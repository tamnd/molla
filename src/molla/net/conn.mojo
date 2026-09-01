"""One accepted socket and everything owed to it.

The state here is small and every field is one of the four things that go wrong
in a non blocking server, so they are worth naming.

Reading is edge triggered, which means readiness fires once when a socket goes
from empty to not empty and never again until it goes empty. Reading once per
notification leaves whatever did not fit in that one call sitting in the kernel
with nothing scheduled to come back for it. That is a hang under load and a
green test suite under none, so `fill` reads until EAGAIN, every time, and the
whole design depends on it.

Writing is the other half. `send` on a non blocking socket takes what fits and
tells you how much, so a short write is the normal case rather than an error.
Retrying in a loop burns a core while the peer is slow. The remainder goes in
the connection's ring, write interest goes on, and the loop goes back to
waiting. When the ring drains, write interest goes off again, because leaving
it on means the poller reports this socket writable on every single pass.

A descriptor cannot be closed while output is still queued, so end of stream is
not end of connection. A peer that sends a request and shuts down its writing
side is still waiting for the answer. `peer_done` says the reading half is
finished, `closing` says close once the ring is empty, and only `closed` means
the descriptor is gone.

The read buffer and the write ring both come from `molla.io` and both belong to
the connection for its whole life. They are not per request, so a keep alive
connection allocates nothing after its first request grew them to the size that
connection needs.
"""

from molla.io.buffer import Buffer
from molla.io.ring import Ring
from molla.sys.errno import EAGAIN, ECONNRESET, EINTR, EPIPE, get_errno
from molla.sys.fd import close
from molla.sys.socket import MAX_IOV, recv, send, write_vectored

comptime READ_CHUNK = 16384
"""How much room `fill` makes before each `recv`.

Sixteen kilobytes because that is one TLS record and a comfortable fit for a
request with headers, and because a smaller chunk means more syscalls on a
connection that is sending a body. It is a reservation, not an allocation: the
buffer only grows if the data actually arrives."""

comptime DEFAULT_READ_CAPACITY = 16384
comptime DEFAULT_WRITE_CAPACITY = 65536
"""Per connection output budget. A response larger than this is written in more
than one pass, which is the point: unbounded output buffering is how a server
with slow clients runs out of memory."""

comptime FILL_EOF = -2
"""`fill` saw end of stream. Distinct from an error, because there may still be
bytes in the buffer worth parsing and output worth writing."""

comptime FILL_ERROR = -1


struct Connection(Movable):
    """A socket, its buffers, and what state it is in."""

    var fd: Int
    var slot: Int
    """Index in the reactor's table. The timing wheel carries this as its token
    so a fired timer names a connection without a lookup."""

    var generation: Int
    """Bumped every time the slot is reused. A stale reference from a fired
    timer or a queued handoff is caught by comparing this rather than by
    hoping."""

    var input: Buffer
    var output: Ring
    var timer: Int
    var last_active_ms: Int
    var peer_done: Bool
    var closing: Bool
    """Close as soon as the output ring is empty. Set by a protocol that has
    answered its last request, and by the reactor on an idle timeout."""

    var closed: Bool
    var write_interest: Bool
    """Whether the poller is currently watching this socket for writability.
    Tracked so interest is only changed when the answer changes, since a
    register call per pass is a syscall per pass for nothing."""

    var producing: Bool
    """The protocol has more to write that no readiness event is going to ask
    for. A streaming response is the case: the client sent one request and is
    not going to send another, and once the ring drains into the socket there is
    no edge left to come back on. Set it and the reactor calls `on_writable`
    while there is room, whether or not the poller said anything."""

    var bytes_in: Int
    var bytes_out: Int
    var short_writes: Int
    """How many times the kernel took less than was offered. A connection with
    a lot of these is a slow reader, which is what backpressure exists for."""

    def __init__(
        out self,
        fd: Int,
        slot: Int,
        generation: Int,
        read_capacity: Int,
        write_capacity: Int,
        counter: Int,
        now_ms: Int,
    ):
        self.fd = fd
        self.slot = slot
        self.generation = generation
        self.input = Buffer(read_capacity, counter)
        self.output = Ring(write_capacity, counter)
        self.timer = -1
        self.last_active_ms = now_ms
        self.peer_done = False
        self.closing = False
        self.closed = False
        self.write_interest = False
        self.producing = False
        self.bytes_in = 0
        self.bytes_out = 0
        self.short_writes = 0

    def reuse(mut self, fd: Int, generation: Int, now_ms: Int):
        """Take over a new socket without giving the buffers back.

        A reactor slot is reused as soon as the connection in it closes, and
        building a fresh `Connection` for it meant a free and a calloc for the
        read buffer and the write ring every time, plus growing the read buffer
        again the first time a request did not fit in the starting size. That is
        a per connection allocation on a server that claims not to have one, and
        on a client that opens a connection per request it is a per request
        allocation.

        So the buffers stay and everything else is set back to what a new
        connection would have. The buffers keep whatever size the last
        connection grew them to, which is the point: after a warm up the traffic
        has already paid for the size it needs.
        """
        self.fd = fd
        self.generation = generation
        self.input.clear()
        self.output.clear()
        self.timer = -1
        self.last_active_ms = now_ms
        self.peer_done = False
        self.closing = False
        self.closed = False
        self.write_interest = False
        self.producing = False
        self.bytes_in = 0
        self.bytes_out = 0
        self.short_writes = 0

    def is_valid(self) -> Bool:
        """Both blocks were allocated. A connection that fails this is closed
        immediately rather than served with a null buffer."""
        return self.input.is_valid() and self.output.is_valid()

    def touch(mut self, now_ms: Int):
        """Say the connection did something. This is the whole of the idle
        timeout refresh, and it is one store rather than a timer operation."""
        self.last_active_ms = now_ms

    def idle_ms(self, now_ms: Int) -> Int:
        return now_ms - self.last_active_ms

    def fill(mut self) -> Int:
        """Read until the socket is empty. Returns bytes read, or a code.

        `FILL_EOF` when the peer closed its writing side, which is not a
        reason to close: there may be a request in the buffer and a response
        owed. `FILL_ERROR` when the connection broke, which is.

        Read into the buffer's own tail, so bytes land where the parser will
        read them and nothing is copied between a stack array and the buffer.
        """
        var total = 0
        while True:
            if not self.input.reserve(READ_CHUNK):
                # Out of memory for this connection. Whatever was read is
                # returned, and the caller decides, which for a server under
                # pressure means closing this one rather than dying.
                return total if total > 0 else FILL_ERROR
            var got = recv(self.fd, self.input.tail_ptr(), READ_CHUNK)
            if got > 0:
                self.input.commit(got)
                self.bytes_in += got
                total += got
                continue
            if got == 0:
                self.peer_done = True
                return total if total > 0 else FILL_EOF
            var code = get_errno()
            if code == EAGAIN:
                return total
            if code == EINTR:
                continue
            if code == ECONNRESET:
                self.peer_done = True
                return total if total > 0 else FILL_EOF
            return FILL_ERROR

    def queue(mut self, data: Span[UInt8, _]) -> Int:
        """Put bytes in the output ring. Returns how many it took.

        Fewer than offered means the ring is full, which is backpressure and
        not an error. A protocol that gets a short answer stops producing and
        waits to be told the socket drained."""
        return self.output.push(data)

    def queue_str(mut self, text: StringSpan) -> Int:
        return self.output.push_str(text)

    def produce(mut self, more: Bool):
        """Say whether there is more to write on the protocol's own initiative.

        Cheaper and narrower than a fifth call on the trait. The reactor needs
        one bit to tell a connection that is finished from one that is waiting
        on nothing but its own producer, and this is that bit.
        """
        self.producing = more

    def writable(self) -> Int:
        """Room left in the output ring. What a protocol checks before
        building a response it cannot queue."""
        return self.output.writable()

    def pending(self) -> Int:
        """Bytes queued and not yet written to the socket."""
        return self.output.readable()

    def wants_write(self) -> Bool:
        return self.output.readable() > 0

    def flush(mut self) -> Int:
        """Write as much of the ring as the kernel will take.

        One `writev` per pass, because the queued bytes are in at most two
        pieces when they wrap the end of the ring, and two pieces is what a
        vectored write is for. The alternative is two `send` calls, and the
        second one is the one that gets a short write.
        """
        var total = 0
        var bases = List[Int](capacity=MAX_IOV)
        var lengths = List[Int](capacity=MAX_IOV)
        while self.output.readable() > 0:
            var pieces = self.output.readable_pieces(bases, lengths)
            if pieces == 0:
                break
            var offered = 0
            for i in range(pieces):
                offered += lengths[i]
            var result = write_vectored(self.fd, bases, lengths)
            if result.is_ok():
                var wrote = result.value
                self.output.consume(wrote)
                self.bytes_out += wrote
                total += wrote
                if wrote < offered:
                    self.short_writes += 1
                    return total
                continue
            var code = result.err
            if code == EAGAIN:
                self.short_writes += 1
                return total
            if code == EINTR:
                continue
            if code == EPIPE or code == ECONNRESET:
                # The peer is gone. Dropping the output is the only option and
                # it is not an error the server should care about.
                self.output.clear()
                self.peer_done = True
                self.closing = True
                return total
            return FILL_ERROR
        return total

    def send_direct(mut self, data: Span[UInt8, _]) -> Int:
        """Write straight to the socket, bypassing the ring.

        Only correct when the ring is empty, because writing around queued
        bytes reorders the stream, so it returns 0 rather than doing that. This
        is the fast path for a response that fits in the kernel's buffer: one
        syscall, no copy into the ring at all. Whatever the kernel would not
        take is the caller's to queue."""
        if self.output.readable() > 0 or len(data) == 0:
            return 0
        var wrote = send(
            self.fd,
            Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=Int(data.unsafe_ptr())
            ),
            len(data),
        )
        if wrote <= 0:
            return 0
        self.bytes_out += wrote
        return wrote

    def finish(mut self):
        """Answer this connection and then close it. What a protocol calls for
        `Connection: close`, and what the reactor calls on an idle timeout."""
        self.closing = True

    def is_finished(self) -> Bool:
        """Nothing more is owed and nothing more is coming."""
        if self.output.readable() > 0:
            return False
        return self.closing or self.peer_done

    def shut(mut self):
        """Close the descriptor. The buffers go when the connection does."""
        if self.closed:
            return
        if self.fd >= 0:
            _ = close(self.fd)
        self.closed = True
