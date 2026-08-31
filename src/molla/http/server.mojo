"""The HTTP/1.1 server for the M0 throughput spike.

Structurally this is the echo server from #2 with a parser in the middle, which
was the whole reason the echo server was shaped the way it was. The event loop,
the edge triggered draining, the pending output buffer and the reaping pass are
unchanged. What is new is that bytes now mean something, so a connection can be
half way through a request across reads, and a single read can contain more than
one request.

Three things here exist because of throughput rather than correctness, and they
are the difference between a number worth quoting and a number that measures the
allocator.

Reads go straight into the connection's buffer with one `recv` per syscall and
no per byte loop. The echo server appended bytes one at a time, which was fine
for proving a shape and is not fine when the point is requests per second.

The read buffer is consumed with a moving offset and only compacted when there
is a partial request left over. A keep alive connection that sends a request,
gets an answer, and sends another can run indefinitely without moving memory.

The response is a `memcpy` of a buffer built at startup. Nothing is formatted
per request except the date, and that is patched once a second in place.

Pipelining works because the parse loop keeps going while there is a complete
request in the buffer, rather than handling one and waiting for another event.
"""

from std.memory import stack_allocation, unsafe_memcpy

from molla.http.request import (
    PARSE_DONE,
    PARSE_FAILED,
    PARSE_NEED_MORE,
    Request,
    parse,
)
from molla.http.response import Responder, build_error
from molla.sys.errno import EAGAIN, ECONNRESET, EINTR, errno_name, get_errno
from molla.sys.fd import close, set_nonblocking
from molla.sys.poll import Poller
from molla.sys.socket import (
    INADDR_LOOPBACK,
    accept,
    listen_tcp,
    local_port,
    recv,
    send,
    set_nodelay,
)

comptime READ_CHUNK = 16384
"""Bigger than the echo server's because a pipelining client can put many
requests in one segment and each extra read is a syscall."""

comptime MAX_EVENTS = 256
comptime BACKLOG = 1024
comptime MAX_REQUEST_BYTES = 65536
"""How much unparsed input one connection may hold. A request that does not fit
is a 431 rather than an unbounded buffer."""

comptime DEFAULT_BODY = "Hello from molla.\n"


struct HttpConnection(Movable):
    var fd: Int
    var input: List[UInt8]
    """Raw storage. Grown to fit and never shrunk, because handing 16 KB back
    to the allocator after every read and asking for it again is a malloc, a
    memset and a free per request, and on a loopback benchmark that is most of
    the time. `input_len` says how much of it is real."""

    var input_len: Int
    var input_at: Int
    """How far into `input` the parser has got. Bytes before this are done."""

    var pending: List[UInt8]
    var pending_at: Int
    var peer_done: Bool
    var closed: Bool
    var should_close: Bool
    """Set by `Connection: close` or by a parse error. The socket still has to
    stay open until the response has actually been written."""

    var requests: Int

    def __init__(out self, fd: Int):
        self.fd = fd
        self.input = List[UInt8]()
        self.input_len = 0
        self.input_at = 0
        self.pending = List[UInt8]()
        self.pending_at = 0
        self.peer_done = False
        self.closed = False
        self.should_close = False
        self.requests = 0

    def pending_bytes(self) -> Int:
        return len(self.pending) - self.pending_at

    def wants_write(self) -> Bool:
        return self.pending_bytes() > 0

    def is_finished(self) -> Bool:
        if self.wants_write():
            return False
        return self.peer_done or self.should_close


def _queue(mut conn: HttpConnection, source: List[UInt8]):
    """Append a prebuilt response to a connection's output.

    A free function taking the connection rather than a method on the server,
    because the borrow checker is right that `self.conns[i]` mutably and
    `self.responder.keep` immutably are two borrows of the same `self`. Passing
    the two fields separately says what is actually true, which is that they do
    not overlap.
    """
    var start = len(conn.pending)
    var count = len(source)
    conn.pending.resize(start + count, 0)
    unsafe_memcpy(
        dest=conn.pending.unsafe_ptr().unsafe_offset(start),
        src=source.unsafe_ptr(),
        count=count,
    )


def _compact(mut conn: HttpConnection):
    """Drop consumed input.

    The common case is that everything parsed, which is a length reset and no
    copying at all. Only a half arrived request needs moving, and then it goes
    to the front rather than pushing the buffer forever. Either way the storage
    stays allocated.

    The move is a manual forward loop through one pointer rather than a memcpy.
    Source and destination overlap here, which memcpy does not promise anything
    about, and taking two pointers into the same list is also the aliasing the
    borrow checker refuses. Destination is below source so forward is correct.
    """
    var at = conn.input_at
    if at == 0:
        return
    var remaining = conn.input_len - at
    if remaining == 0:
        conn.input_len = 0
        conn.input_at = 0
        return
    var p = conn.input.unsafe_ptr()
    for i in range(remaining):
        p.unsafe_store(i, p.unsafe_load(at + i))
    conn.input_len = remaining
    conn.input_at = 0


struct HttpServer(Movable):
    var listener: Int
    var poller: Poller
    var conns: List[HttpConnection]
    var write_interest: List[Bool]
    var responder: Responder
    var scratch: Request
    """One parser state reused for every request on every connection. Parsing is
    synchronous and finishes before the next one starts, so there is no reason
    to allocate a header list per request."""

    var accepted: Int
    var requests: Int
    var errors: Int

    def __init__(out self, address: UInt32, port: UInt16) raises:
        self.listener = listen_tcp(address, port, BACKLOG)
        self.poller = Poller(MAX_EVENTS)
        self.conns = List[HttpConnection]()
        self.write_interest = List[Bool]()
        self.responder = Responder(DEFAULT_BODY)
        self.scratch = Request()
        self.accepted = 0
        self.requests = 0
        self.errors = 0
        try:
            self.poller.add_read(self.listener)
        except e:
            _ = close(self.listener)
            raise e

    def port(self) raises -> UInt16:
        return local_port(self.listener)

    def _index_of(self, fd: Int) -> Int:
        for i in range(len(self.conns)):
            if self.conns[i].fd == fd and not self.conns[i].closed:
                return i
        return -1

    def _accept_all(mut self) raises:
        while True:
            var fd = accept(self.listener)
            if fd < 0:
                var code = get_errno()
                if code == EAGAIN or code == EINTR:
                    return
                raise Error("accept failed: " + errno_name(code))
            set_nonblocking(fd)
            set_nodelay(fd)
            self.poller.add_read(fd)
            self.conns.append(HttpConnection(fd))
            self.write_interest.append(False)
            self.accepted += 1

    def _read(mut self, index: Int) raises:
        """Read until EAGAIN, appending into the connection's input buffer.

        Grows the buffer and reads into the tail so each read is one syscall and
        one length update, rather than a copy through a stack buffer.
        """
        while True:
            if self.conns[index].should_close:
                # Already committed to an error response. Anything still
                # arriving is not going to be answered, so stop reading it.
                return
            var start = self.conns[index].input_len
            if len(self.conns[index].input) < start + READ_CHUNK:
                self.conns[index].input.resize(start + READ_CHUNK, 0)
            var got = recv(
                self.conns[index].fd,
                self.conns[index].input.unsafe_ptr().unsafe_offset(start),
                READ_CHUNK,
            )
            if got > 0:
                self.conns[index].input_len = start + got
                self._parse_all(index)
                continue

            if got == 0:
                self.conns[index].peer_done = True
                return
            var code = get_errno()
            if code == EAGAIN:
                return
            if code == EINTR:
                continue
            if code == ECONNRESET:
                self.conns[index].peer_done = True
                self.conns[index].pending.clear()
                self.conns[index].pending_at = 0
                return
            raise Error("recv failed: " + errno_name(code))

    def _parse_all(mut self, index: Int) raises:
        """Handle every complete request sitting in the buffer.

        Loops rather than handling one, because a pipelining client puts several
        in a single segment and readiness has already been consumed.
        """
        while True:
            var available = (
                self.conns[index].input_len - self.conns[index].input_at
            )
            if available <= 0:
                break

            var rc = parse(
                self.conns[index]
                .input.unsafe_ptr()
                .unsafe_offset(self.conns[index].input_at),
                available,
                self.scratch,
            )

            if rc == PARSE_NEED_MORE:
                if available > MAX_REQUEST_BYTES:
                    self._fail(index, 431)
                break

            if rc == PARSE_FAILED:
                self._fail(index, self.scratch.error_status)
                break

            self.conns[index].input_at += self.scratch.consumed
            self.conns[index].requests += 1
            self.requests += 1

            var keep = self.scratch.keep_alive
            if keep:
                _queue(self.conns[index], self.responder.keep)
            else:
                _queue(self.conns[index], self.responder.close)
                self.conns[index].should_close = True
                break

        _compact(self.conns[index])

    def _fail(mut self, index: Int, status: Int):
        var body = build_error(status)
        _queue(self.conns[index], body)
        self.conns[index].should_close = True
        self.conns[index].input_at = self.conns[index].input_len
        self.errors += 1

    def _flush(mut self, index: Int) raises:
        while self.conns[index].pending_bytes() > 0:
            var start = self.conns[index].pending_at
            var remaining = self.conns[index].pending_bytes()
            var wrote = send(
                self.conns[index].fd,
                self.conns[index].pending.unsafe_ptr().unsafe_offset(start),
                remaining,
            )
            if wrote < 0:
                var code = get_errno()
                if code == EAGAIN:
                    return
                if code == EINTR:
                    continue
                self.conns[index].peer_done = True
                self.conns[index].pending.clear()
                self.conns[index].pending_at = 0
                return
            self.conns[index].pending_at += wrote

        self.conns[index].pending.clear()
        self.conns[index].pending_at = 0

    def _sync_write_interest(mut self, index: Int) raises:
        var wanted = self.conns[index].wants_write()
        if wanted == self.write_interest[index]:
            return
        self.poller.set_write_interest(self.conns[index].fd, wanted)
        self.write_interest[index] = wanted

    def _close(mut self, index: Int):
        if self.conns[index].closed:
            return
        self.poller.remove(self.conns[index].fd)
        _ = close(self.conns[index].fd)
        self.conns[index].closed = True

    def _reap(mut self):
        var i = len(self.conns) - 1
        while i >= 0:
            if self.conns[i].closed:
                _ = self.conns.pop(i)
                _ = self.write_interest.pop(i)
            i -= 1

    def poll_once(mut self, timeout_ms: Int) raises -> Int:
        var count = self.poller.wait(timeout_ms)
        # One clock read per batch, not per request.
        self.responder.refresh()

        for i in range(count):
            var ready = self.poller.event(i)
            if ready.fd == self.listener:
                self._accept_all()
                continue

            var index = self._index_of(ready.fd)
            if index < 0:
                continue

            if ready.readable:
                self._read(index)
            if not self.conns[index].closed:
                self._flush(index)

        for i in range(len(self.conns)):
            if self.conns[i].closed:
                continue
            if self.conns[i].is_finished():
                self._close(i)
            else:
                self._sync_write_interest(i)

        self._reap()
        return count

    def serve(mut self, timeout_ms: Int) raises:
        while True:
            _ = self.poll_once(timeout_ms)

    def shutdown(mut self):
        for i in range(len(self.conns)):
            self._close(i)
        self.conns.clear()
        self.write_interest.clear()
        self.poller.shutdown()
        if self.listener >= 0:
            _ = close(self.listener)
            self.listener = -1


def run_http(port: UInt16) raises:
    """Entry point for `molla http`. Loopback only, per D9."""
    var server = HttpServer(INADDR_LOOPBACK, port)
    print("http server listening on 127.0.0.1:" + String(server.port()))
    print("this is the M0 throughput spike. every path returns the same body.")
    print("ctrl-c to stop.")
    server.serve(1000)
