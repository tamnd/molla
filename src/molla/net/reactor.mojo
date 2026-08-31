"""One event loop, one thread, one set of connections.

A reactor owns a poller, a table of connections, a timing wheel and nothing
that another thread touches, with one deliberate exception. Two reactors share
no state, so there is no lock on the request path and no cache line bouncing
between cores. Scaling is by running more of them.

The exception is the handoff queue, which exists only where the kernel will not
shard the accept queue for us. It is one mutex, taken once per accepted
connection and never during a request, and on Linux it is not used at all.

The design here is the one the M0 echo spike proved out, with the three things
that spike was written to learn kept intact and everything else replaced.

Read until EAGAIN, because readiness is edge triggered and a second
notification for bytes that were already announced is never coming.

Never spin on a short write. The remainder goes in the connection's ring and
write interest goes on, so a slow peer costs one poller registration rather
than a core.

Do not close a descriptor with output still queued. End of stream from the peer
is not end of connection.

What the spike did not have is timeouts, which is why the wheel is here. A
connection that connects and says nothing must not hold a descriptor forever,
and that is not a hypothetical: it is the shape of the simplest denial of
service there is, and it costs the attacker one socket.

Draining is the other thing another thread can ask for. A reactor that is
draining has closed its listeners, so nothing new arrives, and closes each
connection at the first moment that connection owes the client nothing: no
bytes in its output ring, nothing half read in its input, no stream in flight.
A request that is being served when the drain starts is served to the end. When
the deadline passes, whatever is left is closed anyway and counted, because a
shutdown that waits forever for one stuck client is not a shutdown.

The flags another thread touches are atomics rather than plain fields. They
were plain `Bool` until #15 and it worked on both machines, which is the
dangerous kind of working: a non atomic store read from another thread is
undefined rather than merely stale, and the reason to fix it is that the
compiler is allowed to keep `stopping` in a register across the whole loop.

The poller is edge triggered on both platforms, which is a deliberate deviation
from what issue #10 asked for. The issue said edge triggered on Linux and level
triggered on macOS. Running kqueue level triggered would mean the two platforms
need different drain rules, and the drain rule is the part that is easy to get
subtly wrong. Edge triggered everywhere means one rule, `fill` reads until
EAGAIN and that is the only correct way to call it, and the tests exercise the
same code path on both machines. The cost is that kqueue reports a readiness
edge molla might not have drained yet, and the answer to that is the same
EAGAIN loop.
"""

from std.memory import stack_allocation

from molla.net.conn import (
    DEFAULT_READ_CAPACITY,
    DEFAULT_WRITE_CAPACITY,
    FILL_ERROR,
    Connection,
)
from molla.net.wheel import TICK_MS, Wheel
from molla.sys.clock import monotonic_ms
from molla.sys.errno import EAGAIN, EINTR, EMFILE, ENFILE, errno_name, get_errno
from molla.sys.fd import close, set_nonblocking
from molla.sys.poll import Poller, Ready
from molla.sys.socket import (
    SO_SNDBUF,
    accept,
    recv,
    send,
    set_buffer_size,
    set_keepalive,
    set_nodelay,
    socket_pair,
)
from molla.sys.atomic import AtomicBlock, AtomicRef
from molla.sys.queue import MpscQueue
from molla.sys.thread import self_id, set_thread_name

comptime MAX_EVENTS = 256
"""Events pulled from the kernel per pass. Larger batches mean fewer syscalls
and a longer tail before the first connection in the batch is served, and 256
is where those two stop arguing on a thousand connections."""

comptime DEFAULT_IDLE_MS = 60000
"""How long a connection may do nothing before it is closed. A minute is what
keep alive is worth: long enough that a browser reuses the connection for the
next request, short enough that an abandoned socket is not still held when the
next deploy happens."""

comptime WAKE_DRAIN = 64
comptime NO_SLOT = -1

comptime HANDOFF_CAPACITY = 1024
"""Connections one reactor may have waiting to be adopted by another. A full
queue is not a dropped connection: the accepting thread serves it itself, which
is worse balance and better than a refusal."""

comptime DRAIN_TICK_MS = 5
"""How long a draining reactor waits in the poller. Short, because the last
thing a drain waits on is often a connection that has already flushed and needs
one more pass to notice, and a drain that takes a second to see that is a
second on every deploy."""

comptime FLAG_STOPPING = 0
comptime FLAG_DRAINING = 1
comptime FLAG_DEADLINE = 2
comptime FLAG_OPEN = 3
comptime FLAG_DROPPED = 4
comptime FLAG_THREAD = 5
comptime FLAG_COUNT = 6
"""The control block. Six counters, each alone on a cache line, and the only
memory two threads write to. Three are asks from outside, `stopping`,
`draining` and the drain deadline, and three are answers this reactor
publishes, the open connection count, what it had to drop at the deadline, and
which thread it is running on."""

comptime STATE_NEW = 0
comptime STATE_RUNNING = 1
comptime STATE_DRAINING = 2
comptime STATE_STOPPED = 3


def state_name(state: Int) -> String:
    if state == STATE_RUNNING:
        return "running"
    if state == STATE_DRAINING:
        return "draining"
    if state == STATE_STOPPED:
        return "stopped"
    return "new"


comptime SERVICE_ROUNDS = 8
"""How many times one connection may go round the read, produce, write cycle in
a single pass. A budget rather than a limit: whatever is left over is picked up
at the top of the next pass. It exists so one connection with a megabyte in
flight cannot hold the reactor while a thousand others wait."""


trait Protocol(Defaultable, Deinitable, Movable):
    """What a reactor does with the bytes.

    Deliberately four calls and no more. Everything above this, HTTP framing in
    #11 and the API surfaces after it, is written against these, and anything
    a protocol needs that is not here is a sign the boundary is in the wrong
    place rather than a reason to add a fifth.
    """

    def on_open(mut self, mut conn: Connection):
        """A connection was accepted and is in the table."""
        ...

    def on_readable(mut self, mut conn: Connection) -> Bool:
        """Bytes arrived and are in `conn.input`. Return False to close.

        Called after the socket has been drained, so this sees everything that
        was available, not the first chunk of it. What is consumed is the
        protocol's business: leave a partial request in the buffer and it will
        be there next time with more after it."""
        ...

    def on_writable(mut self, mut conn: Connection) -> Bool:
        """The output ring drained and there is room again. Return False to
        close. This is where a streaming response produces its next piece."""
        ...

    def on_close(mut self, mut conn: Connection):
        """The connection is going away. Last chance to account for it."""
        ...


struct Reactor[P: Protocol](Movable):
    """An event loop over one poller and one connection table."""

    var poller: Poller
    var proto: Self.P
    var conns: List[Connection]
    var live: List[Bool]
    """Whether the connection at each slot is in use. Slots are reused rather
    than removed, so an index stays valid and a fired timer can name one."""

    var free_slots: List[Int]
    var slot_of_fd: List[Int]
    """Descriptor to slot, indexed by the descriptor number itself. Descriptors
    are small and dense by definition, which turns what would be a hash lookup
    per event into an array read."""

    var wheel: Wheel
    var listeners: List[Int]
    var wake_read: Int
    var wake_write: Int
    """A socketpair. Another thread writes one byte to make this reactor come
    back from the poller, which is how a handoff and a shutdown both arrive."""

    var handoff: MpscQueue
    """Descriptors another thread has given this one. A ticket queue rather
    than a mutex and a list, so an accepting thread never waits on a reactor
    that happens to be in the middle of a pass."""

    var control: AtomicBlock
    """The flags another thread reads and writes. See `FLAG_STOPPING`."""

    var index: Int
    """Which reactor this is, for the thread name and the SIGQUIT dump."""

    var idle_timeout_ms: Int
    var read_capacity: Int
    var write_capacity: Int
    var counter: Int
    var send_buffer_bytes: Int
    """Kernel send buffer for accepted sockets, or zero to leave it alone.

    Set on the accepted socket rather than on the listener. A listener does pass
    the option down, but macOS then autotunes the inherited buffer back up, so a
    test that asked for eight kilobytes gets a hundred and fifty and quietly
    stops testing backpressure. Setting it after accept sticks on both."""

    var listeners_closed: Bool
    """Set once the drain has closed them, so a drain that takes several passes
    does not try to close a descriptor twice."""

    var accepted: Int
    var closed_count: Int
    var timed_out: Int
    var refused: Int
    """Connections dropped because the process is out of descriptors. Counted
    rather than raised, because running out of descriptors is a load condition
    and not a bug in the loop."""

    var passes: Int
    var fired: List[Int]
    """Reused across passes so expiring timers allocates nothing."""

    var again: List[Int]
    """Slots with work left over from the last pass. Non empty means the next
    pass does not block in the poller."""

    var peers: List[Int]
    """Addresses of the reactors this one may hand a connection to, itself
    included. Empty when the kernel does the sharding, which is the Linux case
    and the one where nothing is handed anywhere."""

    var next_peer: Int
    """Round robin cursor. Not random and not least loaded, because both need
    information this reactor does not have, and round robin over a hash of
    source ports is already close to even."""

    def __init__(
        out self,
        var proto: Self.P,
        idle_timeout_ms: Int,
        counter: Int,
    ) raises:
        self.poller = Poller(MAX_EVENTS)
        self.proto = proto^
        self.conns = List[Connection]()
        self.live = List[Bool]()
        self.free_slots = List[Int]()
        self.slot_of_fd = List[Int]()
        self.wheel = Wheel(monotonic_ms())
        self.listeners = List[Int]()
        self.wake_read = -1
        self.wake_write = -1
        self.handoff = MpscQueue(HANDOFF_CAPACITY)
        self.control = AtomicBlock(FLAG_COUNT)
        self.index = 0
        self.idle_timeout_ms = idle_timeout_ms
        self.read_capacity = DEFAULT_READ_CAPACITY
        self.write_capacity = DEFAULT_WRITE_CAPACITY
        self.counter = counter
        self.send_buffer_bytes = 0
        self.listeners_closed = False
        self.accepted = 0
        self.closed_count = 0
        self.timed_out = 0
        self.refused = 0
        self.passes = 0
        self.fired = List[Int]()
        self.again = List[Int]()
        self.peers = List[Int]()
        self.next_peer = 0

        if not self.handoff.is_valid():
            raise Error("the reactor could not allocate its handoff queue")
        if not self.control.is_valid():
            raise Error("the reactor could not allocate its control block")

        var ends = List[Int]()
        if not socket_pair(ends).is_ok():
            raise Error("the reactor could not open its wakeup channel")
        self.wake_read = ends[0]
        self.wake_write = ends[1]
        set_nonblocking(self.wake_read)
        set_nonblocking(self.wake_write)
        self.poller.add_read(self.wake_read)

    def add_listener(mut self, fd: Int) raises:
        """Take ownership of a listening socket and watch it."""
        self.listeners.append(fd)
        self.poller.add_read(fd)

    def _is_listener(self, fd: Int) -> Bool:
        for i in range(len(self.listeners)):
            if self.listeners[i] == fd:
                return True
        return False

    def _remember(mut self, fd: Int, slot: Int):
        while len(self.slot_of_fd) <= fd:
            self.slot_of_fd.append(NO_SLOT)
        self.slot_of_fd[fd] = slot

    def _slot_for_fd(self, fd: Int) -> Int:
        if fd < 0 or fd >= len(self.slot_of_fd):
            return NO_SLOT
        return self.slot_of_fd[fd]

    def adopt(mut self, fd: Int) raises -> Int:
        """Put an accepted descriptor into the table. Returns its slot.

        The descriptor may come from this reactor's own listener or from
        another thread's handoff, and neither the setup nor the bookkeeping
        differs, so there is one path.
        """
        var now = monotonic_ms()
        var slot: Int
        var generation = 0
        if len(self.free_slots) > 0:
            slot = self.free_slots.pop()
            generation = self.conns[slot].generation + 1
            self.conns[slot] = Connection(
                fd,
                slot,
                generation,
                self.read_capacity,
                self.write_capacity,
                self.counter,
                now,
            )
            self.live[slot] = True
        else:
            slot = len(self.conns)
            self.conns.append(
                Connection(
                    fd,
                    slot,
                    generation,
                    self.read_capacity,
                    self.write_capacity,
                    self.counter,
                    now,
                )
            )
            self.live.append(True)

        if not self.conns[slot].is_valid():
            # Out of memory for this connection's buffers. Refuse it rather
            # than serve it with a null block.
            self.live[slot] = False
            self.free_slots.append(slot)
            _ = close(fd)
            self.refused += 1
            return NO_SLOT

        if self.send_buffer_bytes > 0:
            set_buffer_size(fd, SO_SNDBUF, self.send_buffer_bytes)

        self._remember(fd, slot)
        self.poller.add_read(fd)
        self.conns[slot].timer = self.wheel.add(slot, self.idle_timeout_ms)
        self.accepted += 1
        # Published rather than counted on demand, because the thread that
        # wants this number is not this one and walking the slot table from
        # another thread is a race.
        _ = self._flag(FLAG_OPEN).add(1)
        self.proto.on_open(self.conns[slot])
        return slot

    def _accept_all(mut self, listener: Int) raises:
        """Drain the accept backlog.

        One readiness edge can cover any number of pending connections, so
        accepting once per event would leave the rest waiting for an unrelated
        client to arrive and wake us again.
        """
        while True:
            var fd = accept(listener)
            if fd < 0:
                var code = get_errno()
                if code == EAGAIN or code == EINTR:
                    return
                if code == EMFILE or code == ENFILE:
                    # Out of descriptors. The pending connections stay in the
                    # backlog and the kernel will tell us again, so this is a
                    # pause rather than a failure.
                    self.refused += 1
                    return
                raise Error("accept failed: " + errno_name(code))

            # macOS does not put an accepted socket into non blocking mode and
            # has no accept4 to ask it to, so both platforms set it here.
            set_nonblocking(fd)
            try:
                set_nodelay(fd)
                set_keepalive(fd)
            except:
                # A unix socket has neither. Not being able to set them is not
                # a reason to drop a connection that is otherwise fine.
                pass
            self._place(fd)

    def add_peer(mut self, address: Int):
        """Name another reactor this one may hand connections to.

        Only called where the kernel will not shard the accept queue itself.
        The reactor doing the accepting adds every reactor including itself,
        so the accepting thread keeps its share of the work rather than
        becoming a dispatcher that does nothing else.
        """
        self.peers.append(address)

    def _place(mut self, fd: Int) raises:
        """Serve this connection here, or hand it to the next reactor along."""
        if len(self.peers) == 0:
            _ = self.adopt(fd)
            return
        var target = self.peers[self.next_peer]
        self.next_peer = (self.next_peer + 1) % len(self.peers)
        var me = Int(Pointer(to=self))
        if target == me:
            _ = self.adopt(fd)
            return
        var peer = Pointer[Self, MutAnyOrigin](unsafe_from_address=target)
        if not peer[].handoff_push(fd):
            # The peer could not take it. Serving it here is worse balance and
            # better than dropping a connection somebody is waiting on.
            _ = self.adopt(fd)

    def handoff_push(mut self, fd: Int) -> Bool:
        """Give this reactor a descriptor from another thread.

        The macOS path. The acceptor thread claims a cell in the ticket queue
        and one byte on the wakeup socket brings the reactor back out of the
        poller to collect it. Nothing here can block, so an accepting thread
        never waits on a reactor that is busy.

        False means the queue is full, which is the caller's cue to serve the
        connection itself rather than to retry.
        """
        if not self.handoff.push(fd):
            return False
        self.wake()
        return True

    def wake(mut self):
        """Make this reactor return from the poller. Safe from any thread."""
        if self.wake_write < 0:
            return
        var byte = stack_allocation[1, UInt8]()
        byte.unsafe_store(0, UInt8(1))
        _ = send(self.wake_write, byte, 1)

    def _drain_wake(mut self) raises:
        """Empty the wakeup socket and adopt anything handed over.

        The bytes carry no information. They exist to interrupt the poller, and
        the queue is where the actual work is, which means a burst of wakeups
        collapses into one pass rather than one pass each.
        """
        var scratch = stack_allocation[WAKE_DRAIN, UInt8]()
        while True:
            if recv(self.wake_read, scratch, WAKE_DRAIN) <= 0:
                break

        var fd = 0
        while self.handoff.pop(fd):
            if self.is_draining():
                # Accepted a moment before the drain started and handed to a
                # reactor that is now closing. Taking it would mean answering a
                # request on a server that has already said it is going away.
                _ = close(fd)
                continue
            _ = self.adopt(fd)

    def _close_slot(mut self, slot: Int):
        """Close one connection and give its slot back.

        The generation is bumped by whoever reuses the slot rather than here,
        so a reference held from before this point compares unequal to whatever
        lands in the slot next.
        """
        if slot < 0 or slot >= len(self.conns) or not self.live[slot]:
            return
        self.proto.on_close(self.conns[slot])
        if self.conns[slot].timer >= 0:
            self.wheel.cancel(self.conns[slot].timer)
            self.conns[slot].timer = -1
        var fd = self.conns[slot].fd
        self.poller.remove(fd)
        self.conns[slot].shut()
        self._remember(fd, NO_SLOT)
        self.live[slot] = False
        self.free_slots.append(slot)
        self.closed_count += 1
        _ = self._flag(FLAG_OPEN).sub(1)

    def _sync_write_interest(mut self, slot: Int) raises:
        var wanted = self.conns[slot].wants_write()
        if wanted == self.conns[slot].write_interest:
            return
        self.poller.set_write_interest(self.conns[slot].fd, wanted)
        self.conns[slot].write_interest = wanted

    def _service(mut self, slot: Int, readable: Bool, writable: Bool) raises:
        """Everything one readiness event means for one connection.

        The loop in the middle is not decoration. A connection can read more in
        one pass than its output ring holds, and the leftover sits in the input
        buffer waiting for room. Handling it once and returning would wait for
        the next readiness edge to come back for it, and on an edge triggered
        poller that edge is not coming: the bytes were already announced and the
        socket may be writable the whole time without ever transitioning. That
        is a hang under exactly the load that makes a hang worst.

        So this keeps going while it is making progress, and if it runs out of
        rounds while still making progress it puts the slot on `again`, which
        makes the next pass come straight back to it instead of waiting.
        """
        var now = monotonic_ms()
        var keep = True

        if readable:
            var got = self.conns[slot].fill()
            if got == FILL_ERROR:
                self._close_slot(slot)
                return
            if got > 0:
                self.conns[slot].touch(now)

        var rounds = 0
        var moving = True
        while keep and moving and rounds < SERVICE_ROUNDS:
            rounds += 1
            var input_before = self.conns[slot].input.length
            var output_before = self.conns[slot].pending()
            var wrote_any = 0

            if input_before > 0 and self.conns[slot].writable() > 0:
                keep = self.proto.on_readable(self.conns[slot])
                if not keep:
                    break

            # `on_writable` is for a protocol that produces without being asked,
            # which is what a streaming response is. It only makes sense when
            # there is somewhere to put the output, so either the socket said it
            # had room or the protocol said it has more, and in the second case
            # there is no edge coming and waiting for one is the hang.
            if (writable or self.conns[slot].producing) and self.conns[
                slot
            ].writable() > 0:
                keep = self.proto.on_writable(self.conns[slot])
                if not keep:
                    break

            if self.conns[slot].wants_write():
                var wrote = self.conns[slot].flush()
                if wrote == FILL_ERROR:
                    self._close_slot(slot)
                    return
                if wrote > 0:
                    wrote_any = wrote
                    self.conns[slot].touch(now)

            # Bytes leaving the ring count as progress. Without that, a producer
            # that fills the ring and has it drained inside the same round looks
            # identical to one that did nothing, since both start and end the
            # round with an empty ring, and the loop stops with the stream half
            # written and nothing scheduled to finish it.
            moving = (
                self.conns[slot].input.length != input_before
                or self.conns[slot].pending() != output_before
                or wrote_any > 0
            )

        if keep and moving and rounds == SERVICE_ROUNDS:
            # Still going when the budget ran out. Come back next pass without
            # blocking rather than hoping for an edge that has already been
            # spent.
            self.again.append(slot)

        if not keep:
            self.conns[slot].finish()

        if self.conns[slot].is_finished():
            self._close_slot(slot)
            return
        self._sync_write_interest(slot)

    def _expire(mut self) raises:
        """Close what has been idle too long.

        A fired timer is a question rather than an answer. The connection may
        have been busy since it was armed, in which case it gets a fresh timer
        for whatever is left of its idle allowance. That is why activity does
        not have to cancel anything: touching a connection is one store, and
        the correction happens here at most once per timeout period.
        """
        var now = monotonic_ms()
        _ = self.wheel.advance(now, self.fired)
        for i in range(len(self.fired)):
            var slot = self.fired[i]
            if slot < 0 or slot >= len(self.conns) or not self.live[slot]:
                continue
            self.conns[slot].timer = -1
            var idle = self.conns[slot].idle_ms(now)
            if idle >= self.idle_timeout_ms:
                self.timed_out += 1
                self._close_slot(slot)
                continue
            self.conns[slot].timer = self.wheel.add(
                slot, self.idle_timeout_ms - idle
            )

    def poll_once(mut self, timeout_ms: Int) raises -> Int:
        """Wait once and handle everything that came back.

        Public so tests can step the loop by hand and drive a client on the
        same thread, which is how every test in this layer is written.
        """
        self.passes += 1
        # A connection that ran out of service rounds last pass has work in hand
        # and no reason to wait, so the poller is polled rather than waited on.
        var wait = 0 if len(self.again) > 0 else timeout_ms
        var count = self.poller.wait(wait)

        for i in range(count):
            var ready = self.poller.event(i)
            if ready.fd == self.wake_read:
                self._drain_wake()
                continue
            if self._is_listener(ready.fd):
                self._accept_all(ready.fd)
                continue
            var slot = self._slot_for_fd(ready.fd)
            if slot == NO_SLOT or not self.live[slot]:
                continue
            self._service(slot, ready.readable, ready.writable)

        # Taken and cleared first, because servicing these can add to the list
        # for the next pass and running the list while it grows would not
        # terminate.
        var resume = List[Int]()
        for i in range(len(self.again)):
            resume.append(self.again[i])
        self.again.clear()
        for i in range(len(resume)):
            var slot = resume[i]
            if slot < 0 or slot >= len(self.conns) or not self.live[slot]:
                continue
            self._service(slot, False, False)

        self._expire()
        return count

    def _flag(self, which: Int) -> AtomicRef:
        return self.control.slot(which)

    def is_stopping(self) -> Bool:
        return self._flag(FLAG_STOPPING).load() != 0

    def is_draining(self) -> Bool:
        return self._flag(FLAG_DRAINING).load() != 0

    def state(self) -> Int:
        """What this reactor is doing, as seen from another thread."""
        if self._flag(FLAG_STOPPING).load() != 0:
            return STATE_STOPPED
        if self._flag(FLAG_DRAINING).load() != 0:
            return STATE_DRAINING
        if self._flag(FLAG_THREAD).load() != 0:
            return STATE_RUNNING
        return STATE_NEW

    def open_published(self) -> Int:
        """The open connection count this reactor last published.

        Read by the supervisor thread while the reactor is running, which is
        why it is an atomic and why `connection_count` is not the thing to
        call from outside."""
        return self._flag(FLAG_OPEN).load()

    def dropped(self) -> Int:
        """Connections cut because the drain deadline passed."""
        return self._flag(FLAG_DROPPED).load()

    def thread_id(self) -> Int:
        return self._flag(FLAG_THREAD).load()

    def handoff_depth(self) -> Int:
        return self.handoff.depth()

    def run(mut self) raises:
        """Serve until told to stop. One thread lives in here."""
        _ = set_thread_name("molla-io-" + String(self.index))
        self._flag(FLAG_THREAD).store(self_id())
        while not self.is_stopping():
            var wait = self.wheel.next_timeout_ms()
            if wait < 0:
                wait = TICK_MS * 10
            if self.is_draining():
                wait = DRAIN_TICK_MS
            _ = self.poll_once(wait)
            if self.is_draining():
                self._drain_step()

    def stop(mut self):
        """Ask the loop to finish its pass and return. Safe from any thread.

        Abrupt. Connections in flight lose whatever was still queued for them,
        which is what a test wants at the end and not what a running server
        wants. `begin_drain` is the polite one."""
        self._flag(FLAG_STOPPING).store(1)
        self.wake()

    def begin_drain(mut self, deadline_ms: Int):
        """Stop accepting and start closing. Safe from any thread.

        `deadline_ms` is a monotonic timestamp rather than a duration, so every
        reactor in a server shares one deadline instead of each starting its
        own clock whenever its thread happens to notice.
        """
        self._flag(FLAG_DEADLINE).store(deadline_ms)
        self._flag(FLAG_DRAINING).store(1)
        self.wake()

    def _close_listeners(mut self):
        """Stop accepting, before anything else in the drain.

        The listeners go first so the window where a client can connect and get
        an immediate close is as short as it can be. A connection already in
        the accept backlog is closed by the kernel with the listener, which
        looks to that client like a connection refused and is the right answer
        for a server that is on its way out."""
        if self.listeners_closed:
            return
        self.listeners_closed = True
        for i in range(len(self.listeners)):
            self.poller.remove(self.listeners[i])
            _ = close(self.listeners[i])
        self.listeners.clear()

    def _owes_nothing(self, slot: Int) -> Bool:
        """Whether this connection can be closed without cutting anything off.

        Three questions, and all three have to be no. Is there a response still
        queued, is there a request half read, and is a stream still producing.
        A keep alive connection between requests answers no to all three, which
        is why an idle pool closes the instant a drain starts.
        """
        if self.conns[slot].pending() > 0:
            return False
        if self.conns[slot].input.length > 0:
            return False
        if self.conns[slot].producing:
            return False
        return True

    def _drain_step(mut self) raises:
        """One pass of the drain. Called after the poller, every pass."""
        self._close_listeners()

        var deadline = self._flag(FLAG_DEADLINE).load()
        var expired = deadline > 0 and monotonic_ms() >= deadline

        for i in range(len(self.conns)):
            if not self.live[i]:
                continue
            if expired:
                # Out of time. What is left is cut, and counted so the number
                # ends up in the shutdown line rather than in nobody's log.
                _ = self._flag(FLAG_DROPPED).add(1)
                self._close_slot(i)
                continue
            # One more read before deciding this connection is finished. The
            # poller is edge triggered, so a request that arrived between the
            # last pass and this one has already spent its edge and there is no
            # second one coming. Without this a connection that is about to ask
            # for something looks exactly like one that is asleep, and the
            # difference is a request the client sent and never got an answer
            # to. It costs one recv per idle connection per drain pass.
            self._service(i, True, False)
            if not self.live[i]:
                continue
            if self._owes_nothing(i):
                self._close_slot(i)

        if self._flag(FLAG_OPEN).load() <= 0:
            self._flag(FLAG_STOPPING).store(1)

    def connection_count(self) -> Int:
        var total = 0
        for i in range(len(self.live)):
            if self.live[i]:
                total += 1
        return total

    def shutdown(mut self):
        """Close everything this reactor owns.

        Connections first, so each protocol sees `on_close` while its
        connection is still whole, then the listeners, then the poller.
        """
        for i in range(len(self.conns)):
            if self.live[i]:
                self._close_slot(i)
        for i in range(len(self.listeners)):
            _ = close(self.listeners[i])
        self.listeners.clear()
        if self.wake_read >= 0:
            _ = close(self.wake_read)
            self.wake_read = -1
        if self.wake_write >= 0:
            _ = close(self.wake_write)
            self.wake_write = -1
        self.poller.shutdown()
