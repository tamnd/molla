"""Tests for the timing wheel, the reactor, and the threaded server.

Real sockets on real loopback, and for the server tests, real threads. The
wheel is the one part that can be tested without either, and it is tested with
an explicit clock rather than by sleeping, because a test that sleeps for a
deadline is a test that fails on a loaded CI runner.

The single reactor tests drive the loop by hand with `poll_once` and act as the
client between passes, which is how the M0 spike tests were written and is
still the only way to be both the client and the server without a second
thread. The server tests do use threads, so there the client just talks to a
socket and the server side runs where it will really run.
"""

from std.memory import stack_allocation

from harness import Suite

from molla.net.listener import (
    SHARDED_ACCEPT,
    ListenAddress,
    bound_port,
    open_listener,
)
from molla.net.protocol import EchoProtocol
from molla.net.reactor import Reactor
from molla.net.server import Server, default_workers
from molla.net.wheel import SLOTS, TICK_MS, Wheel
from molla.sys.clock import monotonic_ms
from molla.sys.fd import close
from molla.sys.socket import (
    INADDR_LOOPBACK,
    SO_RCVBUF,
    SO_SNDBUF,
    connect,
    connect_unix,
    recv,
    send,
    set_buffer_size,
    socket_tcp,
)

comptime MAX_STEPS = 400
"""Passes to wait for something before calling it a failure. At a 5 ms timeout
that is two seconds, far longer than loopback needs and short enough that a
hang reads as a failed check rather than a hung job."""

comptime WAIT_MS = 4000
"""How long the threaded tests wait for a server that is running on its own
threads. Generous, because CI runners are shared and a scheduling delay is not
a bug."""


def _client(port: UInt16) raises -> Int:
    """A connected, non blocking client socket. EINPROGRESS is the normal
    answer on loopback and is not an error."""
    var fd = socket_tcp()
    _ = connect(fd, INADDR_LOOPBACK, port)
    return fd


def _send_pattern(fd: Int, first: Int, count: Int) -> Int:
    var chunk = stack_allocation[4096, UInt8]()
    var sent = 0
    while sent < count:
        var batch = min(4096, count - sent)
        for i in range(batch):
            chunk.unsafe_store(i, UInt8((first + sent + i) % 251))
        var wrote = send(fd, chunk, batch)
        if wrote <= 0:
            return sent
        sent += wrote
    return sent


def _drain(fd: Int, want: Int) -> Int:
    """Read up to `want` bytes, stopping when the socket goes quiet."""
    var buf = stack_allocation[4096, UInt8]()
    var got = 0
    var idle = 0
    while got < want and idle < 2000:
        var n = recv(fd, buf, min(4096, want - got))
        if n > 0:
            got += n
            idle = 0
            continue
        idle += 1
    return got


def _check_wheel(mut suite: Suite) raises:
    suite.group("net.wheel")

    var wheel = Wheel(0)
    var fired = List[Int]()

    var early = wheel.add(11, 100)
    var later = wheel.add(22, 1000)
    suite.check(wheel.pending == 2, "two timers are pending")
    suite.check(early != later, "and they have different ids")

    suite.check(wheel.advance(50, fired) == 0, "nothing fires before its tick")
    suite.check(wheel.advance(100, fired) == 1, "the first one fires on time")
    suite.check(fired[0] == 11, "and hands back its token")
    suite.check(wheel.pending == 1, "the other is still waiting")

    suite.check(wheel.advance(900, fired) == 0, "and does not fire early")
    suite.check(wheel.advance(1000, fired) == 1, "it fires on its own tick")
    suite.check(fired[0] == 22, "with its own token")
    suite.check(wheel.pending == 0, "and nothing is left")

    # A deadline past one revolution of level 0 has to cascade down before it
    # can fire. 6.4 seconds is where level 0 ends, so this one starts two
    # levels up.
    var far = Wheel(0)
    var distant = far.add(7, 30000)
    suite.check(far.deadline_of(distant) == 300, "a far deadline is 300 ticks")
    suite.check(far.advance(29000, fired) == 0, "and survives the cascades")
    suite.check(far.pending == 1, "still armed after 290 ticks")
    suite.check(far.advance(30000, fired) == 1, "then fires at its deadline")
    suite.check(fired[0] == 7, "with the token it was given")

    var cancelled = Wheel(0)
    var doomed = cancelled.add(3, 200)
    cancelled.cancel(doomed)
    suite.check(cancelled.pending == 0, "cancelling drops the count at once")
    suite.check(cancelled.advance(1000, fired) == 0, "and it never fires")
    cancelled.cancel(doomed)
    suite.check(cancelled.pending == 0, "cancelling twice is harmless")

    var reused = Wheel(0)
    var first = reused.add(1, 100)
    _ = reused.advance(100, fired)
    var second = reused.add(2, 100)
    suite.check(second == first, "a fired timer's slab entry is reused")

    var many = Wheel(0)
    for i in range(SLOTS * 4):
        _ = many.add(i, (i + 1) * TICK_MS)
    suite.check(many.pending == SLOTS * 4, "256 timers are armed")
    var total = 0
    for _ in range(SLOTS * 4):
        total += many.advance(many.now_tick() * TICK_MS + TICK_MS, fired)
    suite.check(total == SLOTS * 4, "and every one of them fires exactly once")
    suite.check(many.pending == 0, "leaving nothing behind")

    var idle = Wheel(0)
    suite.check(
        idle.next_timeout_ms() == -1, "an empty wheel does not need waking"
    )
    _ = idle.add(1, 5000)
    suite.check(idle.next_timeout_ms() == TICK_MS, "a busy one wakes per tick")


def _step_until_accepted(
    mut reactor: Reactor[EchoProtocol], count: Int
) raises -> Int:
    var steps = 0
    while reactor.accepted < count and steps < MAX_STEPS:
        _ = reactor.poll_once(5)
        steps += 1
    return steps


def _check_reactor(mut suite: Suite) raises:
    suite.group("net.reactor")

    var listener = open_listener(ListenAddress(UInt16(0)), False)
    var port = bound_port(listener)
    var reactor = Reactor[EchoProtocol](EchoProtocol(), 60000, 0)
    reactor.add_listener(listener)

    var client = _client(port)
    _ = _step_until_accepted(reactor, 1)
    suite.check(reactor.accepted == 1, "the reactor accepted a connection")
    suite.check(reactor.connection_count() == 1, "and holds it")
    suite.check(reactor.proto.opened == 1, "the protocol was told")

    var payload = 11
    suite.check(
        _send_pattern(client, 97, payload) == payload, "a client writes"
    )

    var back = stack_allocation[64, UInt8]()
    var got = -1
    var steps = 0
    while got <= 0 and steps < MAX_STEPS:
        _ = reactor.poll_once(5)
        got = recv(client, back, 64)
        steps += 1
    suite.check(got == payload, "and reads the same number of bytes back")

    var same = got == payload
    for i in range(payload):
        if back.unsafe_load(i) != UInt8((97 + i) % 251):
            same = False
    suite.check(same, "with the bytes unchanged")
    suite.check(reactor.proto.echoed == payload, "the protocol counted them")

    # A closed client must be reaped, and the protocol must hear about it.
    _ = close(client)
    steps = 0
    while reactor.connection_count() > 0 and steps < MAX_STEPS:
        _ = reactor.poll_once(5)
        steps += 1
    suite.check(reactor.connection_count() == 0, "a closed peer is reaped")
    suite.check(reactor.proto.closed == 1, "and the protocol was told")
    suite.check(reactor.closed_count == 1, "and the reactor counted it")

    reactor.shutdown()


def _check_backpressure(mut suite: Suite) raises:
    suite.group("net.reactor backpressure")

    # Half a megabyte, against sockets whose buffers are pinned small at both
    # ends. Left to itself the kernel decides how much it will hold, that number
    # differs by platform and grows over the life of a connection, and a test
    # that pushes bytes until it happens to fill either passes by accident or
    # tests nothing. An accepted socket inherits the listener's buffer sizes, so
    # setting it there covers the connection the reactor ends up with.
    var total = 512 * 1024
    var listener = open_listener(ListenAddress(UInt16(0)), False)
    set_buffer_size(listener, SO_SNDBUF, 8192)
    var port = bound_port(listener)
    var reactor = Reactor[EchoProtocol](EchoProtocol(), 60000, 0)
    reactor.add_listener(listener)

    var client = socket_tcp()
    set_buffer_size(client, SO_RCVBUF, 8192)
    _ = connect(client, INADDR_LOOPBACK, port)
    _ = _step_until_accepted(reactor, 1)

    # Push without reading until the whole path is full: the client's send
    # buffer, the reactor's read buffer, its output ring, the server's send
    # buffer and the client's receive buffer. Feeding bytes in step with the
    # reads would never stall anything and would test nothing. Socket buffer
    # sizes differ by platform and auto tune, so this stops on the condition it
    # is looking for rather than after a fixed number of bytes.
    var sent = 0
    var rounds = 0
    while (
        sent < total
        and reactor.conns[0].short_writes == 0
        and rounds < MAX_STEPS
    ):
        while sent < total:
            var want = min(8192, total - sent)
            var wrote = _send_pattern(client, 0, want)
            sent += wrote
            if wrote < want:
                break
        _ = reactor.poll_once(1)
        rounds += 1

    suite.check(
        reactor.conns[0].short_writes > 0, "the kernel took a short write"
    )
    suite.check(reactor.proto.stalled > 0, "and the ring filled up behind it")
    suite.check(
        reactor.conns[0].pending() > 0, "with a response still queued behind it"
    )

    var read_back = 0
    var buf = stack_allocation[8192, UInt8]()
    var steps = 0
    while read_back < total and steps < 200000:
        if sent < total:
            var want = min(8192, total - sent)
            sent += _send_pattern(client, 0, want)
        _ = reactor.poll_once(1)
        while True:
            var n = recv(client, buf, 8192)
            if n <= 0:
                break
            read_back += n
        steps += 1

    suite.check(sent == total, "the client sent half a megabyte")
    suite.check(read_back == total, "and got every byte of it back")
    suite.check(steps < 200000, "without spinning")

    _ = close(client)
    reactor.shutdown()


def _check_idle_timeout(mut suite: Suite) raises:
    suite.group("net.reactor idle timeout")

    # 300 ms, which is three ticks of the wheel. Long enough that a slow
    # machine does not close the connection before the test looks at it, short
    # enough that the test is not the slow part of the suite.
    var listener = open_listener(ListenAddress(UInt16(0)), False)
    var port = bound_port(listener)
    var reactor = Reactor[EchoProtocol](EchoProtocol(), 300, 0)
    reactor.add_listener(listener)

    var client = _client(port)
    _ = _step_until_accepted(reactor, 1)
    suite.check(reactor.connection_count() == 1, "a silent client is accepted")
    suite.check(reactor.timed_out == 0, "and is not closed immediately")

    var started = monotonic_ms()
    var steps = 0
    while reactor.timed_out == 0 and monotonic_ms() - started < 5000:
        _ = reactor.poll_once(5)
        steps += 1
    var waited = monotonic_ms() - started

    suite.check(reactor.timed_out == 1, "a connection that says nothing closes")
    suite.check(waited >= 250, "not before its deadline")
    suite.check(reactor.connection_count() == 0, "and the slot is given back")
    suite.check(reactor.proto.closed == 1, "the protocol heard about it")

    _ = close(client)
    reactor.shutdown()


def _check_unix_socket(mut suite: Suite) raises:
    suite.group("net.reactor unix socket")

    var path = String("/tmp/molla-reactor-test.sock")
    var listener = open_listener(ListenAddress(path), False)
    var reactor = Reactor[EchoProtocol](EchoProtocol(), 60000, 0)
    reactor.add_listener(listener)

    var client = connect_unix(path)
    _ = _step_until_accepted(reactor, 1)
    suite.check(reactor.accepted == 1, "a unix client is accepted")

    var wrote = _send_pattern(client, 65, 4)
    suite.check(wrote == 4, "and can write")

    var back = stack_allocation[16, UInt8]()
    var got = -1
    var steps = 0
    while got <= 0 and steps < MAX_STEPS:
        _ = reactor.poll_once(5)
        got = recv(client, back, 16)
        steps += 1
    suite.check(got == 4, "and reads the same bytes back")
    suite.check(back.unsafe_load(0) == UInt8(65), "over the same reactor")

    _ = close(client)
    reactor.shutdown()

    # Opening it again has to work, which means the stale path was removed
    # rather than left to fail the next bind with EADDRINUSE.
    var again = open_listener(ListenAddress(path), False)
    suite.check(again >= 0, "a leftover socket path does not block a rebind")
    _ = close(again)


def _check_server(mut suite: Suite) raises:
    suite.group("net.server")

    var workers = 4
    var server = Server[EchoProtocol](
        ListenAddress(UInt16(0)), workers, 60000, 0
    )
    var port = server.port
    suite.check(port != 0, "the server bound a port")
    suite.check(default_workers() >= 2, "the default worker count is sane")

    server.start()

    var clients = List[Int]()
    for _ in range(16):
        clients.append(_client(port))

    var deadline = monotonic_ms() + WAIT_MS
    while server.accepted() < 16 and monotonic_ms() < deadline:
        pass
    suite.check(server.accepted() == 16, "sixteen connections were accepted")

    var echoed = 0
    for i in range(len(clients)):
        var sent = _send_pattern(clients[i], i, 64)
        if sent != 64:
            continue
        if _drain(clients[i], 64) == 64:
            echoed += 1
    suite.check(echoed == 16, "and every one of them was echoed")

    var counts = server.spread()
    var busy = 0
    for i in range(len(counts)):
        if counts[i] > 0:
            busy += 1
    if SHARDED_ACCEPT:
        suite.check(
            len(counts) == workers, "every reactor has its own listener"
        )
        suite.check(busy >= 1, "and the kernel spread the connections")
    else:
        suite.check(busy == workers, "round robin used every reactor")

    for i in range(len(clients)):
        _ = close(clients[i])
    server.stop()
    suite.check(server.open_connections() == 0, "stopping closes everything")


def _check_server_unix(mut suite: Suite) raises:
    suite.group("net.server unix socket")

    var path = String("/tmp/molla-server-test.sock")
    var server = Server[EchoProtocol](ListenAddress(path), 2, 60000, 0)
    server.start()

    var client = connect_unix(path)
    var deadline = monotonic_ms() + WAIT_MS
    while server.accepted() < 1 and monotonic_ms() < deadline:
        pass
    suite.check(server.accepted() == 1, "a unix client reaches the server")

    _ = _send_pattern(client, 1, 32)
    suite.check(_drain(client, 32) == 32, "and is echoed on a worker thread")

    _ = close(client)
    server.stop()


def run(mut suite: Suite) raises:
    _check_wheel(suite)
    _check_reactor(suite)
    _check_backpressure(suite)
    _check_idle_timeout(suite)
    _check_unix_socket(suite)
    _check_server(suite)
    _check_server_unix(suite)
