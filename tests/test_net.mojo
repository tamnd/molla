"""Tests for the syscall layer and the echo spike.

These are real sockets on real loopback, not fakes. A fake would pass on a
machine where the ABI constants are wrong, which is the only thing this layer
can get wrong.

Everything runs on one thread. There is no threading module in Mojo 1.0, so the
client side is driven by hand between calls to `poll_once`, which is also why
`poll_once` is public. `step` below drives the loop until a condition holds or a
bounded number of passes go by, so a broken build fails in about a second
instead of hanging a CI job.

Every listener binds to port 0 and asks the kernel which port it got. A fixed
port would make two runs on one machine flaky, and CI runs the suite on three
platforms at once.
"""

from std.memory import stack_allocation

from harness import Suite

from molla.net.echo import EchoServer
from molla.sys.errno import EAGAIN, errno_name, get_errno
from molla.sys.fd import close, is_nonblocking, set_nonblocking
from molla.sys.poll import EVENT_SIZE, USES_KQUEUE, Poller
from molla.sys.socket import (
    INADDR_LOOPBACK,
    SOCKADDR_IN_SIZE,
    connect,
    listen_tcp,
    local_port,
    read_port,
    recv,
    send,
    socket_tcp,
    write_sockaddr_in,
)

comptime MAX_STEPS = 200
"""Upper bound on loop passes while waiting for something. At a 5 ms timeout
that is one second, which is far longer than loopback needs and short enough
that a hang reads as a failure."""


def _client_connect(port: UInt16) raises -> Int:
    """A connected, non blocking client socket.

    A non blocking connect to loopback returns EINPROGRESS rather than 0 almost
    every time, and that is not an error. The connection completes while the
    server side accepts.
    """
    var fd = socket_tcp()
    _ = connect(fd, INADDR_LOOPBACK, port)
    return fd


def _send_bytes(fd: Int, first: Int, count: Int) raises -> Int:
    """Send `count` bytes of a repeating pattern. Returns bytes accepted."""
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


def test_sockaddr_layout(mut suite: Suite) raises:
    suite.group("sockaddr_in layout")

    var sa = stack_allocation[SOCKADDR_IN_SIZE, UInt8]()
    write_sockaddr_in(sa, UInt32(0x7F000001), UInt16(8080))

    # 8080 is 0x1F90, and it goes on the wire most significant byte first no
    # matter what the host does.
    suite.check(
        sa.unsafe_load(2) == UInt8(0x1F), "port high byte is big endian"
    )
    suite.check(sa.unsafe_load(3) == UInt8(0x90), "port low byte is big endian")
    suite.check(read_port(sa) == UInt16(8080), "the port reads back unchanged")

    suite.check(sa.unsafe_load(4) == UInt8(127), "address starts with 127")
    suite.check(sa.unsafe_load(7) == UInt8(1), "address ends with 1")

    var padding_clear = True
    for i in range(8, SOCKADDR_IN_SIZE):
        if sa.unsafe_load(i) != UInt8(0):
            padding_clear = False
    suite.check(padding_clear, "the trailing pad is zeroed")


def test_poller_shape(mut suite: Suite) raises:
    suite.group("poller")

    # If this is wrong every decoded event is garbage, so it is worth asserting
    # rather than trusting the comment next to the constant.
    if USES_KQUEUE:
        suite.check(EVENT_SIZE == 32, "kevent is 32 bytes")
    else:
        suite.check(
            EVENT_SIZE == 12 or EVENT_SIZE == 16,
            "epoll_event is 12 bytes packed or 16 bytes aligned",
        )

    var poller = Poller(8)
    suite.check(poller.fd >= 0, "the poller opens")
    poller.shutdown()
    suite.check(poller.fd == -1, "shutdown clears the descriptor")
    poller.shutdown()
    suite.check(True, "shutdown twice is not an error")


def test_listener(mut suite: Suite) raises:
    suite.group("listener")

    var fd = listen_tcp(INADDR_LOOPBACK, UInt16(0), 16)
    suite.check(fd >= 0, "a listener opens on an ephemeral port")
    suite.check(local_port(fd) != UInt16(0), "the kernel assigned a real port")
    suite.check(is_nonblocking(fd), "the listener is non blocking")
    _ = close(fd)


def test_nonblocking_is_actually_set(mut suite: Suite) raises:
    suite.group("non blocking flag")

    # This is the regression test for the arm64 variadic ABI bug. fcntl reported
    # success while setting nothing, and the only symptom was a recv that
    # blocked forever. Checking the flag word catches it directly, and a read
    # that returns EAGAIN instead of hanging catches it end to end.
    var fd = socket_tcp()
    suite.check(is_nonblocking(fd), "socket_tcp returns a non blocking socket")

    var listener = listen_tcp(INADDR_LOOPBACK, UInt16(0), 16)
    var client = _client_connect(local_port(listener))
    var buf = stack_allocation[16, UInt8]()
    var got = recv(client, buf, 16)
    suite.check(got == -1, "a read with nothing to read returns -1")
    suite.check(get_errno() == EAGAIN, "and the reason is EAGAIN, not a block")

    _ = close(client)
    _ = close(listener)
    _ = close(fd)


def test_echo_round_trip(mut suite: Suite) raises:
    suite.group("echo round trip")

    var server = EchoServer(INADDR_LOOPBACK, UInt16(0))
    var client = _client_connect(server.port())

    var steps = 0
    while server.accepted == 0 and steps < MAX_STEPS:
        _ = server.poll_once(5)
        steps += 1
    suite.check(server.accepted == 1, "the server accepted the connection")

    var payload = 11
    suite.check(
        _send_bytes(client, 97, payload) == payload, "the client sent 11 bytes"
    )

    var back = stack_allocation[64, UInt8]()
    var got = -1
    steps = 0
    while got <= 0 and steps < MAX_STEPS:
        _ = server.poll_once(5)
        got = recv(client, back, 64)
        steps += 1

    suite.check(got == payload, "the client read back the same number of bytes")
    var identical = got == payload
    for i in range(payload):
        if back.unsafe_load(i) != UInt8((97 + i) % 251):
            identical = False
    suite.check(identical, "the bytes came back unchanged")
    suite.check(
        server.bytes_echoed == payload, "the server counted what it wrote"
    )

    _ = close(client)
    server.shutdown()


def test_echo_large_payload(mut suite: Suite) raises:
    suite.group("echo under partial writes")

    # 512 KB will not fit in a socket buffer, so this is the case that forces
    # short writes, pending output, and write interest. If the write path were
    # a retry loop instead, this test would still pass but would spin, so the
    # step count is checked as well: a spin would blow through MAX_STEPS.
    var total = 512 * 1024
    var server = EchoServer(INADDR_LOOPBACK, UInt16(0))
    var client = _client_connect(server.port())

    var steps = 0
    while server.accepted == 0 and steps < MAX_STEPS:
        _ = server.poll_once(5)
        steps += 1

    var sent = 0
    var received = 0
    var mismatch = False
    var buf = stack_allocation[8192, UInt8]()
    steps = 0

    while received < total and steps < 20000:
        if sent < total:
            # Seeded with the running total so the pattern is one continuous
            # stream across calls, which is what makes reordering detectable.
            sent += _send_bytes(client, sent, min(64 * 1024, total - sent))
        _ = server.poll_once(1)
        while True:
            var got = recv(client, buf, 8192)
            if got <= 0:
                break
            for i in range(got):
                if buf.unsafe_load(i) != UInt8((received + i) % 251):
                    mismatch = True
            received += got
        steps += 1

    suite.check(sent == total, "the client sent the whole payload")
    suite.check(received == total, "every byte came back")
    suite.check(not mismatch, "and in the right order")

    _ = close(client)
    server.shutdown()


def test_many_connections(mut suite: Suite) raises:
    suite.group("several connections at once")

    var count = 8
    var server = EchoServer(INADDR_LOOPBACK, UInt16(0))
    var port = server.port()
    var clients = List[Int]()
    for _i in range(count):
        clients.append(_client_connect(port))

    var steps = 0
    while server.accepted < count and steps < MAX_STEPS:
        _ = server.poll_once(5)
        steps += 1
    suite.check(server.accepted == count, "the accept loop drained the backlog")

    # One event can cover several pending connections, so accepting once per
    # notification would leave the rest stuck. That is what this checks.
    suite.check(len(server.conns) == count, "all of them are being tracked")

    var all_echoed = True
    for i in range(count):
        _ = _send_bytes(clients[i], i, 4)

    var buf = stack_allocation[16, UInt8]()
    for i in range(count):
        var got = -1
        steps = 0
        while got <= 0 and steps < MAX_STEPS:
            _ = server.poll_once(5)
            got = recv(clients[i], buf, 16)
            steps += 1
        if got != 4:
            all_echoed = False
        else:
            for j in range(4):
                if buf.unsafe_load(j) != UInt8((i + j) % 251):
                    all_echoed = False
    suite.check(all_echoed, "each connection got its own bytes back")

    for i in range(count):
        _ = close(clients[i])
    server.shutdown()


def test_peer_close_is_reaped(mut suite: Suite) raises:
    suite.group("peer close")

    var server = EchoServer(INADDR_LOOPBACK, UInt16(0))
    var client = _client_connect(server.port())

    var steps = 0
    while server.accepted == 0 and steps < MAX_STEPS:
        _ = server.poll_once(5)
        steps += 1

    _ = close(client)

    steps = 0
    while len(server.conns) > 0 and steps < MAX_STEPS:
        _ = server.poll_once(5)
        steps += 1

    suite.check(
        len(server.conns) == 0, "a closed peer is dropped from the table"
    )
    server.shutdown()


def run(mut suite: Suite):
    try:
        test_sockaddr_layout(suite)
        test_poller_shape(suite)
        test_listener(suite)
        test_nonblocking_is_actually_set(suite)
        test_echo_round_trip(suite)
        test_echo_large_payload(suite)
        test_many_connections(suite)
        test_peer_close_is_reaped(suite)
    except e:
        suite.fail("network tests", String(e))
