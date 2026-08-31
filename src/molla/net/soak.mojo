"""The soak test for the echo spike.

Issue #2 is not done because a round trip works. It is done when the server
holds a thousand concurrent connections for a minute on macOS and Linux without
leaking descriptors or memory. That is the number worth measuring, because
everything that goes wrong in a socket server goes wrong at the point where the
connection table stops being small.

Both sides run on one thread, since there is no threading module to put clients
on another. So a pass looks like: give each client a chance to write, run one
turn of the server loop, give each client a chance to read. That is slower than
real clients would be and it does not measure throughput, which is issue #3's
job. It measures whether the thing stays correct and stays the same size.

Descriptor leaks are caught by opening a socket after teardown and looking at
the number the kernel hands back. Descriptors are allocated lowest free first on
both platforms, so if a run of a thousand connections leaked even one, the probe
comes back high instead of near where the listener started.

Memory is measured with getrusage, which reports a high water mark rather than
current usage. That is weaker than it sounds and the report says so. What it can
prove is the useful direction: if peak memory after the soak is the same as peak
memory once the connections were up, nothing grew while the loop ran.
"""

from std.ffi import c_int, external_call
from std.memory import stack_allocation
from std.sys.info import CompilationTarget
from std.time import monotonic

from molla.net.echo import EchoServer
from molla.sys.errno import EAGAIN, errno_name, get_errno
from molla.sys.fd import close
from molla.sys.socket import (
    INADDR_LOOPBACK,
    connect,
    recv,
    send,
    socket_tcp,
)

comptime RUSAGE_SELF = 0
comptime RU_MAXRSS_OFFSET = 32
"""struct rusage opens with two timevals, 16 bytes each on both platforms, and
ru_maxrss is the long right after them."""

comptime PAYLOAD = 8
"""Bytes per client per pass. Small on purpose. The point is a thousand
independent conversations, not volume."""

comptime CONNECT_BATCH = 64
"""How many clients to connect before letting the server accept. The listen
backlog is 128, so connecting all thousand at once would drop some."""


def max_rss_kb() -> Int:
    """Peak resident set size in kilobytes, or -1 if it cannot be read.

    macOS reports ru_maxrss in bytes and Linux reports it in kilobytes. Same
    field, same struct, different unit, and no flag to ask which.
    """
    var buf = stack_allocation[256, UInt8]()
    for i in range(256):
        buf.unsafe_store(i, UInt8(0))
    var rc = Int(external_call["getrusage", c_int](c_int(RUSAGE_SELF), buf))
    if rc < 0:
        return -1

    var value: UInt64 = 0
    for i in range(8):
        value |= UInt64(buf.unsafe_load(RU_MAXRSS_OFFSET + i)) << (
            UInt64(i) * 8
        )

    comptime if CompilationTarget.is_macos():
        return Int(value) // 1024
    else:
        return Int(value)


def probe_fd() -> Int:
    """Open and close a socket, returning the number it was given.

    A cheap leak check. The kernel hands out the lowest free descriptor, so
    after everything is closed this should land near where the run started.
    """
    var fd = Int(external_call["socket", c_int](c_int(2), c_int(1), c_int(0)))
    if fd < 0:
        return -1
    _ = close(fd)
    return fd


def run_soak(connections: Int, seconds: Int) raises -> Int:
    """Hold `connections` open for `seconds`. Returns 0 if everything held.

    Prints a report either way, because a soak that fails is only useful if it
    says which of the several things it watches went wrong.
    """
    print(
        "soak: "
        + String(connections)
        + " connections for "
        + String(seconds)
        + "s"
    )

    var fd_before = probe_fd()
    var server = EchoServer(INADDR_LOOPBACK, 0)
    var port = server.port()

    var clients = List[Int]()
    var outstanding = List[Int]()
    var connect_failures = 0

    for i in range(connections):
        var fd = socket_tcp()
        if fd < 0:
            connect_failures += 1
            continue
        var rc = connect(fd, INADDR_LOOPBACK, port)
        if (
            rc < 0
            and get_errno() != EAGAIN
            and errno_name(get_errno()) != "EINPROGRESS"
        ):
            connect_failures += 1
            _ = close(fd)
            continue
        clients.append(fd)
        outstanding.append(0)
        if (i + 1) % CONNECT_BATCH == 0:
            _ = server.poll_once(1)

    # Let the accept loop catch up before the clock starts. Bounded so a broken
    # accept path reports instead of hanging.
    var settle = 0
    while server.accepted < len(clients) and settle < 10000:
        _ = server.poll_once(1)
        settle += 1

    print(
        "  connected "
        + String(len(clients))
        + ", accepted "
        + String(server.accepted)
    )
    if connect_failures > 0:
        print("  connect failures " + String(connect_failures))

    var rss_at_start = max_rss_kb()
    var round_trips = 0
    var mismatches = 0
    var short_reads = 0
    var passes = 0

    var out_buf = stack_allocation[PAYLOAD, UInt8]()
    var in_buf = stack_allocation[4096, UInt8]()

    var started = Int(monotonic())
    var deadline = started + seconds * 1_000_000_000

    while Int(monotonic()) < deadline:
        for i in range(len(clients)):
            # One outstanding message per client. Waiting for the echo before
            # sending again is what keeps this a correctness test rather than a
            # queue depth test.
            if outstanding[i] == 0:
                var mark = UInt8(i % 251)
                for j in range(PAYLOAD):
                    out_buf.unsafe_store(j, mark)
                var wrote = send(clients[i], out_buf, PAYLOAD)
                if wrote > 0:
                    outstanding[i] = wrote

        _ = server.poll_once(0)

        for i in range(len(clients)):
            if outstanding[i] == 0:
                continue
            var got = recv(clients[i], in_buf, 4096)
            if got <= 0:
                continue
            var mark = UInt8(i % 251)
            for j in range(got):
                if in_buf.unsafe_load(j) != mark:
                    mismatches += 1
            if got < outstanding[i]:
                short_reads += 1
            outstanding[i] -= min(got, outstanding[i])
            if outstanding[i] == 0:
                round_trips += 1

        passes += 1

    var elapsed_ns = Int(monotonic()) - started
    var rss_at_end = max_rss_kb()
    var conns_held = len(server.conns)

    for i in range(len(clients)):
        _ = close(clients[i])

    var drain = 0
    while len(server.conns) > 0 and drain < 10000:
        _ = server.poll_once(1)
        drain += 1
    var conns_after = len(server.conns)

    server.shutdown()
    var fd_after = probe_fd()

    print(
        "  elapsed        "
        + String(elapsed_ns // 1_000_000)
        + " ms over "
        + String(passes)
        + " passes"
    )
    print(
        "  held           "
        + String(conns_held)
        + " connections at the end of the run"
    )
    print("  round trips    " + String(round_trips))
    print("  bytes echoed   " + String(server.bytes_echoed))
    print("  mismatched     " + String(mismatches))
    print(
        "  short reads    "
        + String(short_reads)
        + " (expected, a stream has no message boundaries)"
    )
    print(
        "  peak rss       "
        + String(rss_at_start)
        + " kB after connect, "
        + String(rss_at_end)
        + " kB at the end"
    )
    print(
        "  probe fd       "
        + String(fd_before)
        + " before, "
        + String(fd_after)
        + " after"
    )

    var ok = True
    if len(clients) != connections:
        print(
            "  FAIL: only "
            + String(len(clients))
            + " of "
            + String(connections)
            + " clients connected"
        )
        ok = False
    if server.accepted != len(clients):
        print(
            "  FAIL: accepted "
            + String(server.accepted)
            + " of "
            + String(len(clients))
        )
        ok = False
    if conns_held != len(clients):
        print(
            "  FAIL: "
            + String(conns_held)
            + " connections survived the run, expected "
            + String(len(clients))
        )
        ok = False
    if mismatches != 0:
        print("  FAIL: " + String(mismatches) + " bytes came back wrong")
        ok = False
    if round_trips == 0:
        print("  FAIL: no round trips completed")
        ok = False
    if conns_after != 0:
        print(
            "  FAIL: "
            + String(conns_after)
            + " connections were not reaped after the clients closed"
        )
        ok = False
    # A few descriptors of slack for whatever the runtime opened along the way.
    if fd_after < 0 or fd_after > fd_before + 4:
        print(
            "  FAIL: descriptors leaked, probe went from "
            + String(fd_before)
            + " to "
            + String(fd_after)
        )
        ok = False
    if rss_at_end > rss_at_start * 2 and rss_at_end - rss_at_start > 16384:
        print("  FAIL: peak memory more than doubled during the run")
        ok = False

    if ok:
        print("  soak passed")
        return 0
    return 1
