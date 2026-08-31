"""The soak test for the reactor.

Issue #10 is not done because an echo works. It is done when the reactor holds a
thousand connections with mixed idle and active traffic for an hour without
leaking descriptors and without latency drifting as the run goes on. Those are
three different failures and this measures all three.

The server runs on its own threads here, which the M0 soak could not do, so the
main thread is nothing but clients. That makes the round trip time worth
measuring: it is a real client handing a message to a real event loop on another
core and waiting for it to come back, rather than one thread pretending to be
both ends.

Mixed traffic means most connections are idle. That is the shape real traffic
has and it is the shape that finds bugs, because an idle connection is one the
reactor has to keep, arm a timer for, and never look at again. A server that
scans its whole connection table per pass looks fine with ten connections and
falls over at a thousand, and the way that shows up is the active connections
getting slower as the idle ones pile up rather than anything failing outright.

Drift is measured by cutting the run into ten segments and comparing the last
one against the first. Latency that is flat across ten segments of an hour is
the claim being made. Latency that climbs means something is accumulating, and
the number says how fast.

Descriptors are checked by opening a socket after teardown and reading the
number the kernel hands back, which is the lowest free one on both platforms. A
run that leaked even one comes back high.
"""

from std.memory import stack_allocation
from std.time import monotonic

from molla.net.listener import ListenAddress
from molla.net.protocol import EchoProtocol
from molla.net.server import Server
from molla.net.soak import max_rss_kb, probe_fd
from molla.sys.fd import close, set_nonblocking
from molla.sys.socket import INADDR_LOOPBACK, connect, recv, send, socket_tcp

comptime SEGMENTS = 10
"""Slices of the run compared against each other. Ten is enough to see a trend
and few enough that each one holds a meaningful number of samples."""

comptime BUCKETS = 28
"""Latency histogram buckets, one per power of two microseconds. Bucket 0 is
under a microsecond and bucket 27 is over two minutes, which covers everything
between a loopback round trip and a hang."""

comptime PAYLOAD = 32
comptime ACTIVE_IN = 8
"""One connection in eight sends. The rest connect and stay silent, which is
what a keep alive pool looks like between requests."""

comptime CONNECT_BATCH = 128
comptime IDLE_MARGIN_MS = 60000
"""Added to the run length to get the server's idle timeout, so a connection
that is idle on purpose is not closed for being idle."""


def _bucket(ns: Int) -> Int:
    """Which power of two microseconds a duration lands in."""
    var us = ns // 1000
    if us <= 0:
        return 0
    var index = 0
    var edge = 1
    while edge < us and index < BUCKETS - 1:
        edge = edge << 1
        index += 1
    return index


def _quantile(
    counts: List[Int], base: Int, total: Int, fraction_num: Int
) -> Int:
    """Upper edge in microseconds of the bucket holding a quantile.

    Reported as the bucket edge rather than an interpolated value, because a
    histogram this coarse cannot honestly claim more precision than the bucket
    it landed in.
    """
    if total == 0:
        return 0
    var want = (total * fraction_num) // 100
    if want < 1:
        want = 1
    var seen = 0
    for i in range(BUCKETS):
        seen += counts[base + i]
        if seen >= want:
            return 1 << i
    return 1 << (BUCKETS - 1)


def run_net_soak(connections: Int, seconds: Int) raises -> Int:
    """Hold `connections` against a threaded server for `seconds`.

    Returns 0 when everything held. Prints the report either way, because a
    soak that fails is only useful if it says which of the things it watches
    went wrong.
    """
    print(
        "net soak: "
        + String(connections)
        + " connections for "
        + String(seconds)
        + "s, one in "
        + String(ACTIVE_IN)
        + " sending"
    )

    var fd_before = probe_fd()
    var idle_timeout = seconds * 1000 + IDLE_MARGIN_MS
    var server = Server[EchoProtocol](
        ListenAddress(UInt16(0)), 0, idle_timeout, 0
    )
    var port = server.port
    print("  workers        " + String(server.workers))
    server.start()

    var clients = List[Int]()
    var active = List[Bool]()
    var sent_at = List[Int]()
    var connect_failures = 0

    for i in range(connections):
        var fd = socket_tcp()
        if fd < 0:
            connect_failures += 1
            continue
        _ = connect(fd, INADDR_LOOPBACK, port)
        set_nonblocking(fd)
        clients.append(fd)
        active.append(i % ACTIVE_IN == 0)
        sent_at.append(0)
        if (i + 1) % CONNECT_BATCH == 0:
            # The backlog is 1024 and the workers are accepting on their own
            # threads, but connecting a thousand sockets as fast as the loop
            # can issue them is still worth pacing.
            _ = server.accepted()

    var settle_deadline = Int(monotonic()) + 30_000_000_000
    while (
        server.accepted() < len(clients) and Int(monotonic()) < settle_deadline
    ):
        pass
    var accepted_at_start = server.accepted()
    print(
        "  connected "
        + String(len(clients))
        + ", accepted "
        + String(accepted_at_start)
    )
    if connect_failures > 0:
        print("  connect failures " + String(connect_failures))

    var rss_at_start = max_rss_kb()
    var open_at_start = server.open_connections()

    var counts = List[Int](length=SEGMENTS * BUCKETS, fill=0)
    var totals = List[Int](length=SEGMENTS, fill=0)
    var sums = List[Int](length=SEGMENTS, fill=0)
    var round_trips = 0
    var mismatches = 0

    var out_buf = stack_allocation[PAYLOAD, UInt8]()
    var in_buf = stack_allocation[4096, UInt8]()

    var started = Int(monotonic())
    var length_ns = seconds * 1_000_000_000
    var deadline = started + length_ns

    while True:
        var now = Int(monotonic())
        if now >= deadline:
            break
        var segment = ((now - started) * SEGMENTS) // length_ns
        if segment >= SEGMENTS:
            segment = SEGMENTS - 1

        for i in range(len(clients)):
            if not active[i] or sent_at[i] != 0:
                continue
            var mark = UInt8(i % 251)
            for j in range(PAYLOAD):
                out_buf.unsafe_store(j, mark)
            var at = Int(monotonic())
            if send(clients[i], out_buf, PAYLOAD) == PAYLOAD:
                sent_at[i] = at

        for i in range(len(clients)):
            if sent_at[i] == 0:
                continue
            var got = recv(clients[i], in_buf, 4096)
            if got <= 0:
                continue
            var elapsed = Int(monotonic()) - sent_at[i]
            sent_at[i] = 0
            round_trips += 1
            var mark = UInt8(i % 251)
            for j in range(got):
                if in_buf.unsafe_load(j) != mark:
                    mismatches += 1
            counts[segment * BUCKETS + _bucket(elapsed)] += 1
            totals[segment] += 1
            sums[segment] += elapsed

    var elapsed_ns = Int(monotonic()) - started
    var rss_at_end = max_rss_kb()
    var open_at_end = server.open_connections()

    for i in range(len(clients)):
        _ = close(clients[i])

    # Give the workers a moment to notice a thousand closed peers before asking
    # how many connections are left.
    var drain_deadline = Int(monotonic()) + 30_000_000_000
    while server.open_connections() > 0 and Int(monotonic()) < drain_deadline:
        pass
    var open_after = server.open_connections()

    server.stop()
    var fd_after = probe_fd()

    print("  elapsed        " + String(elapsed_ns // 1_000_000) + " ms")
    print(
        "  held           "
        + String(open_at_end)
        + " connections at the end of the run, "
        + String(open_at_start)
        + " at the start"
    )
    print("  round trips    " + String(round_trips))
    print("  mismatched     " + String(mismatches))
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

    print("  latency by segment, microseconds")
    print("    segment  samples      mean       p50       p99")
    for s in range(SEGMENTS):
        var mean = (sums[s] // totals[s] // 1000) if totals[s] > 0 else 0
        print(
            "    "
            + String(s)
            + "        "
            + String(totals[s])
            + "  "
            + String(mean)
            + "  "
            + String(_quantile(counts, s * BUCKETS, totals[s], 50))
            + "  "
            + String(_quantile(counts, s * BUCKETS, totals[s], 99))
        )

    var first_p99 = _quantile(counts, 0, totals[0], 99)
    var last_p99 = _quantile(
        counts, (SEGMENTS - 1) * BUCKETS, totals[SEGMENTS - 1], 99
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
    if accepted_at_start != len(clients):
        print(
            "  FAIL: accepted "
            + String(accepted_at_start)
            + " of "
            + String(len(clients))
        )
        ok = False
    if open_at_end != len(clients):
        print(
            "  FAIL: "
            + String(open_at_end)
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
    if open_after != 0:
        print(
            "  FAIL: "
            + String(open_after)
            + " connections were not reaped after the clients closed"
        )
        ok = False
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
    # Four times is a wide gate on purpose. The histogram is powers of two, so
    # two neighbouring buckets are already a factor of two apart and a run that
    # crosses one boundary is noise rather than drift.
    if last_p99 > first_p99 * 4:
        print(
            "  FAIL: p99 drifted from "
            + String(first_p99)
            + " to "
            + String(last_p99)
            + " microseconds"
        )
        ok = False

    if ok:
        print("  net soak passed")
        return 0
    return 1
