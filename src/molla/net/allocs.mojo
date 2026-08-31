"""The allocation assertion, as a command, so CI can fail on it.

Issue #17. The design claims the request path allocates nothing in steady
state, and a claim like that stops being true within a month unless something
checks it. This is the something.

What it does is run a mixed load twice. The first pass is a warm up: every
buffer grows to the size the traffic needs, every connection slot in the
reactor gets made, every response the writer will ever build is built once. The
counter is read after that, the identical load runs again, and the counter has
to read exactly the same number. Not nearly the same. The same.

The load is mixed on purpose, because the interesting allocation is the one on
a path a simpler test does not take. So it covers a plain GET, a HEAD, a 404, a
POST with a body, a chunked POST, a pipelined batch, and both streaming
routes. Anything that grows a buffer the second time round is a regression, and
anything that grows it only on the fifth kind of request would be invisible to a
test that only sent the first.

Two things this cannot see, and both are worth saying out loud. Mojo's own
allocations do not go through `molla.sys.mem` and cannot be counted, so what is
measured is molla's heap traffic rather than the process's. And the counter is
not atomic, which is why this runs one worker: two workers counting against one
counter would lose updates and the number would come out too low, which is the
direction that hides a bug.
"""

from std.memory import stack_allocation

from molla.http.protocol import HttpProtocol
from molla.net.context import ServerContext
from molla.net.listener import ListenAddress
from molla.net.server import Server
from molla.sys.clock import monotonic_ms
from molla.sys.fd import close
from molla.sys.mem import AllocCounter, keep
from molla.sys.signal import ignore_sigpipe
from molla.sys.socket import INADDR_LOOPBACK, connect, recv, send, socket_tcp
from molla.sys.thread import sleep_ms

comptime WAIT_MS = 10000
comptime CHUNK = 8192
comptime STREAM_EVENTS = 4
"""Few, because the point is that a stream allocates nothing per event and not
how fast it goes. Four is enough for the pending and wire buffers to reach
their size on the warm up pass."""


def _client(port: UInt16) raises -> Int:
    var fd = socket_tcp()
    _ = connect(fd, INADDR_LOOPBACK, port)
    return fd


def _send_text(fd: Int, text: StringSpan) -> Int:
    var n = text.byte_length()
    var p = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(text.unsafe_ptr())
    )
    var sent = 0
    var deadline = monotonic_ms() + WAIT_MS
    while sent < n and monotonic_ms() < deadline:
        var wrote = send(fd, p.unsafe_offset(sent), n - sent)
        if wrote <= 0:
            _ = sleep_ms(1)
            continue
        sent += wrote
    return sent


def _await_first(fd: Int) -> Int:
    """Read until this connection has been answered at all.

    Every connection in a round has to be accepted before any of them closes,
    or the number of slots the reactor builds is a matter of timing rather than
    a matter of how many connections there are. A slot costs three allocations
    the first time it is built and nothing afterwards, so a warm up that
    happened to build fifty five of them leaves nine to be built during the
    steady pass, and the whole measurement turns into a coin toss.

    Waiting for the first byte from each connection is the cheapest proof that
    the server has all of them open at once. Nothing asks the server to close
    until that has happened, which is the other half of the same point and is
    why `_closer` is sent separately.
    """
    var buf = stack_allocation[CHUNK, UInt8]()
    var deadline = monotonic_ms() + WAIT_MS
    while monotonic_ms() < deadline:
        var got = recv(fd, buf, CHUNK)
        if got > 0:
            return got
        if got == 0:
            return 0
        _ = sleep_ms(1)
    return 0


def _drain_until_closed(fd: Int) -> Int:
    """Read until the peer hangs up. Returns how many bytes came back.

    The last request on every connection asks for `Connection: close`, so the
    end of the answer is the end of the socket and there is no framing to parse
    on the client side.
    """
    var buf = stack_allocation[CHUNK, UInt8]()
    var total = 0
    var deadline = monotonic_ms() + WAIT_MS
    while monotonic_ms() < deadline:
        var got = recv(fd, buf, CHUNK)
        if got > 0:
            total += got
            continue
        if got == 0:
            break
        _ = sleep_ms(1)
    return total


def _load() -> String:
    """One connection's worth of traffic, as a single pipelined batch.

    Every kind of request molla can currently answer, in one string. Building it
    once and sending it whole means the client is not the thing being measured.
    The request that closes the connection is sent afterwards, by `_closer`, for
    the reason given there.
    """
    var out = String("")
    out += "GET / HTTP/1.1\r\nHost: molla\r\n\r\n"
    out += "HEAD / HTTP/1.1\r\nHost: molla\r\n\r\n"
    out += "GET /healthz HTTP/1.1\r\nHost: molla\r\n\r\n"
    out += "GET /nowhere HTTP/1.1\r\nHost: molla\r\n\r\n"
    out += (
        "POST / HTTP/1.1\r\nHost: molla\r\nContent-Length: 11\r\n\r\nhello"
        " molla"
    )
    out += (
        "POST / HTTP/1.1\r\nHost: molla\r\nTransfer-Encoding: chunked\r\n\r\n"
        "5\r\nhello\r\n6\r\n molla\r\n0\r\n\r\n"
    )
    # Pipelined, because a batch in one segment takes the parse loop round
    # again without a new readiness event, which is a different path through
    # the pump than one request per read.
    for _ in range(8):
        out += "GET / HTTP/1.1\r\nHost: molla\r\n\r\n"
    out += "GET /stream/ndjson HTTP/1.1\r\nHost: molla\r\n\r\n"
    out += "GET /stream/sse HTTP/1.1\r\nHost: molla\r\n\r\n"
    return out^


def _closer() -> String:
    """The request that ends a connection, sent separately from the batch.

    It is not the last line of the batch because a connection the server has
    already closed is a slot the server can already reuse, and with sixty four
    clients the early ones would be closed before the late ones were accepted.
    The reactor would then serve the whole load out of forty or fifty slots, a
    different number every run, and the pass that happened to build more of them
    would look like the regression.
    """
    return String("GET / HTTP/1.1\r\nHost: molla\r\nConnection: close\r\n\r\n")


struct AllocReport(Copyable, ImplicitlyCopyable, Movable):
    """What the two passes cost. A struct rather than printed numbers, so the
    suite can assert on the same measurement the command prints."""

    var warm_allocations: Int
    var warm_bytes_read: Int
    var steady_allocations: Int
    """The number the whole exercise is about. Anything but zero is a
    regression."""

    var steady_bytes_grown: Int
    var steady_bytes_read: Int
    var workers: Int
    var drained: Bool

    def __init__(out self):
        self.warm_allocations = 0
        self.warm_bytes_read = 0
        self.steady_allocations = -1
        self.steady_bytes_grown = 0
        self.steady_bytes_read = 0
        self.workers = 0
        self.drained = False

    def served_the_same(self) -> Bool:
        """Both passes read the same answers back.

        Checked because a pass that quietly answered nothing would allocate
        nothing too, and would otherwise look like the best result this command
        can produce.
        """
        return self.warm_bytes_read > 0 and (
            self.steady_bytes_read == self.warm_bytes_read
        )

    def ok(self) -> Bool:
        return (
            self.steady_allocations == 0
            and self.served_the_same()
            and self.drained
        )


def measure_allocs(connections: Int, rounds: Int) raises -> AllocReport:
    """Run the mixed load twice against a real server and count both passes."""
    _ = ignore_sigpipe()
    var report = AllocReport()

    var counter = AllocCounter()
    if counter.raw() == 0:
        return report^

    # One worker. The counter is not atomic, and a second worker counting
    # against it would lose updates, which is the direction that hides a bug
    # rather than the direction that invents one.
    var context = ServerContext(1, 60000, counter.raw())
    var server = Server[HttpProtocol](ListenAddress(UInt16(0)), context)
    for i in range(len(server.reactors)):
        server.reactors[i].proto.configure_stream(STREAM_EVENTS)
    var port = server.port
    report.workers = server.workers
    server.start()

    var batch = _load()
    report.warm_bytes_read = _pass(port, connections, rounds, batch)
    var after_warm = counter.total()
    var bytes_after_warm = counter.bytes()
    report.warm_allocations = after_warm

    report.steady_bytes_read = _pass(port, connections, rounds, batch)
    report.steady_allocations = counter.total() - after_warm
    report.steady_bytes_grown = counter.bytes() - bytes_after_warm

    report.drained = server.drain().clean
    keep(counter)
    return report^


def run_allocs(connections: Int, rounds: Int) raises -> Int:
    """Run the measurement and print it. Zero when nothing was allocated.

    Prints the numbers either way, because a failure is only useful if it says
    how much.
    """
    print("allocs", connections, "connections,", rounds, "rounds")
    var report = measure_allocs(connections, rounds)
    print("  workers       ", report.workers)
    print(
        "  warm up       ",
        report.warm_allocations,
        "allocations,",
        report.warm_bytes_read,
        "bytes read",
    )
    print(
        "  steady state  ",
        report.steady_allocations,
        "allocations,",
        report.steady_bytes_read,
        "bytes read",
    )
    print("  heap grew by  ", report.steady_bytes_grown, "bytes")
    if not report.served_the_same():
        print("  the two passes did not read the same answers back")
    if not report.drained:
        print("  the server did not drain cleanly")
    print("  result        ", "pass" if report.ok() else "fail")
    return 0 if report.ok() else 1


def _pass(
    port: UInt16, connections: Int, rounds: Int, batch: StringSpan
) raises -> Int:
    """One pass of the load. Returns the total bytes read back.

    Every connection in a round is opened before any of them is written to, all
    of them are written to before any of them is read, and every one of them has
    been answered before any of them closes. That ordering is the difference
    between exercising the reactor's whole table and exercising one slot.
    Opening and finishing one connection at a time lets the server accept,
    answer and free the same slot every time, which proves that one slot is
    reused and nothing about the other sixty three. Not waiting for the answers
    before closing is worse than that: the slot count then depends on how far
    ahead of the accept loop the client got, and the measurement stops being
    repeatable.

    The byte count is compared between passes, which is the check that the
    second pass really did the same work. A pass that quietly answered nothing
    would allocate nothing too, and would otherwise look like a success.
    """
    var total = 0
    for _ in range(rounds):
        var fds = List[Int]()
        for _ in range(connections):
            fds.append(_client(port))
        for i in range(len(fds)):
            _ = _send_text(fds[i], batch)
        for i in range(len(fds)):
            total += _await_first(fds[i])
        var closer = _closer()
        for i in range(len(fds)):
            _ = _send_text(fds[i], closer)
        for i in range(len(fds)):
            total += _drain_until_closed(fds[i])
            _ = close(fds[i])
    return total
