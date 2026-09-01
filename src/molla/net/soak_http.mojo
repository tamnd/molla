"""The one hour soak on the systems layer, with real HTTP traffic.

Issue #18. Everything underneath this has its own tests and the reactor has its
own soak, and none of that answers the question this one asks: does the whole
stack stay the same size and the same speed for an hour while five kinds of
client that annoy it in different ways are all connected at once.

The five kinds are the point, and they run together rather than one after
another. Keep alive clients send request after request down one connection, so
the parser and the writer are reused thousands of times without a fresh
connection to hide state behind, and half of those requests carry a body.
Streaming clients ask for a chunked response and read it whole. Slow readers
ask for the same thing and then read sixty four bytes every hundred
milliseconds, which fills the write ring and holds it full, which is the only
way to keep the backpressure path warm. Abrupt clients send a request and close
without reading a byte of the answer, which is the write to a dead socket that
a server has to survive. Oversized clients announce a body far larger than the
limit and are told 413, which is the error path with a close on the end of it.

Four things are watched, and they are the four ways a server dies slowly rather
than loudly.

Resident memory, sampled per segment on Linux where the current figure can be
read, and as a peak on both platforms. Flat resident memory across ten segments
of an hour is the claim.

Descriptors, by opening a socket after teardown and reading the number the
kernel hands back. This soak reconnects constantly, tens of thousands of times
over an hour, so a leak of one descriptor per connection would be obvious and a
leak of one in a thousand would still show.

Queues, meaning the log ring and the reactor's connection table. A ring that
ends the run with bytes still in it is a flush that stopped keeping up, and
connections that survive the clients closing are a reap that stopped happening.

Latency drift, from the shared histogram in `molla.net.latency`, comparing the
last segment of the run against the first.

The clients all run on this thread while the server runs on its own, which is
what makes the round trip time worth measuring: a real client handing a request
to a real event loop on another core and waiting for the answer to come back.
"""

from std.memory import stack_allocation
from std.time import monotonic

from molla.http.protocol import HttpProtocol
from molla.net.context import ServerContext
from molla.net.latency import LatencyLog
from molla.net.listener import ListenAddress
from molla.net.server import Server
from molla.net.soak import max_rss_kb, probe_fd
from molla.ops.config import LEVEL_WARN
from molla.ops.log import LogPump, LogSink
from molla.ops.metrics import (
    M_CONNECTIONS_ACCEPTED,
    M_HANDLER_ERRORS,
    M_REQUESTS,
    M_RESPONSES_2XX,
    M_RESPONSES_4XX,
    M_RESPONSES_5XX,
    Metrics,
)
from molla.sys.fd import close, set_nonblocking
from molla.sys.file import close_fd, open_read, pread_all
from molla.sys.signal import ignore_sigpipe
from molla.sys.socket import INADDR_LOOPBACK, connect, recv, send, socket_tcp

comptime SEGMENTS = 10
"""Slices of the run compared against each other. Ten is enough to see a trend
and few enough that each one holds a meaningful number of samples."""

comptime ROLE_KEEPALIVE = 0
comptime ROLE_STREAM = 1
comptime ROLE_SLOW = 2
comptime ROLE_ABRUPT = 3
comptime ROLE_OVERSIZED = 4
comptime ROLE_COUNT = 5

comptime SOAK_MAX_BODY = 4096
"""Small, so an oversized request is a few hundred bytes of headers rather than
sixty four megabytes of traffic to prove a limit that is not about size."""

comptime OVERSIZED_LENGTH = 1048576
"""What the oversized client announces. Never sent, because the server answers
413 on the header and does not wait for a body it has already refused."""

comptime MAX_REQUESTS_PER_CONNECTION = 100000000
"""Effectively no limit. The default cap exists so a real server recycles a
connection eventually, and a keep alive client here is meant to hold one
connection for the whole hour so that anything per connection has nowhere to
hide."""

comptime KEEPALIVE_THINK_MS = 5
comptime STREAM_THINK_MS = 50
comptime SLOW_READ_BYTES = 64
comptime SLOW_PAUSE_MS = 100
comptime ABRUPT_LINGER_MS = 2
comptime RECONNECT_MS = 20

comptime READ_CHUNK = 16384
comptime MAX_PENDING = 262144
"""How much unparsed response one client may hold. Reaching it means the
framing came apart, which is a failure rather than a reason to grow."""

comptime SEND_TRIES = 200
comptime STALL_NS = 5_000_000_000
"""How long a client may fail to get a request out before the run says so. A
loopback handshake finishes in microseconds, so anything near this is a socket
that is stuck rather than one that is busy."""

comptime CONNECT_BATCH = 64
comptime SETTLE_NS = 30_000_000_000
comptime IDLE_MARGIN_MS = 60000
"""Added to the run length to get the server's idle timeout, so a slow reader
pausing for a hundred milliseconds is never mistaken for an idle connection."""


def current_rss_kb() -> Int:
    """Resident set size right now, in kilobytes, or -1 where it cannot be
    read.

    Linux publishes it in `/proc/self/statm`, field two, in pages. macOS has no
    equivalent file and the mach call that answers the same question is a
    different kind of dependency, so on macOS this returns -1 and the report
    leans on the peak from getrusage. Saying which platform gets the stronger
    check is better than pretending both do.
    """
    var opened = open_read("/proc/self/statm")
    if opened.is_err():
        return -1
    var fd = opened.value
    var buf = stack_allocation[128, UInt8]()
    var got = pread_all(fd, buf, 128, 0)
    _ = close_fd(fd)
    if got.is_err() or got.value <= 0:
        return -1

    # "size resident shared text lib data dt", in pages. The second field is
    # the one that answers how much of this process is really in memory.
    var field = 0
    var value = 0
    var seen = False
    for i in range(got.value):
        var c = buf.unsafe_load(i)
        if c >= UInt8(48) and c <= UInt8(57):
            value = value * 10 + Int(c) - 48
            seen = True
            continue
        if seen:
            if field == 1:
                return (value * 4096) // 1024
            field += 1
            value = 0
            seen = False
    return -1


def _find(buf: List[UInt8], start: Int, needle: StringSpan) -> Int:
    """Where `needle` starts in `buf` at or after `start`, or -1.

    Plain, and quadratic in the worst case. The haystack is one HTTP response
    and the needle is a handful of bytes, so the clever version would cost more
    to read than it saves.
    """
    var n = needle.byte_length()
    if n == 0 or len(buf) < n:
        return -1
    var p = needle.unsafe_ptr()
    for i in range(start, len(buf) - n + 1):
        var hit = True
        for j in range(n):
            if buf[i + j] != p.unsafe_load(j):
                hit = False
                break
        if hit:
            return i
    return -1


def _matches_ci(buf: List[UInt8], at: Int, text: StringSpan) -> Bool:
    """Case insensitive compare of the buffer at `at` against a lower case
    literal. Only the buffer side is folded, since every caller here passes a
    literal that is already lower case."""
    var n = text.byte_length()
    if at < 0 or at + n > len(buf):
        return False
    var p = text.unsafe_ptr()
    for j in range(n):
        var got = buf[at + j]
        if got >= UInt8(65) and got <= UInt8(90):
            got += 32
        if got != p.unsafe_load(j):
            return False
    return True


def _header_value_int(
    buf: List[UInt8], start: Int, head_end: Int, name: StringSpan
) -> Int:
    """The integer value of a header, or -1 when it is not there.

    Only used for Content-Length, and only on responses this soak asked a
    server it started itself for, so a header that appears twice or has
    something other than digits after it is a bug in that server rather than an
    attack to defend against.
    """
    for i in range(start, head_end):
        if not _matches_ci(buf, i, name):
            continue
        var at = i + name.byte_length()
        while at < head_end and buf[at] == UInt8(32):
            at += 1
        var value = 0
        var seen = False
        while at < head_end and buf[at] >= UInt8(48) and buf[at] <= UInt8(57):
            value = value * 10 + Int(buf[at]) - 48
            seen = True
            at += 1
        return value if seen else -1
    return -1


def _has_header(
    buf: List[UInt8], start: Int, head_end: Int, name: StringSpan
) -> Bool:
    for i in range(start, head_end):
        if _matches_ci(buf, i, name):
            return True
    return False


struct Wire(Movable):
    """One client's unread response bytes, and enough parsing to find the end
    of a response.

    A soak client has to know when an answer finished, and on a keep alive
    connection the only thing that says so is the framing. So this understands
    Content-Length and chunked and nothing else, which is what molla sends. It
    is deliberately not the parser in `molla.http`: that one faces the network,
    this one reads answers, and keeping them apart means a bug in one cannot
    quietly cover for the other.
    """

    var buf: List[UInt8]
    var at: Int
    """First byte of the response being read. Bytes before it are done with,
    and they are dropped when the buffer empties rather than after every
    response, because moving the tail down on every keep alive request is the
    one place this could get expensive."""

    def __init__(out self):
        self.buf = List[UInt8](capacity=READ_CHUNK)
        self.at = 0

    def clear(mut self):
        self.buf.clear()
        self.at = 0

    def pending(self) -> Int:
        return len(self.buf) - self.at

    def feed[o: MutOrigin](mut self, data: Pointer[UInt8, o], count: Int):
        for i in range(count):
            self.buf.append(data.unsafe_load(i))

    def next_status(mut self) -> Int:
        """The status of the next complete response, or -1 for not yet.

        Returns -2 when the bytes cannot be a response molla would send, which
        is a failure of the run rather than something to wait out.
        """
        if self.pending() < 13:
            return -1
        if not _matches_ci(self.buf, self.at, "http/1."):
            return -2
        var head = _find(self.buf, self.at, "\r\n\r\n")
        if head < 0:
            return -1 if self.pending() < MAX_PENDING else -2

        var status = 0
        for i in range(self.at + 9, self.at + 12):
            var c = self.buf[i]
            if c < UInt8(48) or c > UInt8(57):
                return -2
            status = status * 10 + Int(c) - 48

        var body = head + 4
        var length = _header_value_int(
            self.buf, self.at, head, "content-length:"
        )
        var end: Int
        if length >= 0:
            end = body + length
        elif _has_header(self.buf, self.at, head, "transfer-encoding:"):
            # The last chunk is `0\r\n\r\n`, and it either follows the CRLF that
            # ended a previous chunk or is the whole body when there were none.
            if _matches_ci(self.buf, body, "0\r\n\r\n"):
                end = body + 5
            else:
                var term = _find(self.buf, body, "\r\n0\r\n\r\n")
                if term < 0:
                    return -1 if self.pending() < MAX_PENDING else -2
                end = term + 7
        else:
            # No length and no chunking means the body ends when the socket
            # does, and molla never answers that way.
            return -2

        if len(self.buf) < end:
            return -1 if self.pending() < MAX_PENDING else -2
        self.at = end
        if self.at >= len(self.buf):
            self.buf.clear()
            self.at = 0
        return status


def _role_of(index: Int) -> Int:
    """Which kind of client this one is.

    Half are keep alive, because that is what most traffic is and because the
    latency numbers should come mostly from the ordinary case. The other four
    kinds get an eighth each, which at a thousand connections is a hundred and
    twenty five of each, enough that all five things are happening continuously
    rather than occasionally.
    """
    var slot = index % 8
    if slot == 1:
        return ROLE_STREAM
    if slot == 3:
        return ROLE_SLOW
    if slot == 5:
        return ROLE_ABRUPT
    if slot == 7:
        return ROLE_OVERSIZED
    return ROLE_KEEPALIVE


def _role_name(role: Int) -> StaticString:
    if role == ROLE_KEEPALIVE:
        return "keep alive"
    if role == ROLE_STREAM:
        return "streaming"
    if role == ROLE_SLOW:
        return "slow reader"
    if role == ROLE_ABRUPT:
        return "abrupt"
    return "oversized"


def _expected_status(role: Int) -> Int:
    """What every answer to this kind of client has to be.

    Per role rather than a range, because a run where the oversized clients
    started getting 200 and the keep alive clients started getting 413 would
    have exactly the same totals as a clean one.
    """
    return 413 if role == ROLE_OVERSIZED else 200


def _request_for(role: Int, turn: Int) -> String:
    """What this client sends next.

    The keep alive client alternates a plain GET with a POST that carries a
    body, so the body reader is on the hot path for an hour rather than only in
    a test of its own. It posts to `/healthz`, which answers whatever the
    method was, because `/` is GET only and a 405 would close the connection
    this client exists to hold open. The streaming client alternates the two
    framings for the same reason.
    """
    if role == ROLE_KEEPALIVE:
        if turn % 2 == 0:
            return String("GET / HTTP/1.1\r\nHost: soak\r\n\r\n")
        return String(
            "POST /healthz HTTP/1.1\r\nHost: soak\r\nContent-Length:"
            " 11\r\n\r\nhello molla"
        )
    if role == ROLE_STREAM:
        if turn % 2 == 0:
            return String("GET /stream/ndjson HTTP/1.1\r\nHost: soak\r\n\r\n")
        return String("GET /stream/sse HTTP/1.1\r\nHost: soak\r\n\r\n")
    if role == ROLE_SLOW:
        return String("GET /stream/ndjson HTTP/1.1\r\nHost: soak\r\n\r\n")
    if role == ROLE_ABRUPT:
        return String("GET /healthz HTTP/1.1\r\nHost: soak\r\n\r\n")
    return String(
        "POST / HTTP/1.1\r\nHost: soak\r\nContent-Length: "
        + String(OVERSIZED_LENGTH)
        + "\r\n\r\n"
    )


def _try_send(fd: Int, text: StringSpan) -> Int:
    """Push a request out. 1 for gone, 0 for not yet, -1 for a half sent one.

    Not yet is the case that matters and it took a run to find. Every socket
    here is non blocking from the moment it is created, so `connect` returns
    before the handshake finishes and the first write on a fresh socket fails
    until it does. Spinning on that in a tight loop turns a normal wait into a
    failure, so a socket that is not ready yet says so and the caller comes
    back on the next pass.

    Once the first byte is out the connection is up, and the rest of a request
    that is a couple of hundred bytes long is going to follow, so finishing a
    partial write in a bounded loop here is fine.
    """
    var n = text.byte_length()
    var p = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(text.unsafe_ptr())
    )
    var sent = send(fd, p, n)
    if sent <= 0:
        return 0
    var tries = 0
    while sent < n and tries < SEND_TRIES:
        var wrote = send(fd, p.unsafe_offset(sent), n - sent)
        if wrote > 0:
            sent += wrote
            continue
        tries += 1
    return 1 if sent == n else -1


def _open(port: UInt16) raises -> Int:
    var fd = socket_tcp()
    if fd < 0:
        return -1
    _ = connect(fd, INADDR_LOOPBACK, port)
    set_nonblocking(fd)
    return fd


def run_http_soak(connections: Int, seconds: Int) raises -> Int:
    """Hold `connections` of mixed HTTP traffic against a real server for
    `seconds`.

    Returns 0 when everything held. Prints the report either way, because a
    soak that fails is only useful if it says which of the things it watches
    went wrong.
    """
    # The abrupt clients close on a server that is mid write and the oversized
    # ones get closed on, so both ends write to dead sockets all run. Without
    # this the first one of those kills the process.
    _ = ignore_sigpipe()

    print(
        "http soak: "
        + String(connections)
        + " connections for "
        + String(seconds)
        + "s, five kinds of client at once"
    )

    var fd_before = probe_fd()
    var idle_timeout = seconds * 1000 + IDLE_MARGIN_MS
    var server = Server[HttpProtocol](
        ListenAddress(UInt16(0)), ServerContext(0, idle_timeout, 0)
    )

    # Logging and metrics on, because a soak of the systems layer that leaves
    # out the parts an operator would have running is a soak of a server nobody
    # is going to run. The level is warn, so the ring is exercised by the
    # enabled check on every request without an hour of debug lines to flush.
    var sink = LogSink(server.workers, 65536, LEVEL_WARN)
    var metrics = Metrics(server.workers)
    for i in range(len(server.reactors)):
        server.reactors[i].proto.configure(
            0, MAX_REQUESTS_PER_CONNECTION, SOAK_MAX_BODY
        )
        server.reactors[i].proto.configure_ops(
            sink.logger(i), metrics.meter(i), metrics.view(), True
        )
    var pump = LogPump(Int(Pointer(to=sink)))
    pump.start()

    var port = server.port
    print("  workers        " + String(server.workers))
    server.start()

    var fds = List[Int]()
    var roles = List[Int]()
    var wires = List[Wire]()
    var sent_at = List[Int]()
    var next_at = List[Int]()
    var turns = List[Int]()
    var stalled_since = List[Int]()
    var connect_failures = 0

    for i in range(connections):
        var fd = _open(port)
        if fd < 0:
            connect_failures += 1
            continue
        fds.append(fd)
        roles.append(_role_of(i))
        wires.append(Wire())
        sent_at.append(0)
        next_at.append(0)
        turns.append(0)
        stalled_since.append(0)
        if (i + 1) % CONNECT_BATCH == 0:
            # The backlog is deep and the workers accept on their own threads,
            # but issuing a thousand connects as fast as a loop can is still
            # worth pacing.
            _ = server.accepted()

    var settle = Int(monotonic()) + SETTLE_NS
    while server.accepted() < len(fds) and Int(monotonic()) < settle:
        pass
    var accepted_at_start = server.accepted()
    print(
        "  connected      "
        + String(len(fds))
        + ", accepted "
        + String(accepted_at_start)
    )
    if connect_failures > 0:
        print("  connect failures " + String(connect_failures))

    var rss_start = max_rss_kb()
    var live_rss_start = current_rss_kb()
    var rss_by_segment = List[Int](length=SEGMENTS, fill=-1)

    var latency = LatencyLog(SEGMENTS)
    var answers = List[Int](length=ROLE_COUNT, fill=0)
    var wrong_status = List[Int](length=ROLE_COUNT, fill=0)
    var reads = 0
    var malformed = 0
    var reconnects = 0
    var send_failures = 0
    var read_buf = stack_allocation[READ_CHUNK, UInt8]()

    var started = Int(monotonic())
    var length_ns = seconds * 1_000_000_000
    var deadline = started + length_ns
    var segment_seen = -1

    while True:
        var now = Int(monotonic())
        if now >= deadline:
            break
        var segment = ((now - started) * SEGMENTS) // length_ns
        if segment >= SEGMENTS:
            segment = SEGMENTS - 1
        if segment != segment_seen:
            # Once per segment rather than once per pass, because reading
            # /proc is a syscall and this loop is the client.
            segment_seen = segment
            rss_by_segment[segment] = current_rss_kb()

        for i in range(len(fds)):
            var role = roles[i]

            if fds[i] < 0:
                if now >= next_at[i]:
                    fds[i] = _open(port)
                    wires[i].clear()
                    sent_at[i] = 0
                    stalled_since[i] = 0
                    if fds[i] < 0:
                        next_at[i] = now + RECONNECT_MS * 1_000_000
                continue

            if sent_at[i] == 0:
                if now < next_at[i]:
                    continue
                var outcome = _try_send(fds[i], _request_for(role, turns[i]))
                if outcome == 0:
                    # Almost always a handshake that has not finished. Given
                    # five seconds of passes it is a socket that is never going
                    # to take a byte, and that is worth failing the run over.
                    if stalled_since[i] == 0:
                        stalled_since[i] = now
                    elif now - stalled_since[i] > STALL_NS:
                        send_failures += 1
                        _ = close(fds[i])
                        fds[i] = -1
                        stalled_since[i] = 0
                        next_at[i] = now + RECONNECT_MS * 1_000_000
                    continue
                if outcome < 0:
                    send_failures += 1
                    _ = close(fds[i])
                    fds[i] = -1
                    stalled_since[i] = 0
                    next_at[i] = now + RECONNECT_MS * 1_000_000
                    continue
                turns[i] += 1
                stalled_since[i] = 0
                sent_at[i] = now
                if role == ROLE_ABRUPT:
                    # Nothing is ever read. The close is the whole request.
                    next_at[i] = now + ABRUPT_LINGER_MS * 1_000_000
                continue

            if role == ROLE_ABRUPT:
                if now >= next_at[i]:
                    _ = close(fds[i])
                    fds[i] = -1
                    sent_at[i] = 0
                    answers[ROLE_ABRUPT] += 1
                    reconnects += 1
                    next_at[i] = now + RECONNECT_MS * 1_000_000
                continue

            if role == ROLE_SLOW and now < next_at[i]:
                continue

            var want = SLOW_READ_BYTES if role == ROLE_SLOW else READ_CHUNK
            var got = recv(fds[i], read_buf, want)
            if got == 0:
                # The server hung up. Expected after a 413 and not otherwise,
                # so a close in the middle of a response leaves an answer
                # uncounted rather than being quietly tolerated.
                _ = close(fds[i])
                fds[i] = -1
                sent_at[i] = 0
                reconnects += 1
                next_at[i] = now + RECONNECT_MS * 1_000_000
                continue
            if got < 0:
                if role == ROLE_SLOW:
                    next_at[i] = now + SLOW_PAUSE_MS * 1_000_000
                continue

            reads += 1
            wires[i].feed(read_buf, got)
            while True:
                var status = wires[i].next_status()
                if status == -1:
                    break
                if status == -2:
                    malformed += 1
                    _ = close(fds[i])
                    fds[i] = -1
                    wires[i].clear()
                    sent_at[i] = 0
                    next_at[i] = now + RECONNECT_MS * 1_000_000
                    break

                answers[role] += 1
                if status != _expected_status(role):
                    wrong_status[role] += 1
                if role != ROLE_SLOW:
                    # The slow reader's latency is its own doing, so it stays
                    # out of the numbers the drift gate looks at.
                    latency.record(segment, Int(monotonic()) - sent_at[i])
                sent_at[i] = 0

                if role == ROLE_OVERSIZED:
                    # A 413 closes the connection, so there is nothing to be
                    # gained by waiting for the server to say so.
                    _ = close(fds[i])
                    fds[i] = -1
                    wires[i].clear()
                    reconnects += 1
                    next_at[i] = now + RECONNECT_MS * 1_000_000
                    break
                if role == ROLE_KEEPALIVE:
                    next_at[i] = now + KEEPALIVE_THINK_MS * 1_000_000
                elif role == ROLE_STREAM:
                    next_at[i] = now + STREAM_THINK_MS * 1_000_000
                else:
                    next_at[i] = now

            if role == ROLE_SLOW and fds[i] >= 0:
                next_at[i] = now + SLOW_PAUSE_MS * 1_000_000

    var elapsed_ns = Int(monotonic()) - started
    var rss_end = max_rss_kb()
    var live_rss_end = current_rss_kb()

    for i in range(len(fds)):
        if fds[i] >= 0:
            _ = close(fds[i])

    # Give the workers a moment to notice a thousand closed peers before asking
    # how many connections are left.
    var reap_deadline = Int(monotonic()) + SETTLE_NS
    while server.open_connections() > 0 and Int(monotonic()) < reap_deadline:
        pass
    var open_after = server.open_connections()

    # After the pump has stopped, so what is left in the ring is what the flush
    # never got to rather than what it had not reached yet.
    pump.stop()
    var pending_logs = sink.pending()
    var dropped_logs = sink.dropped()
    server.stop()
    var fd_after = probe_fd()

    var requests = metrics.total(M_REQUESTS)
    var responses_2xx = metrics.total(M_RESPONSES_2XX)
    var responses_4xx = metrics.total(M_RESPONSES_4XX)
    var responses_5xx = metrics.total(M_RESPONSES_5XX)
    var handler_errors = metrics.total(M_HANDLER_ERRORS)
    var accepted = metrics.total(M_CONNECTIONS_ACCEPTED)

    var read_answers = 0
    for role in range(ROLE_COUNT):
        if role != ROLE_ABRUPT:
            read_answers += answers[role]

    print("  elapsed        " + String(elapsed_ns // 1_000_000) + " ms")
    print("  accepted       " + String(accepted))
    print("  reconnects     " + String(reconnects))
    print("  socket reads   " + String(reads))
    print("  requests       " + String(requests))
    print(
        "  responses      "
        + String(responses_2xx)
        + " 2xx, "
        + String(responses_4xx)
        + " 4xx, "
        + String(responses_5xx)
        + " 5xx"
    )
    print("  answers read   " + String(read_answers))
    print("  by client kind")
    for role in range(ROLE_COUNT):
        print(
            "    "
            + String(_role_name(role))
            + ": "
            + String(answers[role])
            + (
                " requests sent and dropped" if role
                == ROLE_ABRUPT else " answers read, "
                + String(wrong_status[role])
                + " with the wrong status"
            )
        )
    print(
        "  peak rss       "
        + String(rss_start)
        + " kB after connect, "
        + String(rss_end)
        + " kB at the end"
    )
    if live_rss_start >= 0:
        print(
            "  live rss       "
            + String(live_rss_start)
            + " kB after connect, "
            + String(live_rss_end)
            + " kB at the end"
        )
        var line = String("  live rss by segment, kB ")
        for s in range(SEGMENTS):
            line += " " + String(rss_by_segment[s])
        print(line)
    else:
        print("  live rss       not readable on this platform, peak only")
    print(
        "  probe fd       "
        + String(fd_before)
        + " before, "
        + String(fd_after)
        + " after"
    )
    print(
        "  log ring       "
        + String(pending_logs)
        + " bytes pending, "
        + String(dropped_logs)
        + " records dropped"
    )
    print("  connections    " + String(open_after) + " left after the clients")

    print("  latency by segment, microseconds")
    print("    segment  samples      mean       p50       p99")
    for s in range(SEGMENTS):
        print(
            "    "
            + String(s)
            + "        "
            + String(latency.count(s))
            + "  "
            + String(latency.mean_us(s))
            + "  "
            + String(latency.quantile_us(s, 50))
            + "  "
            + String(latency.quantile_us(s, 99))
        )

    var first_p99 = latency.quantile_us(0, 99)
    var last_p99 = latency.quantile_us(SEGMENTS - 1, 99)

    var ok = True
    if len(fds) != connections:
        print(
            "  FAIL: only "
            + String(len(fds))
            + " of "
            + String(connections)
            + " clients connected"
        )
        ok = False
    if accepted_at_start < len(fds):
        print(
            "  FAIL: accepted "
            + String(accepted_at_start)
            + " of "
            + String(len(fds))
            + " before the run started"
        )
        ok = False
    for role in range(ROLE_COUNT):
        if answers[role] == 0:
            print(
                "  FAIL: the "
                + String(_role_name(role))
                + " clients did nothing"
            )
            ok = False
        if wrong_status[role] != 0:
            print(
                "  FAIL: "
                + String(wrong_status[role])
                + " answers to the "
                + String(_role_name(role))
                + " clients were not "
                + String(_expected_status(role))
            )
            ok = False
    if malformed != 0:
        print(
            "  FAIL: "
            + String(malformed)
            + " responses could not be framed by the client"
        )
        ok = False
    if send_failures != 0:
        print(
            "  FAIL: " + String(send_failures) + " requests could not be sent"
        )
        ok = False
    if responses_5xx != 0 or handler_errors != 0:
        print(
            "  FAIL: "
            + String(responses_5xx)
            + " 5xx responses and "
            + String(handler_errors)
            + " handler errors"
        )
        ok = False
    # The oversized clients are the only ones asking for a 4xx, and every
    # request they send gets one, so the server's own count of 4xx answers has
    # to be at least what they read back.
    if responses_4xx < answers[ROLE_OVERSIZED]:
        print(
            "  FAIL: the clients read "
            + String(answers[ROLE_OVERSIZED])
            + " 413 answers and the server counted "
            + String(responses_4xx)
            + " 4xx"
        )
        ok = False
    if requests < read_answers:
        print(
            "  FAIL: the server counted "
            + String(requests)
            + " requests and the clients read "
            + String(read_answers)
            + " answers"
        )
        ok = False
    if open_after != 0:
        print(
            "  FAIL: "
            + String(open_after)
            + " connections were not reaped after the clients closed"
        )
        ok = False
    if dropped_logs != 0:
        print("  FAIL: " + String(dropped_logs) + " log records were dropped")
        ok = False
    if pending_logs != 0:
        print(
            "  FAIL: "
            + String(pending_logs)
            + " bytes were still in the log ring at the end"
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
    if rss_end > rss_start * 2 and rss_end - rss_start > 16384:
        print("  FAIL: peak memory more than doubled during the run")
        ok = False
    # A quarter more than it was once the connections were up, or four
    # megabytes, whichever is larger. The allowance is there because this
    # process is both ends of every connection and the client side is still
    # growing its buffers into their steady size when the clock starts.
    if live_rss_start > 0 and live_rss_end > 0:
        var allowed = live_rss_start // 4
        if allowed < 4096:
            allowed = 4096
        if live_rss_end - live_rss_start > allowed:
            print(
                "  FAIL: resident memory went from "
                + String(live_rss_start)
                + " kB to "
                + String(live_rss_end)
                + " kB"
            )
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
        print("  http soak passed")
        return 0
    return 1
