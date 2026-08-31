"""Tests for config precedence, the log ring, the counters, and /molla.

Four things get checked here and three of them are ordinary. Config is a table
and a precedence rule, metrics are integers, and the admin routes are HTTP.

The fourth is the acceptance criterion for #16 and it is the reason this file
has an allocation counter in it. Logging at a level that is switched off has to
cost nothing, and nothing is a claim about allocations rather than about time,
because time is noisy and an allocation is not. So a thousand calls to
`log.debug` with the level at info run against a counted buffer and the count
has to come out at exactly zero. Not small. Zero.

The admin routes are checked against a real threaded server rather than by
calling the handler, because the interesting failure is a route that works on
one thread and returns somebody else's counters on four, and calling the
handler directly cannot see it.

Every test that makes a `LogSink` ends with `keep(sink)`. A `Logger` is a set of
addresses into the sink's block, Mojo drops a value at its last use rather than
at the end of the scope, and handing out an address is not a use it can see. A
test that forgets it reads a freed ring, which passes most of the time and fails
on a machine with a different allocator, which is how two of these were found.
"""

from std.ffi import c_int, external_call
from std.memory import stack_allocation

from harness import Suite

from molla.http.protocol import HttpProtocol
from molla.io.buffer import Buffer
from molla.net.context import ServerContext
from molla.net.listener import ListenAddress
from molla.net.server import Server
from molla.ops.config import (
    FROM_DEFAULT,
    FROM_ENV,
    FROM_FILE,
    FROM_FLAG,
    KEY_LOG_LEVEL,
    KEY_WORKERS,
    LEVEL_DEBUG,
    LEVEL_ERROR,
    LEVEL_INFO,
    LEVEL_OFF,
    LEVEL_WARN,
    Config,
    describe_setting,
    env_name,
    level_name,
    level_of,
    load_config,
    source_name,
)
from molla.ops.log import LogPump, LogSink
from molla.ops.metrics import (
    M_BYTES_READ,
    M_CONNECTIONS_ACCEPTED,
    M_REQUESTS,
    M_RESPONSES_2XX,
    M_RESPONSES_4XX,
    M_RESPONSES_5XX,
    Meter,
    Metrics,
)
from molla.sys.clock import monotonic_ms, monotonic_ns, realtime_ns
from molla.sys.fd import close
from molla.sys.file import close_fd, create, pwrite_all, unlink
from molla.sys.mem import AllocCounter, keep
from molla.sys.socket import INADDR_LOOPBACK, connect, recv, send, socket_tcp
from molla.sys.thread import sleep_ms

comptime WAIT_MS = 5000
comptime OFF_ROUNDS = 1000
"""How many disabled log calls the allocation check makes. Enough that a single
allocation anywhere on the path shows up, and small enough to be free."""


def _setenv(name: String, value: String):
    var n = name + "\0"
    var v = value + "\0"
    _ = external_call["setenv", c_int](n.unsafe_ptr(), v.unsafe_ptr(), c_int(1))


def _unsetenv(name: String):
    var n = name + "\0"
    _ = external_call["unsetenv", c_int](n.unsafe_ptr())


def _write_file(path: StringSpan, text: StringSpan) -> Bool:
    var opened = create(path)
    if not opened.is_ok():
        return False
    var fd = opened.value
    var p = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(text.unsafe_ptr())
    )
    var wrote = pwrite_all(fd, p, text.byte_length(), 0)
    _ = close_fd(fd)
    return wrote.is_ok()


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


def _contains(data: List[UInt8], needle: StringSpan) -> Bool:
    var n = needle.byte_length()
    if n == 0 or len(data) < n:
        return False
    var p = needle.unsafe_ptr()
    for start in range(len(data) - n + 1):
        var same = True
        for i in range(n):
            if data[start + i] != p.unsafe_load(i):
                same = False
                break
        if same:
            return True
    return False


def _read_until(fd: Int, needle: StringSpan, mut into: List[UInt8]) -> Bool:
    var buf = stack_allocation[4096, UInt8]()
    var deadline = monotonic_ms() + WAIT_MS
    while monotonic_ms() < deadline:
        if _contains(into, needle):
            return True
        var got = recv(fd, buf, 4096)
        if got > 0:
            for i in range(got):
                into.append(buf.unsafe_load(i))
            continue
        if got == 0:
            return _contains(into, needle)
        _ = sleep_ms(1)
    return _contains(into, needle)


def _ask(port: UInt16, target: StringSpan, mut into: List[UInt8]) raises:
    """One request on one connection, read until the connection closes or the
    terminator arrives. `Connection: close` so the read has an end."""
    var fd = _client(port)
    _ = _send_text(
        fd,
        String("GET ")
        + String(target)
        + " HTTP/1.1\r\nHost: molla\r\nConnection: close\r\n\r\n",
    )
    _ = _read_until(fd, "\r\n\r\n", into)
    # A header terminator is not a body terminator, so keep reading until the
    # peer hangs up. Every admin answer is small enough that this is one more
    # pass through the loop.
    var deadline = monotonic_ms() + 200
    var buf = stack_allocation[4096, UInt8]()
    while monotonic_ms() < deadline:
        var got = recv(fd, buf, 4096)
        if got > 0:
            for i in range(got):
                into.append(buf.unsafe_load(i))
            continue
        if got == 0:
            break
        _ = sleep_ms(1)
    _ = close(fd)


def _check_clocks(mut suite: Suite):
    suite.group("ops clocks")

    var a = monotonic_ns()
    var b = monotonic_ns()
    suite.check(b >= a, "the monotonic clock does not go backwards")
    suite.check(a > 0, "and it is not stuck at zero")

    var wall = realtime_ns()
    # Nanoseconds at the start of 2020. Anything below this is a clock that did
    # not answer, and the point of the check is to catch a `clock_gettime` that
    # silently returned a seconds count.
    suite.check(wall > 1577836800000000000, "the wall clock is a real date")
    suite.check(
        wall % 1000000000 != 0 or wall == 0,
        "and it has sub second precision",
    )


def _check_config_defaults(mut suite: Suite):
    suite.group("ops config defaults")

    var config = Config()
    suite.check(
        config.value(KEY_WORKERS) == "0",
        "workers defaults to nought, which means one per core",
    )
    suite.check(
        config.get(KEY_WORKERS).source == FROM_DEFAULT,
        "and it says the default is where that came from",
    )
    suite.check(
        config.int_value("idle_timeout_ms") == 60000, "idle is a minute"
    )
    suite.check(config.bool_value("metrics"), "metrics are on")
    suite.check(config.index_of("nonsense") < 0, "an unknown key is not found")
    suite.check(
        config.value("nonsense") == "",
        "and reading it gives an empty string rather than a crash",
    )
    suite.check(len(config.problems()) == 0, "the defaults are all valid")
    suite.check(
        config.describe().find("workers = 0 (default)") >= 0,
        "describe prints the value and the source",
    )
    suite.check(
        describe_setting(config.get(KEY_WORKERS)).find("default 0") < 0,
        "and it leaves out the default when the default is the source",
    )
    suite.check(source_name(FROM_FLAG) == "flag", "the sources have names")
    suite.check(source_name(99) == "unknown", "and an unknown one says so")


def _check_config_precedence(mut suite: Suite):
    suite.group("ops config precedence")

    var path = String("/tmp/molla-test-config.conf")
    var written = _write_file(
        path,
        "# a comment\n\nworkers = 3\nlog_level=warn\nidle_timeout_ms = 1000\n",
    )
    suite.check(written, "the test config file was written")

    var config = Config()
    suite.check(config.load_file(path), "and it reads back")
    suite.check(config.int_value(KEY_WORKERS) == 3, "the file sets workers")
    suite.check(config.get(KEY_WORKERS).source == FROM_FILE, "and it says so")
    suite.check(
        config.value(KEY_LOG_LEVEL) == "warn",
        "a line without spaces round the equals works too",
    )

    var key = env_name(KEY_WORKERS)
    suite.check(key == "MOLLA_WORKERS", "the environment name is the key upper")
    _setenv(key, "5")
    config.load_env()
    suite.check(
        config.int_value(KEY_WORKERS) == 5, "the environment beats the file"
    )
    suite.check(config.get(KEY_WORKERS).source == FROM_ENV, "and it says so")

    _ = config.load_flag("--workers=7")
    suite.check(
        config.int_value(KEY_WORKERS) == 7, "a flag beats the environment"
    )
    suite.check(config.get(KEY_WORKERS).source == FROM_FLAG, "and it says so")
    suite.check(
        describe_setting(config.get(KEY_WORKERS)).find("default 0") >= 0,
        "and now the line shows what it overrode",
    )

    suite.check(
        not config.load_flag("--nonsense=1"),
        "a flag for a key that does not exist is refused",
    )
    suite.check(
        not config.load_flag("workers=1"),
        "and so is one that forgot the dashes",
    )

    _ = config.load_flag("--workers=banana")
    var problems = config.problems()
    suite.check(len(problems) == 1, "a value that is not a number is a problem")
    suite.check(
        problems[0].find("banana") >= 0 and problems[0].find("flag") >= 0,
        "and the problem says what it was and where it came from",
    )
    suite.check(
        config.int_value(KEY_WORKERS) == 0,
        "and reading it falls back to the default rather than guessing",
    )

    var args = List[String]()
    args.append(String("molla"))
    args.append(String("config"))
    args.append(String("get"))
    args.append(String("--config=") + path)
    args.append(String("--idle_timeout_ms=250"))
    var loaded = load_config(args)
    suite.check(
        loaded.int_value("idle_timeout_ms") == 250,
        "load_config reads the file, the environment, and the flags in order",
    )
    suite.check(
        loaded.int_value(KEY_WORKERS) == 5,
        "and the environment still wins over the file it read",
    )

    _unsetenv(key)
    _ = unlink(path)


def _check_levels(mut suite: Suite):
    suite.group("ops log levels")

    suite.check(level_of("debug") == LEVEL_DEBUG, "debug parses")
    suite.check(level_of("WARN") == LEVEL_WARN, "and it is case insensitive")
    suite.check(level_of("off") == LEVEL_OFF, "off is a level")
    suite.check(level_of("chatty") < 0, "and anything else is not")
    suite.check(level_name(LEVEL_ERROR) == "error", "the levels have names")
    suite.check(
        level_name(42) == "off",
        "and a level above the last one is silence rather than a name",
    )

    var config = Config()
    _ = config.load_flag("--log_level=chatty")
    var problems = config.problems()
    suite.check(
        len(problems) == 1 and problems[0].find("chatty") >= 0,
        "a level that is not a level is reported like a bad number is",
    )


def _check_log_records(mut suite: Suite):
    suite.group("ops log records")

    var sink = LogSink(2, 8192, LEVEL_INFO)
    suite.check(sink.is_valid(), "a sink with two streams is valid")

    var log = sink.logger(0)
    suite.check(log.enabled(LEVEL_WARN), "warn is on when the level is info")
    suite.check(not log.enabled(LEVEL_DEBUG), "and debug is not")
    suite.check(not log.debug("noise"), "a disabled call writes nothing")
    suite.check(sink.pending() == 0, "and leaves the ring empty")

    suite.check(log.info("server started"), "an enabled call writes")
    var entry = log.begin(LEVEL_WARN)
    entry.message("slow request")
    entry.field("path", "/v1/chat")
    entry.field_int("duration_ns", 1234567)
    entry.field_bool("streamed", True)
    suite.check(entry.end(), "and a record with fields writes")
    suite.check(sink.pending() > 0, "the records are waiting in the ring")

    var out = Buffer(65536, 0)
    suite.check(sink.drain(out) == 2, "and draining takes both")
    var text = String(StringSlice(unsafe_from_utf8=out.bytes()))
    suite.check(text.find('msg="server started"') >= 0, "the message is quoted")
    suite.check(text.find("level=warn") >= 0, "the level is named")
    suite.check(text.find("worker=0") >= 0, "the worker is numbered")
    suite.check(text.find("ts=") == 0, "the timestamp comes first")
    suite.check(
        text.find("duration_ns=1234567") >= 0,
        "an integer field is written as an integer, in nanoseconds",
    )
    suite.check(text.find("streamed=true") >= 0, "a bool field reads as a bool")
    suite.check(sink.pending() == 0, "and the ring is empty again")

    # A quote or a newline in a value would end the record early and let a
    # request header forge a second log line, which is the whole reason the
    # cleaning exists.
    var dirty = log.begin(LEVEL_ERROR)
    dirty.message("bad")
    dirty.field("target", 'a "quoted"\nsecond line')
    _ = dirty.end()
    out.clear()
    _ = sink.drain(out)
    var cleaned = String(StringSlice(unsafe_from_utf8=out.bytes()))
    suite.check(
        cleaned.find("a 'quoted'.second line") >= 0,
        "a quote becomes an apostrophe and a newline becomes a dot",
    )
    suite.check(
        _count_newlines(cleaned) == 1, "so one record is still one line"
    )

    var second = sink.logger(1)
    _ = second.info("from the other worker")
    out.clear()
    _ = sink.drain(out)
    var other = String(StringSlice(unsafe_from_utf8=out.bytes()))
    suite.check(
        other.find("worker=1") >= 0, "each stream writes its own worker number"
    )

    sink.set_level(LEVEL_OFF)
    # The logger goes in a local and the sink is kept alive past it, and both
    # halves of that matter. A `Logger` is addresses into the sink's block, and
    # Mojo drops a value at its last use, so writing this as one expression on
    # `sink.logger(0)` frees the rings before the call that uses them and the
    # level read comes back as whatever the allocator left behind. It read as
    # enabled, which is how this comment got written.
    var silenced = sink.logger(0)
    suite.check(
        not silenced.error("nothing gets through off"),
        "setting the level to off silences every stream",
    )
    suite.check(sink.pending() == 0, "and nothing reaches the ring")

    keep(out)
    keep(sink)


def _count_newlines(text: String) -> Int:
    var n = 0
    var bytes = text.as_bytes()
    for i in range(len(bytes)):
        if bytes[i] == UInt8(10):
            n += 1
    return n


def _check_log_off_allocates_nothing(mut suite: Suite):
    suite.group("ops log costs nothing when off")

    var sink = LogSink(1, 8192, LEVEL_INFO)
    var log = sink.logger(0)
    var counter = AllocCounter()
    var out = Buffer(4096, counter.raw())

    # One call first, so anything the path allocates once is already allocated
    # before the count starts. There is nothing, and proving that is the point,
    # but a check that would pass either way is not worth having.
    _ = log.debug("warming up")
    var before = counter.total()
    for _ in range(OFF_ROUNDS):
        _ = log.debug("this level is off")
    var after = counter.total()
    suite.check(
        after - before == 0,
        "a thousand calls at a disabled level allocate nothing at all",
    )
    suite.check(sink.pending() == 0, "and write nothing")

    # The same loop with the level on, so the zero above is evidence about the
    # level check rather than about the loop being optimised away.
    sink.set_level(LEVEL_DEBUG)
    for _ in range(8):
        _ = log.debug("this level is on")
    suite.check(sink.pending() > 0, "and turning the level on writes again")

    _ = sink.drain(out)
    keep(out)
    keep(sink)
    counter.close()


def _check_log_full_ring(mut suite: Suite):
    suite.group("ops log drops rather than blocks")

    var sink = LogSink(1, 4096, LEVEL_INFO)
    var log = sink.logger(0)
    var written = 0
    for _ in range(4096):
        if log.info("a record that will not fit forever"):
            written += 1
    suite.check(written > 0, "records go in until the ring is full")
    suite.check(written < 4096, "and then they stop going in")
    suite.check(sink.dropped() > 0, "the drops are counted")
    suite.check(
        sink.dropped() + written == 4096,
        "and every call either wrote or was counted as dropped",
    )

    var out = Buffer(65536, 0)
    var taken = sink.drain(out)
    suite.check(taken == written, "draining gets back exactly what went in")
    suite.check(
        log.info("and there is room again"),
        "and the ring takes records again once it has been drained",
    )
    keep(out)
    keep(sink)


def _check_log_pump(mut suite: Suite) raises:
    suite.group("ops log pump")

    var sink = LogSink(1, 8192, LEVEL_INFO)
    var pump = LogPump(Int(Pointer(to=sink)))
    pump.start()
    var log = sink.logger(0)
    for i in range(16):
        _ = log.info("pumped")
    var deadline = monotonic_ms() + WAIT_MS
    while sink.pending() > 0 and monotonic_ms() < deadline:
        _ = sleep_ms(1)
    suite.check(sink.pending() == 0, "the pump empties the ring on its own")
    _ = log.error("written just before the stop")
    pump.stop()
    suite.check(
        sink.pending() == 0,
        (
            "and it flushes once more on the way out rather than losing the"
            " last record"
        ),
    )
    keep(sink)


def _check_meters(mut suite: Suite):
    suite.group("ops metrics")

    var nothing = Meter()
    nothing.inc(M_REQUESTS)
    suite.check(
        nothing.get(M_REQUESTS) == 0,
        "a meter that was never configured counts nothing and does not crash",
    )

    var metrics = Metrics(4)
    suite.check(metrics.is_valid(), "four workers of counters are valid")

    var first = metrics.meter(0)
    var second = metrics.meter(1)
    first.inc(M_REQUESTS)
    first.add(M_BYTES_READ, 100)
    second.inc(M_REQUESTS)
    second.add(M_BYTES_READ, 40)
    suite.check(first.get(M_REQUESTS) == 1, "each worker keeps its own count")
    suite.check(metrics.total(M_REQUESTS) == 2, "and the total adds them up")
    suite.check(metrics.total(M_BYTES_READ) == 140, "for every series")

    first.inc(M_CONNECTIONS_ACCEPTED)
    first.dec(M_CONNECTIONS_ACCEPTED)
    suite.check(
        metrics.total(M_CONNECTIONS_ACCEPTED) == 0, "a gauge goes down again"
    )

    first.observe_status(200)
    first.observe_status(204)
    first.observe_status(404)
    first.observe_status(500)
    first.observe_status(101)
    suite.check(metrics.total(M_RESPONSES_2XX) == 2, "2xx is bucketed")
    suite.check(metrics.total(M_RESPONSES_4XX) == 1, "4xx is bucketed")
    suite.check(metrics.total(M_RESPONSES_5XX) == 1, "5xx is bucketed")

    suite.check(
        metrics.meter(9).get(M_REQUESTS) == 0,
        "a worker number nobody has gets the meter that counts nothing",
    )

    var view = metrics.view()
    suite.check(view.total(M_REQUESTS) == 2, "the view reads the same numbers")
    suite.check(view.uptime_ns() > 0, "and the uptime is running")

    var out = Buffer(65536, 0)
    suite.check(metrics.render(out, "9.9.9"), "the exposition renders")
    var text = String(StringSlice(unsafe_from_utf8=out.bytes()))
    suite.check(
        text.find("# HELP molla_http_requests_total") >= 0,
        "every series has a HELP line",
    )
    suite.check(
        text.find("# TYPE molla_http_requests_total counter") >= 0,
        "and a TYPE line",
    )
    suite.check(
        text.find("molla_http_requests_total 2") >= 0, "and the summed value"
    )
    suite.check(
        text.find("# TYPE molla_connections_open gauge") >= 0,
        "a gauge is typed as a gauge",
    )
    suite.check(
        text.find('molla_build_info{version="9.9.9"} 1') >= 0,
        "and the version goes out as a label on a build info gauge",
    )
    suite.check(
        text.find("molla_http_request_duration_ns_total") >= 0,
        "durations are exported as integer nanoseconds",
    )

    var off = Metrics(2, False)
    var quiet = Buffer(4096, 0)
    _ = off.render(quiet, "9.9.9")
    var said = String(StringSlice(unsafe_from_utf8=quiet.bytes()))
    suite.check(
        said.find("switched off") >= 0,
        "and a server without metrics says so rather than exporting zeroes",
    )

    keep(out)
    keep(quiet)


def _check_admin_routes(mut suite: Suite) raises:
    suite.group("ops admin routes")

    var server = Server[HttpProtocol](
        ListenAddress(UInt16(0)), ServerContext(2, 60000, 0, 2000)
    )
    var sink = LogSink(server.workers, 8192, LEVEL_OFF)
    var metrics = Metrics(server.workers)
    for i in range(len(server.reactors)):
        server.reactors[i].proto.configure_ops(
            sink.logger(i), metrics.meter(i), metrics.view(), True
        )
    var port = server.port
    server.start()

    var version = List[UInt8]()
    _ask(port, "/molla/version", version)
    suite.check(
        _contains(version, "HTTP/1.1 200 OK"), "/molla/version answers 200"
    )
    suite.check(_contains(version, "molla "), "and names the version")
    suite.check(
        _contains(version, "mojo "), "and the toolchain it was built on"
    )

    var health = List[UInt8]()
    _ask(port, "/molla/health", health)
    suite.check(
        _contains(health, "HTTP/1.1 200 OK"), "/molla/health answers 200"
    )
    suite.check(_contains(health, "ok\n"), "with a body a script can test")
    suite.check(
        _contains(health, "Content-Type: text/plain"), "and a plain body"
    )

    var scrape = List[UInt8]()
    _ask(port, "/molla/metrics", scrape)
    suite.check(
        _contains(scrape, "HTTP/1.1 200 OK"), "/molla/metrics answers 200"
    )
    suite.check(
        _contains(scrape, "Content-Type: text/plain; version=0.0.4"),
        "with the content type the exposition format asks for",
    )
    suite.check(
        _contains(scrape, "molla_http_requests_total"),
        "and the exposition in the body",
    )
    suite.check(
        _contains(scrape, "molla_build_info"), "including the build info gauge"
    )

    # The requests that were counted are the ones asked for above, and they are
    # spread across whichever workers accepted them, so this is also the check
    # that the summing works across threads.
    suite.check(
        metrics.total(M_REQUESTS) >= 3,
        "the scrape counted the requests that produced it",
    )
    suite.check(
        metrics.total(M_CONNECTIONS_ACCEPTED) >= 3, "and the connections"
    )
    suite.check(metrics.total(M_RESPONSES_2XX) >= 3, "and the answers")
    suite.check(metrics.total(M_BYTES_READ) > 0, "and the bytes that came in")

    var post = _client(port)
    _ = _send_text(
        post,
        (
            "POST /molla/metrics HTTP/1.1\r\nHost: molla\r\nContent-Length:"
            " 0\r\n\r\n"
        ),
    )
    var refused = List[UInt8]()
    _ = _read_until(post, "\r\n\r\n", refused)
    suite.check(
        _contains(refused, "HTTP/1.1 405"),
        "an admin route only answers GET and HEAD",
    )
    _ = close(post)

    var report = server.drain()
    suite.check(report.clean, "and the server still drains cleanly")
    keep(sink)
    keep(metrics)


def _check_admin_off(mut suite: Suite) raises:
    suite.group("ops admin routes off")

    var server = Server[HttpProtocol](
        ListenAddress(UInt16(0)), ServerContext(2, 60000, 0, 2000)
    )
    var port = server.port
    server.start()

    var reply = List[UInt8]()
    _ask(port, "/molla/metrics", reply)
    suite.check(
        _contains(reply, "HTTP/1.1 404"),
        "a server that was never told to serve /molla returns a 404",
    )

    var alive = List[UInt8]()
    _ask(port, "/healthz", alive)
    suite.check(
        _contains(alive, "HTTP/1.1 200 OK"),
        "and everything else still answers",
    )

    var report = server.drain()
    suite.check(report.clean, "and it drains cleanly")


def run(mut suite: Suite) raises:
    _check_clocks(suite)
    _check_config_defaults(suite)
    _check_config_precedence(suite)
    _check_levels(suite)
    _check_log_records(suite)
    _check_log_off_allocates_nothing(suite)
    _check_log_full_ring(suite)
    _check_log_pump(suite)
    _check_meters(suite)
    _check_admin_routes(suite)
    _check_admin_off(suite)
