"""Counters, one set per worker, added up when somebody scrapes them.

Prometheus is the format and the interesting decision is underneath it. A
counter that every worker increments is a cache line that every core has to
take a turn owning, and molla has one of those per request. So each worker gets
its own copy of every series, on its own cache line, and the sum only happens
when `/molla/metrics` is scraped, which is once every fifteen seconds by a
machine that does not care how long it takes.

The series are a fixed list rather than a registry with names in a map. A map
lookup per increment is more expensive than the increment, and a fixed list
means the render loop has no allocation in it and the compiler knows every
index. Adding a series is a line in `SERIES` and a constant, which is the same
amount of work as registering one would be.

Durations are exported as integer nanoseconds, in a `_ns_total` counter beside
a `_count` counter, because that pair is what Prometheus needs to compute an
average and it is what a histogram degrades to when you do not have one yet.
molla does not have one yet, and saying that here is better than exporting a
number that looks like a quantile and is not.
"""

from molla.io.buffer import Buffer
from molla.http.serialize import write_decimal
from molla.sys.atomic import CACHE_LINE, AtomicBlock, AtomicRef
from molla.sys.clock import monotonic_ns

comptime COUNTER = 0
comptime GAUGE = 1


struct Series(Copyable, ImplicitlyCopyable, Movable):
    """One exported number: what it is called, what it means, and its type."""

    var name: StaticString
    var help: StaticString
    var kind: Int

    def __init__(out self, name: StaticString, help: StaticString, kind: Int):
        self.name = name
        self.help = help
        self.kind = kind


comptime M_REQUESTS = 0
comptime M_RESPONSES_2XX = 1
comptime M_RESPONSES_4XX = 2
comptime M_RESPONSES_5XX = 3
comptime M_CONNECTIONS_ACCEPTED = 4
comptime M_CONNECTIONS_OPEN = 5
comptime M_BYTES_READ = 6
comptime M_BYTES_WRITTEN = 7
comptime M_REQUEST_NS = 8
comptime M_REQUEST_COUNT = 9
comptime M_HANDLER_ERRORS = 10
comptime M_LOG_DROPPED = 11
comptime SERIES_COUNT = 12


def series_list() -> List[Series]:
    """Every series molla exports, in index order.

    A function rather than a comptime list because Mojo 1.0 has no global to
    put one in. It is called once by `render`, which is not a hot path.
    """
    var out = List[Series](capacity=SERIES_COUNT)
    out.append(
        Series(
            "molla_http_requests_total",
            "Requests parsed, whatever the answer was",
            COUNTER,
        )
    )
    out.append(
        Series(
            "molla_http_responses_2xx_total",
            "Responses with a 2xx status",
            COUNTER,
        )
    )
    out.append(
        Series(
            "molla_http_responses_4xx_total",
            "Responses with a 4xx status, which is the client's problem",
            COUNTER,
        )
    )
    out.append(
        Series(
            "molla_http_responses_5xx_total",
            "Responses with a 5xx status, which is ours",
            COUNTER,
        )
    )
    out.append(
        Series(
            "molla_connections_accepted_total",
            "Connections accepted since the process started",
            COUNTER,
        )
    )
    out.append(
        Series("molla_connections_open", "Connections open right now", GAUGE)
    )
    out.append(
        Series("molla_bytes_read_total", "Bytes read from clients", COUNTER)
    )
    out.append(
        Series("molla_bytes_written_total", "Bytes written to clients", COUNTER)
    )
    out.append(
        Series(
            "molla_http_request_duration_ns_total",
            (
                "Nanoseconds spent answering requests, on the monotonic clock."
                " Divide by molla_http_request_duration_count for the mean."
                " There is no histogram yet"
            ),
            COUNTER,
        )
    )
    out.append(
        Series(
            "molla_http_request_duration_count",
            "Requests included in molla_http_request_duration_ns_total",
            COUNTER,
        )
    )
    out.append(
        Series(
            "molla_http_handler_errors_total",
            "Handlers that raised and were turned into a 500",
            COUNTER,
        )
    )
    out.append(
        Series(
            "molla_log_records_dropped_total",
            "Log records dropped because a worker's ring was full",
            COUNTER,
        )
    )
    return out^


struct Meter(Copyable, ImplicitlyCopyable, Movable):
    """One worker's set of counters. Addresses only, so it copies into a
    reactor and crosses a thread boundary the same way a `Logger` does."""

    var base: Int
    """Address of this worker's first counter, or zero for a meter that counts
    nothing. Zero is what a server built without metrics gets, so no call site
    has to check whether metrics are on."""

    def __init__(out self, base: Int = 0):
        self.base = base

    def _at(self, index: Int) -> AtomicRef:
        if self.base == 0 or index < 0 or index >= SERIES_COUNT:
            return AtomicRef(0)
        return AtomicRef(self.base + index * CACHE_LINE)

    def add(self, index: Int, delta: Int):
        _ = self._at(index).add(delta)

    def inc(self, index: Int):
        _ = self._at(index).add(1)

    def dec(self, index: Int):
        _ = self._at(index).sub(1)

    def set(self, index: Int, value: Int):
        self._at(index).store(value)

    def get(self, index: Int) -> Int:
        return self._at(index).load()

    def observe_status(self, status: Int):
        """Bucket a response by status class.

        Three counters rather than one per status code. A per code counter is a
        label with an unbounded range in it, which is the standard way to make
        a metrics backend fall over, and the question anybody actually asks of
        this is what fraction of responses were our fault.
        """
        if status >= 500:
            self.inc(M_RESPONSES_5XX)
        elif status >= 400:
            self.inc(M_RESPONSES_4XX)
        elif status >= 200 and status < 300:
            self.inc(M_RESPONSES_2XX)

    def observe_request_ns(self, nanoseconds: Int):
        self.add(M_REQUEST_NS, nanoseconds)
        self.inc(M_REQUEST_COUNT)


struct MetricsView(Copyable, ImplicitlyCopyable, Movable):
    """Read only access to every worker's counters, by address.

    The reason this exists rather than a pointer to `Metrics` is that the
    protocol which serves `/molla/metrics` lives inside a reactor, the reactors
    live in a list the server owns, and a pointer to a struct that a list can
    move is a pointer that goes stale on the next append. The block's address
    does not move, so the view holds that and nothing else.
    """

    var base: Int
    var workers: Int
    var started_ns: Int

    def __init__(
        out self, base: Int = 0, workers: Int = 0, started_ns: Int = 0
    ):
        self.base = base
        self.workers = workers
        self.started_ns = started_ns

    def is_valid(self) -> Bool:
        return self.base != 0 and self.workers > 0

    def total(self, index: Int) -> Int:
        """One series, summed over every worker.

        The workers are read one at a time while they are still running, so
        this is a reading taken at slightly different moments rather than an
        instant. That is what every counter scrape is, and it is worth knowing
        before somebody compares two series and finds them off by one.
        """
        if not self.is_valid() or index < 0 or index >= SERIES_COUNT:
            return 0
        var sum = 0
        for w in range(self.workers):
            sum += AtomicRef(
                self.base + (w * SERIES_COUNT + index) * CACHE_LINE
            ).load()
        return sum

    def uptime_ns(self) -> Int:
        if self.started_ns == 0:
            return 0
        return monotonic_ns() - self.started_ns

    def render(self, mut out: Buffer, version: StringSpan) -> Bool:
        """The whole exposition, in Prometheus text format.

        Written into a caller's buffer with no intermediate allocation, so the
        admin route costs what a response costs and not what a String
        concatenation costs.
        """
        if not self.is_valid():
            return out.append_str(
                "# metrics are switched off in this process\n"
            )
        var ok = True
        var all = series_list()
        for i in range(len(all)):
            ok = ok and _one(out, all[i], self.total(i))

        ok = ok and out.append_str(
            "# HELP molla_process_uptime_ns Nanoseconds since the process"
            " started, on the monotonic clock\n"
        )
        ok = ok and out.append_str("# TYPE molla_process_uptime_ns gauge\n")
        ok = ok and out.append_str("molla_process_uptime_ns ")
        ok = ok and write_decimal(out, self.uptime_ns())
        ok = ok and out.append_str("\n")

        # A build info gauge that is always 1 and carries the version in a
        # label. It looks odd and it is the convention, because it lets a
        # dashboard join on version without the version being a series of its
        # own that changes cardinality every release.
        ok = ok and out.append_str(
            "# HELP molla_build_info Build information, always 1\n"
        )
        ok = ok and out.append_str("# TYPE molla_build_info gauge\n")
        ok = ok and out.append_str('molla_build_info{version="')
        ok = ok and out.append_str(version)
        ok = ok and out.append_str('"} 1\n')
        return ok

    def snapshot(self) -> String:
        """Every series as one line each, for a log or a test.

        Allocates, and is not on any request path. `/molla/metrics` goes
        through `render`.
        """
        var out = String("")
        var all = series_list()
        for i in range(len(all)):
            out += String(all[i].name) + " " + String(self.total(i)) + "\n"
        return out^


struct Metrics(Movable):
    """Every worker's counters, and the render that adds them up."""

    var block: AtomicBlock
    var workers: Int
    var started_ns: Int
    var enabled: Bool

    def __init__(out self, workers: Int, enabled: Bool = True):
        self.workers = workers if workers > 0 else 1
        self.enabled = enabled
        self.block = AtomicBlock(self.workers * SERIES_COUNT)
        self.started_ns = monotonic_ns()

    def is_valid(self) -> Bool:
        return self.enabled and self.block.is_valid()

    def meter(self, worker: Int) -> Meter:
        """The end one worker holds. An out of range worker gets the counting
        nothing meter rather than somebody else's counters."""
        if not self.is_valid() or worker < 0 or worker >= self.workers:
            return Meter(0)
        return Meter(self.block.address_of(worker * SERIES_COUNT))

    def view(self) -> MetricsView:
        """What the protocol is given, so that serving `/molla/metrics` needs
        no reference to this struct."""
        if not self.is_valid():
            return MetricsView()
        return MetricsView(
            self.block.address_of(0), self.workers, self.started_ns
        )

    def total(self, index: Int) -> Int:
        return self.view().total(index)

    def uptime_ns(self) -> Int:
        return self.view().uptime_ns()

    def render(self, mut out: Buffer, version: StringSpan) -> Bool:
        return self.view().render(out, version)

    def snapshot(self) -> String:
        return self.view().snapshot()


def _one(mut out: Buffer, series: Series, value: Int) -> Bool:
    var ok = out.append_str("# HELP ")
    ok = ok and out.append_str(series.name)
    ok = ok and out.append_str(" ")
    ok = ok and out.append_str(series.help)
    ok = ok and out.append_str("\n# TYPE ")
    ok = ok and out.append_str(series.name)
    ok = ok and out.append_str(
        " counter\n" if series.kind == COUNTER else (" gauge\n")
    )
    ok = ok and out.append_str(series.name)
    ok = ok and out.append_str(" ")
    ok = ok and write_decimal(out, value)
    ok = ok and out.append_str("\n")
    return ok
