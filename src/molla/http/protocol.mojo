"""HTTP/1.1 on the reactor.

`molla.http.server` is the M0 spike: a whole server in one file that answers
every path with the same bytes, kept because it is the evidence behind the
throughput numbers in D1. This is the real one. It implements the four calls in
`molla.net.reactor.Protocol` and gets the event loop, the sharded accept, the
idle timer and the write ring for free.

## One state per slot, not one per request

The reactor holds one protocol object per worker thread and hands it a
`Connection` on every call, so anything that has to survive between calls lives
in `states`, indexed by `conn.slot`. The reactor reuses slots, so `on_open`
resets the entry rather than trusting what the last connection left there.

Each state owns its own `ResponseWriter`, which is what makes the allocation
story work. The writer's buffer grows to whatever that client's responses need
over the first few requests and is then reused forever, so a keep alive
connection serving a million requests allocates for the first few and never
again.

## Why the header block is copied out when there is a body, and not otherwise

Spans in a parsed `Request` are offsets into the connection's read buffer, so
they are only valid until that buffer is consumed. For a request with no body,
which is every GET and HEAD and therefore almost all of them, nothing has to be
consumed before the response is written, so the handler reads the spans
directly and nothing is copied at all.

A request with a body is different. The body has to be consumed out of the read
buffer as it arrives or the buffer grows to the size of the upload, and once it
is consumed the header spans point at whatever came after. So the method and
the target, which are the two things a handler needs after the body is in, are
copied into a per connection buffer first. That buffer is reused across
requests exactly like the writer is, so the copy is a memcpy of a few dozen
bytes and not an allocation.

## Expect: 100-continue

Answered here rather than by the parser, because whether to accept a body is a
policy question and the parser does not make those. A client that sent it is
sitting waiting, so the interim response goes out before the body is read
rather than after, and it is written straight to the ring because forty two
bytes always fit.

## The keep alive cap

A connection is closed after `max_requests`. Not because anything degrades, but
because a connection that never closes never releases its slot, its buffers or
its file descriptor, and on a long running server that is how one client's
misbehaviour becomes everyone's problem. The cap is high enough that a normal
client never reaches it.

## Streaming

A streaming response is the only case where the protocol produces without being
asked, which is what `on_writable` is for. The state is one flag and a
`StreamWriter` per slot, and the pump loop runs the stream to completion or to
backpressure before it looks at the input again.

The two things that needed care are both about ordering. `_push_out` marks a
connection as closing once the last byte of a response is queued, and a
streaming response is only a header block at that point, so it does not count
as closing until the stream is done. And a stream produces into a bounded
staging buffer, so the pump has to stop on `STREAM_FULL` and hand back to the
reactor rather than growing the buffer until the reader catches up.
"""

from molla.http.body import (
    BODY_DONE,
    BODY_FAILED,
    BODY_NEED_MORE,
    BodyReader,
)
from molla.http.request import (
    PARSE_DONE,
    PARSE_FAILED,
    PARSE_NEED_MORE,
    Request,
    span_eq,
)
from molla.http.request import parse as parse_request
from molla.http.serialize import ResponseWriter, write_decimal
from molla.http.stream import (
    STREAM_FULL,
    STREAM_OK,
    StreamWriter,
    ndjson_headers,
    sse_headers,
)
from molla.io.buffer import Buffer
from molla.net.conn import Connection
from molla.net.reactor import Protocol
from molla.ops.log import LEVEL_DEBUG, LEVEL_ERROR, Logger
from molla.ops.metrics import (
    M_BYTES_READ,
    M_BYTES_WRITTEN,
    M_CONNECTIONS_ACCEPTED,
    M_CONNECTIONS_OPEN,
    M_HANDLER_ERRORS,
    M_REQUESTS,
    Meter,
    MetricsView,
)
from molla.build_info import MOJO_PIN, VERSION
from molla.sys.clock import monotonic_ms, monotonic_ns
from molla.sys.mem import as_ptr

comptime DEFAULT_MAX_REQUESTS = 10000
"""Requests on one connection before it is closed."""

comptime DEFAULT_MAX_BODY = 64 << 20
"""Largest request body accepted. Over this is a 413 rather than a disk full."""

comptime MAX_PENDING_HEADER_BYTES = 65536
"""Unparsed input a connection may hold before the request line and headers are
complete. Over this is a 431, and it is what stops a client from opening a
connection and dribbling header bytes forever."""

comptime MAX_TARGET_BYTES = 8192
"""Copied target length. Longer than this was already a 414 in the parser, so
reaching the cap here means something changed there."""

comptime DEFAULT_STREAM_EVENTS = 8
"""Events the stand in streaming routes emit before ending. A demo number, and
the seam where a token loop goes in M2."""


struct ConnState(Movable):
    """Everything about one connection that outlives a single call."""

    var writer: ResponseWriter
    var out_at: Int
    """How much of `writer` has made it into the ring. Non zero means the ring
    filled up and the rest goes out from `on_writable`."""

    var line: Buffer
    """Method then target, copied for a request that has a body."""

    var method_len: Int
    var target_len: Int
    var body: BodyReader
    var reading_body: Bool
    var keep_alive: Bool
    var is_head: Bool
    var closing: Bool
    """A response has been written and the connection goes away once it is out.
    Distinct from the reactor's own closing flag, which cuts the socket."""

    var requests: Int
    var counter: Int
    var max_body: Int

    var stream: StreamWriter
    var streaming: Bool
    """A response whose headers are out and whose body is still being made."""

    var stream_left: Int
    var stream_index: Int
    var scratch: Buffer
    """Where a stream event's payload is built, reused like everything else
    here, so producing an event does not allocate."""

    var metered_in: Int
    var metered_out: Int
    """How much of this connection's byte totals has already been added to the
    meter. A `Connection` counts bytes for its whole life, so the protocol
    reports the difference each time it looks and the counter advances while a
    keep alive connection is still open rather than in one jump when it
    closes."""

    var started_ns: Int
    """When the request being answered arrived, on the monotonic clock. Zero
    between requests."""

    def __init__(out self, counter: Int, max_body: Int):
        self.writer = ResponseWriter(counter)
        self.out_at = 0
        self.line = Buffer(MAX_TARGET_BYTES, counter)
        self.method_len = 0
        self.target_len = 0
        self.body = BodyReader(0, 0, max_body, counter)
        self.reading_body = False
        self.keep_alive = True
        self.is_head = False
        self.closing = False
        self.requests = 0
        self.counter = counter
        self.max_body = max_body
        self.stream = StreamWriter(counter)
        self.streaming = False
        self.stream_left = 0
        self.stream_index = 0
        self.scratch = Buffer(256, counter)
        self.metered_in = 0
        self.metered_out = 0
        self.started_ns = 0

    def reset(mut self):
        """Ready for a new connection in this slot, keeping every buffer."""
        self.writer.reset()
        self.out_at = 0
        self.line.clear()
        self.method_len = 0
        self.target_len = 0
        self.body = BodyReader(0, 0, self.max_body, self.counter)
        self.reading_body = False
        self.keep_alive = True
        self.is_head = False
        self.closing = False
        self.requests = 0
        self.stream.abort()
        self.streaming = False
        self.stream_left = 0
        self.stream_index = 0
        self.scratch.clear()
        self.metered_in = 0
        self.metered_out = 0
        self.started_ns = 0

    def method(self) -> Span[UInt8, MutAnyOrigin]:
        return Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(self.line.base()), length=self.method_len
        )

    def target(self) -> Span[UInt8, MutAnyOrigin]:
        return Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(self.line.base() + self.method_len),
            length=self.target_len,
        )


def _span_is[o: MutOrigin](data: Span[UInt8, o], text: StringSpan) -> Bool:
    if len(data) != text.byte_length():
        return False
    var p = text.unsafe_ptr()
    for i in range(len(data)):
        if data[i] != p.unsafe_load(i):
            return False
    return True


comptime ROUTE_NONE = 0
comptime ROUTE_ROOT = 1
comptime ROUTE_HEALTHZ = 2
comptime ROUTE_SSE = 3
comptime ROUTE_NDJSON = 4
comptime ROUTE_BOOM = 5
comptime ROUTE_ADMIN_VERSION = 6
comptime ROUTE_ADMIN_HEALTH = 7
comptime ROUTE_ADMIN_METRICS = 8
"""One integer per path this stand in answers.

It used to be one Bool per route passed down through three functions, which was
fine at four routes and would have been nine arguments at nine. Resolving once
into an integer also means the two callers, the one that has spans into the read
buffer and the one that has them copied out, agree on the answer by
construction rather than by both being edited the same way.
"""


def _route_of[o: MutOrigin](target: Span[UInt8, o]) -> Int:
    """Which route a target names, or `ROUTE_NONE` for a 404.

    A chain of comparisons rather than a router. Nine paths, all short, all
    known at compile time, and M2 replaces the whole thing with a real router.
    Building one now would be building it twice.
    """
    if _span_is(target, "/"):
        return ROUTE_ROOT
    if _span_is(target, "/healthz"):
        return ROUTE_HEALTHZ
    if _span_is(target, "/stream/sse"):
        return ROUTE_SSE
    if _span_is(target, "/stream/ndjson"):
        return ROUTE_NDJSON
    if _span_is(target, "/boom"):
        return ROUTE_BOOM
    if _span_is(target, "/molla/version"):
        return ROUTE_ADMIN_VERSION
    if _span_is(target, "/molla/health"):
        return ROUTE_ADMIN_HEALTH
    if _span_is(target, "/molla/metrics"):
        return ROUTE_ADMIN_METRICS
    return ROUTE_NONE


def _is_admin(route: Int) -> Bool:
    """Whether a route id is one of the `/molla` three."""
    return (
        route == ROUTE_ADMIN_VERSION
        or route == ROUTE_ADMIN_HEALTH
        or route == ROUTE_ADMIN_METRICS
    )


struct HttpProtocol(Movable, Protocol):
    """HTTP/1.1 request and response framing for the reactor."""

    var states: List[ConnState]
    var req: Request
    """One parser state for the whole thread. Parsing runs to completion inside
    a single call, so there is never a second request being parsed at once."""

    var counter: Int
    var max_requests: Int
    var max_body: Int
    var stream_events: Int

    var opened: Int
    var closed: Int
    var requests: Int
    var errors: Int
    """Responses with a 4xx or 5xx status that molla produced itself, which is
    the number that says the input was bad rather than the handler was."""

    var streams: Int
    var aborted: Int
    """Streams whose client went away before the last chunk."""

    var handler_errors: Int
    """Requests whose handler raised. Counted apart from `errors`, because a
    bad request and a broken handler are somebody else's problem and ours."""

    var allow_fault: Bool
    """Whether the `/boom` route exists. Off unless a caller asks."""

    var admin: Bool
    """Whether the `/molla` routes exist. Off unless a caller asks, so that a
    server nobody configured does not answer questions about itself."""

    var logger: Logger
    var meter: Meter
    """This worker's end of the log ring and of the counters. Both are
    addresses, both are safe to copy into a reactor, and both do nothing at all
    when they were never configured, which is why no call site checks."""

    var metrics: MetricsView
    """Every worker's counters, for the one route that adds them up."""

    def __init__(out self):
        self.states = List[ConnState]()
        self.req = Request()
        self.counter = 0
        self.max_requests = DEFAULT_MAX_REQUESTS
        self.max_body = DEFAULT_MAX_BODY
        self.stream_events = DEFAULT_STREAM_EVENTS
        self.opened = 0
        self.closed = 0
        self.requests = 0
        self.errors = 0
        self.streams = 0
        self.aborted = 0
        self.handler_errors = 0
        self.allow_fault = False
        self.admin = False
        self.logger = Logger()
        self.meter = Meter()
        self.metrics = MetricsView()

    def configure_fault(mut self, enabled: Bool):
        """Turn the raising route on. Only a test or `molla drain` does this."""
        self.allow_fault = enabled

    def configure(mut self, counter: Int, max_requests: Int, max_body: Int):
        """Set the limits before the reactor starts.

        Separate from `__init__` because the trait needs a no argument
        constructor, and taking the settings through a method is less trouble
        than threading a config object through the reactor's parameter.
        """
        self.counter = counter
        self.max_requests = max_requests
        self.max_body = max_body

    def configure_ops(
        mut self,
        logger: Logger,
        meter: Meter,
        metrics: MetricsView,
        admin: Bool,
    ):
        """Hand this worker its log ring, its counters, and the shared view.

        A method for the same reason `configure` is one: the trait needs a no
        argument constructor. The default is a logger and a meter that write
        nowhere, so a server that never calls this is a server that spends
        nothing on either.
        """
        self.logger = logger
        self.meter = meter
        self.metrics = metrics
        self.admin = admin

    def configure_stream(mut self, events: Int):
        """How many events the stand in streaming routes emit. Separate from
        `configure` so a test can turn one knob without restating the rest."""
        self.stream_events = events

    def _ensure(mut self, slot: Int):
        while len(self.states) <= slot:
            self.states.append(ConnState(self.counter, self.max_body))

    def on_open(mut self, mut conn: Connection):
        self._ensure(conn.slot)
        self.states[conn.slot].reset()
        self.opened += 1
        self.meter.inc(M_CONNECTIONS_ACCEPTED)
        self.meter.inc(M_CONNECTIONS_OPEN)

    def on_close(mut self, mut conn: Connection):
        """The socket is going away, so a stream in flight is over.

        `abort` rather than `end`, because there is nothing to write a terminal
        chunk into. A client that hangs up in the middle of a completion is the
        ordinary case, not an error, and the only thing owed to it is releasing
        what was staged for it.
        """
        if conn.slot < len(self.states):
            if self.states[conn.slot].streaming:
                self.aborted += 1
                self.states[conn.slot].stream.abort()
                self.states[conn.slot].streaming = False
            self._meter_bytes(conn.slot, conn)
        self.closed += 1
        self.meter.dec(M_CONNECTIONS_OPEN)

    def on_writable(mut self, mut conn: Connection) -> Bool:
        """The ring drained. Push the rest of the response and carry on."""
        return self._pump(conn)

    def on_readable(mut self, mut conn: Connection) -> Bool:
        return self._pump(conn)

    def _meter_bytes(mut self, slot: Int, mut conn: Connection):
        """Add whatever this connection has moved since the last look.

        A `Connection` counts bytes for its whole life, so the difference is
        what belongs to the meter. Doing it this way rather than once at close
        means a keep alive connection that stays open for an hour shows up in
        the counters during that hour instead of in one jump at the end.
        """
        var moved_in = conn.bytes_in - self.states[slot].metered_in
        if moved_in > 0:
            self.meter.add(M_BYTES_READ, moved_in)
            self.states[slot].metered_in = conn.bytes_in
        var moved_out = conn.bytes_out - self.states[slot].metered_out
        if moved_out > 0:
            self.meter.add(M_BYTES_WRITTEN, moved_out)
            self.states[slot].metered_out = conn.bytes_out

    def _pump(mut self, mut conn: Connection) -> Bool:
        """Move the connection forward as far as it will go.

        One loop rather than one request per call, because a pipelining client
        puts several requests in one segment and the readiness that brought us
        here has already been spent. It stops when the ring is full, when the
        input runs out, or when the connection is finished.
        """
        var slot = conn.slot
        self._ensure(slot)
        while True:
            if not self._push_out(slot, conn):
                # The ring is full. `on_writable` picks this up.
                conn.produce(self.states[slot].streaming)
                self._meter_bytes(slot, conn)
                return True
            if self.states[slot].closing:
                # `finish` rather than returning False, and the difference
                # matters. Returning False tells the reactor to stop servicing
                # the connection, and it stops before flushing, so the response
                # that says the connection is closing never leaves the ring.
                # `finish` says the same thing without cutting the write short:
                # the reactor drains what is queued and closes after.
                self._meter_bytes(slot, conn)
                conn.finish()
                return True

            if self.states[slot].streaming:
                # The headers are out and the body is still being made. Run it
                # until it finishes or until the reader stops keeping up.
                if not self._pump_stream(slot, conn):
                    conn.produce(True)
                    return True
                conn.produce(False)
                continue

            if self.states[slot].reading_body:
                if not self._read_body(slot, conn):
                    return True
                continue

            if conn.input.length == 0:
                self._meter_bytes(slot, conn)
                return True

            var rc = parse_request(
                as_ptr(conn.input.base()), conn.input.length, self.req
            )

            if rc == PARSE_NEED_MORE:
                if conn.input.length > MAX_PENDING_HEADER_BYTES:
                    self._error(slot, conn, 431)
                    continue
                self._meter_bytes(slot, conn)
                return True

            if rc == PARSE_FAILED:
                self._error(slot, conn, self.req.error_status)
                continue

            self._accept(slot, conn)

    def _accept(mut self, slot: Int, mut conn: Connection):
        """Take a parsed request and either answer it or start reading a body.

        Returns without answering when there is a body, and the next turn of
        the pump loop goes through `_read_body` instead.
        """
        self.states[slot].keep_alive = self.req.keep_alive
        self.states[slot].is_head = self.req.is_head
        self.states[slot].requests += 1
        self.requests += 1
        self.meter.inc(M_REQUESTS)
        self.states[slot].started_ns = monotonic_ns()
        if self.states[slot].requests >= self.max_requests:
            self.states[slot].keep_alive = False

        if self.req.expect_continue:
            # Straight to the ring. An interim response is forty two bytes and
            # it has to be out before the client will send the body, so putting
            # it through the writer, which the real response also needs, would
            # mean holding two responses at once for no gain.
            _ = conn.queue_str("HTTP/1.1 100 Continue\r\n\r\n")

        if not self.req.has_body():
            var header_bytes = self.req.header_bytes
            self._respond(slot, conn)
            conn.input.consume(header_bytes)
            return

        if self.req.content_length > self.max_body:
            self._error(slot, conn, 413)
            return

        if not self._copy_line(slot, as_ptr(conn.input.base())):
            self._error(slot, conn, 414)
            return

        self.states[slot].body = BodyReader(
            self.req.body_kind,
            self.req.content_length,
            self.max_body,
            self.counter,
        )
        self.states[slot].reading_body = True
        conn.input.consume(self.req.header_bytes)

    def _copy_line[
        o: MutOrigin
    ](mut self, slot: Int, base: Pointer[UInt8, o]) -> Bool:
        """Keep the method and the target past the point the read buffer moves.

        Appended into one buffer back to back rather than into two, so a
        connection carries one reusable block for both and the offsets say
        where each one is.
        """
        self.states[slot].line.clear()
        var total = self.req.method.length + self.req.target.length
        if total > MAX_TARGET_BYTES:
            return False
        var method = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(Int(base) + self.req.method.start),
            length=self.req.method.length,
        )
        if not self.states[slot].line.append(method):
            return False
        var target = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(Int(base) + self.req.target.start),
            length=self.req.target.length,
        )
        if not self.states[slot].line.append(target):
            return False
        self.states[slot].method_len = self.req.method.length
        self.states[slot].target_len = self.req.target.length
        return True

    def _read_body(mut self, slot: Int, mut conn: Connection) -> Bool:
        """Feed what arrived to the body reader. True when the body finished."""
        if conn.input.length == 0:
            return False
        var status = self.states[slot].body.feed(
            as_ptr(conn.input.base()), 0, conn.input.length
        )
        var took = self.states[slot].body.last_consumed()
        if took > 0:
            conn.input.consume(took)
        if status == BODY_FAILED:
            var code = self.states[slot].body.error_status
            self._error(slot, conn, code if code != 0 else 400)
            return True
        if status == BODY_NEED_MORE:
            return False
        self.states[slot].reading_body = False
        self._respond_after_body(slot, conn)
        return True

    def _push_out(mut self, slot: Int, mut conn: Connection) -> Bool:
        """Move the written response into the ring. False means it did not all
        fit and the rest is owed."""
        var total = self.states[slot].writer.length()
        if self.states[slot].out_at >= total:
            return True
        var at = self.states[slot].out_at
        var chunk = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(self.states[slot].writer.out.base() + at),
            length=total - at,
        )
        var took = conn.queue(chunk)
        self.states[slot].out_at += took
        if self.states[slot].out_at < total:
            return False
        # Fully queued. Reset so the next response starts from zero without
        # having to remember to.
        self.states[slot].writer.reset()
        self.states[slot].out_at = 0
        if not self.states[slot].keep_alive and not self.states[slot].streaming:
            # A streaming response has only had its headers queued at this
            # point, so it is not finished and the connection is not closing
            # yet. `_pump_stream` sets the flag when the last chunk is out.
            self.states[slot].closing = True
        return True

    def _pump_stream(mut self, slot: Int, mut conn: Connection) -> Bool:
        """Produce and flush until the stream ends or the reader falls behind.

        True means the stream is complete and the pump can look at the input
        again. False means there are bytes the ring would not take, and
        `on_writable` comes back to this.
        """
        var now = monotonic_ms()
        while True:
            # Flush before producing rather than after, so a single event on an
            # idle connection is framed and queued on its own instead of waiting
            # for a second one that may be thirty seconds away.
            _ = self.states[slot].stream.flush(conn)
            if self.states[slot].stream.is_done():
                break
            if self.states[slot].stream_left <= 0:
                if self.states[slot].stream.ended:
                    # The terminal chunk is staged and the ring would not take
                    # it, which is the only thing left to wait for.
                    return False
                _ = self.states[slot].stream.end(now)
                continue
            # Producing while the ring is full is deliberate and is what the
            # staging limit is for. A token generator that stalls because a
            # socket is momentarily full is a generator running at the speed of
            # the slowest reader, and the events that pile up go out as one
            # chunk rather than as one chunk each.
            if self._write_stream_event(slot, now) == STREAM_FULL:
                return False
            self.states[slot].stream_left -= 1

        self.states[slot].streaming = False
        self.states[slot].stream_left = 0
        if not self.states[slot].keep_alive:
            self.states[slot].closing = True
        return True

    def _write_stream_event(mut self, slot: Int, now: Int) -> Int:
        """One event from the stand in generator.

        The payload is built into the connection's scratch buffer rather than
        into a String, so the streaming path allocates as little as the plain
        response path does. In M2 this is where a token goes.
        """
        var index = self.states[slot].stream_index
        self.states[slot].stream_index = index + 1
        self.states[slot].scratch.clear()
        _ = self.states[slot].scratch.append_str('{"i":')
        _ = write_decimal(self.states[slot].scratch, index)
        _ = self.states[slot].scratch.append_str("}")
        var payload = self.states[slot].scratch.bytes()
        if self.states[slot].stream.sse:
            return self.states[slot].stream.event("token", payload, "", now)
        return self.states[slot].stream.record(payload, now)

    def _error(mut self, slot: Int, mut conn: Connection, status: Int):
        """Answer with a status and stop reading.

        Everything still in the buffer is dropped. Once framing is in doubt the
        remaining bytes cannot be trusted to be a request rather than the tail
        of the one that went wrong, and reading them is exactly how a smuggled
        request gets a second chance.

        The accounting at the end is the same call `_write_default` makes, and
        it was missing until the HTTP soak in issue #18 sent a million oversized
        bodies and the server reported zero 4xx responses. A 413, a 414 and a
        431 are answers, they are the answers an operator most wants a graph
        of, and they never go through the handler path that was doing the
        counting.
        """
        self.errors += 1
        self.states[slot].keep_alive = False
        self.states[slot].reading_body = False
        _ = self.states[slot].writer.respond_error(
            status, self.states[slot].is_head
        )
        self.states[slot].out_at = 0
        conn.input.clear()
        self._account(slot)

    def _respond(mut self, slot: Int, mut conn: Connection):
        """Answer a request whose spans are still valid in the read buffer."""
        var head = self.req.is_head
        var keep = self.states[slot].keep_alive
        var base = as_ptr(conn.input.base())
        var is_get = span_eq(base, self.req.method, "GET") or head
        var target = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=base.unsafe_offset(self.req.target.start),
            length=self.req.target.length,
        )
        self.states[slot].out_at = 0
        self._write_default(slot, _route_of(target), is_get, keep, head)

    def _respond_after_body(mut self, slot: Int, mut conn: Connection):
        """Answer a request whose body has just finished arriving."""
        var head = self.states[slot].is_head
        var keep = self.states[slot].keep_alive
        var is_get = _span_is(self.states[slot].method(), "GET") or head
        var route = _route_of(self.states[slot].target())
        self.states[slot].out_at = 0
        self._write_default(slot, route, is_get, keep, head)

    def _write_default(
        mut self,
        slot: Int,
        route: Int,
        is_get: Bool,
        keep: Bool,
        head: Bool,
    ):
        """Run the handler, and answer with a 500 if it blows up.

        This is the boundary the whole server depends on and it is three lines.
        A handler that raises is a bug in that handler, and the two things that
        must not follow from it are a client left hanging on a connection that
        will never answer, and a process that takes everybody else's requests
        down with it. So the failure is turned into a 500, the connection is
        closed because nothing knows how much of a response was already
        written, and the reactor never finds out.

        What it does not cover is a genuine crash. A null dereference or a
        stack overflow in a handler takes the process down and no language
        level construct can catch it, which is why the memory rules in
        `molla.sys.mem` matter more here than the error handling does.
        """
        try:
            self._route(slot, route, is_get, keep, head)
        except e:
            self.handler_errors += 1
            self.errors += 1
            self.meter.inc(M_HANDLER_ERRORS)
            var entry = self.logger.begin(LEVEL_ERROR)
            entry.message("handler raised")
            entry.field_int("slot", slot)
            entry.field("detail", String(e))
            _ = entry.end()
            _ = self.states[slot].writer.respond_error(500, head)
            self.states[slot].keep_alive = False
            self.states[slot].streaming = False
        self._account(slot)

    def _account(mut self, slot: Int):
        """Record what the response was and how long it took.

        Called once per request from `_write_default`, after the handler has
        run and whether or not it raised, which is why it is here and not at
        the end of `_route`. The status comes off the writer rather than being
        threaded back out of the handler, since the writer is the thing that
        actually decided.
        """
        var status = self.states[slot].writer.status
        self.meter.observe_status(status)
        if self.states[slot].started_ns > 0:
            self.meter.observe_request_ns(
                monotonic_ns() - self.states[slot].started_ns
            )
            self.states[slot].started_ns = 0
        if status >= 500 or self.logger.enabled(LEVEL_DEBUG):
            var entry = self.logger.begin(
                LEVEL_ERROR if status >= 500 else LEVEL_DEBUG
            )
            entry.message("request")
            entry.field_int("status", status)
            entry.field_int("slot", slot)
            _ = entry.end()

    def _route(
        mut self,
        slot: Int,
        route: Int,
        is_get: Bool,
        keep: Bool,
        head: Bool,
    ) raises:
        """The stand in handler, and the three admin routes that are real.

        The application routes are a stand in: a root, a health check and two
        streams, which is enough to exercise framing end to end and is
        deliberately not a router. M2 replaces them with one, and the seam is
        here so that when it does, nothing above or below has to move.

        The `/molla` routes are not a stand in. They are what #16 asks for and
        what an operator gets: a version, a health check that answers before
        anything else is ready, and the Prometheus exposition. They are served
        on the same port as everything else, which is a decision worth being
        explicit about. A second listener on a second port is the safer answer
        and it is also a second thing to configure, a second thing to expose in
        a container, and a second thing to forget. Everything here is
        information molla already prints on startup, and the day one of these
        routes can change something is the day the port question gets asked
        again.

        `/boom` raises, and only exists when a caller has asked for it with
        `configure_fault`. A server that answers an unauthenticated request by
        running the error path on purpose is not something to ship on by
        default, and a property nobody can trigger is not something to claim
        either, so the tests and `molla drain` turn it on and nothing else
        does.
        """
        if route == ROUTE_BOOM and self.allow_fault:
            raise Error("the handler for /boom raised, which is its whole job")
        if route == ROUTE_HEALTHZ:
            _ = self.states[slot].writer.respond_str(
                200, "text/plain", "ok\n", keep, head
            )
            return
        if route == ROUTE_SSE or route == ROUTE_NDJSON:
            if not is_get:
                self._refuse(slot, 405, head)
                return
            self._start_stream(slot, route == ROUTE_SSE, keep, head)
            return
        if _is_admin(route):
            if not self.admin:
                self._refuse(slot, 404, head)
                return
            if not is_get:
                self._refuse(slot, 405, head)
                return
            self._admin(slot, route, keep, head)
            return
        if route != ROUTE_ROOT:
            self._refuse(slot, 404, head)
            return
        if not is_get:
            self._refuse(slot, 405, head)
            return
        _ = self.states[slot].writer.respond_str(
            200, "text/plain", "Hello from molla.\n", keep, head
        )

    def _refuse(mut self, slot: Int, status: Int, head: Bool):
        """Answer with a status molla chose and stop reading this connection.

        Every refusal in the handler did these same three things and one of
        them forgot the keep alive once, so they are one call now.
        """
        self.errors += 1
        _ = self.states[slot].writer.respond_error(status, head)
        self.states[slot].keep_alive = False

    def _admin(mut self, slot: Int, route: Int, keep: Bool, head: Bool):
        """Answer one of the `/molla` routes.

        The body is built in the connection's scratch buffer, which is the same
        buffer a stream event uses and is reused across requests, so a scrape
        costs a memcpy rather than an allocation. A metrics exposition is a few
        kilobytes and the buffer grows to that on the first scrape and stays
        there.
        """
        self.states[slot].scratch.clear()
        var kind = StaticString("text/plain")
        if route == ROUTE_ADMIN_HEALTH:
            _ = self.states[slot].scratch.append_str("ok\n")
        elif route == ROUTE_ADMIN_VERSION:
            # Not JSON, on purpose. This is the answer to a person running
            # curl, it is the same information `molla version` prints, and
            # anything that wants to parse it wants /molla/metrics instead.
            _ = self.states[slot].scratch.append_str("molla ")
            _ = self.states[slot].scratch.append_str(VERSION)
            _ = self.states[slot].scratch.append_str("\nmojo ")
            _ = self.states[slot].scratch.append_str(MOJO_PIN)
            _ = self.states[slot].scratch.append_str("\n")
        else:
            # The exposition format wants this exact content type, down to the
            # version parameter, or Prometheus falls back to a guess.
            kind = "text/plain; version=0.0.4; charset=utf-8"
            _ = self.metrics.render(self.states[slot].scratch, VERSION)
        _ = self.states[slot].writer.respond(
            200, kind, self.states[slot].scratch.bytes(), keep, head
        )

    def _start_stream(mut self, slot: Int, sse: Bool, keep: Bool, head: Bool):
        """Write the header block and arm the stream.

        The headers go through the ordinary writer and out through `_push_out`
        like any other response. Only the body is different, which is the point
        of keeping streaming out of `ResponseWriter` entirely.
        """
        var ok: Bool
        if sse:
            ok = sse_headers(self.states[slot].writer, keep, head)
        else:
            ok = ndjson_headers(self.states[slot].writer, keep, head)
        if not ok:
            self.errors += 1
            _ = self.states[slot].writer.respond_error(500, head)
            self.states[slot].keep_alive = False
            return
        self.streams += 1
        self.states[slot].stream.begin(sse, head, monotonic_ms())
        self.states[slot].streaming = True
        self.states[slot].stream_left = self.stream_events
        self.states[slot].stream_index = 0
