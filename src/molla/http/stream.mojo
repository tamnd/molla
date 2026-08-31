"""Server-sent events and NDJSON, framed over chunked transfer encoding.

A completion is not a response, it is a response that takes thirty seconds to
arrive. Everything about that is different: the length is unknown when the
headers go out, the client is reading while the server is still generating, and
the slowest part of the system is usually the reader rather than the writer.
This is the framing layer for that, and it is deliberately separate from
`serialize.mojo`, which writes a response whose body is already in hand.

## Flush per event, coalesce only under backpressure

The rule the issue asks for is that an event goes out on its own as soon as it
exists, and that events only get combined when the socket cannot take them.
That falls out of holding two buffers rather than one.

`pending` holds framed payload with no chunk framing on it yet. `wire` holds
chunk framed bytes that have not all been accepted by the connection's output
ring. `flush` drains `wire` first, and only when `wire` is empty does it wrap
whatever is in `pending` into a single chunk.

The caller flushes before producing rather than after. In the normal case,
where the ring has room, that means each event is staged, wrapped and queued
before the next one exists, which is one chunk per event and no waiting for a
second event that may be thirty seconds away. When the ring is full, `flush`
cannot empty `wire`, later events pile up in `pending`, and the next flush that
gets room wraps all of them into one chunk.

Nothing special cases this. Coalescing is what happens when the writer is
faster than the reader, which is the only time it is wanted.

The producer is expected to keep going while the ring is full rather than stall
with it, which is what the staging limit is for. A token generator that stops
because a socket is momentarily full is a generator running at the speed of the
slowest reader connected to it.

## Backpressure is a return value, not a bigger buffer

`staged` is capped at `limit`. A producer that would push past it gets
`STREAM_FULL` and is expected to stop and wait to be asked again, which for the
reactor means `on_writable`.

The alternative, growing the buffer until the reader catches up, is how a
server with one slow client runs out of memory. A local inference server makes
that worse rather than better: the thing on the other end of a slow SSE stream
is often a browser tab that has been backgrounded, and the thing producing is a
GPU that does not care.

## The heartbeat is a decision, not a timer

`heartbeat_due` takes the current time and answers, and `heartbeat` stages the
comment. Neither reads a clock. The reactor's `Protocol` trait is four calls on
purpose and none of them is a tick, so the wakeup that drives this belongs to
whatever is producing tokens. Passing the time in also makes the test
deterministic instead of a sleep.

A heartbeat is only due when nothing is staged. If bytes are already waiting
because the reader is slow, the stream is not silent and another comment on the
pile helps nobody.

There are two reasons to send it. The documented one is that proxies drop a
connection with no traffic on it, and a long prefill produces no traffic at
all. The other one is ours: the reactor touches a connection when a flush
actually writes, so a stream that never writes looks idle to our own idle
timeout and gets closed by the server it is running on.

NDJSON has no heartbeat. There is no comment syntax to hide one in, and a blank
line is not portably ignored by readers, so an NDJSON stream that has to
survive a proxy needs the application to send a real progress record.

## Framing is validated, because this is where clients quietly break

An event name or id containing a newline ends the event early and the rest of
it arrives as fields of the next one. An NDJSON record containing a newline is
two records. Both produce a client that misbehaves with no error anywhere, so
both are refused with `STREAM_INVALID` rather than written.

Data is different. A payload with newlines in it is legal SSE and is handled by
splitting it across several `data:` lines, which is what the format is for. All
three line endings are split on, because a lone CR inside a payload is a line
break to an SSE client and is not one to a naive writer.
"""

from molla.http.serialize import (
    CRLF,
    ResponseWriter,
    chunk_header,
    write_decimal,
)
from molla.io.buffer import Buffer
from molla.net.conn import Connection
from molla.sys.mem import as_ptr

comptime STREAM_OK = 0
"""Staged. Not necessarily sent, and not necessarily framed yet."""

comptime STREAM_FULL = 1
"""The staging limit is reached. Stop producing and wait to be asked again."""

comptime STREAM_CLOSED = 2
"""The stream has ended or been aborted. Nothing more will be sent."""

comptime STREAM_INVALID = 3
"""The caller passed something that would corrupt the framing."""

comptime DEFAULT_STREAM_LIMIT = 256 << 10
"""Bytes a stream may hold for a reader that is not keeping up, before the
producer is told to stop."""

comptime DEFAULT_HEARTBEAT_MS = 15000
"""Silence before a comment goes out, in milliseconds."""

comptime _CR: UInt8 = 13
comptime _LF: UInt8 = 10
comptime _NUL: UInt8 = 0


def sse_headers(
    mut w: ResponseWriter, keep_alive: Bool, head_only: Bool = False
) -> Bool:
    """The header block for `text/event-stream`.

    `X-Accel-Buffering: no` is not a standard header and is here anyway,
    because nginx buffers a proxied response by default and a buffered SSE
    stream arrives all at once at the end, which looks exactly like a server
    that is hanging.
    """
    if not w.start(200, head_only):
        return False
    if not w.header("Content-Type", "text/event-stream"):
        return False
    if not w.header("Cache-Control", "no-cache"):
        return False
    if not w.header("Transfer-Encoding", "chunked"):
        return False
    if not w.header("X-Accel-Buffering", "no"):
        return False
    if not w.header("Server", "molla"):
        return False
    if not w.header_date():
        return False
    if not w.header_connection(keep_alive):
        return False
    return w.end_headers()


def ndjson_headers(
    mut w: ResponseWriter, keep_alive: Bool, head_only: Bool = False
) -> Bool:
    """The header block for `application/x-ndjson`.

    `application/x-ndjson` rather than `application/jsonl` or a bare
    `application/json`, because it is what the clients this has to work with
    already send and expect, and because `application/json` on a stream tells a
    proxy it may buffer the whole thing to parse it.
    """
    if not w.start(200, head_only):
        return False
    if not w.header("Content-Type", "application/x-ndjson"):
        return False
    if not w.header("Cache-Control", "no-cache"):
        return False
    if not w.header("Transfer-Encoding", "chunked"):
        return False
    if not w.header("X-Accel-Buffering", "no"):
        return False
    if not w.header("Server", "molla"):
        return False
    if not w.header_date():
        return False
    if not w.header_connection(keep_alive):
        return False
    return w.end_headers()


def _has_break(data: Span[UInt8, _]) -> Bool:
    """True if a CR, LF or NUL is anywhere in the bytes."""
    for i in range(len(data)):
        var c = data[i]
        if c == _CR or c == _LF or c == _NUL:
            return True
    return False


def _lines(data: Span[UInt8, _]) -> Int:
    """How many `data:` lines a payload turns into.

    CRLF is one line ending and not two. A payload that ends with a line
    ending gets one more empty line, which the counting here already gives
    because the break itself is what increments.
    """
    var count = 1
    var i = 0
    var n = len(data)
    while i < n:
        if data[i] == _CR:
            count += 1
            i += 2 if i + 1 < n and data[i + 1] == _LF else 1
        elif data[i] == _LF:
            count += 1
            i += 1
        else:
            i += 1
    return count


def _str_has_break(text: StringSpan) -> Bool:
    var p = text.unsafe_ptr()
    for i in range(text.byte_length()):
        var c = p.unsafe_load(i)
        if c == _CR or c == _LF or c == _NUL:
            return True
    return False


struct StreamWriter(Movable):
    """One streaming response body, from the first event to the last chunk."""

    var pending: Buffer
    """Framed payload with no chunk framing on it yet. Everything here goes out
    as one chunk, which is what makes coalescing free."""

    var wire: Buffer
    """Chunk framed bytes. Not all of them are in the ring yet."""

    var wire_at: Int
    """How much of `wire` the ring has taken."""

    var limit: Int
    var sse: Bool
    var head_only: Bool
    var ended: Bool
    """`end` has been called. The terminal chunk may not be out yet."""

    var done: Bool
    """The terminal chunk has been queued. The response is complete."""

    var closed: Bool
    """Aborted. Nothing further is staged and nothing staged will be sent."""

    var last_write_ms: Int
    var heartbeat_ms: Int
    var events: Int
    var payload_bytes: Int
    var coalesced: Int
    """Chunks that carried more than one event, which is the count that says
    the reader was slower than the writer."""

    var _staged_events: Int

    def __init__(
        out self,
        counter: Int,
        limit: Int = DEFAULT_STREAM_LIMIT,
        capacity: Int = 4096,
    ):
        self.pending = Buffer(capacity, counter)
        self.wire = Buffer(capacity, counter)
        self.wire_at = 0
        self.limit = limit
        self.sse = True
        self.head_only = False
        self.ended = False
        self.done = False
        self.closed = False
        self.last_write_ms = 0
        self.heartbeat_ms = DEFAULT_HEARTBEAT_MS
        self.events = 0
        self.payload_bytes = 0
        self.coalesced = 0
        self._staged_events = 0

    def begin(mut self, sse: Bool, head_only: Bool, now_ms: Int):
        """Start a new stream on this writer, keeping both buffers.

        Called per response rather than per connection, for the same reason
        `ResponseWriter.reset` is: the capacity reached on the first stream is
        the capacity the next one needs.
        """
        self.pending.clear()
        self.wire.clear()
        self.wire_at = 0
        self.sse = sse
        self.head_only = head_only
        self.ended = False
        self.done = False
        self.closed = False
        self.last_write_ms = now_ms
        self.events = 0
        self.payload_bytes = 0
        self.coalesced = 0
        self._staged_events = 0

    def staged(self) -> Int:
        """Bytes held for a reader that has not taken them."""
        return self.pending.length + (self.wire.length - self.wire_at)

    def room(self) -> Int:
        var left = self.limit - self.staged()
        return left if left > 0 else 0

    def blocked(self) -> Bool:
        """True when there are bytes the ring would not take."""
        return self.staged() > 0

    def is_done(self) -> Bool:
        return self.done

    def _fits(self, need: Int) -> Bool:
        return self.staged() + need <= self.limit

    def abort(mut self):
        """Give up on the stream and drop what was staged.

        Used when the peer goes away mid response. Dropping the staged bytes
        rather than trying to write them is the point: the socket is gone, and
        a half chunk written into a closed connection is not more correct than
        nothing.
        """
        self.closed = True
        self.pending.clear()
        self.wire.clear()
        self.wire_at = 0
        self._staged_events = 0

    def comment(mut self, text: StringSpan, now_ms: Int) -> Int:
        """An SSE comment line, which every client ignores.

        Not available on NDJSON, which has no syntax for one.
        """
        if self.closed or self.ended:
            return STREAM_CLOSED
        if not self.sse:
            return STREAM_INVALID
        if _str_has_break(text):
            return STREAM_INVALID
        if self.head_only:
            self.last_write_ms = now_ms
            return STREAM_OK
        var need = text.byte_length() + 4
        if not self._fits(need):
            return STREAM_FULL
        if not self.pending.append_str(": "):
            return STREAM_FULL
        if not self.pending.append_str(text):
            return STREAM_FULL
        if not self.pending.append_byte(_LF):
            return STREAM_FULL
        if not self.pending.append_byte(_LF):
            return STREAM_FULL
        self._staged_events += 1
        self.last_write_ms = now_ms
        return STREAM_OK

    def heartbeat_due(self, now_ms: Int) -> Bool:
        """Whether a comment should go out now.

        Silence means nothing produced and nothing waiting. A stream with bytes
        staged behind a slow reader is busy, not silent.
        """
        if not self.sse or self.closed or self.ended or self.head_only:
            return False
        if self.staged() > 0:
            return False
        return now_ms - self.last_write_ms >= self.heartbeat_ms

    def heartbeat(mut self, now_ms: Int) -> Int:
        return self.comment("ping", now_ms)

    def retry(mut self, ms: Int, now_ms: Int) -> Int:
        """The `retry:` field, which sets the client's reconnect delay.

        Worth sending once at the start of a stream. The browser default is
        three seconds, and a client that reconnects three seconds after a
        dropped completion starts the whole generation again.
        """
        if self.closed or self.ended:
            return STREAM_CLOSED
        if not self.sse:
            return STREAM_INVALID
        if ms < 0:
            return STREAM_INVALID
        if self.head_only:
            self.last_write_ms = now_ms
            return STREAM_OK
        if not self._fits(32):
            return STREAM_FULL
        if not self.pending.append_str("retry: "):
            return STREAM_FULL
        if not write_decimal(self.pending, ms):
            return STREAM_FULL
        if not self.pending.append_byte(_LF):
            return STREAM_FULL
        if not self.pending.append_byte(_LF):
            return STREAM_FULL
        self._staged_events += 1
        self.last_write_ms = now_ms
        return STREAM_OK

    def event(
        mut self,
        name: StringSpan,
        data: Span[UInt8, _],
        id: StringSpan,
        now_ms: Int,
    ) -> Int:
        """One SSE event. An empty `name` or `id` leaves the field out.

        The data is split across `data:` lines on every line ending, so a
        payload that happens to contain one arrives as one event with a
        newline in it rather than as a truncated event and some garbage.
        """
        if self.closed or self.ended:
            return STREAM_CLOSED
        if not self.sse:
            return STREAM_INVALID
        if _str_has_break(name) or _str_has_break(id):
            return STREAM_INVALID
        self.events += 1
        self.payload_bytes += len(data)
        if self.head_only:
            self.last_write_ms = now_ms
            return STREAM_OK

        # Counted rather than bounded by the worst case, which would be seven
        # bytes of `data: ` framing per payload byte. A worst case bound makes
        # the effective limit a seventh of the real one, so a producer with a
        # payload larger than that gets STREAM_FULL forever against a limit it
        # would actually fit in, which is a deadlock and not backpressure.
        var need = (
            name.byte_length()
            + id.byte_length()
            + 16
            + len(data)
            + 7 * _lines(data)
        )
        if not self._fits(need):
            return STREAM_FULL

        if name.byte_length() > 0:
            if not self.pending.append_str("event: "):
                return STREAM_FULL
            if not self.pending.append_str(name):
                return STREAM_FULL
            if not self.pending.append_byte(_LF):
                return STREAM_FULL
        if id.byte_length() > 0:
            if not self.pending.append_str("id: "):
                return STREAM_FULL
            if not self.pending.append_str(id):
                return STREAM_FULL
            if not self.pending.append_byte(_LF):
                return STREAM_FULL

        var at = 0
        var n = len(data)
        while True:
            var stop = at
            while stop < n and data[stop] != _CR and data[stop] != _LF:
                stop += 1
            if not self.pending.append_str("data: "):
                return STREAM_FULL
            if stop > at:
                var line = Span[UInt8, MutAnyOrigin](
                    unsafe_ptr=as_ptr(Int(data.unsafe_ptr()) + at),
                    length=stop - at,
                )
                if not self.pending.append(line):
                    return STREAM_FULL
            if not self.pending.append_byte(_LF):
                return STREAM_FULL
            if stop >= n:
                break
            # CRLF is one line ending and not two, so a payload written on a
            # Windows box does not arrive with a blank line between every line.
            if data[stop] == _CR and stop + 1 < n and data[stop + 1] == _LF:
                at = stop + 2
            else:
                at = stop + 1
            if at >= n:
                # A trailing line ending means one more empty line, which is a
                # real part of the payload.
                if not self.pending.append_str("data: "):
                    return STREAM_FULL
                if not self.pending.append_byte(_LF):
                    return STREAM_FULL
                break

        if not self.pending.append_byte(_LF):
            return STREAM_FULL
        self._staged_events += 1
        self.last_write_ms = now_ms
        return STREAM_OK

    def event_str(
        mut self,
        name: StringSpan,
        data: StringSpan,
        id: StringSpan,
        now_ms: Int,
    ) -> Int:
        var span = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(Int(data.unsafe_ptr())),
            length=data.byte_length(),
        )
        return self.event(name, span, id, now_ms)

    def record(mut self, data: Span[UInt8, _], now_ms: Int) -> Int:
        """One NDJSON record, which is the bytes and a newline.

        A record containing a line ending is refused. It is not a record with a
        newline in it, it is two records, and the second one is not valid JSON.
        """
        if self.closed or self.ended:
            return STREAM_CLOSED
        if self.sse:
            return STREAM_INVALID
        if _has_break(data):
            return STREAM_INVALID
        self.events += 1
        self.payload_bytes += len(data)
        if self.head_only:
            self.last_write_ms = now_ms
            return STREAM_OK
        if not self._fits(len(data) + 1):
            return STREAM_FULL
        if not self.pending.append(data):
            return STREAM_FULL
        if not self.pending.append_byte(_LF):
            return STREAM_FULL
        self._staged_events += 1
        self.last_write_ms = now_ms
        return STREAM_OK

    def record_str(mut self, data: StringSpan, now_ms: Int) -> Int:
        var span = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(Int(data.unsafe_ptr())),
            length=data.byte_length(),
        )
        return self.record(span, now_ms)

    def end(mut self, now_ms: Int) -> Int:
        """No more events. The terminal chunk goes out on the next flush."""
        if self.closed:
            return STREAM_CLOSED
        if self.ended:
            return STREAM_OK
        self.ended = True
        self.last_write_ms = now_ms
        if self.head_only:
            self.done = True
        return STREAM_OK

    def _drain(mut self, mut conn: Connection) -> Int:
        if self.wire_at >= self.wire.length:
            return 0
        var chunk = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(self.wire.base() + self.wire_at),
            length=self.wire.length - self.wire_at,
        )
        var took = conn.queue(chunk)
        self.wire_at += took
        return took

    def _frame(mut self) -> Bool:
        """Wrap everything pending into exactly one chunk."""
        if self._staged_events > 1:
            self.coalesced += 1
        if not chunk_header(self.wire, self.pending.length):
            return False
        if not self.wire.append(self.pending.bytes()):
            return False
        if not self.wire.append_str(CRLF):
            return False
        self.pending.clear()
        self._staged_events = 0
        return True

    def flush(mut self, mut conn: Connection) -> Int:
        """Push as much as the ring will take. Returns bytes queued.

        Not an error when it returns less than is staged. That is the whole
        point of the two buffers, and `blocked` is how the caller asks.
        """
        if self.closed or self.head_only:
            return 0
        var total = 0
        while True:
            total += self._drain(conn)
            if self.wire_at < self.wire.length:
                return total
            self.wire.clear()
            self.wire_at = 0
            if conn.writable() == 0:
                return total
            if self.pending.length > 0:
                if not self._frame():
                    return total
                continue
            if self.ended and not self.done:
                if not self.wire.append_str("0\r\n\r\n"):
                    return total
                self.done = True
                continue
            return total
