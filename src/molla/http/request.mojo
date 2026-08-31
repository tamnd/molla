"""An HTTP/1.1 request parser that does not copy.

Every field is a `Span`, which is an offset and a length into the connection's
read buffer. Nothing here allocates, nothing here builds a `String`, and the
caller owns the bytes for exactly as long as it owns the buffer. That is the
whole design, and it is why the parser takes a raw pointer rather than something
friendlier: the moment a header value becomes a `String` we are doing a malloc
per header per request, which is the cost this spike exists to avoid.

The parser is incremental. It is handed everything received so far and answers
one of three things: this is a complete request and it used N bytes, this is a
valid prefix and I need more, or this is broken and here is the status code to
send back. A connection may hold a half arrived request across many reads, so
"not done yet" has to be an ordinary answer rather than an error.

On strictness. This rejects the request smuggling shapes rather than guessing,
because guessing is how a proxy and a server end up disagreeing about where one
request ends and the next begins. Specifically: whitespace between the header
name and the colon is a 400, a message carrying both Content-Length and
Transfer-Encoding is a 400, and a repeated Content-Length is a 400. RFC 9112
requires the first two. The third is allowed to be accepted when the values
agree, and we do not, because there is no legitimate sender that does it.

Bare LF is accepted as a line terminator even though CRLF is what the grammar
says. Real clients emit it and RFC 9112 permits recognising it.
"""

from std.memory import Pointer

comptime MAX_REQUEST_LINE = 8192
"""Longest request line we will read. Past this the answer is 414 rather than
growing the buffer, because the only senders that need more are attacks."""

comptime MAX_HEADERS = 64
comptime MAX_HEADER_BYTES = 16384

comptime PARSE_NEED_MORE = 0
comptime PARSE_DONE = 1
comptime PARSE_FAILED = 2

comptime SP: UInt8 = 32
comptime HTAB: UInt8 = 9
comptime CR: UInt8 = 13
comptime LF: UInt8 = 10
comptime COLON: UInt8 = 58


@fieldwise_init
struct Span(Copyable, ImplicitlyCopyable, Movable):
    """Where something lives in the read buffer. Not the bytes themselves."""

    var start: Int
    var length: Int

    def is_empty(self) -> Bool:
        return self.length == 0


@fieldwise_init
struct Header(Copyable, ImplicitlyCopyable, Movable):
    var name: Span
    var value: Span


def _lower(c: UInt8) -> UInt8:
    if c >= 65 and c <= 90:
        return c + 32
    return c


def _is_token[o: MutOrigin](buf: Pointer[UInt8, o], span: Span) -> Bool:
    """Whether a span is a valid RFC 9110 token.

    Used on the method and on header names. A separator or a control character
    here is what lets a crafted header name split a message downstream.
    """
    if span.length == 0:
        return False
    for i in range(span.length):
        var c = buf.unsafe_load(span.start + i)
        if c <= 32 or c >= 127:
            return False
        if (
            c == 34
            or c == 40
            or c == 41
            or c == 44
            or c == 47
            or c == 58
            or c == 59
            or c == 60
            or c == 61
            or c == 62
            or c == 63
            or c == 64
            or c == 91
            or c == 92
            or c == 93
            or c == 123
            or c == 125
        ):
            return False
    return True


def span_eq_ci[
    o: MutOrigin
](buf: Pointer[UInt8, o], span: Span, lit: StringSpan) -> Bool:
    """Compare a span to an all lowercase literal, ignoring case.

    Header names are case insensitive, so `Content-Length` and `content-length`
    have to hit the same branch. The literal side is assumed already lowercase
    so only one side needs folding.
    """
    if span.length != lit.byte_length():
        return False
    var p = lit.unsafe_ptr()
    for i in range(span.length):
        if _lower(buf.unsafe_load(span.start + i)) != p.unsafe_load(i):
            return False
    return True


def span_eq[
    o: MutOrigin
](buf: Pointer[UInt8, o], span: Span, lit: StringSpan) -> Bool:
    if span.length != lit.byte_length():
        return False
    var p = lit.unsafe_ptr()
    for i in range(span.length):
        if buf.unsafe_load(span.start + i) != p.unsafe_load(i):
            return False
    return True


struct Request(Movable):
    """One parsed request. Valid only while the buffer it points into is."""

    var method: Span
    var target: Span
    var minor_version: Int
    """0 or 1. We do not serve 0.9 and anything above 1.1 is a 505."""

    var headers: List[Header]
    var content_length: Int
    """-1 when the header is absent, which is not the same as 0."""

    var keep_alive: Bool
    var consumed: Int
    """Bytes of the buffer this request occupied, headers and body together."""

    var error_status: Int
    """Set when parsing failed. 0 otherwise."""

    def __init__(out self):
        self.method = Span(0, 0)
        self.target = Span(0, 0)
        self.minor_version = 1
        self.headers = List[Header]()
        self.content_length = -1
        self.keep_alive = True
        self.consumed = 0
        self.error_status = 0

    def reset(mut self):
        """Reuse the header list rather than freeing and reallocating it.

        A server that parses a request per connection per pass and allocates a
        list each time is measuring the allocator, not the parser.
        """
        self.headers.clear()
        self.method = Span(0, 0)
        self.target = Span(0, 0)
        self.minor_version = 1
        self.content_length = -1
        self.keep_alive = True
        self.consumed = 0
        self.error_status = 0

    def header[
        o: MutOrigin
    ](self, buf: Pointer[UInt8, o], name: StringSpan) -> Span:
        """The value of a header by lowercase name, or an empty span."""
        for i in range(len(self.headers)):
            if span_eq_ci(buf, self.headers[i].name, name):
                return self.headers[i].value
        return Span(0, 0)


def _fail(mut req: Request, status: Int) -> Int:
    req.error_status = status
    return PARSE_FAILED


def parse[
    o: MutOrigin
](buf: Pointer[UInt8, o], length: Int, mut req: Request) -> Int:
    """Parse one request from the front of `buf`.

    Returns `PARSE_DONE`, `PARSE_NEED_MORE`, or `PARSE_FAILED`. On `PARSE_DONE`
    every span in `req` is an offset from `buf` and `req.consumed` says how much
    of the buffer to drop before looking for the next request.
    """
    req.reset()

    var at = 0

    # A recipient may skip empty lines before the request line. Some clients
    # emit a stray CRLF after a body, and the alternative to skipping it is to
    # fail the next request on a connection that was working.
    while at < length:
        var c = buf.unsafe_load(at)
        if c == CR and at + 1 < length and buf.unsafe_load(at + 1) == LF:
            at += 2
        elif c == LF:
            at += 1
        else:
            break
    if at >= length:
        return PARSE_NEED_MORE

    var line_start = at

    # Method.
    var method_start = at
    while at < length and buf.unsafe_load(at) != SP:
        if at - line_start > MAX_REQUEST_LINE:
            return _fail(req, 414)
        if buf.unsafe_load(at) == CR or buf.unsafe_load(at) == LF:
            return _fail(req, 400)
        at += 1
    if at >= length:
        return PARSE_NEED_MORE
    req.method = Span(method_start, at - method_start)
    if not _is_token(buf, req.method):
        return _fail(req, 400)
    at += 1

    # Target. No parsing of it here beyond finding its end, because what a
    # target means is the router's problem and this layer should not have an
    # opinion about it.
    var target_start = at
    while at < length and buf.unsafe_load(at) != SP:
        if at - line_start > MAX_REQUEST_LINE:
            return _fail(req, 414)
        var c = buf.unsafe_load(at)
        if c == CR or c == LF:
            return _fail(req, 400)
        # Controls and raw spaces in a target are how a request line gets split.
        if c < 33 or c == 127:
            return _fail(req, 400)
        at += 1
    if at >= length:
        return PARSE_NEED_MORE
    req.target = Span(target_start, at - target_start)
    if req.target.length == 0:
        return _fail(req, 400)
    at += 1

    # Version. The shape and the value are two different failures. A line that
    # is not HTTP-version at all is malformed and gets a 400. A well formed
    # version we do not speak gets a 505, which is what RFC 9110 says and what
    # lets a client tell "you are broken" apart from "I am too new for you".
    if at + 8 > length:
        return PARSE_NEED_MORE
    if not span_eq(buf, Span(at, 5), "HTTP/"):
        return _fail(req, 400)
    var major = buf.unsafe_load(at + 5)
    var dot = buf.unsafe_load(at + 6)
    var minor = buf.unsafe_load(at + 7)
    if major < 48 or major > 57 or dot != 46 or minor < 48 or minor > 57:
        return _fail(req, 400)
    if major != 49:
        return _fail(req, 505)
    if minor == 48:
        req.minor_version = 0
    elif minor == 49:
        req.minor_version = 1
    else:
        return _fail(req, 505)
    at += 8

    # End of the request line.
    if at >= length:
        return PARSE_NEED_MORE
    if buf.unsafe_load(at) == CR:
        if at + 1 >= length:
            return PARSE_NEED_MORE
        if buf.unsafe_load(at + 1) != LF:
            return _fail(req, 400)
        at += 2
    elif buf.unsafe_load(at) == LF:
        at += 1
    else:
        return _fail(req, 400)

    # HTTP/1.0 defaults to closing, 1.1 defaults to staying open. Connection
    # can override either way and is read below.
    req.keep_alive = req.minor_version == 1

    var headers_start = at
    var seen_content_length = False
    var seen_transfer_encoding = False

    while True:
        if at >= length:
            return PARSE_NEED_MORE
        if at - headers_start > MAX_HEADER_BYTES:
            return _fail(req, 431)

        # Blank line ends the header block.
        var c = buf.unsafe_load(at)
        if c == CR:
            if at + 1 >= length:
                return PARSE_NEED_MORE
            if buf.unsafe_load(at + 1) != LF:
                return _fail(req, 400)
            at += 2
            break
        if c == LF:
            at += 1
            break

        # Obsolete line folding. A continuation line starting with whitespace
        # used to be legal and is now a 400, because a proxy that unfolds and a
        # server that does not will disagree about the value.
        if c == SP or c == HTAB:
            return _fail(req, 400)

        var name_start = at
        while at < length and buf.unsafe_load(at) != COLON:
            var n = buf.unsafe_load(at)
            if n == CR or n == LF:
                return _fail(req, 400)
            at += 1
        if at >= length:
            return PARSE_NEED_MORE
        var name = Span(name_start, at - name_start)
        # No space is allowed before the colon. This is the check that stops
        # "Content-Length : 5" being read as a header by one hop and ignored by
        # the next.
        if not _is_token(buf, name):
            return _fail(req, 400)
        at += 1

        # Optional whitespace after the colon is not part of the value.
        while at < length:
            var w = buf.unsafe_load(at)
            if w == SP or w == HTAB:
                at += 1
            else:
                break
        if at >= length:
            return PARSE_NEED_MORE

        var value_start = at
        while at < length:
            var v = buf.unsafe_load(at)
            if v == CR or v == LF:
                break
            at += 1
        if at >= length:
            return PARSE_NEED_MORE
        var value_end = at

        # Trailing whitespace is not part of the value either.
        while value_end > value_start:
            var t = buf.unsafe_load(value_end - 1)
            if t == SP or t == HTAB:
                value_end -= 1
            else:
                break

        if buf.unsafe_load(at) == CR:
            if at + 1 >= length:
                return PARSE_NEED_MORE
            if buf.unsafe_load(at + 1) != LF:
                return _fail(req, 400)
            at += 2
        else:
            at += 1

        if len(req.headers) >= MAX_HEADERS:
            return _fail(req, 431)

        var value = Span(value_start, value_end - value_start)
        req.headers.append(Header(name, value))

        if span_eq_ci(buf, name, "content-length"):
            if seen_content_length:
                return _fail(req, 400)
            seen_content_length = True
            var n = 0
            if value.length == 0:
                return _fail(req, 400)
            for i in range(value.length):
                var d = buf.unsafe_load(value.start + i)
                if d < 48 or d > 57:
                    return _fail(req, 400)
                n = n * 10 + Int(d - 48)
                if n > 1 << 40:
                    return _fail(req, 413)
            req.content_length = n
        elif span_eq_ci(buf, name, "transfer-encoding"):
            seen_transfer_encoding = True
        elif span_eq_ci(buf, name, "connection"):
            if span_eq_ci(buf, value, "close"):
                req.keep_alive = False
            elif span_eq_ci(buf, value, "keep-alive"):
                req.keep_alive = True

    # Both framing headers on one message is the classic smuggling setup. The
    # spec says to treat Transfer-Encoding as authoritative and close, and we
    # reject outright, which is stricter and cheaper to reason about.
    if seen_content_length and seen_transfer_encoding:
        return _fail(req, 400)

    if seen_transfer_encoding:
        # Chunked decoding is M1 work. Saying 501 is honest, and it is very
        # much better than reading the chunk header as a body.
        return _fail(req, 501)

    var body = 0
    if req.content_length > 0:
        body = req.content_length
        if at + body > length:
            return PARSE_NEED_MORE

    req.consumed = at + body
    return PARSE_DONE
