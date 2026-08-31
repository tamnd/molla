"""An HTTP/1.1 request parser that does not copy, and does not guess.

Every field is a `Span`, which is an offset and a length into the connection's
read buffer. Nothing here allocates, nothing here builds a `String`, and the
caller owns the bytes for exactly as long as it owns the buffer. That is the
whole design, and it is why the parser takes a raw pointer rather than
something friendlier: the moment a header value becomes a `String` we are doing
a malloc per header per request.

The parser is incremental. It is handed everything received so far and answers
one of three things: the header block is complete and it ended at byte N, this
is a valid prefix and I need more, or this is broken and here is the status to
send back. A connection may hold a half arrived request across many reads, so
"not done yet" has to be an ordinary answer rather than an error.

`parse` stops at the end of the header block and does not wait for a body. It
decides how the body is framed and records that in `body_kind`, and reading it
is `BodyReader`'s job in `body.mojo`. That split is what lets a body larger
than memory exist at all: a parser that only returns a complete message can
never see one.

## On strictness

Every ambiguity in HTTP/1.1 framing is a place where two hops can disagree
about where one message ends and the next begins, and every one of those is a
request smuggling bug. So this parser rejects rather than resolves. In
particular:

A bare LF is not a line terminator. RFC 9112 says a recipient MAY recognise one
and this parser does not, which is stricter than the spec allows for. The
reason is that a bare LF is the single most productive smuggling primitive
there is: a front end that treats CRLF as the only terminator and a back end
that also accepts LF will read different messages out of the same bytes. Every
real client sends CRLF. The ones that do not are almost all attacks, and the
few that are not get a clear 400 rather than a silently different reading.

Whitespace between a header name and its colon is a 400, which RFC 9112
requires. Obsolete line folding is a 400, because a hop that unfolds and a hop
that does not will disagree about the value.

Content-Length and Transfer-Encoding on one message is a 400. The spec says to
treat Transfer-Encoding as authoritative and close the connection, and we
reject outright, which is stricter and much easier to reason about. A repeated
Content-Length is a 400 even when the values agree, because there is no
legitimate sender that does it.

Transfer-Encoding must be exactly `chunked`. Any other coding, or chunked with
anything after it, is a 501 rather than an attempt to make sense of it.

An HTTP/1.1 request with no Host, or with two, is a 400. That is what routes a
request, and a message that does not say where it is going unambiguously is not
one we should be guessing about.

More than one space between the method, the target and the version is a 400.
Some parsers skip runs of whitespace there, and a target that starts with a
space then means one thing to them and another to us.
"""

from std.memory import Pointer

from molla.http.scan import find_byte, find_ctl

comptime MAX_REQUEST_LINE = 8192
"""Longest request line we will read, per issue #11. Past this the answer is
414 rather than growing the buffer, because the only senders that need more are
attacks."""

comptime MAX_HEADERS = 128
"""Most header fields on one message. Past this is 431."""

comptime MAX_HEADER_BYTES = 65536
"""Largest header block. Past this is 431. This is the number that stops a
sender opening a connection and streaming header bytes forever without ever
sending the blank line."""

comptime MAX_CONTENT_LENGTH = 1 << 40
"""A terabyte. Not a real limit on what can be served, since bodies spill to
disk, but a bound on what an integer parsed out of a header may be so that
arithmetic on it downstream cannot wrap."""

comptime PARSE_NEED_MORE = 0
comptime PARSE_DONE = 1
comptime PARSE_FAILED = 2

comptime BODY_NONE = 0
"""No Content-Length and no Transfer-Encoding, so there is no body. Not the
same as a body of length zero, though for a request the two behave alike."""

comptime BODY_LENGTH = 1
"""Content-Length bytes follow the header block."""

comptime BODY_CHUNKED = 2
"""Transfer-Encoding: chunked. The length is not known in advance."""

comptime SP: UInt8 = 32
comptime HTAB: UInt8 = 9
comptime CR: UInt8 = 13
comptime LF: UInt8 = 10
comptime COLON: UInt8 = 58
comptime COMMA: UInt8 = 44


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

    Used on the method and on every header name. A separator or a control
    character here is what lets a crafted header name split a message
    downstream.
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


def _trim[o: MutOrigin](buf: Pointer[UInt8, o], span: Span) -> Span:
    """Drop leading and trailing spaces and tabs. OWS is not part of a value."""
    var start = span.start
    var end = span.start + span.length
    while start < end:
        var c = buf.unsafe_load(start)
        if c == SP or c == HTAB:
            start += 1
        else:
            break
    while end > start:
        var c = buf.unsafe_load(end - 1)
        if c == SP or c == HTAB:
            end -= 1
        else:
            break
    return Span(start, end - start)


def list_has_token[
    o: MutOrigin
](buf: Pointer[UInt8, o], value: Span, want: StringSpan) -> Bool:
    """Whether a comma separated header value contains a given token.

    `Connection: keep-alive, Upgrade` has to answer yes to both, and a parser
    that compares the whole value against "close" gets `Connection: close,
    TE` wrong. The comparison is case insensitive and each element is trimmed,
    which is what the grammar says.
    """
    var at = value.start
    var end = value.start + value.length
    while at <= end:
        var comma = find_byte(buf, at, end, COMMA)
        var stop = end if comma < 0 else comma
        var element = _trim(buf, Span(at, stop - at))
        if span_eq_ci(buf, element, want):
            return True
        if comma < 0:
            return False
        at = comma + 1
    return False


struct Request(Movable):
    """One parsed request. Valid only while the buffer it points into is."""

    var method: Span
    var target: Span
    var minor_version: Int
    """0 or 1. We do not serve 0.9 and anything above 1.1 is a 505."""

    var headers: List[Header]

    var body_kind: Int
    """`BODY_NONE`, `BODY_LENGTH` or `BODY_CHUNKED`."""

    var content_length: Int
    """-1 unless `body_kind` is `BODY_LENGTH`."""

    var keep_alive: Bool
    var expect_continue: Bool
    """The client sent `Expect: 100-continue` and is waiting to be told to send
    the body. Answering is the protocol's job, not the parser's."""

    var is_head: Bool
    """HEAD needs the same headers as GET and none of the body, and getting
    that wrong desynchronises a keep alive connection, so it is decided once
    here rather than at every place that writes a response."""

    var header_bytes: Int
    """Bytes from the front of the buffer through the blank line. Where the
    body starts, and what to drop when there is no body."""

    var error_status: Int
    """Set when parsing failed. 0 otherwise."""

    def __init__(out self):
        self.method = Span(0, 0)
        self.target = Span(0, 0)
        self.minor_version = 1
        self.headers = List[Header]()
        self.body_kind = BODY_NONE
        self.content_length = -1
        self.keep_alive = True
        self.expect_continue = False
        self.is_head = False
        self.header_bytes = 0
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
        self.body_kind = BODY_NONE
        self.content_length = -1
        self.keep_alive = True
        self.expect_continue = False
        self.is_head = False
        self.header_bytes = 0
        self.error_status = 0

    def header[
        o: MutOrigin
    ](self, buf: Pointer[UInt8, o], name: StringSpan) -> Span:
        """The value of a header by lowercase name, or an empty span."""
        for i in range(len(self.headers)):
            if span_eq_ci(buf, self.headers[i].name, name):
                return self.headers[i].value
        return Span(0, 0)

    def has_body(self) -> Bool:
        return self.body_kind != BODY_NONE and self.content_length != 0


def _fail(mut req: Request, status: Int) -> Int:
    req.error_status = status
    return PARSE_FAILED


def parse[
    o: MutOrigin
](buf: Pointer[UInt8, o], length: Int, mut req: Request) -> Int:
    """Parse the header block of one request from the front of `buf`.

    Returns `PARSE_DONE`, `PARSE_NEED_MORE`, or `PARSE_FAILED`. On `PARSE_DONE`
    every span in `req` is an offset from `buf`, `req.header_bytes` is where
    the body begins, and `req.body_kind` says how to read it.
    """
    req.reset()

    var at = 0

    # RFC 9112 says a server should ignore at least one empty line received
    # before the request line, because some old clients emit a stray CRLF after
    # a body. Exactly one, and it has to be a real CRLF: allowing a run of them
    # gives a sender a way to pad a message until a length aware hop in front
    # of us has stopped counting.
    if at + 1 < length and buf.unsafe_load(at) == CR:
        if buf.unsafe_load(at + 1) == LF:
            at += 2
    if at >= length:
        return PARSE_NEED_MORE

    var line_start = at

    # The request line ends at the first LF. Finding it first bounds every
    # scan after it, so a malformed line cannot run off into the body.
    var line_limit = min(length, line_start + MAX_REQUEST_LINE + 2)
    var lf = find_byte(buf, at, line_limit, LF)
    if lf < 0:
        if length - line_start > MAX_REQUEST_LINE:
            return _fail(req, 414)
        return PARSE_NEED_MORE
    if lf - line_start > MAX_REQUEST_LINE:
        return _fail(req, 414)
    if lf == line_start or buf.unsafe_load(lf - 1) != CR:
        return _fail(req, 400)
    var line_end = lf - 1

    # Method.
    var sp1 = find_byte(buf, line_start, line_end, SP)
    if sp1 < 0:
        return _fail(req, 400)
    req.method = Span(line_start, sp1 - line_start)
    if not _is_token(buf, req.method):
        return _fail(req, 400)
    req.is_head = span_eq(buf, req.method, "HEAD")

    # Target. What a target means is the router's problem and this layer has no
    # opinion about it beyond where it ends and that it holds no byte that
    # could split the line.
    var target_start = sp1 + 1
    var sp2 = find_byte(buf, target_start, line_end, SP)
    if sp2 < 0:
        return _fail(req, 400)
    req.target = Span(target_start, sp2 - target_start)
    if req.target.length == 0:
        return _fail(req, 400)
    for i in range(req.target.length):
        var c = buf.unsafe_load(req.target.start + i)
        if c < 33 or c == 127:
            return _fail(req, 400)

    # Version. The shape and the value are two different failures. A line that
    # is not HTTP-version at all is malformed and gets a 400. A well formed
    # version we do not speak gets a 505, which is what RFC 9110 says and what
    # lets a client tell "you are broken" apart from "I am too new for you".
    var version_start = sp2 + 1
    if line_end - version_start != 8:
        return _fail(req, 400)
    if not span_eq(buf, Span(version_start, 5), "HTTP/"):
        return _fail(req, 400)
    var major = buf.unsafe_load(version_start + 5)
    var dot = buf.unsafe_load(version_start + 6)
    var minor = buf.unsafe_load(version_start + 7)
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

    at = lf + 1

    # HTTP/1.0 defaults to closing, 1.1 defaults to staying open. Connection
    # can override either way and is read below.
    req.keep_alive = req.minor_version == 1

    var headers_start = at
    var seen_content_length = False
    var seen_transfer_encoding = False
    var seen_host = 0

    while True:
        if at - headers_start > MAX_HEADER_BYTES:
            return _fail(req, 431)
        if at >= length:
            return PARSE_NEED_MORE

        var field_limit = min(length, headers_start + MAX_HEADER_BYTES + 2)
        var end = find_byte(buf, at, field_limit, LF)
        if end < 0:
            if length - headers_start > MAX_HEADER_BYTES:
                return _fail(req, 431)
            return PARSE_NEED_MORE
        if end == at or buf.unsafe_load(end - 1) != CR:
            return _fail(req, 400)
        var field_end = end - 1

        # The blank line ends the header block.
        if field_end == at:
            at = end + 1
            break

        # Obsolete line folding. A continuation line starting with whitespace
        # used to be legal and is now a 400.
        var first = buf.unsafe_load(at)
        if first == SP or first == HTAB:
            return _fail(req, 400)

        # Nothing between here and the terminator may be a control character.
        # One scan over the whole field catches an embedded NUL, a lone CR and
        # a DEL in either the name or the value, which is the check that stops
        # a crafted header from being read as two by a less careful hop.
        if find_ctl(buf, at, field_end) >= 0:
            return _fail(req, 400)

        var colon = find_byte(buf, at, field_end, COLON)
        if colon < 0:
            return _fail(req, 400)
        var name = Span(at, colon - at)
        # `_is_token` rejects a space before the colon, which is the check that
        # stops "Content-Length : 5" being a header to one hop and not to the
        # next.
        if not _is_token(buf, name):
            return _fail(req, 400)

        var value = _trim(buf, Span(colon + 1, field_end - colon - 1))

        if len(req.headers) >= MAX_HEADERS:
            return _fail(req, 431)
        req.headers.append(Header(name, value))

        if span_eq_ci(buf, name, "content-length"):
            if seen_content_length:
                return _fail(req, 400)
            seen_content_length = True
            if value.length == 0:
                return _fail(req, 400)
            var n = 0
            for i in range(value.length):
                var d = buf.unsafe_load(value.start + i)
                if d < 48 or d > 57:
                    return _fail(req, 400)
                n = n * 10 + Int(d - 48)
                if n > MAX_CONTENT_LENGTH:
                    return _fail(req, 413)
            req.content_length = n
        elif span_eq_ci(buf, name, "transfer-encoding"):
            if seen_transfer_encoding:
                return _fail(req, 400)
            seen_transfer_encoding = True
            # Only chunked, and only on its own. `gzip, chunked` is legal in
            # the grammar and we do not implement it, and saying so is much
            # better than reading a compressed stream as a chunk header.
            if not span_eq_ci(buf, value, "chunked"):
                return _fail(req, 501)
        elif span_eq_ci(buf, name, "connection"):
            if list_has_token(buf, value, "close"):
                req.keep_alive = False
            elif list_has_token(buf, value, "keep-alive"):
                req.keep_alive = True
        elif span_eq_ci(buf, name, "host"):
            seen_host += 1
        elif span_eq_ci(buf, name, "expect"):
            if span_eq_ci(buf, value, "100-continue"):
                req.expect_continue = True
            else:
                # An expectation we do not understand has to be refused rather
                # than ignored, or the client waits for a response that is not
                # coming.
                return _fail(req, 417)

        at = end + 1

    # Both framing headers on one message is the classic smuggling setup.
    if seen_content_length and seen_transfer_encoding:
        return _fail(req, 400)

    # HTTP/1.1 requires exactly one Host. Zero is what a smuggled request looks
    # like once a front end has stripped the outer one, and two is a request
    # that two hops will route differently.
    if req.minor_version == 1 and seen_host != 1:
        return _fail(req, 400)
    if seen_host > 1:
        return _fail(req, 400)

    if seen_transfer_encoding:
        req.body_kind = BODY_CHUNKED
    elif seen_content_length:
        req.body_kind = BODY_LENGTH
    else:
        req.body_kind = BODY_NONE
        req.content_length = -1

    req.header_bytes = at
    return PARSE_DONE
