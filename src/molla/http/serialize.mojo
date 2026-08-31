"""Writing a response, one buffer and one queue call.

The M0 spike in `response.mojo` builds a whole response once and memcpys it per
request, which is the right thing when there is one body and the point is to
measure the socket. A real server has a different body every time, so this is
the other half: assemble a status line and a header block into a scratch buffer
that belongs to the connection, then hand the whole thing to the ring in one
call.

One buffer per connection, reused across requests and never freed between them,
is the whole allocation story. `reset` sets the length back to zero and keeps
the capacity, so after the first few requests on a connection the writer has
grown to whatever that client's responses need and stops allocating entirely.
That is what issue #17 asserts, and it is why the writer is a struct with a
lifetime rather than a function that returns a `List`.

`Date` is refreshed at most once a second rather than formatted per response.
The formatter is Howard Hinnant's civil_from_days from `response.mojo`, integer
only with no table and no libc call behind it.

## HEAD

A HEAD response carries exactly the headers the GET would have carried,
including Content-Length, and none of the body. Getting that wrong does not
produce a wrong page, it desynchronises the connection: send a body the client
is not expecting and the next request on that connection is read out of the
middle of it. So the writer is told once, at `start`, whether this is a HEAD,
and `body` then counts the bytes without writing them. No call site has to
remember.
"""

from molla.http.response import DATE_LENGTH, format_http_date
from molla.io.buffer import Buffer
from molla.sys.clock import unix_time
from molla.sys.mem import as_ptr

comptime CRLF = "\r\n"


def status_text(status: Int) -> StaticString:
    """The reason phrase. Clients ignore it and humans reading a capture do
    not, which is the entire argument for having the table."""
    if status == 100:
        return "Continue"
    if status == 200:
        return "OK"
    if status == 201:
        return "Created"
    if status == 202:
        return "Accepted"
    if status == 204:
        return "No Content"
    if status == 206:
        return "Partial Content"
    if status == 301:
        return "Moved Permanently"
    if status == 302:
        return "Found"
    if status == 304:
        return "Not Modified"
    if status == 307:
        return "Temporary Redirect"
    if status == 308:
        return "Permanent Redirect"
    if status == 400:
        return "Bad Request"
    if status == 401:
        return "Unauthorized"
    if status == 403:
        return "Forbidden"
    if status == 404:
        return "Not Found"
    if status == 405:
        return "Method Not Allowed"
    if status == 408:
        return "Request Timeout"
    if status == 409:
        return "Conflict"
    if status == 411:
        return "Length Required"
    if status == 413:
        return "Content Too Large"
    if status == 414:
        return "URI Too Long"
    if status == 415:
        return "Unsupported Media Type"
    if status == 417:
        return "Expectation Failed"
    if status == 422:
        return "Unprocessable Content"
    if status == 429:
        return "Too Many Requests"
    if status == 431:
        return "Request Header Fields Too Large"
    if status == 499:
        return "Client Closed Request"
    if status == 500:
        return "Internal Server Error"
    if status == 501:
        return "Not Implemented"
    if status == 502:
        return "Bad Gateway"
    if status == 503:
        return "Service Unavailable"
    if status == 504:
        return "Gateway Timeout"
    if status == 505:
        return "HTTP Version Not Supported"
    if status == 507:
        return "Insufficient Storage"
    return "Unknown"


def _decimal_width(value: Int) -> Int:
    var n = value
    var width = 1
    while n >= 10:
        n //= 10
        width += 1
    return width


def write_decimal(mut out: Buffer, value: Int) -> Bool:
    """Decimal into a buffer, written in place rather than through a digit list.

    The reverse-a-list version allocates, and a response with a Content-Length
    allocating once per request is the thing #17 measures. A free function
    because the streaming writers in `stream.mojo` need the same thing into
    their own buffers, and one implementation is one place to get it wrong.
    """
    if value == 0:
        return out.append_byte(UInt8(48))
    var negative = value < 0
    var n = -value if negative else value
    var width = _decimal_width(n)
    var total = width + 1 if negative else width
    if not out.reserve(total):
        return False
    var at = out.length
    var p = out.ptr()
    if negative:
        p.unsafe_store(at, UInt8(45))
        at += 1
    var i = at + width - 1
    while n > 0:
        p.unsafe_store(i, UInt8(48 + n % 10))
        n //= 10
        i -= 1
    out.commit(total)
    return True


struct ResponseWriter(Movable):
    """A scratch buffer and the state needed to write one response into it."""

    var out: Buffer
    var head_only: Bool
    """Set at `start`. Makes `body` write the length and not the bytes."""

    var body_bytes: Int
    """What Content-Length said, whether or not the bytes were written."""

    var date: Buffer
    """The IMF-fixdate, held as bytes so it can be patched in place and
    appended without a format call."""

    var date_second: Int

    var status: Int
    """The status of the response being written, or 0 before `start`.

    Here rather than returned out of the handler, because the writer is the
    thing that actually decided and every path through the handler goes through
    `start`. It is what the metrics and the log line read."""

    def __init__(out self, counter: Int, capacity: Int = 1024):
        self.out = Buffer(capacity, counter)
        self.head_only = False
        self.body_bytes = 0
        self.status = 0
        self.date = Buffer(DATE_LENGTH, counter)
        self.date.commit(DATE_LENGTH)
        self.date_second = 0
        self._refresh_date()

    def _refresh_date(mut self):
        var now = unix_time()
        if now == self.date_second:
            return
        self.date_second = now
        format_http_date(as_ptr(self.date.base()), 0, now)

    def reset(mut self):
        """Ready for the next response, keeping the capacity."""
        self.out.clear()
        self.head_only = False
        self.body_bytes = 0
        self.status = 0

    def bytes(self) -> Span[UInt8, MutAnyOrigin]:
        return self.out.bytes()

    def length(self) -> Int:
        return self.out.length

    def start(mut self, status: Int, head_only: Bool = False) -> Bool:
        """Write the status line. Always HTTP/1.1, even in reply to a 1.0
        request, which RFC 9110 allows and which is what every server does."""
        self.reset()
        self.head_only = head_only
        self.status = status
        if not self.out.append_str("HTTP/1.1 "):
            return False
        if not self.write_int(status):
            return False
        if not self.out.append_str(" "):
            return False
        if not self.out.append_str(status_text(status)):
            return False
        return self.out.append_str(CRLF)

    def write_int(mut self, value: Int) -> Bool:
        return write_decimal(self.out, value)

    def header(mut self, name: StringSpan, value: StringSpan) -> Bool:
        if not self.out.append_str(name):
            return False
        if not self.out.append_str(": "):
            return False
        if not self.out.append_str(value):
            return False
        return self.out.append_str(CRLF)

    def header_int(mut self, name: StringSpan, value: Int) -> Bool:
        if not self.out.append_str(name):
            return False
        if not self.out.append_str(": "):
            return False
        if not self.write_int(value):
            return False
        return self.out.append_str(CRLF)

    def header_date(mut self) -> Bool:
        self._refresh_date()
        if not self.out.append_str("Date: "):
            return False
        var span = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(self.date.base()), length=DATE_LENGTH
        )
        if not self.out.append(span):
            return False
        return self.out.append_str(CRLF)

    def header_connection(mut self, keep_alive: Bool) -> Bool:
        """Sent explicitly either way.

        A 1.1 response with no Connection header means keep alive and a 1.0 one
        means close, so leaving it out is correct and unreadable. Saying it is
        two dozen bytes and removes a whole class of argument with a proxy.
        """
        if keep_alive:
            return self.header("Connection", "keep-alive")
        return self.header("Connection", "close")

    def end_headers(mut self) -> Bool:
        return self.out.append_str(CRLF)

    def body(mut self, data: Span[UInt8, _]) -> Bool:
        """Write the body, or count it and write nothing for a HEAD.

        Call after `end_headers`, and after a `Content-Length` that agrees with
        `len(data)`, which `respond` does for you.
        """
        self.body_bytes = len(data)
        if self.head_only:
            return True
        return self.out.append(data)

    def body_str(mut self, text: StringSpan) -> Bool:
        self.body_bytes = text.byte_length()
        if self.head_only:
            return True
        return self.out.append_str(text)

    def respond(
        mut self,
        status: Int,
        content_type: StringSpan,
        body: Span[UInt8, _],
        keep_alive: Bool,
        head_only: Bool = False,
    ) -> Bool:
        """The whole thing, for the ordinary case where the body is in hand."""
        if not self.start(status, head_only):
            return False
        if not self.header("Content-Type", content_type):
            return False
        if not self.header_int("Content-Length", len(body)):
            return False
        if not self.header("Server", "molla"):
            return False
        if not self.header_date():
            return False
        if not self.header_connection(keep_alive):
            return False
        if not self.end_headers():
            return False
        return self.body(body)

    def respond_str(
        mut self,
        status: Int,
        content_type: StringSpan,
        text: StringSpan,
        keep_alive: Bool,
        head_only: Bool = False,
    ) -> Bool:
        if not self.start(status, head_only):
            return False
        if not self.header("Content-Type", content_type):
            return False
        if not self.header_int("Content-Length", text.byte_length()):
            return False
        if not self.header("Server", "molla"):
            return False
        if not self.header_date():
            return False
        if not self.header_connection(keep_alive):
            return False
        if not self.end_headers():
            return False
        return self.body_str(text)

    def respond_error(mut self, status: Int, head_only: Bool = False) -> Bool:
        """An error response, which always closes.

        The body is the reason phrase and a newline, so a human running curl
        against a failing server sees what happened without needing the status
        line printed.
        """
        var text = status_text(status)
        if not self.start(status, head_only):
            return False
        if not self.header("Content-Type", "text/plain"):
            return False
        if not self.header_int("Content-Length", text.byte_length() + 1):
            return False
        if not self.header("Server", "molla"):
            return False
        if not self.header_date():
            return False
        if not self.header_connection(False):
            return False
        if not self.end_headers():
            return False
        self.body_bytes = text.byte_length() + 1
        if self.head_only:
            return True
        if not self.out.append_str(text):
            return False
        return self.out.append_byte(UInt8(10))

    def respond_continue(mut self) -> Bool:
        """`HTTP/1.1 100 Continue` and a blank line, and nothing else.

        An interim response is not a response. It does not end the message and
        the real one follows on the same connection, so this deliberately does
        not write Date, Server or Connection: adding them is legal and makes a
        capture harder to read for no gain.
        """
        self.reset()
        if not self.out.append_str("HTTP/1.1 100 Continue"):
            return False
        if not self.out.append_str(CRLF):
            return False
        return self.out.append_str(CRLF)


def chunk_header(mut out: Buffer, size: Int) -> Bool:
    """A chunked transfer size line, which is hex and lowercase.

    Here rather than in `body.mojo` because that module reads chunks and this
    one writes them, and #12's streaming writers are the caller.
    """
    if size == 0:
        return out.append_str("0\r\n")
    var digits = 0
    var n = size
    while n > 0:
        n >>= 4
        digits += 1
    if not out.reserve(digits + 2):
        return False
    var at = out.length
    var p = out.ptr()
    var i = at + digits - 1
    n = size
    while n > 0:
        var d = Int(n & 15)
        p.unsafe_store(i, UInt8(48 + d) if d < 10 else UInt8(87 + d))
        n >>= 4
        i -= 1
    p.unsafe_store(at + digits, UInt8(13))
    p.unsafe_store(at + digits + 1, UInt8(10))
    out.commit(digits + 2)
    return True
