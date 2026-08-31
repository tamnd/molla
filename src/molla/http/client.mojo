"""A small HTTPS client, enough to talk to a registry and nothing more.

This is not a general purpose HTTP client and should not grow into one. It does
GET, it follows redirects, it understands `Content-Length` and chunked bodies,
and it stops there. No connection reuse, no pipelining, no compression, no
cookies. Every request opens a connection, sends `Connection: close`, and reads
until the server hangs up. For pulling a handful of blobs that is the right
trade, and it means the body framing has two cases instead of six.

Bodies are read into memory whole. A model file will not fit that way and M3
will need a streaming version, but the M0 question is whether TLS works at all,
and answering it with a `List[UInt8]` keeps the parts that could fail down to
the ones under test.

The parser here is deliberately sloppier than the one in `request.mojo`. That
one faces the network and has to reject malformed input from strangers. This one
talks to servers we chose, so it accepts what they send and raises on anything
it cannot make sense of.
"""

from molla.build_info import VERSION
from molla.tls.client import TlsClient
from molla.tls.policy import TlsPolicy

comptime CHUNK_SIZE = 16384
"""How much to ask TLS for at a time. Two TLS records, roughly."""

comptime MAX_LINE = 8192
"""Longest status line or header line accepted. ghcr redirect URLs run past
1500 bytes, so the usual 4096 is not enough headroom."""

comptime MAX_HEADERS = 100

comptime MAX_BODY = 512 * 1024 * 1024
"""A ceiling so a bad `Content-Length` cannot ask us to allocate the world."""

comptime MAX_REDIRECTS = 5

comptime DEFAULT_HTTPS_PORT: UInt16 = 443


struct Url(Copyable, ImplicitlyCopyable, Movable):
    """A parsed https URL. Scheme is implied because nothing else is allowed."""

    var host: String
    var port: UInt16
    var path: String
    """Path and query together, ready to go on the request line."""

    def __init__(out self, host: String, port: UInt16, path: String):
        self.host = host
        self.port = port
        self.path = path

    def text(self) -> String:
        var out = String("https://") + self.host
        if self.port != DEFAULT_HTTPS_PORT:
            out += ":" + String(self.port)
        return out + self.path


def parse_url(text: String) raises -> Url:
    """Split an absolute https URL.

    Plain http is rejected rather than supported. molla pulls signed artifacts
    over the network and there is no case where falling back to cleartext is
    the helpful thing to do.
    """
    if not text.startswith("https://"):
        if text.startswith("http://"):
            raise Error("refusing to fetch over plain http: " + text)
        raise Error("not an absolute https URL: " + text)

    var rest = text[byte=8:]
    var slash = rest.find("/")
    var authority: String
    var path: String
    if slash < 0:
        authority = String(rest)
        path = "/"
    else:
        authority = String(rest[byte=0:slash])
        path = String(rest[byte=slash:])

    var port = DEFAULT_HTTPS_PORT
    var colon = authority.find(":")
    if colon >= 0:
        var digits = String(authority[byte = colon + 1 :])
        var bare = String(authority[byte=0:colon])
        authority = bare
        try:
            port = UInt16(Int(digits))
        except:
            raise Error("bad port in URL: " + text)

    if authority.byte_length() == 0:
        raise Error("no host in URL: " + text)
    return Url(authority, port, path)


struct Header(Copyable, ImplicitlyCopyable, Movable):
    var name: String
    """Lowercased on the way in, so lookups do not have to case fold."""

    var value: String

    def __init__(out self, name: String, value: String):
        self.name = name
        self.value = value


struct Response(Movable):
    var status: Int
    var headers: List[Header]
    var body: List[UInt8]

    def __init__(out self):
        self.status = 0
        self.headers = List[Header]()
        self.body = List[UInt8]()

    def header(self, name: String) -> String:
        """The first value for `name`, or an empty string. `name` must already
        be lowercase."""
        for i in range(len(self.headers)):
            if self.headers[i].name == name:
                return self.headers[i].value
        return String("")

    def take_body(deinit self) -> List[UInt8]:
        """Hand the body out and drop the rest.

        Moving a field out of a live struct is not allowed, and a blob is too
        big to copy just to get it past a return statement, so the whole
        response is consumed instead.
        """
        return self.body^

    def body_text(self) -> String:
        return String(StringSpan(unsafe_from_utf8=self.body))


def _lower(text: StringSpan) -> String:
    var out = List[UInt8]()
    var p = text.unsafe_ptr()
    for i in range(text.byte_length()):
        var c = p.unsafe_load(i)
        if c >= 65 and c <= 90:
            c += 32
        out.append(c)
    return String(StringSpan(unsafe_from_utf8=out))


def _trim(text: StringSpan) -> String:
    var p = text.unsafe_ptr()
    var start = 0
    var stop = text.byte_length()
    while start < stop:
        var c = p.unsafe_load(start)
        if c != 32 and c != 9:
            break
        start += 1
    while stop > start:
        var c = p.unsafe_load(stop - 1)
        if c != 32 and c != 9:
            break
        stop -= 1
    return String(text[byte=start:stop])


struct Reader(Movable):
    """Buffered reads over one TLS connection.

    TLS hands back whatever fits in a record, so a header line arrives in
    pieces and a body arrives with the tail of the last header still in the
    buffer. Everything above needs a byte stream, so there is a buffer and a
    cursor, and the buffer is compacted rather than reallocated when the cursor
    runs ahead of it.
    """

    var conn: TlsClient
    var buf: List[UInt8]
    var pos: Int
    var scratch: List[UInt8]
    var at_eof: Bool

    def __init__(out self, var conn: TlsClient):
        self.conn = conn^
        self.buf = List[UInt8]()
        self.pos = 0
        self.scratch = List[UInt8]()
        self.scratch.resize(CHUNK_SIZE, 0)
        self.at_eof = False

    def close(mut self):
        self.conn.close()

    def _fill(mut self) raises -> Int:
        """Pull one more chunk in. Returns the number of bytes added, zero at
        end of stream."""
        if self.at_eof:
            return 0

        # Drop what has already been handed out before growing the buffer,
        # otherwise a large body keeps every header byte alive under it.
        if self.pos > 0:
            var kept = List[UInt8]()
            for i in range(self.pos, len(self.buf)):
                kept.append(self.buf[i])
            self.buf = kept^
            self.pos = 0

        var n = self.conn.read(self.scratch.unsafe_ptr(), CHUNK_SIZE)
        if n <= 0:
            self.at_eof = True
            return 0
        for i in range(n):
            self.buf.append(self.scratch[i])
        return n

    def read_line(mut self) raises -> String:
        """One CRLF terminated line, without the terminator.

        A bare LF is accepted too. Servers that send one are out of spec, but
        rejecting the response is a worse outcome than parsing it.
        """
        # Counted from the cursor rather than from the start of the buffer,
        # because _fill compacts and every absolute index would move under us.
        var scanned = 0
        while True:
            var at = self.pos + scanned
            while at < len(self.buf):
                if self.buf[at] == 10:
                    var stop = at
                    if stop > self.pos and self.buf[stop - 1] == 13:
                        stop -= 1
                    var out = List[UInt8]()
                    for i in range(self.pos, stop):
                        out.append(self.buf[i])
                    self.pos = at + 1
                    return String(StringSpan(unsafe_from_utf8=out))
                at += 1
            scanned = at - self.pos
            if scanned > MAX_LINE:
                raise Error("header line longer than " + String(MAX_LINE))
            if self._fill() == 0:
                raise Error("connection closed mid header")

    def read_exact(mut self, count: Int, mut into: List[UInt8]) raises:
        var want = count
        while want > 0:
            if self.pos == len(self.buf):
                if self._fill() == 0:
                    raise Error(
                        "connection closed with "
                        + String(want)
                        + " bytes still expected"
                    )
            var take = len(self.buf) - self.pos
            if take > want:
                take = want
            for i in range(self.pos, self.pos + take):
                into.append(self.buf[i])
            self.pos += take
            want -= take

    def read_to_eof(mut self, mut into: List[UInt8]) raises:
        while True:
            for i in range(self.pos, len(self.buf)):
                into.append(self.buf[i])
            self.pos = len(self.buf)
            if len(into) > MAX_BODY:
                raise Error("response body over " + String(MAX_BODY) + " bytes")
            if self._fill() == 0:
                return


def _hex_value(c: UInt8) raises -> Int:
    if c >= 48 and c <= 57:
        return Int(c) - 48
    if c >= 97 and c <= 102:
        return Int(c) - 87
    if c >= 65 and c <= 70:
        return Int(c) - 55
    raise Error("not a hex digit")


def _read_chunked(mut reader: Reader, mut into: List[UInt8]) raises:
    """Transfer-Encoding: chunked, without the trailer parsing.

    Trailers are read and dropped. Nothing molla asks for sends one, and a
    trailer that mattered would have to be signed to be worth acting on.
    """
    while True:
        var line = reader.read_line()
        var semi = line.find(";")
        if semi >= 0:
            var head = String(line[byte=0:semi])
            line = head
        line = _trim(line)
        if line.byte_length() == 0:
            raise Error("empty chunk size line")

        var size = 0
        var p = line.unsafe_ptr()
        for i in range(line.byte_length()):
            size = size * 16 + _hex_value(p.unsafe_load(i))
        if size == 0:
            # Trailer section, ends at the first blank line.
            while True:
                var trailer = reader.read_line()
                if trailer.byte_length() == 0:
                    return

        if len(into) + size > MAX_BODY:
            raise Error("chunked body over " + String(MAX_BODY) + " bytes")
        reader.read_exact(size, into)
        var terminator = reader.read_line()
        if terminator.byte_length() != 0:
            raise Error("chunk not followed by CRLF")


def _build_request(url: Url, accept: String, authorization: String) -> String:
    var out = String("GET ") + url.path + " HTTP/1.1\r\n"
    out += "Host: " + url.host
    if url.port != DEFAULT_HTTPS_PORT:
        out += ":" + String(url.port)
    out += "\r\n"
    out += "User-Agent: molla/" + VERSION + "\r\n"
    if accept.byte_length() > 0:
        out += "Accept: " + accept + "\r\n"
    if authorization.byte_length() > 0:
        out += "Authorization: " + authorization + "\r\n"
    # No keep alive. See the module docstring.
    out += "Connection: close\r\n\r\n"
    return out


def get_once(
    url: Url,
    accept: String,
    authorization: String,
    policy: TlsPolicy = TlsPolicy(),
) raises -> Response:
    """One request, one connection, no redirect following."""
    var conn = TlsClient(url.host, url.port, policy)
    var reader = Reader(conn^)

    var request = _build_request(url, accept, authorization)
    reader.conn.write_all(request.as_bytes())

    var response = Response()

    var status_line = reader.read_line()
    if not status_line.startswith("HTTP/1."):
        reader.close()
        raise Error("not an HTTP/1.x response: " + status_line)
    var first = status_line.find(" ")
    if first < 0:
        reader.close()
        raise Error("no status code in: " + status_line)
    var after = String(status_line[byte = first + 1 :])
    var second = after.find(" ")
    var code = after if second < 0 else String(after[byte=0:second])
    try:
        response.status = Int(code)
    except:
        reader.close()
        raise Error("bad status code in: " + status_line)

    while True:
        var line = reader.read_line()
        if line.byte_length() == 0:
            break
        if len(response.headers) >= MAX_HEADERS:
            reader.close()
            raise Error("more than " + String(MAX_HEADERS) + " headers")
        var colon = line.find(":")
        if colon < 0:
            # Folded continuation lines are obsolete and nothing we talk to
            # sends them, so a line without a colon means we lost sync.
            reader.close()
            raise Error("header without a colon: " + line)
        response.headers.append(
            Header(_lower(line[byte=0:colon]), _trim(line[byte = colon + 1 :]))
        )

    var encoding = _lower(response.header("transfer-encoding"))
    var length = response.header("content-length")
    try:
        if encoding.find("chunked") >= 0:
            _read_chunked(reader, response.body)
        elif length.byte_length() > 0:
            var want = Int(length)
            if want < 0 or want > MAX_BODY:
                raise Error("implausible Content-Length: " + length)
            reader.read_exact(want, response.body)
        elif response.status == 204 or response.status == 304:
            pass
        else:
            reader.read_to_eof(response.body)
    except e:
        reader.close()
        raise e

    reader.close()
    return response^


def _is_redirect(status: Int) -> Bool:
    return (
        status == 301
        or status == 302
        or status == 303
        or status == 307
        or status == 308
    )


def get(
    url_text: String,
    accept: String = String(""),
    authorization: String = String(""),
    policy: TlsPolicy = TlsPolicy(),
) raises -> Response:
    """GET a URL, following redirects.

    The `Authorization` header is dropped when a redirect crosses to another
    host. Registries rely on this: ghcr.io answers a blob request with a 307 to
    a signed URL on githubusercontent.com, and sending the registry bearer token
    to a different host would be handing a credential to whoever the redirect
    named.

    The policy is carried across the redirect unchanged, which sounds like the
    dangerous choice and is the safe one, because it names hosts rather than
    holding a switch. Turning verification off for a registry leaves it on for
    the CDN the registry redirected to, and that host is chosen by the response.
    """
    var url = parse_url(url_text)
    var auth = authorization
    var hops = 0

    while True:
        var response = get_once(url, accept, auth, policy)
        if not _is_redirect(response.status):
            return response^

        var location = response.header("location")
        if location.byte_length() == 0:
            raise Error(
                String(response.status) + " redirect with no Location header"
            )

        hops += 1
        if hops > MAX_REDIRECTS:
            raise Error("more than " + String(MAX_REDIRECTS) + " redirects")

        var next: Url
        if location.startswith("https://") or location.startswith("http://"):
            next = parse_url(location)
        elif location.startswith("/"):
            next = Url(url.host, url.port, location)
        else:
            raise Error("cannot follow relative redirect: " + location)

        if next.host != url.host:
            auth = String("")
        url = next
