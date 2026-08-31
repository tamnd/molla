"""Tests for the HTTP/1.1 parser, the body reader, the writer and the protocol.

The parser tests run against a byte buffer directly, because the parser's whole
contract is about offsets into a buffer and going through a socket to check that
would only add ways to be flaky.

The server tests use real loopback sockets and drive the client by hand between
calls to `poll_once`, same as `test_net.mojo`, for the same reason: there is no
second thread to put a client on.

Most of these are the cases that are easy to get wrong and hard to notice.
Requests split across two reads, two requests in one read, and the smuggling
shapes. A parser that only ever sees whole requests arriving one at a time looks
correct right up until it meets a real client.

The hostile corpus at the bottom of the parser section is the one worth keeping
honest. Every entry in it is a real technique out of a request smuggling writeup
rather than a fuzzer's idea of a bad string, and the assertion is not just that
each is refused but that the refusal is the same one every time. A parser that
accepts one of these is not slightly wrong, it is a way for one client to put a
request in another client's connection.
"""

from std.memory import stack_allocation

from harness import Suite

from molla.http.body import (
    BODY_DONE,
    BODY_FAILED,
    BODY_NEED_MORE,
    BodyReader,
)
from molla.http.multipart import (
    MULTIPART_DONE,
    MULTIPART_FAILED,
    MULTIPART_NEED_MORE,
    MultipartReader,
    boundary_of,
)
from molla.http.protocol import HttpProtocol
from molla.http.request import (
    BODY_CHUNKED,
    BODY_LENGTH,
    BODY_NONE,
    PARSE_DONE,
    PARSE_FAILED,
    PARSE_NEED_MORE,
    Request,
    Span,
    parse,
    span_eq,
)
from molla.http.response import DATE_LENGTH, Responder, format_http_date
from molla.http.scan import count_byte, find_byte, find_crlf, find_ctl
from molla.http.serialize import ResponseWriter, chunk_header, status_text
from molla.http.server import HttpServer
from molla.io.buffer import Buffer
from molla.net.listener import ListenAddress, bound_port, open_listener
from molla.net.reactor import Reactor
from molla.sys.errno import EAGAIN, get_errno
from molla.sys.fd import close
from molla.sys.file import FileInfo, exists, stat_path
from molla.sys.mem import AllocCounter
from molla.sys.socket import INADDR_LOOPBACK, connect, recv, send, socket_tcp

comptime MAX_STEPS = 200
comptime SCRATCH = 65536


def _load[o: MutOrigin](buf: Pointer[UInt8, o], text: StringSpan) -> Int:
    """Copy a literal into a buffer and return its length."""
    var p = text.unsafe_ptr()
    var n = text.byte_length()
    for i in range(n):
        buf.unsafe_store(i, p.unsafe_load(i))
    return n


def _parse_text[
    o: MutOrigin
](buf: Pointer[UInt8, o], text: StringSpan, mut req: Request) -> Int:
    var n = _load(buf, text)
    return parse(buf, n, req)


def test_parse_request_line(mut suite: Suite) raises:
    suite.group("request line")

    var buf = stack_allocation[SCRATCH, UInt8]()
    var req = Request()

    var rc = _parse_text(buf, "GET /hello HTTP/1.1\r\nHost: x\r\n\r\n", req)
    suite.check(rc == PARSE_DONE, "a plain GET parses")
    suite.check(span_eq(buf, req.method, "GET"), "the method is GET")
    suite.check(span_eq(buf, req.target, "/hello"), "the target is /hello")
    suite.check(req.minor_version == 1, "the version is 1.1")
    suite.check(req.keep_alive, "1.1 keeps the connection open by default")
    suite.check(len(req.headers) == 1, "one header was seen")
    suite.check(req.content_length == -1, "no body means no content length")
    suite.check(req.body_kind == BODY_NONE, "and no framing")
    suite.check(
        req.header_bytes == 32, "the header block ends at the blank line"
    )

    rc = _parse_text(buf, "GET / HTTP/1.0\r\n\r\n", req)
    suite.check(rc == PARSE_DONE, "HTTP/1.0 parses without a host")
    suite.check(not req.keep_alive, "1.0 closes by default")

    rc = _parse_text(
        buf, "GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n", req
    )
    suite.check(rc == PARSE_DONE, "1.0 with keep-alive parses")
    suite.check(req.keep_alive, "and it overrides the 1.0 default")

    rc = _parse_text(
        buf, "GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", req
    )
    suite.check(rc == PARSE_DONE, "1.1 with close parses")
    suite.check(not req.keep_alive, "and it overrides the 1.1 default")

    # Connection is a list, and keep-alive next to something else still means
    # keep-alive. A parser that compares the whole value closes here.
    rc = _parse_text(
        buf,
        "GET / HTTP/1.1\r\nHost: x\r\nConnection: keep-alive, Upgrade\r\n\r\n",
        req,
    )
    suite.check(rc == PARSE_DONE, "a connection list parses")
    suite.check(req.keep_alive, "and keep-alive is found inside it")

    rc = _parse_text(
        buf,
        "GET / HTTP/1.1\r\nHost: x\r\nConnection: Upgrade, close\r\n\r\n",
        req,
    )
    suite.check(not req.keep_alive, "close is found inside a list too")

    rc = _parse_text(buf, "HEAD / HTTP/1.1\r\nHost: x\r\n\r\n", req)
    suite.check(rc == PARSE_DONE and req.is_head, "HEAD is recognised once")

    # RFC 9112 says a server should ignore an empty line before the request
    # line, because old clients emit one after a POST body.
    rc = _parse_text(buf, "\r\nGET / HTTP/1.1\r\nHost: x\r\n\r\n", req)
    suite.check(rc == PARSE_DONE, "one leading blank line is skipped")


def test_parse_incremental(mut suite: Suite) raises:
    suite.group("partial requests")

    var buf = stack_allocation[SCRATCH, UInt8]()
    var req = Request()
    var whole = "GET /split HTTP/1.1\r\nHost: x\r\nAccept: */*\r\n\r\n"
    var total = _load(buf, whole)

    # Every prefix short of the whole thing has to say "need more" rather than
    # failing or, worse, succeeding on half a request.
    var all_need_more = True
    var wrong_at = -1
    for n in range(1, total):
        if parse(buf, n, req) != PARSE_NEED_MORE:
            all_need_more = False
            if wrong_at < 0:
                wrong_at = n
    suite.check(
        all_need_more,
        "every prefix of a request is incomplete, not broken"
        + (
            " (first bad length " + String(wrong_at) + ")" if wrong_at
            >= 0 else ""
        ),
    )
    suite.check(
        parse(buf, total, req) == PARSE_DONE, "the whole request parses"
    )


def test_parse_framing(mut suite: Suite) raises:
    suite.group("body framing")

    var buf = stack_allocation[SCRATCH, UInt8]()
    var req = Request()

    # The parser now stops at the blank line. Where the body goes is the body
    # reader's business, and the parser's job is to say how it is framed.
    var n = _load(
        buf, "POST /x HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhello"
    )
    var rc = parse(buf, n, req)
    suite.check(rc == PARSE_DONE, "a request with a body parses")
    suite.check(req.content_length == 5, "the content length is read")
    suite.check(req.body_kind == BODY_LENGTH, "and the framing is by length")
    suite.check(req.header_bytes == n - 5, "the header block ends before it")
    suite.check(req.has_body(), "and it has a body")

    # The parser must not wait for the body. A server that does cannot answer
    # a request whose body is larger than its read buffer.
    suite.check(
        parse(buf, n - 5, req) == PARSE_DONE,
        "the headers alone are a complete parse",
    )

    n = _load(
        buf,
        "POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n",
    )
    rc = parse(buf, n, req)
    suite.check(rc == PARSE_DONE, "chunked parses")
    suite.check(req.body_kind == BODY_CHUNKED, "and is framed as chunked")
    suite.check(req.content_length == -1, "with no content length")

    n = _load(buf, "GET / HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n")
    rc = parse(buf, n, req)
    suite.check(rc == PARSE_DONE, "a zero length body parses")
    suite.check(not req.has_body(), "and counts as no body at all")

    n = _load(
        buf,
        (
            "POST / HTTP/1.1\r\nHost: x\r\nExpect:"
            " 100-continue\r\nContent-Length: 3\r\n\r\n"
        ),
    )
    rc = parse(buf, n, req)
    suite.check(rc == PARSE_DONE, "expect 100-continue parses")
    suite.check(req.expect_continue, "and is recorded for the protocol")

    # Two requests in one buffer. The second must start exactly where the first
    # said it ended, which is the property pipelining depends on.
    n = _load(
        buf,
        "GET /a HTTP/1.1\r\nHost: x\r\n\r\nGET /b HTTP/1.1\r\nHost: x\r\n\r\n",
    )
    rc = parse(buf, n, req)
    suite.check(rc == PARSE_DONE, "the first of two pipelined requests parses")
    suite.check(span_eq(buf, req.target, "/a"), "and it is the first one")
    var first = req.header_bytes
    rc = parse(buf.unsafe_offset(first), n - first, req)
    suite.check(rc == PARSE_DONE, "the second one parses from where it ended")
    suite.check(
        span_eq(buf.unsafe_offset(first), req.target, "/b"),
        "and it is the second one",
    )


def _rejects[
    o: MutOrigin
](
    mut suite: Suite,
    buf: Pointer[UInt8, o],
    mut req: Request,
    text: StringSpan,
    status: Int,
    why: StringSpan,
):
    """Assert that a request is refused with a particular status."""
    var rc = _parse_text(buf, text, req)
    suite.check(
        rc == PARSE_FAILED and req.error_status == status,
        String(why)
        + " (got "
        + String(rc)
        + "/"
        + String(req.error_status)
        + ")",
    )


def test_parse_rejects(mut suite: Suite) raises:
    suite.group("malformed and hostile requests")

    var buf = stack_allocation[SCRATCH, UInt8]()
    var req = Request()

    # The framing pair. Both headers on one message is the original request
    # smuggling setup and there is no reading of it that two hops agree on.
    _rejects(
        suite,
        buf,
        req,
        (
            "POST / HTTP/1.1\r\nHost: x\r\nContent-Length:"
            " 5\r\nTransfer-Encoding: chunked\r\n\r\nhello"
        ),
        400,
        "content-length with transfer-encoding is rejected",
    )
    _rejects(
        suite,
        buf,
        req,
        (
            "POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\nContent-Length:"
            " 6\r\n\r\nhello"
        ),
        400,
        "a repeated content-length is rejected",
    )
    _rejects(
        suite,
        buf,
        req,
        (
            "POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding:"
            " chunked\r\nTransfer-Encoding: chunked\r\n\r\n"
        ),
        400,
        "a repeated transfer-encoding is rejected",
    )
    _rejects(
        suite,
        buf,
        req,
        (
            "POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: gzip,"
            " chunked\r\n\r\n"
        ),
        501,
        "a transfer-encoding we cannot apply is 501, not ignored",
    )

    # Whitespace tricks around the name. Every one of these is a header to one
    # implementation and not to the next, which is the whole game.
    _rejects(
        suite,
        buf,
        req,
        "GET / HTTP/1.1\r\nHost: x\r\nContent-Length : 5\r\n\r\n",
        400,
        "whitespace before the colon is rejected",
    )
    _rejects(
        suite,
        buf,
        req,
        "GET / HTTP/1.1\r\nHost: x\r\n\tmore\r\n\r\n",
        400,
        "obsolete line folding is rejected",
    )
    _rejects(
        suite,
        buf,
        req,
        "GET / HTTP/1.1\r\nHost: x\r\n Content-Length: 5\r\n\r\n",
        400,
        "a space folded header is rejected",
    )

    # Line terminators. A bare LF is the single most productive smuggling
    # primitive there is, because a hop that treats it as a terminator and one
    # that does not will disagree about where a request ends.
    _rejects(
        suite,
        buf,
        req,
        "GET / HTTP/1.1\nHost: x\n\n",
        400,
        "a bare LF is not a line terminator",
    )
    _rejects(
        suite,
        buf,
        req,
        "GET / HTTP/1.1\r\nHost: x\nContent-Length: 5\r\n\r\n",
        400,
        "a bare LF inside the header block is rejected",
    )
    _rejects(
        suite,
        buf,
        req,
        "GET / HTTP/1.1\r\nHost: x\rContent-Length: 5\r\n\r\n",
        400,
        "a bare CR inside a header is rejected",
    )

    # Content-Length values that are not a plain decimal number.
    _rejects(
        suite,
        buf,
        req,
        "POST / HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n",
        400,
        "a non numeric content-length is rejected",
    )
    _rejects(
        suite,
        buf,
        req,
        "POST / HTTP/1.1\r\nHost: x\r\nContent-Length: +5\r\n\r\n",
        400,
        "a signed content-length is rejected",
    )
    _rejects(
        suite,
        buf,
        req,
        "POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 5 5\r\n\r\n",
        400,
        "two numbers in one content-length is rejected",
    )
    _rejects(
        suite,
        buf,
        req,
        "POST / HTTP/1.1\r\nHost: x\r\nContent-Length: \r\n\r\n",
        400,
        "an empty content-length is rejected",
    )

    # The request line itself.
    _rejects(suite, buf, req, "GET / HTTP/2.0\r\n\r\n", 505, "2.0 is a 505")
    _rejects(
        suite,
        buf,
        req,
        "GE\tT / HTTP/1.1\r\n\r\n",
        400,
        "a control character in the method is rejected",
    )
    _rejects(
        suite,
        buf,
        req,
        "GET  / HTTP/1.1\r\nHost: x\r\n\r\n",
        400,
        "two spaces in the request line is rejected",
    )
    _rejects(
        suite,
        buf,
        req,
        "GET /\tx HTTP/1.1\r\nHost: x\r\n\r\n",
        400,
        "a tab in the target is rejected",
    )
    _rejects(
        suite, buf, req, "GET /\r\nHost: x\r\n\r\n", 400, "no version is a 400"
    )

    # Host. Zero is what a smuggled request looks like once a front end has
    # stripped the outer one, and two is a request two hops route differently.
    _rejects(
        suite,
        buf,
        req,
        "GET / HTTP/1.1\r\n\r\n",
        400,
        "HTTP/1.1 without a host is rejected",
    )
    _rejects(
        suite,
        buf,
        req,
        "GET / HTTP/1.1\r\nHost: a\r\nHost: b\r\n\r\n",
        400,
        "two hosts are rejected",
    )

    # An expectation we do not implement has to be refused, because the client
    # is sitting there waiting for an answer that would never come.
    _rejects(
        suite,
        buf,
        req,
        (
            "POST / HTTP/1.1\r\nHost: x\r\nExpect: something\r\nContent-Length:"
            " 1\r\n\r\n"
        ),
        417,
        "an unknown expectation is a 417",
    )

    # An embedded NUL is the classic way to make a name mean one thing to a
    # C parser and another to a length based one.
    var n = _load(buf, "GET / HTTP/1.1\r\nHost: x\r\nX-A: 1\r\n\r\n")
    buf.unsafe_store(30, UInt8(0))
    var rc = parse(buf, n, req)
    suite.check(
        rc == PARSE_FAILED and req.error_status == 400,
        "a NUL inside a header value is rejected",
    )


def test_scan(mut suite: Suite) raises:
    suite.group("simd scanning")

    var buf = stack_allocation[512, UInt8]()
    for i in range(512):
        buf.unsafe_store(i, UInt8(65))
    buf.unsafe_store(200, UInt8(13))
    buf.unsafe_store(201, UInt8(10))
    buf.unsafe_store(300, UInt8(0))

    suite.check(find_byte(buf, 0, 512, UInt8(10)) == 201, "an LF is found")
    suite.check(
        find_byte(buf, 202, 512, UInt8(10)) == -1,
        "and not found once it is behind the start",
    )
    suite.check(find_crlf(buf, 0, 512) == 200, "a CRLF is found at its CR")
    suite.check(find_ctl(buf, 0, 512) == 200, "the CR counts as a control")
    suite.check(find_ctl(buf, 202, 512) == 300, "and so does an embedded NUL")
    suite.check(
        find_ctl(buf, 202, 300) == -1, "with nothing in between flagged"
    )
    suite.check(count_byte(buf, 0, 100, UInt8(65)) == 100, "bytes are counted")

    # The tail. A scanner that only looks at whole vector widths misses
    # anything in the last few bytes, and a header block ends in the tail more
    # often than not.
    var short = stack_allocation[7, UInt8]()
    for i in range(7):
        short.unsafe_store(i, UInt8(97))
    short.unsafe_store(6, UInt8(10))
    suite.check(
        find_byte(short, 0, 7, UInt8(10)) == 6,
        "a match in the last byte of a short buffer is found",
    )

    # A byte at every offset, so an off by one in the vector loop shows up
    # rather than hiding behind an alignment that happens to work.
    var each = stack_allocation[130, UInt8]()
    var all_found = True
    for at in range(130):
        for i in range(130):
            each.unsafe_store(i, UInt8(65))
        each.unsafe_store(at, UInt8(10))
        if find_byte(each, 0, 130, UInt8(10)) != at:
            all_found = False
    suite.check(all_found, "a target at every offset is found at that offset")


def _feed_body[
    o: MutOrigin
](mut reader: BodyReader, buf: Pointer[UInt8, o], text: StringSpan) -> Int:
    var n = _load(buf, text)
    return reader.feed(buf, 0, n)


def _body_is(mut reader: BodyReader, text: StringSpan) -> Bool:
    var got = reader.bytes()
    if len(got) != text.byte_length():
        return False
    var p = text.unsafe_ptr()
    for i in range(len(got)):
        if got[i] != p.unsafe_load(i):
            return False
    return True


def test_body_reader(mut suite: Suite) raises:
    suite.group("request bodies")

    var buf = stack_allocation[SCRATCH, UInt8]()
    var counter = AllocCounter()

    var fixed = BodyReader(BODY_LENGTH, 5, 1 << 20, counter.raw())
    suite.check(
        _feed_body(fixed, buf, "hello") == BODY_DONE, "a fixed body completes"
    )
    suite.check(_body_is(fixed, "hello"), "with the right bytes")
    suite.check(fixed.last_consumed() == 5, "and takes exactly its length")
    _ = fixed^

    # Trailing bytes belong to the next request and must be left alone.
    var trailing = BodyReader(BODY_LENGTH, 5, 1 << 20, counter.raw())
    suite.check(
        _feed_body(trailing, buf, "helloGET /") == BODY_DONE,
        "a fixed body completes with more behind it",
    )
    suite.check(
        trailing.last_consumed() == 5, "and does not eat the next request"
    )
    _ = trailing^

    # Split across feeds, which is what actually happens on a socket.
    var split = BodyReader(BODY_LENGTH, 5, 1 << 20, counter.raw())
    suite.check(
        _feed_body(split, buf, "he") == BODY_NEED_MORE,
        "half a fixed body is incomplete",
    )
    suite.check(
        _feed_body(split, buf, "llo") == BODY_DONE, "and the rest finishes it"
    )
    suite.check(_body_is(split, "hello"), "with the halves joined in order")
    _ = split^

    var chunked = BodyReader(BODY_CHUNKED, -1, 1 << 20, counter.raw())
    suite.check(
        _feed_body(chunked, buf, "5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n")
        == BODY_DONE,
        "a chunked body completes",
    )
    suite.check(_body_is(chunked, "hello world"), "with the chunks joined")
    _ = chunked^

    # A chunk extension and a trailer, both legal and both easy to misread as
    # part of the size or the body.
    var extended = BodyReader(BODY_CHUNKED, -1, 1 << 20, counter.raw())
    suite.check(
        _feed_body(extended, buf, "5;a=b\r\nhello\r\n0\r\nX-T: 1\r\n\r\n")
        == BODY_DONE,
        "a chunk extension and a trailer are skipped",
    )
    suite.check(_body_is(extended, "hello"), "leaving only the body")
    _ = extended^

    # One byte at a time, driven the way a connection drives it: what the
    # reader did not consume stays in front of what arrives next. A chunk size
    # line has to arrive whole, so on this schedule the reader spends most of
    # its feeds taking nothing, and a caller that assumed every feed consumed
    # everything would lose the size line.
    var dribbled = BodyReader(BODY_CHUNKED, -1, 1 << 20, counter.raw())
    var whole = "5\r\nhello\r\n0\r\n\r\n"
    var n = _load(buf, whole)
    var status = BODY_NEED_MORE
    var fed = 0
    var have = 0
    for _ in range(n):
        have += 1
        status = dribbled.feed(buf.unsafe_offset(fed), 0, have)
        var took = dribbled.last_consumed()
        fed += took
        have -= took
    suite.check(status == BODY_DONE, "a chunked body arriving byte by byte")
    suite.check(_body_is(dribbled, "hello"), "reads the same as in one go")
    _ = dribbled^

    var bad = BodyReader(BODY_CHUNKED, -1, 1 << 20, counter.raw())
    suite.check(
        _feed_body(bad, buf, "zz\r\nhello\r\n0\r\n\r\n") == BODY_FAILED,
        "a non hex chunk size is refused",
    )
    _ = bad^

    var over = BodyReader(BODY_LENGTH, 100, 10, counter.raw())
    suite.check(
        _feed_body(over, buf, "hello") == BODY_FAILED,
        "a body over the limit is refused",
    )
    suite.check(over.error_status == 413, "with a 413")
    _ = over^

    counter.close()


def test_body_spill(mut suite: Suite) raises:
    suite.group("bodies that spill to disk")

    var counter = AllocCounter()
    var big = stack_allocation[4096, UInt8]()
    for i in range(4096):
        big.unsafe_store(i, UInt8(65 + i % 26))

    # A threshold far below the body, so the switch to a file happens part way
    # through and the bytes already held have to be carried over rather than
    # lost. That carry is the part worth testing.
    var reader = BodyReader(BODY_LENGTH, 4096, 1 << 20, counter.raw(), 1024)
    var status = reader.feed(big, 0, 4096)
    suite.check(status == BODY_DONE, "a body over the threshold completes")
    suite.check(not reader.in_memory(), "and is not in memory")
    suite.check(reader.total == 4096, "with every byte accounted for")
    suite.check(reader.spill_written == 4096, "and every byte written")

    var path = reader.spill_path
    suite.check(path.byte_length() > 0, "it has a path")
    suite.check(exists(path), "and the file is there")
    var info = FileInfo()
    _ = stat_path(path, info)
    suite.check(info.size == 4096, "with the right size on disk")

    # The file is the reader's, so it goes when the reader does. A server that
    # leaves one of these behind per upload fills the disk in an afternoon.
    _ = reader^
    suite.check(not exists(path), "and it is removed with the reader")

    counter.close()


def _writer_text(mut w: ResponseWriter) -> String:
    var out = String("")
    var got = w.bytes()
    for i in range(len(got)):
        out += chr(Int(got[i]))
    return out


def test_serialize(mut suite: Suite) raises:
    suite.group("writing responses")

    var counter = AllocCounter()
    var w = ResponseWriter(counter.raw())

    suite.check(status_text(200) == "OK", "a status has its reason phrase")
    suite.check(
        status_text(431) == "Request Header Fields Too Large",
        "even the long ones",
    )

    _ = w.respond_str(200, "text/plain", "hi", True, False)
    var text = _writer_text(w)
    suite.check(text.startswith("HTTP/1.1 200 OK\r\n"), "the status line")
    suite.check("Content-Length: 2\r\n" in text, "the content length")
    suite.check("Content-Type: text/plain\r\n" in text, "the content type")
    suite.check("Date: " in text, "a date")
    suite.check("Connection: keep-alive\r\n" in text, "and a connection header")
    suite.check(text.endswith("\r\n\r\nhi"), "with the body after a blank line")

    # HEAD. The headers are the ones a GET would have carried, including the
    # length, and the body is not there. Getting this wrong does not produce a
    # wrong page, it desynchronises the connection.
    _ = w.respond_str(200, "text/plain", "hi", True, True)
    text = _writer_text(w)
    suite.check("Content-Length: 2\r\n" in text, "a HEAD still says the length")
    suite.check(text.endswith("\r\n\r\n"), "but sends no body")
    suite.check(w.body_bytes == 2, "and counts what it did not send")

    _ = w.respond_error(404, False)
    text = _writer_text(w)
    suite.check(text.startswith("HTTP/1.1 404 Not Found"), "an error status")
    suite.check("Connection: close\r\n" in text, "an error always closes")
    suite.check(text.endswith("Not Found\n"), "and says so in the body")

    _ = w.respond_continue()
    suite.check(
        _writer_text(w) == "HTTP/1.1 100 Continue\r\n\r\n",
        "an interim response is the status line and nothing else",
    )

    # Numbers are written in place. The version that built a digit list
    # allocated once per response, which is exactly what #17 forbids.
    _ = w.start(200, False)
    _ = w.header_int("X-N", 0)
    _ = w.header_int("X-N", 1234567890)
    _ = w.header_int("X-N", -42)
    text = _writer_text(w)
    suite.check("X-N: 0\r\n" in text, "zero writes as one digit")
    suite.check("X-N: 1234567890\r\n" in text, "a long number writes in full")
    suite.check("X-N: -42\r\n" in text, "and a negative one keeps its sign")

    var out = Buffer(64, counter.raw())
    _ = chunk_header(out, 255)
    _ = chunk_header(out, 0)
    var hex = String("")
    var raw = out.bytes()
    for i in range(len(raw)):
        hex += chr(Int(raw[i]))
    suite.check(hex == "ff\r\n0\r\n", "a chunk size line is lowercase hex")

    counter.close()


def test_serialize_does_not_allocate(mut suite: Suite) raises:
    suite.group("the response path does not allocate")

    var counter = AllocCounter()
    var w = ResponseWriter(counter.raw())

    # Warm up. The first few responses grow the buffer to whatever this client
    # needs, and that growth is allocation. The claim is about the steady
    # state, not about the first request on a connection, and saying so is
    # more useful than a number that quietly depends on the initial capacity.
    for i in range(8):
        _ = w.respond_str(200, "text/plain", "Hello from molla.\n", True, False)
        _ = i

    var before = counter.total()
    for i in range(1000):
        _ = w.respond_str(200, "text/plain", "Hello from molla.\n", True, False)
        _ = w.respond_error(404, False)
        _ = i
    var after = counter.total()
    suite.check(
        after == before,
        "two thousand responses allocate nothing ("
        + String(after - before)
        + ")",
    )

    counter.close()


def test_multipart(mut suite: Suite) raises:
    suite.group("multipart form data")

    suite.check(
        boundary_of("multipart/form-data; boundary=abc") == "abc",
        "an unquoted boundary is read",
    )
    suite.check(
        boundary_of('multipart/form-data; boundary="a;b"') == "a;b",
        "a quoted boundary may contain a delimiter",
    )
    suite.check(
        boundary_of("multipart/form-data") == "",
        "a missing boundary is empty rather than a guess",
    )
    suite.check(
        boundary_of("text/plain; xboundary=no") == "",
        "and a parameter that merely ends in boundary does not match",
    )

    var buf = stack_allocation[SCRATCH, UInt8]()
    var counter = AllocCounter()

    var body = (
        "--ab12\r\nContent-Disposition: form-data;"
        ' name="field"\r\n\r\nhello\r\n--ab12\r\nContent-Disposition:'
        ' form-data; name="file"; filename="x.bin"\r\nContent-Type:'
        " application/octet-stream\r\n\r\nBODY\r\n--ab12--\r\n"
    )

    var mp = MultipartReader("ab12", counter.raw())
    var n = _load(buf, body)
    suite.check(mp.feed(buf, 0, n) == MULTIPART_DONE, "a form post completes")
    suite.check(len(mp.parts) == 2, "with both parts")
    var field = mp.part("field")
    var file = mp.part("file")
    suite.check(field == 0 and file == 1, "found by name, in order")
    suite.check(mp.parts[field].size == 5, "the field has its content")
    suite.check(mp.parts[file].filename == "x.bin", "the file has its filename")
    suite.check(
        mp.parts[file].content_type == "application/octet-stream",
        "and its content type",
    )
    _ = mp^

    # One byte at a time. A boundary landing across two feeds is the case the
    # carry buffer exists for, and feeding single bytes puts a split in every
    # possible place at once.
    var slow = MultipartReader("ab12", counter.raw())
    var status = MULTIPART_NEED_MORE
    for i in range(n):
        status = slow.feed(buf.unsafe_offset(i), 0, 1)
    suite.check(status == MULTIPART_DONE, "the same body arriving byte by byte")
    suite.check(len(slow.parts) == 2, "produces the same two parts")
    suite.check(slow.parts[0].size == 5, "with the same content length")
    suite.check(slow.parts[1].size == 4, "for both of them")
    _ = slow^

    # A backslash escaped quote in a filename, which is legal and which a
    # parser that stops at the first quote truncates.
    var quoted = MultipartReader("ab12", counter.raw())
    n = _load(
        buf,
        (
            '--ab12\r\nContent-Disposition: form-data; name="f";'
            ' filename="a\\"b.txt"\r\n\r\nz\r\n--ab12--\r\n'
        ),
    )
    suite.check(quoted.feed(buf, 0, n) == MULTIPART_DONE, "an escaped quote")
    suite.check(
        quoted.parts[0].filename == 'a"b.txt', "is kept in the filename"
    )
    _ = quoted^

    # An email content transfer encoding. Reading the part as raw bytes when
    # the sender said it was base64 hands a handler the wrong content with no
    # sign that it did, so it is refused instead.
    var encoded = MultipartReader("ab12", counter.raw())
    n = _load(
        buf,
        (
            "--ab12\r\nContent-Disposition: form-data;"
            ' name="f"\r\nContent-Transfer-Encoding:'
            " base64\r\n\r\nAAA=\r\n--ab12--\r\n"
        ),
    )
    suite.check(
        encoded.feed(buf, 0, n) == MULTIPART_FAILED,
        "base64 content transfer encoding is refused rather than misread",
    )
    _ = encoded^

    counter.close()


def _date_is[
    o: MutOrigin
](buf: Pointer[UInt8, o], expected: StringSpan) -> Bool:
    """Whether the formatted date at the front of `buf` matches.

    A helper rather than three inline loops because a pointer's origin is part
    of its type, so one `p` variable cannot be reassigned across three different
    string literals.
    """
    var p = expected.unsafe_ptr()
    for i in range(DATE_LENGTH):
        if buf.unsafe_load(i) != p.unsafe_load(i):
            return False
    return True


def test_http_date(mut suite: Suite) raises:
    suite.group("date header")

    var buf = stack_allocation[64, UInt8]()
    for i in range(64):
        buf.unsafe_store(i, UInt8(0))

    # The example date from RFC 9110, which is a known Sunday.
    format_http_date(buf, 0, 784111777)
    suite.check(
        _date_is(buf, "Sun, 06 Nov 1994 08:49:37 GMT"),
        "the RFC 9110 example date formats exactly",
    )

    # The epoch itself, a Thursday, which is the case an off by one in the
    # weekday arithmetic gets wrong.
    format_http_date(buf, 0, 0)
    suite.check(
        _date_is(buf, "Thu, 01 Jan 1970 00:00:00 GMT"),
        "the epoch formats as a Thursday",
    )

    # A leap day, which is what the civil_from_days shifted year is for.
    format_http_date(buf, 0, 1709164800)
    suite.check(
        _date_is(buf, "Thu, 29 Feb 2024 00:00:00 GMT"),
        "29 February 2024 formats correctly",
    )


def test_responder(mut suite: Suite) raises:
    suite.group("prebuilt responses")

    var r = Responder("hi\n")
    suite.check(len(r.keep) > 0, "the keep alive response is built")
    suite.check(
        len(r.close) > len(r.keep),
        "the closing response is longer, it carries Connection: close",
    )
    # The date must actually have been written, not left as the space padding
    # the buffer was reserved with.
    suite.check(
        r.keep[r.keep_date_at] != UInt8(32),
        "the date placeholder was filled in",
    )


def _client(port: UInt16) raises -> Int:
    var fd = socket_tcp()
    _ = connect(fd, INADDR_LOOPBACK, port)
    return fd


def _send_text(fd: Int, text: StringSpan) raises -> Int:
    var buf = stack_allocation[8192, UInt8]()
    var n = text.byte_length()
    var p = text.unsafe_ptr()
    for i in range(n):
        buf.unsafe_store(i, p.unsafe_load(i))
    return send(fd, buf, n)


def _read_all(fd: Int, mut server: HttpServer, want: Int) raises -> String:
    """Turn the loop until at least `want` bytes have come back."""
    var out = String("")
    var buf = stack_allocation[8192, UInt8]()
    var got_total = 0
    for _ in range(MAX_STEPS):
        _ = server.poll_once(1)
        var got = recv(fd, buf, 8192)
        if got > 0:
            got_total += got
            for i in range(got):
                out += chr(Int(buf.unsafe_load(i)))
        if got_total >= want:
            break
    return out


def test_server_round_trip(mut suite: Suite) raises:
    suite.group("http round trip")

    var server = HttpServer(INADDR_LOOPBACK, UInt16(0))
    var port = server.port()
    var client = _client(port)

    for _ in range(MAX_STEPS):
        _ = server.poll_once(1)
        if server.accepted > 0:
            break
    suite.check(server.accepted == 1, "the server accepted the connection")

    _ = _send_text(client, "GET /hello HTTP/1.1\r\nHost: x\r\n\r\n")
    var reply = _read_all(client, server, 20)
    suite.check(reply.startswith("HTTP/1.1 200 OK"), "the reply is a 200")
    suite.check("Content-Length:" in reply, "it carries a content length")
    suite.check("Date: " in reply, "it carries a date")
    suite.check("Hello from molla." in reply, "and the body came back")
    suite.check(server.requests == 1, "the server counted one request")

    _ = close(client)
    server.shutdown()


def test_server_keep_alive(mut suite: Suite) raises:
    suite.group("keep alive and pipelining")

    var server = HttpServer(INADDR_LOOPBACK, UInt16(0))
    var port = server.port()
    var client = _client(port)
    for _ in range(MAX_STEPS):
        _ = server.poll_once(1)
        if server.accepted > 0:
            break

    # Three requests down one connection, one at a time.
    for _ in range(3):
        _ = _send_text(client, "GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        var reply = _read_all(client, server, 20)
        _ = reply
    suite.check(server.requests == 3, "three requests on one connection")
    suite.check(len(server.conns) == 1, "and the connection is still open")

    # Two more in a single write, which is the pipelining case.
    _ = _send_text(
        client,
        "GET /a HTTP/1.1\r\nHost: x\r\n\r\nGET /b HTTP/1.1\r\nHost: x\r\n\r\n",
    )
    var reply = _read_all(client, server, 200)
    suite.check(server.requests == 5, "both pipelined requests were handled")
    var first = reply.find("HTTP/1.1 200")
    var second = reply.find("HTTP/1.1 200", first + 1)
    suite.check(second > first, "and both answers came back, in order")

    _ = close(client)
    server.shutdown()


def test_server_closes_when_asked(mut suite: Suite) raises:
    suite.group("connection close")

    var server = HttpServer(INADDR_LOOPBACK, UInt16(0))
    var port = server.port()
    var client = _client(port)
    for _ in range(MAX_STEPS):
        _ = server.poll_once(1)
        if server.accepted > 0:
            break

    _ = _send_text(
        client, "GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
    )
    var reply = _read_all(client, server, 20)
    suite.check("Connection: close" in reply, "the reply says it is closing")

    # The response has to be fully written before the socket goes away, so the
    # close happens on a later pass and not during the parse.
    var gone = False
    for _ in range(MAX_STEPS):
        _ = server.poll_once(1)
        if len(server.conns) == 0:
            gone = True
            break
    suite.check(gone, "and the connection is dropped afterwards")

    _ = close(client)
    server.shutdown()


def test_server_rejects_bad_request(mut suite: Suite) raises:
    suite.group("server side rejection")

    var server = HttpServer(INADDR_LOOPBACK, UInt16(0))
    var port = server.port()
    var client = _client(port)
    for _ in range(MAX_STEPS):
        _ = server.poll_once(1)
        if server.accepted > 0:
            break

    _ = _send_text(client, "GET / HTTP/1.1\r\nHost : x\r\n\r\n")
    var reply = _read_all(client, server, 20)
    suite.check(reply.startswith("HTTP/1.1 400"), "a bad request gets a 400")
    suite.check(server.errors == 1, "and the server counted it")

    _ = close(client)
    server.shutdown()


def test_server_split_request(mut suite: Suite) raises:
    suite.group("request split across reads")

    var server = HttpServer(INADDR_LOOPBACK, UInt16(0))
    var port = server.port()
    var client = _client(port)
    for _ in range(MAX_STEPS):
        _ = server.poll_once(1)
        if server.accepted > 0:
            break

    # Send the request line, turn the loop, then send the rest. The server must
    # hold the fragment rather than answering or failing.
    _ = _send_text(client, "GET /slow HTTP/1.1\r\nHo")
    for _ in range(10):
        _ = server.poll_once(1)
    suite.check(server.requests == 0, "half a request is not answered")
    suite.check(server.errors == 0, "and it is not an error either")

    _ = _send_text(client, "st: x\r\n\r\n")
    var reply = _read_all(client, server, 20)
    suite.check(reply.startswith("HTTP/1.1 200"), "the rest completes it")
    suite.check(server.requests == 1, "and it counts as exactly one request")

    _ = close(client)
    server.shutdown()


def _proto_read(
    fd: Int, mut reactor: Reactor[HttpProtocol], want: Int
) raises -> String:
    """Turn the reactor until at least `want` bytes have come back."""
    var out = String("")
    var buf = stack_allocation[8192, UInt8]()
    var got_total = 0
    for _ in range(MAX_STEPS):
        _ = reactor.poll_once(1)
        var got = recv(fd, buf, 8192)
        if got > 0:
            got_total += got
            for i in range(got):
                out += chr(Int(buf.unsafe_load(i)))
        if got_total >= want:
            break
    return out


def _proto_accept(mut reactor: Reactor[HttpProtocol], want: Int = 1) raises:
    """Turn the loop until `want` connections have been accepted in total.

    A count rather than a flag, because a test that opens a second connection
    against the same reactor would otherwise see the first one's accept and
    carry on before the second was in the table.
    """
    for _ in range(MAX_STEPS):
        _ = reactor.poll_once(1)
        if reactor.accepted >= want:
            return


def test_protocol_round_trip(mut suite: Suite) raises:
    suite.group("http on the reactor")

    var listener = open_listener(ListenAddress(UInt16(0)), False)
    var port = bound_port(listener)
    var reactor = Reactor[HttpProtocol](HttpProtocol(), 60000, 0)
    reactor.add_listener(listener)
    var client = _client(port)
    _proto_accept(reactor)
    suite.check(reactor.accepted == 1, "the reactor accepted a connection")

    _ = _send_text(client, "GET / HTTP/1.1\r\nHost: x\r\n\r\n")
    var reply = _proto_read(client, reactor, 20)
    suite.check(reply.startswith("HTTP/1.1 200 OK"), "a GET gets a 200")
    suite.check("Hello from molla." in reply, "with the body")
    suite.check(reactor.proto.requests == 1, "and it was counted once")

    _ = _send_text(client, "GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n")
    reply = _proto_read(client, reactor, 20)
    suite.check("\r\n\r\nok\n" in reply, "the health route answers")

    _ = _send_text(client, "GET /nope HTTP/1.1\r\nHost: x\r\n\r\n")
    reply = _proto_read(client, reactor, 20)
    suite.check(reply.startswith("HTTP/1.1 404"), "an unknown route is a 404")

    _ = close(client)
    reactor.shutdown()


def test_protocol_keep_alive(mut suite: Suite) raises:
    suite.group("http keep alive on the reactor")

    var listener = open_listener(ListenAddress(UInt16(0)), False)
    var port = bound_port(listener)
    var reactor = Reactor[HttpProtocol](HttpProtocol(), 60000, 0)
    reactor.add_listener(listener)
    var client = _client(port)
    _proto_accept(reactor)

    for _ in range(4):
        _ = _send_text(client, "GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        var reply = _proto_read(client, reactor, 20)
        _ = reply
    suite.check(reactor.proto.requests == 4, "four requests on one connection")
    suite.check(reactor.connection_count() == 1, "and it is still open")

    # Pipelining. Both requests are in one segment, and the readiness that
    # brought the reactor here has already been spent, so a protocol that
    # handles one and waits leaves the second sitting there forever.
    _ = _send_text(
        client,
        (
            "GET /healthz HTTP/1.1\r\nHost: x\r\n\r\nGET /healthz"
            " HTTP/1.1\r\nHost: x\r\n\r\n"
        ),
    )
    var reply = _proto_read(client, reactor, 200)
    suite.check(reactor.proto.requests == 6, "both pipelined requests answered")
    var first = reply.find("HTTP/1.1 200")
    var second = reply.find("HTTP/1.1 200", first + 1)
    suite.check(second > first, "and both answers came back, in order")

    _ = close(client)
    reactor.shutdown()


def test_protocol_bodies(mut suite: Suite) raises:
    suite.group("http bodies on the reactor")

    var listener = open_listener(ListenAddress(UInt16(0)), False)
    var port = bound_port(listener)
    var reactor = Reactor[HttpProtocol](HttpProtocol(), 60000, 0)
    reactor.add_listener(listener)
    var client = _client(port)
    _proto_accept(reactor)

    _ = _send_text(
        client,
        "POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 5\r\n\r\nhello",
    )
    var reply = _proto_read(client, reactor, 20)
    suite.check(
        reply.startswith("HTTP/1.1 405"), "a POST to a GET route is 405"
    )

    # A body has to be read to the end even when the answer does not need it,
    # or the bytes are read as the next request on the connection.
    _ = close(client)
    client = _client(port)
    _proto_accept(reactor, 2)
    _ = _send_text(
        client,
        (
            "POST /healthz HTTP/1.1\r\nHost: x\r\nContent-Length:"
            " 5\r\n\r\nhelloGET /healthz HTTP/1.1\r\nHost: x\r\n\r\n"
        ),
    )
    reply = _proto_read(client, reactor, 300)
    var first = reply.find("HTTP/1.1 200")
    var second = reply.find("HTTP/1.1 200", first + 1)
    suite.check(
        second > first,
        "a request after a body is read as a request and not as body",
    )

    # Chunked, split so the size line and the data land in different reads.
    _ = close(client)
    client = _client(port)
    _proto_accept(reactor, 3)
    _ = _send_text(
        client,
        (
            "POST /healthz HTTP/1.1\r\nHost: x\r\nTransfer-Encoding:"
            " chunked\r\n\r\n5\r\nhel"
        ),
    )
    for _ in range(10):
        _ = reactor.poll_once(1)
    _ = _send_text(client, "lo\r\n0\r\n\r\n")
    reply = _proto_read(client, reactor, 20)
    suite.check(
        reply.startswith("HTTP/1.1 200"), "a split chunked body is answered"
    )

    _ = close(client)
    reactor.shutdown()


def test_protocol_expect_and_head(mut suite: Suite) raises:
    suite.group("expect and head on the reactor")

    var listener = open_listener(ListenAddress(UInt16(0)), False)
    var port = bound_port(listener)
    var reactor = Reactor[HttpProtocol](HttpProtocol(), 60000, 0)
    reactor.add_listener(listener)
    var client = _client(port)
    _proto_accept(reactor)

    # The client is waiting to be told to send, so the interim response has to
    # go out before the body arrives rather than with the real one.
    _ = _send_text(
        client,
        (
            "POST /healthz HTTP/1.1\r\nHost: x\r\nExpect:"
            " 100-continue\r\nContent-Length: 2\r\n\r\n"
        ),
    )
    var interim = _proto_read(client, reactor, 25)
    suite.check(
        interim.startswith("HTTP/1.1 100 Continue\r\n\r\n"),
        "a 100 continue goes out before the body",
    )
    _ = _send_text(client, "hi")
    var reply = _proto_read(client, reactor, 20)
    suite.check(reply.startswith("HTTP/1.1 200"), "and then the real answer")

    _ = _send_text(client, "HEAD /healthz HTTP/1.1\r\nHost: x\r\n\r\n")
    reply = _proto_read(client, reactor, 20)
    suite.check(reply.startswith("HTTP/1.1 200"), "a HEAD is answered")
    suite.check("Content-Length: 3" in reply, "with the length a GET would say")
    suite.check(reply.endswith("\r\n\r\n"), "and nothing after the blank line")

    _ = close(client)
    reactor.shutdown()


def test_protocol_rejects(mut suite: Suite) raises:
    suite.group("http rejection on the reactor")

    var listener = open_listener(ListenAddress(UInt16(0)), False)
    var port = bound_port(listener)
    var reactor = Reactor[HttpProtocol](HttpProtocol(), 60000, 0)
    reactor.add_listener(listener)
    var client = _client(port)
    _proto_accept(reactor)

    # A smuggling attempt, and then a request behind it. The second one must
    # never be answered: once framing is in doubt the rest of the buffer
    # cannot be trusted to be a request rather than the tail of the bad one.
    _ = _send_text(
        client,
        (
            "POST / HTTP/1.1\r\nHost: x\r\nContent-Length:"
            " 6\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\nGET /healthz"
            " HTTP/1.1\r\nHost: x\r\n\r\n"
        ),
    )
    var reply = _proto_read(client, reactor, 20)
    suite.check(reply.startswith("HTTP/1.1 400"), "a smuggled pair is a 400")
    suite.check(
        reply.find("HTTP/1.1 200") < 0,
        "and what came behind it is not answered",
    )

    var gone = False
    for _ in range(MAX_STEPS):
        _ = reactor.poll_once(1)
        if reactor.connection_count() == 0:
            gone = True
            break
    suite.check(gone, "the connection is closed after a framing error")

    _ = close(client)
    reactor.shutdown()


def run(mut suite: Suite):
    try:
        test_parse_request_line(suite)
        test_parse_incremental(suite)
        test_parse_framing(suite)
        test_parse_rejects(suite)
        test_scan(suite)
        test_body_reader(suite)
        test_body_spill(suite)
        test_serialize(suite)
        test_serialize_does_not_allocate(suite)
        test_multipart(suite)
        test_http_date(suite)
        test_responder(suite)
        test_server_round_trip(suite)
        test_server_keep_alive(suite)
        test_server_closes_when_asked(suite)
        test_server_rejects_bad_request(suite)
        test_server_split_request(suite)
        test_protocol_round_trip(suite)
        test_protocol_keep_alive(suite)
        test_protocol_bodies(suite)
        test_protocol_expect_and_head(suite)
        test_protocol_rejects(suite)
    except e:
        suite.fail("test_http raised", String(e))
