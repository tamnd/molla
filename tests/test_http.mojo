"""Tests for the HTTP parser and the server loop.

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
"""

from std.memory import stack_allocation

from harness import Suite

from molla.http.request import (
    PARSE_DONE,
    PARSE_FAILED,
    PARSE_NEED_MORE,
    Request,
    Span,
    parse,
    span_eq,
)
from molla.http.response import DATE_LENGTH, Responder, format_http_date
from molla.http.server import HttpServer
from molla.sys.errno import EAGAIN, get_errno
from molla.sys.fd import close
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
    suite.check(req.consumed == 32, "consumed the whole request and no more")

    rc = _parse_text(buf, "GET / HTTP/1.0\r\n\r\n", req)
    suite.check(rc == PARSE_DONE, "HTTP/1.0 parses")
    suite.check(not req.keep_alive, "1.0 closes by default")

    rc = _parse_text(
        buf, "GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n", req
    )
    suite.check(rc == PARSE_DONE, "1.0 with keep-alive parses")
    suite.check(req.keep_alive, "and it overrides the 1.0 default")

    rc = _parse_text(buf, "GET / HTTP/1.1\r\nConnection: close\r\n\r\n", req)
    suite.check(rc == PARSE_DONE, "1.1 with close parses")
    suite.check(not req.keep_alive, "and it overrides the 1.1 default")

    # Bare LF instead of CRLF. Permitted, and clients really do send it.
    rc = _parse_text(buf, "GET / HTTP/1.1\nHost: x\n\n", req)
    suite.check(rc == PARSE_DONE, "bare LF line endings parse")


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


def test_parse_body_and_pipelining(mut suite: Suite) raises:
    suite.group("bodies and pipelining")

    var buf = stack_allocation[SCRATCH, UInt8]()
    var req = Request()

    var n = _load(buf, "POST /x HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello")
    var rc = parse(buf, n, req)
    suite.check(rc == PARSE_DONE, "a request with a body parses")
    suite.check(req.content_length == 5, "the content length is read")
    suite.check(req.consumed == n, "the body counts towards what was consumed")

    # One byte short of the declared body is not a complete request.
    suite.check(
        parse(buf, n - 1, req) == PARSE_NEED_MORE,
        "a body that has not all arrived is incomplete",
    )

    # Two requests in one buffer. The second must start exactly where the first
    # said it ended, which is the property pipelining depends on.
    n = _load(buf, "GET /a HTTP/1.1\r\n\r\nGET /b HTTP/1.1\r\n\r\n")
    rc = parse(buf, n, req)
    suite.check(rc == PARSE_DONE, "the first of two pipelined requests parses")
    suite.check(span_eq(buf, req.target, "/a"), "and it is the first one")
    var first = req.consumed
    rc = parse(buf.unsafe_offset(first), n - first, req)
    suite.check(rc == PARSE_DONE, "the second one parses from where it ended")
    suite.check(
        span_eq(buf.unsafe_offset(first), req.target, "/b"),
        "and it is the second one",
    )


def test_parse_rejects(mut suite: Suite) raises:
    suite.group("malformed and hostile requests")

    var buf = stack_allocation[SCRATCH, UInt8]()
    var req = Request()

    var rc = _parse_text(
        buf,
        (
            "POST / HTTP/1.1\r\nContent-Length: 5\r\nTransfer-Encoding:"
            " chunked\r\n\r\nhello"
        ),
        req,
    )
    suite.check(
        rc == PARSE_FAILED and req.error_status == 400,
        "content-length with transfer-encoding is rejected",
    )

    rc = _parse_text(
        buf,
        (
            "POST / HTTP/1.1\r\nContent-Length: 5\r\nContent-Length:"
            " 6\r\n\r\nhello"
        ),
        req,
    )
    suite.check(
        rc == PARSE_FAILED and req.error_status == 400,
        "a repeated content-length is rejected",
    )

    rc = _parse_text(buf, "GET / HTTP/1.1\r\nHost : x\r\n\r\n", req)
    suite.check(
        rc == PARSE_FAILED and req.error_status == 400,
        "whitespace before the colon is rejected",
    )

    rc = _parse_text(buf, "GET / HTTP/1.1\r\nHost: x\r\n\tmore\r\n\r\n", req)
    suite.check(
        rc == PARSE_FAILED and req.error_status == 400,
        "obsolete line folding is rejected",
    )

    rc = _parse_text(buf, "GET / HTTP/1.1\r\nContent-Length: abc\r\n\r\n", req)
    suite.check(
        rc == PARSE_FAILED and req.error_status == 400,
        "a non numeric content-length is rejected",
    )

    rc = _parse_text(buf, "GET / HTTP/2.0\r\n\r\n", req)
    suite.check(
        rc == PARSE_FAILED and req.error_status == 505,
        "an unsupported minor version is 505",
    )

    rc = _parse_text(buf, "GE\tT / HTTP/1.1\r\n\r\n", req)
    suite.check(
        rc == PARSE_FAILED and req.error_status == 400,
        "a control character in the method is rejected",
    )

    rc = _parse_text(
        buf,
        "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
        req,
    )
    suite.check(
        rc == PARSE_FAILED and req.error_status == 501,
        "chunked is 501 rather than silently misframed",
    )


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
    var buf = stack_allocation[4096, UInt8]()
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

    _ = _send_text(client, "GET / HTTP/1.1\r\nConnection: close\r\n\r\n")
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


def run(mut suite: Suite):
    try:
        test_parse_request_line(suite)
        test_parse_incremental(suite)
        test_parse_body_and_pipelining(suite)
        test_parse_rejects(suite)
        test_http_date(suite)
        test_responder(suite)
        test_server_round_trip(suite)
        test_server_keep_alive(suite)
        test_server_closes_when_asked(suite)
        test_server_rejects_bad_request(suite)
        test_server_split_request(suite)
    except e:
        suite.fail("test_http raised", String(e))
