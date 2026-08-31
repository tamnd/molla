"""Tests for the SSE and NDJSON writers.

Three groups. The framing tests run against the writer directly, because the
whole contract is what bytes come out and going through a socket to look at
them would only add ways to be flaky. The protocol tests run a real stream over
a real loopback socket. The slow reader test is the one the issue actually asks
for and is the reason this file exists.

The slow reader is not literally one byte per second, because a test that takes
a minute to say yes is a test nobody runs. It is one byte per pass of the loop
against a server with the socket buffers and the output ring pinned small, which
puts the writer into backpressure on almost every pass and keeps it there for
the whole stream. That is the state the one byte per second client would be in,
reached in a second rather than in ten minutes.
"""

from std.memory import stack_allocation

from harness import Suite

from molla.http.protocol import HttpProtocol
from molla.http.serialize import ResponseWriter
from molla.http.stream import (
    STREAM_CLOSED,
    STREAM_FULL,
    STREAM_INVALID,
    STREAM_OK,
    StreamWriter,
    ndjson_headers,
    sse_headers,
)
from molla.net.listener import ListenAddress, bound_port, open_listener
from molla.net.reactor import Reactor
from molla.sys.fd import close
from molla.sys.mem import AllocCounter
from molla.sys.socket import (
    INADDR_LOOPBACK,
    SO_RCVBUF,
    SO_SNDBUF,
    connect,
    recv,
    send,
    set_buffer_size,
    socket_tcp,
)

comptime MAX_STEPS = 600


def _text(s: Span[UInt8, MutAnyOrigin]) -> String:
    var out = String("")
    for i in range(len(s)):
        out += chr(Int(s[i]))
    return out


def _staged(mut w: StreamWriter) -> String:
    """What is waiting, unframed. Only meaningful before a flush."""
    return _text(w.pending.bytes())


def test_sse_framing(mut suite: Suite) raises:
    suite.group("sse framing")

    var counter = AllocCounter()
    var w = StreamWriter(counter.raw())
    w.begin(True, False, 0)

    _ = w.event_str("token", "hello", "1", 0)
    suite.check(
        _staged(w) == "event: token\nid: 1\ndata: hello\n\n",
        "an event carries its name, id, data and a blank line",
    )

    w.begin(True, False, 0)
    _ = w.event_str("", "plain", "", 0)
    suite.check(
        _staged(w) == "data: plain\n\n",
        "an empty name and id leave the fields out",
    )

    # A payload with newlines is legal SSE and becomes several data lines. Any
    # of the three line endings counts, because a lone CR is a line break to an
    # SSE client and is not one to a writer that only looks for LF.
    w.begin(True, False, 0)
    _ = w.event_str("", "a\nb\r\nc\rd", "", 0)
    suite.check(
        _staged(w) == "data: a\ndata: b\ndata: c\ndata: d\n\n",
        "a payload with all three line endings splits into four data lines",
    )

    w.begin(True, False, 0)
    _ = w.event_str("", "trailing\n", "", 0)
    suite.check(
        _staged(w) == "data: trailing\ndata: \n\n",
        "a trailing newline is a real empty line and is kept",
    )

    w.begin(True, False, 0)
    suite.check(
        w.event_str("bad\nname", "x", "", 0) == STREAM_INVALID,
        "a newline in the event name is refused",
    )
    suite.check(
        w.event_str("", "x", "bad\rid", 0) == STREAM_INVALID,
        "and so is one in the id",
    )
    suite.check(
        w.event_str("", "x", "nul\0id", 0) == STREAM_INVALID,
        "and a NUL in the id",
    )
    suite.check(w.staged() == 0, "and none of the three staged anything")

    w.begin(True, False, 0)
    _ = w.comment("ping", 0)
    suite.check(_staged(w) == ": ping\n\n", "a comment is a colon line")
    w.begin(True, False, 0)
    _ = w.retry(1500, 0)
    suite.check(_staged(w) == "retry: 1500\n\n", "retry carries the number")

    w.begin(True, False, 0)
    suite.check(
        w.record_str("{}", 0) == STREAM_INVALID,
        "an NDJSON record on an SSE stream is refused",
    )


def test_ndjson_framing(mut suite: Suite) raises:
    suite.group("ndjson framing")

    var counter = AllocCounter()
    var w = StreamWriter(counter.raw())
    w.begin(False, False, 0)

    _ = w.record_str('{"a":1}', 0)
    _ = w.record_str('{"a":2}', 0)
    suite.check(
        _staged(w) == '{"a":1}\n{"a":2}\n',
        "records are one line each",
    )

    w.begin(False, False, 0)
    suite.check(
        w.record_str('{"a":"x\ny"}', 0) == STREAM_INVALID,
        "a record with a newline in it is refused, because it is two records",
    )
    suite.check(
        w.record_str('{"a":"x\ry"}', 0) == STREAM_INVALID,
        "and a carriage return is refused for the same reason",
    )
    suite.check(w.staged() == 0, "and neither one staged anything")

    suite.check(
        w.event_str("token", "x", "", 0) == STREAM_INVALID,
        "an SSE event on an NDJSON stream is refused",
    )
    suite.check(
        w.comment("ping", 0) == STREAM_INVALID,
        "and NDJSON has no comment to hide a heartbeat in",
    )
    suite.check(
        not w.heartbeat_due(1000000),
        "so a silent NDJSON stream never says a heartbeat is due",
    )


def test_stream_backpressure(mut suite: Suite) raises:
    suite.group("stream backpressure")

    var counter = AllocCounter()
    var w = StreamWriter(counter.raw(), 64)
    w.begin(True, False, 0)

    var accepted = 0
    for _ in range(100):
        if w.event_str("", "0123456789", "", 0) != STREAM_OK:
            break
        accepted += 1
    suite.check(accepted > 0, "the first events are taken")
    suite.check(accepted < 100, "and a producer that keeps going is stopped")
    suite.check(
        w.staged() <= 64,
        "staged bytes never go past the limit (" + String(w.staged()) + ")",
    )
    suite.check(
        w.event_str("", "0123456789", "", 0) == STREAM_FULL,
        "and it keeps saying so rather than growing",
    )

    _ = w.end(0)
    suite.check(
        w.event_str("", "x", "", 0) == STREAM_CLOSED,
        "an event after the end is refused",
    )

    w.begin(True, False, 0)
    _ = w.event_str("", "x", "", 0)
    w.abort()
    suite.check(w.staged() == 0, "abort drops what was staged")
    suite.check(
        w.event_str("", "x", "", 0) == STREAM_CLOSED,
        "and an aborted stream takes nothing more",
    )


def test_heartbeat(mut suite: Suite) raises:
    suite.group("stream heartbeat")

    var counter = AllocCounter()
    var w = StreamWriter(counter.raw())
    w.begin(True, False, 1000)

    suite.check(not w.heartbeat_due(1000), "nothing is due immediately")
    suite.check(not w.heartbeat_due(15999), "nor a millisecond early")
    suite.check(w.heartbeat_due(16000), "one is due after fifteen seconds")

    _ = w.event_str("", "x", "", 16000)
    suite.check(not w.heartbeat_due(16000), "an event resets the silence")

    # A stream with bytes waiting for a slow reader is busy, not silent, and
    # another comment on the pile would help nobody.
    suite.check(
        not w.heartbeat_due(40000),
        "a stream with staged bytes is not silent",
    )

    _ = w.heartbeat(60000)
    suite.check(": ping\n" in _staged(w), "the heartbeat is a comment line")

    _ = w.end(60000)
    suite.check(not w.heartbeat_due(200000), "an ended stream never wants one")


def test_stream_headers(mut suite: Suite) raises:
    suite.group("stream headers")

    var counter = AllocCounter()
    var w = ResponseWriter(counter.raw())

    _ = sse_headers(w, True, False)
    var head = _text(w.bytes())
    suite.check(head.startswith("HTTP/1.1 200 OK\r\n"), "sse answers 200")
    suite.check(
        "Content-Type: text/event-stream\r\n" in head, "with the event type"
    )
    suite.check(
        "Transfer-Encoding: chunked\r\n" in head,
        "and chunked framing, since the length is not known yet",
    )
    suite.check("Cache-Control: no-cache\r\n" in head, "and no caching")
    suite.check(
        "X-Accel-Buffering: no\r\n" in head,
        "and the header that stops nginx buffering the whole stream",
    )
    suite.check(
        "Content-Length" not in head,
        "and no Content-Length, which is the point",
    )

    _ = ndjson_headers(w, True, False)
    head = _text(w.bytes())
    suite.check(
        "Content-Type: application/x-ndjson\r\n" in head,
        "ndjson answers with the ndjson type",
    )


def _client(port: UInt16, rcvbuf: Int = 0) raises -> Int:
    var fd = socket_tcp()
    if rcvbuf > 0:
        set_buffer_size(fd, SO_RCVBUF, rcvbuf)
    _ = connect(fd, INADDR_LOOPBACK, port)
    return fd


def _send_text(fd: Int, text: StringSpan) raises -> Int:
    var buf = stack_allocation[8192, UInt8]()
    var n = text.byte_length()
    var p = text.unsafe_ptr()
    for i in range(n):
        buf.unsafe_store(i, p.unsafe_load(i))
    return send(fd, buf, n)


def _accept(mut reactor: Reactor[HttpProtocol], want: Int = 1) raises:
    for _ in range(MAX_STEPS):
        _ = reactor.poll_once(1)
        if reactor.accepted >= want:
            return


def _read_until(
    fd: Int, mut reactor: Reactor[HttpProtocol], marker: StringSpan
) raises -> String:
    """Turn the loop until `marker` has come back, or the budget runs out."""
    var out = String("")
    var buf = stack_allocation[8192, UInt8]()
    for _ in range(MAX_STEPS):
        _ = reactor.poll_once(1)
        var got = recv(fd, buf, 8192)
        if got > 0:
            for i in range(got):
                out += chr(Int(buf.unsafe_load(i)))
        if marker in out:
            break
    return out


def _chunks(body: StringSpan) -> Int:
    """How many non empty chunks the body is framed as."""
    var count = 0
    var at = 0
    var n = body.byte_length()
    var raw = body.as_bytes()
    while at < n:
        var line = at
        while line < n and raw[line] != 13:
            line += 1
        if line + 1 >= n or raw[line + 1] != 10:
            return count
        var size = 0
        for i in range(at, line):
            var c = Int(raw[i])
            if c >= 48 and c <= 57:
                size = size * 16 + c - 48
            elif c >= 97 and c <= 102:
                size = size * 16 + c - 87
            elif c >= 65 and c <= 70:
                size = size * 16 + c - 55
            else:
                return count
        at = line + 2
        if size == 0:
            return count
        count += 1
        at += size + 2
    return count


def _dechunk(body: StringSpan) -> String:
    """Undo chunked transfer encoding. Returns an empty string on bad framing.

    Written out by hand rather than run through `BodyReader`, because the point
    of the test is to check what the writer produced, and checking it with the
    reader from the same change would agree with itself.
    """
    var out = String("")
    var at = 0
    var n = body.byte_length()
    var raw = body.as_bytes()
    while at < n:
        var line = at
        while line < n and raw[line] != 13:
            line += 1
        if line + 1 >= n or raw[line + 1] != 10:
            return String("")
        var size = 0
        for i in range(at, line):
            var c = Int(raw[i])
            if c >= 48 and c <= 57:
                size = size * 16 + c - 48
            elif c >= 97 and c <= 102:
                size = size * 16 + c - 87
            elif c >= 65 and c <= 70:
                size = size * 16 + c - 55
            else:
                return String("")
        at = line + 2
        if size == 0:
            return out
        if at + size + 2 > n:
            return String("")
        for i in range(at, at + size):
            out += chr(Int(raw[i]))
        at += size
        if raw[at] != 13 or raw[at + 1] != 10:
            return String("")
        at += 2
    return String("")


def test_protocol_sse(mut suite: Suite) raises:
    suite.group("sse on the reactor")

    var listener = open_listener(ListenAddress(UInt16(0)), False)
    var port = bound_port(listener)
    var reactor = Reactor[HttpProtocol](HttpProtocol(), 60000, 0)
    reactor.proto.configure_stream(4)
    reactor.add_listener(listener)
    var client = _client(port)
    _accept(reactor)

    _ = _send_text(client, "GET /stream/sse HTTP/1.1\r\nHost: x\r\n\r\n")
    var reply = _read_until(client, reactor, "0\r\n\r\n")
    suite.check(reply.startswith("HTTP/1.1 200 OK"), "the stream answers 200")
    suite.check(
        "Content-Type: text/event-stream" in reply, "with the event type"
    )
    suite.check(reactor.proto.streams == 1, "and the stream was counted")

    var split = reply.find("\r\n\r\n")
    var body = _dechunk(reply[byte = split + 4 :])
    suite.check(
        body
        == 'event: token\ndata: {"i":0}\n\nevent: token\ndata: {"i":1}\n\n'
        'event: token\ndata: {"i":2}\n\nevent: token\ndata: {"i":3}\n\n',
        "and four events came out of the chunked body",
    )

    # A reader that keeps up gets one chunk per event. The alternative, holding
    # events until there are enough to be worth a write, is how a token stream
    # arrives in bursts and looks like a server that is thinking.
    suite.check(
        _chunks(reply[byte = split + 4 :]) == 4,
        "one chunk per event when the reader keeps up",
    )
    suite.check(
        reactor.proto.states[0].stream.coalesced == 0,
        "and nothing was coalesced, because nothing had to be",
    )

    # The stream ended without closing the connection, so the next request goes
    # on the same one. A streaming response that always closes is a streaming
    # response that costs a connection setup per completion.
    _ = _send_text(client, "GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n")
    reply = _read_until(client, reactor, "ok\n")
    suite.check(
        reply.startswith("HTTP/1.1 200"),
        "and the connection is reusable after the last chunk",
    )

    _ = close(client)
    reactor.shutdown()


def test_protocol_ndjson(mut suite: Suite) raises:
    suite.group("ndjson on the reactor")

    var listener = open_listener(ListenAddress(UInt16(0)), False)
    var port = bound_port(listener)
    var reactor = Reactor[HttpProtocol](HttpProtocol(), 60000, 0)
    reactor.proto.configure_stream(3)
    reactor.add_listener(listener)
    var client = _client(port)
    _accept(reactor)

    _ = _send_text(client, "GET /stream/ndjson HTTP/1.1\r\nHost: x\r\n\r\n")
    var reply = _read_until(client, reactor, "0\r\n\r\n")
    suite.check(
        "Content-Type: application/x-ndjson" in reply, "ndjson answers 200"
    )
    var split = reply.find("\r\n\r\n")
    var body = _dechunk(reply[byte = split + 4 :])
    suite.check(
        body == '{"i":0}\n{"i":1}\n{"i":2}\n',
        "and three records came back, one per line",
    )

    # HEAD on a stream is the headers and nothing else. Sending a body to a
    # HEAD does not produce a wrong page, it desynchronises the connection.
    _ = _send_text(client, "HEAD /stream/sse HTTP/1.1\r\nHost: x\r\n\r\n")
    reply = _read_until(client, reactor, "\r\n\r\n")
    suite.check(
        reply.startswith("HTTP/1.1 200"), "a HEAD on a stream answers 200"
    )
    suite.check(
        reply.endswith("\r\n\r\n") and "data:" not in reply,
        "with the headers and not one byte of body",
    )

    _ = _send_text(client, "POST /stream/sse HTTP/1.1\r\nHost: x\r\n\r\n")
    reply = _read_until(client, reactor, "\r\n\r\n")
    suite.check(
        reply.startswith("HTTP/1.1 405"), "a POST to a stream route is a 405"
    )

    _ = close(client)
    reactor.shutdown()


def test_slow_reader(mut suite: Suite) raises:
    suite.group("stream against a slow reader")

    # Everything is pinned small so backpressure is the normal state rather
    # than something that happens once if the timing is right. The listener's
    # send buffer is set because an accepted socket inherits it, and the
    # reactor's write ring is set because that is the other half of the path.
    var listener = open_listener(ListenAddress(UInt16(0)), False)
    set_buffer_size(listener, SO_SNDBUF, 8192)
    var port = bound_port(listener)
    var reactor = Reactor[HttpProtocol](HttpProtocol(), 60000, 0)
    reactor.write_capacity = 2048
    reactor.proto.configure_stream(400)
    reactor.add_listener(listener)
    var client = _client(port, 8192)
    _accept(reactor)

    _ = _send_text(client, "GET /stream/sse HTTP/1.1\r\nHost: x\r\n\r\n")

    # One byte per pass. The writer spends the whole stream unable to queue
    # what it has, which is the state a one byte per second client puts it in.
    var out = String("")
    var buf = stack_allocation[8192, UInt8]()
    var done = False
    for _ in range(200000):
        _ = reactor.poll_once(0)
        var got = recv(client, buf, 1)
        if got > 0:
            out += chr(Int(buf.unsafe_load(0)))
        if out.endswith("0\r\n\r\n"):
            done = True
            break
    suite.check(done, "the whole stream arrives one byte at a time")

    var split = out.find("\r\n\r\n")
    var body = _dechunk(out[byte = split + 4 :])
    suite.check(body.byte_length() > 0, "and the chunk framing is intact")

    # One check for four hundred events rather than four hundred checks, with
    # the first index that went wrong in the message so a failure says which.
    var seen = 0
    var wrong = -1
    var at = body.find('data: {"i":')
    while at >= 0:
        var want = 'data: {"i":' + String(seen) + "}"
        if body.find(want, at) != at and wrong < 0:
            wrong = seen
        seen += 1
        at = body.find('data: {"i":', at + 1)
    suite.check(
        seen == 400 and wrong < 0,
        (
            "all 400 events arrived in order (saw "
            + String(seen)
            + ", first out of place "
            + String(wrong)
            + ")"
        ),
    )
    suite.check(
        reactor.proto.states[0].stream.coalesced > 0,
        "and events were coalesced into shared chunks under backpressure",
    )

    _ = close(client)
    reactor.shutdown()


def test_stream_hangup(mut suite: Suite) raises:
    suite.group("stream terminated mid response")

    var listener = open_listener(ListenAddress(UInt16(0)), False)
    set_buffer_size(listener, SO_SNDBUF, 8192)
    var port = bound_port(listener)
    var reactor = Reactor[HttpProtocol](HttpProtocol(), 60000, 0)
    reactor.write_capacity = 2048
    reactor.proto.configure_stream(20000)
    reactor.add_listener(listener)
    var client = _client(port, 8192)
    _accept(reactor)

    _ = _send_text(client, "GET /stream/sse HTTP/1.1\r\nHost: x\r\n\r\n")

    # Read enough to know the stream is running, then hang up in the middle of
    # it. This is the ordinary case for a completion: a browser tab closes.
    var buf = stack_allocation[8192, UInt8]()
    var got_any = False
    for _ in range(MAX_STEPS):
        _ = reactor.poll_once(1)
        if recv(client, buf, 8192) > 0:
            got_any = True
            break
    suite.check(got_any, "the stream started")
    _ = close(client)

    var gone = False
    for _ in range(MAX_STEPS):
        _ = reactor.poll_once(1)
        if reactor.connection_count() == 0:
            gone = True
            break
    suite.check(gone, "the connection is released rather than hanging")
    suite.check(
        reactor.proto.aborted == 1, "and the stream was recorded as aborted"
    )
    suite.check(
        reactor.proto.states[0].stream.staged() == 0,
        "with nothing left staged for a client that is gone",
    )

    reactor.shutdown()


def run(mut suite: Suite):
    try:
        test_sse_framing(suite)
        test_ndjson_framing(suite)
        test_stream_backpressure(suite)
        test_heartbeat(suite)
        test_stream_headers(suite)
        test_protocol_sse(suite)
        test_protocol_ndjson(suite)
        test_slow_reader(suite)
        test_stream_hangup(suite)
    except e:
        suite.fail("test_stream raised", String(e))
