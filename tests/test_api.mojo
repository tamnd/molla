"""Tests for `molla.api.openai` and the routes that use it.

None of this needs a model file. Parsing a request, refusing a field, and
building a response are all pure functions over bytes, and the routes are
checked against a server started without a model, which answers 503 and proves
the dispatch reached them. What a model does with a prompt is `test_engine`'s
job and the fleet run's.

The response tests read the bytes back rather than comparing to a fixed string.
A fixed string is a test of the key order, and the key order is not something
the specification says anything about. What matters is that `finish_reason` is
present and null on a middle chunk, which is a thing an SDK checks and a thing
that is easy to get wrong by leaving the key out entirely.
"""

from std.memory import stack_allocation

from harness import Suite

from molla.api.openai import (
    DEFAULT_MAX_TOKENS,
    FINISH_LENGTH,
    FINISH_STOP,
    add_text_choice,
    begin_text_body,
    end_text_body,
    parse_chat,
    parse_completions,
    write_chat_body,
    write_chat_chunk,
    write_error,
    write_models,
    write_text_chunk,
)
from molla.http.protocol import HttpProtocol
from molla.io.buffer import Buffer
from molla.json.dom import Document, parse
from molla.json.reader import Reader
from molla.json.serialize import Writer
from molla.net.listener import ListenAddress, bound_port, open_listener
from molla.net.reactor import Reactor
from molla.sys.fd import close
from molla.sys.socket import (
    INADDR_LOOPBACK,
    connect,
    recv,
    send,
    socket_tcp,
)

comptime MAX_STEPS = 400
"""Turns of the reactor a request gets before a test gives up. Generous,
because a slow machine under a full suite is still a machine that answers."""


def _text_of(w: Writer) -> String:
    return String(StringSpan(unsafe_from_utf8=w.bytes()))


struct Body(Movable):
    """A json body, parsed, with everything the parsers want to borrow.

    One struct because the document holds offsets into the buffer and the
    reader owns the scratch the parse needed, so all three have to outlive the
    request that came out of them.
    """

    var buf: Buffer
    var reader: Reader
    var doc: Document
    var scratch: Writer

    def __init__(out self, text: StringSpan) raises:
        self.buf = Buffer(8192, 0)
        self.reader = Reader(0)
        self.doc = Document(0)
        self.scratch = Writer(0, 8192)
        _ = self.buf.append_str(text)
        if not parse(self.doc, self.reader, self.buf.bytes()):
            raise Error("the test body is not valid json: " + String(text))


struct ApiRequestLike(Movable):
    """What a test wants to look at, flattened out of an `ApiRequest`.

    An `ApiRequest` borrows nothing, but the document it was parsed from has to
    stay alive for the parse itself, so the helpers above build one of these and
    let the document go. It is a test fixture and not a design.
    """

    var stream: Bool
    var max_tokens: Int
    var model: String
    var have_model: Bool
    var prompt: String
    var prompts: Int
    var temperature: Float32
    var stops: Int
    var biases: Int
    var echo: Bool
    var uses_ids: Bool

    def __init__(
        out self,
        stream: Bool,
        max_tokens: Int,
        var model: String,
        have_model: Bool,
        var prompt: String,
        prompts: Int,
        temperature: Float32,
        stops: Int,
        biases: Int,
        echo: Bool,
        uses_ids: Bool,
    ):
        self.stream = stream
        self.max_tokens = max_tokens
        self.model = model^
        self.have_model = have_model
        self.prompt = prompt^
        self.prompts = prompts
        self.temperature = temperature
        self.stops = stops
        self.biases = biases
        self.echo = echo
        self.uses_ids = uses_ids


def _chat(text: StringSpan) raises -> ApiRequestLike:
    var body = Body(text)
    var req = parse_chat(body.doc, body.scratch)
    return ApiRequestLike(
        req.stream,
        req.max_tokens,
        req.model,
        req.have_model,
        req.messages_json,
        req.prompts(),
        req.sampling.temperature,
        len(req.stops),
        len(req.bias_ids),
        req.echo,
        req.uses_ids,
    )


def _completions(text: StringSpan) raises -> ApiRequestLike:
    var body = Body(text)
    var req = parse_completions(body.doc)
    var first = String("")
    if len(req.texts) > 0:
        first = req.texts[0]
    var out = ApiRequestLike(
        req.stream,
        req.max_tokens,
        req.model,
        req.have_model,
        first,
        req.prompts(),
        req.sampling.temperature,
        len(req.stops),
        len(req.bias_ids),
        req.echo,
        req.uses_ids,
    )
    return out^


def _refused(text: StringSpan, chat: Bool) -> Bool:
    """True when the parse raised, which is what a 400 is built from."""
    try:
        var body = Body(text)
        if chat:
            _ = parse_chat(body.doc, body.scratch)
        else:
            _ = parse_completions(body.doc)
        return False
    except:
        return True


def test_parse_chat(mut suite: Suite) raises:
    suite.group("parsing a chat request")

    var plain = _chat('{"messages":[{"role":"user","content":"hi"}]}')
    suite.check(not plain.stream, "stream defaults to off")
    suite.check(
        plain.max_tokens == DEFAULT_MAX_TOKENS, "and max_tokens to a limit"
    )
    suite.check(not plain.have_model, "a body with no model says so")
    suite.check(plain.temperature == 1.0, "temperature defaults to OpenAI's 1")
    suite.check(
        '"role":"user"' in plain.prompt,
        "the messages come back out as json for the template",
    )

    var full = _chat(
        '{"model":"m","stream":true,"max_tokens":7,"temperature":0.2,'
        '"top_p":0.9,"stop":["a","b"],"logit_bias":{"5":1.5},'
        '"messages":[{"role":"system","content":"s"},'
        '{"role":"user","content":"u"}]}'
    )
    suite.check(full.stream, "stream is read")
    suite.check(full.max_tokens == 7, "max_tokens is read")
    suite.check(full.model == "m", "and the model name")
    suite.check(full.have_model, "which is known to have been given")
    suite.check(full.temperature == 0.2, "temperature is read")
    suite.check(full.stops == 2, "both stop strings are kept")
    suite.check(full.biases == 1, "and the logit bias")

    var newer = _chat(
        '{"max_completion_tokens":11,'
        '"messages":[{"role":"user","content":"hi"}]}'
    )
    suite.check(newer.max_tokens == 11, "max_completion_tokens is the new name")

    var one_stop = _chat(
        '{"stop":"END","messages":[{"role":"user","content":"hi"}]}'
    )
    suite.check(one_stop.stops == 1, "a bare string is one stop")


def test_parse_completions(mut suite: Suite) raises:
    suite.group("parsing a completions request")

    var one = _completions('{"prompt":"hello"}')
    suite.check(one.prompts == 1, "a string is one prompt")
    suite.check(one.prompt == "hello", "and it is the string")
    suite.check(not one.uses_ids, "which is text and not token ids")

    var many = _completions('{"prompt":["a","b","c"]}')
    suite.check(many.prompts == 3, "an array of strings is three prompts")

    var ids = _completions('{"prompt":[1,2,3]}')
    suite.check(ids.prompts == 1, "an array of ints is one prompt")
    suite.check(ids.uses_ids, "already tokenized")

    var id_lists = _completions('{"prompt":[[1,2],[3,4]]}')
    suite.check(id_lists.prompts == 2, "an array of arrays is two of those")
    suite.check(id_lists.uses_ids, "also already tokenized")

    var echo = _completions('{"prompt":"x","echo":true}')
    suite.check(echo.echo, "echo is read")

    suite.check(
        _refused('{"prompt":["a",1]}', False),
        "a prompt array that mixes strings and ids is refused",
    )
    suite.check(_refused("{}", False), "and a request with no prompt at all")
    suite.check(_refused('{"prompt":[]}', False), "and an empty prompt array")


def test_parse_refusals(mut suite: Suite) raises:
    suite.group("fields this build does not implement are refused")

    var head = '{"messages":[{"role":"user","content":"hi"}],'
    suite.check(_refused(head + '"tools":[]}', True), "tools")
    suite.check(_refused(head + '"functions":[]}', True), "functions")
    suite.check(_refused(head + '"tool_choice":"auto"}', True), "tool_choice")
    suite.check(
        _refused(head + '"response_format":{"type":"json_object"}}', True),
        "response_format",
    )
    suite.check(_refused(head + '"logprobs":true}', True), "logprobs")
    suite.check(_refused(head + '"top_logprobs":3}', True), "top_logprobs")
    suite.check(_refused(head + '"n":2}', True), "n above one")
    suite.check(
        _refused('{"prompt":"x","best_of":2}', False), "best_of on completions"
    )
    suite.check(
        _refused('{"prompt":"x","suffix":"y"}', False), "suffix on completions"
    )

    # False is not a request for something, so it is not a refusal. A client
    # library that sends `logprobs: false` by default would otherwise be unable
    # to talk to this server at all.
    suite.check(
        not _refused(head + '"logprobs":false}', True),
        "but logprobs false is not asking for anything",
    )
    suite.check(
        not _refused(head + '"n":1}', True), "and n of one is what we do anyway"
    )

    suite.check(
        _refused(head + '"temperature":5}', True),
        "a temperature outside OpenAI's range is refused",
    )
    suite.check(
        _refused(head + '"top_p":2}', True), "and a top_p outside its range"
    )
    suite.check(
        _refused('{"messages":[]}', True), "and a chat with no messages at all"
    )
    suite.check(_refused("[1,2]", True), "and a body that is not an object")


def test_write_responses(mut suite: Suite) raises:
    suite.group("building openai responses")

    var w = Writer(0, 8192)

    suite.check(
        write_error(w, "no", "invalid_request_error", ""), "an error is built"
    )
    var err = _text_of(w)
    suite.check('"error"' in err, "wrapped in an error object")
    suite.check('"message":"no"' in err, "with the message")
    suite.check('"param":null' in err, "and param written out as null")
    suite.check('"code":null' in err, "and code too")

    suite.check(write_models(w, "m", 7, False), "a model list is built")
    var list = _text_of(w)
    suite.check('"object":"list"' in list, "as a list")
    suite.check('"data":[' in list, "with a data array")
    suite.check('"owned_by":"molla"' in list, "and an owner")

    suite.check(write_models(w, "m", 7, True), "and one model on its own")
    suite.check('"object":"model"' in _text_of(w), "without the list envelope")

    suite.check(
        write_chat_body(w, "id", 7, "m", "hello", FINISH_STOP, 3, 2),
        "a whole chat completion is built",
    )
    var chat = _text_of(w)
    suite.check('"object":"chat.completion"' in chat, "named correctly")
    suite.check('"role":"assistant"' in chat, "with an assistant message")
    suite.check('"content":"hello"' in chat, "carrying the text")
    suite.check('"finish_reason":"stop"' in chat, "and why it stopped")
    suite.check('"total_tokens":5' in chat, "and usage that adds up")

    suite.check(
        write_chat_body(w, "id", 7, "m", "x", FINISH_LENGTH, 1, 1),
        "a completion cut off by the limit is built",
    )
    suite.check(
        '"finish_reason":"length"' in _text_of(w), "and says length instead"
    )


def test_write_chunks(mut suite: Suite) raises:
    suite.group("building streaming chunks")

    var w = Writer(0, 8192)
    var text = String(" there")

    suite.check(
        write_chat_chunk(w, "i", 7, "m", True, "".as_bytes(), False, 0, 0, 0),
        "the opening chunk is built",
    )
    var first = _text_of(w)
    suite.check('"object":"chat.completion.chunk"' in first, "named correctly")
    suite.check('"role":"assistant"' in first, "carrying the role")
    suite.check('"finish_reason":null' in first, "with a null finish reason")
    suite.check('"usage"' not in first, "and no usage yet")

    suite.check(
        write_chat_chunk(
            w, "i", 7, "m", False, text.as_bytes(), False, 0, 0, 0
        ),
        "a token chunk is built",
    )
    var middle = _text_of(w)
    suite.check('"content":" there"' in middle, "carrying the text")
    suite.check('"role"' not in middle, "and no role after the first")
    suite.check('"finish_reason":null' in middle, "still a null finish reason")

    suite.check(
        write_chat_chunk(
            w, "i", 7, "m", False, "".as_bytes(), True, FINISH_STOP, 4, 6
        ),
        "the closing chunk is built",
    )
    var last = _text_of(w)
    suite.check('"delta":{}' in last, "with an empty delta")
    suite.check('"finish_reason":"stop"' in last, "and a real finish reason")
    suite.check('"total_tokens":10' in last, "and the usage")

    suite.check(
        write_text_chunk(w, "i", 7, "m", text.as_bytes(), False, 0, 0, 0),
        "a completions chunk is built",
    )
    var textual = _text_of(w)
    # Not `text_completion.chunk`. See the note on `write_text_chunk`.
    suite.check('"object":"text_completion"' in textual, "named as the SDK")
    suite.check('"text":" there"' in textual, "carrying the text")
    suite.check('"logprobs":null' in textual, "with logprobs written as null")

    suite.check(begin_text_body(w, "i", 7, "m"), "a text body opens")
    suite.check(add_text_choice(w, "a", 0, FINISH_STOP), "a first choice")
    suite.check(add_text_choice(w, "b", 1, FINISH_LENGTH), "a second choice")
    suite.check(end_text_body(w, 2, 3), "and it closes")
    var body = _text_of(w)
    suite.check('"index":0' in body, "the first choice is numbered")
    suite.check('"index":1' in body, "and so is the second")
    suite.check('"total_tokens":5' in body, "with usage over both")


def _client(port: UInt16) raises -> Int:
    var fd = socket_tcp()
    _ = connect(fd, INADDR_LOOPBACK, port)
    return fd


def _send_text(fd: Int, text: StringSpan) -> Int:
    var n = text.byte_length()
    var p = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(text.unsafe_ptr())
    )
    var sent = 0
    while sent < n:
        var wrote = send(fd, p.unsafe_offset(sent), n - sent)
        if wrote <= 0:
            break
        sent += wrote
    return sent


def _read(
    fd: Int, mut reactor: Reactor[HttpProtocol], want: Int
) raises -> String:
    var out = String("")
    var buf = stack_allocation[8192, UInt8]()
    var total = 0
    for _ in range(MAX_STEPS):
        _ = reactor.poll_once(1)
        var got = recv(fd, buf, 8192)
        if got > 0:
            total += got
            for i in range(got):
                out += chr(Int(buf.unsafe_load(i)))
        if total >= want:
            break
    return out


def _post(target: StringSpan, body: StringSpan) -> String:
    return (
        String("POST ")
        + String(target)
        + " HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        + "Content-Length: "
        + String(body.byte_length())
        + "\r\n\r\n"
        + String(body)
    )


def test_routes(mut suite: Suite) raises:
    """The API routes on a server that was started without a model.

    A 503 rather than a 404 is the whole point of the test. It says the target
    resolved, the method check ran, and the handler is the API handler, which
    is everything about the dispatch that a model would not tell us anything
    more about.
    """
    suite.group("openai routes without a model")

    var listener = open_listener(ListenAddress(UInt16(0)), False)
    var port = bound_port(listener)
    var reactor = Reactor[HttpProtocol](HttpProtocol(), 60000, 0)
    reactor.add_listener(listener)
    var client = _client(port)
    for _ in range(MAX_STEPS):
        _ = reactor.poll_once(1)
        if reactor.accepted >= 1:
            break

    _ = _send_text(
        client, _post("/v1/chat/completions", '{"messages":[{"a":1}]}')
    )
    var chat = _read(client, reactor, 40)
    suite.check(chat.startswith("HTTP/1.1 503"), "chat says it has no model")
    suite.check('"type":"server_error"' in chat, "in the error envelope")
    suite.check(
        "application/json" in chat, "and it is served as json, not as text"
    )

    _ = _send_text(client, _post("/v1/completions", '{"prompt":"x"}'))
    var text = _read(client, reactor, 40)
    suite.check(text.startswith("HTTP/1.1 503"), "and so does completions")

    _ = _send_text(client, "GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n")
    var models = _read(client, reactor, 40)
    suite.check(models.startswith("HTTP/1.1 503"), "and so does the model list")

    # A GET on a completion is a 405 before anything looks for a model, because
    # the method is wrong whatever the server was started with.
    _ = _send_text(
        client, "GET /v1/chat/completions HTTP/1.1\r\nHost: x\r\n\r\n"
    )
    var wrong = _read(client, reactor, 40)
    suite.check(
        wrong.startswith("HTTP/1.1 405"), "a GET on a completion is 405"
    )
    suite.check("a completion is a POST" in wrong, "and says why")

    _ = _send_text(client, "GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n")
    var health = _read(client, reactor, 40)
    suite.check(
        health.startswith("HTTP/1.1 200"),
        "and the routes that were here before are unaffected",
    )

    # Last, because a 404 is a refusal and a refusal closes the connection.
    _ = _send_text(client, "GET /v1/nope HTTP/1.1\r\nHost: x\r\n\r\n")
    var missing = _read(client, reactor, 40)
    suite.check(
        missing.startswith("HTTP/1.1 404"), "an unknown v1 route is still a 404"
    )

    _ = close(client)
    reactor.shutdown()


def run(mut suite: Suite):
    try:
        test_parse_chat(suite)
        test_parse_completions(suite)
        test_parse_refusals(suite)
        test_write_responses(suite)
        test_write_chunks(suite)
        test_routes(suite)
    except e:
        suite.fail("test_api raised", String(e))
