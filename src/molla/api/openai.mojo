"""The OpenAI request and response bodies, read and written.

Two halves. Reading turns a request body into an `ApiRequest`, which is the set
of things molla can actually act on, and refuses anything else. Writing turns a
result into the exact JSON the OpenAI clients expect, key by key, including the
keys whose value is always null.

## Refusing rather than ignoring

A field this build does not implement is an error and not a shrug. `tools`,
`response_format` and `logprobs` are real features with real behaviour, and a
server that accepts a request carrying one of them and answers as though it were
not there is telling the caller their tool call was considered and not taken.
That failure is silent, it looks like the model being bad at tool use, and it
costs somebody an afternoon. So the fields that are not implemented are named,
and naming them is why the list is written out rather than being a default of
ignoring the unknown.

Fields that carry no behaviour are ignored, because refusing those breaks
clients for nothing. `user` is a label for somebody else's billing, `stream_options`
asks for usage that is sent anyway, and neither changes a token.

## Nulls are written

`param` and `code` in the error envelope, and `logprobs` in a completion choice,
are written as null rather than left out. The OpenAI SDKs model them as optional
fields and both spellings decode, but the shape people compare against when they
are debugging is the one in the documentation, and that one has the keys.

## Defaults are OpenAI's, not molla's

A request with no temperature gets one, because that is what the documentation
says the default is and it is what the caller is expecting to have happened.
`molla generate` defaults to greedy instead, and the difference is deliberate:
the command line is a tool for looking at a model, where a second run producing
different text is a nuisance, and the API is an implementation of a specification
somebody else wrote.
"""

from molla.engine.sample import GREEDY, SamplerConfig
from molla.json.dom import (
    JS_ARRAY,
    JS_BOOL,
    JS_DOUBLE,
    JS_INT,
    JS_NULL,
    JS_OBJECT,
    JS_STRING,
    NO_NODE,
    Document,
)
from molla.json.serialize import Writer

comptime DEFAULT_MAX_TOKENS = 256
"""What a request with no `max_tokens` gets.

OpenAI's answer is the rest of the context window, which on a model with a
hundred and thirty thousand positions and a scalar decode is a request that
never finishes. A number small enough to come back, large enough for an answer,
and printed in the docs is the honest version until there is a scheduler.
"""

comptime MAX_STOPS = 4
"""What OpenAI allows, and what is allowed here."""


struct ApiRequest(Movable):
    """One chat or completions request, reduced to what the engine can act on.
    """

    var chat: Bool
    """A chat request renders messages through the model's template. A
    completions request takes the prompt as it arrives."""

    var model: String
    var have_model: Bool
    var stream: Bool
    var max_tokens: Int
    var echo: Bool
    """Completions only. Put the prompt in front of what was generated."""

    var stops: List[String]
    var sampling: SamplerConfig
    var bias_ids: List[Int]
    var bias_vals: List[Float32]
    var have_seed: Bool
    """Whether the caller pinned the seed. Without one the server picks, so
    that two identical requests are two samples rather than one twice."""

    var messages_json: String
    """The `messages` array as JSON, to be handed to the chat template. Kept as
    text rather than as a parsed structure because that is the shape
    `Template.render_object` wants and because molla has no business having an
    opinion about what a message contains."""

    var texts: List[String]
    """Completions prompts, when they arrived as text."""

    var id_prompts: List[List[Int]]
    """Completions prompts, when they arrived as token ids."""

    var uses_ids: Bool

    def __init__(out self, chat: Bool):
        self.chat = chat
        self.model = String("")
        self.have_model = False
        self.stream = False
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.echo = False
        self.stops = List[String]()
        self.sampling = SamplerConfig()
        # OpenAI's default, not molla's. See the module docstring.
        self.sampling.temperature = 1.0
        self.bias_ids = List[Int]()
        self.bias_vals = List[Float32]()
        self.have_seed = False
        self.messages_json = String("")
        self.texts = List[String]()
        self.id_prompts = List[List[Int]]()
        self.uses_ids = False

    def prompts(self) -> Int:
        """How many completions this request asks for."""
        if self.chat:
            return 1
        return len(self.id_prompts) if self.uses_ids else len(self.texts)


def _text(doc: Document, node: Int) -> String:
    """A string node as an owned `String`."""
    return String(StringSpan(unsafe_from_utf8=doc.text(node)))


def _refuse_unsupported(doc: Document, root: Int, chat: Bool) raises:
    """Name every field that would change the answer and does not.

    Ordered so the first thing a caller hears about is the thing they most
    likely sent on purpose.
    """
    if doc.get(root, "tools") != NO_NODE:
        raise Error(
            "tools are not implemented in this build, and answering as though"
            " the request had not asked for them would look like the model"
            " deciding not to call one"
        )
    if doc.get(root, "functions") != NO_NODE:
        raise Error("functions are not implemented in this build")
    if doc.get(root, "tool_choice") != NO_NODE:
        raise Error("tool_choice is not implemented in this build")
    if doc.get(root, "response_format") != NO_NODE:
        raise Error(
            "response_format is not implemented in this build, so a request"
            " for json would get prose and no error"
        )
    var logprobs = doc.get(root, "logprobs")
    if logprobs != NO_NODE and doc.kind(logprobs) != JS_NULL:
        if not (doc.kind(logprobs) == JS_BOOL and not doc.as_bool(logprobs)):
            raise Error("logprobs are not implemented in this build")
    if doc.get(root, "top_logprobs") != NO_NODE:
        raise Error("top_logprobs are not implemented in this build")
    if not chat:
        if doc.get(root, "suffix") != NO_NODE:
            raise Error("suffix is not implemented in this build")
        if doc.get(root, "best_of") != NO_NODE:
            raise Error("best_of is not implemented in this build")
    var n = doc.get(root, "n")
    if n != NO_NODE and doc.kind(n) != JS_NULL and doc.as_int(n, 1) != 1:
        raise Error(
            "this build answers one completion per prompt, so n has to be one"
        )


def _read_stops(doc: Document, root: Int, mut req: ApiRequest) raises:
    """`stop` as a string or as an array of up to four of them."""
    var node = doc.get(root, "stop")
    if node == NO_NODE or doc.kind(node) == JS_NULL:
        return
    if doc.kind(node) == JS_STRING:
        req.stops.append(_text(doc, node))
        return
    if doc.kind(node) != JS_ARRAY:
        raise Error("stop has to be a string or an array of strings")
    var child = doc.first_child(node)
    while child != NO_NODE:
        if doc.kind(child) != JS_STRING:
            raise Error("every entry in stop has to be a string")
        if len(req.stops) >= MAX_STOPS:
            raise Error("stop takes at most " + String(MAX_STOPS) + " strings")
        req.stops.append(_text(doc, child))
        child = doc.next_sibling(child)


def _read_bias(doc: Document, root: Int, mut req: ApiRequest) raises:
    """`logit_bias`, an object whose keys are token ids written as strings."""
    var node = doc.get(root, "logit_bias")
    if node == NO_NODE or doc.kind(node) == JS_NULL:
        return
    if doc.kind(node) != JS_OBJECT:
        raise Error("logit_bias has to be an object keyed by token id")
    var child = doc.first_child(node)
    while child != NO_NODE:
        var key = String(StringSpan(unsafe_from_utf8=doc.key(child)))
        var id: Int
        try:
            id = atol(key)
        except:
            raise Error("'" + key + "' is a logit_bias key and not a token id")
        if doc.kind(child) != JS_INT and doc.kind(child) != JS_DOUBLE:
            raise Error("a logit_bias value has to be a number")
        req.bias_ids.append(id)
        req.bias_vals.append(Float32(doc.as_double(child, 0.0)))
        child = doc.next_sibling(child)


def _read_sampling(doc: Document, root: Int, mut req: ApiRequest) raises:
    """The settings that are the same on both routes.

    `top_k`, `min_p`, `typical_p`, `repeat_penalty` and `repeat_last_n` are not
    OpenAI's. They are read because every local runner accepts them and every
    client that talks to one sends them, and leaving them out would mean a
    request that works against llama.cpp silently samples differently here.
    """
    var temp = doc.get(root, "temperature")
    if temp != NO_NODE and doc.kind(temp) != JS_NULL:
        req.sampling.temperature = Float32(doc.as_double(temp, 1.0))
    var top_p = doc.get(root, "top_p")
    if top_p != NO_NODE and doc.kind(top_p) != JS_NULL:
        req.sampling.top_p = Float32(doc.as_double(top_p, 1.0))
    var top_k = doc.get(root, "top_k")
    if top_k != NO_NODE and doc.kind(top_k) != JS_NULL:
        req.sampling.top_k = doc.as_int(top_k, 0)
    var min_p = doc.get(root, "min_p")
    if min_p != NO_NODE and doc.kind(min_p) != JS_NULL:
        req.sampling.min_p = Float32(doc.as_double(min_p, 0.0))
    var typical = doc.get(root, "typical_p")
    if typical != NO_NODE and doc.kind(typical) != JS_NULL:
        req.sampling.typical_p = Float32(doc.as_double(typical, 1.0))
    var repeat = doc.get(root, "repeat_penalty")
    if repeat != NO_NODE and doc.kind(repeat) != JS_NULL:
        req.sampling.repeat_penalty = Float32(doc.as_double(repeat, 1.0))
    var window = doc.get(root, "repeat_last_n")
    if window != NO_NODE and doc.kind(window) != JS_NULL:
        req.sampling.repeat_last_n = doc.as_int(window, 64)
    var freq = doc.get(root, "frequency_penalty")
    if freq != NO_NODE and doc.kind(freq) != JS_NULL:
        req.sampling.frequency_penalty = Float32(doc.as_double(freq, 0.0))
    var presence = doc.get(root, "presence_penalty")
    if presence != NO_NODE and doc.kind(presence) != JS_NULL:
        req.sampling.presence_penalty = Float32(doc.as_double(presence, 0.0))
    var seed = doc.get(root, "seed")
    if seed != NO_NODE and doc.kind(seed) != JS_NULL:
        var value = doc.as_int(seed, 0)
        if value < 0:
            raise Error("a seed cannot be negative")
        req.sampling.seed = UInt64(value)
        req.have_seed = True

    # OpenAI's penalties run from minus two to two and its temperature from
    # zero to two. Checked here rather than in the sampler, because these are
    # the API's limits and the sampler's are the arithmetic's.
    if req.sampling.temperature < 0 or req.sampling.temperature > 2:
        raise Error("temperature has to be between zero and two")
    if (
        req.sampling.frequency_penalty < -2
        or req.sampling.frequency_penalty > 2
    ):
        raise Error("frequency_penalty has to be between minus two and two")
    if req.sampling.presence_penalty < -2 or req.sampling.presence_penalty > 2:
        raise Error("presence_penalty has to be between minus two and two")
    req.sampling.check()


def _read_common(doc: Document, root: Int, mut req: ApiRequest) raises:
    var model = doc.get(root, "model")
    if model != NO_NODE and doc.kind(model) == JS_STRING:
        req.model = _text(doc, model)
        req.have_model = True
    var stream = doc.get(root, "stream")
    if stream != NO_NODE and doc.kind(stream) != JS_NULL:
        if doc.kind(stream) != JS_BOOL:
            raise Error("stream has to be true or false")
        req.stream = doc.as_bool(stream)

    # `max_completion_tokens` is what the field is called now and `max_tokens`
    # is what every client still sends, so both are read and the newer one
    # wins.
    var limit = doc.get(root, "max_tokens")
    var newer = doc.get(root, "max_completion_tokens")
    if newer != NO_NODE and doc.kind(newer) != JS_NULL:
        limit = newer
    if limit != NO_NODE and doc.kind(limit) != JS_NULL:
        var value = doc.as_int(limit, DEFAULT_MAX_TOKENS)
        if value <= 0:
            raise Error("max_tokens has to be at least one")
        req.max_tokens = value

    _read_stops(doc, root, req)
    _read_bias(doc, root, req)
    _read_sampling(doc, root, req)


def _copy_value(doc: Document, node: Int, mut w: Writer) -> Bool:
    """Write one parsed value back out as JSON.

    The chat template takes its variables as a JSON object, and the messages
    are already a parsed subtree of the request, so they go back through the
    writer rather than being sliced out of the request bytes by hand. The
    round trip is also what escapes them correctly: a message whose content
    contained a quote arrived decoded and has to leave encoded again.
    """
    var kind = doc.kind(node)
    if kind == JS_OBJECT:
        if not w.begin_object():
            return False
        var child = doc.first_child(node)
        while child != NO_NODE:
            if not w.key_bytes(doc.key(child)):
                return False
            if not _copy_value(doc, child, w):
                return False
            child = doc.next_sibling(child)
        return w.end_object()
    if kind == JS_ARRAY:
        if not w.begin_array():
            return False
        var child = doc.first_child(node)
        while child != NO_NODE:
            if not _copy_value(doc, child, w):
                return False
            child = doc.next_sibling(child)
        return w.end_array()
    if kind == JS_STRING:
        return w.string_bytes(doc.text(node))
    if kind == JS_INT:
        return w.int(doc.as_int(node))
    if kind == JS_DOUBLE:
        return w.double(doc.as_double(node))
    if kind == JS_BOOL:
        return w.bool(doc.as_bool(node))
    return w.null()


def parse_chat(doc: Document, mut w: Writer) raises -> ApiRequest:
    """A `/v1/chat/completions` body. `w` is scratch for the messages copy."""
    var root = doc.root
    if doc.kind(root) != JS_OBJECT:
        raise Error("a chat request has to be a json object")
    var req = ApiRequest(True)
    _refuse_unsupported(doc, root, True)
    _read_common(doc, root, req)

    var messages = doc.get(root, "messages")
    if messages == NO_NODE:
        raise Error("a chat request needs messages")
    if doc.kind(messages) != JS_ARRAY:
        raise Error("messages has to be an array")
    if doc.size(messages) == 0:
        raise Error("a chat request needs at least one message")
    w.reset()
    if not _copy_value(doc, messages, w) or not w.complete():
        raise Error("the messages did not fit in the buffer for them")
    req.messages_json = String(StringSpan(unsafe_from_utf8=w.bytes()))
    return req^


def _read_id_prompt(doc: Document, node: Int) raises -> List[Int]:
    var ids = List[Int]()
    var child = doc.first_child(node)
    while child != NO_NODE:
        if doc.kind(child) != JS_INT:
            raise Error("a token id prompt has to be an array of integers")
        ids.append(doc.as_int(child))
        child = doc.next_sibling(child)
    if len(ids) == 0:
        raise Error("a prompt with no tokens has nothing to continue")
    return ids^


def parse_completions(doc: Document) raises -> ApiRequest:
    """A `/v1/completions` body.

    The prompt has four spellings and all four are here, because the two that
    are not a plain string are what evaluation harnesses send. A string is one
    prompt, an array of strings is several, an array of integers is one prompt
    already tokenized, and an array of arrays of integers is several of those.
    Telling the last two apart is what the first element is, which is also how
    the reference implementation does it.
    """
    var root = doc.root
    if doc.kind(root) != JS_OBJECT:
        raise Error("a completions request has to be a json object")
    var req = ApiRequest(False)
    _refuse_unsupported(doc, root, False)
    _read_common(doc, root, req)
    req.echo = doc.get_bool(root, "echo", False)

    var prompt = doc.get(root, "prompt")
    if prompt == NO_NODE or doc.kind(prompt) == JS_NULL:
        raise Error("a completions request needs a prompt")
    if doc.kind(prompt) == JS_STRING:
        req.texts.append(_text(doc, prompt))
        return req^
    if doc.kind(prompt) != JS_ARRAY:
        raise Error(
            "prompt has to be a string, an array of strings, or an array of"
            " token ids"
        )
    var first = doc.first_child(prompt)
    if first == NO_NODE:
        raise Error("a prompt with nothing in it has nothing to continue")
    if doc.kind(first) == JS_INT:
        req.uses_ids = True
        req.id_prompts.append(_read_id_prompt(doc, prompt))
        return req^
    var child = first
    while child != NO_NODE:
        if doc.kind(child) == JS_STRING:
            if req.uses_ids:
                raise Error(
                    "a prompt array holds either strings or token ids, not both"
                )
            req.texts.append(_text(doc, child))
        elif doc.kind(child) == JS_ARRAY:
            if len(req.texts) > 0:
                raise Error(
                    "a prompt array holds either strings or token ids, not both"
                )
            req.uses_ids = True
            req.id_prompts.append(_read_id_prompt(doc, child))
        else:
            raise Error(
                "a prompt array holds strings, token id arrays, or token ids"
            )
        child = doc.next_sibling(child)
    return req^


comptime FINISH_STOP = 0
comptime FINISH_LENGTH = 1


def finish_name(reason: Int) -> StaticString:
    return "length" if reason == FINISH_LENGTH else "stop"


def write_error(
    mut w: Writer, message: StringSpan, kind: StringSpan, code: StringSpan
) -> Bool:
    """The error envelope, with the nulls written out.

    `{"error": {"message": ..., "type": ..., "param": ..., "code": ...}}`, which
    is the shape the SDKs raise from and the shape people paste into issues.
    `param` is always null here because nothing on this path can attribute a
    failure to one named field with enough confidence to say so.
    """
    w.reset()
    if not w.begin_object():
        return False
    if not w.key("error") or not w.begin_object():
        return False
    if not w.field_str("message", message):
        return False
    if not w.field_str("type", kind):
        return False
    if not w.field_null("param"):
        return False
    if code.byte_length() == 0:
        if not w.field_null("code"):
            return False
    elif not w.field_str("code", code):
        return False
    if not w.end_object() or not w.end_object():
        return False
    return w.complete()


def write_models(
    mut w: Writer, id: StringSpan, created: Int, one: Bool
) -> Bool:
    """`/v1/models` and `/v1/models/{id}`, which differ by the envelope.

    One model, because this build loads one file and there is nothing honest to
    put in a longer list. The id is the whole reference the server was started
    with rather than a short name, so that a client which round trips it back
    in the `model` field of a request gets a match.
    """
    w.reset()
    if not one:
        if not w.begin_object():
            return False
        if not w.field_str("object", "list"):
            return False
        if not w.key("data") or not w.begin_array():
            return False
    if not w.begin_object():
        return False
    if not w.field_str("id", id):
        return False
    if not w.field_str("object", "model"):
        return False
    if not w.field_int("created", created):
        return False
    if not w.field_str("owned_by", "molla"):
        return False
    if not w.end_object():
        return False
    if not one:
        if not w.end_array() or not w.end_object():
            return False
    return w.complete()


def _write_usage(mut w: Writer, prompt: Int, completion: Int) -> Bool:
    if not w.key("usage") or not w.begin_object():
        return False
    if not w.field_int("prompt_tokens", prompt):
        return False
    if not w.field_int("completion_tokens", completion):
        return False
    if not w.field_int("total_tokens", prompt + completion):
        return False
    return w.end_object()


def _write_head(
    mut w: Writer,
    id: StringSpan,
    object: StringSpan,
    created: Int,
    model: StringSpan,
) -> Bool:
    if not w.begin_object():
        return False
    if not w.field_str("id", id):
        return False
    if not w.field_str("object", object):
        return False
    if not w.field_int("created", created):
        return False
    if not w.field_str("model", model):
        return False
    return w.key("choices") and w.begin_array()


def write_chat_body(
    mut w: Writer,
    id: StringSpan,
    created: Int,
    model: StringSpan,
    content: StringSpan,
    reason: Int,
    prompt_tokens: Int,
    completion_tokens: Int,
) -> Bool:
    """A whole `chat.completion`, for a request that did not ask to stream."""
    w.reset()
    if not _write_head(w, id, "chat.completion", created, model):
        return False
    if not w.begin_object():
        return False
    if not w.field_int("index", 0):
        return False
    if not w.key("message") or not w.begin_object():
        return False
    if not w.field_str("role", "assistant"):
        return False
    if not w.field_str("content", content):
        return False
    if not w.end_object():
        return False
    if not w.field_str("finish_reason", finish_name(reason)):
        return False
    if not w.end_object() or not w.end_array():
        return False
    if not _write_usage(w, prompt_tokens, completion_tokens):
        return False
    if not w.end_object():
        return False
    return w.complete()


def write_chat_chunk(
    mut w: Writer,
    id: StringSpan,
    created: Int,
    model: StringSpan,
    role: Bool,
    content: Span[UInt8, _],
    last: Bool,
    reason: Int,
    prompt_tokens: Int,
    completion_tokens: Int,
) -> Bool:
    """One `chat.completion.chunk`.

    Three shapes go through here and they are the three the specification has.
    The first chunk carries the role and no text, every chunk after it carries
    text and a null finish reason, and the last one carries an empty delta, the
    reason, and the usage.
    """
    w.reset()
    if not _write_head(w, id, "chat.completion.chunk", created, model):
        return False
    if not w.begin_object():
        return False
    if not w.field_int("index", 0):
        return False
    if not w.key("delta") or not w.begin_object():
        return False
    if role:
        if not w.field_str("role", "assistant"):
            return False
        if not w.field_str("content", ""):
            return False
    elif not last:
        if not w.key("content") or not w.string_bytes(content):
            return False
    if not w.end_object():
        return False
    if last:
        if not w.field_str("finish_reason", finish_name(reason)):
            return False
    elif not w.field_null("finish_reason"):
        return False
    if not w.end_object() or not w.end_array():
        return False
    if last:
        if not _write_usage(w, prompt_tokens, completion_tokens):
            return False
    if not w.end_object():
        return False
    return w.complete()


def _write_text_choice(
    mut w: Writer, text: Span[UInt8, _], index: Int, reason: Int, last: Bool
) -> Bool:
    if not w.begin_object():
        return False
    if not w.key("text") or not w.string_bytes(text):
        return False
    if not w.field_int("index", index):
        return False
    if not w.field_null("logprobs"):
        return False
    if last:
        if not w.field_str("finish_reason", finish_name(reason)):
            return False
    elif not w.field_null("finish_reason"):
        return False
    return w.end_object()


def begin_text_body(
    mut w: Writer, id: StringSpan, created: Int, model: StringSpan
) -> Bool:
    """Open a `text_completion`, for the caller to append choices to.

    Open rather than whole, because a completions request can carry several
    prompts and each one is generated in turn. The choices go in with
    `add_text_choice` and `end_text_body` closes it.
    """
    w.reset()
    return _write_head(w, id, "text_completion", created, model)


def add_text_choice(
    mut w: Writer, text: StringSpan, index: Int, reason: Int
) -> Bool:
    return _write_text_choice(w, text.as_bytes(), index, reason, True)


def end_text_body(
    mut w: Writer, prompt_tokens: Int, completion_tokens: Int
) -> Bool:
    if not w.end_array():
        return False
    if not _write_usage(w, prompt_tokens, completion_tokens):
        return False
    if not w.end_object():
        return False
    return w.complete()


def write_text_chunk(
    mut w: Writer,
    id: StringSpan,
    created: Int,
    model: StringSpan,
    content: Span[UInt8, _],
    last: Bool,
    reason: Int,
    prompt_tokens: Int,
    completion_tokens: Int,
) -> Bool:
    """One streaming `text_completion` chunk.

    Note the object name. A streaming completions chunk is a `text_completion`
    and not a `text_completion.chunk`, which reads like an oversight in the
    specification and is what the clients check for.
    """
    w.reset()
    if not _write_head(w, id, "text_completion", created, model):
        return False
    if not _write_text_choice(w, content, 0, reason, last):
        return False
    if not w.end_array():
        return False
    if last:
        if not _write_usage(w, prompt_tokens, completion_tokens):
            return False
    if not w.end_object():
        return False
    return w.complete()
