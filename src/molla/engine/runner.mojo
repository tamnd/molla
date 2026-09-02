"""One loaded model, and the state of the one request that is using it.

Everything a route needs and nothing a route should know about. The protocol
holds the address of one of these and calls four things on it: render a prompt,
start a generation, take a token, and say what the last one produced. What a
GGUF is, what a template is, and how a token becomes bytes stay on this side of
that line.

## One sequence

There is one session, one sampler and one set of counters here, because M2
decodes one sequence at a time. A second request arriving while one is running
is refused rather than queued or interleaved, which is a worse server and an
honest one: queueing without a scheduler means a client waiting on a socket with
no idea it is second, and interleaving without paging means two sequences
writing over each other's cache. The scheduler is M3 and it is the thing that
makes this a `List` instead of a field.

`busy` is only ever observed across a streaming response, since a streaming
request goes back to the event loop between tokens and another connection can be
serviced in the gap. A request that is not streaming holds the worker for its
whole generation, which is the same statement said less politely.

## Stop strings are held, not searched afterwards

A stop string can straddle two tokens, and it can be half emitted before it is
known to be one. So the generated text is kept whole here, and what a caller may
send is the part of it that cannot still turn out to be the beginning of a stop
string. That means a token is sometimes produced and nothing goes out for it,
which is correct, and it is why the streaming loop asks for a delta rather than
assuming one token is one chunk.
"""

from molla.engine.bind import Bound, bind
from molla.engine.sample import Sampler, SamplerConfig
from molla.engine.session import Session as Decode
from molla.jinja.template import Template
from molla.model.gguf import Gguf
from molla.model.load import Weights, load, plan_load
from molla.model.repack import RepackCache, model_key, open_cache
from molla.model.spec import read_geometry
from molla.sys.clock import unix_time
from molla.sys.device import default_device
from molla.sys.mem import AllocCounter
from molla.tokenizer.tokenizer import DecodeStream, Session, Tokenizer

comptime REASON_STOP = 0
comptime REASON_LENGTH = 1
"""Why a generation ended. The same two values `molla.api.openai` writes as
`stop` and `length`, spelled here so the engine does not import the API."""

comptime DEFAULT_CONTEXT = 4096
"""Positions to make room for when nobody says. The cache is four bytes an
element, so allocating whatever the file allows would be gigabytes for a
conversation of two lines."""

comptime RunnerPtr = Pointer[Runner, MutAnyOrigin]
"""How the protocol reaches the runner.

By address for the same reason the logger and the metrics view are: the protocol
lives inside a reactor, the reactors live in a list the server owns, and the
runner is a local of the function that started the server. Nothing here owns the
runner and nothing here outlives it.
"""


def runner_at(address: Int) -> RunnerPtr:
    """The runner an address names. Only ever called on a checked address."""
    return RunnerPtr(unsafe_from_address=address)


def address_of(ref runner: Runner) -> Int:
    return Int(Pointer(to=runner))


struct Runner(Movable):
    """A model file, ready to answer, plus whatever it is in the middle of."""

    var g: Gguf
    var weights: Weights
    var cache: RepackCache
    """The repacked weights beside the model, when there are any. Held for the
    same reason `g` is, which is that `b` points into it."""

    var b: Bound
    """Addresses inside the two mappings `g` and `cache` hold. Nothing here owns
    bytes, so those two outliving `b` is the whole of the lifetime rule."""

    var tokenizer: Tokenizer
    var counter: AllocCounter
    var chat: Template
    var has_chat: Bool
    """Whether the file carried a chat template. Without one the completions
    route still works and the chat route says why it does not."""

    var session: Decode
    var sampler: Sampler
    var decoder: DecodeStream

    var id: String
    """What `/v1/models` reports and what a request's `model` is matched
    against. The whole reference the server was given, so a client that round
    trips it gets a match."""

    var created: Int
    var eos: Int
    var context: Int
    var bos_text: String
    var eos_text: String

    var busy: Bool
    var left: Int
    var produced: Int
    var prompt_tokens: Int
    var reason: Int
    var text: String
    """Everything this generation has decoded, truncated at a stop string once
    one has been seen."""

    var emitted: Int
    """Bytes of `text` already handed to the client."""

    var stops: List[String]
    var seq: Int
    """Requests answered, which is what makes a response id unique."""

    def __init__(
        out self,
        model_path: String,
        tokenizer_path: String,
        id: String,
        context: Int,
    ) raises:
        var g = Gguf(model_path)
        var dev = default_device()

        # Everything stays in the mapping, for the reason `molla generate`
        # gives: the kernels are host kernels, so a tensor on a card is one
        # they cannot read.
        #
        # A miss repacks while it loads and a hit does not, so the first start
        # against a model pays once and every start after it binds straight to
        # the cache. This run binds to whatever was there when it opened, which
        # on a miss is the file, so the repack a miss writes is for the next
        # start and not for this one.
        var cache = open_cache(model_path, model_key(g))
        var repack_for = String("") if cache.usable else model_path
        var weights = load(g, plan_load(g, dev, 0), 0, False, repack_for)
        var b = bind(g, cache)
        var geometry = read_geometry(g)
        var want = context if context > 0 else DEFAULT_CONTEXT
        if geometry.context_length > 0 and want > geometry.context_length:
            want = geometry.context_length

        var counter = AllocCounter()
        var tokenizer = Tokenizer(tokenizer_path, counter.raw())
        var source = g.string_or("tokenizer.chat_template", "")
        self.has_chat = source.byte_length() > 0
        self.chat = Template(source)

        self.eos = g.uint_or("tokenizer.ggml.eos_token_id", -1)
        var bos = g.uint_or("tokenizer.ggml.bos_token_id", -1)
        self.bos_text = _token_text(tokenizer, bos)
        self.eos_text = _token_text(tokenizer, self.eos)

        self.session = Decode(b, want)
        self.sampler = Sampler(SamplerConfig(), b.vocab())
        self.decoder = DecodeStream(True)
        self.context = want
        self.g = g^
        self.weights = weights^
        self.cache = cache^
        self.b = b^
        self.tokenizer = tokenizer^
        self.counter = counter
        self.id = id
        self.created = unix_time()
        self.busy = False
        self.left = 0
        self.produced = 0
        self.prompt_tokens = 0
        self.reason = REASON_STOP
        self.text = String("")
        self.emitted = 0
        self.stops = List[String]()
        self.seq = 0

    def close(mut self):
        self.g.close()

    def describe(self) -> String:
        return (
            self.g.architecture()
            + ", "
            + String(self.b.block_count())
            + " layers, "
            + String(self.b.width())
            + " wide, "
            + String(self.context)
            + " positions"
        )

    def answers_to(self, name: String) -> Bool:
        """Whether a request's `model` field names this model.

        The whole reference matches, which is what a client that read
        `/v1/models` will send back. The last path segment matches too, because
        the reference is a file path and nobody wants to type a home directory
        into a curl command to be told the server has no model. Nothing else
        matches: a request naming a model this server did not load is a 404 and
        not a silent redirect to the only one there is.
        """
        if name == self.id:
            return True
        var cut = self.id.rfind("/")
        if cut < 0:
            cut = self.id.rfind("\\")
        if cut < 0:
            return False
        return name == String(self.id[byte = cut + 1 : self.id.byte_length()])

    def next_id(mut self, prefix: StringSpan) -> String:
        """A response id nothing else will have.

        The start time and a counter. Not a random string, because there is no
        randomness here that is not a sampler's, and a client that treats these
        as opaque cannot tell the difference.
        """
        self.seq += 1
        return String(prefix) + String(self.created) + "-" + String(self.seq)

    def render(self, messages_json: String) raises -> String:
        """Messages through the model's own chat template.

        The variable set is the one the conformance corpus uses, which is the
        one 494 real templates were checked against in #22. `tools` and
        `documents` are passed as null rather than left out, because a template
        that branches on them reads better against a null than against a name
        that is not there, and because that is the shape the oracle compared.
        """
        if not self.has_chat:
            raise Error(
                "this model file carries no chat template, so there is nothing"
                " to turn messages into a prompt with, and /v1/completions"
                " takes a prompt directly"
            )
        var vars = String('{"messages": ')
        vars += messages_json
        vars += ', "tools": null, "documents": null'
        vars += ', "add_generation_prompt": true'
        vars += ', "bos_token": '
        vars += _quote(self.bos_text)
        vars += ', "eos_token": '
        vars += _quote(self.eos_text)
        vars += "}"
        return self.chat.render_object(vars)

    def encode(self, text: String, rendered: Bool) raises -> List[Int]:
        """Text to ids. `rendered` says the chat template wrote it.

        The difference is the beginning of text token. A template writes one
        into the text itself, so the post processor must not add a second, and
        a model whose prompt starts with two of them answers differently in a
        way nothing reports.
        """
        var session = Session()
        var ids = List[Int]()
        if rendered:
            self.tokenizer.encode_rendered(text, session, ids)
        else:
            self.tokenizer.encode(text, True, session, ids)
        return ids^

    def detokenize(self, ids: List[Int]) raises -> String:
        """Ids back to text, for a completions request that asked to be echoed
        a prompt it had sent as token ids."""
        return self.tokenizer.decode(ids, True)

    def start(
        mut self,
        prompt: List[Int],
        config: SamplerConfig,
        bias_ids: List[Int],
        bias_vals: List[Float32],
        limit: Int,
        var stops: List[String],
    ) raises:
        """Prefill, and get ready to hand out tokens.

        The prompt goes into the sampler as well as into the model, so the
        penalties see the whole conversation rather than only the part this
        answer has written.
        """
        if len(prompt) == 0:
            raise Error("the prompt encoded to no tokens")
        if len(prompt) >= self.context:
            raise Error(
                "the prompt is "
                + String(len(prompt))
                + " tokens and this server was started with room for "
                + String(self.context)
            )
        var take = limit
        if take > self.context - len(prompt):
            take = self.context - len(prompt)

        self.session.reset()
        self.sampler = Sampler(config, self.b.vocab())
        for i in range(len(bias_ids)):
            self.sampler.bias(bias_ids[i], bias_vals[i])
        for i in range(len(prompt)):
            self.sampler.observe(prompt[i])
        self.decoder = DecodeStream(True)
        self.stops = stops^
        self.text = String("")
        self.emitted = 0
        self.produced = 0
        self.prompt_tokens = len(prompt)
        self.reason = REASON_LENGTH if take == 0 else REASON_STOP
        self.left = take
        self.busy = True
        self.session.prefill(self.b, prompt)

    def advance(mut self) raises -> Bool:
        """One more token, or False because there are no more.

        False is not an error. It means the model asked to stop, a stop string
        matched, or the budget ran out, and `reason` says which.
        """
        if self.left <= 0:
            self.reason = REASON_LENGTH
            return False
        var next = self.session.pick(self.sampler)
        if next == self.eos:
            self.reason = REASON_STOP
            return False
        self.text += self.decoder.step(self.tokenizer, next)
        self.produced += 1
        self.left -= 1
        var cut = self._stop_at()
        if cut >= 0:
            var kept = String(self.text[byte=0:cut])
            self.text = kept
            if self.emitted > cut:
                self.emitted = cut
            self.reason = REASON_STOP
            return False
        self.session.step(self.b, next)
        return True

    def finish(mut self):
        """Give the model back. Called however the generation ended."""
        self.busy = False

    def _stop_at(self) -> Int:
        """Where a stop string begins in the generated text, or minus one."""
        for i in range(len(self.stops)):
            if self.stops[i].byte_length() == 0:
                continue
            var at = self.text.find(self.stops[i])
            if at >= 0:
                return at
        return -1

    def _held(self) -> Int:
        """Bytes at the end of the text that could still become a stop string.

        The longest suffix of what has been generated that is also a proper
        prefix of some stop string. Sending those and finding out one token
        later that they were the first half of a stop is not recoverable, since
        they have left.
        """
        var have = self.text.byte_length()
        var most = 0
        for i in range(len(self.stops)):
            var stop = self.stops[i]
            var k = stop.byte_length() - 1
            if k > have:
                k = have
            while k > most:
                if self.text[byte = have - k : have] == stop[byte=0:k]:
                    most = k
                    break
                k -= 1
        return most

    def delta(mut self, done: Bool) -> String:
        """The text a client has not been sent yet and safely can be.

        `done` says no more tokens are coming, which is what makes the held
        back tail safe: nothing can extend it into a stop string any more.
        """
        var end = self.text.byte_length()
        if not done:
            end -= self._held()
        if end <= self.emitted:
            return String("")
        var out = String(self.text[byte = self.emitted : end])
        self.emitted = end
        return out

    def all_text(self) -> String:
        """Everything generated, which is what a non streaming answer sends."""
        return self.text


def _token_text(tokenizer: Tokenizer, id: Int) -> String:
    """One token's bytes as text, empty when there is no such token."""
    if id < 0:
        return String("")
    var bytes = List[UInt8]()
    if not tokenizer.token_bytes(id, bytes):
        return String("")
    return String(StringSpan(unsafe_from_utf8=bytes))


def _quote(text: String) -> String:
    """A JSON string, for the two token texts that go into the template vars.

    A special token is a short run of printable ASCII in every file anybody
    ships, so this handles the escapes JSON requires and nothing more. It is
    here rather than through the JSON writer because those two are the only
    strings on this path and a writer would be a buffer to carry for them.
    """
    var digits = String("0123456789abcdef")
    var out = List[UInt8]()
    out.append(0x22)
    var bytes = text.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == 0x22 or c == 0x5C:
            out.append(0x5C)
            out.append(c)
        elif c == 0x0A:
            out.append(0x5C)
            out.append(0x6E)
        elif c == 0x0D:
            out.append(0x5C)
            out.append(0x72)
        elif c == 0x09:
            out.append(0x5C)
            out.append(0x74)
        elif c < 0x20:
            out.append(0x5C)
            out.append(0x75)
            out.append(0x30)
            out.append(0x30)
            out.append(digits.as_bytes()[Int(c >> 4)])
            out.append(digits.as_bytes()[Int(c & 15)])
        else:
            out.append(c)
    out.append(0x22)
    return String(StringSpan(unsafe_from_utf8=out))
