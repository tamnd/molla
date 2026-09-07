"""One loaded model, and the state of the one request that is using it.

Everything a route needs and nothing a route should know about. The protocol
holds the address of one of these and calls four things on it: render a prompt,
start a generation, take a token, and say what the last one produced. What a
GGUF is, what a template is, and how a token becomes bytes stay on this side of
that line.

## Several requests, one pool

A request in flight is a job, and there are as many of them as the server was
started with slots. The tokens live in the batch below, which is where the model
can reach them. What a job holds is the part of a request that is text rather
than tokens: what has been decoded, how much of it has gone out, the stop
strings, and why it ended. A batch has no business knowing any of that and a
protocol has no business knowing what a cell is.

The pool is shared rather than divided. A request is admitted for its prompt
plus everything it is allowed to generate, and it is refused when the pool has
no room for that, so one long request can have the whole pool while it lasts and
several short ones fit alongside each other. Dividing the context by the slot
count up front would refuse a long request on an idle server, which is the
worse of the two failures.

A job's handle is the slot it sits in, and it is valid until `finish`. After
that the slot is free and whatever is read off it belongs to whoever came next.

## Who drives the loop

Whichever connection calls `advance` steps the batch, and a step carries every
job that has work. So a connection asking for its own next token pays for
everybody's, and the jobs it advanced find their tokens waiting when their own
connections come round. That is what makes this interleave without a thread: the
reactor already hands control back between tokens, and the step that happens in
one connection's turn is the step all of them needed.

A request that is not streaming still holds the worker for its whole generation.
It drives the batch while it does, so the streaming connections behind it keep
producing, but nothing else on that worker is answered until it is done.

## Two backends, one of them present

The sequence lives in a host session or in a device batch and never in both, so
both are optionals and exactly one is filled. A variant would be tidier and Mojo
has no shape for one that does not cost more than this does. What it buys is
that a server started on the host does not allocate a device cache and a server
started on a card does not allocate a host one, and a kv cache at four bytes an
element is not a thing to allocate twice for the sake of a field.

The host path is one job whatever the slot count, because the host has no batch
and never had one. A server started on the host with room for more is told so
rather than quietly serving one at a time.

Which one it is was decided before the file was opened, by
`molla.engine.backend`, and it is printed at startup and served on
`/molla/version` for the same reason `molla generate` prints it: a server that
quietly fell back to the host looks exactly like a slow card.

## Stop strings are held, not searched afterwards

A stop string can straddle two tokens, and it can be half emitted before it is
known to be one. So the generated text is kept whole here, and what a caller may
send is the part of it that cannot still turn out to be the beginning of a stop
string. That means a token is sometimes produced and nothing goes out for it,
which is correct, and it is why the streaming loop asks for a delta rather than
assuming one token is one chunk.
"""

from molla.engine.backend import Backend
from molla.engine.batch import DeviceBatch, open_batch
from molla.engine.bind import Bound, bind
from molla.engine.device import device_context, load_on_device
from molla.engine.sample import Sampler, SamplerConfig
from molla.engine.session import Session as Decode
from molla.jinja.template import Template
from molla.model.gguf import Gguf
from molla.model.load import Weights, load, plan_load
from molla.model.repack import RepackCache, model_key, open_cache
from molla.model.spec import read_geometry
from molla.nn.gpu import PREFILL_CHUNK
from molla.nn.repack import CACHE_F16
from molla.sys.clock import unix_time
from molla.sys.device import Device
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

comptime DEFAULT_SLOTS = 1
"""Requests in flight when nobody says.

One, so that a server nobody configured behaves the way it did before there was
a batch: the whole pool is available to the one request being answered, and
memory is what it was. Asking for more is asking to share the pool, and that is
a decision about a machine rather than a default anything can be given.
"""

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


struct Job(Movable):
    """One request in flight, in the terms a protocol answers in.

    Everything here is about text. The tokens are in the batch, and the only
    number that crosses between the two is `taken`, which says how much of what
    the stream has written this job has turned into text already.
    """

    var live: Bool
    var at: Int
    """Which stream of the batch this job is, which is its own handle on the
    device path and zero on the host one."""

    var decoder: DecodeStream
    var text: String
    """Everything this generation has decoded, truncated at a stop string once
    one has been seen."""

    var emitted: Int
    """Bytes of `text` already handed to the client."""

    var taken: Int
    """Tokens of the stream's output already turned into text."""

    var produced: Int
    var prompt_tokens: Int
    var left: Int
    var reason: Int
    var stops: List[String]

    def __init__(out self):
        """A free slot. Every field is set again by `begin`."""
        self.live = False
        self.at = -1
        self.decoder = DecodeStream(True)
        self.text = String("")
        self.emitted = 0
        self.taken = 0
        self.produced = 0
        self.prompt_tokens = 0
        self.left = 0
        self.reason = REASON_STOP
        self.stops = List[String]()

    def begin(
        mut self,
        at: Int,
        prompt_tokens: Int,
        take: Int,
        var stops: List[String],
    ):
        self.live = True
        self.at = at
        self.decoder = DecodeStream(True)
        self.text = String("")
        self.emitted = 0
        self.taken = 0
        self.produced = 0
        self.prompt_tokens = prompt_tokens
        self.left = take
        self.reason = REASON_LENGTH if take == 0 else REASON_STOP
        self.stops = stops^

    def stop_at(self) -> Int:
        """Where a stop string begins in the generated text, or minus one."""
        for i in range(len(self.stops)):
            if self.stops[i].byte_length() == 0:
                continue
            var at = self.text.find(self.stops[i])
            if at >= 0:
                return at
        return -1

    def held(self) -> Int:
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

    var session: Optional[Decode]
    var batch: Optional[DeviceBatch]
    """The sequences, in host memory or on the card. Exactly one of them is
    filled and `backend.on_device` says which. The host one holds a sequence and
    the device one holds as many as there are slots."""

    var backend: Backend
    """Where this server computes, and why. Reported at startup and on
    `/molla/version`, because it is not visible in an answer."""

    var sampler: Sampler
    """The host path's sampler. The device path keeps one a stream, inside the
    batch, because the recent window the penalties read is a sequence's own."""

    var id: String
    """What `/v1/models` reports and what a request's `model` is matched
    against. The whole reference the server was given, so a client that round
    trips it gets a match."""

    var created: Int
    var eos: Int
    var context: Int
    var bos_text: String
    var eos_text: String

    var jobs: List[Job]
    """One entry a slot, dead until a request takes it. Fixed length, so a
    handle is a subscript and nothing has to be looked up."""

    var slots: Int
    var seq: Int
    """Requests answered, which is what makes a response id unique."""

    def __init__(
        out self,
        model_path: String,
        tokenizer_path: String,
        id: String,
        context: Int,
        backend: Backend = Backend(),
        form: Int = CACHE_F16,
        slots: Int = DEFAULT_SLOTS,
        fair: Bool = False,
    ) raises:
        if slots < 1:
            raise Error("a server needs room for at least one request")
        if slots > 1 and not backend.on_device:
            raise Error(
                "--slots is a device setting and this server is running on the"
                " host, where there is one sequence and no batch to put a"
                " second one in"
            )
        if fair and not backend.on_device:
            raise Error(
                "--fair is a device setting and this server is running on the"
                " host, where there is one sequence and so nothing to be fair"
                " between"
            )
        var g = Gguf(model_path)
        var dev = backend.device
        var geometry = read_geometry(g)
        var want = context if context > 0 else DEFAULT_CONTEXT
        if geometry.context_length > 0 and want > geometry.context_length:
            want = geometry.context_length

        # The cache is opened before either plan, because the plan is what
        # decides which copy of each weight the read stage warms, and a cache
        # that turns up afterwards is one the plan could not use.
        var cache = open_cache(model_path, model_key(g))
        var weights: Weights
        var b: Bound
        self.session = None
        self.batch = None

        if backend.on_device:
            var ctx = device_context(dev.index)
            weights = load_on_device(g, cache, model_path, dev, ctx)
            # The same file bound twice. Once against the residency, which is
            # what the kernels read, and once without it, which is where the
            # norm gains are readable so they can be uploaded. A `Bound` owns no
            # bytes, so the second is a list of addresses and not a second copy
            # of anything.
            b = bind(g, cache, weights.residency())
            self.batch = open_batch(
                ctx, bind(g, cache), b, want, slots, PREFILL_CHUNK, form, fair
            )
        else:
            # Everything stays in the mapping, because host kernels cannot read
            # a tensor on a card.
            #
            # A miss repacks while it loads and a hit does not, so the first
            # start against a model pays once and every start after it binds
            # straight to the cache. This run binds to whatever was there when
            # it opened, which on a miss is the file, so the repack a miss
            # writes is for the next start and not for this one.
            if form != CACHE_F16:
                raise Error(
                    "--cache-type is a device setting and this server is"
                    " running on the host"
                )
            var repack_for = String("") if cache.usable else model_path
            weights = load(g, plan_load(g, dev, 0, cache), 0, False, repack_for)
            b = bind(g, cache)
            self.session = Decode(b, want)

        var counter = AllocCounter()
        var tokenizer = Tokenizer(tokenizer_path, counter.raw())
        var source = g.string_or("tokenizer.chat_template", "")
        self.has_chat = source.byte_length() > 0
        self.chat = Template(source)

        self.eos = g.uint_or("tokenizer.ggml.eos_token_id", -1)
        var bos = g.uint_or("tokenizer.ggml.bos_token_id", -1)
        self.bos_text = _token_text(tokenizer, bos)
        self.eos_text = _token_text(tokenizer, self.eos)

        self.backend = backend
        self.sampler = Sampler(SamplerConfig(), b.vocab())
        self.context = want
        self.g = g^
        self.weights = weights^
        self.cache = cache^
        self.b = b^
        self.tokenizer = tokenizer^
        self.counter = counter
        self.id = id
        self.created = unix_time()
        self.slots = slots
        self.jobs = List[Job]()
        for _ in range(slots):
            self.jobs.append(Job())
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

    def repack(self) -> String:
        """Whether this model bound to a repack cache, in one line.

        Said out loud at startup for the same reason `molla load` says it: a
        repack that reruns on every start is the thing the cache exists to
        prevent, and a server that is quietly doing it every time looks exactly
        like one that is not.
        """
        if self.cache.usable:
            return (
                String(self.cache.count())
                + " tensors from cache, "
                + String(self.cache.bytes() // (1 << 20))
                + " MiB"
            )
        return self.cache.reason

    def running_on(self) -> String:
        """Which backend answers, in one line, with the reason when there is
        one.

        The reason is only ever there after `auto` stayed on the host, which is
        the case somebody looking at a slow server needs told rather than left
        to work out from a token rate.
        """
        var out = self.backend.describe()
        if self.backend.note.byte_length() > 0:
            out += ", " + self.backend.note
        return out^

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

    def free_slot(self) -> Int:
        """A slot nothing is using, or minus one."""
        for i in range(len(self.jobs)):
            if not self.jobs[i].live:
                return i
        return -1

    def running(self) -> Int:
        """Requests in flight, for the line the server prints when it stops."""
        var n = 0
        for i in range(len(self.jobs)):
            if self.jobs[i].live:
                n += 1
        return n

    def start(
        mut self,
        prompt: List[Int],
        config: SamplerConfig,
        bias_ids: List[Int],
        bias_vals: List[Float32],
        limit: Int,
        var stops: List[String],
    ) raises -> Int:
        """Take a request, and return the handle everything else takes.

        Minus one means the server is full: every slot is answering something,
        or the pool has no region long enough for what this request could come
        to hold. That is a refusal and not an error, since the same request
        would be taken a moment later, and it is why it comes back as a number
        rather than as a raise. What raises is a request that would be refused
        however empty the server was.

        Nothing is computed here on the device path. The prompt is admitted and
        the first pass over it happens on the first `advance`, which is what
        lets a long prompt ride along with other jobs' decodes rather than
        stopping them while it prefills.

        The prompt goes into the sampler as well as into the model, so the
        penalties see the whole conversation rather than only the part this
        answer has written. `DeviceBatch.admit` does that for a stream and the
        host path does it here.
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

        var job = self.free_slot()
        if job < 0:
            return -1

        var at = 0
        if self.batch:
            # A request that would fit on its own and does not fit beside the
            # ones already running is refused here rather than admitted and
            # preempted later, which is #32's rule.
            if not self.batch.value().fits(len(prompt) + take):
                return -1
            at = self.batch.value().admit(prompt.copy(), take, self.eos, config)
            for i in range(len(bias_ids)):
                self.batch.value().bias(at, bias_ids[i], bias_vals[i])
        else:
            self.session.value().reset()
            self.sampler = Sampler(config, self.b.vocab())
            for i in range(len(bias_ids)):
                self.sampler.bias(bias_ids[i], bias_vals[i])
            for i in range(len(prompt)):
                self.sampler.observe(prompt[i])
            self.session.value().prefill(self.b, prompt)
        self.jobs[job].begin(at, len(prompt), take, stops^)
        return job

    def _check(self, job: Int) raises:
        if job < 0 or job >= len(self.jobs):
            raise Error("there is no job " + String(job))
        if not self.jobs[job].live:
            raise Error("job " + String(job) + " has already been finished")

    def _next(mut self, job: Int) raises -> Int:
        """The next token for one job, or minus one because there are no more.

        On the device path this is where the batch is stepped, and a step
        carries every job that has work rather than only this one. So a
        connection asking for its own token pays for everybody's, and the jobs
        it advanced find theirs waiting.
        """
        var at = self.jobs[job].at
        if not self.batch:
            var token = self.session.value().pick(self.sampler)
            if token == self.eos:
                return -1
            self.session.value().step(self.b, token)
            return token
        while self.batch.value().produced(at) <= self.jobs[job].taken:
            if not self.batch.value().busy(at):
                return -1
            if self.batch.value().step() == 0:
                return -1
        var token = self.batch.value().token(at, self.jobs[job].taken)
        self.jobs[job].taken += 1
        return token

    def advance(mut self, job: Int) raises -> Bool:
        """One more token for one job, or False because there are no more.

        False is not an error. It means the model asked to stop, a stop string
        matched, or the budget ran out, and `reason_of` says which.
        """
        self._check(job)
        if self.jobs[job].left <= 0:
            self.jobs[job].reason = REASON_LENGTH
            return False
        var next = self._next(job)
        if next < 0:
            # The stream ran out. On the device path the batch knows whether the
            # stop token or the limit is what did it, and on the host path
            # reaching here at all means the stop token, since the limit is the
            # check above.
            var stopped = True
            if self.batch:
                stopped = self.batch.value().ended(self.jobs[job].at)
            self.jobs[job].reason = REASON_STOP if stopped else REASON_LENGTH
            return False
        self.jobs[job].text += self.jobs[job].decoder.step(self.tokenizer, next)
        self.jobs[job].produced += 1
        self.jobs[job].left -= 1
        var cut = self.jobs[job].stop_at()
        if cut >= 0:
            var kept = String(self.jobs[job].text[byte=0:cut])
            self.jobs[job].text = kept
            if self.jobs[job].emitted > cut:
                self.jobs[job].emitted = cut
            self.jobs[job].reason = REASON_STOP
            # The stream is told to stop as well as the job, or it would keep
            # taking a place in every step until its own limit ran out, decoding
            # text nobody is going to be sent.
            if self.batch:
                self.batch.value().halt(self.jobs[job].at)
            return False
        return True

    def finish(mut self, job: Int) raises:
        """Give the slot back. Called however the generation ended.

        The stream goes with it, which is what hands its region of the pool back
        for the next request. Nothing may be read off this job afterwards.
        """
        if job < 0 or job >= len(self.jobs):
            return
        if not self.jobs[job].live:
            return
        if self.batch:
            self.batch.value().drop(self.jobs[job].at)
        self.jobs[job].live = False

    def delta(mut self, job: Int, done: Bool) raises -> String:
        """The text a client has not been sent yet and safely can be.

        `done` says no more tokens are coming, which is what makes the held
        back tail safe: nothing can extend it into a stop string any more.
        """
        self._check(job)
        var end = self.jobs[job].text.byte_length()
        if not done:
            end -= self.jobs[job].held()
        if end <= self.jobs[job].emitted:
            return String("")
        var out = String(
            self.jobs[job].text[byte = self.jobs[job].emitted : end]
        )
        self.jobs[job].emitted = end
        return out

    def all_text(self, job: Int) raises -> String:
        """Everything generated, which is what a non streaming answer sends."""
        self._check(job)
        return self.jobs[job].text

    def reason_of(self, job: Int) raises -> Int:
        self._check(job)
        return self.jobs[job].reason

    def produced_of(self, job: Int) raises -> Int:
        self._check(job)
        return self.jobs[job].produced

    def prompt_of(self, job: Int) raises -> Int:
        self._check(job)
        return self.jobs[job].prompt_tokens


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
