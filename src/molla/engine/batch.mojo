"""One step loop over many sequences sharing a card, a model and a pool.

`DeviceSession` is one sequence: it owns a cache, a scratch and a position, and
its loop is prefill and then a token at a time until something stops it. This is
the other shape. Several sequences share one cache, one scratch and one pass, and
the loop is a step at a time over all of them at once.

## What a step is

Every stream that has work contributes tokens to one batch, and the batch is one
forward pass. A stream that is still reading its prompt contributes as much of it
as the cap has room for. A stream that is generating contributes the one token it
last produced. So a long prompt rides along with other streams' decodes rather
than stopping them, and that is chunked prefill: there is no separate mechanism
for it and there does not need to be, because the cap a batch has anyway is what
cuts the prompt up. See
[docs/validation/batching.md](../../../docs/validation/batching.md).

The cap is the knob that trades time to first token against inter token latency.
A large cap gets a prompt in fast and makes every decode behind it late. A small
one keeps the streams smooth and makes the prompt take longer to start answering.

## What the loop does not do

It does not decide who waits. Admission refuses a stream that does not fit and
that is the whole of the policy here, which is FIFO in the sense that a stream
that was admitted keeps its region until it is done. A fair mode that round robins
by session is the stage after this one, and it goes above this rather than inside
it, because what it changes is which streams are offered and not how a step runs.
"""

from std.sys.info import has_accelerator

from max.gpu.host import DeviceContext

from molla.engine.bind import Bound
from molla.engine.device import DeviceKvCache
from molla.engine.sample import Sampler, SamplerConfig
from molla.nn.gpu import MM_GROUPS, PREFILL_CHUNK, SPAN, DeviceVec
from molla.nn.gpu_block import DeviceModel, DeviceScratch, device_forward
from molla.nn.model import frequency_factors
from molla.nn.repack import CACHE_F16
from molla.nn.tensor import Buffer


struct Stream(Movable):
    """One sequence in a batch: what it owes, where it is, and what it wrote.

    Its own sampler, because the recent window the penalties read is this
    sequence's history and the draw counter is this sequence's, and both of
    those being shared is the same bug wearing two hats.
    """

    var seq: Int
    """Which lease of the shared cache this holds. See `DeviceKvCache.admit`."""

    var prompt: List[Int]
    var fed: Int
    """How much of the prompt has gone through a pass."""

    var pos: Int
    """How many positions this sequence has consumed, prompt and output both."""

    var next: Int
    """The token the following step feeds, or negative while the prompt lasts.

    A generated token is fed at the step after the one that produced it, which
    is what makes a decode one token rather than two: the token that was sampled
    from this step's logits is the token this stream contributes to the next.
    """

    var want: Int
    """How many tokens are still to be generated."""

    var stop: Int
    """The token that ends this stream, or negative for none."""

    var out: List[Int]
    var live: Bool
    var sampler: Sampler

    def __init__(
        out self,
        seq: Int,
        var prompt: List[Int],
        want: Int,
        stop: Int,
        var sampler: Sampler,
    ):
        self.seq = seq
        self.prompt = prompt^
        self.fed = 0
        self.pos = 0
        self.next = -1
        self.want = want
        self.stop = stop
        self.out = List[Int]()
        self.live = True
        self.sampler = sampler^

    def prefilling(self) -> Bool:
        return self.fed < len(self.prompt)

    def working(self) -> Bool:
        """Whether this stream has a token to put in the next batch."""
        return self.live and (self.prefilling() or self.want > 0)


struct DeviceBatch(Movable):
    """Many sequences over one model, one pool and one pass a step.

    Everything is allocated when this is made. A step takes no memory on either
    side, which is the same rule `DeviceSession` follows and it matters more
    here, because a step that allocated would allocate once a stream.
    """

    var ctx: DeviceContext
    var model: DeviceModel
    var cache: DeviceKvCache

    var scratch: DeviceScratch
    """Sized for a whole batch of tokens and for as many answers as there are
    slots, since every stream in a step can want one."""

    var x: DeviceVec
    """The residual stream for the batch, one row a token."""

    var solo: DeviceScratch
    """The same again for a step that turned out to carry one token.

    Which happens whenever one stream is left with work, so it is the whole of
    the tail of any run and all of a run with one stream in it. A scratch is
    sized for the run it serves rather than for the largest run it might serve:
    the attention scores scale with the chunk and the context together, and the
    norm reads a vector the width of what it was handed. This is the same pair
    `DeviceSession` holds and it is held here for the same reason.
    """

    var solox: DeviceVec
    """The residual stream for that step, one row wide."""

    var logits: Buffer
    """The answers a step brought back, a row a stream that got one."""

    var streams: List[Stream]
    var cap: Int
    """The most tokens one step carries. See the module docstring."""

    var slots: Int
    """The most streams that can be in flight at once."""

    var form: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        var model: DeviceModel,
        kv_width: Int,
        context: Int,
        slots: Int = 16,
        cap: Int = PREFILL_CHUNK,
        form: Int = CACHE_F16,
    ) raises:
        """A model already on the card, and a pool that many sequences share.

        The model comes in rather than being built here, so that anything with a
        `DeviceModel` can be batched over. `open_batch` is what a caller holding
        a file and its bindings uses.

        `context` is the pool, not one sequence's context. A stream asks for the
        positions it needs when it is admitted and gets a region that long, so
        the pool is however much there is to hand out and the sum of what the
        streams hold cannot exceed it. That is the whole of admission and
        `DeviceKvCache.admit` is where it lives.
        """
        if context <= 0:
            raise Error("a batch needs a pool with room for a token")
        if slots < 1:
            raise Error("a batch needs room for at least one stream")
        if cap < 1:
            raise Error("a batch has to carry at least one token a step")
        self.ctx = ctx
        self.model = model^
        self.cache = DeviceKvCache(
            ctx, self.model.block_count(), context, kv_width, form
        )
        self.cap = cap
        if self.cap > self.cache.paging.chunk():
            self.cap = self.cache.paging.chunk()
        self.slots = slots
        self.scratch = DeviceScratch(
            ctx,
            self.model.specs[0],
            context,
            self.model.vocab(),
            self.cap,
            slots,
        )
        # Rounded up the way the scratch is and for the same reason: the
        # residual stream is what the first matmul of a layer reads.
        var block = SPAN * MM_GROUPS
        var rows = (self.cap + block - 1) // block * block
        self.x = DeviceVec(ctx, rows * self.model.width())
        self.solo = DeviceScratch(
            ctx, self.model.specs[0], context, self.model.vocab(), 1, slots
        )
        self.solox = DeviceVec(ctx, self.model.width())
        self.logits = Buffer(self.model.vocab(), slots)
        self.streams = List[Stream]()
        self.form = form
        # The pool arrives with its whole index handed to the sequence a session
        # would own. Nothing here is that session, so it goes back and every
        # region this hands out is one it admitted.
        self.cache.evict(0)

    def admit(
        mut self,
        var prompt: List[Int],
        limit: Int,
        stop: Int = -1,
        sampling: SamplerConfig = SamplerConfig(),
    ) raises -> Int:
        """Take a stream if the pool has room for its worst case, or refuse.

        The worst case is the prompt plus everything it is allowed to generate,
        because that is what it will hold if it runs to the limit, and a request
        admitted on anything less is a request that may have to be preempted.
        Preempting costs the prefill again, so waiting is cheaper than admitting
        and regretting it. This is #32's admission rule.

        The prompt goes into the sampler here as well as into the model, so the
        penalties see the whole text, which is what the single sequence path
        does and what llama.cpp does.
        """
        if len(prompt) == 0:
            raise Error("a stream needs a prompt with at least one token")
        if limit < 0:
            raise Error("a stream cannot be asked for a negative number")
        if len(self.streams) >= self.slots:
            raise Error(
                "every one of the "
                + String(self.slots)
                + " stream slots is taken"
            )
        var seq = self.cache.admit(len(prompt) + limit)
        var sampler = Sampler(sampling, self.model.vocab())
        for i in range(len(prompt)):
            sampler.observe(prompt[i])
        var at = len(self.streams)
        self.streams.append(Stream(seq, prompt^, limit, stop, sampler^))
        return at

    def drop(mut self, at: Int) raises:
        """Give a stream's region and cells back, whether or not it finished."""
        if at < 0 or at >= len(self.streams):
            raise Error("there is no stream " + String(at))
        if not self.streams[at].live:
            return
        self.streams[at].live = False
        self.cache.evict(self.streams[at].seq)

    def working(self) -> Int:
        """How many streams have a token for the next step."""
        var n = 0
        for i in range(len(self.streams)):
            if self.streams[i].working():
                n += 1
        return n

    def step(mut self) raises -> Int:
        """One batch through the stack, and a token for everything that got one.

        Returns how many tokens the batch carried, which is zero when there was
        no work, and that is how a caller knows to stop.

        The batch is filled in stream order until the cap is reached. A stream
        that does not fit this step is not skipped in any lasting sense, because
        the streams before it shrink as they finish and it moves up. What it
        does mean is that a long prompt admitted first delays a short one behind
        it, and that is what the fair mode in the stage after this exists to
        change.
        """
        var tokens = List[Int]()
        var rows = List[Int]()
        var answered = List[Int]()
        var deepest = 0
        for i in range(len(self.streams)):
            if not self.streams[i].working():
                continue
            if len(tokens) >= self.cap:
                break
            var take = 1
            if self.streams[i].prefilling():
                take = len(self.streams[i].prompt) - self.streams[i].fed
                if take > self.cap - len(tokens):
                    take = self.cap - len(tokens)
            var at = len(tokens)
            if self.streams[i].prefilling():
                for _ in range(take):
                    tokens.append(self.streams[i].prompt[self.streams[i].fed])
                    self.streams[i].fed += 1
            elif self.streams[i].next < 0:
                raise Error(
                    "stream "
                    + String(i)
                    + " owes "
                    + String(self.streams[i].want)
                    + " tokens and has nothing to continue from"
                )
            else:
                tokens.append(self.streams[i].next)
            var pos = self.streams[i].pos
            _ = self.cache.place_for(self.streams[i].seq, pos, take, True, at)
            self.streams[i].pos = pos + take
            if pos + take - 1 > deepest:
                deepest = pos + take - 1
            # A stream still owing prompt gets no answer, because the logits of
            # a token in the middle of a prompt are a prediction nobody asked
            # for. One that has just finished its prompt gets the same answer
            # the single sequence path samples its first token from.
            #
            # A step where nobody wants an answer still runs the head once, on
            # the last token of the chunk, because an empty row list is how a
            # single sequence asks for its last token and that meaning cannot
            # be taken away. It is one norm and one head matvec of work nobody
            # reads, on a step where every stream is mid prompt.
            if not self.streams[i].prefilling():
                rows.append(at + take - 1)
                answered.append(i)
        if len(tokens) == 0:
            return 0

        self.cache.paging.mixed(len(tokens))
        var many = len(tokens) > 1
        device_forward(
            self.ctx,
            self.model,
            self.scratch if many else self.solo,
            self.x if many else self.solox,
            tokens,
            deepest,
            0,
            self.cache.keys,
            self.cache.values,
            self.cache.paging,
            self.form,
            rows,
        )
        self.ctx.synchronize()
        if len(rows) > 0:
            if many:
                self.scratch.logits.copy_out(self.logits)
            else:
                self.solo.logits.copy_out(self.logits)
        for r in range(len(answered)):
            var i = answered[r]
            var token = self.streams[i].sampler.pick(self.logits, r)
            self.streams[i].want -= 1
            if token == self.streams[i].stop:
                self.streams[i].next = -1
                self.streams[i].want = 0
                continue
            self.streams[i].out.append(token)
            self.streams[i].next = token
        return len(tokens)

    def run(mut self, steps: Int = 0) raises -> Int:
        """Step until nothing has work, or until `steps` of them have run.

        The bound is there so a caller can drive this a step at a time without
        writing the loop again, and zero means no bound.
        """
        var ran = 0
        while steps == 0 or ran < steps:
            if self.step() == 0:
                break
            ran += 1
        return ran

    def produced(self, at: Int) raises -> Int:
        """How many tokens a stream has generated so far.

        What a caller watching a step loop wants, because the answer changes on
        the steps that stream was answered on and not on the others, and that is
        how the time between one stream's tokens gets measured without the loop
        having to keep a clock of its own.
        """
        if at < 0 or at >= len(self.streams):
            raise Error("there is no stream " + String(at))
        return len(self.streams[at].out)

    def output(self, at: Int) raises -> List[Int]:
        """What a stream generated, which is a copy and not the stream's own."""
        if at < 0 or at >= len(self.streams):
            raise Error("there is no stream " + String(at))
        var out = List[Int]()
        for i in range(len(self.streams[at].out)):
            out.append(self.streams[at].out[i])
        return out^


def open_batch(
    ctx: DeviceContext,
    host: Bound,
    dev: Bound,
    context: Int,
    slots: Int = 16,
    cap: Int = PREFILL_CHUNK,
    form: Int = CACHE_F16,
) raises -> Optional[DeviceBatch]:
    """The batch behind the accelerator guard, the way `open_session` is.

    Optional because a build with no device code in it has nothing to return,
    and a `None` on the other side is what says so.
    """
    comptime if has_accelerator():
        var trained = dev.geometry.context_length
        if trained > 0 and context > trained:
            raise Error(
                "asked for a pool of "
                + String(context)
                + " and the file says the model was trained to "
                + String(trained)
            )
        var model = DeviceModel(
            ctx,
            dev.arch,
            dev.specs,
            host.model,
            dev.model,
            host.layers,
            dev.layers,
            frequency_factors(host.model),
        )
        return DeviceBatch(
            ctx, model^, dev.kv_width(), context, slots, cap, form
        )
    return None
