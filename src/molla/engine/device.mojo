"""One sequence on the device, from a prompt to tokens.

The device twin of `molla.engine.session`, and the same six operations in the
same order. What differs is where the numbers are between them. The cache is
device memory the attention kernel writes and reads without either half ever
being copied, the residual stream never leaves, and the one transfer a token
pays for is the row of logits somebody has to look at to pick the next one.

## Why the context lives here

A CUDA process gets one `DeviceContext`. Constructing a second one succeeds and
then hangs on the first buffer allocated against it, with the card idle and every
thread asleep in a futex wait, which is a failure that looks like a slow model
rather than like an error. Metal does not mind, so this is a rule the fleet finds
and a laptop does not.

So the session owns the one context and hands it to everything below it. The
load takes it as an argument, the pool allocates against it, the model's small
weights are uploaded through it, and every kernel is queued on it. Nothing under
this module makes one.

## What a step costs

One synchronize, at the end. Everything a token needs is queued first and waited
for once, which is what makes a token one command buffer rather than several
hundred, and it is the whole reason this module exists rather than the host
session calling device kernels one at a time.

The logits are not downloaded by a step, only by `fetch`. A prefill runs the
forward pass once per prompt token and throws away every logit but the last, so
downloading each one would be a transfer per token for a number nobody reads.
`pick` fetches, which means the sampler always sees the last step's answer and
the prompt costs no transfers at all.

## Opening one, from three callers

`device_context`, `load_on_device` and `open_session` are here rather than in
whichever caller wanted a session first, because there are three of them now:
the server, `molla generate`, and the logit oracle. Each of the three has to
make the one context, refuse a model that does not entirely fit, and write a
planar cache when there is none, and three copies of that is three chances to
get the refusal wrong in the direction that reads zeros and answers anyway.

Each of them is behind a `comptime if` because `max-core` resolves the device
architecture when molla is compiled, so a call that is not guarded is what stops
the build on a machine with no GPU. Three of the five boxes we test on have
none.
"""

from std.sys.info import has_accelerator

from max.gpu.host import DeviceContext

from molla.engine.bind import Bound
from molla.engine.cache import check_room, slot_of
from molla.engine.sample import Sampler
from molla.model.gguf import Gguf
from molla.model.load import Weights, device_refusal, load, plan_load
from molla.model.repack import RepackCache, model_key, open_cache
from molla.nn.gpu import SPAN, DeviceVec
from molla.nn.gpu_block import (
    PREFILL_CHUNK,
    DeviceModel,
    DeviceScratch,
    device_forward,
)
from molla.nn.model import frequency_factors
from molla.nn.tensor import Buffer
from molla.sys.device import Device


struct DeviceKvCache(Movable):
    """One sequence's keys and values, in device memory, one vector per layer.

    The same shape as `molla.engine.cache.KvCache` and the same policy, which is
    not a coincidence and not a copy: the two questions that are policy rather
    than storage, which slot a position goes in and what happens when the context
    fills, are free functions over there and this calls them. The day a slot
    stops being a position both caches change together.
    """

    var keys: List[DeviceVec]
    """`layers` vectors of `context * kv_width` floats."""

    var values: List[DeviceVec]
    """The same, and the same length, always."""

    var layers: Int
    var context: Int
    var kv_width: Int
    var filled: Int

    def __init__(
        out self, ctx: DeviceContext, layers: Int, context: Int, kv_width: Int
    ) raises:
        """Allocate the whole thing up front.

        More pressing here than on the host. A device allocation in the middle
        of a decode is a stall the driver may serve by evicting something else it
        was holding, and on a card that is nearly full it is an out of memory
        error several tokens into an answer. The size is known the moment the
        context length is chosen, so it is taken then and the failure, if there
        is going to be one, happens before any output has been printed.
        """
        if layers <= 0:
            raise Error("a cache needs at least one layer")
        if context <= 0:
            raise Error("a cache needs room for at least one position")
        if kv_width <= 0:
            raise Error("a cache needs a positive key width")
        self.layers = layers
        self.context = context
        self.kv_width = kv_width
        self.filled = 0
        self.keys = List[DeviceVec]()
        self.values = List[DeviceVec]()
        var per = context * kv_width
        for _ in range(layers):
            self.keys.append(DeviceVec(ctx, per))
            self.values.append(DeviceVec(ctx, per))

    def bytes(self) -> Int:
        """What this occupies on the card, which is worth reporting first."""
        return 2 * self.layers * self.context * self.kv_width * 4

    def reset(mut self):
        """Forget the sequence without giving back the memory."""
        self.filled = 0

    def slot_for(self, pos: Int) raises -> Int:
        return slot_of(pos, self.context)

    def room(self) -> Int:
        return self.context - self.filled

    def reserve(mut self, count: Int) raises:
        check_room(count, self.room(), self.context)

    def advance(mut self, count: Int = 1) raises:
        self.reserve(count)
        self.filled += count


struct DeviceSession(Movable):
    """One sequence in flight on a card: the context, the model, and the cache.

    Everything a token needs is allocated when the session is made, on the
    device and on the host both. A decode step touches no allocator on either
    side.
    """

    var ctx: DeviceContext
    """The one context this process has. See the module docstring."""

    var model: DeviceModel
    var cache: DeviceKvCache
    var scratch: DeviceScratch

    var batch: DeviceScratch
    """The same scratch again, sized for a prefill chunk.

    A second one rather than one sized for the chunk and used for both, because
    the attention scratch scales with the chunk and the context together and a
    decode would then be holding a hundred times the scores it reads.
    """

    var wide: DeviceVec
    """The residual stream for a prefill chunk, one row a token."""

    var x: DeviceVec
    """The residual stream, one token wide, device side from the embedding
    lookup to the final norm."""

    var logits: Buffer
    """One row, `vocab` long, on the host. The only thing a token brings back,
    and it comes back when somebody asks rather than when a step ends."""

    var pos: Int
    """How many positions this sequence has consumed. The next token is at
    `pos`, and the cache's `filled` agrees with it until something evicts."""

    var batched: Bool
    """Whether the last pass was a chunk, so `fetch` knows which logits are the
    live ones.

    The two scratches each own a logits vector and a pass writes exactly one of
    them, so reading the other gives whatever the last decode left there. That
    is a silent wrong answer rather than an error, because the buffer is the
    right shape and full of real numbers from an earlier token."""

    def __init__(
        out self, ctx: DeviceContext, host: Bound, dev: Bound, context: Int
    ) raises:
        """A model that is already loaded and bound twice, ready for a token.

        `host` and `dev` are the same file bound against the same repack cache,
        one without a residency and one with it. Neither owns bytes, so this is
        two lists of addresses rather than two copies of a model, and both are
        needed: the matrices are read out of the pool by the kernels and the norm
        gains are read out of the mapping to be uploaded.
        """
        if context <= 0:
            raise Error("a session needs room for at least one token")
        var trained = dev.geometry.context_length
        if trained > 0 and context > trained:
            raise Error(
                "asked for a context of "
                + String(context)
                + " and the file says the model was trained to "
                + String(trained)
            )
        self.ctx = ctx
        self.model = DeviceModel(
            ctx,
            dev.arch,
            dev.specs,
            host.model,
            dev.model,
            host.layers,
            dev.layers,
            frequency_factors(host.model),
        )
        self.cache = DeviceKvCache(
            ctx, dev.block_count(), context, dev.kv_width()
        )
        self.scratch = DeviceScratch(ctx, dev.specs[0], context, dev.vocab())
        var chunk = PREFILL_CHUNK
        if chunk > context:
            chunk = context
        self.batch = DeviceScratch(
            ctx, dev.specs[0], context, dev.vocab(), chunk
        )
        # The same `SPAN` rows of slack the scratch carries, for the same
        # reason: the residual stream is what the first matmul of a layer reads.
        self.wide = DeviceVec(ctx, (chunk + SPAN) * dev.width())
        self.x = DeviceVec(ctx, dev.width())
        self.logits = Buffer(dev.vocab())
        self.pos = 0
        self.batched = False

    def context(self) -> Int:
        return self.cache.context

    def reset(mut self):
        """Start a new sequence in the same memory."""
        self.cache.reset()
        self.pos = 0
        self.batched = False

    def run(mut self, tokens: List[Int]) raises:
        """A run of tokens through the stack in one pass, waited for once.

        The wait is here rather than in `fetch` because a prefill that never
        waits hands the driver every launch of every chunk at once, and the
        point of the design is one synchronize a pass and not none.
        """
        var n = len(tokens)
        if n == 0:
            raise Error("a forward pass needs at least one token")
        self.cache.reserve(n)
        var slot = self.cache.slot_for(self.pos)
        device_forward(
            self.ctx,
            self.model,
            self.batch if n > 1 else self.scratch,
            self.wide if n > 1 else self.x,
            tokens,
            self.pos,
            slot,
            self.cache.keys,
            self.cache.values,
        )
        self.ctx.synchronize()
        self.batched = n > 1
        self.pos += n
        for _ in range(n):
            self.cache.advance()

    def step(mut self, token: Int) raises:
        """One token through the whole stack, leaving its logits on the card.

        Queued as one run and waited for once. The wait is here rather than in
        `fetch` because a prefill of a few thousand tokens that never waits is a
        few hundred thousand launches handed to a driver that has to hold all of
        them, and the point of the design is one synchronize per token and not
        none.
        """
        var one: List[Int] = [token]
        self.run(one)

    def fetch(mut self) raises:
        """Bring the last step's logits back to the host.

        A copy rather than a mapping, because a mapping anywhere in the process
        costs 1.3 GiB on its first call and this one would reach it on the first
        token even with every upload fixed.
        """
        if self.pos == 0:
            raise Error("there are no logits until a token has been run")
        if self.batched:
            self.batch.logits.copy_out(self.logits)
        else:
            self.scratch.logits.copy_out(self.logits)

    def prefill(mut self, tokens: List[Int]) raises:
        """The prompt, in chunks. Only the last token's logits are worth reading.

        A chunk is a matmul with the chunk length as its free dimension rather
        than that many matvecs, so a prompt costs the launches of one token per
        chunk instead of the launches of one token per token. The chunk size is
        what bounds the attention scratch, which is the only thing here that
        grows with both the chunk and the context.

        A trace records one residual stream a layer, so a session that is
        tracing goes back to a token at a time. That path is what the logit
        corpus compares against the host and it has to stay the one the corpus
        was written for.
        """
        if len(tokens) == 0:
            raise Error("a prompt with no tokens has nothing to continue")
        self.cache.reserve(len(tokens))
        if self.scratch.tracing or self.batch.chunk == 1:
            for i in range(len(tokens)):
                self.step(tokens[i])
            return
        var at = 0
        while at < len(tokens):
            var take = len(tokens) - at
            if take > self.batch.chunk:
                take = self.batch.chunk
            var run = List[Int]()
            for i in range(take):
                run.append(tokens[at + i])
            self.run(run)
            at += take

    def pick(mut self, mut sampler: Sampler) raises -> Int:
        """One token, from the logits the last step left on the card."""
        self.fetch()
        return sampler.pick(self.logits)

    def generate(
        mut self,
        mut sampler: Sampler,
        prompt: List[Int],
        limit: Int,
        stop: Int = -1,
    ) raises -> List[Int]:
        """A prompt in, up to `limit` new tokens out.

        The same loop the host session runs, down to the prompt going into the
        sampler as well as into the model so the penalties see the whole text.
        """
        if limit < 0:
            raise Error("cannot generate a negative number of tokens")
        self.cache.reserve(len(prompt) + limit)
        for i in range(len(prompt)):
            sampler.observe(prompt[i])
        self.prefill(prompt)
        var out = List[Int]()
        for _ in range(limit):
            var next = self.pick(sampler)
            if next == stop:
                break
            out.append(next)
            self.step(next)
        return out^


def device_context(index: Int) raises -> DeviceContext:
    """The one context this process is going to use, or a refusal.

    One, and made here rather than wherever it is first wanted. A CUDA process
    gets a single `DeviceContext`: a second one constructs and then hangs on the
    first allocation against it, with the card idle and every thread asleep on a
    futex. So the load and the session are handed the same one, and this is the
    only place it comes from.
    """
    comptime if has_accelerator():
        return DeviceContext(device_id=index)
    raise Error(
        "this build has no device code in it, so there is no device context to"
        " make. Accelerator support is decided when molla is compiled, not when"
        " it is run"
    )


def open_session(
    ctx: DeviceContext, host: Bound, dev: Bound, context: Int
) raises -> Optional[DeviceSession]:
    """The session, behind the same guard for the same reason.

    Optional because the caller holds one of these or a host session and never
    both, and a `None` on the other side is what says which.
    """
    comptime if has_accelerator():
        return DeviceSession(ctx, host, dev, context)
    raise Error(
        "this build has no device code in it, so there is no device session to"
        " run a sequence in"
    )


def load_on_device(
    g: Gguf,
    mut cache: RepackCache,
    model_path: String,
    dev: Device,
    ctx: DeviceContext,
    label: String = "  repack        ",
) raises -> Weights:
    """Put every weight of this model on the card, in the planar layout.

    Every weight, and it refuses rather than placing what fits. A device kernel
    handed a host address reads zeros without faulting, so a model that half
    fits is a server that answers fluently out of layers that saw nothing, and
    nothing in the answer says so. The refusal is checked against the plan
    before any bytes move, so a model that will not fit says so in a second
    rather than after a minute of copying.

    The repack runs first and on its own when there is no cache. A load that
    writes the cache on the way past writes it from the file it is reading, so
    the bytes it copied to the card would be the ggml ones, which is the layout
    these kernels cannot read.

    `label` is the column the repack line is printed under, because the three
    callers report in three different shapes and the one thing this function
    prints has to line up with the rest of whichever one is asking.
    """
    comptime if has_accelerator():
        var refusal = device_refusal(g)
        if refusal.byte_length() > 0:
            raise Error(refusal)
        if not cache.usable:
            print(label + " writing a planar cache first, " + cache.reason)
            var warm = load(
                g, plan_load(g, dev, 0, cache), 0, False, model_path
            )
            _ = warm^
            cache.close()
            cache = open_cache(model_path, model_key(g))
            if not cache.usable:
                raise Error(
                    "the repack cache was written and still cannot be used: "
                    + cache.reason
                )

        var plan = plan_load(g, dev, -1, cache)
        if plan.left_behind > 0:
            raise Error(
                "the device forward pass needs every weight on the card and"
                " this plan leaves "
                + String(plan.left_behind)
                + " of them in the mapping, which would be read as zeros. This"
                " model does not fit on "
                + dev.name
            )
        return load(g, plan^, 0, False, "", ctx)

    # Only reachable on a build with no device code, where the block above is
    # not compiled at all. Every caller has already refused for the same reason,
    # so this is the compiler being told what a function with a return type owes
    # rather than a second check.
    raise Error(
        "this build has no device code in it, so there is nothing to load onto"
    )
