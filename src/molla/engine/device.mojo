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
"""

from max.gpu.host import DeviceContext

from molla.engine.bind import Bound
from molla.engine.cache import check_room, slot_of
from molla.engine.sample import Sampler
from molla.nn.gpu import DeviceVec
from molla.nn.gpu_block import DeviceModel, DeviceScratch, device_forward
from molla.nn.model import frequency_factors
from molla.nn.tensor import Buffer


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

    var x: DeviceVec
    """The residual stream, one token wide, device side from the embedding
    lookup to the final norm."""

    var logits: Buffer
    """One row, `vocab` long, on the host. The only thing a token brings back,
    and it comes back when somebody asks rather than when a step ends."""

    var pos: Int
    """How many positions this sequence has consumed. The next token is at
    `pos`, and the cache's `filled` agrees with it until something evicts."""

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
        self.x = DeviceVec(ctx, dev.width())
        self.logits = Buffer(dev.vocab())
        self.pos = 0

    def context(self) -> Int:
        return self.cache.context

    def reset(mut self):
        """Start a new sequence in the same memory."""
        self.cache.reset()
        self.pos = 0

    def step(mut self, token: Int) raises:
        """One token through the whole stack, leaving its logits on the card.

        Queued as one run and waited for once. The wait is here rather than in
        `fetch` because a prefill of a few thousand tokens that never waits is a
        few hundred thousand launches handed to a driver that has to hold all of
        them, and the point of the design is one synchronize per token and not
        none.
        """
        self.cache.reserve(1)
        var slot = self.cache.slot_for(self.pos)
        device_forward(
            self.ctx,
            self.model,
            self.scratch,
            self.x,
            token,
            self.pos,
            slot,
            self.cache.keys,
            self.cache.values,
        )
        self.ctx.synchronize()
        self.pos += 1
        self.cache.advance()

    def fetch(mut self) raises:
        """Bring the last step's logits back to the host."""
        if self.pos == 0:
            raise Error("there are no logits until a token has been run")
        self.scratch.logits.download(self.logits)

    def prefill(mut self, tokens: List[Int]) raises:
        """The prompt. Only the last token's logits are worth anything."""
        if len(tokens) == 0:
            raise Error("a prompt with no tokens has nothing to continue")
        self.cache.reserve(len(tokens))
        for i in range(len(tokens)):
            self.step(tokens[i])

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
