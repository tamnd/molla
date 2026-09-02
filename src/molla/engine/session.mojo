"""One sequence, from a prompt to tokens.

Prefill and decode are the same loop here, not two paths. A prefill runs the
forward pass once per prompt token and throws away every logit but the last, a
decode runs it once per generated token and keeps each one, and the only
difference between them is whether anybody looks at the answer. Writing them as
two functions would be two orderings of the same six operations, and the failure
mode from #26 applies just as well here: a prefill that rotates by the wrong
position produces fluent output rather than a crash, and it would only do it on
the path that has no oracle.

That is also why the check that matters is not an arithmetic one. Prefilling n
tokens and then decoding has to leave the cache byte for byte identical to
feeding those n tokens one at a time, because that equality is the statement
that the two routes agree about position. Numbers that are merely close would
pass while one route was off by one.

A prefill here is still one position at a time. Attention takes a single query
and a run of keys by construction, so a real batched prefill that multiplies a
whole prompt at once is a different kernel and not a different loop, and it is
worth having only once there is something to measure it against. What this
gives is correct tokens, which is what the milestone asks for.

The token is chosen by a `Sampler`, which the caller owns rather than the
session. A sampler carries the recent window the penalties look at and the draw
counter the randomness comes from, and both of those belong to the request that
asked for them: two requests against the same weights want their own settings
and their own seed, and a session that made its own sampler would be handing
every caller the same one.
"""

from molla.engine.bind import Bound
from molla.engine.cache import KvCache
from molla.engine.sample import Sampler
from molla.nn.block import Scratch
from molla.nn.model import forward, frequency_factors
from molla.nn.tensor import Buffer


struct Session(Movable):
    """One sequence in flight: where it is, what it has seen, and its scratch.

    Everything a token needs is allocated when the session is made. A decode
    step touches no allocator, which is not a micro optimisation but the
    difference between a steady token rate and one that stalls whenever the
    allocator decides to do some work.
    """

    var cache: KvCache
    var scratch: Scratch
    var x: Buffer
    """The residual stream, one token wide."""

    var logits: Buffer
    """One row, `vocab` long, holding the last token that was run."""

    var factors: List[Float32]
    """`rope_freqs.weight` as float32, read once. Empty when the file has
    none."""

    var pos: Int
    """How many positions this sequence has consumed. The next token is at
    `pos`, and the cache's `filled` agrees with it until something evicts."""

    def __init__(out self, b: Bound, context: Int) raises:
        if context <= 0:
            raise Error("a session needs room for at least one token")
        var trained = b.geometry.context_length
        if trained > 0 and context > trained:
            raise Error(
                "asked for a context of "
                + String(context)
                + " and the file says the model was trained to "
                + String(trained)
            )
        self.cache = KvCache(b.block_count(), context, b.kv_width())
        self.scratch = Scratch(b.specs[0], context)
        self.x = Buffer(b.width())
        self.logits = Buffer(b.vocab())
        self.factors = frequency_factors(b.model)
        self.pos = 0

    def context(self) -> Int:
        return self.cache.context

    def reset(mut self):
        """Start a new sequence in the same memory."""
        self.cache.reset()
        self.pos = 0

    def step(mut self, b: Bound, token: Int) raises:
        """One token through the whole stack, leaving its logits behind.

        The position and the slot come from two different places on purpose.
        The position is the sequence's, which is what rope and the sliding
        window ask about, and the slot is the cache's answer to where that
        position goes.
        """
        self.cache.reserve(1)
        var slot = self.cache.slot_for(self.pos)
        forward(
            b.arch,
            b.model,
            b.specs,
            b.layers,
            self.scratch,
            self.x,
            token,
            self.pos,
            slot,
            self.cache.keys,
            self.cache.values,
            self.factors,
            self.logits,
        )
        self.pos += 1
        self.cache.advance()

    def prefill(mut self, b: Bound, tokens: List[Int]) raises:
        """The prompt. Only the last token's logits are worth anything.

        The rest are the distribution over what came next in a text that
        already says what came next, and computing them is unavoidable because
        each one writes the key the next token attends to.
        """
        if len(tokens) == 0:
            raise Error("a prompt with no tokens has nothing to continue")
        self.cache.reserve(len(tokens))
        for i in range(len(tokens)):
            self.step(b, tokens[i])

    def pick(mut self, mut sampler: Sampler) raises -> Int:
        """One token, from the logits the last step left."""
        return sampler.pick(self.logits)

    def generate(
        mut self,
        b: Bound,
        mut sampler: Sampler,
        prompt: List[Int],
        limit: Int,
        stop: Int = -1,
    ) raises -> List[Int]:
        """A prompt in, up to `limit` new tokens out.

        Stops early on `stop`, which is the end of turn token and is not
        included in what comes back. A caller that wants to see it can compare
        the length against the limit, and a caller that wants to keep going can
        pass a stop of minus one.

        The prompt goes into the sampler as well as into the model. The
        penalties are meant to see the whole text and not only the part this
        run wrote, so a prompt that already repeats a phrase counts against
        repeating it again.
        """
        if limit < 0:
            raise Error("cannot generate a negative number of tokens")
        self.cache.reserve(len(prompt) + limit)
        for i in range(len(prompt)):
            sampler.observe(prompt[i])
        self.prefill(b, prompt)
        var out = List[Int]()
        for _ in range(limit):
            var next = self.pick(sampler)
            if next == stop:
                break
            out.append(next)
            self.step(b, next)
        return out^
