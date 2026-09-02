"""Turning a row of logits into one token.

A sampler is a pipeline of filters over a candidate set, and the order they run
in is part of the definition rather than an implementation detail. Top-p after
top-k means the mass is measured over what top-k left, and the same two the
other way round is a different sampler that people will report as a bug. The
order here is llama.cpp's, because that is what every preset in circulation was
tuned against:

    grammar mask, logit bias, penalties, temperature, top-k, top-p, min-p,
    typical, sample

Grammar masking is M4. The hook is here and it does nothing, which is a
deliberate placeholder rather than an omission: it has to run before the
penalties so that a token the grammar forbids is gone before anything measures
how much probability mass is left.

A logit bias is added rather than multiplied, so it means the same thing at
every temperature. It goes in before the penalties because a caller pushing a
token down and the penalties pushing the same token down should compose, and it
goes in after the mask because a bias on a token the grammar forbids should not
bring it back.

Greedy is the exact argmax and not a temperature approaching zero. Those differ.
A temperature of one thousandth divides logits that already differ by tens, and
the exponential of that overflows to infinity and comes back as a NaN, so the
limit that is supposed to be greedy is instead undefined. A temperature of zero
here means take the largest and do not build a distribution at all.

The randomness is counter based and not a stream. A stream sampler draws its
next float from wherever the last draw left the state, so two sequences sharing
a generator get tokens that depend on the interleaving, and the same seed and
the same prompt give different output depending on what else was in the batch.
That is not reproducibility, it is reproducibility as long as nobody else is
using the server. A counter based generator hashes the seed together with the
position, so draw `n` of a sequence is a pure function of the two and the batch
cannot be observed from inside it.
"""

from std.math import exp, log

from molla.nn.kernel import argmax
from molla.nn.tensor import Buffer

comptime GREEDY = Float32(0)
"""A temperature of zero, which means take the argmax."""


def _mix(value: UInt64) -> UInt64:
    """splitmix64's finalizer, which is the cheapest good 64 bit avalanche.

    Not a generator by itself. It is the mixing function, applied to a counter
    rather than iterated on a state, which is what makes the draw a pure
    function of what goes in.
    """
    var z = value
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def draw(seed: UInt64, counter: UInt64) -> Float32:
    """A uniform float in `[0, 1)` from a seed and a counter.

    Twenty four bits, which is every value a float32 can represent in that
    range without rounding two counters onto the same float. Taking more bits
    and dividing by two to the sixty four would round anyway and would hide
    that it had.
    """
    var h = _mix(seed ^ _mix(counter + 0x9E3779B97F4A7C15))
    return Float32(Int(h >> 40)) / Float32(1 << 24)


struct SamplerConfig(Copyable, ImplicitlyCopyable, Movable):
    """What a request asked for. Every field's off value is its default."""

    var temperature: Float32
    """Zero is greedy. One leaves the distribution as the model gave it."""

    var top_k: Int
    """Keep the `k` most likely. Zero is off."""

    var top_p: Float32
    """Keep the smallest set whose probability sums past `p`. One is off."""

    var min_p: Float32
    """Drop anything less likely than `min_p` times the most likely. Zero is
    off. This is the one that scales with how confident the model is: a peaked
    distribution keeps almost nothing and a flat one keeps almost everything,
    where a fixed top-k does the same thing to both."""

    var typical_p: Float32
    """Keep the smallest set, ordered by distance from the distribution's own
    entropy, whose mass sums past `p`. One is off. This drops the tokens that
    are surprising and also the ones that are too obvious, which is what makes
    it different from every other filter here."""

    var repeat_penalty: Float32
    """Divides the logit of anything in the recent window. One is off."""

    var frequency_penalty: Float32
    """Subtracted once per appearance in the window. Zero is off."""

    var presence_penalty: Float32
    """Subtracted once for appearing at all. Zero is off."""

    var repeat_last_n: Int
    """How far back the penalties look. Zero disables all three."""

    var seed: UInt64
    """The same seed and the same prompt give the same tokens, whatever else
    the server is doing."""

    def __init__(out self):
        """Greedy, with every filter off. The one configuration that has no
        randomness in it and so is the one every test starts from."""
        self.temperature = GREEDY
        self.top_k = 0
        self.top_p = 1.0
        self.min_p = 0.0
        self.typical_p = 1.0
        self.repeat_penalty = 1.0
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        self.repeat_last_n = 64
        self.seed = 0

    def greedy(self) -> Bool:
        return self.temperature <= 0

    def penalizing(self) -> Bool:
        if self.repeat_last_n <= 0:
            return False
        return (
            self.repeat_penalty != 1.0
            or self.frequency_penalty != 0.0
            or self.presence_penalty != 0.0
        )

    def check(self) raises:
        if self.temperature < 0:
            raise Error("a temperature cannot be negative")
        if self.top_k < 0:
            raise Error("a top-k cannot be negative")
        if self.top_p <= 0 or self.top_p > 1:
            raise Error("a top-p has to be above zero and at most one")
        if self.min_p < 0 or self.min_p > 1:
            raise Error("a min-p has to be between zero and one")
        if self.typical_p <= 0 or self.typical_p > 1:
            raise Error("a typical-p has to be above zero and at most one")
        if self.repeat_penalty <= 0:
            raise Error("a repetition penalty of zero or less erases a logit")
        if self.repeat_last_n < 0:
            raise Error("a penalty window cannot be negative")


struct Sampler(Movable):
    """One sequence's sampler: its settings, its recent tokens, its counter.

    Per sequence and not shared. The recent window is this sequence's history
    and the counter is this sequence's draw number, and both of those being
    global is the same bug wearing two hats.
    """

    var config: SamplerConfig

    var recent: List[Int]
    """The penalty window as a ring. `filled` says how much of it is real."""

    var head: Int
    var filled: Int

    var ids: List[Int]
    """The candidate set, sized to the vocabulary once."""

    var vals: List[Float32]
    """Logits while they are being transformed, then probabilities."""

    var kept: Int
    """How many of `ids` are still candidates. The filters shrink this rather
    than erasing entries, so nothing is copied between stages."""

    var drawn: Int
    """How many tokens this sequence has sampled. The RNG counter."""

    var bias_ids: List[Int]
    """Tokens the request wants pushed up or down. Here rather than in the
    config because it is the one setting whose size depends on the request, and
    a config with a list in it stops being a value anybody can copy about."""

    var bias_vals: List[Float32]
    """Added to the logit of the token at the same index."""

    def __init__(out self, config: SamplerConfig, vocab: Int) raises:
        config.check()
        if vocab <= 0:
            raise Error("a sampler needs a vocabulary")
        self.config = config
        self.recent = List[Int]()
        for _ in range(config.repeat_last_n):
            self.recent.append(-1)
        self.head = 0
        self.filled = 0
        self.ids = List[Int]()
        self.vals = List[Float32]()
        for i in range(vocab):
            self.ids.append(i)
            self.vals.append(0.0)
        self.kept = 0
        self.drawn = 0
        self.bias_ids = List[Int]()
        self.bias_vals = List[Float32]()

    def vocab(self) -> Int:
        return len(self.ids)

    def bias(mut self, token: Int, amount: Float32) raises:
        """Push one token up or down by `amount` logits.

        In the units logits are in, so the same number means the same thing at
        every temperature, and a bias of minus one hundred is a ban rather than
        a strong hint. Naming the same token twice adds up, which is what a
        caller that merged two maps means.
        """
        if token < 0 or token >= self.vocab():
            raise Error(
                "a logit bias names token "
                + String(token)
                + " and the vocabulary is "
                + String(self.vocab())
                + " wide"
            )
        self.bias_ids.append(token)
        self.bias_vals.append(amount)

    def biasing(self) -> Bool:
        return len(self.bias_ids) > 0

    def reset(mut self):
        """A new sequence in the same sampler."""
        self.head = 0
        self.filled = 0
        self.kept = 0
        self.drawn = 0

    def observe(mut self, token: Int):
        """Record a token so the penalties can see it.

        Prompt tokens count. A model asked to continue a text that already
        repeats itself should be penalised for continuing to, and llama.cpp
        does the same, so a preset tuned there behaves the same here.
        """
        if len(self.recent) == 0:
            return
        self.recent[self.head] = token
        self.head = (self.head + 1) % len(self.recent)
        if self.filled < len(self.recent):
            self.filled += 1

    def count_of(self, token: Int) -> Int:
        """How many times `token` appears in the window."""
        var n = 0
        for i in range(self.filled):
            if self.recent[i] == token:
                n += 1
        return n

    def logprobs(self, logits: Buffer, mut out: List[Float32]) raises:
        """The natural log of the softmax of the logits as the model gave them.

        Before the penalties and before the temperature, which is what a client
        asking for logprobs means. A logprob computed after the transforms
        describes the sampler's settings rather than the model, and two clients
        with different temperatures would get different numbers for the same
        model on the same prompt and reasonably call that a bug.

        Computed as `z - max - log(sum(exp(z - max)))` rather than as the log of
        a softmax, so a token with a probability below what a float32 can hold
        comes back as a large negative number instead of minus infinity.
        """
        var n = logits.elements()
        if len(out) != n:
            raise Error("logprobs wants an output the width of the vocabulary")
        var top = logits.data[0]
        for i in range(1, n):
            if logits.data[i] > top:
                top = logits.data[i]
        var sum = Float64(0)
        for i in range(n):
            sum += Float64(exp(logits.data[i] - top))
        var offset = Float32(log(sum))
        for i in range(n):
            out[i] = logits.data[i] - top - offset

    def pick(mut self, logits: Buffer) raises -> Int:
        """One token, through the whole pipeline.

        `logits` is read and not written. A caller that wants logprobs asks for
        them from the same buffer afterwards and gets the model's numbers, which
        would not be true if the transforms happened in place.
        """
        if logits.elements() != self.vocab():
            raise Error(
                "the sampler was built for a vocabulary of "
                + String(self.vocab())
                + " and got "
                + String(logits.elements())
                + " logits"
            )

        # Greedy first, before anything is copied. An exact argmax over the
        # penalised logits is the whole computation, and the penalties are the
        # only transform that can change which one is largest: temperature and
        # every filter here are monotonic, so they cannot move the maximum.
        if self.config.greedy():
            if not self.config.penalizing() and not self.biasing():
                var at = argmax(logits.data, 0, self.vocab())
                if at < 0:
                    raise Error("there are no logits to pick from")
                self._took(at)
                return at

        self._load(logits)
        self._mask()
        self._bias()
        self._penalise()

        if self.config.greedy():
            var best = 0
            for i in range(1, self.kept):
                if self.vals[i] > self.vals[best]:
                    best = i
            var at = self.ids[best]
            self._took(at)
            return at

        self._temperature()
        self._softmax()
        self._top_k()
        self._top_p()
        self._min_p()
        self._typical()
        var at = self._choose()
        self._took(at)
        return at

    def _took(mut self, token: Int):
        self.observe(token)
        self.drawn += 1

    def _load(mut self, logits: Buffer):
        for i in range(self.vocab()):
            self.ids[i] = i
            self.vals[i] = logits.data[i]
        self.kept = self.vocab()

    def _mask(mut self):
        """Where a grammar drops the tokens it forbids. M4.

        First on purpose. A grammar that runs after top-p is a grammar applied
        to a set that was already truncated on the assumption every token was
        allowed, and the result is a sampler that occasionally has nothing left
        to choose from.
        """
        return

    def _bias(mut self):
        """The request's own adjustments to particular tokens.

        Over the candidate set rather than by index into the vocabulary,
        because the mask above is free to drop entries and a bias written to a
        slot rather than to a token is a bias applied to the wrong word. Two
        entries naming the same token add up, which is what a caller that built
        its map by merging two of them means.
        """
        if not self.biasing():
            return
        for i in range(self.kept):
            var t = self.ids[i]
            for j in range(len(self.bias_ids)):
                if self.bias_ids[j] == t:
                    self.vals[i] += self.bias_vals[j]

    def _penalise(mut self):
        if not self.config.penalizing():
            return
        for i in range(self.kept):
            var n = self.count_of(self.ids[i])
            if n == 0:
                continue
            var v = self.vals[i]
            # Divide a positive logit and multiply a negative one, so both
            # move the token down by the same factor. Doing one operation to
            # both signs would penalise half the vocabulary and reward the
            # other half, and a penalty that promotes a token for having just
            # been used is worse than no penalty at all.
            if v > 0:
                v /= self.config.repeat_penalty
            else:
                v *= self.config.repeat_penalty
            v -= Float32(n) * self.config.frequency_penalty
            v -= self.config.presence_penalty
            self.vals[i] = v

    def _temperature(mut self):
        var t = self.config.temperature
        if t == 1.0:
            return
        for i in range(self.kept):
            self.vals[i] /= t

    def _softmax(mut self):
        var top = self.vals[0]
        for i in range(1, self.kept):
            if self.vals[i] > top:
                top = self.vals[i]
        var sum = Float32(0)
        for i in range(self.kept):
            self.vals[i] = exp(self.vals[i] - top)
            sum += self.vals[i]
        if sum <= 0:
            return
        for i in range(self.kept):
            self.vals[i] /= sum

    def _sort(mut self):
        """The candidates in descending probability, by heapsort.

        Heapsort rather than quicksort because a logit row is not random. It
        arrives with structure that a naive pivot handles badly, and the worst
        case of a quicksort over a hundred thousand candidates is a visible
        stall on one token in a conversation, which is exactly the kind of
        intermittent slowness nobody ever tracks down.
        """
        var n = self.kept
        for start in range(n // 2 - 1, -1, -1):
            self._sift(start, n)
        for end in range(n - 1, 0, -1):
            self._swap(0, end)
            self._sift(0, end)
        # A heapsort of a max heap leaves ascending order, and every filter
        # below wants the likely end first.
        var i = 0
        var j = n - 1
        while i < j:
            self._swap(i, j)
            i += 1
            j -= 1

    def _sift(mut self, at: Int, n: Int):
        var root = at
        while True:
            var child = root * 2 + 1
            if child >= n:
                return
            if child + 1 < n and self.vals[child + 1] > self.vals[child]:
                child += 1
            if self.vals[root] >= self.vals[child]:
                return
            self._swap(root, child)
            root = child

    def _swap(mut self, a: Int, b: Int):
        var v = self.vals[a]
        self.vals[a] = self.vals[b]
        self.vals[b] = v
        var i = self.ids[a]
        self.ids[a] = self.ids[b]
        self.ids[b] = i

    def _top_k(mut self):
        var k = self.config.top_k
        if k <= 0 or k >= self.kept:
            return
        self._sort()
        self.kept = k

    def _top_p(mut self):
        var p = self.config.top_p
        if p >= 1.0 or self.kept <= 1:
            return
        self._sort()
        var sum = Float32(0)
        for i in range(self.kept):
            sum += self.vals[i]
            # Past the threshold and not at it. The token that carries the mass
            # over the line is the last one that can be chosen, so a top-p of
            # nine tenths keeps the tokens that make up the first ninety
            # percent and the one straddling the boundary.
            if sum >= p:
                self.kept = i + 1
                return

    def _min_p(mut self):
        var floor = self.config.min_p
        if floor <= 0 or self.kept <= 1:
            return
        var top = self.vals[0]
        for i in range(1, self.kept):
            if self.vals[i] > top:
                top = self.vals[i]
        var cut = top * floor
        var write = 0
        for i in range(self.kept):
            if self.vals[i] >= cut:
                self._swap(write, i)
                write += 1
        if write > 0:
            self.kept = write

    def _typical(mut self):
        """Keep the tokens whose surprise is closest to the average surprise.

        The distribution's entropy is what it expects to be surprised by, and
        this orders candidates by how far their own surprise is from that,
        nearest first, then keeps the smallest prefix carrying `p` of the mass.
        So it drops the tail and also the top when the top is more obvious than
        the distribution as a whole, which no other filter here does.
        """
        var p = self.config.typical_p
        if p >= 1.0 or self.kept <= 1:
            return

        var entropy = Float32(0)
        for i in range(self.kept):
            if self.vals[i] > 0:
                entropy -= self.vals[i] * log(self.vals[i])

        # Sorted by distance rather than by probability, so the ordering the
        # other filters left is not the one this walks. The probabilities move
        # with their ids, which is why every reorder here goes through `_swap`.
        var distance = List[Float32]()
        for i in range(self.kept):
            var surprise = Float32(0)
            if self.vals[i] > 0:
                surprise = -log(self.vals[i])
            var d = surprise - entropy
            distance.append(d if d >= 0 else -d)
        for i in range(1, self.kept):
            var j = i
            while j > 0 and distance[j - 1] > distance[j]:
                var d = distance[j - 1]
                distance[j - 1] = distance[j]
                distance[j] = d
                self._swap(j - 1, j)
                j -= 1

        var sum = Float32(0)
        for i in range(self.kept):
            sum += self.vals[i]
            if sum >= p:
                self.kept = i + 1
                return

    def _choose(mut self) raises -> Int:
        """One draw from what is left, renormalised.

        Renormalised because the filters removed mass and the draw is over what
        remains. Walking the unnormalised weights against a uniform in `[0, 1)`
        would fall off the end whenever the kept mass was under one, and the
        obvious repair of returning the last candidate makes the least likely
        survivor the most likely outcome.
        """
        if self.kept <= 0:
            raise Error("every candidate was filtered out")
        var sum = Float32(0)
        for i in range(self.kept):
            sum += self.vals[i]
        if sum <= 0:
            return self.ids[0]
        var target = draw(self.config.seed, UInt64(self.drawn)) * sum
        var acc = Float32(0)
        for i in range(self.kept):
            acc += self.vals[i]
            if target < acc:
                return self.ids[i]
        # Only reachable when the accumulation rounds short of the sum it was
        # computed from, which float32 does. The last candidate is the right
        # answer there rather than a fallback.
        return self.ids[self.kept - 1]
