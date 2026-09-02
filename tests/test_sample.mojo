"""Tests for the sampling pipeline.

Sampling is checked by what comes out of it over many draws rather than by one
call, because a filter that is off by one candidate produces a token that was
always allowed and a distribution that is quietly wrong. So most of what is
here samples a few hundred times against a distribution whose probabilities are
exact powers of two, and asks which tokens were reachable.

The logits are `log` of the intended probability, so a softmax at temperature
one gives back one half, one quarter, one eighth and two sixteenths. That makes
every threshold below a number a person can check rather than one this file
computed.
"""

from std.math import exp, log

from harness import Suite

from molla.engine.sample import Sampler, SamplerConfig, draw
from molla.nn.tensor import Buffer

comptime VOCAB = 5


def _logits() -> Buffer:
    """Probabilities of a half, a quarter, an eighth and two sixteenths."""
    var out = Buffer(VOCAB)
    out.data[0] = Float32(log(Float64(8)))
    out.data[1] = Float32(log(Float64(4)))
    out.data[2] = Float32(log(Float64(2)))
    out.data[3] = 0.0
    out.data[4] = 0.0
    return out^


def _reachable(mut s: Sampler, logits: Buffer, draws: Int) raises -> Int:
    """How many distinct tokens `draws` samples ever produced."""
    var seen = List[Bool]()
    for _ in range(VOCAB):
        seen.append(False)
    for _ in range(draws):
        seen[s.pick(logits)] = True
    var n = 0
    for i in range(VOCAB):
        if seen[i]:
            n += 1
    return n


def _close(a: Float32, b: Float32, within: Float32) -> Bool:
    var d = a - b
    if d < 0:
        d = -d
    return d <= within


def run(mut suite: Suite) raises:
    test_greedy(suite)
    test_seeded(suite)
    test_counter(suite)
    test_temperature(suite)
    test_top_k(suite)
    test_top_p(suite)
    test_min_p(suite)
    test_typical(suite)
    test_penalties(suite)
    test_bias(suite)
    test_logprobs(suite)
    test_errors(suite)


def test_greedy(mut suite: Suite) raises:
    suite.group("sampler greedy")

    var logits = _logits()
    var s = Sampler(SamplerConfig(), VOCAB)
    suite.check(s.pick(logits) == 0, "greedy takes the largest logit")

    var same = True
    for _ in range(20):
        if s.pick(logits) != 0:
            same = False
    suite.check(same, "and takes it every time, since nothing is random")

    # A greedy pick is the argmax and not the limit of a temperature going to
    # zero. A temperature small enough to be greedy is small enough to divide
    # logits into an exponential that overflows.
    var tie = Buffer(VOCAB)
    for i in range(VOCAB):
        tie.data[i] = 2.0
    var flat = Sampler(SamplerConfig(), VOCAB)
    suite.check(
        flat.pick(tie) == 0, "an exact tie goes to the lower token number"
    )

    var big = Buffer(VOCAB)
    big.data[0] = 800.0
    big.data[1] = 900.0
    big.data[2] = 100.0
    big.data[3] = 0.0
    big.data[4] = -900.0
    var huge = Sampler(SamplerConfig(), VOCAB)
    suite.check(
        huge.pick(big) == 1,
        "and logits far outside what an exponential holds are still ordered",
    )


def test_seeded(mut suite: Suite) raises:
    """The done criterion from #28: a hundred runs, one sequence."""
    suite.group("sampler seeds")

    var logits = _logits()
    var config = SamplerConfig()
    config.temperature = 1.0
    config.seed = 20260901

    var first = List[Int]()
    var s = Sampler(config, VOCAB)
    for _ in range(20):
        first.append(s.pick(logits))

    var stable = True
    for _ in range(100):
        var again = Sampler(config, VOCAB)
        for i in range(20):
            if again.pick(logits) != first[i]:
                stable = False
    suite.check(stable, "the same seed gives the same twenty tokens, 100 times")

    var other = config
    other.seed = 20260902
    var different = Sampler(other, VOCAB)
    var differs = False
    for i in range(20):
        if different.pick(logits) != first[i]:
            differs = True
    suite.check(differs, "and a different seed gives something else")

    var reset = Sampler(config, VOCAB)
    for _ in range(5):
        _ = reset.pick(logits)
    reset.reset()
    var restarted = True
    for i in range(20):
        if reset.pick(logits) != first[i]:
            restarted = False
    suite.check(restarted, "resetting a sampler starts the sequence again")


def test_counter(mut suite: Suite) raises:
    """The draw is a function of the seed and the count, not of a stream."""
    suite.group("sampler counter")

    suite.check(
        draw(1, 0) == draw(1, 0) and draw(1, 7) == draw(1, 7),
        "the same seed and counter give the same float",
    )
    suite.check(
        draw(1, 0) != draw(1, 1) and draw(1, 0) != draw(2, 0),
        "and moving either one moves the answer",
    )

    var inside = True
    for i in range(256):
        var v = draw(99, UInt64(i))
        if v < 0.0 or v >= 1.0:
            inside = False
    suite.check(inside, "every draw is in the half open unit interval")

    # The point of a counter rather than a stream. Two sequences sharing a
    # server take their draws interleaved, and with a shared stream that
    # interleaving would decide what each of them got.
    var logits = _logits()
    var config = SamplerConfig()
    config.temperature = 1.0
    config.seed = 4242

    var alone = List[Int]()
    var solo = Sampler(config, VOCAB)
    for _ in range(12):
        alone.append(solo.pick(logits))

    var other = config
    other.seed = 777
    var mine = Sampler(config, VOCAB)
    var theirs = Sampler(other, VOCAB)
    var unaffected = True
    for i in range(12):
        _ = theirs.pick(logits)
        if mine.pick(logits) != alone[i]:
            unaffected = False
        _ = theirs.pick(logits)
    suite.check(
        unaffected,
        "a second sequence drawing in between changes nothing about the first",
    )


def test_temperature(mut suite: Suite) raises:
    suite.group("sampler temperature")

    var logits = _logits()

    var cold = SamplerConfig()
    cold.temperature = 0.01
    cold.seed = 5
    var frozen = Sampler(cold, VOCAB)
    suite.check(
        _reachable(frozen, logits, 200) == 1,
        "a temperature near zero only ever reaches the most likely token",
    )

    var hot = SamplerConfig()
    hot.temperature = 100.0
    hot.seed = 5
    var flat = Sampler(hot, VOCAB)
    suite.check(
        _reachable(flat, logits, 200) == VOCAB,
        "and a large one reaches the whole vocabulary",
    )

    var one = SamplerConfig()
    one.temperature = 1.0
    one.seed = 5
    var plain = Sampler(one, VOCAB)
    suite.check(
        _reachable(plain, logits, 400) == VOCAB,
        "a temperature of one leaves every token reachable",
    )


def test_top_k(mut suite: Suite) raises:
    suite.group("sampler top-k")

    var logits = _logits()
    var config = SamplerConfig()
    config.temperature = 1.0
    config.seed = 11

    config.top_k = 1
    var one = Sampler(config, VOCAB)
    suite.check(
        _reachable(one, logits, 200) == 1,
        "a top-k of one is the argmax however hot it is",
    )

    config.top_k = 2
    var two = Sampler(config, VOCAB)
    suite.check(
        _reachable(two, logits, 400) == 2,
        "and a top-k of two reaches exactly the two most likely",
    )

    config.top_k = VOCAB + 10
    var all = Sampler(config, VOCAB)
    suite.check(
        _reachable(all, logits, 400) == VOCAB,
        "a top-k past the end of the vocabulary keeps all of it",
    )


def test_top_p(mut suite: Suite) raises:
    suite.group("sampler top-p")

    var logits = _logits()
    var config = SamplerConfig()
    config.temperature = 1.0
    config.seed = 13

    # The mass runs 0.5, then 0.75, then 0.875. A p of one half is reached by
    # the first token alone.
    config.top_p = 0.5
    var half = Sampler(config, VOCAB)
    suite.check(
        _reachable(half, logits, 200) == 1,
        "a top-p that the first token already covers keeps only it",
    )

    config.top_p = 0.7
    var most = Sampler(config, VOCAB)
    suite.check(
        _reachable(most, logits, 400) == 2,
        "and the token that carries the sum past the line is kept, not dropped",
    )

    config.top_p = 0.8
    var more = Sampler(config, VOCAB)
    suite.check(
        _reachable(more, logits, 600) == 3,
        "a p above three quarters reaches the third",
    )


def test_min_p(mut suite: Suite) raises:
    suite.group("sampler min-p")

    var logits = _logits()
    var config = SamplerConfig()
    config.temperature = 1.0
    config.seed = 17

    # Relative to the most likely, which here is one half. A min-p of 0.3 cuts
    # at 0.15 and keeps the half and the quarter.
    config.min_p = 0.3
    var strict = Sampler(config, VOCAB)
    suite.check(
        _reachable(strict, logits, 400) == 2,
        "min-p cuts at a fraction of the most likely token, not at an absolute",
    )

    config.min_p = 0.2
    var loose = Sampler(config, VOCAB)
    suite.check(
        _reachable(loose, logits, 600) == 3,
        "and lowering it lets the next one through",
    )

    # The same filter against a flat distribution keeps everything, which is
    # the property a fixed top-k does not have.
    var flat = Buffer(VOCAB)
    for i in range(VOCAB):
        flat.data[i] = 1.0
    config.min_p = 0.9
    var even = Sampler(config, VOCAB)
    suite.check(
        _reachable(even, flat, 400) == VOCAB,
        "a distribution with no favourite keeps all of it at any min-p",
    )


def test_typical(mut suite: Suite) raises:
    suite.group("sampler typical")

    var logits = _logits()
    var config = SamplerConfig()
    config.temperature = 1.0
    config.seed = 19

    config.typical_p = 0.4
    var narrow = Sampler(config, VOCAB)
    var reachable = _reachable(narrow, logits, 400)
    suite.check(
        reachable >= 1 and reachable < VOCAB,
        "a typical-p below one drops part of the vocabulary",
    )

    # The part it drops is not the tail. Ordering by distance from the
    # distribution's own entropy is what makes this filter different, and the
    # cheapest statement of that is that a flat distribution has every token at
    # exactly the entropy, so none of them is atypical.
    var flat = Buffer(VOCAB)
    for i in range(VOCAB):
        flat.data[i] = 1.0
    config.typical_p = 0.5
    var even = Sampler(config, VOCAB)
    var kept = _reachable(even, flat, 600)
    suite.check(
        kept >= 3,
        "and a distribution where every token is equally surprising keeps most",
    )


def test_penalties(mut suite: Suite) raises:
    suite.group("sampler penalties")

    var close = Buffer(VOCAB)
    close.data[0] = 3.0
    close.data[1] = 2.9
    close.data[2] = 1.0
    close.data[3] = 0.0
    close.data[4] = -1.0

    # Greedy throughout, so what is checked is the penalised ordering and not
    # a draw. Token zero starts ahead by a tenth of a logit.
    var config = SamplerConfig()
    var plain = Sampler(config, VOCAB)
    suite.check(plain.pick(close) == 0, "unpenalised, the first token wins")

    var freq = config
    freq.frequency_penalty = 0.2
    var counted = Sampler(freq, VOCAB)
    suite.check(
        counted.pick(close) == 0,
        "a frequency penalty does nothing to a token nobody has seen",
    )
    suite.check(
        counted.pick(close) == 1,
        "and takes it below its rival once it has been used once",
    )

    var presence = config
    presence.presence_penalty = 0.2
    var flagged = Sampler(presence, VOCAB)
    flagged.observe(0)
    flagged.observe(0)
    flagged.observe(0)
    suite.check(
        flagged.pick(close) == 1,
        "a presence penalty applies to a token that appeared at all",
    )

    var repeat = config
    repeat.repeat_penalty = 1.5
    var divided = Sampler(repeat, VOCAB)
    divided.observe(0)
    suite.check(
        divided.pick(close) == 1,
        "a repetition penalty divides a positive logit",
    )

    # The negative half of the rule. Token four is the only one below zero, so
    # penalising every other token has to leave it last rather than promote it.
    var negative = Sampler(repeat, VOCAB)
    for i in range(VOCAB):
        negative.observe(i)
    suite.check(
        negative.pick(close) != 4,
        "and multiplies a negative one, so a penalty never promotes a token",
    )

    var windowed = config
    windowed.frequency_penalty = 0.2
    windowed.repeat_last_n = 2
    var forgetting = Sampler(windowed, VOCAB)
    forgetting.observe(0)
    forgetting.observe(1)
    forgetting.observe(1)
    suite.check(
        forgetting.count_of(0) == 0 and forgetting.count_of(1) == 2,
        "the window forgets what fell off the back of it",
    )

    var off = config
    off.frequency_penalty = 0.2
    off.repeat_last_n = 0
    var unpenalised = Sampler(off, VOCAB)
    unpenalised.observe(0)
    unpenalised.observe(0)
    unpenalised.observe(0)
    suite.check(
        unpenalised.pick(close) == 0,
        "and a window of zero turns all three penalties off",
    )


def test_bias(mut suite: Suite) raises:
    suite.group("sampler logit bias")

    var logits = _logits()

    var config = SamplerConfig()
    var lifted = Sampler(config, VOCAB)
    lifted.bias(2, 10.0)
    suite.check(
        lifted.pick(logits) == 2,
        "a bias lifts a token past one the model preferred",
    )

    var banned = Sampler(config, VOCAB)
    banned.bias(0, -100.0)
    suite.check(
        banned.pick(logits) == 1,
        "and a large negative one is a ban on the token it names",
    )

    # Twice, because a caller that merged two maps means the sum. Token two is
    # an eighth against token zero's half, so one logit of bias leaves it
    # behind and two put it in front.
    var summed = Sampler(config, VOCAB)
    summed.bias(2, 1.0)
    summed.bias(2, 1.0)
    suite.check(
        summed.pick(logits) == 2, "two entries for the same token add up"
    )

    var short = Sampler(config, VOCAB)
    short.bias(2, 1.0)
    suite.check(
        short.pick(logits) == 0, "and half of it is not enough on its own"
    )

    # In logits, so the same bias means the same thing whatever the temperature
    # is. A bias applied to probabilities would be a different ban at every
    # temperature and the same request would behave differently.
    var hot = SamplerConfig()
    hot.temperature = 2.0
    hot.seed = 11
    var only = Sampler(hot, VOCAB)
    only.bias(0, -100.0)
    only.bias(1, -100.0)
    only.bias(3, -100.0)
    only.bias(4, -100.0)
    var stuck = True
    for _ in range(200):
        if only.pick(logits) != 2:
            stuck = False
    suite.check(stuck, "a ban holds at a temperature that would draw anything")

    # After the mask and before the penalties, which is the placement the two
    # of them compose in. The bias puts token two ahead by ten and a frequency
    # penalty of eleven per appearance takes it back off again.
    var both = SamplerConfig()
    both.frequency_penalty = 11.0
    var composed = Sampler(both, VOCAB)
    composed.bias(2, 10.0)
    suite.check(
        composed.pick(logits) == 2, "a biased token wins before it is used"
    )
    suite.check(
        composed.pick(logits) != 2,
        "and the penalties still see it afterwards",
    )

    var outside = Sampler(config, VOCAB)
    var failed = False
    try:
        outside.bias(VOCAB, 1.0)
    except:
        failed = True
    suite.check(failed, "a bias on a token outside the vocabulary is refused")


def test_logprobs(mut suite: Suite) raises:
    suite.group("sampler logprobs")

    var logits = _logits()
    var out = List[Float32]()
    for _ in range(VOCAB):
        out.append(0.0)

    var config = SamplerConfig()
    config.temperature = 0.5
    config.frequency_penalty = 1.0
    config.seed = 23
    var s = Sampler(config, VOCAB)
    s.observe(0)
    s.observe(0)
    s.logprobs(logits, out)

    var total = Float32(0)
    for i in range(VOCAB):
        total += exp(out[i])
    suite.check(
        _close(total, 1.0, 1e-5),
        "the logprobs are a distribution that sums to one",
    )

    # The distribution the model gave, not the one the sampler is going to draw
    # from. A temperature of a half and a frequency penalty are both set above
    # and neither may appear in these numbers.
    suite.check(
        _close(out[0], Float32(log(0.5)), 1e-5),
        "and they are the model's numbers, before the temperature",
    )
    suite.check(
        _close(out[1], Float32(log(0.25)), 1e-5)
        and _close(out[2], Float32(log(0.125)), 1e-5),
        "and before the penalties, whatever the sequence has been doing",
    )

    # Shifting every logit by a constant is the same distribution, and a
    # softmax that did not subtract its maximum would say otherwise.
    var shifted = Buffer(VOCAB)
    for i in range(VOCAB):
        shifted.data[i] = logits.data[i] + 500.0
    var also = List[Float32]()
    for _ in range(VOCAB):
        also.append(0.0)
    s.logprobs(shifted, also)
    var agree = True
    for i in range(VOCAB):
        if not _close(also[i], out[i], 1e-4):
            agree = False
    suite.check(agree, "adding a constant to every logit changes nothing")


def test_errors(mut suite: Suite) raises:
    suite.group("sampler errors")

    var config = SamplerConfig()
    var failed = False
    try:
        _ = Sampler(config, 0)
    except:
        failed = True
    suite.check(failed, "a sampler with no vocabulary is refused")

    var bad = config
    bad.temperature = -1.0
    failed = False
    try:
        _ = Sampler(bad, VOCAB)
    except:
        failed = True
    suite.check(failed, "so is a negative temperature")

    bad = config
    bad.top_p = 0.0
    failed = False
    try:
        _ = Sampler(bad, VOCAB)
    except:
        failed = True
    suite.check(failed, "and a top-p of zero, which would keep nothing")

    bad = config
    bad.top_p = 1.5
    failed = False
    try:
        _ = Sampler(bad, VOCAB)
    except:
        failed = True
    suite.check(failed, "and one above the whole distribution")

    bad = config
    bad.repeat_penalty = 0.0
    failed = False
    try:
        _ = Sampler(bad, VOCAB)
    except:
        failed = True
    suite.check(failed, "and a repetition penalty that would erase a logit")

    var s = Sampler(config, VOCAB)
    var wrong = Buffer(VOCAB + 1)
    failed = False
    try:
        _ = s.pick(wrong)
    except:
        failed = True
    suite.check(
        failed, "a logit row that is not the width of the vocabulary is refused"
    )

    var out = List[Float32]()
    for _ in range(VOCAB + 2):
        out.append(0.0)
    var logits = _logits()
    failed = False
    try:
        s.logprobs(logits, out)
    except:
        failed = True
    suite.check(failed, "and so is an output of the wrong width for logprobs")
