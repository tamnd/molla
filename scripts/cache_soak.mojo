"""Runs one long sequence through both cache forms and compares as it goes.

Not part of the library and not part of the test suite. The suite and the logit
corpus both read a cache a few positions after they wrote it, which is the one
thing a quantized cache is not at risk from. What a q8_0 cache actually risks is
a rounding that only matters once a key has been sitting in the cache for
thousands of positions and is being read by every token since. Nothing that runs
in a few seconds can see that, so this is the thing that looks.

The comparison is teacher forced. The same fixed token sequence goes through an
f16 session and a q8_0 session, one token at a time, and every token is the one
the text says rather than the one the model picked. Free running generation
would be the more natural soak and it is the wrong measurement: the two forms
disagree about one token early, the texts diverge, and from then on the two runs
are answering different questions and every later difference is that divergence
rather than the cache. Teacher forcing holds the prefix identical at every
position, so the only difference left between the two runs is what the cache did
to the keys and values.

Three numbers come out of it.

**Top one agreement, bucketed by position.** Whether the two forms predict the
same next token, counted over every position and reported in eighths of the run.
This is the number that answers the question, because a rounding that
accumulates shows up as a later eighth agreeing less than an earlier one.

**The divergence of the whole distribution.** At every checkpoint, the
Kullback Leibler divergence of the q8_0 row from the f16 row in nats, over the
full vocabulary. Agreement counts only the winner and this counts everything.

**The worst shift in the head.** The largest change in log probability over the
tokens f16 gave at least a thousandth of the mass to. A form that keeps the
ranking and stretches the spacing moves this and moves neither of the others.

The control pass is worth running at least once on a machine before believing
any of it. It runs f16 twice and compares those, and the answer has to be exact:
same pick at every position, zero divergence at every checkpoint. A run whose
control is not exact is measuring the machine and not the cache.

Usage:

    mojo run -I src scripts/cache_soak.mojo model.gguf tok.json text.txt \\
        --context=4096 --every=256 --device=auto [--control]
"""

from std.math import exp, log
from std.sys import argv, exit

from max.gpu.host import DeviceContext

from molla.engine.backend import Backend, parse_backend, pick
from molla.engine.bind import Bound, bind
from molla.engine.device import DeviceSession, device_context, load_on_device
from molla.model.gguf import Gguf
from molla.model.repack import model_key, open_cache
from molla.model.spec import read_geometry
from molla.nn.repack import CACHE_F16, CACHE_Q8, cache_type_name
from molla.sys.clock import monotonic_ms
from molla.sys.device import devices
from molla.sys.mem import AllocCounter
from molla.sys.mmap import Mapping
from molla.tokenizer.tokenizer import Session, Tokenizer

comptime KL_TOL = Float64(4e-3)
"""How far the q8_0 distribution may sit from the f16 one at a checkpoint.

Nats, and the whole vocabulary rather than the head. For scale, two rows that
disagree about nothing but the fourth decimal of every log probability land near
1e-6, and a row that has moved a token out of the top ten lands near 1e-2.

Worst measured is 3.1e-4 on Llama 3.1 8B at Q4_K_M over 8192 positions and
1.5e-3 on SmolLM2 135M over 512. The number is where it is because of the small
model and not the large one, which is the same spread the logit corpus shows and
is probably the same cause: a 135M has fewer heads to average an error over and
a flatter distribution for it to move."""

comptime GROWTH_TOL = Float64(4.0)
"""How much larger the divergence in the last quarter of the run may be than in
the first quarter.

The point of the whole exercise is this ratio and not the absolute number. A
cache whose error is bounded gives something near one, because a key rounded at
position 40 is no worse when it is read at position 4000 than when it was read
at position 41. A cache whose error accumulates gives a number that grows with
the context and there is no context long enough to be safe from it."""

comptime AGREE_TOL = Float64(0.90)
"""How often the two forms have to predict the same next token, in any eighth of
the run and not only overall.

Any eighth rather than overall, because an overall number that stays high while
the last eighth collapses is the exact failure this is looking for."""

comptime HEAD_MASS = Float64(1e-3)
"""How much probability f16 has to give a token before its log probability is
worth comparing. Below this the log is a large negative number arrived at
through an exponential and it moves for reasons that have nothing to do with the
cache."""


struct Trace(Movable):
    """What one pass over the sequence leaves behind.

    Copied out of the session rather than borrowed from it, because the second
    pass reuses the same memory the first one ran in.
    """

    var picks: List[Int]
    """The argmax at every position, which is the model's next token."""

    var rows: List[List[Float64]]
    """Log probabilities over the whole vocabulary, at the checkpoints."""

    var at: List[Int]
    """Which position each of those rows was taken at."""

    var ms: Int

    def __init__(out self):
        self.picks = List[Int]()
        self.rows = List[List[Float64]]()
        self.at = List[Int]()
        self.ms = 0


def _read(path: String) raises -> String:
    var mapping = Mapping(path)
    var data = Span[UInt8, MutAnyOrigin](
        unsafe_ptr=mapping.base(), length=mapping.length
    )
    var out = String(StringSpan(unsafe_from_utf8=data))
    mapping.close()
    return out^


def _log_softmax(logits: List[Float32]) raises -> List[Float64]:
    """The row with its arbitrary additive constant removed.

    Log probabilities and not logits, for the reason the logit oracle gives: a
    row of logits means nothing until the constant is gone, and two rows that
    differ by a constant are the same distribution.
    """
    var n = len(logits)
    if n == 0:
        raise Error("a distribution over no tokens is not one")
    var top = Float64(logits[0])
    for i in range(1, n):
        if Float64(logits[i]) > top:
            top = Float64(logits[i])
    var acc = Float64(0)
    for i in range(n):
        acc += exp(Float64(logits[i]) - top)
    var lse = top + log(acc)
    var out = List[Float64]()
    out.reserve(n)
    for i in range(n):
        out.append(Float64(logits[i]) - lse)
    return out^


def _kl(p: List[Float64], q: List[Float64]) -> Float64:
    """The divergence of `q` from `p`, in nats, both of them log probabilities.

    This direction and not the other, because `p` is f16 and the question is how
    much of what f16 believes q8_0 has lost. The reverse divergence would weight
    the tail, which is where a quantized cache is allowed to be wrong.
    """
    var acc = Float64(0)
    for i in range(len(p)):
        acc += exp(p[i]) * (p[i] - q[i])
    return acc if acc > 0 else 0


def _head_shift(p: List[Float64], q: List[Float64]) -> Float64:
    """The worst move in log probability among the tokens f16 cares about."""
    var floor = log(HEAD_MASS)
    var worst = Float64(0)
    for i in range(len(p)):
        if p[i] < floor:
            continue
        var d = p[i] - q[i]
        if d < 0:
            d = -d
        if d > worst:
            worst = d
    return worst


def _rank(row: List[Float64], token: Int) -> Int:
    """Where `token` sits in `row`, counting from one."""
    var above = 0
    for i in range(len(row)):
        if row[i] > row[token]:
            above += 1
    return above + 1


def _pass(
    ctx: DeviceContext,
    host: Bound,
    b: Bound,
    context: Int,
    form: Int,
    tokens: List[Int],
    every: Int,
) raises -> Trace:
    """The sequence through one session, a token at a time, watched all the way.

    A token at a time and not in chunks, on purpose. A chunked prefill is the
    fast path and it is not the path a conversation spends its life in, and more
    to the point a chunk only reads the cache once for the whole chunk. Stepping
    means every position reads every position before it, which is the traffic
    this is asking a question about.
    """
    var s = DeviceSession(ctx, host, b, context, form)
    var vocab = b.vocab()
    var out = Trace()
    out.picks.reserve(len(tokens))
    var started = monotonic_ms()
    for i in range(len(tokens)):
        s.step(tokens[i])
        s.fetch()
        var top = 0
        var best = s.logits.data[0]
        for v in range(1, vocab):
            if s.logits.data[v] > best:
                best = s.logits.data[v]
                top = v
        out.picks.append(top)
        if (i + 1) % every == 0 or i == len(tokens) - 1:
            var row = List[Float32]()
            row.reserve(vocab)
            for v in range(vocab):
                row.append(s.logits.data[v])
            out.rows.append(_log_softmax(row))
            out.at.append(i)
    out.ms = monotonic_ms() - started
    _ = s^
    return out^


def _fill(base: List[Int], want: Int) raises -> List[Int]:
    """The token sequence, tiled until it fills the context.

    Tiled from the second token rather than the first, so the beginning of
    sequence marker appears once and at the beginning, which is the only place a
    model has ever been shown one.
    """
    if len(base) == 0:
        raise Error("the text encoded to no tokens")
    var out = List[Int]()
    for i in range(len(base)):
        out.append(base[i])
    while len(out) < want:
        if len(base) == 1:
            out.append(base[0])
            continue
        for i in range(1, len(base)):
            if len(out) >= want:
                break
            out.append(base[i])
    while len(out) > want:
        _ = out.pop()
    return out^


def _agreement(a: Trace, b: Trace, parts: Int) raises -> List[Float64]:
    """How often the two agreed, in `parts` equal stretches of the run."""
    var n = len(a.picks)
    if n != len(b.picks):
        raise Error("the two passes ran different numbers of tokens")
    var out = List[Float64]()
    for p in range(parts):
        var lo = n * p // parts
        var hi = n * (p + 1) // parts
        var same = 0
        for i in range(lo, hi):
            if a.picks[i] == b.picks[i]:
                same += 1
        out.append(Float64(same) / Float64(hi - lo) if hi > lo else 1.0)
    return out^


def _mean(values: List[Float64], lo: Int, hi: Int) -> Float64:
    var acc = Float64(0)
    for i in range(lo, hi):
        acc += values[i]
    return acc / Float64(hi - lo) if hi > lo else 0


def _compare(name: String, a: Trace, b: Trace, exact: Bool) raises -> Int:
    """Print the comparison and return the number of complaints."""
    var bad = 0
    var parts = 8
    if len(a.picks) < parts:
        parts = 1
    var agree = _agreement(a, b, parts)

    var kls = List[Float64]()
    print("  " + name)
    print("    position   divergence   head shift   f16 top rank")
    for c in range(len(a.rows)):
        var kl = _kl(a.rows[c], b.rows[c])
        kls.append(kl)
        var shift = _head_shift(a.rows[c], b.rows[c])
        var rank = _rank(b.rows[c], a.picks[a.at[c]])
        print(
            "    "
            + String(a.at[c])
            + "   "
            + String(kl)
            + "   "
            + String(shift)
            + "   "
            + String(rank)
        )
        if exact and kl != 0:
            print("    the control diverged at position " + String(a.at[c]))
            bad += 1
        if not exact and kl > KL_TOL:
            print(
                "    divergence "
                + String(kl)
                + " at position "
                + String(a.at[c])
                + " is over "
                + String(KL_TOL)
            )
            bad += 1

    var line = String("    agreement by eighth  ")
    for p in range(len(agree)):
        line += String(agree[p]) + " "
    print(line)
    for p in range(len(agree)):
        if exact and agree[p] != 1.0:
            print("    the control picked differently in eighth " + String(p))
            bad += 1
            break
        if not exact and agree[p] < AGREE_TOL:
            print(
                "    agreement "
                + String(agree[p])
                + " in eighth "
                + String(p)
                + " is under "
                + String(AGREE_TOL)
            )
            bad += 1

    if len(kls) >= 4 and not exact:
        var quarter = len(kls) // 4
        var early = _mean(kls, 0, quarter)
        var late = _mean(kls, len(kls) - quarter, len(kls))
        var grew = late / early if early > 0 else 0
        print(
            "    divergence early "
            + String(early)
            + ", late "
            + String(late)
            + ", grew by "
            + String(grew)
        )
        if early > 0 and grew > GROWTH_TOL:
            print(
                "    the divergence grew by "
                + String(grew)
                + ", over "
                + String(GROWTH_TOL)
            )
            bad += 1
    return bad


def _resolve(value: String) raises -> Backend:
    var all = devices()
    var fits = List[Bool]()
    for _ in range(len(all)):
        fits.append(True)
    return pick(parse_backend(value), all, fits)


def _flag(arg: String, name: String) -> String:
    if not arg.startswith(name):
        return String("")
    return String(arg[byte = name.byte_length() : arg.byte_length()])


def main() raises:
    var args = argv()
    if len(args) < 4:
        print(
            "usage: cache_soak <model.gguf> <tokenizer.json> <text file>"
            " [--context=4096] [--every=256] [--device=auto] [--control]"
        )
        exit(2)
    var model_path = String(args[1])
    var tokenizer_path = String(args[2])
    var text_path = String(args[3])
    var context = 4096
    var every = 256
    var want = String("auto")
    var control = False
    for i in range(4, len(args)):
        var arg = String(args[i])
        if arg == "--control":
            control = True
        elif _flag(arg, "--context=") != "":
            context = Int(_flag(arg, "--context="))
        elif _flag(arg, "--every=") != "":
            every = Int(_flag(arg, "--every="))
        elif _flag(arg, "--device=") != "":
            want = _flag(arg, "--device=")
        else:
            print("unknown flag " + arg)
            exit(2)
    if every <= 0 or context <= every:
        print("--every has to be positive and smaller than the context")
        exit(2)

    var backend = _resolve(want)
    if not backend.on_device:
        print(
            "this needs a device. The host path holds no quantized cache and"
            " there is nothing here to compare"
        )
        exit(2)
    print("backend  " + backend.describe())

    var g = Gguf(model_path)
    var cache = open_cache(model_path, model_key(g))
    var ctx = device_context(backend.device.index)
    var weights = load_on_device(g, cache, model_path, backend.device, ctx)
    var b = bind(g, cache, weights.residency())
    var host = bind(g, cache)

    var geometry = read_geometry(g)
    if geometry.context_length > 0 and context > geometry.context_length:
        context = geometry.context_length

    var counter = AllocCounter()
    var tokenizer = Tokenizer(tokenizer_path, counter.raw())
    var session = Session()
    var base = List[Int]()
    tokenizer.encode(_read(text_path), True, session, base)
    var tokens = _fill(base, context)
    print(
        "sequence  "
        + String(len(tokens))
        + " tokens, "
        + String(len(base))
        + " of text tiled to fill a context of "
        + String(context)
        + ", "
        + String(len(tokens) // every)
        + " checkpoints"
    )

    var bad = 0
    var f16 = _pass(ctx, host, b, context, CACHE_F16, tokens, every)
    print("  " + cache_type_name(CACHE_F16) + "  " + String(f16.ms) + " ms")
    if control:
        var again = _pass(ctx, host, b, context, CACHE_F16, tokens, every)
        print(
            "  "
            + cache_type_name(CACHE_F16)
            + " again  "
            + String(again.ms)
            + " ms"
        )
        bad += _compare(String("f16 against itself"), f16, again, True)
    var q8 = _pass(ctx, host, b, context, CACHE_Q8, tokens, every)
    print("  " + cache_type_name(CACHE_Q8) + "  " + String(q8.ms) + " ms")
    bad += _compare(String("q8_0 against f16"), f16, q8, False)

    _ = weights^
    cache.close()
    g.close()
    if bad > 0:
        print(String(bad) + " complaints on " + backend.describe())
        exit(1)
    print(
        "the q8_0 cache holds up over "
        + String(len(tokens))
        + " positions on "
        + backend.describe()
    )
