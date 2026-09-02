"""Checks a whole forward pass against llama.cpp, layer by layer.

Not part of the library and not part of the test suite. `tests/test_kernel.mojo`
catches a kernel that computes the wrong thing. Nothing in the suite catches
correct kernels wired together wrongly, because a right attention with a wrong
rope base is a stack of pieces that each pass their own test and produce fluent
text from the wrong distribution. That is what this is for.

For each case in the manifest it loads the GGUF that
`scripts/gen-logits.py` recorded llama.cpp against, prefills the token ids that
are written in the reference rather than tokenizing anything, and compares
three things.

**The residual stream after every layer.** The reference has one line per
snapshot with llama.cpp's sum over the tensor and six of its values. molla
records the same snapshots through `Scratch.tracing`, and the first snapshot
that disagrees is a layer number rather than a mismatch somewhere in a
hundred million multiplies. Snapshot zero is the embedding, so a failure there
is the lookup or the file offsets and not any layer at all, and the last
snapshot is the final norm, so a case whose snapshots all agree and whose
logits do not is the output head and nothing else.

**The top of the final distribution.** Sixty four tokens with their log
probabilities, in order.

**An evenly spaced sample of the rest of it.** A kernel that gets the ranking
right and the spacing wrong keeps every token in the head where it was, and
this is what notices.

Log probabilities and not logits, on both sides. A row of logits has an
arbitrary additive constant in it, and log softmax is the same row with that
constant removed, so comparing log probabilities compares the only part of the
answer that means anything.

The tolerances are absolute numbers written down in one place below with what
was measured against them. Read them before trusting a pass.

Usage:

    mojo run -I src scripts/logit_oracle.mojo \
        scripts/logit_cases.txt scripts/logits corpus/logits
"""

from std.math import exp, log, sqrt
from std.sys import argv, exit

from molla.engine.bind import bind
from molla.engine.session import Session as Decode
from molla.model.gguf import Gguf
from molla.model.load import load, plan_load
from molla.model.repack import model_key, open_cache
from molla.sys.device import default_device
from molla.sys.mmap import Mapping

comptime SUM_TOL = Float64(2e-3)
"""How far the sum over a snapshot may sit from llama.cpp's, as a fraction of
the total absolute mass of that snapshot.

Relative to the mass and not to the sum, for the reason `docs/validation/
kernel.md` gives about dot products: a sum over thousands of signed activations
cancels down to far less than the numbers that went into it, so an error that
is nothing against the activations can be most of the total. Measuring against
the total would fail on the snapshots that happen to cancel well.

Worst across the corpus is 5.1e-4, on Qwen 2.5. The eleven SmolLM2 cases are
all under 1.1e-4. A wrong rope base moves this by tens of per cent."""

comptime ELEM_TOL = Float64(8e-2)
"""How far one sampled value may sit from llama.cpp's, relative to the larger
of its own magnitude and the root mean square of the vector it sits in.

Worst across the corpus is 3.1e-2, on Llama 3.1, with Qwen 2.5 just under it
at 3.0e-2 and every SmolLM2 case under 1.5e-3. That spread is the reason this
number is where it is rather than twenty times tighter, and it is not
explained. It is not the quantization, because SmolLM2 at four bits agrees ten
times more closely than either larger model at four or five. See
`docs/validation/logits.md`."""

comptime LOGIT_TOL = Float64(2e-1)
"""How far one log probability may sit from llama.cpp's, absolute.

Absolute rather than relative because a log probability already is a logarithm,
so a fixed offset here is a fixed ratio of probability.

Worst across the corpus is 9.6e-2 on Qwen 2.5 and 6.7e-2 on Llama 3.1, against
1.0e-2 for the worst SmolLM2 case, which is the same spread the sampled values
show and almost certainly the same cause. The tail of the distribution runs a
little wider than the head on every case, which is expected: those numbers are
around minus twenty five and every rounding on the way there has been through
an exponential."""


def _read(path: String) raises -> List[UInt8]:
    var mapping = Mapping(path)
    var out = List[UInt8]()
    out.reserve(mapping.length)
    var data = Span[UInt8, MutAnyOrigin](
        unsafe_ptr=mapping.base(), length=mapping.length
    )
    for i in range(len(data)):
        out.append(data[i])
    mapping.close()
    return out^


def _lines(data: List[UInt8]) -> List[String]:
    var out = List[String]()
    var one = List[UInt8]()
    for i in range(len(data)):
        if data[i] == 10:
            out.append(String(StringSpan(unsafe_from_utf8=Span(one))))
            one.clear()
        else:
            one.append(data[i])
    if len(one) > 0:
        out.append(String(StringSpan(unsafe_from_utf8=Span(one))))
    return out^


def _split(line: String, sep: UInt8) -> List[String]:
    var out = List[String]()
    var one = List[UInt8]()
    var data = line.as_bytes()
    for i in range(len(data)):
        if data[i] == sep:
            out.append(String(StringSpan(unsafe_from_utf8=Span(one))))
            one.clear()
        else:
            one.append(data[i])
    out.append(String(StringSpan(unsafe_from_utf8=Span(one))))
    return out^


def _words(line: String) -> List[String]:
    """Split on spaces, dropping the empty runs a double space leaves."""
    var raw = _split(line, 32)
    var out = List[String]()
    for i in range(len(raw)):
        if raw[i].byte_length() > 0:
            out.append(raw[i])
    return out^


def _abs(x: Float64) -> Float64:
    return -x if x < 0 else x


def _max(a: Float64, b: Float64) -> Float64:
    return a if a > b else b


struct Snapshot(Copyable, Movable):
    """One `layer` line of a reference."""

    var rows: Int
    """How many positions llama.cpp still had in the tensor. It drops every
    position but the last once nothing downstream reads them, so this is the
    prompt length for most layers and one for the last."""

    var total: Float64
    var sampled: List[Float64]

    def __init__(out self):
        self.rows = 0
        self.total = 0.0
        self.sampled = List[Float64]()


struct Case(Movable):
    var name: String
    var model: String
    var bytes: Int
    var layers: Int
    var width: Int
    var vocab: Int
    var tokens: List[Int]
    var snapshots: List[Snapshot]
    var top_id: List[Int]
    var top_lp: List[Float64]
    var probe_id: List[Int]
    var probe_lp: List[Float64]

    def __init__(out self):
        self.name = String("")
        self.model = String("")
        self.bytes = 0
        self.layers = 0
        self.width = 0
        self.vocab = 0
        self.tokens = List[Int]()
        self.snapshots = List[Snapshot]()
        self.top_id = List[Int]()
        self.top_lp = List[Float64]()
        self.probe_id = List[Int]()
        self.probe_lp = List[Float64]()


def _parse(path: String) raises -> Case:
    var out = Case()
    var lines = _lines(_read(path))
    for i in range(len(lines)):
        var line = lines[i]
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        var w = _words(line)
        var key = w[0]
        if key == "name":
            out.name = w[1]
        elif key == "model":
            out.model = w[1]
        elif key == "bytes":
            out.bytes = atol(w[1])
        elif key == "layers":
            out.layers = atol(w[1])
        elif key == "width":
            out.width = atol(w[1])
        elif key == "vocab":
            out.vocab = atol(w[1])
        elif key == "tokens":
            for j in range(1, len(w)):
                out.tokens.append(atol(w[j]))
        elif key == "layer":
            var one = Snapshot()
            one.rows = atol(w[2])
            one.total = Float64(w[3])
            for j in range(4, len(w)):
                one.sampled.append(Float64(w[j]))
            if len(one.sampled) != 6:
                raise Error(
                    "a layer line with "
                    + String(len(one.sampled))
                    + " sampled values"
                )
            out.snapshots.append(one^)
        elif key == "top":
            out.top_id.append(atol(w[1]))
            out.top_lp.append(Float64(w[2]))
        elif key == "probe":
            out.probe_id.append(atol(w[1]))
            out.probe_lp.append(Float64(w[2]))
    if len(out.snapshots) != out.layers + 2:
        raise Error(
            String(len(out.snapshots))
            + " snapshots for a model with "
            + String(out.layers)
            + " layers, which wants two more than that"
        )
    if len(out.tokens) == 0:
        raise Error("a case with no tokens has nothing to run")
    return out^


def _log_softmax(logits: List[Float32]) -> List[Float64]:
    """Logits minus their log sum of exponentials.

    Shifted by the largest before the exponential, which is the ordinary
    guard: a logit of thirty exponentiates to ten to the thirteen and a row of
    them overflows a float32 long before the sum means anything.
    """
    var n = len(logits)
    var top = Float64(logits[0])
    for i in range(n):
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


def _found(spec: Case, token: Int) -> Float64:
    """llama.cpp's log probability for one token, from what was recorded.

    Only the top of the distribution and a sample of the tail are in the file,
    so a token in neither gets a number low enough that any comparison against
    it comes out large, which is the truth: llama.cpp put it below every one of
    the sixty four it did write down.
    """
    for i in range(len(spec.top_id)):
        if spec.top_id[i] == token:
            return spec.top_lp[i]
    for i in range(len(spec.probe_id)):
        if spec.probe_id[i] == token:
            return spec.probe_lp[i]
    return -1e30


def _check(spec: Case, model_dir: String) raises -> Int:
    """The number of complaints, or minus one when the model is not here.

    Prints one line per case, plus a detail line for each thing that
    disagreed. A missing model is a skip because no machine in the fleet has
    every file the corpus names and the large ones are large.
    """
    var path = model_dir + "/" + spec.model
    var size: Int
    try:
        var probe = Mapping(path)
        size = probe.length
        probe.close()
    except:
        print("  " + spec.name + "  skipped, no " + path)
        return -1
    if size != spec.bytes:
        print(
            "  "
            + spec.name
            + "  FAIL, "
            + spec.model
            + " is "
            + String(size)
            + " bytes and the reference was taken against "
            + String(spec.bytes)
        )
        return 1

    var g = Gguf(path)
    var cache = open_cache(path, model_key(g))
    var weights = load(g, plan_load(g, default_device(), 0), 0, False, "")
    var b = bind(g, cache)
    if b.width() != spec.width:
        raise Error(
            "the file is "
            + String(b.width())
            + " wide and the reference says "
            + String(spec.width)
        )
    if b.block_count() != spec.layers:
        raise Error(
            "the file has "
            + String(b.block_count())
            + " layers and the reference says "
            + String(spec.layers)
        )
    if b.vocab() != spec.vocab:
        raise Error(
            "the file has a vocabulary of "
            + String(b.vocab())
            + " and the reference says "
            + String(spec.vocab)
        )

    var d = Decode(b, len(spec.tokens) + 1)
    d.scratch.tracing = True
    d.prefill(b, spec.tokens)

    var n = len(spec.tokens)
    var w = spec.width
    var per = d.scratch.snapshots(w) // n
    if per != spec.layers + 2:
        raise Error(
            "molla took "
            + String(per)
            + " snapshots per token and the reference has "
            + String(spec.layers + 2)
        )

    var bad = 0
    var first = -1
    var worst_sum = Float64(0)
    var worst_elem = Float64(0)
    for k in range(per):
        var want = spec.snapshots[k].copy()
        # llama.cpp keeps every position until the last layer and then keeps
        # only the one the head reads, so the reference says how many rows its
        # sum covered and molla sums the same tail of the prompt.
        var from_pos = n - want.rows
        if from_pos < 0:
            from_pos = 0
        var total = Float64(0)
        var mass = Float64(0)
        var row_sq = Float64(0)
        for p in range(from_pos, n):
            var at = (p * per + k) * w
            for i in range(w):
                var v = Float64(d.scratch.trace[at + i])
                total += v
                mass += _abs(v)
                if p == n - 1:
                    row_sq += v * v
        var gap = _abs(total - want.total) / _max(mass, 1.0)
        if gap > worst_sum:
            worst_sum = gap
        var complained = gap > SUM_TOL
        if complained:
            bad += 1
            print(
                "  "
                + spec.name
                + "  snapshot "
                + String(k)
                + " sums to "
                + String(total)
                + " where llama.cpp had "
                + String(want.total)
                + ", "
                + String(gap)
                + " of its mass"
            )

        # A value is measured against its own magnitude or against the root
        # mean square of the vector it sits in, whichever is larger. The floor
        # matters because a residual stream is not flat: Qwen 2.5 carries a
        # handful of channels in the thousands next to hundreds that are around
        # a tenth, and the error a dot product makes in any one output is set
        # by the length of the whole input vector and not by the size of that
        # output. Asking a channel of 0.04 to land within a per cent of itself
        # while a channel of four thousand goes through the same multiply is
        # asking for a precision no arrangement of float32 was going to give.
        var scale = sqrt(row_sq / Float64(w))
        var last = ((n - 1) * per + k) * w
        var at_index = [0, 1, 2, w - 3, w - 2, w - 1]
        for j in range(6):
            var got = Float64(d.scratch.trace[last + at_index[j]])
            var theirs = want.sampled[j]
            var off = _abs(got - theirs) / _max(_abs(theirs), scale)
            if off > worst_elem:
                worst_elem = off
            if off > ELEM_TOL:
                bad += 1
                complained = True
                print(
                    "  "
                    + spec.name
                    + "  snapshot "
                    + String(k)
                    + " value "
                    + String(at_index[j])
                    + " is "
                    + String(got)
                    + " where llama.cpp had "
                    + String(theirs)
                )
        if complained and first < 0:
            first = k

    var logits = List[Float32]()
    logits.reserve(spec.vocab)
    for i in range(spec.vocab):
        logits.append(d.logits.data[i])
    var lp = _log_softmax(logits)

    var worst_top = Float64(0)
    var wrong_rank = 0
    for i in range(len(spec.top_id)):
        var off = _abs(lp[spec.top_id[i]] - spec.top_lp[i])
        if off > worst_top:
            worst_top = off
        if off > LOGIT_TOL:
            bad += 1
            print(
                "  "
                + spec.name
                + "  token "
                + String(spec.top_id[i])
                + " at rank "
                + String(i)
                + " has log probability "
                + String(lp[spec.top_id[i]])
                + " where llama.cpp had "
                + String(spec.top_lp[i])
            )
    # The ranking on its own, which is what a caller sampling greedily gets.
    # Reported and not asserted below rank zero, because two tokens a
    # thousandth apart in the tail swapping places is float arithmetic and not
    # a wrong kernel.
    for i in range(1, len(spec.top_id)):
        if lp[spec.top_id[i]] > lp[spec.top_id[i - 1]]:
            wrong_rank += 1
    var mine = 0
    for i in range(spec.vocab):
        if lp[i] > lp[mine]:
            mine = i
    if mine != spec.top_id[0]:
        # Losing the argmax only counts when llama.cpp had a clear winner. On
        # `The capital of France is` at four bits, llama.cpp puts ` Paris` and
        # ` a` three hundredths of a log probability apart, which is less than
        # the tolerance either of them is checked against, so which one comes
        # out first is a coin the arithmetic flips and not an answer either
        # program got wrong.
        var margin = spec.top_lp[0] - _found(spec, mine)
        if margin > LOGIT_TOL:
            bad += 1
            print(
                "  "
                + spec.name
                + "  greedy picks "
                + String(mine)
                + " where llama.cpp picks "
                + String(spec.top_id[0])
                + ", which llama.cpp had "
                + String(margin)
                + " ahead"
            )
        else:
            print(
                "  "
                + spec.name
                + "  greedy picks "
                + String(mine)
                + " over llama.cpp's "
                + String(spec.top_id[0])
                + ", "
                + String(margin)
                + " apart and inside the tolerance"
            )

    var worst_probe = Float64(0)
    for i in range(len(spec.probe_id)):
        var off = _abs(lp[spec.probe_id[i]] - spec.probe_lp[i])
        if off > worst_probe:
            worst_probe = off
        if off > LOGIT_TOL:
            bad += 1
            print(
                "  "
                + spec.name
                + "  token "
                + String(spec.probe_id[i])
                + " has log probability "
                + String(lp[spec.probe_id[i]])
                + " where llama.cpp had "
                + String(spec.probe_lp[i])
            )

    var line = "  " + spec.name + "  " + String(spec.layers) + " layers, "
    line += String(n) + " tokens"
    line += ", sum " + String(worst_sum)
    line += ", value " + String(worst_elem)
    line += ", head " + String(worst_top)
    line += ", tail " + String(worst_probe)
    if wrong_rank > 0:
        line += ", " + String(wrong_rank) + " swaps"
    if bad == 0:
        print(line + "  ok")
    else:
        print(line + "  FAIL, first disagreement in snapshot " + String(first))

    cache.close()
    g.close()
    _ = weights^
    return bad


def main() raises:
    var args = argv()
    if len(args) != 4:
        print("usage: logit_oracle <cases.txt> <reference-dir> <model-dir>")
        exit(2)
    var cases = String(args[1])
    var ref_dir = String(args[2])
    var model_dir = String(args[3])

    var names = List[String]()
    var lines = _lines(_read(cases))
    for i in range(len(lines)):
        if lines[i].byte_length() == 0 or lines[i].startswith("#"):
            continue
        names.append(_split(lines[i], 9)[0])

    var bad = 0
    var ran = 0
    var skipped = 0
    for i in range(len(names)):
        var path = ref_dir + "/" + names[i] + ".txt"
        var spec: Case
        try:
            spec = _parse(path)
        except e:
            print("  " + names[i] + "  no reference, " + String(e))
            skipped += 1
            continue
        var out = _check(spec, model_dir)
        if out < 0:
            skipped += 1
        else:
            bad += out
            ran += 1

    if ran == 0:
        print(
            "nothing ran, "
            + String(skipped)
            + " cases skipped. Run scripts/gen-logits.py to build them."
        )
        exit(1)
    if bad > 0:
        print(String(bad) + " disagreements with llama.cpp")
        exit(1)
    var line = String(ran) + " cases agree with llama.cpp"
    if skipped > 0:
        line += ", " + String(skipped) + " skipped"
    print(line)
