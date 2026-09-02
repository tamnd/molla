"""One transformer layer, against a reference written from scratch.

The kernels in `molla.nn.kernel` are already checked against dequantizing the
same bytes, and rope against ggml's own loop. What a layer adds on top of those
is the order they get called in and which argument goes where, and that is not
something an oracle for arithmetic can catch. A layer that runs the mlp before
attention, or normalises the output instead of the input, or rotates the key
after storing it, produces numbers in the right range every time.

So the reference here is a second implementation rather than a second source of
numbers. It walks plain `List[Float32]` weights with naive loops and does not
call anything in `molla.nn`, apart from `exp`, `sqrt`, `cos` and `sin` out of
the standard library. If `block.mojo` calls things in the wrong order, the two
disagree.

There is one case with hand computed values as well, because two
implementations that agree can still both be wrong. Identity projections, gains
of one and a single key at position zero make a layer collapse to
`x + normed(x)`, which is arithmetic a person can check.
"""

from std.math import cos, exp, sin, sqrt, tanh

from harness import Suite

from molla.nn.attention import AttnSpec
from molla.nn.block import (
    ACT_GELU,
    ACT_SILU,
    BlockSpec,
    LayerWeights,
    Scratch,
    activation_name,
    attention_layer,
    layer,
    mlp_layer,
)
from molla.nn.quant import Q_F32
from molla.nn.rope import RopeSpec
from molla.nn.tensor import Buffer, Tensor
from molla.sys.mem import keep

comptime WIDTH = 8
comptime HEADS = 2
comptime HEAD_DIM = 4
comptime HIDDEN = 8
comptime EPS = Float32(1e-5)


def _close(a: Float32, b: Float32, within: Float32) -> Bool:
    var d = a - b
    if d < 0:
        d = -d
    return d <= within


def _f32_bytes(values: List[Float32]) -> List[UInt8]:
    var out = List[UInt8]()
    for v in values:
        var bits = Int(v.to_bits())
        for shift in range(4):
            out.append(UInt8((bits >> (shift * 8)) & 0xFF))
    return out^


struct Arena(Movable):
    """Somewhere for weight bytes to live for as long as the tensors do.

    A `Tensor` holds an integer address, so nothing the compiler can see keeps
    the bytes alive. One list of byte lists, kept alive with `keep` at the end
    of each test, is one place to get that right instead of a dozen.
    """

    var held: List[List[UInt8]]

    def __init__(out self):
        self.held = List[List[UInt8]]()

    def tensor(mut self, values: List[Float32], cols: Int, rows: Int) -> Tensor:
        self.held.append(_f32_bytes(values))
        var last = len(self.held) - 1
        return Tensor(Int(self.held[last].unsafe_ptr()), Q_F32, cols, rows)


def _ones(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(1.0)
    return out^


def _identity(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for r in range(n):
        for c in range(n):
            out.append(Float32(1.0) if r == c else Float32(0.0))
    return out^


def _dense(cols: Int, rows: Int, seed: Int) -> List[Float32]:
    """Small deterministic weights, spread either side of zero.

    Not random. A layer test that fails should fail the same way twice, and the
    values only have to be varied enough that a transposed index or a swapped
    projection shows up.
    """
    var out = List[Float32]()
    for r in range(rows):
        for c in range(cols):
            var n = (r * 13 + c * 7 + seed * 5) % 9
            out.append(Float32(n - 4) / 8.0)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


def _ref_rms(mut x: List[Float32], at: Int, n: Int, gain: List[Float32]):
    var sum = Float64(0)
    for i in range(n):
        sum += Float64(x[at + i]) * Float64(x[at + i])
    var scale = Float32(1.0 / sqrt(sum / Float64(n) + Float64(EPS)))
    for i in range(n):
        x[at + i] = x[at + i] * scale * gain[i]


def _ref_matvec(
    w: List[Float32],
    cols: Int,
    rows: Int,
    x: List[Float32],
    at_in: Int,
    mut out: List[Float32],
    at_out: Int,
):
    for r in range(rows):
        var sum = Float32(0)
        for c in range(cols):
            sum += w[r * cols + c] * x[at_in + c]
        out[at_out + r] = sum


def _ref_rope(mut x: List[Float32], at: Int, pos: Int, base: Float32):
    """Neox pairing, no scaling, the whole head rotated."""
    var pairs = HEAD_DIM // 2
    var step = Float32(1.0)
    var ratio = Float32(1.0) / Float32(sqrt(Float64(base)))
    for pair in range(pairs):
        var theta = Float32(pos) * step
        var c = Float32(cos(Float64(theta)))
        var s = Float32(sin(Float64(theta)))
        var lo = at + pair
        var hi = lo + pairs
        var a = x[lo]
        var b = x[hi]
        x[lo] = a * c - b * s
        x[hi] = a * s + b * c
        step = step * ratio


def _ref_attend(
    q: List[Float32],
    keys: List[Float32],
    values: List[Float32],
    count: Int,
    kv_heads: Int,
    mut out: List[Float32],
):
    var scale = Float32(1.0 / sqrt(Float64(HEAD_DIM)))
    var group = HEADS // kv_heads
    var kv_width = kv_heads * HEAD_DIM
    for h in range(HEADS):
        var kh = h // group
        var raw = List[Float32]()
        for t in range(count):
            var dot = Float32(0)
            for d in range(HEAD_DIM):
                dot += (
                    q[h * HEAD_DIM + d] * keys[t * kv_width + kh * HEAD_DIM + d]
                )
            raw.append(dot * scale)
        var top = raw[0]
        for t in range(count):
            if raw[t] > top:
                top = raw[t]
        var total = Float32(0)
        for t in range(count):
            raw[t] = Float32(exp(Float64(raw[t] - top)))
            total += raw[t]
        for d in range(HEAD_DIM):
            var sum = Float32(0)
            for t in range(count):
                sum += raw[t] / total * values[t * kv_width + kh * HEAD_DIM + d]
            out[h * HEAD_DIM + d] = sum


def _ref_silu(v: Float32) -> Float32:
    return v / (1.0 + Float32(exp(Float64(-v))))


def _ref_gelu(v: Float32) -> Float32:
    var c = Float32(0.7978845608028654)
    var inner = c * (v + 0.044715 * v * v * v)
    return 0.5 * v * (1.0 + Float32(tanh(Float64(inner))))


def run(mut suite: Suite) raises:
    test_names(suite)
    test_hand(suite)
    test_against_reference(suite)
    test_qk_norm(suite)
    test_post_norms(suite)
    test_mlp_shapes(suite)
    test_cache(suite)
    test_errors(suite)


def _spec() raises -> BlockSpec:
    return BlockSpec(
        AttnSpec(HEADS, HEADS, HEAD_DIM),
        RopeSpec(HEAD_DIM, 10000.0),
        WIDTH,
        HIDDEN,
        EPS,
    )


def test_names(mut suite: Suite) raises:
    suite.group("block activations")
    suite.check(
        activation_name(ACT_SILU) == "silu"
        and activation_name(ACT_GELU) == "gelu",
        "the two activations have names",
    )
    suite.check(
        activation_name(99) == "unknown",
        "and one that is not in the list says so rather than picking",
    )


def test_hand(mut suite: Suite) raises:
    """Identity weights, gains of one, one key at position zero.

    Every projection passes its input through, rope at position zero is a
    rotation by nothing, and a softmax over one score is one. So attention
    collapses to `x + normed(x)` and the mlp to `x + silu(normed(x)) *
    normed(x)`, both of which are worked out below rather than produced by
    anything.
    """
    suite.group("block by hand")

    var arena = Arena()
    var spec = _spec()
    var w = LayerWeights()
    w.attn_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    w.ffn_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    w.wq = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    w.wk = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    w.wv = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    w.wo = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    w.gate = arena.tensor(_identity(WIDTH), WIDTH, HIDDEN)
    w.up = arena.tensor(_identity(WIDTH), WIDTH, HIDDEN)
    w.down = arena.tensor(_identity(HIDDEN), HIDDEN, WIDTH)
    w.check(spec)

    var x = Buffer(WIDTH)
    for i in range(WIDTH):
        x.data[i] = Float32(i + 1)

    var s = Scratch(spec, 4)
    var keys = _zeros(WIDTH)
    var values = _zeros(WIDTH)
    attention_layer(spec, w, x, s, keys, values, 0, 0, List[Float32](), False)

    # The mean square of one through eight is 204 / 8, which is 25.5.
    var divisor = Float32(sqrt(Float64(25.5) + Float64(EPS)))
    var ok = True
    for i in range(WIDTH):
        var v = Float32(i + 1)
        if not _close(x.data[i], v + v / divisor, 1e-5):
            ok = False
    suite.check(ok, "attention with identity weights adds the normed input")

    suite.check(
        _close(keys[0], 1.0 / divisor, 1e-6),
        "and the key that went into the cache is the normed input too",
    )
    suite.check(_close(values[3], 4.0 / divisor, 1e-6), "and so is the value")

    # x is now a scalar multiple of one through eight, and an rms norm is scale
    # invariant, so the mlp normalises back to exactly the same vector.
    var normed = List[Float32]()
    for i in range(WIDTH):
        normed.append(Float32(i + 1) / divisor)
    var before = List[Float32]()
    for i in range(WIDTH):
        before.append(x.data[i])

    mlp_layer(spec, w, x, s)
    ok = True
    for i in range(WIDTH):
        var want = before[i] + _ref_silu(normed[i]) * normed[i]
        if not _close(x.data[i], want, 1e-5):
            ok = False
    suite.check(ok, "and a gated silu mlp adds silu of the normed input by it")

    keep(arena)


def test_against_reference(mut suite: Suite) raises:
    """Weights that are not identity, at a position that is not zero."""
    suite.group("block against a reference")

    var arena = Arena()
    var spec = _spec()

    var attn_gain = _dense(WIDTH, 1, 1)
    var ffn_gain = _dense(WIDTH, 1, 2)
    var wq = _dense(WIDTH, WIDTH, 3)
    var wk = _dense(WIDTH, WIDTH, 4)
    var wv = _dense(WIDTH, WIDTH, 5)
    var wo = _dense(WIDTH, WIDTH, 6)
    var gate = _dense(WIDTH, HIDDEN, 7)
    var up = _dense(WIDTH, HIDDEN, 8)
    var down = _dense(HIDDEN, WIDTH, 9)

    var w = LayerWeights()
    w.attn_norm = arena.tensor(attn_gain, WIDTH, 1)
    w.ffn_norm = arena.tensor(ffn_gain, WIDTH, 1)
    w.wq = arena.tensor(wq, WIDTH, WIDTH)
    w.wk = arena.tensor(wk, WIDTH, WIDTH)
    w.wv = arena.tensor(wv, WIDTH, WIDTH)
    w.wo = arena.tensor(wo, WIDTH, WIDTH)
    w.gate = arena.tensor(gate, WIDTH, HIDDEN)
    w.up = arena.tensor(up, WIDTH, HIDDEN)
    w.down = arena.tensor(down, HIDDEN, WIDTH)
    w.check(spec)

    var start = List[Float32]()
    for i in range(WIDTH):
        start.append(Float32(i % 5) - 2.0 + Float32(i) / 10.0)

    # Two keys already in the cache, so the softmax has something to do and the
    # new key lands at slot two.
    var seeded_k = _dense(WIDTH * 3, 1, 11)
    var seeded_v = _dense(WIDTH * 3, 1, 12)

    var x = Buffer(WIDTH)
    for i in range(WIDTH):
        x.data[i] = start[i]
    var keys = List[Float32]()
    var values = List[Float32]()
    for i in range(WIDTH * 3):
        keys.append(seeded_k[i])
        values.append(seeded_v[i])
    var s = Scratch(spec, 8)
    layer(spec, w, x, s, keys, values, 2, 2, List[Float32](), False)

    # The reference, written out.
    var r = List[Float32]()
    for i in range(WIDTH):
        r.append(start[i])
    var rk = List[Float32]()
    var rv = List[Float32]()
    for i in range(WIDTH * 3):
        rk.append(seeded_k[i])
        rv.append(seeded_v[i])

    var normed = List[Float32]()
    for i in range(WIDTH):
        normed.append(r[i])
    _ref_rms(normed, 0, WIDTH, attn_gain)

    var q = _zeros(WIDTH)
    _ref_matvec(wq, WIDTH, WIDTH, normed, 0, q, 0)
    _ref_matvec(wk, WIDTH, WIDTH, normed, 0, rk, WIDTH * 2)
    _ref_matvec(wv, WIDTH, WIDTH, normed, 0, rv, WIDTH * 2)
    for h in range(HEADS):
        _ref_rope(q, h * HEAD_DIM, 2, 10000.0)
        _ref_rope(rk, WIDTH * 2 + h * HEAD_DIM, 2, 10000.0)

    var mixed = _zeros(WIDTH)
    _ref_attend(q, rk, rv, 3, HEADS, mixed)
    var projected = _zeros(WIDTH)
    _ref_matvec(wo, WIDTH, WIDTH, mixed, 0, projected, 0)
    for i in range(WIDTH):
        r[i] += projected[i]

    var normed2 = List[Float32]()
    for i in range(WIDTH):
        normed2.append(r[i])
    _ref_rms(normed2, 0, WIDTH, ffn_gain)
    var g = _zeros(HIDDEN)
    var u = _zeros(HIDDEN)
    _ref_matvec(gate, WIDTH, HIDDEN, normed2, 0, g, 0)
    _ref_matvec(up, WIDTH, HIDDEN, normed2, 0, u, 0)
    for i in range(HIDDEN):
        g[i] = _ref_silu(g[i]) * u[i]
    var out2 = _zeros(WIDTH)
    _ref_matvec(down, HIDDEN, WIDTH, g, 0, out2, 0)
    for i in range(WIDTH):
        r[i] += out2[i]

    var ok = True
    for i in range(WIDTH):
        if not _close(x.data[i], r[i], 1e-4):
            ok = False
    suite.check(ok, "a whole layer matches a second implementation of it")

    var cached = True
    for i in range(WIDTH):
        if not _close(keys[WIDTH * 2 + i], rk[WIDTH * 2 + i], 1e-5):
            cached = False
        if not _close(values[WIDTH * 2 + i], rv[WIDTH * 2 + i], 1e-5):
            cached = False
    suite.check(cached, "and so does the key and value it left in the cache")

    var untouched = True
    for i in range(WIDTH * 2):
        if keys[i] != seeded_k[i] or values[i] != seeded_v[i]:
            untouched = False
    suite.check(untouched, "and the slots before it are not written to")

    keep(arena)


def test_qk_norm(mut suite: Suite) raises:
    """Qwen 3's per head norms, which no metadata key announces."""
    suite.group("block query and key norms")

    var arena = Arena()
    var spec = _spec()
    var w = LayerWeights()
    w.attn_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    w.ffn_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    w.wq = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    w.wk = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    w.wv = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    w.wo = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    w.gate = arena.tensor(_identity(WIDTH), WIDTH, HIDDEN)
    w.up = arena.tensor(_identity(WIDTH), WIDTH, HIDDEN)
    w.down = arena.tensor(_identity(HIDDEN), HIDDEN, WIDTH)
    w.q_norm = arena.tensor(_ones(HEAD_DIM), HEAD_DIM, 1)
    w.k_norm = arena.tensor(_ones(HEAD_DIM), HEAD_DIM, 1)
    w.check(spec)

    var x = Buffer(WIDTH)
    for i in range(WIDTH):
        x.data[i] = Float32(i + 1)
    var s = Scratch(spec, 4)
    var keys = _zeros(WIDTH)
    var values = _zeros(WIDTH)
    attention_layer(spec, w, x, s, keys, values, 0, 0, List[Float32](), False)

    # The key was normalised per head, so each head of it has a root mean
    # square of one. The value was not, which is what says the norm went to the
    # right two of the three projections.
    var ok = True
    for h in range(HEADS):
        var sum = Float64(0)
        for d in range(HEAD_DIM):
            var v = Float64(keys[h * HEAD_DIM + d])
            sum += v * v
        if not _close(Float32(sqrt(sum / Float64(HEAD_DIM))), 1.0, 1e-4):
            ok = False
    suite.check(ok, "each head of the key comes out with unit root mean square")

    var divisor = Float32(sqrt(Float64(25.5) + Float64(EPS)))
    suite.check(
        _close(values[0], 1.0 / divisor, 1e-6),
        "and the value is not normalised, only the query and the key",
    )

    # The same layer without the norms gives a different answer, which is the
    # check that they are doing anything at all.
    var plain = LayerWeights()
    plain.attn_norm = w.attn_norm
    plain.ffn_norm = w.ffn_norm
    plain.wq = w.wq
    plain.wk = w.wk
    plain.wv = w.wv
    plain.wo = w.wo
    plain.gate = w.gate
    plain.up = w.up
    plain.down = w.down
    var y = Buffer(WIDTH)
    for i in range(WIDTH):
        y.data[i] = Float32(i + 1)
    var keys2 = _zeros(WIDTH)
    var values2 = _zeros(WIDTH)
    attention_layer(
        spec, plain, y, s, keys2, values2, 0, 0, List[Float32](), False
    )
    suite.check(
        not _close(keys[7], keys2[7], 1e-4),
        "and leaving the norms out changes the key that gets cached",
    )

    keep(arena)


def test_post_norms(mut suite: Suite) raises:
    """Gemma's norms on the way out of a sublayer."""
    suite.group("block post norms")

    var arena = Arena()
    var spec = _spec()
    var w = LayerWeights()
    w.attn_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    w.ffn_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    w.wq = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    w.wk = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    w.wv = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    w.wo = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    w.gate = arena.tensor(_identity(WIDTH), WIDTH, HIDDEN)
    w.up = arena.tensor(_identity(WIDTH), WIDTH, HIDDEN)
    w.down = arena.tensor(_identity(HIDDEN), HIDDEN, WIDTH)
    w.attn_post_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    w.check(spec)

    var x = Buffer(WIDTH)
    for i in range(WIDTH):
        x.data[i] = Float32(i + 1)
    var s = Scratch(spec, 4)
    var keys = _zeros(WIDTH)
    var values = _zeros(WIDTH)
    attention_layer(spec, w, x, s, keys, values, 0, 0, List[Float32](), False)

    # Without the post norm the sublayer adds `normed(x)`, whose own mean
    # square is one, so normalising it again by a gain of one gives back the
    # same vector. That makes this case check that the norm is applied to the
    # output rather than that it changes it, which is the placement question.
    var divisor = Float32(sqrt(Float64(25.5) + Float64(EPS)))
    var ok = True
    for i in range(WIDTH):
        var v = Float32(i + 1)
        if not _close(x.data[i], v + v / divisor, 1e-4):
            ok = False
    suite.check(
        ok, "a post norm over an already normed output leaves it where it was"
    )

    # Scale the gain and the output moves with it, which says the norm is on
    # the sublayer's output and not on the residual stream.
    var loud = LayerWeights()
    loud.attn_norm = w.attn_norm
    loud.ffn_norm = w.ffn_norm
    loud.wq = w.wq
    loud.wk = w.wk
    loud.wv = w.wv
    loud.wo = w.wo
    loud.gate = w.gate
    loud.up = w.up
    loud.down = w.down
    var twos = List[Float32]()
    for _ in range(WIDTH):
        twos.append(2.0)
    loud.attn_post_norm = arena.tensor(twos, WIDTH, 1)

    var y = Buffer(WIDTH)
    for i in range(WIDTH):
        y.data[i] = Float32(i + 1)
    var keys2 = _zeros(WIDTH)
    var values2 = _zeros(WIDTH)
    attention_layer(
        spec, loud, y, s, keys2, values2, 0, 0, List[Float32](), False
    )
    ok = True
    for i in range(WIDTH):
        var v = Float32(i + 1)
        if not _close(y.data[i], v + 2.0 * v / divisor, 1e-4):
            ok = False
    suite.check(
        ok,
        (
            "and a gain of two doubles what the sublayer contributes and not"
            " the stream it joins"
        ),
    )

    keep(arena)


def test_mlp_shapes(mut suite: Suite) raises:
    suite.group("block mlp shapes")

    var arena = Arena()
    var norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    var up = _dense(WIDTH, HIDDEN, 21)
    var down = _dense(HIDDEN, WIDTH, 22)
    var up_t = arena.tensor(up, WIDTH, HIDDEN)
    var down_t = arena.tensor(down, HIDDEN, WIDTH)

    var start = List[Float32]()
    for i in range(WIDTH):
        start.append(Float32(i) - 3.0)

    # Not gated: one projection through an activation and back down.
    var spec = _spec()
    spec.gated = False
    var w = LayerWeights()
    w.attn_norm = norm
    w.ffn_norm = norm
    w.wq = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    w.wk = w.wq
    w.wv = w.wq
    w.wo = w.wq
    w.up = up_t
    w.down = down_t
    w.check(spec)

    var x = Buffer(WIDTH)
    for i in range(WIDTH):
        x.data[i] = start[i]
    var s = Scratch(spec, 4)
    mlp_layer(spec, w, x, s)

    var r = List[Float32]()
    for i in range(WIDTH):
        r.append(start[i])
    var normed = List[Float32]()
    for i in range(WIDTH):
        normed.append(r[i])
    _ref_rms(normed, 0, WIDTH, _ones(WIDTH))
    var u = _zeros(HIDDEN)
    _ref_matvec(up, WIDTH, HIDDEN, normed, 0, u, 0)
    for i in range(HIDDEN):
        u[i] = _ref_silu(u[i])
    var out = _zeros(WIDTH)
    _ref_matvec(down, HIDDEN, WIDTH, u, 0, out, 0)

    var ok = True
    for i in range(WIDTH):
        if not _close(x.data[i], r[i] + out[i], 1e-4):
            ok = False
    suite.check(ok, "a non gated mlp is one projection through an activation")

    # Gated with gelu, which is Gemma.
    var gated = _spec()
    gated.act = ACT_GELU
    var gate = _dense(WIDTH, HIDDEN, 23)
    var w2 = LayerWeights()
    w2.attn_norm = norm
    w2.ffn_norm = norm
    w2.wq = w.wq
    w2.wk = w.wq
    w2.wv = w.wq
    w2.wo = w.wq
    w2.gate = arena.tensor(gate, WIDTH, HIDDEN)
    w2.up = up_t
    w2.down = down_t
    w2.check(gated)

    var y = Buffer(WIDTH)
    for i in range(WIDTH):
        y.data[i] = start[i]
    mlp_layer(gated, w2, y, s)

    var g = _zeros(HIDDEN)
    var u2 = _zeros(HIDDEN)
    _ref_matvec(gate, WIDTH, HIDDEN, normed, 0, g, 0)
    _ref_matvec(up, WIDTH, HIDDEN, normed, 0, u2, 0)
    for i in range(HIDDEN):
        g[i] = _ref_gelu(g[i]) * u2[i]
    var out2 = _zeros(WIDTH)
    _ref_matvec(down, HIDDEN, WIDTH, g, 0, out2, 0)

    ok = True
    for i in range(WIDTH):
        if not _close(y.data[i], r[i] + out2[i], 1e-4):
            ok = False
    suite.check(ok, "and a gated gelu mlp gates through the other one")

    keep(arena)


def test_cache(mut suite: Suite) raises:
    """Two tokens in a row, which is the smallest thing that is a sequence."""
    suite.group("block over two positions")

    var arena = Arena()
    var spec = _spec()
    var w = LayerWeights()
    w.attn_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    w.ffn_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    w.wq = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    w.wk = w.wq
    w.wv = w.wq
    w.wo = w.wq
    w.gate = arena.tensor(_identity(WIDTH), WIDTH, HIDDEN)
    w.up = w.gate
    w.down = arena.tensor(_identity(HIDDEN), HIDDEN, WIDTH)

    var s = Scratch(spec, 4)
    var keys = _zeros(WIDTH * 2)
    var values = _zeros(WIDTH * 2)

    var first = Buffer(WIDTH)
    for i in range(WIDTH):
        first.data[i] = Float32(i + 1)
    attention_layer(
        spec, w, first, s, keys, values, 0, 0, List[Float32](), False
    )
    var slot0 = List[Float32]()
    for i in range(WIDTH):
        slot0.append(keys[i])

    var second = Buffer(WIDTH)
    for i in range(WIDTH):
        second.data[i] = Float32(WIDTH - i)
    attention_layer(
        spec, w, second, s, keys, values, 1, 1, List[Float32](), False
    )

    var same = True
    for i in range(WIDTH):
        if keys[i] != slot0[i]:
            same = False
    suite.check(same, "the second token leaves the first one's key alone")

    var wrote = False
    for i in range(WIDTH):
        if keys[WIDTH + i] != 0.0:
            wrote = True
    suite.check(wrote, "and writes its own into the next slot")

    # The second token's output has to depend on the first, which is the whole
    # reason a cache exists. Running the same token at slot zero with an empty
    # cache gives a different answer.
    var alone = Buffer(WIDTH)
    for i in range(WIDTH):
        alone.data[i] = Float32(WIDTH - i)
    var solo_k = _zeros(WIDTH)
    var solo_v = _zeros(WIDTH)
    attention_layer(
        spec, w, alone, s, solo_k, solo_v, 0, 0, List[Float32](), False
    )
    var differs = False
    for i in range(WIDTH):
        if not _close(alone.data[i], second.data[i], 1e-5):
            differs = True
    suite.check(differs, "and its output depends on the key that came before")

    keep(arena)


def test_errors(mut suite: Suite) raises:
    suite.group("block errors")

    var arena = Arena()
    var spec = _spec()
    var full = LayerWeights()
    full.attn_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    full.ffn_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    full.wq = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    full.wk = full.wq
    full.wv = full.wq
    full.wo = full.wq
    full.gate = arena.tensor(_identity(WIDTH), WIDTH, HIDDEN)
    full.up = full.gate
    full.down = arena.tensor(_identity(HIDDEN), HIDDEN, WIDTH)

    var raised = False
    try:
        LayerWeights().check(spec)
    except:
        raised = True
    suite.check(raised, "a layer with no weights at all is refused")

    raised = False
    var no_gate = full
    no_gate.gate = Tensor.none()
    try:
        no_gate.check(spec)
    except:
        raised = True
    suite.check(raised, "and a gated layer with no gate projection")

    raised = False
    var plain = _spec()
    plain.gated = False
    try:
        full.check(plain)
    except:
        raised = True
    suite.check(
        raised,
        "and a gate projection on a layer the table says does not gate",
    )

    raised = False
    var half_norm = full
    half_norm.q_norm = arena.tensor(_ones(HEAD_DIM), HEAD_DIM, 1)
    try:
        half_norm.check(spec)
    except:
        raised = True
    suite.check(raised, "and a query norm with no key norm beside it")

    raised = False
    var wrong = full
    wrong.wo = arena.tensor(_ones(WIDTH * 2), WIDTH, 2)
    try:
        wrong.check(_spec())
    except:
        raised = True
    suite.check(raised, "and an output projection of the wrong shape")

    raised = False
    var x = Buffer(WIDTH)
    var s = Scratch(spec, 4)
    var keys = _zeros(WIDTH)
    var values = _zeros(WIDTH)
    try:
        attention_layer(
            spec, full, x, s, keys, values, 1, 1, List[Float32](), False
        )
    except:
        raised = True
    suite.check(raised, "and a cache slot the cache has no room for")

    raised = False
    var narrow = Buffer(WIDTH - 1)
    try:
        mlp_layer(spec, full, narrow, s)
    except:
        raised = True
    suite.check(raised, "and a residual stream that is not the layer's width")

    keep(arena)
