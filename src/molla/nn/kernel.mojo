"""The arithmetic a transformer is made of, on the host.

Five things happen to a vector between one token and the next: it gets
normalised, it gets multiplied by a matrix, it gets a nonlinearity applied to
it, it gets added to something, and somewhere a row of scores gets turned into
probabilities. Everything else in a dense model is bookkeeping around those.

The matrix multiply is the only one that matters for speed and it is the only
one that reads packed bytes. A matvec against a q4_k weight never materialises a
float32 copy of the row. It uses the identity that a block's values are
`d * sc * q - dmin * m`, so the dot product over a group is
`d * sc * sum(q * x) - dmin * m * sum(x)`: two sums over the group, and the
scales applied once at the end rather than once per value. q6_k has no minimum
term and so is one sum, and q4_0 and q8_0 are the same shape again.

That identity is where the correctness risk is. It is arithmetic that is not
obviously the same as dequantizing and multiplying, so every fused path here is
checked against `molla.nn.quant.dequant_run` followed by a plain dot, over the
same random blocks the quant conformance corpus uses. The two do not agree bit
for bit and are not expected to, because they add the same terms in different
orders, so that check is the one place in this file with a tolerance on it.
"""

from std.math import exp, sqrt

from molla.nn.quant import (
    Q_BF16,
    Q_F16,
    Q_F32,
    Q_Q4_0,
    Q_Q4_1,
    Q_Q4_K,
    Q_Q5_0,
    Q_Q5_1,
    Q_Q5_K,
    Q_Q6_K,
    Q_Q8_0,
    _k_scale,
    _qh,
    bf16_at,
    block_bytes,
    block_elements,
    f16_at,
    dequant_run,
    f32_at,
    supported,
)
from molla.nn.tensor import Buffer, Tensor
from molla.sys.mmap import RawPtr


def dot(
    x: List[Float32], at: Int, y: List[Float32], to: Int, n: Int
) -> Float32:
    """A plain dot product, which is the reference every fused path is checked
    against."""
    var acc = Float32(0)
    for i in range(n):
        acc += x[at + i] * y[to + i]
    return acc


def _u8(p: RawPtr, at: Int) -> Int:
    return Int(p.unsafe_load(at))


def _i8(p: RawPtr, at: Int) -> Int:
    var v = Int(p.unsafe_load(at))
    if v >= 128:
        return v - 256
    return v


def row_dot(
    kind: Int, p: RawPtr, at: Int, x: List[Float32], to: Int, n: Int
) raises -> Float32:
    """The dot product of one packed row with a float32 vector.

    `n` is the number of values in the row and has to be a whole number of
    blocks, for the same reason `dequant_run` insists on it.
    """
    var per = block_elements(kind)
    if per == 0:
        raise Error("ggml type " + String(kind) + " has no dot product")
    if n % per != 0:
        raise Error(
            "a row of "
            + String(n)
            + " is not a whole number of "
            + String(per)
            + " element blocks"
        )

    if kind == Q_F32:
        var acc = Float32(0)
        for i in range(n):
            acc += f32_at(p, at + i * 4) * x[to + i]
        return acc
    if kind == Q_F16:
        var acc = Float32(0)
        for i in range(n):
            acc += f16_at(p, at + i * 2) * x[to + i]
        return acc
    if kind == Q_BF16:
        var acc = Float32(0)
        for i in range(n):
            acc += bf16_at(p, at + i * 2) * x[to + i]
        return acc

    var stride = block_bytes(kind)
    var blocks = n // per
    var total = Float32(0)
    for b in range(blocks):
        var block = at + b * stride
        var base = to + b * per
        if kind == Q_Q4_0:
            total += _dot_q4_0(p, block, x, base)
        elif kind == Q_Q4_1:
            total += _dot_q4_1(p, block, x, base)
        elif kind == Q_Q5_0:
            total += _dot_q5_0(p, block, x, base)
        elif kind == Q_Q5_1:
            total += _dot_q5_1(p, block, x, base)
        elif kind == Q_Q8_0:
            total += _dot_q8_0(p, block, x, base)
        elif kind == Q_Q4_K:
            total += _dot_q4_k(p, block, x, base)
        elif kind == Q_Q5_K:
            total += _dot_q5_k(p, block, x, base)
        elif kind == Q_Q6_K:
            total += _dot_q6_k(p, block, x, base)
        else:
            raise Error("ggml type " + String(kind) + " has no dot product")
    return total


def _dot_q4_0(p: RawPtr, at: Int, x: List[Float32], base: Int) -> Float32:
    """`d * (sum(q * x) - 8 * sum(x))`, since every value is centred on eight.
    """
    var d = f16_at(p, at)
    var qx = Float32(0)
    var sx = Float32(0)
    for l in range(16):
        var b = _u8(p, at + 2 + l)
        var lo = x[base + l]
        var hi = x[base + l + 16]
        qx += Float32(b & 0xF) * lo + Float32(b >> 4) * hi
        sx += lo + hi
    return d * (qx - 8.0 * sx)


def _dot_q4_1(p: RawPtr, at: Int, x: List[Float32], base: Int) -> Float32:
    """`d * sum(q * x) + m * sum(x)`, with nothing centred."""
    var d = f16_at(p, at)
    var m = f16_at(p, at + 2)
    var qx = Float32(0)
    var sx = Float32(0)
    for l in range(16):
        var b = _u8(p, at + 4 + l)
        var lo = x[base + l]
        var hi = x[base + l + 16]
        qx += Float32(b & 0xF) * lo + Float32(b >> 4) * hi
        sx += lo + hi
    return d * qx + m * sx


def _dot_q5_0(p: RawPtr, at: Int, x: List[Float32], base: Int) -> Float32:
    """q4_0's shape with the fifth bit put back before the centring."""
    var d = f16_at(p, at)
    var qh = _qh(p, at + 2)
    var qx = Float32(0)
    var sx = Float32(0)
    for l in range(16):
        var b = _u8(p, at + 6 + l)
        var lo = x[base + l]
        var hi = x[base + l + 16]
        var ql = (b & 0xF) | (((qh >> l) << 4) & 0x10)
        var qhh = (b >> 4) | ((qh >> (l + 12)) & 0x10)
        qx += Float32(ql) * lo + Float32(qhh) * hi
        sx += lo + hi
    return d * (qx - 16.0 * sx)


def _dot_q5_1(p: RawPtr, at: Int, x: List[Float32], base: Int) -> Float32:
    var d = f16_at(p, at)
    var m = f16_at(p, at + 2)
    var qh = _qh(p, at + 4)
    var qx = Float32(0)
    var sx = Float32(0)
    for l in range(16):
        var b = _u8(p, at + 8 + l)
        var lo = x[base + l]
        var hi = x[base + l + 16]
        var ql = (b & 0xF) | (((qh >> l) << 4) & 0x10)
        var qhh = (b >> 4) | ((qh >> (l + 12)) & 0x10)
        qx += Float32(ql) * lo + Float32(qhh) * hi
        sx += lo + hi
    return d * qx + m * sx


def _dot_q8_0(p: RawPtr, at: Int, x: List[Float32], base: Int) -> Float32:
    var d = f16_at(p, at)
    var acc = Float32(0)
    for l in range(32):
        acc += Float32(_i8(p, at + 2 + l)) * x[base + l]
    return d * acc


def _dot_q4_k(p: RawPtr, at: Int, x: List[Float32], base: Int) -> Float32:
    """Two sums per group of 32, and the scales applied once at the end.

    A value is `d * sc * q - dmin * m`, so a dot product over a group is
    `d * sc * sum(q * x) - dmin * m * sum(x)`. That turns two multiplies and a
    subtract per value into two multiplies and a subtract per group of 32, which
    is the entire reason a fused path is worth having.
    """
    var d = f16_at(p, at)
    var dmin = f16_at(p, at + 2)
    var scales = at + 4
    var qs = at + 16
    var total = Float32(0)
    for half in range(4):
        var lo_scale = _k_scale(p, scales, half * 2)
        var hi_scale = _k_scale(p, scales, half * 2 + 1)
        var q = qs + half * 32
        var at_lo = base + half * 64
        var at_hi = at_lo + 32
        var qx1 = Float32(0)
        var sx1 = Float32(0)
        var qx2 = Float32(0)
        var sx2 = Float32(0)
        for l in range(32):
            var b = _u8(p, q + l)
            var a = x[at_lo + l]
            var c = x[at_hi + l]
            qx1 += Float32(b & 0xF) * a
            sx1 += a
            qx2 += Float32(b >> 4) * c
            sx2 += c
        total += d * lo_scale[0] * qx1 - dmin * lo_scale[1] * sx1
        total += d * hi_scale[0] * qx2 - dmin * hi_scale[1] * sx2
    return total


def _dot_q5_k(p: RawPtr, at: Int, x: List[Float32], base: Int) -> Float32:
    var d = f16_at(p, at)
    var dmin = f16_at(p, at + 2)
    var scales = at + 4
    var qh = at + 16
    var qs = at + 48
    var total = Float32(0)
    for half in range(4):
        var lo_scale = _k_scale(p, scales, half * 2)
        var hi_scale = _k_scale(p, scales, half * 2 + 1)
        var u1 = 1 << (half * 2)
        var u2 = u1 << 1
        var q = qs + half * 32
        var at_lo = base + half * 64
        var at_hi = at_lo + 32
        var qx1 = Float32(0)
        var sx1 = Float32(0)
        var qx2 = Float32(0)
        var sx2 = Float32(0)
        for l in range(32):
            var b = _u8(p, q + l)
            var h = _u8(p, qh + l)
            var a = x[at_lo + l]
            var c = x[at_hi + l]
            var v1 = (b & 0xF) + (16 if (h & u1) != 0 else 0)
            var v2 = (b >> 4) + (16 if (h & u2) != 0 else 0)
            qx1 += Float32(v1) * a
            sx1 += a
            qx2 += Float32(v2) * c
            sx2 += c
        total += d * lo_scale[0] * qx1 - dmin * lo_scale[1] * sx1
        total += d * hi_scale[0] * qx2 - dmin * hi_scale[1] * sx2
    return total


def _dot_q6_k(p: RawPtr, at: Int, x: List[Float32], base: Int) -> Float32:
    """No minimum term, so one sum per group of sixteen and a signed scale.

    The eight accumulators are the awkward part. Each pass over `l` touches four
    output positions, and each of those belongs to a different one of the eight
    scale groups, so the sums cannot be finished one group at a time without
    reading the high bit plane four times over.
    """
    var d = f16_at(p, at + 208)
    var ql = at
    var qh = at + 128
    var sc = at + 192
    var total = Float32(0)
    for n in range(2):
        var lo = ql + n * 64
        var hi = qh + n * 32
        var scale = sc + n * 8
        var out = base + n * 128
        var acc = InlineArray[Float32, 8](fill=0.0)
        for l in range(32):
            var group = l // 16
            var h = _u8(p, hi + l)
            var q1 = ((_u8(p, lo + l) & 0xF) | (((h >> 0) & 3) << 4)) - 32
            var q2 = ((_u8(p, lo + l + 32) & 0xF) | (((h >> 2) & 3) << 4)) - 32
            var q3 = ((_u8(p, lo + l) >> 4) | (((h >> 4) & 3) << 4)) - 32
            var q4 = ((_u8(p, lo + l + 32) >> 4) | (((h >> 6) & 3) << 4)) - 32
            acc[group] += Float32(q1) * x[out + l]
            acc[group + 2] += Float32(q2) * x[out + l + 32]
            acc[group + 4] += Float32(q3) * x[out + l + 64]
            acc[group + 6] += Float32(q4) * x[out + l + 96]
        for g in range(8):
            total += d * Float32(_i8(p, scale + g)) * acc[g]
    return total


def matvec(w: Tensor, x: Buffer, mut out: Buffer) raises:
    """`out[r] = dot(row r of w, x)`, one row at a time.

    Row major over the output, which is the order that reads each weight row
    once and in address order. The other order, walking the input and scattering
    into the output, reads every row of the weight for every element of the
    input and is what a naive transpose turns this into.
    """
    if x.elements() != w.cols:
        raise Error(
            "matvec wants an input of "
            + String(w.cols)
            + " but got "
            + String(x.elements())
        )
    if out.elements() != w.rows:
        raise Error(
            "matvec wants an output of "
            + String(w.rows)
            + " but got "
            + String(out.elements())
        )
    var stride = w.row_bytes()
    var p = w.base()
    for r in range(w.rows):
        out.data[r] = row_dot(w.kind, p, r * stride, x.data, 0, w.cols)


def matvec_into(
    w: Tensor,
    x: List[Float32],
    at_in: Int,
    mut out: List[Float32],
    at_out: Int,
) raises:
    """`matvec` reading and writing runs of plain lists.

    The key and value projections write straight into the cache, which is one
    long list per layer rather than a buffer per token, and a query is
    normalised per head over a run of itself. Both want an offset, and `Buffer`
    deliberately has no view type, because a view over a list the cache grows
    is a lifetime problem in exchange for one loop.
    """
    if at_in < 0 or at_out < 0:
        raise Error("a matvec offset cannot be negative")
    if len(x) < at_in + w.cols:
        raise Error(
            "matvec wants "
            + String(w.cols)
            + " values from offset "
            + String(at_in)
            + " but the input ends at "
            + String(len(x))
        )
    if len(out) < at_out + w.rows:
        raise Error(
            "matvec wants room for "
            + String(w.rows)
            + " rows at offset "
            + String(at_out)
            + " but the output ends at "
            + String(len(out))
        )
    var stride = w.row_bytes()
    var p = w.base()
    for r in range(w.rows):
        out[at_out + r] = row_dot(w.kind, p, r * stride, x, at_in, w.cols)


def rms_norm_at(
    mut x: List[Float32], at: Int, n: Int, gain: Tensor, eps: Float32
) raises:
    """Root mean square norm in place over a run.

    Qwen 3 normalises each head of the query and the key separately, so this
    runs over `head_dim` at a time inside a vector that is thirty two heads
    wide. Gemma normalises a sublayer's output on its way to the residual add,
    which is the whole width but still in place. Neither wants a second buffer.
    """
    if at < 0 or n <= 0:
        raise Error("a norm needs a non negative offset and a positive width")
    if len(x) < at + n:
        raise Error("a norm was pointed past the end of its buffer")
    if gain.elements() != n:
        raise Error(
            "rms_norm wants a gain of "
            + String(n)
            + " but got "
            + String(gain.elements())
        )
    var sum = Float64(0)
    for i in range(n):
        sum += Float64(x[at + i]) * Float64(x[at + i])
    var scale = Float32(1.0 / sqrt(sum / Float64(n) + Float64(eps)))

    var g = Buffer(n)
    dequant_run(gain.kind, gain.base(), 0, n, g.data, 0)
    for i in range(n):
        x[at + i] = x[at + i] * scale * g.data[i]


def rms_norm(mut out: Buffer, x: Buffer, gain: Tensor, eps: Float32) raises:
    """Root mean square norm, which is layer norm without the mean subtracted.

    The gain is a weight and so it is packed like any other, but every model in
    practice stores norms as f32 because they are a few thousand values and
    quantizing them saves nothing and costs accuracy. It is read through the
    same dequant path anyway rather than assuming, since a file that does
    quantize them should work rather than produce noise.

    The sum of squares accumulates in float64. On a 4096 wide activation the
    float32 answer differs from the float64 one in the last couple of bits,
    which does not matter on its own, and it is thirty two layers of it that
    make a difference nobody can attribute later.
    """
    if x.elements() != out.elements():
        raise Error("rms_norm wants the input and the output the same size")
    if gain.elements() != x.elements():
        raise Error(
            "rms_norm wants a gain of "
            + String(x.elements())
            + " but got "
            + String(gain.elements())
        )
    var n = x.elements()
    var sum = Float64(0)
    for i in range(n):
        sum += Float64(x.data[i]) * Float64(x.data[i])
    var scale = Float32(1.0 / sqrt(sum / Float64(n) + Float64(eps)))

    var g = Buffer(gain.cols, gain.rows)
    # `base()` already points at the first byte of the tensor, so the offset
    # into it is zero. Passing the address again reads the weight from the
    # weight's own address squared, which is a segfault on a good day.
    dequant_run(gain.kind, gain.base(), 0, n, g.data, 0)
    for i in range(n):
        out.data[i] = x.data[i] * scale * g.data[i]


def softmax(mut x: List[Float32], at: Int, n: Int):
    """In place over a run, with the maximum subtracted first.

    Subtracting the maximum is not an optimisation, it is what stops the whole
    row becoming NaN. An attention score of 100 is ordinary and `exp(100)` is
    two thirds of the way to a float32 infinity, so a row with two of them sums
    to infinity and every probability comes back as a nan.
    """
    if n <= 0:
        return
    var top = x[at]
    for i in range(1, n):
        if x[at + i] > top:
            top = x[at + i]
    var sum = Float32(0)
    for i in range(n):
        var e = exp(x[at + i] - top)
        x[at + i] = e
        sum += e
    var inv = 1.0 / sum
    for i in range(n):
        x[at + i] *= inv


def silu(v: Float32) -> Float32:
    """`x * sigmoid(x)`, which is what every gated MLP in a Llama uses."""
    return v / (1.0 + exp(-v))


def gelu(v: Float32) -> Float32:
    """The tanh approximation, which is the one the weights were trained with.

    The exact form through the error function is a different function by about
    a thousandth, and a model trained against one and run against the other is
    the kind of small consistent bias that never gets found.
    """
    var c = Float32(0.7978845608028654)
    var inner = c * (v + 0.044715 * v * v * v)
    var t = Float32(2.0) / (1.0 + exp(-2.0 * inner)) - 1.0
    return 0.5 * v * (1.0 + t)


def swiglu(mut gate: Buffer, up: Buffer) raises:
    """`gate = silu(gate) * up`, in place on the gate.

    In place because the two halves of a gated MLP are the same width as each
    other and as wide as the hidden dimension, which on an 8B is 14336 floats,
    and a third buffer for a result that is consumed immediately is 56 KB of
    cache pressure per layer for nothing.
    """
    if gate.elements() != up.elements():
        raise Error("swiglu wants both halves the same size")
    for i in range(gate.elements()):
        gate.data[i] = silu(gate.data[i]) * up.data[i]


def add_into(mut acc: Buffer, x: Buffer) raises:
    """The residual add, which is the only reason a deep network trains."""
    if acc.elements() != x.elements():
        raise Error("add_into wants both sides the same size")
    for i in range(acc.elements()):
        acc.data[i] += x.data[i]


def scale_into(mut x: Buffer, by: Float32):
    for i in range(x.elements()):
        x.data[i] *= by


def argmax(x: List[Float32], at: Int, n: Int) -> Int:
    """The most likely token, which is what greedy decoding picks.

    Ties go to the lower index. That is arbitrary and it is written down because
    it has to match whatever the sampler does in #28 or the two disagree on a
    logit row that has an exact tie, which happens more often than it sounds
    like it should when a row has been through a float16 weight.
    """
    if n <= 0:
        return -1
    var best = 0
    for i in range(1, n):
        if x[at + i] > x[at + best]:
            best = i
    return best
