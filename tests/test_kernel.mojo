"""The host kernels.

Two kinds of check here. The small ones pin arithmetic a person can do on
paper, a softmax that sums to one, a norm with a known divisor, a gelu at a
value someone can look up. Those are the ones that say which side is wrong when
something disagrees.

The larger one is the fused matvec against packed weights. Every quantized path
in `row_dot` is checked against dequantizing the same bytes with
`molla.nn.quant.dequant_run` and taking a plain dot product, over random blocks
built the same way the conformance corpus builds them. That check has a
tolerance because the two orders of summation are genuinely different and it
would be wrong to expect them to agree bit for bit. The tolerance is relative to
the sum of the term magnitudes rather than to the answer, since a dot product
that cancels down to nearly zero has an absolute error set by the terms that
cancelled and not by what is left.
"""

from std.math import sqrt
from std.memory import bitcast

from harness import Suite

from molla.nn.kernel import (
    add_into,
    argmax,
    dot,
    gelu,
    matvec,
    rms_norm,
    row_dot,
    scale_into,
    silu,
    softmax,
    swiglu,
)
from molla.nn.quant import (
    Q_BF16,
    Q_F16,
    Q_F32,
    Q_Q4_0,
    Q_Q4_K,
    Q_Q5_K,
    Q_Q6_K,
    Q_Q8_0,
    block_bytes,
    block_elements,
    dequant_run,
)
from molla.nn.tensor import Buffer, Tensor
from molla.sys.mem import keep
from molla.sys.mmap import RawPtr


def _ptr(ref bytes: List[UInt8]) -> RawPtr:
    return RawPtr(unsafe_from_address=Int(bytes.unsafe_ptr()))


def _close(a: Float32, b: Float32, within: Float32) -> Bool:
    var d = a - b
    if d < 0:
        d = -d
    return d <= within


struct Rng(Copyable, ImplicitlyCopyable, Movable):
    """The same xorshift the json tests use, so the inputs are not the ones any
    of this was written against."""

    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt64:
        self.state ^= self.state << 13
        self.state ^= self.state >> 7
        self.state ^= self.state << 17
        return self.state

    def byte(mut self) -> UInt8:
        return UInt8(self.next() & 0xFF)

    def unit(mut self) -> Float32:
        """A float in [-1, 1)."""
        return Float32(Int(self.next() & 0xFFFF)) / 32768.0 - 1.0


# Four float16 bit patterns with different exponents and both signs. A scale
# field has to be a real number, since two random bytes read as a float16 are a
# NaN or an infinity often enough to matter and comparing NaN against NaN says
# nothing about either side.
def _scale_bits(pick: Int) -> Int:
    if pick == 0:
        return 0x3C00
    if pick == 1:
        return 0xB800
    if pick == 2:
        return 0x2E66
    return 0x3555


def _blocks(kind: Int, count: Int, seed: UInt64) raises -> List[UInt8]:
    """`count` blocks of random bytes with the scale fields made sane."""
    var stride = block_bytes(kind)
    var out = List[UInt8]()
    var rng = Rng(seed)
    for _ in range(count * stride):
        out.append(rng.byte())
    for b in range(count):
        var at = b * stride
        if kind == Q_Q6_K:
            _put16(out, at + 208, _scale_bits(Int(rng.next() & 3)))
        else:
            _put16(out, at, _scale_bits(Int(rng.next() & 3)))
            if kind == Q_Q4_K or kind == Q_Q5_K:
                _put16(out, at + 2, _scale_bits(Int(rng.next() & 3)))
    return out^


def _put16(mut b: List[UInt8], at: Int, value: Int):
    b[at] = UInt8(value & 0xFF)
    b[at + 1] = UInt8((value >> 8) & 0xFF)


def _putf32(mut b: List[UInt8], at: Int, value: Float32):
    var bits = Int(bitcast[DType.uint32, 1](value))
    b[at] = UInt8(bits & 0xFF)
    b[at + 1] = UInt8((bits >> 8) & 0xFF)
    b[at + 2] = UInt8((bits >> 16) & 0xFF)
    b[at + 3] = UInt8((bits >> 24) & 0xFF)


def _f32_bytes(values: List[Float32]) -> List[UInt8]:
    var out = List[UInt8]()
    for _ in range(len(values) * 4):
        out.append(0)
    for i in range(len(values)):
        _putf32(out, i * 4, values[i])
    return out^


def run(mut suite: Suite) raises:
    test_scalars(suite)
    test_softmax(suite)
    test_elementwise(suite)
    test_rms_norm(suite)
    test_row_dot(suite)
    test_fused_matches_dequant(suite)
    test_matvec(suite)


def test_scalars(mut suite: Suite) raises:
    suite.group("kernel scalars")

    var a = List[Float32]()
    var b = List[Float32]()
    for i in range(4):
        a.append(Float32(i + 1))
        b.append(2.0)
    suite.check(dot(a, 0, b, 0, 4) == 20.0, "a dot product adds the products")
    suite.check(dot(a, 1, b, 0, 2) == 10.0, "and it starts where it is told to")
    suite.check(dot(a, 0, b, 0, 0) == 0.0, "an empty dot product is zero")

    suite.check(silu(0.0) == 0.0, "silu of zero is zero")
    suite.check(
        _close(silu(1.0), 0.7310586, 1e-6), "and silu of one is one sigmoid"
    )
    suite.check(silu(-20.0) > -1e-6, "silu decays to zero on the left")
    suite.check(
        _close(gelu(1.0), 0.8411920, 1e-5), "gelu of one is about 0.84119"
    )
    suite.check(gelu(0.0) == 0.0, "and gelu of zero is zero")
    suite.check(
        _close(gelu(-1.0), -0.1588080, 1e-5),
        "and gelu is not odd, it is small and negative at minus one",
    )

    var pick = List[Float32]()
    pick.append(1.0)
    pick.append(9.0)
    pick.append(3.0)
    suite.check(argmax(pick, 0, 3) == 1, "argmax finds the largest")
    pick[2] = 9.0
    suite.check(argmax(pick, 0, 3) == 1, "and a tie goes to the lower index")
    suite.check(argmax(pick, 0, 0) == -1, "and an empty row has no argmax")


def test_softmax(mut suite: Suite) raises:
    suite.group("kernel softmax")

    var x = List[Float32]()
    x.append(1.0)
    x.append(2.0)
    x.append(3.0)
    softmax(x, 0, 3)
    suite.check(
        _close(x[0] + x[1] + x[2], 1.0, 1e-6), "a softmax row sums to one"
    )
    suite.check(_close(x[0], 0.09003057, 1e-6), "and the smallest is 0.0900")
    suite.check(x[0] < x[1] and x[1] < x[2], "and the order is kept")

    # The whole reason the maximum is subtracted. exp(800) is an infinity in
    # float32 and a row that sums to infinity comes back as nothing but NaN.
    var big = List[Float32]()
    big.append(800.0)
    big.append(801.0)
    softmax(big, 0, 2)
    suite.check(
        _close(big[0] + big[1], 1.0, 1e-6),
        "and a row of eight hundreds still sums to one",
    )
    suite.check(big[0] == big[0], "rather than coming back as NaN")

    var one = List[Float32]()
    one.append(-5.0)
    softmax(one, 0, 1)
    suite.check(one[0] == 1.0, "a row of one is certain")

    var run_of = List[Float32]()
    for _ in range(4):
        run_of.append(0.0)
    run_of[1] = 1.0
    run_of[2] = 1.0
    softmax(run_of, 1, 2)
    suite.check(
        run_of[0] == 0.0 and run_of[3] == 0.0 and _close(run_of[1], 0.5, 1e-6),
        "and a softmax over part of a list leaves the rest alone",
    )


def test_elementwise(mut suite: Suite) raises:
    suite.group("kernel elementwise")

    var gate = Buffer(3)
    var up = Buffer(3)
    for i in range(3):
        gate.data[i] = 1.0
        up.data[i] = Float32(i + 1)
    swiglu(gate, up)
    suite.check(
        _close(gate.data[0], 0.7310586, 1e-6)
        and _close(gate.data[2], 2.1931758, 1e-5),
        "swiglu is silu of the gate times the other half",
    )

    var acc = Buffer(3)
    acc.fill(1.0)
    var add = Buffer(3)
    add.fill(0.5)
    add_into(acc, add)
    suite.check(acc.data[1] == 1.5, "a residual add adds")
    scale_into(acc, 2.0)
    suite.check(acc.data[1] == 3.0, "and a scale scales")

    var wrong = Buffer(4)
    var raised = False
    try:
        add_into(acc, wrong)
    except:
        raised = True
    suite.check(raised, "adding a different size is an error")

    raised = False
    try:
        swiglu(gate, wrong)
    except:
        raised = True
    suite.check(raised, "and so is a gate and an up that disagree")


def test_rms_norm(mut suite: Suite) raises:
    suite.group("kernel rms_norm")

    # Four values whose mean square is 7.5, so the divisor is a number that can
    # be checked without running the code that produces it.
    var x = Buffer(4)
    for i in range(4):
        x.data[i] = Float32(i + 1)
    var ones = List[Float32]()
    for _ in range(4):
        ones.append(1.0)
    var gain_bytes = _f32_bytes(ones)
    var gain = Tensor(Int(gain_bytes.unsafe_ptr()), Q_F32, 4, 1)
    var out = Buffer(4)
    rms_norm(out, x, gain, 1e-5)

    var expect = Float32(1.0 / sqrt(Float64(7.5) + 1e-5))
    suite.check(
        _close(out.data[0], expect, 1e-6),
        "rms_norm divides by the root mean square",
    )
    suite.check(
        _close(out.data[3], 4.0 * expect, 1e-5),
        "and it is the same divisor for every element",
    )

    var half = List[Float32]()
    for _ in range(4):
        half.append(0.5)
    var half_bytes = _f32_bytes(half)
    var half_gain = Tensor(Int(half_bytes.unsafe_ptr()), Q_F32, 4, 1)
    rms_norm(out, x, half_gain, 1e-5)
    suite.check(
        _close(out.data[0], 0.5 * expect, 1e-6), "and the gain multiplies"
    )
    # A `Tensor` holds an address and not a reference, so nothing the compiler
    # can see keeps the bytes it points at alive. Without this the list is freed
    # the moment the tensor is built and the norm reads whatever came next.
    keep(half_bytes)

    var zero = Buffer(4)
    rms_norm(out, zero, gain, 1e-5)
    suite.check(
        out.data[0] == 0.0 and out.data[0] == out.data[0],
        "an all zero input norms to zero rather than to NaN",
    )

    var small = Buffer(2)
    var raised = False
    try:
        rms_norm(small, x, gain, 1e-5)
    except:
        raised = True
    suite.check(raised, "and an output of the wrong size is an error")

    raised = False
    var short_gain = Tensor(Int(gain_bytes.unsafe_ptr()), Q_F32, 2, 1)
    try:
        rms_norm(out, x, short_gain, 1e-5)
    except:
        raised = True
    suite.check(raised, "and so is a gain of the wrong size")
    keep(gain_bytes)


def test_row_dot(mut suite: Suite) raises:
    suite.group("kernel row_dot on unpacked rows")

    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(-3.0)
    values.append(0.5)
    var raw = _f32_bytes(values)
    var x = List[Float32]()
    for i in range(4):
        x.append(Float32(i + 1))
    suite.check(
        row_dot(Q_F32, _ptr(raw), 0, x, 0, 4) == -2.0,
        "a float32 row multiplies straight through",
    )

    # 1.0 and 2.0 as float16, then the same two again.
    var h = List[UInt8]()
    for _ in range(8):
        h.append(0)
    _put16(h, 0, 0x3C00)
    _put16(h, 2, 0x4000)
    _put16(h, 4, 0x3C00)
    _put16(h, 6, 0x4000)
    suite.check(
        row_dot(Q_F16, _ptr(h), 0, x, 0, 4) == 1.0 + 4.0 + 3.0 + 8.0,
        "and a float16 row is read two bytes at a time",
    )

    # bfloat16 is the top half of a float32, so 1.0 is 0x3F80 and 2.0 is 0x4000.
    var bf = List[UInt8]()
    for _ in range(8):
        bf.append(0)
    _put16(bf, 0, 0x3F80)
    _put16(bf, 2, 0x4000)
    _put16(bf, 4, 0x3F80)
    _put16(bf, 6, 0x4000)
    suite.check(
        row_dot(Q_BF16, _ptr(bf), 0, x, 0, 4) == 1.0 + 4.0 + 3.0 + 8.0,
        "and a bfloat16 row is the top half of a float32",
    )

    var raised = False
    try:
        _ = row_dot(Q_Q4_0, _ptr(raw), 0, x, 0, 4)
    except:
        raised = True
    suite.check(raised, "a row that is not a whole number of blocks is refused")

    raised = False
    try:
        _ = row_dot(99, _ptr(raw), 0, x, 0, 4)
    except:
        raised = True
    suite.check(raised, "and a type with no decoder has no dot product either")
    keep(raw)
    keep(h)
    keep(bf)


def _names() -> List[Int]:
    var out = List[Int]()
    out.append(Q_Q4_0)
    out.append(Q_Q8_0)
    out.append(Q_Q4_K)
    out.append(Q_Q5_K)
    out.append(Q_Q6_K)
    return out^


def _label(kind: Int) -> String:
    if kind == Q_Q4_0:
        return "q4_0"
    if kind == Q_Q8_0:
        return "q8_0"
    if kind == Q_Q4_K:
        return "q4_k"
    if kind == Q_Q5_K:
        return "q5_k"
    return "q6_k"


def test_fused_matches_dequant(mut suite: Suite) raises:
    """The check the fused paths exist to be measured against.

    Four blocks of random bytes per format, decoded the slow way and multiplied
    the obvious way, against the fused path reading the same bytes. The bound is
    relative to the sum of the term magnitudes because a dot product that
    cancels is left holding an absolute error set by the terms that cancelled.
    """
    suite.group("kernel fused matvec agrees with dequant")

    var kinds = _names()
    for i in range(len(kinds)):
        var kind = kinds[i]
        var per = block_elements(kind)
        var n = per * 4
        var packed = _blocks(kind, 4, 0x9E3779B97F4A7C15 + UInt64(kind))

        var w = List[Float32]()
        for _ in range(n):
            w.append(0.0)
        dequant_run(kind, _ptr(packed), 0, n, w, 0)

        var rng = Rng(0x1234567 + UInt64(kind))
        var x = List[Float32]()
        for _ in range(n):
            x.append(rng.unit())

        var slow = Float32(0)
        var mag = Float32(0)
        for j in range(n):
            var term = w[j] * x[j]
            slow += term
            mag += term if term >= 0 else -term

        var fast = row_dot(kind, _ptr(packed), 0, x, 0, n)
        suite.check(
            _close(fast, slow, 1e-4 * mag + 1e-6),
            _label(kind)
            + " fused over "
            + String(n)
            + " values matches dequantizing first",
        )


def test_matvec(mut suite: Suite) raises:
    suite.group("kernel matvec")

    # Two rows of three, in ggml order, so cols is the fast axis and the matrix
    # reads 1 2 3 then 4 5 6 in memory.
    var values = List[Float32]()
    values.append(1.0)
    values.append(2.0)
    values.append(3.0)
    values.append(4.0)
    values.append(5.0)
    values.append(6.0)
    var raw = _f32_bytes(values)
    var w = Tensor(Int(raw.unsafe_ptr()), Q_F32, 3, 2)

    var x = Buffer(3)
    x.data[0] = 1.0
    x.data[1] = 0.0
    x.data[2] = -1.0
    var out = Buffer(2)
    matvec(w, x, out)
    suite.check(
        out.data[0] == -2.0 and out.data[1] == -2.0,
        "matvec takes a row at a time and writes one output per row",
    )

    x.data[1] = 1.0
    matvec(w, x, out)
    suite.check(
        out.data[0] == 0.0 and out.data[1] == 3.0,
        "and each row gets its own dot product",
    )

    var wrong_in = Buffer(4)
    var raised = False
    try:
        matvec(w, wrong_in, out)
    except:
        raised = True
    suite.check(raised, "an input that is not as wide as the weight is refused")

    var wrong_out = Buffer(5)
    raised = False
    try:
        matvec(w, x, wrong_out)
    except:
        raised = True
    suite.check(
        raised, "and so is an output that does not have one slot per row"
    )
    keep(raw)
