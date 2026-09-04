"""The planar layout against the format it came from.

Two claims are being made and they need different evidence.

The first is that the repack loses nothing. A planar row decoded back to
float32 should be the same numbers as decoding the original blocks, and not
approximately: every folding this layout does is exact in float32, because
`d * sc * q - dmin * m` regrouped as `(d * sc) * q + (-(dmin * m))` is the same
sequence of operations with the constants computed early, and a centring like
q4_0's minus eight is exact on an integer. So the check here is equality, not a
tolerance, and a tolerance would hide the one bug worth catching, which is a
group index off by one putting the right scales on the wrong values.

The second is that the planar dot product agrees with the fused ggml one. That
one does need a tolerance, because the two sum the same terms in different
orders, and it is the same tolerance and the same reasoning as the fused paths
already carry in `tests/test_kernel.mojo`.

Random bytes rather than a real model, four blocks per format, with the scale
fields forced to be real numbers because two random bytes read as a float16 are
a NaN often enough that comparing them would say nothing.
"""

from molla.nn.kernel import matvec, row_dot
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
    block_bytes,
    block_elements,
    dequant_run,
)
from molla.nn.repack import (
    LAYOUT_GGML,
    LAYOUT_PLANAR,
    LAYOUT_VERSION,
    QUANT_I8,
    QUANT_S4,
    QUANT_S5,
    QUANT_S6,
    QUANT_U4,
    QUANT_U5,
    group_size,
    has_min,
    planar_groups,
    planar_quant_bytes,
    planar_row_bytes,
    planar_row_dot,
    planar_run,
    quant_bits,
    quant_form,
    quant_high_bits,
    repack_row,
    group_shift,
    repackable,
    unpack_run,
)
from molla.nn.tensor import WHERE_DEVICE, Buffer, Tensor
from molla.sys.mmap import RawPtr

from std.sys.info import has_accelerator

from max.gpu.host import DeviceContext

from molla.nn.gpu import device_matvec
from molla.sys.mem import keep

from harness import Suite


def _ptr(ref bytes: List[UInt8]) -> RawPtr:
    return RawPtr(unsafe_from_address=Int(bytes.unsafe_ptr()))


def _close(a: Float32, b: Float32, within: Float32) -> Bool:
    var d = a - b
    if d < 0:
        d = -d
    return d <= within


struct Rng(Copyable, ImplicitlyCopyable, Movable):
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
        return Float32(Int(self.next() & 0xFFFF)) / 32768.0 - 1.0


def _scale_bits(pick: Int) -> Int:
    if pick == 0:
        return 0x3C00
    if pick == 1:
        return 0xB800
    if pick == 2:
        return 0x2E66
    return 0x3555


def _put16(mut b: List[UInt8], at: Int, value: Int):
    b[at] = UInt8(value & 0xFF)
    b[at + 1] = UInt8((value >> 8) & 0xFF)


def _blocks(kind: Int, count: Int, seed: UInt64) raises -> List[UInt8]:
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
            if (
                kind == Q_Q4_K
                or kind == Q_Q5_K
                or kind == Q_Q4_1
                or kind == Q_Q5_1
            ):
                _put16(out, at + 2, _scale_bits(Int(rng.next() & 3)))
    return out^


def _zeros(n: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for _ in range(n):
        out.append(0)
    return out^


def _floats(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


def _kinds() -> List[Int]:
    var out = List[Int]()
    out.append(Q_Q4_0)
    out.append(Q_Q4_1)
    out.append(Q_Q5_0)
    out.append(Q_Q5_1)
    out.append(Q_Q8_0)
    out.append(Q_Q4_K)
    out.append(Q_Q5_K)
    out.append(Q_Q6_K)
    return out^


def _label(kind: Int) -> String:
    if kind == Q_Q4_0:
        return "q4_0"
    if kind == Q_Q4_1:
        return "q4_1"
    if kind == Q_Q5_0:
        return "q5_0"
    if kind == Q_Q5_1:
        return "q5_1"
    if kind == Q_Q8_0:
        return "q8_0"
    if kind == Q_Q4_K:
        return "q4_k"
    if kind == Q_Q5_K:
        return "q5_k"
    return "q6_k"


def run(mut suite: Suite) raises:
    test_geometry(suite)
    test_round_trip(suite)
    test_planar_dot(suite)
    test_planar_matvec(suite)
    test_refusals(suite)


def test_geometry(mut suite: Suite) raises:
    suite.group("nn.repack geometry")

    suite.check(LAYOUT_GGML != LAYOUT_PLANAR, "the two layouts are distinct")
    suite.check(
        LAYOUT_VERSION >= 1, "the layout has a version to key a cache on"
    )

    suite.check(not repackable(Q_F32), "f32 is already the layout kernels want")
    suite.check(not repackable(Q_F16), "and f16 would lose precision in a byte")
    suite.check(not repackable(Q_BF16), "and so would bf16")

    var kinds = _kinds()
    for i in range(len(kinds)):
        var kind = kinds[i]
        suite.check(repackable(kind), _label(kind) + " has a planar form")

    suite.check(group_size(Q_Q4_K) == 32, "a q4_k group is thirty two values")
    suite.check(
        group_size(Q_Q6_K) == 16,
        "and a q6_k group is sixteen, because that is how many scales it has",
    )
    suite.check(group_size(Q_F32) == 0, "an unrepackable type has no group")

    suite.check(has_min(Q_Q4_1), "q4_1 keeps its minimum")
    suite.check(has_min(Q_Q4_K), "and q4_k folds one per group")
    suite.check(
        not has_min(Q_Q4_0),
        "q4_0 does not, because the centring went into the byte",
    )
    suite.check(not has_min(Q_Q6_K), "and neither does q6_k")

    suite.check(planar_groups(Q_Q8_0, 256) == 8, "256 q8_0 values is 8 groups")
    suite.check(planar_groups(Q_Q6_K, 256) == 16, "and 16 q6_k ones")

    # Every group in the table has to be a power of two, because the device
    # kernels index their scale plane with a shift and not a divide. A type
    # added with a group of 24 fails here rather than reading whichever scale
    # `i >> 4` happens to land on.
    for i in range(len(kinds)):
        var g = group_size(kinds[i])
        var s = group_shift(g)
        suite.check(
            s >= 0 and (1 << s) == g,
            "a " + _label(kinds[i]) + " group is a power of two",
        )
    suite.check(group_shift(32) == 5, "a group of 32 is a shift of 5")
    suite.check(group_shift(16) == 4, "and a group of 16 is a shift of 4")
    suite.check(group_shift(24) == -1, "a group that is not one has no shift")

    suite.check(
        quant_form(Q_Q4_0) == QUANT_S4, "q4_0 quants are signed nibbles"
    )
    suite.check(
        quant_form(Q_Q4_1) == QUANT_U4, "q4_1 keeps its nibbles unsigned"
    )
    suite.check(quant_form(Q_Q4_K) == QUANT_U4, "and so does q4_k")
    suite.check(
        quant_form(Q_Q5_K) == QUANT_U5,
        "an unsigned five bit type is two planes",
    )
    suite.check(
        quant_form(Q_Q5_0) == QUANT_S5, "and q5_0 is the same two read lower"
    )
    suite.check(quant_form(Q_Q6_K) == QUANT_S6, "q6_k is the six bit form")
    suite.check(quant_form(Q_Q8_0) == QUANT_I8, "an eight bit one takes a byte")
    suite.check(quant_bits(Q_Q4_K) == 4, "four bit types are four bits wide")
    suite.check(quant_bits(Q_Q5_0) == 5, "five bit types are five")
    suite.check(quant_bits(Q_Q6_K) == 6, "and q6_k is six")
    suite.check(quant_bits(Q_Q8_0) == 8, "q8_0 is the only eight")

    suite.check(
        quant_high_bits(QUANT_U4) == 0, "a four bit form has no second plane"
    )
    suite.check(quant_high_bits(QUANT_S5) == 1, "a five bit form carries one")
    suite.check(quant_high_bits(QUANT_S6) == 2, "and a six bit form two")

    suite.check(
        planar_quant_bytes(Q_Q8_0, 256) == 256,
        "an eight bit quant plane is a byte a value",
    )
    suite.check(
        planar_quant_bytes(Q_Q4_K, 256) == 128,
        "and a four bit one is half that",
    )
    suite.check(
        planar_quant_bytes(Q_Q5_0, 256) == 128 + 32,
        "five bits is a nibble plane and a bit plane",
    )
    suite.check(
        planar_quant_bytes(Q_Q6_K, 256) == 128 + 64,
        "and six bits is a nibble plane and a two bit one",
    )

    # The quant plane at the type's own width, plus one scale plane, or two
    # planes when there is a minimum.
    suite.check(
        planar_row_bytes(Q_Q8_0, 256) == 256 + 8 * 4,
        "a centred row is the quants and one scale plane",
    )
    suite.check(
        planar_row_bytes(Q_Q4_K, 256) == 128 + 8 * 4 + 8 * 4,
        "and a row with a minimum carries a second plane",
    )
    suite.check(
        planar_row_bytes(Q_Q6_K, 256) == 128 + 64 + 16 * 4,
        "q6_k has twice the scales and no minimum",
    )
    for i in range(len(kinds)):
        var kind = kinds[i]
        suite.check(
            planar_row_bytes(kind, 512) % 4 == 0,
            _label(kind) + " rows are a multiple of four bytes long",
        )

    var raised = False
    try:
        _ = planar_row_bytes(Q_Q4_K, 300)
    except:
        raised = True
    suite.check(raised, "a row that is not whole groups has no planar size")


def test_round_trip(mut suite: Suite) raises:
    """Repack and decode gives back exactly what the blocks decode to.

    Exactly, with no tolerance. Every step of the fold is either an integer
    operation or a float32 multiply of the same two numbers the decoder
    multiplies, so any difference at all is a bug and not rounding.
    """
    suite.group("nn.repack decodes to the same numbers")

    var kinds = _kinds()
    for i in range(len(kinds)):
        var kind = kinds[i]
        var per = block_elements(kind)
        var n = per * 4
        var packed = _blocks(kind, 4, 0x243F6A8885A308D3 + UInt64(kind))
        var planar = _zeros(planar_row_bytes(kind, n))
        repack_row(kind, _ptr(packed), 0, n, _ptr(planar), 0)

        var want = _floats(n)
        dequant_run(kind, _ptr(packed), 0, n, want, 0)
        var got = _floats(n)
        planar_run(kind, _ptr(planar), 0, n, got, 0)

        var same = True
        var first = -1
        for j in range(n):
            if got[j] != want[j]:
                same = False
                if first < 0:
                    first = j
        suite.check(
            same,
            _label(kind)
            + " repacked and decoded is bit for bit the blocks decoded"
            + ("" if first < 0 else ", first differs at " + String(first)),
        )

        # The dispatcher has to pick the same two readers, since everything that
        # holds a tensor rather than a type number goes through it.
        var by_layout = _floats(n)
        unpack_run(kind, LAYOUT_PLANAR, _ptr(planar), 0, n, by_layout, 0)
        var routed = True
        for j in range(n):
            if by_layout[j] != want[j]:
                routed = False
        suite.check(
            routed, "and unpack_run routes " + _label(kind) + " by layout"
        )


def test_planar_dot(mut suite: Suite) raises:
    suite.group("nn.repack planar dot agrees with the fused one")

    var kinds = _kinds()
    for i in range(len(kinds)):
        var kind = kinds[i]
        var per = block_elements(kind)
        var n = per * 4
        var packed = _blocks(kind, 4, 0x13198A2E03707344 + UInt64(kind))
        var planar = _zeros(planar_row_bytes(kind, n))
        repack_row(kind, _ptr(packed), 0, n, _ptr(planar), 0)

        var w = _floats(n)
        dequant_run(kind, _ptr(packed), 0, n, w, 0)

        var rng = Rng(0xA4093822299F31D0 + UInt64(kind))
        var x = List[Float32]()
        for _ in range(n):
            x.append(rng.unit())

        var slow = Float32(0)
        var mag = Float32(0)
        for j in range(n):
            var term = w[j] * x[j]
            slow += term
            mag += term if term >= 0 else -term

        var planar_total = planar_row_dot(kind, _ptr(planar), 0, x, 0, n)
        suite.check(
            _close(planar_total, slow, 1e-4 * mag + 1e-6),
            _label(kind) + " planar dot matches dequantizing first",
        )

        var fused = row_dot(kind, _ptr(packed), 0, x, 0, n)
        suite.check(
            _close(planar_total, fused, 1e-4 * mag + 1e-6),
            "and matches the fused " + _label(kind) + " path on the blocks",
        )


def test_planar_matvec(mut suite: Suite) raises:
    """A whole tensor, so the row stride is exercised and not just one row.

    A stride bug is invisible with one row and is the failure that reads as a
    model which is fine for its first output and noise after it, so the tensor
    here has several rows and every one of them is checked.
    """
    suite.group("nn.repack matvec over a planar tensor")

    var kinds = _kinds()
    for i in range(len(kinds)):
        var kind = kinds[i]
        var per = block_elements(kind)
        var cols = per * 2
        var rows = 3
        var packed = _blocks(kind, 2 * rows, 0x082EFA98EC4E6C89 + UInt64(kind))

        var stride = planar_row_bytes(kind, cols)
        var planar = _zeros(stride * rows)
        var block_stride = block_bytes(kind) * (cols // per)
        for r in range(rows):
            repack_row(
                kind,
                _ptr(packed),
                r * block_stride,
                cols,
                _ptr(planar),
                r * stride,
            )

        var w = Tensor(Int(packed.unsafe_ptr()), kind, cols, rows)
        var wp = w.as_planar(Int(planar.unsafe_ptr()))
        suite.check(
            wp.row_bytes() == stride,
            _label(kind) + " row_bytes follows the layout",
        )
        suite.check(
            wp.row(1) - wp.row(0) == stride,
            "and so does the address of the second row",
        )

        var rng = Rng(0x452821E638D01377 + UInt64(kind))
        var x = Buffer(cols)
        for j in range(cols):
            x.data[j] = rng.unit()

        var fused = Buffer(rows)
        matvec(w, x, fused)
        var got = Buffer(rows)
        matvec(wp, x, got)

        var agree = True
        for r in range(rows):
            var mag = Float32(0)
            var row = _floats(cols)
            dequant_run(kind, _ptr(packed), r * block_stride, cols, row, 0)
            for j in range(cols):
                var term = row[j] * x.data[j]
                mag += term if term >= 0 else -term
            if not _close(got.data[r], fused.data[r], 1e-4 * mag + 1e-6):
                agree = False
        suite.check(
            agree,
            "a planar matvec over "
            + _label(kind)
            + " matches the same weight read as blocks",
        )


def test_refusals(mut suite: Suite) raises:
    suite.group("nn.repack refusals")

    var packed = _blocks(Q_Q4_K, 1, 0xBE5466CF34E90C6C)
    var planar = _zeros(planar_row_bytes(Q_Q4_K, 256))

    var raised = False
    try:
        repack_row(Q_F32, _ptr(packed), 0, 256, _ptr(planar), 0)
    except:
        raised = True
    suite.check(raised, "a type with no planar form cannot be repacked")

    raised = False
    try:
        repack_row(Q_Q4_K, _ptr(packed), 0, 300, _ptr(planar), 0)
    except:
        raised = True
    suite.check(raised, "and neither can a partial block")

    repack_row(Q_Q4_K, _ptr(packed), 0, 256, _ptr(planar), 0)

    raised = False
    try:
        _ = planar_row_dot(Q_Q4_K, _ptr(planar), 0, _floats(256), 0, 300)
    except:
        raised = True
    suite.check(raised, "a planar dot wants whole groups too")

    raised = False
    try:
        _ = planar_row_dot(Q_F16, _ptr(planar), 0, _floats(256), 0, 256)
    except:
        raised = True
    suite.check(raised, "and a type with no planar form has no planar dot")

    raised = False
    var small = _floats(16)
    try:
        planar_run(Q_Q4_K, _ptr(planar), 0, 256, small, 0)
    except:
        raised = True
    suite.check(raised, "and decoding will not write past the end of a list")


def run_on_device(mut suite: Suite, ctx: DeviceContext) raises:
    """The same weights the host tests use, read by the device matvec.

    The host tests above cover every type molla repacks and the device tests in
    `test_gpu.mojo` cover one, which was fine while the device had two ways of
    reading a row and every type went down one of them. It has four now, and
    which one a type takes is decided by `quant_form`, so the thing worth
    checking is that every type gets an answer and not that one representative
    does.

    Sixty four columns and five rows, which is two 32 value blocks a row for the
    small types and a quarter of a 256 value block for the k types. The k types
    need a whole block, so they get 256 columns instead, which is the only shape
    difference between the two halves of the table.
    """
    suite.group("nn.repack on the device")

    comptime if not has_accelerator():
        suite.check(True, "skipped, this build has no device code in it")
        return
    else:
        var kinds = _kinds()
        for ki in range(len(kinds)):
            var kind = kinds[ki]
            var per = block_elements(kind)
            var cols = per if per > 64 else 64
            var rows = 5
            var stride = planar_row_bytes(kind, cols)
            var block_stride = block_bytes(kind) * (cols // per)

            var packed = _blocks(
                kind, rows * (cols // per), 0xC0FFEE0D15EA5E00 + UInt64(kind)
            )

            var x = Buffer(cols)
            var rng = Rng(0x9E3779B97F4A7C15 + UInt64(kind))
            for j in range(cols):
                x.data[j] = rng.unit()

            var pool = ctx.enqueue_create_buffer[DType.uint8](stride * rows)
            var want = Buffer(rows)
            # Repacked straight into the pool's host mapping and read back out
            # of the same mapping, so the host reference and the device kernel
            # are looking at one copy of the bytes.
            with pool.map_to_host() as mapped:
                var p = RawPtr(unsafe_from_address=Int(mapped.unsafe_ptr()))
                for r in range(rows):
                    repack_row(
                        kind,
                        _ptr(packed),
                        r * block_stride,
                        cols,
                        p,
                        r * stride,
                    )
                for r in range(rows):
                    want.data[r] = planar_row_dot(
                        kind, p, r * stride, x.data, 0, cols
                    )

            var w = Tensor(
                Int(pool.unsafe_ptr()),
                kind,
                cols,
                rows,
                LAYOUT_PLANAR,
                WHERE_DEVICE,
            )
            var got = Buffer(rows)
            device_matvec(ctx, w, x, got)
            # The tensor holds the pool's address and not the pool, for the
            # reason `test_gpu.mojo` gives where the same line is.
            keep(pool)

            var peak = Float32(0)
            var worst = Float32(0)
            for r in range(rows):
                var m = want.data[r] if want.data[r] > 0 else -want.data[r]
                if m > peak:
                    peak = m
                var gap = got.data[r] - want.data[r]
                if gap < 0:
                    gap = -gap
                if gap > worst:
                    worst = gap

            suite.check(
                peak > 0,
                "the " + _label(kind) + " reference is not all zeros",
            )
            suite.check(
                worst <= peak * Float32(1e-5),
                "and the device matvec over "
                + _label(kind)
                + " agrees with the host",
            )
