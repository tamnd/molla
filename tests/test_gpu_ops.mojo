"""The rest of a block on the device, against the host functions it mirrors.

Every check here is the same shape: run the host version that already has tests
of its own, run the device version on the same input, and compare. That is
deliberate and it is the only kind of test worth having for this. A device
softmax that is self consistent and disagrees with the host one by a thousandth
is a model that answers differently depending on which backend it was started
with, and nobody would ever trace that back to a reduction order.

Unlike `test_gpu`, none of this runs without an accelerator. The matvec could
test its refusals anywhere because a `Tensor` is four integers, and these take
device vectors, which need a real context to exist at all. So the whole file is
one compile time branch and it is a skip on the three machines in the fleet with
no GPU in them.

The tolerances are relative to the peak magnitude of the reference rather than
per element, because half of these produce values that pass through zero and a
relative error at an element that is nearly zero is a number with no meaning in
it.

Three of them are looser than the rest, for three different reasons. The norm
and the attention reduce over a few hundred terms and the host accumulates that
in float64, which Metal has not got, so the device sums in float32 through a
tree. Rope is loose because the host takes its `cos` and `sin` in float64 and
the device takes them in float32. A softcapped attention is looser still
because the cap multiplies its own `tanh` by fifty, so a difference in the last
digit of one arrives at the softmax fifty times larger.

None of the three is a fault on either side, and all three are written down in
`docs/validation/kernels.md` with the figures `scripts/block_oracle.mojo`
printed on each GPU.
"""

from std.math import sqrt
from std.sys.info import has_accelerator

from max.gpu.host import DeviceContext

from harness import Suite

from molla.nn.attention import AttnSpec, attend
from molla.nn.gpu import DeviceHalf, DeviceInts, DeviceVec
from molla.nn.gpu_ops import (
    attend_partials,
    device_add_into,
    device_argmax,
    device_attend,
    device_attend_paged,
    device_store_kv,
    device_store_kv_at,
    device_gelu,
    device_geglu,
    device_rms_norm,
    device_rms_norm_inplace,
    device_rope,
    device_scale_into,
    device_silu,
    device_softmax,
    device_swiglu,
    RopeTables,
)
from molla.nn.kernel import (
    add_into,
    argmax,
    gelu,
    rms_norm,
    scale_into,
    silu,
    softmax,
    swiglu,
)
from molla.nn.quant import Q_F32
from molla.nn.repack import CACHE_BLOCK, CACHE_F16, CACHE_Q8, cache_row
from molla.nn.rope import RopeSpec, rotate_heads
from molla.nn.tensor import Buffer, Tensor


def run(mut suite: Suite) raises:
    """Nothing here runs without a device, so this is the skip and no more.

    The tests are in `run_on_device`, which `main` calls with the one context
    the process owns. A CUDA process gets one `DeviceContext` and hangs on the
    first allocation against a second, so no test module may make its own.
    """
    comptime if not has_accelerator():
        suite.group("gpu ops")
        suite.check(True, "skipped, this build has no device code in it")


def run_on_device(mut suite: Suite, ctx: DeviceContext) raises:
    comptime if not has_accelerator():
        return
    else:
        test_norm(suite, ctx)
        test_softmax(suite, ctx)
        test_activations(suite, ctx)
        test_elementwise(suite, ctx)
        test_argmax(suite, ctx)
        test_rope(suite, ctx)
        test_attend(suite, ctx)
        test_attend_q8(suite, ctx)
        test_attend_paged(suite, ctx)
        test_store_scatter(suite, ctx)
        test_refusals(suite, ctx)


def _wave(n: Int, seed: Int) -> Buffer:
    """Something with both signs and a spread of magnitudes.

    A ramp would pass a norm that had the scale wrong by a constant, since every
    element would be off the same way and a relative comparison against a
    smoothly varying reference hides it. This is not random either, because a
    test that fails one run in fifty is a test people learn to rerun.
    """
    var b = Buffer(n)
    for i in range(n):
        var t = Float32((i * 37 + seed * 11) % 197) / Float32(197)
        b.data[i] = (
            (t - Float32(0.5))
            * Float32(4.0)
            * (Float32(1.0) + Float32((i + seed) % 7))
        )
    return b^


def _worst(got: Buffer, want: Buffer) -> Float32:
    """Largest difference, relative to the largest value in the reference."""
    var peak = Float32(0)
    for i in range(want.elements()):
        var v = want.data[i]
        if v < 0:
            v = -v
        if v > peak:
            peak = v
    if peak == 0:
        peak = Float32(1)
    var worst = Float32(0)
    for i in range(want.elements()):
        var d = got.data[i] - want.data[i]
        if d < 0:
            d = -d
        if d > worst:
            worst = d
    return worst / peak


def _tensor_of(values: List[Float32]) -> Tensor:
    """An f32 weight view over a host list, for the host functions that take one.

    The gain of a norm is a weight and the host `rms_norm` reads it through the
    dequant path like any other. The device one takes a device vector instead,
    because a norm weight is a few thousand values read every token of every
    layer and never written, so it is dequantized once when the model binds.
    This is what lets one test feed both.
    """
    return Tensor(Int(values.unsafe_ptr()), Q_F32, len(values), 1)


def test_norm(mut suite: Suite, ctx: DeviceContext) raises:
    suite.group("device rms_norm")

    var n = 512
    var x = _wave(n, 1)
    var gain = _wave(n, 5)
    for i in range(n):
        # A gain near one, which is what a trained norm weight looks like. A
        # gain that straddles zero would let a sign error through.
        gain.data[i] = Float32(1.0) + gain.data[i] * Float32(0.1)

    var want = Buffer(n)
    rms_norm(want, x, _tensor_of(gain.data), Float32(1e-5))

    var dx = DeviceVec(ctx, n)
    var dg = DeviceVec(ctx, n)
    var dout = DeviceVec(ctx, n)
    dx.upload(x)
    dg.upload(gain)
    device_rms_norm(ctx, dx, dg, dout, Float32(1e-5))
    ctx.synchronize()
    var got = Buffer(n)
    dout.download(got)

    # Looser than the matvec's 1e-5 because the host sums the squares in
    # float64 and this cannot, Metal having no float64 at all. A tree in
    # float32 is what replaces it, and over 512 terms it lands within a few
    # times 1e-7 of the float64 answer, which is better than a sequential
    # float32 sum would do and is still not the same number.
    var worst = _worst(got, want)
    suite.check(worst < 2e-6, "a device norm matches the host one")
    if worst >= 2e-6:
        suite.fail("device rms_norm", "worst " + String(worst))

    # In place, which is what Qwen and Gemma both want and is where an
    # implementation that read its input after writing its output would show up.
    var dsame = DeviceVec(ctx, n)
    dsame.upload(x)
    device_rms_norm_inplace(ctx, dsame, dg, Float32(1e-5))
    ctx.synchronize()
    var inplace = Buffer(n)
    dsame.download(inplace)
    suite.check(
        _worst(inplace, want) < 2e-6, "and it gives the same answer in place"
    )


def test_softmax(mut suite: Suite, ctx: DeviceContext) raises:
    suite.group("device softmax")

    var n = 300
    var x = _wave(n, 2)
    var want = Buffer(n)
    want.copy_from(x)
    softmax(want.data, 0, n)

    var dx = DeviceVec(ctx, n)
    dx.upload(x)
    device_softmax(ctx, dx, n)
    ctx.synchronize()
    var got = Buffer(n)
    dx.download(got)
    suite.check(_worst(got, want) < 1e-6, "a device softmax matches the host")

    var total = Float32(0)
    for i in range(n):
        total += got.data[i]
    suite.check(
        total > Float32(0.9999) and total < Float32(1.0001),
        "and what comes out sums to one",
    )

    # The whole reason the maximum is subtracted. `exp(100)` is two thirds of
    # the way to a float32 infinity, so a row with a couple of these in it comes
    # back as nans from any implementation that skipped the subtraction.
    var big = Buffer(n)
    for i in range(n):
        big.data[i] = Float32(90.0) + Float32(i % 11)
    var want_big = Buffer(n)
    want_big.copy_from(big)
    softmax(want_big.data, 0, n)
    var dbig = DeviceVec(ctx, n)
    dbig.upload(big)
    device_softmax(ctx, dbig, n)
    ctx.synchronize()
    var got_big = Buffer(n)
    dbig.download(got_big)
    suite.check(
        _worst(got_big, want_big) < 1e-6,
        "and a row of large scores does not overflow",
    )


def test_activations(mut suite: Suite, ctx: DeviceContext) raises:
    suite.group("device activations")

    var n = 640
    var gate = _wave(n, 3)
    var up = _wave(n, 4)

    var want = Buffer(n)
    want.copy_from(gate)
    swiglu(want, up)

    var dgate = DeviceVec(ctx, n)
    var dup = DeviceVec(ctx, n)
    dgate.upload(gate)
    dup.upload(up)
    device_swiglu(ctx, dgate, dup)
    ctx.synchronize()
    var got = Buffer(n)
    dgate.download(got)
    suite.check(_worst(got, want) < 1e-6, "a device swiglu matches the host")

    var want_silu = Buffer(n)
    var want_gelu = Buffer(n)
    for i in range(n):
        want_silu.data[i] = silu(gate.data[i])
        want_gelu.data[i] = gelu(gate.data[i])

    var ds = DeviceVec(ctx, n)
    ds.upload(gate)
    device_silu(ctx, ds)
    var dg = DeviceVec(ctx, n)
    dg.upload(gate)
    device_gelu(ctx, dg)
    ctx.synchronize()
    var got_silu = Buffer(n)
    var got_gelu = Buffer(n)
    ds.download(got_silu)
    dg.download(got_gelu)
    suite.check(_worst(got_silu, want_silu) < 1e-6, "and so does silu alone")
    suite.check(_worst(got_gelu, want_gelu) < 1e-6, "and gelu alone")

    # Gemma gates with gelu rather than silu, and the two differ by a few
    # percent in the middle of their range, which is a model that is subtly
    # worse rather than a model that is broken.
    var want_geglu = Buffer(n)
    for i in range(n):
        want_geglu.data[i] = gelu(gate.data[i]) * up.data[i]
    var dgg = DeviceVec(ctx, n)
    dgg.upload(gate)
    device_geglu(ctx, dgg, dup)
    ctx.synchronize()
    var got_geglu = Buffer(n)
    dgg.download(got_geglu)
    suite.check(
        _worst(got_geglu, want_geglu) < 1e-6, "and a geglu gates with gelu"
    )


def test_elementwise(mut suite: Suite, ctx: DeviceContext) raises:
    suite.group("device elementwise")

    var n = 1000
    var acc = _wave(n, 6)
    var x = _wave(n, 7)

    var want = Buffer(n)
    want.copy_from(acc)
    add_into(want, x)

    var dacc = DeviceVec(ctx, n)
    var dx = DeviceVec(ctx, n)
    dacc.upload(acc)
    dx.upload(x)
    device_add_into(ctx, dacc, dx)
    ctx.synchronize()
    var got = Buffer(n)
    dacc.download(got)
    suite.check(_worst(got, want) == 0, "a residual add is exact on both sides")

    var want_scaled = Buffer(n)
    want_scaled.copy_from(x)
    scale_into(want_scaled, Float32(0.125))
    var dscale = DeviceVec(ctx, n)
    dscale.upload(x)
    device_scale_into(ctx, dscale, Float32(0.125))
    ctx.synchronize()
    var got_scaled = Buffer(n)
    dscale.download(got_scaled)
    suite.check(_worst(got_scaled, want_scaled) == 0, "and so is a scale")

    # A length that is not a multiple of the tile, which is where a kernel that
    # covered its vector with a single pass rather than a strided loop would
    # leave the tail untouched.
    var odd = 1003
    var tail = _wave(odd, 8)
    var want_tail = Buffer(odd)
    want_tail.copy_from(tail)
    scale_into(want_tail, Float32(2.0))
    var dtail = DeviceVec(ctx, odd)
    dtail.upload(tail)
    device_scale_into(ctx, dtail, Float32(2.0))
    ctx.synchronize()
    var got_tail = Buffer(odd)
    dtail.download(got_tail)
    suite.check(
        _worst(got_tail, want_tail) == 0,
        "and a length that is not a whole number of tiles has no tail left",
    )


def test_argmax(mut suite: Suite, ctx: DeviceContext) raises:
    suite.group("device argmax")

    var n = 777
    var x = _wave(n, 9)
    x.data[513] = Float32(99.0)

    var dx = DeviceVec(ctx, n)
    var dout = DeviceVec(ctx, 1)
    dx.upload(x)
    device_argmax(ctx, dx, dout, n)
    ctx.synchronize()
    suite.check(
        Int(dout.at(0)) == argmax(x.data, 0, n),
        "a device argmax finds the same index as the host",
    )
    suite.check(Int(dout.at(0)) == 513, "which is the one that was planted")

    # A tie, which is not a corner case in a logit row that has been through a
    # float16 weight. Both sides have to break it the same way or the two
    # backends produce different tokens from the same model.
    var tie = Buffer(n)
    for i in range(n):
        tie.data[i] = Float32(1.0)
    var dtie = DeviceVec(ctx, n)
    dtie.upload(tie)
    device_argmax(ctx, dtie, dout, n)
    ctx.synchronize()
    suite.check(
        Int(dout.at(0)) == argmax(tie.data, 0, n) and Int(dout.at(0)) == 0,
        "and a tie goes to the lower index on both",
    )

    # Every value negative, which catches a reduction that started from zero
    # rather than from the lowest float there is.
    var down = Buffer(n)
    for i in range(n):
        down.data[i] = Float32(-1000.0) - Float32((i * 13) % 91)
    var ddown = DeviceVec(ctx, n)
    ddown.upload(down)
    device_argmax(ctx, ddown, dout, n)
    ctx.synchronize()
    suite.check(
        Int(dout.at(0)) == argmax(down.data, 0, n),
        "and a row with nothing positive in it still has a largest value",
    )


def test_rope(mut suite: Suite, ctx: DeviceContext) raises:
    suite.group("device rope")

    var heads = 4
    var head_dim = 64

    # Llama 3, which is neox pairing on a base of 500000.
    var spec = RopeSpec(head_dim, Float32(500000.0))
    var none = List[Float32]()
    _rope_case(suite, ctx, spec, heads, head_dim, 137, none, False, "llama 3")

    # Far enough along a sequence to catch an angle whose error scales with the
    # position, which is what a device that formed its own frequency steps in
    # float32 had. That version passed at position 137 and was out by the fourth
    # digit here, so a test that only ever looked at a small position would have
    # called it correct.
    _rope_case(
        suite, ctx, spec, heads, head_dim, 4096, none, False, "position 4096"
    )

    # The other pairing, which is what a converted Llama 2 wants because its
    # weights were permuted to suit. Same rotation, different memory order, and
    # the wrong one is a model that is fluent for a few tokens and then is not.
    var adjacent = RopeSpec(head_dim, Float32(10000.0))
    adjacent.neox = False
    _rope_case(
        suite, ctx, adjacent, heads, head_dim, 41, none, False, "adjacent pairs"
    )

    # Position zero, where every angle is zero and the rotation is the
    # identity. A kernel that had the sign of the angle backwards passes every
    # other case in this test and fails nothing, so it gets its own.
    _rope_case(suite, ctx, spec, heads, head_dim, 0, none, False, "position 0")

    # A rotary dimension shorter than the head, which is what a partial rotary
    # model has. The elements past `dim` carry content and zeroing them is a
    # silent loss rather than an error.
    var partial = RopeSpec(32, Float32(10000.0))
    _rope_case(
        suite, ctx, partial, heads, head_dim, 77, none, False, "partial rotary"
    )

    # YaRN, which is the only path that reads the correction range the host
    # computes and hands over.
    var yarn = RopeSpec.yarn(
        head_dim, Float32(10000.0), Float32(4.0), 2048, True
    )
    _rope_case(suite, ctx, yarn, heads, head_dim, 611, none, False, "yarn")

    # The per pair frequency factors a Llama 3.1 file carries as
    # `rope_freqs.weight`.
    var factors = List[Float32]()
    for i in range(head_dim // 2):
        factors.append(Float32(1.0) + Float32(i % 5) * Float32(0.25))
    _rope_case(
        suite, ctx, spec, heads, head_dim, 200, factors, True, "freq factors"
    )


def _rope_case(
    mut suite: Suite,
    ctx: DeviceContext,
    spec: RopeSpec,
    heads: Int,
    head_dim: Int,
    pos: Int,
    factors: List[Float32],
    use_factors: Bool,
    name: String,
) raises:
    var n = heads * head_dim
    var x = _wave(n, pos + 1)

    var want = Buffer(n)
    want.copy_from(x)
    rotate_heads(spec, want, heads, head_dim, pos, factors, use_factors)

    var dx = DeviceVec(ctx, n)
    dx.upload(x)
    var tables = RopeTables(ctx, spec, factors, use_factors)
    device_rope(ctx, spec, dx, 0, heads, head_dim, pos, tables)
    ctx.synchronize()
    var got = Buffer(n)
    dx.download(got)

    # Both sides take their frequency steps from `molla.nn.rope.step_table`, so
    # what is left here is that the host takes its `cos` and `sin` in float64
    # and the device takes them in float32. That does not grow with the
    # position, which is the point of the table and is why this sits at the same
    # tolerance as the reductions rather than an order above them.
    var worst = _worst(got, want)
    suite.check(worst < 2e-6, "device rope matches the host for " + name)
    if worst >= 2e-6:
        suite.fail("device rope " + name, "worst " + String(worst))


def test_attend(mut suite: Suite, ctx: DeviceContext) raises:
    suite.group("device attention")

    # Grouped query attention with four query heads to a kv head, which is the
    # shape every recent Llama and Qwen uses, and the mapping is where an off by
    # one reads the wrong head's keys and still produces plausible numbers.
    var spec = AttnSpec(8, 2, 32)
    _attend_case(suite, ctx, spec, 40, 39, "grouped query", 2e-6)

    # One key, which is the first token of a sequence and is where a softmax
    # over a single score has to come back as exactly one.
    _attend_case(suite, ctx, spec, 1, 0, "a single key", 2e-6)

    # Multi head, where every query head has its own kv head.
    var mha = AttnSpec(4, 4, 32)
    _attend_case(suite, ctx, mha, 20, 19, "multi head", 2e-6)

    # A sliding window, which is what masks most of the cache on a Mistral or a
    # Gemma 3 local layer. The device version scores the masked keys anyway and
    # leaves them where they are rather than packing them to the front the way
    # the host does, so this is the case that proves the two agree.
    var windowed = AttnSpec(4, 2, 32)
    windowed.window = 8
    _attend_case(suite, ctx, windowed, 40, 39, "a sliding window", 2e-6)

    # A window with attention sinks, where the first few keys stay visible no
    # matter how far the window has moved past them.
    var sinks = AttnSpec(4, 2, 32)
    sinks.window = 8
    sinks.sinks = 3
    _attend_case(suite, ctx, sinks, 40, 39, "sinks outside the window", 2e-6)

    # Logit softcapping, which Gemma 2 applies to every attention score.
    var capped = AttnSpec(4, 2, 32)
    capped.softcap = Float32(50.0)
    # A cap of 50 multiplies its own tanh, so a float32 tanh a part in ten
    # million from the host's float64 one reaches the softmax fifty times that.
    # Hence the wider gate, which is set by a constant in a model file rather
    # than by anything about the kernel.
    _attend_case(suite, ctx, capped, 30, 29, "softcapped scores", 1e-5)

    # Long enough that the keys are cut into slices and joined by a second
    # kernel, which is the path a decode takes at any real context. Four heads
    # and 512 keys gives four slices. The host reference is unchanged, which is
    # the point: a softmax split into pieces and put back together is the same
    # softmax, and this is what says so.
    var split = AttnSpec(4, 2, 32)
    _attend_case(suite, ctx, split, 512, 511, "a context split in slices", 3e-6)

    # The same split with a window that masks everything before the last 40
    # keys, so the first three slices see nothing at all. A slice with no
    # visible key has no maximum to subtract and has to be dropped rather than
    # normalised, and getting that wrong weights every masked key at one, which
    # is a wrong answer rather than a nan.
    var split_win = AttnSpec(4, 2, 32)
    split_win.window = 40
    _attend_case(
        suite, ctx, split_win, 512, 511, "slices a window empties", 3e-6
    )

    # And the same again with sinks, where the visible keys are in the first
    # slice and the last one and the slices between them are empty.
    var split_sink = AttnSpec(4, 2, 32)
    split_sink.window = 40
    split_sink.sinks = 3
    _attend_case(suite, ctx, split_sink, 512, 511, "slices around sinks", 3e-6)

    var raised = False
    try:
        var blind = AttnSpec(4, 2, 32)
        blind.window = 1
        var q = DeviceVec(ctx, 4 * 32)
        var k = DeviceHalf(ctx, 40 * 2 * 32)
        var v = DeviceHalf(ctx, 40 * 2 * 32)
        var o = DeviceVec(ctx, 4 * 32)
        var s = DeviceVec(ctx, 4 * 40)
        var part = DeviceVec(ctx, attend_partials(blind, 1, 40))
        device_attend(ctx, blind, q, k, v, 40, 100, o, s, part)
    except:
        raised = True
    suite.check(
        raised,
        (
            "a position that can see no keys is refused rather than divided by"
            " zero"
        ),
    )


def _attend_case(
    mut suite: Suite,
    ctx: DeviceContext,
    spec: AttnSpec,
    count: Int,
    pos: Int,
    name: String,
    gate: Float32,
) raises:
    var width = spec.heads * spec.head_dim
    var kv_width = spec.kv_heads * spec.head_dim

    var q = _wave(width, count)
    var keys = List[Float32]()
    var values = List[Float32]()
    # Rounded to half on the way into the list rather than on the way into the
    # cache, because the cache is float16 and the host reference has to be given
    # the values the device will actually read. Rounding twice in two places is
    # the same number here and reads as a comparison of like with like.
    for i in range(count * kv_width):
        keys.append(
            Float32(
                Float16(Float32((i * 31 % 173)) / Float32(173) - Float32(0.5))
            )
        )
        values.append(
            Float32(
                Float16(Float32((i * 17 % 149)) / Float32(149) - Float32(0.5))
            )
        )

    var want = Buffer(width)
    var scratch = List[Float32]()
    for _ in range(count):
        scratch.append(0.0)
    attend(spec, q, keys, values, count, pos, want, scratch)

    var dq = DeviceVec(ctx, width)
    var dk = DeviceHalf(ctx, count * kv_width)
    var dv = DeviceHalf(ctx, count * kv_width)
    var dout = DeviceVec(ctx, width)
    var dscores = DeviceVec(ctx, spec.heads * count)
    # Whatever `attend_partials` asks for, which is what decides whether this
    # case goes through one kernel or through the split and the join. A case
    # with a few dozen keys takes the single kernel however wide the grid is,
    # and the cases with a few hundred take the split, so both paths are checked
    # against the same host reference by the same code.
    var dpart = DeviceVec(ctx, attend_partials(spec, 1, count))
    dq.upload(q)
    dk.upload_run(keys, 0, count * kv_width)
    dv.upload_run(values, 0, count * kv_width)
    device_attend(ctx, spec, dq, dk, dv, count, pos, dout, dscores, dpart)
    ctx.synchronize()
    var got = Buffer(width)
    dout.download(got)

    # The same allowance the norm gets and for the same reason. A score is a
    # dot product over `head_dim` and the sum over keys is another reduction on
    # top of it, both in float32 here against float64 on the host.
    var worst = _worst(got, want)
    suite.check(worst < gate, "device attention matches the host for " + name)
    if worst >= gate:
        suite.fail("device attention " + name, "worst " + String(worst))


def test_attend_paged(mut suite: Suite, ctx: DeviceContext) raises:
    suite.group("device attention over a cell pool")

    # The same grouped query shape the contiguous test starts with, forty
    # positions living in a pool of sixty four cells.
    var spec = AttnSpec(8, 2, 32)
    _attend_paged_case(suite, ctx, spec, 40, 64, 39, "grouped query", 2e-6)

    # One position in a pool that is nearly all somebody else's, which is what
    # the first token of a new sequence meets on a busy card.
    _attend_paged_case(suite, ctx, spec, 1, 48, 0, "a single cell", 2e-6)

    # A sliding window over cells that are not in order, which is where a
    # window counted in cells rather than in positions gives the wrong answer.
    var windowed = AttnSpec(4, 2, 32)
    windowed.window = 8
    _attend_paged_case(suite, ctx, windowed, 40, 64, 39, "a window", 2e-6)

    var sinks = AttnSpec(4, 2, 32)
    sinks.window = 8
    sinks.sinks = 3
    _attend_paged_case(suite, ctx, sinks, 40, 64, 39, "sinks scattered", 2e-6)

    # Long enough to be cut into slices. Every slice holds a mix of this
    # sequence's cells and cells it may not read, which the contiguous split
    # never sees, and a slice that happens to hold none of them has to come
    # back as nothing seen rather than as a division by zero.
    var split = AttnSpec(4, 2, 32)
    _attend_paged_case(
        suite, ctx, split, 512, 600, 511, "a pool in slices", 3e-6
    )

    var split_win = AttnSpec(4, 2, 32)
    split_win.window = 40
    _attend_paged_case(
        suite, ctx, split_win, 512, 600, 511, "slices a window empties", 3e-6
    )

    # A window that is a prefix of the vectors it was uploaded in, which is what
    # every step does: the buffers are the pool and the window is how much of it
    # the pool is using. The cells past the window here would all be visible if
    # they were read.
    _attend_paged_case(
        suite, ctx, spec, 40, 64, 39, "a window inside its buffer", 2e-6, 96
    )
    _attend_paged_case(
        suite, ctx, split, 512, 600, 511, "and one cut into slices", 3e-6, 424
    )

    var raised = False
    try:
        var short = AttnSpec(4, 2, 32)
        var q = DeviceVec(ctx, 4 * 32)
        var k = DeviceHalf(ctx, 40 * 2 * 32)
        var v = DeviceHalf(ctx, 40 * 2 * 32)
        var o = DeviceVec(ctx, 4 * 32)
        var s = DeviceVec(ctx, 4 * 40)
        var part = DeviceVec(ctx, attend_partials(short, 1, 40))
        var cellv = DeviceInts(ctx, 40)
        var host = List[Int32]()
        for i in range(20):
            host.append(Int32(i))
        device_attend_paged(
            ctx, short, q, k, v, host, cellv, 40, 39, o, s, part
        )
    except:
        raised = True
    suite.check(
        raised, "a host window shorter than the one asked for is refused"
    )

    raised = False
    try:
        var short = AttnSpec(4, 2, 32)
        var q = DeviceVec(ctx, 4 * 32)
        var k = DeviceHalf(ctx, 40 * 2 * 32)
        var v = DeviceHalf(ctx, 40 * 2 * 32)
        var o = DeviceVec(ctx, 4 * 32)
        var s = DeviceVec(ctx, 4 * 40)
        var part = DeviceVec(ctx, attend_partials(short, 1, 40))
        var cellv = DeviceInts(ctx, 20)
        var host = List[Int32]()
        for i in range(40):
            host.append(Int32(i))
        device_attend_paged(
            ctx, short, q, k, v, host, cellv, 40, 39, o, s, part
        )
    except:
        raised = True
    suite.check(raised, "and so is a device window that does not reach it")


def _attend_paged_case(
    mut suite: Suite,
    ctx: DeviceContext,
    spec: AttnSpec,
    count: Int,
    cells: Int,
    pos: Int,
    name: String,
    gate: Float32,
    slack: Int = 0,
) raises:
    """`count` positions scattered over a pool of `cells`, against the host.

    `slack` cells past the window, on both sides, holding a position the query
    can see and numbers it must not read. A step reads a prefix of vectors that
    are allocated once at the size of the pool, so the entries past the window
    are always there and a kernel that took the buffer's length for the window
    would produce a number that is wrong by however much they weigh.

    The scatter is a stride of seven, which is coprime with every pool size
    here, so a position lands nowhere near the order it was written in and the
    cells between the positions hold junk that no query may read. Half of those
    are free and the other half hold a position past the query's, which are the
    two reasons a cell can be invisible and are the two the kernel has to get
    right. A kernel that ignored the mask and walked the pool as a run would
    still produce a number, and it would not be this one. The free half holds
    an infinity rather than a plausible float, for the reason the loop that
    writes it gives.

    The reference is the host attention given the same window, which is the
    same comparison `_attend_case` makes one level down: the host paged path is
    checked against the host contiguous one in `tests/test_attention.mojo`, so
    what is being checked here is the kernel rather than the idea.
    """
    var width = spec.heads * spec.head_dim
    var kv_width = spec.kv_heads * spec.head_dim
    var pool = cells + slack

    var q = _wave(width, count)

    # Junk over the whole pool first, from a different generator than the keys,
    # so a row read out of the wrong cell cannot happen to hold the numbers the
    # right cell holds.
    var keys = List[Float32]()
    var values = List[Float32]()
    for i in range(pool * kv_width):
        keys.append(
            Float32(
                Float16(Float32((i * 53 % 101)) / Float32(101) - Float32(0.5))
            )
        )
        values.append(
            Float32(
                Float16(Float32((i * 41 % 97)) / Float32(97) - Float32(0.5))
            )
        )

    # Minus one rather than `CELL_FREE`, because the kernel is told a position
    # is negative and nothing more, and a test that imported the engine's name
    # for it would be checking that two constants agree instead.
    var held = List[Int]()
    for c in range(cells):
        held.append(-1 if c % 2 == 0 else pos + 1 + c % 5)
    # Past the window, a position the query can see, so that reading one of
    # these changes the answer rather than being masked off anyway.
    for _ in range(slack):
        held.append(0)
    for i in range(count):
        var cell = i * 7 % cells
        held[cell] = i
        for d in range(kv_width):
            var at = i * kv_width + d
            keys[cell * kv_width + d] = Float32(
                Float16(Float32((at * 31 % 173)) / Float32(173) - Float32(0.5))
            )
            values[cell * kv_width + d] = Float32(
                Float16(Float32((at * 17 % 149)) / Float32(149) - Float32(0.5))
            )

    # A free cell holds whatever the driver left there, and half of the wrong
    # bits is an infinity, so the free cells here hold one. That is what a
    # fresh allocation on the 4090 can produce and what a Metal one cannot,
    # since it comes back zeroed. The cells holding another sequence's future
    # stay finite, because that sequence really did write numbers there.
    #
    # This passes whether or not the kernel skips a masked row or multiplies it
    # by zero, since neither target turns zero times an infinity into a nan
    # today. It is here as the canary for the day one of them does, which is a
    # toolchain changing its floating point rather than anything in molla
    # changing, and the failure it would announce is a nan logit.
    for c in range(cells):
        if held[c] != -1:
            continue
        for d in range(kv_width):
            keys[c * kv_width + d] = Float32(1e30)
            values[c * kv_width + d] = Float32(1e30)

    var want = Buffer(width)
    var scratch = List[Float32]()
    for _ in range(cells):
        scratch.append(0.0)
    attend(spec, q, keys, values, cells, pos, want, scratch, held)

    var host = List[Int32]()
    for c in range(pool):
        host.append(Int32(held[c]))

    var dq = DeviceVec(ctx, width)
    var dk = DeviceHalf(ctx, pool * kv_width)
    var dv = DeviceHalf(ctx, pool * kv_width)
    var dout = DeviceVec(ctx, width)
    var dscores = DeviceVec(ctx, spec.heads * cells)
    var dpart = DeviceVec(ctx, attend_partials(spec, 1, cells))
    var cellv = DeviceInts(ctx, pool)
    cellv.queue_in(host)
    dq.upload(q)
    dk.upload_run(keys, 0, pool * kv_width)
    dv.upload_run(values, 0, pool * kv_width)
    device_attend_paged(
        ctx, spec, dq, dk, dv, host, cellv, cells, pos, dout, dscores, dpart
    )
    ctx.synchronize()
    var got = Buffer(width)
    dout.download(got)

    var worst = _worst(got, want)
    suite.check(
        worst < gate, "paged device attention matches the host for " + name
    )
    if worst >= gate:
        suite.fail("paged device attention " + name, "worst " + String(worst))


def _quantize_q8(values: List[Float32], mut out: List[Float32]):
    """What a q8_0 cache row holds, computed on the host.

    Written out rather than borrowed from the repack, because the repack
    quantizes a weight matrix and this is the cache, and a reference that shares
    its arithmetic with the thing it is checking is not one. Blocks of
    `CACHE_BLOCK`, a factor of the largest magnitude over 127, rounded to nearest
    away from zero, and the result is what the kernel is expected to read back.
    """
    var n = len(values)
    for i in range(0, n, CACHE_BLOCK):
        var amax = Float32(0)
        for k in range(CACHE_BLOCK):
            var a = values[i + k]
            if a < 0:
                a = -a
            if a > amax:
                amax = a
        var inv = Float32(0) if amax == 0 else Float32(127.0) / amax
        var s = Float32(Float16(amax / Float32(127.0)))
        for k in range(CACHE_BLOCK):
            var v = values[i + k] * inv
            var r = v + Float32(0.5) if v >= 0 else v - Float32(0.5)
            var q = Int(r)
            if q > 127:
                q = 127
            if q < -127:
                q = -127
            out.append(Float32(q) * s)


def test_attend_q8(mut suite: Suite, ctx: DeviceContext) raises:
    """The q8_0 cache, stored by one kernel and read back by another.

    The round trip is the test. A host quantizer says what the row is supposed
    to hold, the store kernel writes it, attention reads it, and the answer has
    to match the host attending to the values the host quantizer produced. That
    catches the two mistakes this layout invites, which are a scale plane read
    at the wrong offset and a block whose factor belongs to its neighbour, and
    neither shows up as anything but slightly wrong numbers.
    """
    suite.group("device attention over a q8 cache")

    var spec = AttnSpec(8, 2, 32)
    var width = spec.heads * spec.head_dim
    var kv_width = spec.kv_heads * spec.head_dim
    var count = 40
    var pos = count - 1

    var q = _wave(width, 3)
    var raw_k = List[Float32]()
    var raw_v = List[Float32]()
    for i in range(count * kv_width):
        raw_k.append(Float32((i * 31 % 173)) / Float32(173) - Float32(0.5))
        raw_v.append(Float32((i * 17 % 149)) / Float32(149) - Float32(0.5))
    var keys = List[Float32]()
    var values = List[Float32]()
    _quantize_q8(raw_k, keys)
    _quantize_q8(raw_v, values)

    var want = Buffer(width)
    var scratch = List[Float32]()
    for _ in range(count):
        scratch.append(0.0)
    attend(spec, q, keys, values, count, pos, want, scratch)

    var row = cache_row(CACHE_Q8, kv_width)
    var dq = DeviceVec(ctx, width)
    var src = DeviceVec(ctx, count * kv_width)
    var dk = DeviceHalf(ctx, count * row)
    var dv = DeviceHalf(ctx, count * row)
    var dout = DeviceVec(ctx, width)
    var dscores = DeviceVec(ctx, spec.heads * count)
    var dpart = DeviceVec(ctx, attend_partials(spec, 1, count))
    dq.upload(q)
    src.copy_in(raw_k)
    device_store_kv(ctx, dk, 0, src, kv_width, count, CACHE_Q8)
    ctx.synchronize()
    src.copy_in(raw_v)
    device_store_kv(ctx, dv, 0, src, kv_width, count, CACHE_Q8)
    device_attend(
        ctx,
        spec,
        dq,
        dk,
        dv,
        count,
        pos,
        dout,
        dscores,
        dpart,
        1,
        CACHE_Q8,
    )
    ctx.synchronize()
    var got = Buffer(width)
    dout.download(got)

    var worst = _worst(got, want)
    suite.check(worst < 2e-6, "a q8 cache reads back what the host quantized")
    if worst >= 2e-6:
        suite.fail("q8 attention", "worst " + String(worst))

    # A width that is not a whole number of blocks has no q8 row, and the
    # refusal is on the host where it can say so rather than in a kernel that
    # would round the width down and drop the tail of every key.
    var raised = False
    try:
        _ = cache_row(CACHE_Q8, 48)
    except:
        raised = True
    suite.check(raised, "a width that is not whole blocks has no q8 row")


def test_store_scatter(mut suite: Suite, ctx: DeviceContext) raises:
    """A row lands in the cell it was handed and nowhere else.

    Both halves of that matter and the second one is the one worth building a
    test around. A scatter that writes the right rows and also writes a row it
    was not asked for corrupts another sequence's cache, which reads as fluent
    text about somebody else's conversation and is not traceable to anything.
    So the pool is filled with a pattern first, and every cell the index vector
    does not name has to still hold it afterwards.

    What a placed row is compared against is the same rows written by the
    contiguous store, bit for bit, rather than a reimplementation of the
    quantizer. The two stores have to agree exactly or a cache written on one
    path and read on the other is a different cache.
    """
    suite.group("device scattered cache store")

    var kv_width = 64
    var rows = 4
    var cells = 8

    # Out of order, not contiguous, and not starting at zero, which is what a
    # pool with other sequences in it hands back.
    var order = List[Int32]()
    order.append(5)
    order.append(1)
    order.append(6)
    order.append(0)

    var idx = DeviceInts(ctx, rows)
    idx.queue_in(order)
    ctx.synchronize()
    suite.check(idx.at(2) == 6, "the index vector arrives on the card")

    var raw = List[Float32]()
    for i in range(rows * kv_width):
        raw.append(Float32((i * 29 % 211)) / Float32(211) - Float32(0.5))
    var pattern = List[Float32]()
    for i in range(cells * kv_width):
        pattern.append(Float32(1) + Float32(i % 5))

    var forms = List[Int]()
    forms.append(CACHE_F16)
    forms.append(CACHE_Q8)
    for fi in range(len(forms)):
        var form = forms[fi]
        var row = cache_row(form, kv_width)
        var name = " at f16" if form == CACHE_F16 else " at q8"

        var src = DeviceVec(ctx, rows * kv_width)
        var wide = DeviceVec(ctx, cells * kv_width)
        var flat = DeviceHalf(ctx, rows * row)
        var pool = DeviceHalf(ctx, cells * row)
        src.copy_in(raw)
        wide.copy_in(pattern)
        device_store_kv(ctx, flat, 0, src, kv_width, rows, form)
        device_store_kv(ctx, pool, 0, wide, kv_width, cells, form)
        ctx.synchronize()

        var want = List[Int]()
        flat.download_bits(want)
        var before = List[Int]()
        pool.download_bits(before)

        device_store_kv_at(ctx, pool, idx, src, kv_width, rows, form)
        ctx.synchronize()
        var after = List[Int]()
        pool.download_bits(after)

        # A placed row is compared over the halves a store actually writes,
        # which at q8 is the quant plane and the scale plane and not the pad
        # `CACHE_ALIGN` leaves at the end. Nothing writes that pad, on either
        # path, so on a card whose fresh allocations are not zero the two pools
        # hold different rubbish there and comparing it would be comparing what
        # the driver last had in that page. A cell nobody wrote is still
        # compared over the whole row, because that is the same buffer read
        # twice and the pad has to come back unchanged like everything else.
        var live = row
        if form == CACHE_Q8:
            live = kv_width // 2 + kv_width // CACHE_BLOCK

        var placed = 0
        var kept = 0
        for c in range(cells):
            var from_row = -1
            for t in range(rows):
                if Int(order[t]) == c:
                    from_row = t
            var same = True
            var span = row if from_row < 0 else live
            for k in range(span):
                var expect = before[c * row + k]
                if from_row >= 0:
                    expect = want[from_row * row + k]
                if after[c * row + k] != expect:
                    same = False
            if from_row >= 0:
                placed += 1 if same else 0
            else:
                kept += 1 if same else 0
        suite.check(
            placed == rows, "every row lands in the cell it was given" + name
        )
        suite.check(kept == cells - rows, "and no other cell is written" + name)

    var pool = DeviceHalf(ctx, cells * kv_width)
    var src = DeviceVec(ctx, rows * kv_width)
    var raised = False
    try:
        device_store_kv_at(ctx, pool, idx, src, kv_width, rows + 1)
    except:
        raised = True
    suite.check(raised, "more rows than the index vector holds is refused")

    raised = False
    try:
        var narrow = DeviceHalf(ctx, kv_width - 1)
        device_store_kv_at(ctx, narrow, idx, src, kv_width, rows)
    except:
        raised = True
    suite.check(raised, "and so is a pool with no room for one row")


def test_refusals(mut suite: Suite, ctx: DeviceContext) raises:
    """The shape checks, which are the cheap half of not corrupting memory.

    A kernel cannot raise and a device write past the end of a buffer is not a
    fault on either target, it is another buffer changing under whatever owns
    it. So every entry point checks its sizes on the host before it queues
    anything, and these are those checks.
    """
    suite.group("device ops refusals")

    var n = 64
    var a = DeviceVec(ctx, n)
    var b = DeviceVec(ctx, n)
    var small = DeviceVec(ctx, 32)
    var one = DeviceVec(ctx, 1)

    var raised = False
    try:
        device_rms_norm(ctx, a, small, b, Float32(1e-5))
    except:
        raised = True
    suite.check(raised, "a norm with a gain of the wrong width is refused")

    raised = False
    try:
        device_rms_norm(ctx, a, a, small, Float32(1e-5))
    except:
        raised = True
    suite.check(raised, "and one whose output is a different size")

    raised = False
    try:
        device_rms_norm_inplace(ctx, a, small, Float32(1e-5))
    except:
        raised = True
    suite.check(raised, "and the in place form checks its gain the same way")

    raised = False
    try:
        device_softmax(ctx, a, n + 1)
    except:
        raised = True
    suite.check(raised, "a softmax over more than the vector holds is refused")

    raised = False
    try:
        device_softmax(ctx, a, 0)
    except:
        raised = True
    suite.check(raised, "and so is one over nothing")

    raised = False
    try:
        device_swiglu(ctx, a, small)
    except:
        raised = True
    suite.check(
        raised, "a gated MLP with halves of different widths is refused"
    )

    raised = False
    try:
        device_add_into(ctx, a, small)
    except:
        raised = True
    suite.check(raised, "and an add of two different widths")

    raised = False
    try:
        device_argmax(ctx, a, one, n + 1)
    except:
        raised = True
    suite.check(raised, "an argmax past the end of its vector is refused")

    var spec = RopeSpec(64, Float32(10000.0))
    var none = List[Float32]()
    var tables = RopeTables(ctx, spec, none, False)
    raised = False
    try:
        device_rope(ctx, spec, a, 0, 2, 64, 0, tables)
    except:
        raised = True
    suite.check(raised, "rope over more heads than the vector holds is refused")

    raised = False
    try:
        device_rope(ctx, spec, a, 0, 1, 32, 0, tables)
    except:
        raised = True
    suite.check(raised, "and a rotary dimension wider than the head")

    raised = False
    try:
        _ = RopeTables(ctx, spec, none, True)
    except:
        raised = True
    suite.check(
        raised, "and building tables that promise factors and hold none"
    )

    # Tables built for a narrower spec than the one they are used with. Which is
    # the mistake the new type makes possible, so it gets a check: a model that
    # alternates two rope specs has two sets of these, and reaching for the
    # wrong one is a plausible thing for a forward pass to do.
    raised = False
    try:
        var narrow = RopeTables(
            ctx, RopeSpec(16, Float32(10000.0)), none, False
        )
        device_rope(ctx, spec, a, 0, 1, 64, 0, narrow)
    except:
        raised = True
    suite.check(raised, "and tables built for a narrower rotary dimension")

    var attn = AttnSpec(2, 1, 32)
    var q = DeviceVec(ctx, 64)
    var o = DeviceVec(ctx, 64)
    var kv = DeviceHalf(ctx, 4 * 32)
    var scores = DeviceVec(ctx, 2 * 4)
    var part = DeviceVec(ctx, attend_partials(attn, 1, 4))
    raised = False
    try:
        device_attend(ctx, attn, q, kv, kv, 8, 7, o, scores, part)
    except:
        raised = True
    suite.check(
        raised, "attention over more keys than the cache holds is refused"
    )

    raised = False
    try:
        device_attend(ctx, attn, q, kv, kv, 4, 3, o, one, part)
    except:
        raised = True
    suite.check(raised, "and one with too little room for its scores")

    raised = False
    try:
        device_attend(ctx, attn, q, kv, kv, 0, 0, o, scores, part)
    except:
        raised = True
    suite.check(raised, "and one with no keys at all")
