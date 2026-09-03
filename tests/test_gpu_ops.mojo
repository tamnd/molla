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
from molla.nn.gpu import DeviceVec
from molla.nn.gpu_ops import (
    device_add_into,
    device_argmax,
    device_attend,
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
    var n = heads * head_dim

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

    var raised = False
    try:
        var blind = AttnSpec(4, 2, 32)
        blind.window = 1
        var q = DeviceVec(ctx, 4 * 32)
        var k = DeviceVec(ctx, 40 * 2 * 32)
        var v = DeviceVec(ctx, 40 * 2 * 32)
        var o = DeviceVec(ctx, 4 * 32)
        var s = DeviceVec(ctx, 4 * 40)
        device_attend(ctx, blind, q, k, v, 40, 100, o, s)
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
    for i in range(count * kv_width):
        keys.append(Float32((i * 31 % 173)) / Float32(173) - Float32(0.5))
        values.append(Float32((i * 17 % 149)) / Float32(149) - Float32(0.5))

    var want = Buffer(width)
    var scratch = List[Float32]()
    for _ in range(count):
        scratch.append(0.0)
    attend(spec, q, keys, values, count, pos, want, scratch)

    var dq = DeviceVec(ctx, width)
    var dk = DeviceVec(ctx, count * kv_width)
    var dv = DeviceVec(ctx, count * kv_width)
    var dout = DeviceVec(ctx, width)
    var dscores = DeviceVec(ctx, spec.heads * count)
    dq.upload(q)
    dk.upload_run(keys, 0, count * kv_width)
    dv.upload_run(values, 0, count * kv_width)
    device_attend(ctx, spec, dq, dk, dv, count, pos, dout, dscores)
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
    var kv = DeviceVec(ctx, 4 * 32)
    var scores = DeviceVec(ctx, 2 * 4)
    raised = False
    try:
        device_attend(ctx, attn, q, kv, kv, 8, 7, o, scores)
    except:
        raised = True
    suite.check(
        raised, "attention over more keys than the cache holds is refused"
    )

    raised = False
    try:
        device_attend(ctx, attn, q, kv, kv, 4, 3, o, one)
    except:
        raised = True
    suite.check(raised, "and one with too little room for its scores")

    raised = False
    try:
        device_attend(ctx, attn, q, kv, kv, 0, 0, o, scores)
    except:
        raised = True
    suite.check(raised, "and one with no keys at all")
