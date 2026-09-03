"""Checks every device operation in a block against the host one it mirrors.

The companion to `kernel_oracle`, which does the same job for the matvec. That
one needs the quantization corpus and this one needs nothing, because everything
it compares is float32 arithmetic on numbers it makes up, so it runs on any
machine with a GPU and no fixtures at all.

It exists next to `tests/test_gpu_ops.mojo` rather than instead of it. The test
asserts and prints nothing, which is what a suite should do, and that means the
only figure it ever reports is one that already failed. This prints the distance
for every operation whether or not it is inside tolerance, which is what a
document recording per target numerics needs, and it is where the numbers in
`docs/validation/kernels.md` came from.

The gate is per operation and not global. Four of them have their own tolerance
and the reasons are different in each case, so a single number across all of
them would either be too loose to catch a real fault in the tight ones or would
fail the loose ones for behaving as designed.

Usage:

    mojo run -I src scripts/block_oracle.mojo
"""

from std.sys import exit
from std.sys.info import has_accelerator

from max.gpu.host import DeviceContext

from molla.nn.attention import AttnSpec, attend
from molla.nn.gpu import DeviceVec
from molla.nn.gpu_ops import (
    device_add_into,
    device_argmax,
    device_attend,
    device_geglu,
    device_gelu,
    device_rms_norm,
    device_rope,
    RopeTables,
    device_scale_into,
    device_silu,
    device_softmax,
    device_swiglu,
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
from molla.sys.device import build_target_arch

comptime TIGHT = 1e-6
"""For the operations that are elementwise or nearly so.

An add, a scale, an activation and a softmax all touch each element once and in
an order that does not matter, so the two sides differ only where the host used
a float64 intermediate. Anything above this in one of them is a real fault and
not an accumulation difference.
"""

comptime WIDE = 2e-6
"""For the norm, and for attention.

Both reduce over a few hundred terms, and the host accumulates that in float64
while Metal has no float64 to accumulate in. A float32 tree is what replaces it,
which is closer to the float64 answer than a sequential float32 sum would be,
and is still not the same number.
"""

comptime CAPPED = 1e-5
"""For attention with a logit softcap, which is Gemma 2 and nothing else here.

The cap multiplies its `tanh` by itself, and the cap is 50, so a float32 `tanh`
that is a part in ten million from the float64 one the host takes arrives at the
softmax fifty times that. Everything downstream of it is the ordinary attention
path, so this is the one tolerance in the file that is set by a constant in a
model file rather than by the arithmetic, and a model with a larger cap would
want a larger number here.

It is not a fudge for a difference nobody looked at. Going through `exp` rather
than the target's own `tanh` took the 4090 from 2e-5 to 1.9e-6, which is where
the float32 exponential runs out, and this leaves the five times headroom the
other gates have rather than the five percent that number would have had.
"""

comptime ANGLED = 2e-6
"""For rope, where what is left is the difference between two sine functions.

This was 2e-5 for one afternoon and the number is worth keeping in view, because
loosening it would have been the wrong fix and it was the obvious one. The
device formed the frequency step from a float32 exponential where the host used
a float64 one, about 1e-7 apart, and the angle multiplies that by the position,
so the two drifted apart as a sequence got longer: inside 4e-6 of peak at
position 137 and 1.5e-4 at position 4096. A tolerance wide enough to cover
position 4096 would have been a tolerance that covered a real fault at every
position, and it would have gone on being not quite wide enough as contexts got
longer.

`molla.nn.rope.step_table` is the fix, and it is why the position 4096 case
below now reports the same figure as the position 137 one. Both sides take the
same float32 steps, do the same three operations in the same order, and differ
only in that the host takes its `cos` and `sin` in float64.
"""


def _abs(x: Float32) -> Float32:
    return -x if x < 0 else x


def _wave(n: Int, seed: Int) -> Buffer:
    var b = Buffer(n)
    for i in range(n):
        var t = Float32((i * 37 + seed * 11) % 197) / Float32(197)
        b.data[i] = (
            (t - Float32(0.5))
            * Float32(4.0)
            * (Float32(1.0) + Float32((i + seed) % 7))
        )
    return b^


def _report(
    mut bad: Int, name: String, got: Buffer, want: Buffer, gate: Float32
):
    """Print how far apart two answers are, and count it if it is too far.

    Relative to the peak magnitude of the reference rather than per element, for
    the reason `kernel_oracle` gives: half of these produce values that pass
    through zero, and a relative error at an element that is nearly zero is a
    number with no meaning in it.
    """
    var peak = Float32(0)
    for i in range(want.elements()):
        var m = _abs(want.data[i])
        if m > peak:
            peak = m
    var worst = Float32(0)
    for i in range(want.elements()):
        var gap = _abs(got.data[i] - want.data[i])
        if gap > worst:
            worst = gap
    var relative = worst / peak if peak > 0 else Float32(0)

    var mark = "ok  " if relative <= gate else "FAIL"
    print(
        mark,
        name,
        " peak",
        peak,
        " worst",
        worst,
        "(" + String(relative) + " of peak)",
    )
    if relative > gate:
        bad += 1


def _report_abs(
    mut bad: Int, name: String, got: Buffer, want: Buffer, gate: Float32
):
    """The same, measured absolutely, which is what a softmax wants.

    Everything else here is measured against the peak magnitude of the
    reference, and for a softmax that is the wrong question in a way that gets
    the answer backwards. A probability vector over two thousand entries has a
    peak near a thousandth, so dividing by it inflates a difference that is
    already at the last bit of a float32, and it inflates it most for the
    flattest distributions, which are the easiest case rather than the hardest.
    A probability is its own scale and the honest measure of one is how far it
    moved.
    """
    var worst = Float32(0)
    for i in range(want.elements()):
        var gap = _abs(got.data[i] - want.data[i])
        if gap > worst:
            worst = gap
    var mark = "ok  " if worst <= gate else "FAIL"
    print(mark, name, " worst", worst, "(absolute)")
    if worst > gate:
        bad += 1


def _norm(ctx: DeviceContext, mut bad: Int) raises:
    for n in [256, 512, 4096]:
        var x = _wave(n, 1)
        var gain = _wave(n, 5)
        for i in range(n):
            gain.data[i] = Float32(1.0) + gain.data[i] * Float32(0.1)

        var want = Buffer(n)
        rms_norm(
            want,
            x,
            Tensor(Int(gain.data.unsafe_ptr()), Q_F32, n, 1),
            Float32(1e-5),
        )

        var dx = DeviceVec(ctx, n)
        var dg = DeviceVec(ctx, n)
        var dout = DeviceVec(ctx, n)
        dx.upload(x)
        dg.upload(gain)
        device_rms_norm(ctx, dx, dg, dout, Float32(1e-5))
        ctx.synchronize()
        var got = Buffer(n)
        dout.download(got)
        _report(bad, "rms_norm " + String(n), got, want, Float32(WIDE))


def _softmax(ctx: DeviceContext, mut bad: Int) raises:
    for n in [64, 300, 2048]:
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
        _report_abs(bad, "softmax " + String(n), got, want, Float32(TIGHT))


def _activations(ctx: DeviceContext, mut bad: Int) raises:
    var n = 4096
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
    _report(bad, "swiglu", got, want, Float32(TIGHT))

    var want_geglu = Buffer(n)
    for i in range(n):
        want_geglu.data[i] = gelu(gate.data[i]) * up.data[i]
    var dgg = DeviceVec(ctx, n)
    dgg.upload(gate)
    device_geglu(ctx, dgg, dup)
    ctx.synchronize()
    var got_geglu = Buffer(n)
    dgg.download(got_geglu)
    _report(bad, "geglu", got_geglu, want_geglu, Float32(TIGHT))

    var want_silu = Buffer(n)
    var want_gelu = Buffer(n)
    for i in range(n):
        want_silu.data[i] = silu(gate.data[i])
        want_gelu.data[i] = gelu(gate.data[i])
    var ds = DeviceVec(ctx, n)
    ds.upload(gate)
    device_silu(ctx, ds)
    var dg2 = DeviceVec(ctx, n)
    dg2.upload(gate)
    device_gelu(ctx, dg2)
    ctx.synchronize()
    var got_silu = Buffer(n)
    var got_gelu = Buffer(n)
    ds.download(got_silu)
    dg2.download(got_gelu)
    _report(bad, "silu", got_silu, want_silu, Float32(TIGHT))
    _report(bad, "gelu", got_gelu, want_gelu, Float32(TIGHT))


def _elementwise(ctx: DeviceContext, mut bad: Int) raises:
    # Not a multiple of the tile width, so the tail of the strided walk is
    # exercised rather than assumed.
    var n = 4093
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
    _report(bad, "add_into", got, want, Float32(0))

    var want_scaled = Buffer(n)
    want_scaled.copy_from(x)
    scale_into(want_scaled, Float32(0.125))
    var dscale = DeviceVec(ctx, n)
    dscale.upload(x)
    device_scale_into(ctx, dscale, Float32(0.125))
    ctx.synchronize()
    var got_scaled = Buffer(n)
    dscale.download(got_scaled)
    _report(bad, "scale_into", got_scaled, want_scaled, Float32(0))


def _argmax(ctx: DeviceContext, mut bad: Int) raises:
    """Reported as an index match rather than a distance.

    An argmax has no tolerance to be inside. It is the same index or it is a
    different token, and half a token is not a thing.
    """
    var n = 128256
    var x = _wave(n, 9)
    x.data[99991] = Float32(1000.0)

    var dx = DeviceVec(ctx, n)
    var dout = DeviceVec(ctx, 1)
    dx.upload(x)
    device_argmax(ctx, dx, dout, n)
    ctx.synchronize()
    var host = argmax(x.data, 0, n)
    var device = Int(dout.at(0))
    var mark = "ok  " if host == device else "FAIL"
    print(mark, "argmax over a vocabulary  host", host, " device", device)
    if host != device:
        bad += 1


def _rope(ctx: DeviceContext, mut bad: Int) raises:
    var heads = 8
    var head_dim = 128

    var llama3 = RopeSpec(head_dim, Float32(500000.0))
    _rope_case(ctx, bad, llama3, heads, head_dim, 137, "rope llama 3")
    _rope_case(ctx, bad, llama3, heads, head_dim, 4096, "rope llama 3 far out")

    var adjacent = RopeSpec(head_dim, Float32(10000.0))
    adjacent.neox = False
    _rope_case(ctx, bad, adjacent, heads, head_dim, 41, "rope adjacent pairs")

    var yarn = RopeSpec.yarn(
        head_dim, Float32(10000.0), Float32(4.0), 2048, True
    )
    _rope_case(ctx, bad, yarn, heads, head_dim, 611, "rope yarn")

    # The two bases a Gemma 3 layer alternates between, which is the case where
    # getting the spec per layer rather than per model matters.
    var local = RopeSpec(head_dim, Float32(10000.0))
    _rope_case(ctx, bad, local, heads, head_dim, 300, "rope gemma 3 local")
    var glob = RopeSpec(head_dim, Float32(1000000.0))
    _rope_case(ctx, bad, glob, heads, head_dim, 300, "rope gemma 3 global")


def _rope_case(
    ctx: DeviceContext,
    mut bad: Int,
    spec: RopeSpec,
    heads: Int,
    head_dim: Int,
    pos: Int,
    name: String,
) raises:
    var n = heads * head_dim
    var x = _wave(n, pos + 1)
    var want = Buffer(n)
    want.copy_from(x)
    var none = List[Float32]()
    rotate_heads(spec, want, heads, head_dim, pos, none, False)

    var dx = DeviceVec(ctx, n)
    dx.upload(x)
    var tables = RopeTables(ctx, spec, none, False)
    device_rope(ctx, spec, dx, 0, heads, head_dim, pos, tables)
    ctx.synchronize()
    var got = Buffer(n)
    dx.download(got)
    _report(bad, name, got, want, Float32(ANGLED))


def _attention(ctx: DeviceContext, mut bad: Int) raises:
    var gqa = AttnSpec(32, 8, 128)
    _attend_case(
        ctx, bad, gqa, 512, 511, "attention grouped query", Float32(WIDE)
    )

    var mha = AttnSpec(8, 8, 64)
    _attend_case(ctx, bad, mha, 128, 127, "attention multi head", Float32(WIDE))

    var windowed = AttnSpec(8, 2, 64)
    windowed.window = 128
    _attend_case(
        ctx, bad, windowed, 512, 511, "attention sliding window", Float32(WIDE)
    )

    var sinks = AttnSpec(8, 2, 64)
    sinks.window = 128
    sinks.sinks = 4
    _attend_case(
        ctx, bad, sinks, 512, 511, "attention window with sinks", Float32(WIDE)
    )

    var capped = AttnSpec(8, 2, 64)
    capped.softcap = Float32(50.0)
    _attend_case(
        ctx, bad, capped, 256, 255, "attention softcapped", Float32(CAPPED)
    )


def _attend_case(
    ctx: DeviceContext,
    mut bad: Int,
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
    _report(bad, name, got, want, gate)


def main():
    comptime if not has_accelerator():
        print(
            "this build has no device code in it, so there are no device"
            " operations to check. Build on a machine with a GPU"
        )
        exit(2)
    else:
        try:
            var ctx = DeviceContext()
            print(
                "built for",
                build_target_arch(),
                "running on",
                ctx.api(),
                ctx.name(),
            )
            print(
                "tolerances",
                TIGHT,
                "elementwise,",
                WIDE,
                "reductions,",
                ANGLED,
                "rope,",
                CAPPED,
                "softcap, all as a fraction of the peak magnitude",
            )
            var bad = 0
            _norm(ctx, bad)
            _softmax(ctx, bad)
            _activations(ctx, bad)
            _elementwise(ctx, bad)
            _argmax(ctx, bad)
            _rope(ctx, bad)
            _attention(ctx, bad)
            if bad > 0:
                print(String(bad) + " operations are outside tolerance")
                exit(1)
            print("every operation agrees with the host inside tolerance")
        except e:
            print("block_oracle:", e)
            exit(1)
