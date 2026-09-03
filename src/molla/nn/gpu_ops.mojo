"""Everything in a transformer block that is not the matvec, on the device.

Individually these are tiny. A norm over 4096 floats and a residual add over
4096 floats are nothing next to a 4096 by 14336 matvec, and on a host that is
the end of the argument. On a device it is not, because the thing that costs is
not the arithmetic, it is leaving. A round trip between two kernels is a
synchronize, a copy out, a copy back and a launch, and at decode shapes that is
more than every small operation in a block put together. So they go across as a
set or the matvec they surround gains nothing, which is what #142 is for.

Same rule as the matvec in `molla.nn.gpu`. One source per operation, compiled
for Metal and for CUDA, with the divergence in compile time parameters rather
than in a second file. A host reference already exists for every one of these in
`molla.nn.kernel`, `molla.nn.rope` and `molla.nn.attention`, and
`scripts/block_oracle.mojo` runs both and prints the distance.

Two places where the device answer is not the host answer bit for bit, both
recorded in `docs/validation/kernels.md` rather than left to be discovered:

The host accumulates a sum of squares in float64 and this cannot, because Metal
has no float64 at all. A tree reduction in float32 is what replaces it, and over
a few thousand terms it is closer to the float64 answer than a sequential
float32 sum would be, because the error grows with the depth of the association
and a tree is logarithmic where a loop is linear.

The host computes rope angles through float64 `cos` and `sin` and this uses the
float32 ones. That is a difference of a few times 1e-7 in an angle, which is
what the tolerance in the validation document is stated to allow.

Nothing here allocates. Every entry point takes device vectors that already
exist and queues a launch without synchronizing, so a block is a couple of dozen
of these on one stream and the caller waits once at the end.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import cos, exp, sin, sqrt
from std.memory import AddressSpace, stack_allocation
from std.sys.info import has_accelerator

from max.gpu import barrier
from max.gpu.host import DeviceContext

from molla.nn.attention import AttnSpec
from molla.nn.gpu import TILE, DeviceVec
from molla.nn.rope import RopeSpec, corr_range, step_table

comptime NEG_INF = Float32(-3.4028234663852886e38)
"""What a masked key scores.

The lowest finite float32 rather than an actual infinity. The host packs the
visible keys together and never exponentiates a masked one, and this cannot,
because packing is a sequential operation and there are `count` threads. So the
masked ones stay in place and are exponentiated to zero, which needs the
subtraction `score - max` not to be a nan, and `-inf` minus a finite maximum is
`-inf` while `-inf` minus `-inf` is a nan. Using the lowest finite value instead
means the arithmetic is ordinary everywhere and the exponential still underflows
to exactly zero.
"""


def _block_sum[tile: Int](value: Float32) -> Float32:
    """Sum one value per thread across the block, answer in every thread.

    A tree over shared memory rather than a warp shuffle. A shuffle would be
    faster and the warp width is 32 on one vendor and 32 on the other today,
    which is exactly the sort of thing that is true until it is not, and D7 says
    a target specific constant belongs in a launch parameter and not in the body
    of a function.
    """
    var part = stack_allocation[
        tile, Float32, address_space=AddressSpace.SHARED
    ]()
    var t = Int(thread_idx.x)
    part[unsafe_offset=t] = value
    barrier()
    var step = tile // 2
    while step > 0:
        if t < step:
            part[unsafe_offset=t] = (
                part[unsafe_offset=t] + part[unsafe_offset=t + step]
            )
        barrier()
        step //= 2
    var total = part[unsafe_offset=0]
    # Every thread reads slot zero before anything is allowed to overwrite it,
    # which matters because these run back to back inside one kernel and the
    # allocation is the same shared memory each time.
    barrier()
    return total


def _block_max[tile: Int](value: Float32) -> Float32:
    """The largest value in the block, in every thread."""
    var part = stack_allocation[
        tile, Float32, address_space=AddressSpace.SHARED
    ]()
    var t = Int(thread_idx.x)
    part[unsafe_offset=t] = value
    barrier()
    var step = tile // 2
    while step > 0:
        if t < step:
            var other = part[unsafe_offset=t + step]
            if other > part[unsafe_offset=t]:
                part[unsafe_offset=t] = other
        barrier()
        step //= 2
    var top = part[unsafe_offset=0]
    barrier()
    return top


def rms_norm_kernel[
    tile: Int
](
    x: Pointer[Float32, MutAnyOrigin],
    g: Pointer[Float32, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    n_dev: Int32,
    eps: Float32,
):
    """`o[i] = x[i] * rsqrt(mean(x*x) + eps) * g[i]`, one block over the row.

    One block and not one per element because the scale is a reduction over the
    whole row, so a second kernel to apply it would mean writing the sum to
    memory and reading it back. At 4096 wide with 128 threads that is 32
    elements each, which is enough work to be worth a launch and little enough
    that the reduction is most of the time.
    """
    var n = Int(n_dev)
    var t = Int(thread_idx.x)

    var acc = Float32(0)
    var i = t
    while i < n:
        var v = x[unsafe_offset=i]
        acc += v * v
        i += tile
    var total = _block_sum[tile](acc)

    var scale = Float32(1.0) / sqrt(total / Float32(n) + eps)
    i = t
    while i < n:
        o[unsafe_offset=i] = x[unsafe_offset=i] * scale * g[unsafe_offset=i]
        i += tile


def softmax_kernel[tile: Int](x: Pointer[Float32, MutAnyOrigin], n_dev: Int32):
    """In place over a run, with the maximum subtracted first.

    Subtracting the maximum is not an optimisation here any more than it is on
    the host. An attention score of 100 is ordinary and `exp(100)` is two thirds
    of the way to a float32 infinity, so a row with two of them sums to infinity
    and every probability comes back as a nan.
    """
    var n = Int(n_dev)
    var t = Int(thread_idx.x)

    var mine = NEG_INF
    var i = t
    while i < n:
        var v = x[unsafe_offset=i]
        if v > mine:
            mine = v
        i += tile
    var top = _block_max[tile](mine)

    var acc = Float32(0)
    i = t
    while i < n:
        var e = exp(x[unsafe_offset=i] - top)
        x[unsafe_offset=i] = e
        acc += e
        i += tile
    var total = _block_sum[tile](acc)

    var inv = Float32(1.0) / total
    i = t
    while i < n:
        x[unsafe_offset=i] = x[unsafe_offset=i] * inv
        i += tile


comptime ACT_SILU = 0
comptime ACT_GELU = 1


def _activate[kind: Int](v: Float32) -> Float32:
    """The gate function, chosen at compile time.

    A parameter rather than a branch because it is the same value for every
    element of every layer of a model, so a runtime test would be a predictable
    branch executed fourteen thousand times per layer to reach the same side.
    """
    comptime if kind == ACT_SILU:
        return v / (Float32(1.0) + exp(-v))
    else:
        # The tanh approximation, which is the one the weights were trained
        # with. The exact form through the error function is a different
        # function by about a thousandth, and a model trained against one and
        # run against the other is a small consistent bias nobody ever finds.
        var c = Float32(0.7978845608028654)
        var inner = c * (v + Float32(0.044715) * v * v * v)
        var e = Float32(2.0) / (Float32(1.0) + exp(Float32(-2.0) * inner))
        return Float32(0.5) * v * (Float32(1.0) + (e - Float32(1.0)))


def swiglu_kernel[
    kind: Int
](
    gate: Pointer[Float32, MutAnyOrigin],
    up: Pointer[Float32, MutAnyOrigin],
    n_dev: Int32,
):
    """`gate = act(gate) * up`, in place on the gate.

    In place for the reason the host version is in place: the two halves of a
    gated MLP are as wide as the hidden dimension, 14336 floats on an 8B, and a
    third buffer for a result consumed immediately is 56 KB of traffic per layer
    for nothing.
    """
    var n = Int(n_dev)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    while i < n:
        gate[unsafe_offset=i] = (
            _activate[kind](gate[unsafe_offset=i]) * up[unsafe_offset=i]
        )
        i += stride


def act_kernel[kind: Int](x: Pointer[Float32, MutAnyOrigin], n_dev: Int32):
    """The gate function alone, for a model with no up projection to fold in."""
    var n = Int(n_dev)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    while i < n:
        x[unsafe_offset=i] = _activate[kind](x[unsafe_offset=i])
        i += stride


def add_into_kernel(
    acc: Pointer[Float32, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    n_dev: Int32,
):
    """The residual add, which is the only reason a deep network trains."""
    var n = Int(n_dev)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    while i < n:
        acc[unsafe_offset=i] = acc[unsafe_offset=i] + x[unsafe_offset=i]
        i += stride


def scale_into_kernel(
    x: Pointer[Float32, MutAnyOrigin], n_dev: Int32, by: Float32
):
    var n = Int(n_dev)
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    var stride = Int(grid_dim.x * block_dim.x)
    while i < n:
        x[unsafe_offset=i] = x[unsafe_offset=i] * by
        i += stride


def argmax_kernel[
    tile: Int
](
    x: Pointer[Float32, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    n_dev: Int32,
):
    """The index of the largest value, written as a float in `o[0]`.

    A float because the whole device side of a block deals in float32 vectors
    and one integer output would mean a second buffer type carried everywhere
    for one value. A vocabulary is at most a few hundred thousand, float32
    represents every integer up to sixteen million exactly, so nothing is lost.

    Ties go to the lower index, which is not arbitrary here: the host `argmax`
    does the same, and a device that broke ties the other way would disagree
    with it on a logit row that has an exact tie, which happens more often than
    it sounds like it should once a row has been through a float16 weight. That
    is why the reduction compares indices when the values are equal instead of
    taking whichever half it looked at first.
    """
    var n = Int(n_dev)
    var t = Int(thread_idx.x)

    var best = 0
    var top = NEG_INF
    var i = t
    while i < n:
        var v = x[unsafe_offset=i]
        if v > top:
            top = v
            best = i
        i += tile
    if t >= n:
        best = n

    var vals = stack_allocation[
        tile, Float32, address_space=AddressSpace.SHARED
    ]()
    var idxs = stack_allocation[
        tile, Int32, address_space=AddressSpace.SHARED
    ]()
    vals[unsafe_offset=t] = top
    idxs[unsafe_offset=t] = Int32(best)
    barrier()
    var step = tile // 2
    while step > 0:
        if t < step:
            var mine = vals[unsafe_offset=t]
            var other = vals[unsafe_offset=t + step]
            var take = other > mine
            if other == mine:
                take = idxs[unsafe_offset=t + step] < idxs[unsafe_offset=t]
            if take:
                vals[unsafe_offset=t] = other
                idxs[unsafe_offset=t] = idxs[unsafe_offset=t + step]
        barrier()
        step //= 2
    if t == 0:
        o[unsafe_offset=0] = Float32(Int(idxs[unsafe_offset=0]))


def rope_kernel[
    neox: Bool, with_factors: Bool
](
    x: Pointer[Float32, MutAnyOrigin],
    steps: Pointer[Float32, MutAnyOrigin],
    factors: Pointer[Float32, MutAnyOrigin],
    at_dev: Int32,
    head_dim_dev: Int32,
    dim_dev: Int32,
    pos_dev: Int32,
    scale: Float32,
    ext_factor: Float32,
    attn_factor: Float32,
    low: Float32,
    high: Float32,
):
    """Rotate a row of heads that sit end to end, one block per head.

    The angles are recomputed per head rather than shared, which is what the
    host version does too and for the same reason: every head at a position gets
    the same angle, and a table would be a second buffer and a second launch to
    fill it for a `cos` and a `sin` per pair.

    Three of the arguments are values the host worked out rather than the raw
    spec, and they are here for accuracy rather than for speed. `steps` is the
    frequency step per pair and `low` and `high` are the YaRN correction range.
    None of the three depends on the position or the head, and all three are
    formed on the host through float64 `log`, `exp` and `pow`, which is where
    the agreement with the host reference comes from. Forming the step here from
    a float32 exponential instead costs about 1e-7 in relative terms, the angle
    multiplies that by the position, and by position 4096 it is a disagreement
    in the fourth digit of a rotated value. `molla.nn.rope.step_table` is the
    other end of this.

    What is left in the kernel is the same three operations the host does in the
    same order, so the two agree to the accuracy of `cos` and `sin` at any
    position rather than to something that decays along a sequence.

    The pairing is a compile time parameter. Neox pairs element `i` with element
    `i + dim/2`, which is what Qwen and most others want, and the other pairs
    adjacent elements, which is what a converted Llama wants because its weights
    were permuted to suit. Elements past `dim` are not touched, which is what
    partial rotary means: they carry content and zeroing them throws it away.
    """
    var head_dim = Int(head_dim_dev)
    var dim = Int(dim_dev)
    var pos = Int(pos_dev)
    var pairs = dim // 2
    var head = Int(block_idx.x)
    var at = Int(at_dev) + head * head_dim

    var pair = Int(thread_idx.x)
    while pair < pairs:
        var freq = Float32(1.0)
        comptime if with_factors:
            freq = factors[unsafe_offset=pair]

        var step = steps[unsafe_offset=pair]
        var extrap = Float32(pos) * step / freq
        var interp = scale * extrap
        var theta = interp
        if ext_factor != 0:
            var mix = _ramp(low, high, pair) * ext_factor
            theta = interp * (Float32(1.0) - mix) + extrap * mix
        var turn = _reduce_angle(theta)
        var c = cos(turn) * attn_factor
        var s = sin(turn) * attn_factor

        var lo = at + pair
        var hi = at + pair + pairs
        comptime if not neox:
            lo = at + pair * 2
            hi = lo + 1

        var a = x[unsafe_offset=lo]
        var b = x[unsafe_offset=hi]
        x[unsafe_offset=lo] = a * c - b * s
        x[unsafe_offset=hi] = a * s + b * c
        pair += Int(block_dim.x)


comptime TWO_PI_HI = Float32(6.28125)
"""Two pi, truncated to eight significant bits.

Eight because the rest of the split below only works if `k * TWO_PI_HI` is exact
in float32, and a float32 has 24 bits of mantissa, so eight here leaves sixteen
for `k`. That covers an angle of up to 65536 turns, which is a position of about
411000 at the fastest pair, and no context is near that.
"""

comptime TWO_PI_MID = Float32(0.0019353072)
"""What is left of two pi after the first term."""

comptime TWO_PI_LO = Float32(1.0253132e-11)
"""And after the second. Three terms reproduce two pi to about 1e-18, which is
far past float32 and is the point: the error in the reduction has to come from
the multiplications rather than from the constant."""

comptime INV_TWO_PI = Float32(0.15915494)


def _reduce_angle(theta: Float32) -> Float32:
    """Bring an angle into one turn before it reaches `cos` and `sin`.

    This is here because the two GPUs disagreed and only one of them was wrong.
    A rope angle is the position times a frequency, so at the fastest pair it is
    the position itself, and by position 4096 that is 652 whole turns. Metal
    takes a float32 `cos` of that and lands within 1e-7 of the float64 answer.
    CUDA does not: it reduced the argument in a way that costs about 4e-4 there,
    growing with the position, which is the shape of a bug that looks fine in
    every short test and degrades a long context.

    Doing the reduction here rather than trusting either one is what makes the
    two targets agree, and it is cheap. Cody and Waite's method, with two pi
    split into three terms so that the first product is exact and the other two
    carry the bits it dropped.

    The angle is never negative, so this does not handle that. `pos` is a
    position, `step` is positive by construction and the YaRN blend is between
    two positive numbers, so there is no path to one.
    """
    var k = Float32(Int(theta * INV_TWO_PI + Float32(0.5)))
    var r = theta - k * TWO_PI_HI
    r = r - k * TWO_PI_MID
    return r - k * TWO_PI_LO


def _tanh(x: Float32) -> Float32:
    """`tanh` through one exponential rather than the target's own.

    Same reason as `_reduce_angle` and found the same way. The host softcaps a
    score in float64 and multiplies the result by the cap, which is 50 on a
    Gemma 2, so an error in the tanh arrives at the softmax fifty times larger.
    The float32 `tanh` CUDA provides is far enough from the float64 one for that
    to show, and the float32 `exp` both targets provide is not.

    Written on the negative side so the exponential is of a non positive number
    and lands in `(0, 1]`. The other arrangement overflows for a score a few
    times the cap, which is not rare.
    """
    var v = x if x >= 0 else -x
    var e = exp(Float32(-2.0) * v)
    var r = (Float32(1.0) - e) / (Float32(1.0) + e)
    return r if x >= 0 else -r


def _ramp(low: Float32, high: Float32, pair: Int) -> Float32:
    """The YaRN blend for one pair, clamped to the unit interval.

    Mirrors `molla.nn.rope._ramp` term for term. The guard on a zero width range
    is what stops a model whose correction range collapses from dividing by
    zero, and it is the same guard the host has.
    """
    var span = high - low
    if span < Float32(0.001):
        span = Float32(0.001)
    var at = (Float32(pair) - low) / span
    if at < 0:
        at = 0
    if at > 1:
        at = 1
    return Float32(1.0) - at


def attend_kernel[
    tile: Int
](
    q: Pointer[Float32, MutAnyOrigin],
    keys: Pointer[Float32, MutAnyOrigin],
    values: Pointer[Float32, MutAnyOrigin],
    scores: Pointer[Float32, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    count_dev: Int32,
    pos_dev: Int32,
    head_dim_dev: Int32,
    kv_width_dev: Int32,
    group_dev: Int32,
    window_dev: Int32,
    sinks_dev: Int32,
    scale: Float32,
    softcap: Float32,
):
    """One query against `count` keys, one block per query head.

    Three phases in one kernel rather than three kernels, because the thing
    between them is a block wide reduction and a block wide reduction is
    something a block can do to itself. Splitting them would put the scores
    through device memory twice more and add two launches per head per layer.

    The masked keys stay where they are and score `NEG_INF` instead of being
    packed to the front the way the host packs them. Packing is a sequential
    operation and there are `tile` threads here, and the two agree anyway: the
    maximum is taken over the same finite values, and a masked key exponentiates
    to exactly zero and contributes nothing to the sum or to the output.

    `scores` is scratch of at least `heads * count`, passed in rather than
    allocated for the reason the host version takes it: a decode calls this once
    per layer per token.
    """
    var count = Int(count_dev)
    var pos = Int(pos_dev)
    var head_dim = Int(head_dim_dev)
    var kv_width = Int(kv_width_dev)
    var group = Int(group_dev)
    var window = Int(window_dev)
    var sinks = Int(sinks_dev)

    var h = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var kvh = h // group
    var qa = h * head_dim
    var sa = h * count

    var mine = NEG_INF
    var j = t
    while j < count:
        var visible = j < sinks or window <= 0 or j > pos - window
        var s = NEG_INF
        if visible:
            var ka = j * kv_width + kvh * head_dim
            var acc = Float32(0)
            for d in range(head_dim):
                acc += q[unsafe_offset=qa + d] * keys[unsafe_offset=ka + d]
            s = acc * scale
            if softcap > 0:
                s = softcap * _tanh(s / softcap)
        scores[unsafe_offset=sa + j] = s
        if s > mine:
            mine = s
        j += tile
    var top = _block_max[tile](mine)

    var acc_sum = Float32(0)
    j = t
    while j < count:
        var e = exp(scores[unsafe_offset=sa + j] - top)
        scores[unsafe_offset=sa + j] = e
        acc_sum += e
        j += tile
    var total = _block_sum[tile](acc_sum)
    var inv = Float32(1.0) / total

    # One thread per element of the head, each walking every key. The other way
    # round would be one thread per key accumulating into `head_dim` outputs,
    # which needs an atomic or a second reduction per element.
    var d = t
    while d < head_dim:
        var acc = Float32(0)
        for j2 in range(count):
            var va = j2 * kv_width + kvh * head_dim
            acc += scores[unsafe_offset=sa + j2] * values[unsafe_offset=va + d]
        o[unsafe_offset=qa + d] = acc * inv
        d += tile


def _grid(n: Int) -> Int:
    """Blocks for an elementwise launch, capped.

    Enough blocks to cover the vector at one element per thread, and never more
    than 256 of them. The cap is there because a residual add over 4096 floats
    would otherwise be 32 blocks doing 128 elements each and the launch would
    cost more than the work; past a few hundred blocks the loop inside the
    kernel is cheaper than the scheduling.
    """
    var blocks = (n + TILE - 1) // TILE
    if blocks < 1:
        return 1
    if blocks > 256:
        return 256
    return blocks


def device_rms_norm(
    ctx: DeviceContext,
    x: DeviceVec,
    gain: DeviceVec,
    mut out: DeviceVec,
    eps: Float32,
) raises:
    """`out = x * rsqrt(mean(x*x) + eps) * gain`.

    The gain arrives as a device vector and not as a `Tensor`, because a norm
    weight is a few thousand f32 values that are read every token of every
    layer and never change, so it is dequantized once when the model binds
    rather than per call.
    """
    _check_norm(x.elements(), gain.elements(), out.elements())
    _need_device()
    comptime if has_accelerator():
        _norm_launch(ctx, x.ptr(), gain.ptr(), out.ptr(), x.elements(), eps)


def device_rms_norm_inplace(
    ctx: DeviceContext, mut x: DeviceVec, gain: DeviceVec, eps: Float32
) raises:
    """The same norm, over the vector it read.

    Which is what Qwen wants for each head of a query and what Gemma wants for
    a sublayer output on its way to the residual add, and neither wants a second
    buffer for it. It is a second entry point rather than a note saying `x` and
    `out` may be the same vector, because passing one vector to both arguments
    of `device_rms_norm` is an aliasing error the compiler refuses to build, and
    a rule the compiler enforces beats a rule in a docstring.

    The kernel underneath is the one `device_rms_norm` launches, unchanged. It
    is safe over itself because the reduction finishes for the whole block
    before any thread writes, and after that each thread reads and writes the
    same index.
    """
    _check_norm(x.elements(), gain.elements(), x.elements())
    _need_device()
    comptime if has_accelerator():
        _norm_launch(ctx, x.ptr(), gain.ptr(), x.ptr(), x.elements(), eps)


def _check_norm(n: Int, gain: Int, written: Int) raises:
    if gain != n:
        raise Error(
            "rms_norm wants a gain of " + String(n) + " but got " + String(gain)
        )
    if written != n:
        raise Error("rms_norm wants the input and the output the same size")


def _norm_launch(
    ctx: DeviceContext,
    x: Pointer[Float32, MutAnyOrigin],
    gain: Pointer[Float32, MutAnyOrigin],
    out_ptr: Pointer[Float32, MutAnyOrigin],
    n: Int,
    eps: Float32,
) raises:
    ctx.enqueue_function[rms_norm_kernel[TILE]](
        x,
        gain,
        out_ptr,
        Int32(n),
        eps,
        grid_dim=(1, 1, 1),
        block_dim=(TILE, 1, 1),
    )


def device_softmax(ctx: DeviceContext, mut x: DeviceVec, n: Int) raises:
    """In place over the first `n` values."""
    if n <= 0 or n > x.elements():
        raise Error(
            "a softmax over "
            + String(n)
            + " does not fit in a vector of "
            + String(x.elements())
        )
    _need_device()
    comptime if has_accelerator():
        ctx.enqueue_function[softmax_kernel[TILE]](
            x.ptr(),
            Int32(n),
            grid_dim=(1, 1, 1),
            block_dim=(TILE, 1, 1),
        )


def device_swiglu(
    ctx: DeviceContext, mut gate: DeviceVec, up: DeviceVec
) raises:
    """`gate = silu(gate) * up`, in place on the gate."""
    _gated(ctx, gate, up, ACT_SILU)


def device_geglu(ctx: DeviceContext, mut gate: DeviceVec, up: DeviceVec) raises:
    """`gate = gelu(gate) * up`, which is what a Gemma MLP wants."""
    _gated(ctx, gate, up, ACT_GELU)


def _gated(
    ctx: DeviceContext, mut gate: DeviceVec, up: DeviceVec, kind: Int
) raises:
    if gate.elements() != up.elements():
        raise Error("a gated MLP wants both halves the same size")
    _need_device()
    comptime if has_accelerator():
        var n = gate.elements()
        if kind == ACT_SILU:
            ctx.enqueue_function[swiglu_kernel[ACT_SILU]](
                gate.ptr(),
                up.ptr(),
                Int32(n),
                grid_dim=(_grid(n), 1, 1),
                block_dim=(TILE, 1, 1),
            )
        else:
            ctx.enqueue_function[swiglu_kernel[ACT_GELU]](
                gate.ptr(),
                up.ptr(),
                Int32(n),
                grid_dim=(_grid(n), 1, 1),
                block_dim=(TILE, 1, 1),
            )


def device_silu(ctx: DeviceContext, mut x: DeviceVec) raises:
    _need_device()
    comptime if has_accelerator():
        var n = x.elements()
        ctx.enqueue_function[act_kernel[ACT_SILU]](
            x.ptr(),
            Int32(n),
            grid_dim=(_grid(n), 1, 1),
            block_dim=(TILE, 1, 1),
        )


def device_gelu(ctx: DeviceContext, mut x: DeviceVec) raises:
    _need_device()
    comptime if has_accelerator():
        var n = x.elements()
        ctx.enqueue_function[act_kernel[ACT_GELU]](
            x.ptr(),
            Int32(n),
            grid_dim=(_grid(n), 1, 1),
            block_dim=(TILE, 1, 1),
        )


def device_add_into(
    ctx: DeviceContext, mut acc: DeviceVec, x: DeviceVec
) raises:
    if acc.elements() != x.elements():
        raise Error("add_into wants both sides the same size")
    _need_device()
    comptime if has_accelerator():
        var n = acc.elements()
        ctx.enqueue_function[add_into_kernel](
            acc.ptr(),
            x.ptr(),
            Int32(n),
            grid_dim=(_grid(n), 1, 1),
            block_dim=(TILE, 1, 1),
        )


def device_scale_into(ctx: DeviceContext, mut x: DeviceVec, by: Float32) raises:
    _need_device()
    comptime if has_accelerator():
        var n = x.elements()
        ctx.enqueue_function[scale_into_kernel](
            x.ptr(),
            Int32(n),
            by,
            grid_dim=(_grid(n), 1, 1),
            block_dim=(TILE, 1, 1),
        )


def device_argmax(
    ctx: DeviceContext, x: DeviceVec, mut out: DeviceVec, n: Int
) raises:
    """The index of the largest of the first `n` values, into `out[0]`."""
    if n <= 0 or n > x.elements():
        raise Error(
            "an argmax over "
            + String(n)
            + " does not fit in a vector of "
            + String(x.elements())
        )
    _need_device()
    comptime if has_accelerator():
        ctx.enqueue_function[argmax_kernel[TILE]](
            x.ptr(),
            out.ptr(),
            Int32(n),
            grid_dim=(1, 1, 1),
            block_dim=(TILE, 1, 1),
        )


struct RopeTables(Movable):
    """The per pair constants a rope kernel reads, uploaded once.

    One of these per rope spec, built when a model binds and kept for as long as
    it is loaded. A model has one, or two when it alternates a local and a
    global spec the way Gemma 3 does, so this is a few hundred bytes on the
    device in exchange for never recomputing a step and never being a position
    dependent distance from the host answer.

    `factors` is `rope_freqs.weight`, one value per pair, which is how a Llama
    3.1 file carries its scaling. A model without that tensor passes an empty
    list and the kernel is compiled without the read, so nothing has to invent a
    vector of ones.
    """

    var steps: DeviceVec
    var factors: DeviceVec
    var use_factors: Bool

    def __init__(
        out self,
        ctx: DeviceContext,
        spec: RopeSpec,
        factors: List[Float32],
        use_factors: Bool,
    ) raises:
        var pairs = spec.dim // 2
        if pairs <= 0:
            raise Error("a rotary dimension has to be positive and even")
        if use_factors and len(factors) < pairs:
            raise Error(
                "rope wants "
                + String(pairs)
                + " frequency factors but got "
                + String(len(factors))
            )
        var table = step_table(spec)
        self.steps = DeviceVec(ctx, pairs)
        self.steps.upload_run(table, 0, pairs)
        # A one element vector when there are no factors, because a device
        # buffer of length zero is not a thing and the kernel never reads it.
        self.factors = DeviceVec(ctx, pairs if use_factors else 1)
        if use_factors:
            self.factors.upload_run(factors, 0, pairs)
        self.use_factors = use_factors


def device_rope(
    ctx: DeviceContext,
    spec: RopeSpec,
    mut x: DeviceVec,
    at: Int,
    heads: Int,
    head_dim: Int,
    pos: Int,
    tables: RopeTables,
) raises:
    """Rotate `heads` heads laid end to end at `at`, in place.

    Which is both of the shapes a block needs. The query comes out of its
    projection as a row of heads, and a key is rotated once where it lies in the
    cache on the way in, at `slot * kv_heads * head_dim`, rather than every time
    it is read.

    `tables` has to have been built from this same spec. That is not checked
    beyond the width, because the two things that would catch it are storing a
    copy of the spec to compare against and trusting the caller, and a spec has
    no equality yet.
    """
    if spec.dim % 2 != 0:
        raise Error("a rotary dimension has to be even")
    if head_dim < spec.dim:
        raise Error(
            "a rotary dimension of "
            + String(spec.dim)
            + " does not fit in a head of "
            + String(head_dim)
        )
    if at < 0 or x.elements() < at + heads * head_dim:
        raise Error(
            "rope wants "
            + String(heads * head_dim)
            + " values from offset "
            + String(at)
            + " but the vector ends at "
            + String(x.elements())
        )
    var pairs = spec.dim // 2
    if tables.steps.elements() < pairs:
        raise Error(
            "rope wants "
            + String(pairs)
            + " frequency steps but the tables hold "
            + String(tables.steps.elements())
        )
    if tables.use_factors and tables.factors.elements() < pairs:
        raise Error(
            "rope wants "
            + String(pairs)
            + " frequency factors but the tables hold "
            + String(tables.factors.elements())
        )
    _need_device()
    comptime if has_accelerator():
        var low = Float32(0)
        var high = Float32(0)
        if spec.ext_factor != 0:
            var ends = corr_range(spec)
            low = ends[0]
            high = ends[1]
        if spec.neox and tables.use_factors:
            _rope[True, True](
                ctx, spec, x, tables, at, heads, head_dim, pos, low, high
            )
        elif spec.neox:
            _rope[True, False](
                ctx, spec, x, tables, at, heads, head_dim, pos, low, high
            )
        elif tables.use_factors:
            _rope[False, True](
                ctx, spec, x, tables, at, heads, head_dim, pos, low, high
            )
        else:
            _rope[False, False](
                ctx, spec, x, tables, at, heads, head_dim, pos, low, high
            )


def _rope[
    neox: Bool, with_factors: Bool
](
    ctx: DeviceContext,
    spec: RopeSpec,
    mut x: DeviceVec,
    tables: RopeTables,
    at: Int,
    heads: Int,
    head_dim: Int,
    pos: Int,
    low: Float32,
    high: Float32,
) raises:
    ctx.enqueue_function[rope_kernel[neox, with_factors]](
        x.ptr(),
        tables.steps.ptr(),
        tables.factors.ptr(),
        Int32(at),
        Int32(head_dim),
        Int32(spec.dim),
        Int32(pos),
        spec.scale,
        spec.ext_factor,
        spec.attn_factor,
        low,
        high,
        grid_dim=(heads, 1, 1),
        block_dim=(TILE, 1, 1),
    )


def device_attend(
    ctx: DeviceContext,
    spec: AttnSpec,
    q: DeviceVec,
    keys: DeviceVec,
    values: DeviceVec,
    count: Int,
    pos: Int,
    mut out: DeviceVec,
    mut scores: DeviceVec,
) raises:
    """One query against `count` keys, writing `heads * head_dim` values.

    The one thing this refuses that the host version also refuses is a position
    that can see no keys at all, and it is checked here rather than in the
    kernel because a kernel cannot raise. A window and a sink count that between
    them mask everything is a configuration error rather than a numerical one,
    and it produces a division by a zero sum, which arrives as a buffer of nans
    several layers later.
    """
    var width = spec.heads * spec.head_dim
    if q.elements() < width:
        raise Error(
            "attention wants a query of "
            + String(width)
            + " but got "
            + String(q.elements())
        )
    if out.elements() < width:
        raise Error(
            "attention wants an output of "
            + String(width)
            + " but got "
            + String(out.elements())
        )
    if count <= 0:
        raise Error("attention needs at least one key to look at")
    var kv_width = spec.kv_heads * spec.head_dim
    if (
        keys.elements() < count * kv_width
        or values.elements() < count * kv_width
    ):
        raise Error(
            "attention wants "
            + String(count * kv_width)
            + " keys and values but got "
            + String(keys.elements())
            + " and "
            + String(values.elements())
        )
    if scores.elements() < spec.heads * count:
        raise Error(
            "attention wants scratch for "
            + String(spec.heads * count)
            + " scores but got "
            + String(scores.elements())
        )
    var seen = 0
    for t in range(count):
        if spec.visible(t, pos):
            seen += 1
    if seen == 0:
        raise Error(
            "position "
            + String(pos)
            + " can see no keys at all, which a window of "
            + String(spec.window)
            + " with "
            + String(spec.sinks)
            + " sinks should never produce"
        )
    _need_device()
    comptime if has_accelerator():
        ctx.enqueue_function[attend_kernel[TILE]](
            q.ptr(),
            keys.ptr(),
            values.ptr(),
            scores.ptr(),
            out.ptr(),
            Int32(count),
            Int32(pos),
            Int32(spec.head_dim),
            Int32(kv_width),
            Int32(spec.group()),
            Int32(spec.window),
            Int32(spec.sinks),
            spec.scale,
            spec.softcap,
            grid_dim=(spec.heads, 1, 1),
            block_dim=(TILE, 1, 1),
        )


def _need_device() raises:
    """The one refusal every entry point here shares.

    No host fallback, for the reason `molla.nn.gpu` gives at length: a target
    that cannot pass numerics is unsupported and says so, and a silent fallback
    is how a target stays listed as supported for a year after it stopped
    working.
    """
    comptime if not has_accelerator():
        raise Error(
            "this build has no device code in it, so there are no device"
            " kernels to run. Accelerator support is decided when molla is"
            " compiled, not when it is run"
        )
