"""Rotary position embedding, and the four ways people have stretched it.

Rope encodes a token's position by rotating pairs of numbers inside each head
by an angle proportional to the position. Pair `i` turns at `base ** (-2i/d)`
radians per step, so the first pair goes round every few tokens and the last
takes longer than any context anyone has trained on. An attention score between
two positions then depends on the angle between them, which is their distance,
and nothing else. That is the whole idea and it is one line of trigonometry.

Everything after that is people trying to run a model past the context it was
trained on. There are four schemes in the wild and they are not four
implementations:

Nothing. `scale` is one and `ext_factor` is zero. Every model runs this at or
below its trained length.

Linear, sometimes called position interpolation. Divide every position by the
factor, so a 8192 position lands where 4096 used to. `scale` is `1 / factor`
and nothing else changes. It squeezes the high frequency pairs into angles the
model never saw during training, which is why it degrades on short range detail.

NTK aware. Raise the base instead of scaling the position, so the low frequency
pairs stretch and the high frequency ones stay where they were. This needs no
code at all: it is `base * factor ** (d / (d - 2))` computed once when the spec
is built, and from there it is an ordinary rope. It is here as a constructor and
not as a branch, which is the honest shape of it.

YaRN. Do the interpolation, but only on the pairs that need it. A pair whose
wavelength is shorter than the trained context has seen every angle already and
is left alone; a pair whose wavelength is longer than the trained context has
never completed a turn and is interpolated fully; the ones in between are
blended across a ramp. `beta_fast` and `beta_slow` are where that ramp starts
and ends, measured in whole turns. It also warms the attention logits by
`1 + 0.1 * ln(1 / scale)`, which is not a position thing at all, it is a
correction for the entropy that interpolation takes out of the softmax.

Llama 3.1 is a fifth in the papers and not a fifth here. Its scheme is per pair
frequency factors, and llama.cpp computes them at conversion time and writes
them into the file as `rope_freqs.weight`. So the file arrives with the answer
in it and `rotate` divides by it. There is nothing to implement and that is
worth knowing rather than rediscovering.

The pairing differs by architecture and it is not a detail. Llama pairs adjacent
elements, `(0,1)`, `(2,3)`, and so on, because the conversion script permutes
the query and key weights so that layout works. Qwen and most others pair an
element with the one `d/2` along. Same rotation, different memory order, and
picking the wrong one gives a model that is fluent for a few tokens and then is
not.
"""

from std.math import ceil, cos, exp, floor, log, pi, sin

from molla.nn.tensor import Buffer


struct RopeSpec(Copyable, ImplicitlyCopyable, Movable):
    """Everything needed to rotate one head at one position."""

    var dim: Int
    """How many elements at the front of the head get rotated. Usually the whole
    head. A model with partial rotary leaves the tail alone, and the tail is
    copied rather than zeroed, because those elements carry content."""

    var base: Float32
    """The theta base. 10000 originally, 500000 for Llama 3, 1000000 for Qwen 3.
    NTK aware scaling is a change to this number and nothing else."""

    var scale: Float32
    """What the position is multiplied by. One for no scaling, `1 / factor` for
    linear interpolation and for the interpolated end of a YaRN ramp."""

    var neox: Bool
    """True pairs element `i` with element `i + dim/2`, which is what Qwen and
    most others want. False pairs adjacent elements, which is what a converted
    Llama wants because its weights were permuted to suit."""

    var ext_factor: Float32
    """How much YaRN. Zero is off and the ramp is not computed at all. One is
    the full scheme. Between the two is a partial blend, which nobody ships but
    ggml allows and so does this."""

    var attn_factor: Float32
    """A flat multiplier on the sine and cosine, which scales the attention
    logits. One unless a model asks otherwise. YaRN multiplies its own
    correction into this rather than replacing it."""

    var beta_fast: Float32
    """Where the YaRN ramp starts, in whole turns of a pair over the trained
    context. 32 in the paper. Pairs faster than this are left alone."""

    var beta_slow: Float32
    """Where the ramp ends. 1 in the paper. Pairs slower than this are
    interpolated all the way."""

    var orig_context: Int
    """The context the model was trained on, which is what a wavelength gets
    compared against. Only read when `ext_factor` is not zero."""

    def __init__(out self, dim: Int, base: Float32):
        """A rope that does nothing but rotate, in neox pairing."""
        self.dim = dim
        self.base = base
        self.scale = 1.0
        self.neox = True
        self.ext_factor = 0.0
        self.attn_factor = 1.0
        self.beta_fast = 32.0
        self.beta_slow = 1.0
        self.orig_context = 0

    @staticmethod
    def linear(
        dim: Int, base: Float32, factor: Float32, neox: Bool
    ) raises -> Self:
        """Position interpolation. A factor of one is no scaling."""
        if factor <= 0:
            raise Error("a rope scaling factor has to be positive")
        var out = Self(dim, base)
        out.neox = neox
        out.scale = 1.0 / factor
        return out

    @staticmethod
    def ntk(
        dim: Int, base: Float32, factor: Float32, neox: Bool
    ) raises -> Self:
        """NTK aware scaling, which is a different base and nothing else.

        `base * factor ** (dim / (dim - 2))`. There is no branch anywhere else
        in this file for it, because after the base is chosen it is an ordinary
        rope, and pretending otherwise would put a mode flag in the hot loop for
        a computation that happens once.
        """
        if factor <= 0:
            raise Error("a rope scaling factor has to be positive")
        if dim <= 2:
            raise Error("ntk scaling needs a rotary dimension above two")
        var power = Float32(dim) / Float32(dim - 2)
        var scaled = base * _powf(factor, power)
        var out = Self(dim, scaled)
        out.neox = neox
        return out

    @staticmethod
    def yarn(
        dim: Int,
        base: Float32,
        factor: Float32,
        orig_context: Int,
        neox: Bool,
    ) raises -> Self:
        """YaRN with the paper's ramp, 32 turns to 1 turn.

        The attention correction is folded in here rather than at rotate time,
        since it depends only on the scale and computing `log` once per head per
        token to get the same number back is waste.
        """
        if factor <= 0:
            raise Error("a rope scaling factor has to be positive")
        if orig_context <= 0:
            raise Error("yarn needs the context the model was trained on")
        var out = Self(dim, base)
        out.neox = neox
        out.scale = 1.0 / factor
        out.ext_factor = 1.0
        out.orig_context = orig_context
        out.attn_factor = 1.0 + 0.1 * Float32(log(Float64(factor)))
        return out


def _powf(x: Float32, y: Float32) -> Float32:
    if x <= 0:
        return 0.0
    return Float32(exp(Float64(y) * log(Float64(x))))


def corr_dim(spec: RopeSpec, turns: Float32) -> Float32:
    """Which pair index completes `turns` whole rotations over the trained
    context.

    Inverting `wavelength(i) = 2 pi base ** (2i/d)` against
    `orig_context / turns` and solving for `i`. It comes out as a real number
    and is not rounded here, because the two ends of the ramp round in opposite
    directions and doing it at the call site keeps that visible.
    """
    var top = Float32(spec.orig_context) / (turns * 2.0 * Float32(pi))
    return (
        Float32(spec.dim)
        * Float32(log(Float64(top)))
        / (2.0 * Float32(log(Float64(spec.base))))
    )


def corr_range(spec: RopeSpec) -> Tuple[Float32, Float32]:
    """The two ends of the YaRN ramp, clamped into the head.

    The low end floors and the high end ceils, so the ramp covers at least the
    pairs it was meant to and a degenerate spec where the two ends meet still
    leaves a ramp one pair wide rather than a division by zero.
    """
    var low = Float32(floor(Float64(corr_dim(spec, spec.beta_fast))))
    var high = Float32(ceil(Float64(corr_dim(spec, spec.beta_slow))))
    if low < 0:
        low = 0.0
    var top = Float32(spec.dim - 1)
    if high > top:
        high = top
    return (low, high)


def _ramp(low: Float32, high: Float32, pair: Int) -> Float32:
    """One minus the position along the ramp, clamped to zero and one.

    The denominator has a floor on it because a spec whose two ends land on the
    same pair is legal and would otherwise divide by zero. `0.001` is ggml's
    number and it is kept rather than improved on, since the point is to agree
    with the reference and not to be tidier than it.
    """
    var span = high - low
    if span < 0.001:
        span = 0.001
    var at = (Float32(pair) - low) / span
    if at < 0:
        at = 0.0
    if at > 1:
        at = 1.0
    return 1.0 - at


def angle(
    spec: RopeSpec, pos: Int, pair: Int, freq: Float32
) raises -> Tuple[Float32, Float32]:
    """The cosine and sine for one pair at one position, with the multiplier
    already folded in.

    `freq` is the per pair frequency factor a Llama 3.1 file carries in
    `rope_freqs.weight`. Pass one when there is no such tensor.
    """
    if pair < 0 or pair * 2 >= spec.dim:
        raise Error(
            "pair "
            + String(pair)
            + " is outside a rotary dimension of "
            + String(spec.dim)
        )
    if freq == 0:
        raise Error("a rope frequency factor of zero has no angle")

    # The angle this pair would turn through with no scaling at all.
    var step = _powf(spec.base, -2.0 * Float32(pair) / Float32(spec.dim))
    var extrap = Float32(pos) * step / freq
    var interp = spec.scale * extrap

    var theta = interp
    if spec.ext_factor != 0:
        var ends = corr_range(spec)
        var mix = _ramp(ends[0], ends[1], pair) * spec.ext_factor
        theta = interp * (1.0 - mix) + extrap * mix
    return (
        Float32(cos(Float64(theta))) * spec.attn_factor,
        Float32(sin(Float64(theta))) * spec.attn_factor,
    )


def rotate(spec: RopeSpec, mut x: List[Float32], at: Int, pos: Int) raises:
    """Rotate one head in place with no frequency factors."""
    var ones = List[Float32]()
    rotate_scaled(spec, x, at, pos, ones, False)


def rotate_scaled(
    spec: RopeSpec,
    mut x: List[Float32],
    at: Int,
    pos: Int,
    factors: List[Float32],
    use_factors: Bool,
) raises:
    """Rotate one head in place, dividing each pair's angle by its factor.

    `factors` is `rope_freqs.weight`, one value per pair, which is how a Llama
    3.1 file carries its scaling. Elements past `spec.dim` are not touched,
    which is what partial rotary means: they carry content and zeroing them
    would throw it away.
    """
    if spec.dim % 2 != 0:
        raise Error("a rotary dimension has to be even")
    var pairs = spec.dim // 2
    if use_factors and len(factors) < pairs:
        raise Error(
            "rope wants "
            + String(pairs)
            + " frequency factors but got "
            + String(len(factors))
        )

    for pair in range(pairs):
        var freq = factors[pair] if use_factors else Float32(1.0)
        var turn = angle(spec, pos, pair, freq)
        var c = turn[0]
        var s = turn[1]

        var lo: Int
        var hi: Int
        if spec.neox:
            lo = at + pair
            hi = at + pair + pairs
        else:
            lo = at + pair * 2
            hi = lo + 1

        var a = x[lo]
        var b = x[hi]
        x[lo] = a * c - b * s
        x[hi] = a * s + b * c


def rotate_heads(
    spec: RopeSpec,
    mut x: Buffer,
    heads: Int,
    head_dim: Int,
    pos: Int,
    factors: List[Float32],
    use_factors: Bool,
) raises:
    """Rotate a whole row of heads that sit end to end in one buffer.

    Which is how q and k come out of their projections, so this is what a block
    actually calls. Every head at the same position gets the same angles, and
    they are recomputed per head rather than cached, which is a `cos` and a
    `sin` per pair per head and is not where the time goes.
    """
    if head_dim < spec.dim:
        raise Error(
            "a rotary dimension of "
            + String(spec.dim)
            + " does not fit in a head of "
            + String(head_dim)
        )
    if x.elements() < heads * head_dim:
        raise Error(
            "rope wants "
            + String(heads * head_dim)
            + " values but the buffer holds "
            + String(x.elements())
        )
    for h in range(heads):
        rotate_scaled(spec, x.data, h * head_dim, pos, factors, use_factors)
