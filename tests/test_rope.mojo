"""Rope, against values taken from the ggml formula.

The expected numbers in here were produced by `scripts/rope_ref.py`, which is
`ggml_compute_forward_rope_f32` and `rope_yarn` written out in Python with
nothing rearranged. That is a transcription of the reference and not an
independent derivation, which is the point: the question these answer is
whether molla rotates a vector the same way llama.cpp does, and a second
derivation from the paper would answer a different and less useful question,
since a model's weights were trained against the implementation rather than
against the paper.

The tolerance is 1e-5 on values of order one to nine. Both sides compute the
same expression, but one does it in float64 and rounds and the other works in
float32 throughout, so the last bit or two differ.
"""

from std.math import cos

from harness import Suite

from molla.nn.rope import (
    RopeSpec,
    angle,
    corr_dim,
    corr_range,
    rotate,
    rotate_heads,
    rotate_scaled,
    step_table,
)
from molla.nn.tensor import Buffer


def _close(a: Float32, b: Float32, within: Float32) -> Bool:
    var d = a - b
    if d < 0:
        d = -d
    return d <= within


def _fresh() -> List[Float32]:
    """One to eight, so a rotated value is easy to tell from an untouched one.
    """
    var x = List[Float32]()
    for i in range(8):
        x.append(Float32(i + 1))
    return x^


def _matches(got: List[Float32], want: List[Float32], within: Float32) -> Bool:
    if len(got) != len(want):
        return False
    for i in range(len(got)):
        if not _close(got[i], want[i], within):
            return False
    return True


def _want(
    a: Float32,
    b: Float32,
    c: Float32,
    d: Float32,
    e: Float32,
    f: Float32,
    g: Float32,
    h: Float32,
) -> List[Float32]:
    var out = List[Float32]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    out.append(e)
    out.append(f)
    out.append(g)
    out.append(h)
    return out^


def run(mut suite: Suite) raises:
    test_plain(suite)
    test_pairing(suite)
    test_linear(suite)
    test_ntk(suite)
    test_yarn(suite)
    test_factors(suite)
    test_heads(suite)
    test_steps(suite)
    test_errors(suite)


def test_plain(mut suite: Suite) raises:
    suite.group("rope with no scaling")

    var at_zero = _fresh()
    rotate(RopeSpec(8, 10000.0), at_zero, 0, 0)
    suite.check(
        _matches(at_zero, _fresh(), 1e-6),
        "position zero rotates by nothing and leaves the vector alone",
    )

    var x = _fresh()
    rotate(RopeSpec(8, 10000.0), x, 0, 3)
    suite.check(
        _matches(
            x,
            _want(
                -1.6955925,
                0.1375517,
                2.7886816,
                3.9759820,
                -4.8088425,
                6.3230593,
                7.0868367,
                8.0119640,
            ),
            1e-5,
        ),
        "and position three matches ggml at every pair",
    )

    # A rotation preserves length, which is a property of the operation rather
    # than of any one implementation of it, so this holds even if the reference
    # values above were transcribed wrong.
    var before = Float32(0)
    var after = Float32(0)
    var start = _fresh()
    for i in range(8):
        before += start[i] * start[i]
        after += x[i] * x[i]
    suite.check(
        _close(before, after, 1e-3), "and a rotation keeps the vector's length"
    )

    var turn = angle(RopeSpec(8, 10000.0), 3, 1, 1.0)
    suite.check(
        _close(turn[0], 0.9553365, 1e-6) and _close(turn[1], 0.2955202, 1e-6),
        "the angle for one pair is the cosine and sine of position over base",
    )


def test_pairing(mut suite: Suite) raises:
    suite.group("rope pairing")

    var neox = _fresh()
    var spec = RopeSpec(8, 10000.0)
    rotate(spec, neox, 0, 3)

    var norm_spec = RopeSpec(8, 10000.0)
    norm_spec.neox = False
    var norm = _fresh()
    rotate(norm_spec, norm, 0, 3)

    suite.check(
        _matches(
            norm,
            _want(
                -1.2722325,
                -1.8388650,
                1.6839286,
                4.7079066,
                4.8177772,
                6.1472777,
                6.9759685,
                8.0209640,
            ),
            1e-5,
        ),
        "adjacent pairing matches ggml, which is what a converted Llama wants",
    )
    suite.check(
        not _matches(neox, norm, 1e-3),
        (
            "and it is a different answer from pairing across the half, which"
            " is the bug that reads as a model going wrong after a few tokens"
        ),
    )

    # Partial rotary. Four elements rotate and four do not, and the four that do
    # not keep the values they came in with rather than being zeroed.
    var partial = _fresh()
    rotate(RopeSpec(4, 10000.0), partial, 0, 3)
    suite.check(
        partial[4] == 5.0
        and partial[5] == 6.0
        and partial[6] == 7.0
        and partial[7] == 8.0,
        "a rotary dimension shorter than the head leaves the tail untouched",
    )
    suite.check(partial[0] != 1.0, "and rotates the front of it")


def test_linear(mut suite: Suite) raises:
    suite.group("rope with linear scaling")

    var x = _fresh()
    rotate(RopeSpec.linear(8, 10000.0, 4.0, True), x, 0, 8)
    suite.check(
        _matches(
            x,
            _want(
                -4.9626340,
                0.7681172,
                2.8594094,
                3.9839920,
                -1.1714368,
                6.2777381,
                7.0585960,
                8.0079840,
            ),
            1e-5,
        ),
        "a factor of four puts position eight where two used to be",
    )

    var scaled = _fresh()
    rotate(RopeSpec.linear(8, 10000.0, 4.0, True), scaled, 0, 8)
    var plain = _fresh()
    rotate(RopeSpec(8, 10000.0), plain, 0, 2)
    suite.check(
        _matches(scaled, plain, 1e-4),
        "and that is exactly what dividing the position by four means",
    )

    var one = _fresh()
    rotate(RopeSpec.linear(8, 10000.0, 1.0, True), one, 0, 3)
    var none = _fresh()
    rotate(RopeSpec(8, 10000.0), none, 0, 3)
    suite.check(
        _matches(one, none, 1e-6), "and a factor of one is no scaling at all"
    )


def test_ntk(mut suite: Suite) raises:
    suite.group("rope with ntk scaling")

    var spec = RopeSpec.ntk(8, 10000.0, 4.0, True)
    suite.check(
        _close(spec.base, 63496.04, 1.0),
        "ntk scaling raises the base to factor to the d over d minus two",
    )
    suite.check(
        spec.scale == 1.0 and spec.ext_factor == 0.0,
        "and changes nothing else, which is why it needs no code in the loop",
    )

    var raised = False
    try:
        _ = RopeSpec.ntk(2, 10000.0, 4.0, True)
    except:
        raised = True
    suite.check(raised, "a rotary dimension of two has no ntk exponent")


def test_yarn(mut suite: Suite) raises:
    suite.group("rope with yarn")

    var spec = RopeSpec.yarn(8, 10000.0, 4.0, 2048, True)
    var ends = corr_range(spec)
    suite.check(
        ends[0] == 1.0 and ends[1] == 3.0,
        (
            "the ramp runs from the pair that turns 32 times to the one that"
            " turns once"
        ),
    )
    suite.check(
        _close(spec.attn_factor, 1.1386294, 1e-6),
        (
            "and the logits get warmed by one plus a tenth of the log of the"
            " factor"
        ),
    )

    var x = _fresh()
    rotate(spec, x, 0, 8)
    suite.check(
        _matches(
            x,
            _want(
                -5.7982327,
                -3.3142350,
                3.0132651,
                4.5362906,
                0.2981593,
                6.3933501,
                8.1311684,
                9.1181263,
            ),
            1e-5,
        ),
        "and the whole thing matches ggml at every pair",
    )

    # The first pair is outside the ramp on the fast side, so it should be
    # rotated as if nothing were scaled, up to the flat logit warming.
    var fast = angle(spec, 8, 0, 1.0)
    var unscaled = angle(RopeSpec(8, 10000.0), 8, 0, 1.0)
    suite.check(
        _close(fast[0], unscaled[0] * spec.attn_factor, 1e-5),
        "a pair faster than the ramp is not interpolated at all",
    )

    suite.check(
        corr_dim(spec, 32.0) < corr_dim(spec, 1.0),
        (
            "a pair that turns 32 times over the trained context comes before"
            " one that turns once"
        ),
    )

    var raised = False
    try:
        _ = RopeSpec.yarn(8, 10000.0, 4.0, 0, True)
    except:
        raised = True
    suite.check(raised, "yarn without a trained context length is refused")


def test_factors(mut suite: Suite) raises:
    suite.group("rope with per pair factors")

    var ff = List[Float32]()
    ff.append(1.0)
    ff.append(2.0)
    ff.append(4.0)
    ff.append(8.0)

    var x = _fresh()
    rotate_scaled(RopeSpec(8, 10000.0), x, 0, 3, ff, True)
    suite.check(
        _matches(
            x,
            _want(
                -1.6955925,
                1.0809134,
                2.9474161,
                3.9969997,
                -4.8088425,
                6.2315027,
                7.0223029,
                8.0014994,
            ),
            1e-5,
        ),
        (
            "each pair's angle is divided by its own factor, which is how a"
            " Llama 3.1 file carries its scaling"
        ),
    )

    var ones = List[Float32]()
    for _ in range(4):
        ones.append(1.0)
    var with_ones = _fresh()
    rotate_scaled(RopeSpec(8, 10000.0), with_ones, 0, 3, ones, True)
    var without = _fresh()
    rotate(RopeSpec(8, 10000.0), without, 0, 3)
    suite.check(
        _matches(with_ones, without, 1e-6),
        "and factors of one are the same as no factors",
    )

    var raised = False
    var short = List[Float32]()
    short.append(1.0)
    var y = _fresh()
    try:
        rotate_scaled(RopeSpec(8, 10000.0), y, 0, 3, short, True)
    except:
        raised = True
    suite.check(raised, "one factor short is an error rather than a read past")


def test_heads(mut suite: Suite) raises:
    suite.group("rope over a row of heads")

    var b = Buffer(16)
    for h in range(2):
        for i in range(8):
            b.data[h * 8 + i] = Float32(i + 1)
    var none = List[Float32]()
    rotate_heads(RopeSpec(8, 10000.0), b, 2, 8, 3, none, False)

    var one = _fresh()
    rotate(RopeSpec(8, 10000.0), one, 0, 3)
    var same = True
    for i in range(8):
        if not _close(b.data[i], one[i], 1e-6):
            same = False
        if not _close(b.data[8 + i], one[i], 1e-6):
            same = False
    suite.check(same, "every head at the same position gets the same rotation")

    var raised = False
    var small = Buffer(8)
    try:
        rotate_heads(RopeSpec(8, 10000.0), small, 2, 8, 3, none, False)
    except:
        raised = True
    suite.check(
        raised, "a buffer too small for the heads it was given is an error"
    )

    raised = False
    try:
        rotate_heads(RopeSpec(16, 10000.0), b, 2, 8, 3, none, False)
    except:
        raised = True
    suite.check(raised, "and so is a rotary dimension wider than the head")


def test_steps(mut suite: Suite) raises:
    """The frequency step table a device kernel is handed instead of a base.

    It has to be the same value `angle` uses and not merely close to it, which
    is the whole reason it exists. `angle` folds the step into a cosine, so the
    way to compare them without exporting a private is to rotate a pair whose
    input makes the answer the step's own cosine and sine, at position one where
    the angle is the step itself.
    """
    suite.group("rope frequency steps")

    var spec = RopeSpec(64, 500000.0)
    var steps = step_table(spec)
    suite.check(len(steps) == 32, "there is one step per pair")
    suite.check(steps[0] == 1.0, "and the first pair turns a whole radian")

    var falls = True
    for i in range(1, len(steps)):
        if steps[i] >= steps[i - 1]:
            falls = False
    suite.check(falls, "and every pair after it turns more slowly")

    var worst = Float32(0)
    for pair in range(len(steps)):
        var turn = angle(spec, 1, pair, 1.0)
        var want_c = Float32(cos(Float64(steps[pair])))
        var gap = turn[0] - want_c
        if gap < 0:
            gap = -gap
        if gap > worst:
            worst = gap
    suite.check(
        worst == 0,
        "and the table holds exactly what angle works out for itself",
    )

    # The two Gemma 3 alternates, which is the case where a table cached against
    # the wrong spec would be read for the wrong layer.
    var local = step_table(RopeSpec(64, 10000.0))
    var glob = step_table(RopeSpec(64, 1000000.0))
    suite.check(
        local[16] != glob[16], "and two bases give two different tables"
    )


def test_errors(mut suite: Suite) raises:
    suite.group("rope errors")

    var raised = False
    try:
        _ = RopeSpec.linear(8, 10000.0, 0.0, True)
    except:
        raised = True
    suite.check(raised, "a scaling factor of zero is refused")

    raised = False
    var odd = _fresh()
    try:
        rotate(RopeSpec(7, 10000.0), odd, 0, 3)
    except:
        raised = True
    suite.check(raised, "and so is an odd rotary dimension, which has no pairs")

    raised = False
    try:
        _ = angle(RopeSpec(8, 10000.0), 3, 4, 1.0)
    except:
        raised = True
    suite.check(raised, "a pair past the end of the head has no angle")

    raised = False
    try:
        _ = angle(RopeSpec(8, 10000.0), 3, 0, 0.0)
    except:
        raised = True
    suite.check(raised, "and a frequency factor of zero has none either")
