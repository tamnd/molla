"""Attention over a span of keys and values.

Small enough to work out by hand, which is the point. The vectors here are two
elements wide and there are never more than four of them, so an expected output
is a line of arithmetic rather than something another program said. The
properties that hold whatever the numbers are, that the probabilities sum to
one and that the output is a convex combination of the values, are checked
separately, since those catch a mistake in the setup that the expected values
would agree with.
"""

from harness import Suite

from molla.nn.attention import AttnSpec, attend
from molla.nn.tensor import Buffer


def _close(a: Float32, b: Float32, within: Float32) -> Bool:
    var d = a - b
    if d < 0:
        d = -d
    return d <= within


def _scratch(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


def _list(*values: Float32) -> List[Float32]:
    var out = List[Float32]()
    for v in values:
        out.append(v)
    return out^


def run(mut suite: Suite) raises:
    test_shape(suite)
    test_visible(suite)
    test_math(suite)
    test_grouped(suite)
    test_window(suite)
    test_softcap(suite)
    test_errors(suite)


def test_shape(mut suite: Suite) raises:
    suite.group("attention shape")

    var mha = AttnSpec(8, 8, 64)
    suite.check(mha.group() == 1, "multi head is one query head per key head")
    suite.check(mha.kv_head_of(7) == 7, "and head seven reads key head seven")

    var gqa = AttnSpec(32, 8, 128)
    suite.check(gqa.group() == 4, "an 8B Llama shares one key head with four")
    suite.check(
        gqa.kv_head_of(0) == 0
        and gqa.kv_head_of(3) == 0
        and gqa.kv_head_of(4) == 1
        and gqa.kv_head_of(31) == 7,
        "and the four that share it are consecutive",
    )

    var mqa = AttnSpec(16, 1, 64)
    suite.check(
        mqa.group() == 16 and mqa.kv_head_of(15) == 0,
        "multi query is the same thing with one key head",
    )

    suite.check(
        _close(AttnSpec(8, 8, 64).scale, 0.125, 1e-7),
        "the default scale is one over the root of the head dimension",
    )

    var raised = False
    try:
        _ = AttnSpec(6, 4, 64)
    except:
        raised = True
    suite.check(
        raised,
        "query heads that are not a whole multiple of key heads are refused",
    )

    raised = False
    try:
        _ = AttnSpec(0, 1, 64)
    except:
        raised = True
    suite.check(raised, "and so is a layer with no heads")


def test_visible(mut suite: Suite) raises:
    suite.group("attention masking")

    var open = AttnSpec(1, 1, 2)
    suite.check(
        open.visible(0, 1000) and open.visible(999, 1000),
        "with no window every key is visible",
    )

    var windowed = AttnSpec(1, 1, 2)
    windowed.window = 4
    suite.check(
        windowed.visible(3, 3)
        and windowed.visible(0, 3)
        and not windowed.visible(0, 4),
        "a window of four sees itself and the three before it",
    )
    suite.check(
        windowed.visible(100, 100) and not windowed.visible(96, 100),
        "and the window moves with the position",
    )

    var sunk = AttnSpec(1, 1, 2)
    sunk.window = 4
    sunk.sinks = 2
    suite.check(
        sunk.visible(0, 100) and sunk.visible(1, 100),
        "sink tokens stay visible however far the window has moved",
    )
    suite.check(
        not sunk.visible(2, 100),
        "and the third token is not a sink and falls outside",
    )


def test_math(mut suite: Suite) raises:
    """One head, two elements wide, three keys, worked out on paper.

    q is `[1, 0]` and the keys are `[1,0]`, `[0,1]` and `[1,1]`, so the raw
    scores are `1/sqrt(2)`, `0` and `1/sqrt(2)`. The softmax of those is
    `0.401112`, `0.197776`, `0.401112`, and the values are the same three
    vectors, so the output is `[0.802224, 0.598888]`.
    """
    suite.group("attention arithmetic")

    var spec = AttnSpec(1, 1, 2)
    var q = Buffer(2)
    q.data[0] = 1.0
    q.data[1] = 0.0
    var keys = _list(1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    var values = _list(1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    var out = Buffer(2)
    var scores = _scratch(3)
    attend(spec, q, keys, values, 3, 2, out, scores)

    suite.check(
        _close(out.data[0], 0.8022242, 1e-6)
        and _close(out.data[1], 0.5988879, 1e-6),
        "three keys against one query give the weighted values",
    )
    suite.check(
        _close(scores[0] + scores[1] + scores[2], 1.0, 1e-6),
        "and the weights are a distribution",
    )
    suite.check(
        _close(scores[0], scores[2], 1e-7) and scores[1] < scores[0],
        "and two keys that score the same get the same weight",
    )

    # One key is the degenerate case and it should come back as that key's
    # value exactly, since a softmax over one number is one.
    var single = Buffer(2)
    var one_score = _scratch(1)
    attend(spec, q, keys, values, 1, 0, single, one_score)
    suite.check(
        single.data[0] == 1.0 and single.data[1] == 0.0,
        "a single key is copied through untouched",
    )

    # A convex combination cannot leave the box the values are in. This holds
    # for any query at all, which is what makes it worth checking separately
    # from the numbers above.
    var far = Buffer(2)
    far.data[0] = -50.0
    far.data[1] = 17.0
    var boxed = Buffer(2)
    attend(spec, far, keys, values, 3, 2, boxed, scores)
    suite.check(
        boxed.data[0] >= 0.0
        and boxed.data[0] <= 1.0
        and boxed.data[1] >= 0.0
        and boxed.data[1] <= 1.0,
        "and an output stays inside the values it was mixed from",
    )


def test_grouped(mut suite: Suite) raises:
    suite.group("attention grouped query")

    # Four query heads, two key heads, one element per head. Key head 0 holds
    # 1 and key head 1 holds 10, so which key head a query read is visible in
    # the answer rather than having to be inferred.
    var spec = AttnSpec(4, 2, 1)
    var q = Buffer(4)
    for h in range(4):
        q.data[h] = 1.0
    var keys = _list(1.0, 10.0)
    var values = _list(3.0, 30.0)
    var out = Buffer(4)
    var scores = _scratch(1)
    attend(spec, q, keys, values, 1, 0, out, scores)

    suite.check(
        out.data[0] == 3.0 and out.data[1] == 3.0,
        "the first two heads read the first key head",
    )
    suite.check(
        out.data[2] == 30.0 and out.data[3] == 30.0,
        "and the last two read the second",
    )


def test_window(mut suite: Suite) raises:
    """Four keys at position three, with and without a window.

    The same setup three ways, so the numbers say which keys were actually
    looked at rather than only that something came out.
    """
    suite.group("attention with a window")

    var q = Buffer(2)
    q.data[0] = 1.0
    q.data[1] = 0.0
    var keys = _list(1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0)
    var values = _list(1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0)
    var out = Buffer(2)
    var scores = _scratch(4)

    var open = AttnSpec(1, 1, 2)
    attend(open, q, keys, values, 4, 3, out, scores)
    suite.check(
        _close(out.data[0], 0.4423620, 1e-6)
        and _close(out.data[1], 1.2273995, 1e-6),
        "with no window all four keys are mixed",
    )

    var windowed = AttnSpec(1, 1, 2)
    windowed.window = 2
    attend(windowed, q, keys, values, 4, 3, out, scores)
    suite.check(
        _close(out.data[0], 0.3302385, 1e-6)
        and _close(out.data[1], 1.6697615, 1e-6),
        "a window of two at position three sees only the last two",
    )

    var sunk = AttnSpec(1, 1, 2)
    sunk.window = 2
    sunk.sinks = 1
    attend(sunk, q, keys, values, 4, 3, out, scores)
    suite.check(
        _close(out.data[0], 0.4965102, 1e-6)
        and _close(out.data[1], 1.2552348, 1e-6),
        "and one sink token puts the first key back in without the second",
    )


def test_softcap(mut suite: Suite) raises:
    suite.group("attention with a softcap")

    var spec = AttnSpec(1, 1, 2)
    spec.softcap = 2.0
    var q = Buffer(2)
    q.data[0] = 1.0
    q.data[1] = 0.0
    var keys = _list(1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    var values = _list(1.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    var out = Buffer(2)
    var scores = _scratch(3)
    attend(spec, q, keys, values, 3, 2, out, scores)
    suite.check(
        _close(out.data[0], 0.7977343, 1e-6)
        and _close(out.data[1], 0.6011329, 1e-6),
        "a cap of two squashes the scores through a tanh before the softmax",
    )

    # A score far above the cap has to land just under it, which is what stops
    # one key taking the whole row.
    var huge = AttnSpec(1, 1, 2)
    huge.softcap = 2.0
    var big_q = Buffer(2)
    big_q.data[0] = 1000.0
    big_q.data[1] = 0.0
    attend(huge, big_q, keys, values, 3, 2, out, scores)
    suite.check(
        scores[1] > 0.01,
        (
            "and a query a thousand times too large still leaves the other keys"
            " some weight"
        ),
    )


def test_errors(mut suite: Suite) raises:
    suite.group("attention errors")

    var spec = AttnSpec(1, 1, 2)
    var q = Buffer(2)
    var keys = _list(1.0, 0.0, 0.0, 1.0)
    var values = _list(1.0, 0.0, 0.0, 1.0)
    var out = Buffer(2)
    var scores = _scratch(2)

    var raised = False
    try:
        attend(spec, q, keys, values, 0, 0, out, scores)
    except:
        raised = True
    suite.check(raised, "attending to no keys at all is an error")

    raised = False
    try:
        attend(spec, q, keys, values, 3, 2, out, scores)
    except:
        raised = True
    suite.check(raised, "and so is claiming more keys than were handed over")

    raised = False
    var small = Buffer(1)
    try:
        attend(spec, small, keys, values, 2, 1, out, scores)
    except:
        raised = True
    suite.check(raised, "and a query narrower than the heads it claims")

    raised = False
    var short = _scratch(1)
    try:
        attend(spec, q, keys, values, 2, 1, out, short)
    except:
        raised = True
    suite.check(raised, "and scratch too small to hold the scores")

    # A window that lands entirely before the keys that exist would leave a
    # head with nothing to attend to, and a softmax over nothing is not zero,
    # it is undefined. Saying so beats writing NaN into the residual stream.
    raised = False
    var impossible = AttnSpec(1, 1, 2)
    impossible.window = 1
    try:
        attend(impossible, q, keys, values, 2, 99, out, scores)
    except:
        raised = True
    suite.check(raised, "and a window that can see no key at all")
