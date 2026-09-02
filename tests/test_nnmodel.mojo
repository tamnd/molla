"""Everything above a layer: the embedding, the head, and the stack.

Named `test_nnmodel` rather than `test_model` because there is already a
`molla.model` package for what a file says, and two test modules a letter apart
is how a check ends up registered twice and run never.

The model here is four wide with one head and two layers, over a vocabulary of
three tokens. That is small enough that the embedding lookup, the tie between
the head and the embedding, and the softcap can each be checked against a
number rather than against another program.
"""

from std.math import sqrt, tanh

from harness import Suite

from molla.model.spec import ARCH_GEMMA2, ARCH_LLAMA, ARCH_QWEN3
from molla.nn.arch import arch_of
from molla.nn.attention import AttnSpec
from molla.nn.block import BlockSpec, LayerWeights, Scratch
from molla.nn.model import (
    ModelWeights,
    embed,
    forward,
    frequency_factors,
    head,
)
from molla.nn.quant import Q_F32
from molla.nn.rope import RopeSpec
from molla.nn.tensor import Buffer, Tensor
from molla.sys.mem import keep

comptime WIDTH = 4
comptime VOCAB = 3
comptime HIDDEN = 4
comptime EPS = Float32(1e-5)


def _close(a: Float32, b: Float32, within: Float32) -> Bool:
    var d = a - b
    if d < 0:
        d = -d
    return d <= within


def _f32_bytes(values: List[Float32]) -> List[UInt8]:
    var out = List[UInt8]()
    for v in values:
        var bits = Int(v.to_bits())
        for shift in range(4):
            out.append(UInt8((bits >> (shift * 8)) & 0xFF))
    return out^


struct Arena(Movable):
    var held: List[List[UInt8]]

    def __init__(out self):
        self.held = List[List[UInt8]]()

    def tensor(mut self, values: List[Float32], cols: Int, rows: Int) -> Tensor:
        self.held.append(_f32_bytes(values))
        var last = len(self.held) - 1
        return Tensor(Int(self.held[last].unsafe_ptr()), Q_F32, cols, rows)


def _ones(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(1.0)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


def _identity(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for r in range(n):
        for c in range(n):
            out.append(Float32(1.0) if r == c else Float32(0.0))
    return out^


def _rows() -> List[Float32]:
    """Three embedding rows, distinct enough to tell apart at a glance."""
    var out = List[Float32]()
    for token in range(VOCAB):
        for i in range(WIDTH):
            out.append(Float32(token * 10 + i + 1))
    return out^


def _spec() raises -> BlockSpec:
    return BlockSpec(
        AttnSpec(1, 1, WIDTH), RopeSpec(WIDTH, 10000.0), WIDTH, HIDDEN, EPS
    )


def run(mut suite: Suite) raises:
    test_embed(suite)
    test_tied(suite)
    test_softcap(suite)
    test_factors(suite)
    test_forward(suite)
    test_trace(suite)
    test_errors(suite)


def test_embed(mut suite: Suite) raises:
    suite.group("model embedding")

    var arena = Arena()
    var w = ModelWeights()
    w.embedding = arena.tensor(_rows(), WIDTH, VOCAB)
    w.output_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)

    var x = Buffer(WIDTH)
    embed(w, arch_of(ARCH_LLAMA), 1, x)
    suite.check(
        x.data[0] == 11.0 and x.data[3] == 14.0,
        "an embedding lookup copies the token's row",
    )

    embed(w, arch_of(ARCH_LLAMA), 0, x)
    suite.check(
        x.data[0] == 1.0 and x.data[3] == 4.0,
        "and a different row for a different token",
    )

    # Gemma multiplies by the root of the width, which is two here.
    embed(w, arch_of(ARCH_GEMMA2), 0, x)
    suite.check(
        _close(x.data[0], 2.0, 1e-5) and _close(x.data[3], 8.0, 1e-5),
        "and a gemma scales it by the root of the model width",
    )

    var raised = False
    try:
        embed(w, arch_of(ARCH_LLAMA), VOCAB, x)
    except:
        raised = True
    suite.check(raised, "a token past the end of the vocabulary is an error")

    raised = False
    try:
        embed(w, arch_of(ARCH_LLAMA), -1, x)
    except:
        raised = True
    suite.check(raised, "and so is a negative one")

    keep(arena)


def test_tied(mut suite: Suite) raises:
    """A tied head reads the embedding, an untied one reads its own tensor."""
    suite.group("model output head")

    var arena = Arena()
    var spec = _spec()
    var w = ModelWeights()
    w.embedding = arena.tensor(_rows(), WIDTH, VOCAB)
    w.output_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    w.check(spec)

    suite.check(w.tied(), "a model with no output tensor has a tied head")
    suite.check(w.vocab() == VOCAB, "and its vocabulary is the embedding's")

    var x = Buffer(WIDTH)
    for i in range(WIDTH):
        x.data[i] = Float32(i + 1)
    var normed = Buffer(WIDTH)
    var logits = Buffer(VOCAB)
    head(w, arch_of(ARCH_LLAMA), spec, x, normed, logits)

    # The mean square of one through four is 7.5, so every element of the
    # normed vector is itself over the root of that, and each logit is the dot
    # of that with one embedding row.
    var divisor = Float32(sqrt(Float64(7.5) + Float64(EPS)))
    var want0 = Float32(0)
    for i in range(WIDTH):
        want0 += Float32(i + 1) / divisor * Float32(i + 1)
    suite.check(
        _close(logits.data[0], want0, 1e-4),
        "a tied head takes the dot of the normed stream with each row",
    )

    # The same model with a head of its own gives different logits, which is
    # what says tying is read off the file rather than assumed.
    var other = ModelWeights()
    other.embedding = w.embedding
    other.output_norm = w.output_norm
    other.output = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    suite.check(
        not other.tied() and other.vocab() == WIDTH,
        "and an untied head has its own vocabulary size",
    )
    var wide = Buffer(WIDTH)
    head(other, arch_of(ARCH_LLAMA), spec, x, normed, wide)
    suite.check(
        _close(wide.data[0], 1.0 / divisor, 1e-5),
        "and an untied head reads its own tensor",
    )

    keep(arena)


def test_softcap(mut suite: Suite) raises:
    suite.group("model logit softcap")

    var arena = Arena()
    var spec = _spec()
    var w = ModelWeights()
    # An embedding with one enormous row, so an uncapped logit would be huge.
    var rows = List[Float32]()
    for token in range(VOCAB):
        for _ in range(WIDTH):
            rows.append(Float32(1000.0) if token == 2 else Float32(1.0))
    w.embedding = arena.tensor(rows, WIDTH, VOCAB)
    w.output_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)

    var x = Buffer(WIDTH)
    for i in range(WIDTH):
        x.data[i] = Float32(i + 1)
    var normed = Buffer(WIDTH)
    var plain = Buffer(VOCAB)
    head(w, arch_of(ARCH_LLAMA), spec, x, normed, plain)
    suite.check(plain.data[2] > 1000.0, "without a cap a logit can be huge")

    var capped = Buffer(VOCAB)
    head(w, arch_of(ARCH_GEMMA2), spec, x, normed, capped)
    suite.check(
        capped.data[2] <= 30.0 and capped.data[2] > 29.9,
        "and gemma2's cap of thirty pulls it just under",
    )
    suite.check(
        _close(
            capped.data[0],
            30.0 * Float32(tanh(Float64(plain.data[0] / 30.0))),
            1e-4,
        ),
        "and the small ones go through the same tanh rather than being left",
    )

    keep(arena)


def test_factors(mut suite: Suite) raises:
    suite.group("model rope frequency factors")

    var arena = Arena()
    var w = ModelWeights()
    w.embedding = arena.tensor(_rows(), WIDTH, VOCAB)
    w.output_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)

    var none = frequency_factors(w)
    suite.check(
        len(none) == 0,
        "a file with no rope_freqs tensor gives no frequency factors",
    )

    var values = List[Float32]()
    values.append(1.0)
    values.append(4.0)
    w.rope_freqs = arena.tensor(values, 2, 1)
    var got = frequency_factors(w)
    suite.check(
        len(got) == 2 and got[0] == 1.0 and got[1] == 4.0,
        "and a file with one reads them once rather than per token",
    )

    var raised = False
    var wrong = w
    wrong.rope_freqs = arena.tensor(_ones(3), 3, 1)
    try:
        wrong.check(_spec())
    except:
        raised = True
    suite.check(
        raised,
        "and a count that does not match the rotary dimension is refused",
    )

    keep(arena)


def test_forward(mut suite: Suite) raises:
    """Two layers, two tokens, with weights that make the answer checkable.

    Every projection is the identity and every gain is one, so a layer adds the
    normed stream twice, once from attention and once from the mlp. What is
    being checked is that both layers ran, in order, into their own slice of
    the cache, and that the head saw the result.
    """
    suite.group("model forward")

    var arena = Arena()
    var spec = _spec()
    var w = ModelWeights()
    w.embedding = arena.tensor(_rows(), WIDTH, VOCAB)
    w.output_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)

    var ident = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    var gain = arena.tensor(_ones(WIDTH), WIDTH, 1)
    var lw = LayerWeights()
    lw.attn_norm = gain
    lw.ffn_norm = gain
    lw.wq = ident
    lw.wk = ident
    lw.wv = ident
    lw.wo = ident
    lw.gate = ident
    lw.up = ident
    lw.down = ident
    lw.check(spec)

    var specs = List[BlockSpec]()
    var layers = List[LayerWeights]()
    var keys = List[List[Float32]]()
    var values = List[List[Float32]]()
    for _ in range(2):
        specs.append(spec)
        layers.append(lw)
        keys.append(_zeros(WIDTH * 2))
        values.append(_zeros(WIDTH * 2))

    var s = Scratch(spec, 4)
    var x = Buffer(WIDTH)
    var logits = Buffer(VOCAB)
    var a = arch_of(ARCH_LLAMA)
    forward(
        a,
        w,
        specs,
        layers,
        s,
        x,
        1,
        0,
        0,
        keys,
        values,
        List[Float32](),
        logits,
    )

    suite.check(
        logits.elements() == VOCAB, "a forward pass gives one logit per token"
    )

    # Token one's row is 11 through 14, whose mean square is 630 over 4, so
    # the first layer's attention adds each element over the root of that.
    var d0 = Float32(sqrt(Float64(157.5) + Float64(EPS)))
    suite.check(
        _close(keys[0][0], 11.0 / d0, 1e-4),
        "the first layer cached the normed embedding as its key",
    )
    suite.check(
        keys[1][0] != 0.0 and keys[1][0] != keys[0][0],
        "and the second layer cached something of its own",
    )

    var untouched = True
    for i in range(WIDTH):
        if keys[0][WIDTH + i] != 0.0:
            untouched = False
    suite.check(untouched, "and neither of them wrote past slot zero")

    # A second token at slot one has to see the first, which is what the cache
    # is for, so its logits differ from the same token run on its own.
    var second = Buffer(WIDTH)
    var with_history = Buffer(VOCAB)
    forward(
        a,
        w,
        specs,
        layers,
        s,
        second,
        2,
        1,
        1,
        keys,
        values,
        List[Float32](),
        with_history,
    )

    var fresh_k = List[List[Float32]]()
    var fresh_v = List[List[Float32]]()
    for _ in range(2):
        fresh_k.append(_zeros(WIDTH * 2))
        fresh_v.append(_zeros(WIDTH * 2))
    var alone = Buffer(WIDTH)
    var no_history = Buffer(VOCAB)
    forward(
        a,
        w,
        specs,
        layers,
        s,
        alone,
        2,
        0,
        0,
        fresh_k,
        fresh_v,
        List[Float32](),
        no_history,
    )

    var differs = False
    for i in range(VOCAB):
        if not _close(with_history.data[i], no_history.data[i], 1e-5):
            differs = True
    suite.check(differs, "and a token's logits depend on the tokens before it")

    keep(arena)


def test_trace(mut suite: Suite) raises:
    """The residual stream recorded on the way through.

    What matters is the numbering. Snapshot zero has to be the embedding
    before any layer touched it and snapshot n has to be what layer n minus
    one left, because `scripts/logit_oracle.mojo` turns a disagreement into a
    layer number by that indexing and an off by one there would name the wrong
    layer every time. So the ends are checked against values that come from
    somewhere other than the trace: the embedding row itself, and the stream
    `forward` finished with.
    """
    suite.group("model trace")

    var arena = Arena()
    var spec = _spec()
    var w = ModelWeights()
    w.embedding = arena.tensor(_rows(), WIDTH, VOCAB)
    w.output_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)

    var ident = arena.tensor(_identity(WIDTH), WIDTH, WIDTH)
    var gain = arena.tensor(_ones(WIDTH), WIDTH, 1)
    var lw = LayerWeights()
    lw.attn_norm = gain
    lw.ffn_norm = gain
    lw.wq = ident
    lw.wk = ident
    lw.wv = ident
    lw.wo = ident
    lw.gate = ident
    lw.up = ident
    lw.down = ident

    var specs = List[BlockSpec]()
    var layers = List[LayerWeights]()
    var keys = List[List[Float32]]()
    var values = List[List[Float32]]()
    for _ in range(2):
        specs.append(spec)
        layers.append(lw)
        keys.append(_zeros(WIDTH * 2))
        values.append(_zeros(WIDTH * 2))

    var s = Scratch(spec, 4)
    var a = arch_of(ARCH_LLAMA)
    var x = Buffer(WIDTH)
    var logits = Buffer(VOCAB)

    suite.check(not s.tracing, "a scratch does not record until it is asked")
    forward(
        a,
        w,
        specs,
        layers,
        s,
        x,
        1,
        0,
        0,
        keys,
        values,
        List[Float32](),
        logits,
    )
    suite.check(
        len(s.trace) == 0, "and a pass with it off leaves nothing behind"
    )

    s.tracing = True
    var fresh_k = List[List[Float32]]()
    var fresh_v = List[List[Float32]]()
    for _ in range(2):
        fresh_k.append(_zeros(WIDTH * 2))
        fresh_v.append(_zeros(WIDTH * 2))
    var traced = Buffer(WIDTH)
    forward(
        a,
        w,
        specs,
        layers,
        s,
        traced,
        1,
        0,
        0,
        fresh_k,
        fresh_v,
        List[Float32](),
        logits,
    )

    suite.check(
        s.snapshots(WIDTH) == 4,
        "two layers give four snapshots for a token",
    )

    var lookup = Buffer(WIDTH)
    embed(w, a, 1, lookup)
    var same = True
    for i in range(WIDTH):
        if s.trace[i] != lookup.data[i]:
            same = False
    suite.check(same, "and the first one is the embedding, before any layer")

    var last = 2 * WIDTH
    var ended = True
    for i in range(WIDTH):
        if s.trace[last + i] != traced.data[i]:
            ended = False
    suite.check(ended, "and the one after that is what the last layer left")

    var normed = 3 * WIDTH
    var final = True
    for i in range(WIDTH):
        if s.trace[normed + i] != s.norm.data[i]:
            final = False
    suite.check(final, "and the last one is the norm the head read")

    var moved = False
    for i in range(WIDTH):
        if s.trace[WIDTH + i] != s.trace[i]:
            moved = True
    suite.check(moved, "and the one between them is neither of those")

    # A second token appends rather than replacing, so a prefill of n tokens
    # leaves n times the snapshots in the order they were taken. The oracle
    # indexes into that as position times snapshots plus layer and would read
    # the wrong token if this ever started overwriting.

    var second = Buffer(WIDTH)
    forward(
        a,
        w,
        specs,
        layers,
        s,
        second,
        2,
        1,
        1,
        fresh_k,
        fresh_v,
        List[Float32](),
        logits,
    )
    suite.check(s.snapshots(WIDTH) == 8, "a second token appends its own four")
    var kept = True
    for i in range(WIDTH):
        if s.trace[i] != lookup.data[i]:
            kept = False
    suite.check(kept, "and does not disturb the first token's")

    s.forget()
    suite.check(
        s.snapshots(WIDTH) == 0 and s.tracing,
        "forgetting empties the trace and leaves the switch alone",
    )

    keep(arena)


def test_errors(mut suite: Suite) raises:
    suite.group("model errors")

    var arena = Arena()
    var spec = _spec()

    var raised = False
    try:
        ModelWeights().check(spec)
    except:
        raised = True
    suite.check(raised, "a model with no embedding is refused")

    raised = False
    var no_norm = ModelWeights()
    no_norm.embedding = arena.tensor(_rows(), WIDTH, VOCAB)
    try:
        no_norm.check(spec)
    except:
        raised = True
    suite.check(raised, "and one with no output norm")

    raised = False
    var narrow = ModelWeights()
    narrow.embedding = arena.tensor(_ones(2 * VOCAB), 2, VOCAB)
    narrow.output_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    try:
        narrow.check(spec)
    except:
        raised = True
    suite.check(
        raised, "and an embedding that is not the width the layers want"
    )

    var w = ModelWeights()
    w.embedding = arena.tensor(_rows(), WIDTH, VOCAB)
    w.output_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    var s = Scratch(spec, 4)
    var x = Buffer(WIDTH)
    var logits = Buffer(VOCAB)
    var a = arch_of(ARCH_LLAMA)

    raised = False
    var empty_specs = List[BlockSpec]()
    var empty_layers = List[LayerWeights]()
    var empty_k = List[List[Float32]]()
    var empty_v = List[List[Float32]]()
    try:
        forward(
            a,
            w,
            empty_specs,
            empty_layers,
            s,
            x,
            0,
            0,
            0,
            empty_k,
            empty_v,
            List[Float32](),
            logits,
        )
    except:
        raised = True
    suite.check(raised, "a model with no layers is not a model")

    raised = False
    var specs = List[BlockSpec]()
    specs.append(spec)
    var layers = List[LayerWeights]()
    var keys = List[List[Float32]]()
    var values = List[List[Float32]]()
    keys.append(_zeros(WIDTH))
    values.append(_zeros(WIDTH))
    try:
        forward(
            a,
            w,
            specs,
            layers,
            s,
            x,
            0,
            0,
            0,
            keys,
            values,
            List[Float32](),
            logits,
        )
    except:
        raised = True
    suite.check(raised, "and one with more specs than sets of weights")

    keep(arena)
