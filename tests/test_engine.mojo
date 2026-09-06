"""The cache, and the loop that fills it.

The model here is four wide, one head, two layers, three tokens of vocabulary,
which is the same shape `test_nnmodel` uses and small enough that a cache can be
compared position by position.

The check this module exists for is the one at the end. Prefilling a prompt and
then decoding has to leave exactly the same bytes in the cache as feeding those
same tokens one at a time, and it has to be exact rather than close, because a
route that is off by one position produces numbers that are close and a
tolerance would pass it.
"""

from harness import Suite

from molla.engine.bind import Bound
from molla.engine.cache import CELL_FREE, MAX_SEQS, CellTable, KvCache
from molla.engine.sample import Sampler, SamplerConfig
from molla.engine.session import Session
from molla.model.spec import ARCH_LLAMA, Geometry
from molla.nn.arch import arch_of
from molla.nn.attention import AttnSpec
from molla.nn.block import BlockSpec, LayerWeights
from molla.nn.model import ModelWeights
from molla.nn.quant import Q_F32
from molla.nn.rope import RopeSpec
from molla.nn.tensor import Tensor
from molla.sys.mem import keep

comptime WIDTH = 4
comptime VOCAB = 3
comptime HIDDEN = 4
comptime LAYERS = 2
comptime EPS = Float32(1e-5)


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


def _dense(cols: Int, rows: Int, seed: Int) -> List[Float32]:
    """Weights that are not the identity and not symmetric.

    An identity projection hides an ordering mistake, because a layer that
    multiplies by one twice in the wrong order gets the same answer.
    """
    var out = List[Float32]()
    for r in range(rows):
        for c in range(cols):
            var n = (r * 13 + c * 7 + seed * 5) % 9
            out.append(Float32(n - 4) / 8.0)
    return out^


def _rows() -> List[Float32]:
    var out = List[Float32]()
    for token in range(VOCAB):
        for i in range(WIDTH):
            out.append(Float32(token * 10 + i + 1) / 16.0)
    return out^


def _geometry() -> Geometry:
    return Geometry(
        block_count=LAYERS,
        context_length=64,
        embedding_length=WIDTH,
        feed_forward_length=HIDDEN,
        head_count=1,
        head_count_kv=1,
        key_length=WIDTH,
        value_length=WIDTH,
        expert_count=0,
        expert_used_count=0,
        rope_dimension_count=WIDTH,
        rope_freq_base=10000.0,
        rope_scale_factor=0.0,
        rope_scaling="unstated",
        epsilon=1e-5,
        head_dim_stated=True,
        kv_stated=True,
        rope_dims_stated=True,
    )


def _spec() raises -> BlockSpec:
    return BlockSpec(
        AttnSpec(1, 1, WIDTH), RopeSpec(WIDTH, 10000.0), WIDTH, HIDDEN, EPS
    )


def _layer(mut arena: Arena, seed: Int) -> LayerWeights:
    var w = LayerWeights()
    w.attn_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    w.wq = arena.tensor(_dense(WIDTH, WIDTH, seed), WIDTH, WIDTH)
    w.wk = arena.tensor(_dense(WIDTH, WIDTH, seed + 1), WIDTH, WIDTH)
    w.wv = arena.tensor(_dense(WIDTH, WIDTH, seed + 2), WIDTH, WIDTH)
    w.wo = arena.tensor(_dense(WIDTH, WIDTH, seed + 3), WIDTH, WIDTH)
    w.ffn_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)
    w.gate = arena.tensor(_dense(WIDTH, HIDDEN, seed + 4), WIDTH, HIDDEN)
    w.up = arena.tensor(_dense(WIDTH, HIDDEN, seed + 5), WIDTH, HIDDEN)
    w.down = arena.tensor(_dense(HIDDEN, WIDTH, seed + 6), HIDDEN, WIDTH)
    return w


def _bound(mut arena: Arena) raises -> Bound:
    var model = ModelWeights()
    model.embedding = arena.tensor(_rows(), WIDTH, VOCAB)
    model.output_norm = arena.tensor(_ones(WIDTH), WIDTH, 1)

    var layers = List[LayerWeights]()
    var specs = List[BlockSpec]()
    for i in range(LAYERS):
        layers.append(_layer(arena, i * 3))
        specs.append(_spec())
    return Bound(arch_of(ARCH_LLAMA), _geometry(), model, layers^, specs^)


def run(mut suite: Suite) raises:
    test_cache_shape(suite)
    test_cache_room(suite)
    test_cache_errors(suite)
    test_cells(suite)
    test_cells_sharing(suite)
    test_cells_errors(suite)
    test_session_step(suite)
    test_prefill_matches_decode(suite)
    test_generate(suite)
    test_session_errors(suite)


def test_cache_shape(mut suite: Suite) raises:
    suite.group("cache shape")

    var c = KvCache(3, 8, 4)
    suite.check(len(c.keys) == 3, "one key list per layer")
    suite.check(len(c.values) == 3, "one value list per layer")
    suite.check(
        len(c.keys[0]) == 32 and len(c.values[0]) == 32,
        "each one is the context by the key width",
    )
    suite.check(
        c.bytes() == 2 * 3 * 8 * 4 * 4,
        "and the size it reports is keys and values, four bytes each",
    )
    suite.check(c.filled == 0, "a new cache has nothing in it")


def test_cache_room(mut suite: Suite) raises:
    suite.group("cache room")

    var c = KvCache(1, 4, 2)
    suite.check(c.room() == 4, "an empty cache has room for the whole context")
    c.advance(3)
    suite.check(c.filled == 3 and c.room() == 1, "and less after advancing")
    suite.check(c.slot_for(3) == 3, "a slot is a position, for now")

    var overflowed = False
    try:
        c.advance(2)
    except:
        overflowed = True
    suite.check(overflowed, "a sequence that does not fit is refused")
    suite.check(
        c.filled == 3,
        "and a refused reservation leaves the cache where it was",
    )

    c.reset()
    suite.check(
        c.filled == 0 and c.room() == 4,
        "reset gives the room back without freeing anything",
    )


def test_cache_errors(mut suite: Suite) raises:
    suite.group("cache errors")

    var failed = False
    try:
        _ = KvCache(0, 8, 4)
    except:
        failed = True
    suite.check(failed, "a cache with no layers is refused")

    failed = False
    try:
        _ = KvCache(2, 0, 4)
    except:
        failed = True
    suite.check(failed, "so is one with no room")

    var c = KvCache(1, 4, 2)
    failed = False
    try:
        _ = c.slot_for(-1)
    except:
        failed = True
    suite.check(failed, "a negative position has no slot")

    failed = False
    try:
        _ = c.slot_for(4)
    except:
        failed = True
    suite.check(failed, "and neither does one past the end")


def test_cells(mut suite: Suite) raises:
    """A cell holds a position for a sequence, and gives it back.

    The pool here is four cells so that it can be filled, which is the case
    worth having: a table that never runs out never exercises the wrap in the
    search or the reuse of a cell somebody let go of.
    """
    suite.group("cell table")

    var t = CellTable(4)
    suite.check(
        t.size() == 4 and t.free() == 4 and t.live == 0,
        "a new pool is all free",
    )

    var a = List[Int]()
    t.alloc_run(0, 0, 4, a)
    suite.check(len(a) == 4, "a run of four hands back four cells")
    suite.check(t.live == 4 and t.free() == 0, "and fills the pool")
    suite.check(t.position(a[2]) == 2, "a cell knows the position it holds")
    suite.check(t.owns(a[2], 0), "and the sequence that put it there")
    suite.check(not t.owns(a[2], 1), "and nobody else")
    suite.check(t.cell_of(0, 3) == a[3], "a position finds its cell")
    suite.check(t.held_by(0) == 4, "and the sequence holds all four")

    var full = False
    try:
        _ = t.alloc(0, 4)
    except:
        full = True
    suite.check(full, "a full pool refuses the next token")

    suite.check(t.release(0, 0, 2) == 2, "releasing two positions frees two")
    suite.check(t.live == 2 and t.free() == 2, "and the pool says so")
    suite.check(
        t.position(a[0]) == CELL_FREE and t.position(a[1]) == CELL_FREE,
        "a freed cell holds nothing",
    )
    suite.check(
        t.cell_of(0, 0) == CELL_FREE, "and the position it held is gone"
    )
    suite.check(t.held_by(0) == 2, "while the rest of the sequence stays")

    # The search starts where the last one stopped, which after filling the
    # pool is back at cell zero, so this is the wrap and the reuse at once.
    var again = t.alloc(0, 4)
    suite.check(again == a[0], "the next token takes a cell that was let go")
    suite.check(t.position(again) == 4, "at the position it was asked for")

    var refused = False
    try:
        var b = List[Int]()
        t.alloc_run(1, 0, 2, b)
    except:
        refused = True
    suite.check(refused, "a run that does not fit is refused whole")
    suite.check(
        t.free() == 1 and t.held_by(1) == 0,
        "and leaves nothing of itself behind",
    )

    t.reset()
    suite.check(
        t.free() == 4 and t.held_by(0) == 0, "reset gives every cell back"
    )


def test_cells_sharing(mut suite: Suite) raises:
    """Two sequences on one set of cells, and neither of them copying.

    This is the property the whole cell form is for. A prefix that two requests
    share is one set of cells with two bits set, and the first request to leave
    frees only the part the other one never claimed.
    """
    suite.group("cell sharing")

    var t = CellTable(8)
    var a = List[Int]()
    t.alloc_run(0, 0, 4, a)

    suite.check(t.share(0, 1, 0, -1) == 4, "sharing a whole sequence is four")
    suite.check(t.live == 4, "and costs no cells")
    suite.check(t.held_by(1) == 4, "the second sequence holds the same four")
    suite.check(t.cell_of(1, 2) == a[2], "and reads them at the same position")

    suite.check(t.release(0, 0, -1) == 0, "the first one leaving frees nothing")
    suite.check(
        t.held_by(0) == 0 and t.held_by(1) == 4,
        "but it does stop holding them",
    )
    suite.check(t.release_all(1) == 4, "the last one out frees all four")
    suite.check(t.live == 0, "and the pool is empty")

    # A prefix hit shares part of a sequence rather than all of it, so the
    # range has to be honoured on both ends.
    t.reset()
    var c = List[Int]()
    t.alloc_run(0, 0, 4, c)
    suite.check(t.share(0, 2, 0, 2) == 2, "sharing a prefix takes the prefix")
    suite.check(t.held_by(2) == 2, "and only the prefix")
    suite.check(
        t.cell_of(2, 2) == CELL_FREE, "the part past it belongs to nobody else"
    )
    suite.check(
        t.release(0, 0, -1) == 2,
        "so the owner leaving frees what was not shared",
    )
    suite.check(t.held_by(2) == 2, "and leaves the shared prefix alone")

    # Two sequences can hold the same position in different cells, which is
    # what makes a position a sequence's and not the pool's.
    t.reset()
    var one = t.alloc(0, 0)
    var two = t.alloc(1, 0)
    suite.check(one != two, "two sequences at position zero get two cells")
    suite.check(
        t.position(one) == 0 and t.position(two) == 0,
        "both of which hold position zero",
    )
    suite.check(t.cell_of(1, 0) == two, "and each finds its own")


def test_cells_errors(mut suite: Suite) raises:
    suite.group("cell table errors")

    var failed = False
    try:
        _ = CellTable(0)
    except:
        failed = True
    suite.check(failed, "a pool with no cells is refused")

    var t = CellTable(4)
    failed = False
    try:
        _ = t.alloc(MAX_SEQS, 0)
    except:
        failed = True
    suite.check(failed, "a sequence the owner word cannot hold is refused")

    suite.check(
        t.alloc(MAX_SEQS - 1, 0) >= 0, "the last one it can hold is not"
    )

    failed = False
    try:
        _ = t.alloc(-1, 0)
    except:
        failed = True
    suite.check(failed, "and neither is a negative sequence")

    failed = False
    try:
        _ = t.alloc(0, -1)
    except:
        failed = True
    suite.check(failed, "a negative position has no cell")

    failed = False
    try:
        _ = t.position(4)
    except:
        failed = True
    suite.check(failed, "a cell past the end of the pool is refused")

    failed = False
    try:
        _ = t.owns(-1, 0)
    except:
        failed = True
    suite.check(failed, "and so is a negative one")


def test_session_step(mut suite: Suite) raises:
    suite.group("session step")

    var arena = Arena()
    var b = _bound(arena)
    var s = Session(b, 16)
    suite.check(s.pos == 0, "a new session is at position zero")
    suite.check(
        s.logits.elements() == VOCAB,
        "and its logits are as wide as the vocabulary",
    )

    s.step(b, 1)
    suite.check(s.pos == 1, "a step consumes a position")
    suite.check(s.cache.filled == 1, "and writes one into the cache")

    var wrote = False
    for i in range(WIDTH):
        if s.cache.keys[0][i] != 0.0:
            wrote = True
    suite.check(wrote, "the first layer's key for position zero is not empty")

    var second_untouched = True
    for i in range(WIDTH):
        if s.cache.keys[0][WIDTH + i] != 0.0:
            second_untouched = False
    suite.check(second_untouched, "and slot one has not been written yet")

    var first = List[Float32]()
    for i in range(WIDTH):
        first.append(s.cache.keys[0][i])

    s.step(b, 2)
    suite.check(s.pos == 2, "a second step consumes another")

    var kept = True
    for i in range(WIDTH):
        if s.cache.keys[0][i] != first[i]:
            kept = False
    suite.check(kept, "and leaves the first token's key exactly as it was")

    var moved = False
    for i in range(WIDTH):
        if s.cache.keys[0][WIDTH + i] != 0.0:
            moved = True
    suite.check(moved, "while writing its own into the next slot")

    keep(arena)


def test_prefill_matches_decode(mut suite: Suite) raises:
    suite.group("prefill matches decode")

    var arena = Arena()
    var b = _bound(arena)

    var prompt = List[Int]()
    prompt.append(1)
    prompt.append(2)
    prompt.append(0)

    var bulk = Session(b, 16)
    bulk.prefill(b, prompt)

    var one_at_a_time = Session(b, 16)
    for i in range(len(prompt)):
        one_at_a_time.step(b, prompt[i])

    suite.check(
        bulk.pos == one_at_a_time.pos and bulk.pos == 3,
        "both routes end at the same position",
    )

    var keys_match = True
    var values_match = True
    for l in range(LAYERS):
        for i in range(len(prompt) * WIDTH):
            if bulk.cache.keys[l][i] != one_at_a_time.cache.keys[l][i]:
                keys_match = False
            if bulk.cache.values[l][i] != one_at_a_time.cache.values[l][i]:
                values_match = False
    suite.check(keys_match, "and every cached key is identical, bit for bit")
    suite.check(values_match, "and so is every cached value")

    var logits_match = True
    for i in range(VOCAB):
        if bulk.logits.data[i] != one_at_a_time.logits.data[i]:
            logits_match = False
    suite.check(logits_match, "and the last token's logits agree exactly")

    keep(arena)


def test_generate(mut suite: Suite) raises:
    suite.group("session generate")

    var arena = Arena()
    var b = _bound(arena)

    var prompt = List[Int]()
    prompt.append(1)

    var greedy = Sampler(SamplerConfig(), VOCAB)
    var s = Session(b, 16)
    var out = s.generate(b, greedy, prompt, 4)
    suite.check(len(out) == 4, "four tokens asked for and four came back")
    for i in range(len(out)):
        suite.check(
            out[i] >= 0 and out[i] < VOCAB,
            "every token generated is in the vocabulary",
        )
    suite.check(
        s.pos == len(prompt) + len(out),
        "the prompt and the continuation are both in the cache",
    )

    var second = Sampler(SamplerConfig(), VOCAB)
    var again = Session(b, 16)
    var repeat = again.generate(b, second, prompt, 4)
    var same = len(repeat) == len(out)
    for i in range(len(out)):
        if repeat[i] != out[i]:
            same = False
    suite.check(same, "greedy decoding gives the same answer twice")

    var third = Sampler(SamplerConfig(), VOCAB)
    var stopped = Session(b, 16)
    var early = stopped.generate(b, third, prompt, 4, out[0])
    suite.check(
        len(early) == 0,
        "a stop token that is the first thing picked ends it at once",
    )

    keep(arena)


def test_session_errors(mut suite: Suite) raises:
    suite.group("session errors")

    var arena = Arena()
    var b = _bound(arena)

    var failed = False
    try:
        _ = Session(b, 0)
    except:
        failed = True
    suite.check(failed, "a session with no room is refused")

    failed = False
    try:
        _ = Session(b, 65)
    except:
        failed = True
    suite.check(
        failed, "so is one asking for more context than the file was trained on"
    )

    var s = Session(b, 4)
    var empty = List[Int]()
    failed = False
    try:
        s.prefill(b, empty)
    except:
        failed = True
    suite.check(failed, "a prompt with no tokens has nothing to continue")

    var long = List[Int]()
    for _ in range(5):
        long.append(1)
    failed = False
    try:
        s.prefill(b, long)
    except:
        failed = True
    suite.check(failed, "and a prompt longer than the context is refused")
    suite.check(
        s.pos == 0,
        "before any of it is computed, so the session is still usable",
    )

    keep(arena)
