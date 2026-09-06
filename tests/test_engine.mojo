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
    test_cells_window(suite)
    test_cells_ring(suite)
    test_cells_eviction(suite)
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


def test_cells_window(mut suite: Suite) raises:
    """How far attention reads, and what it finds when it gets there.

    The window is the one number a step has to get right for the mask to be
    cheap. Too short and a sequence loses the tail of its own context silently,
    too long and every token pays for cells that hold nothing.
    """
    suite.group("cell window")

    var t = CellTable(16)
    suite.check(t.top == 0, "an empty pool has nothing to read")
    suite.check(t.window(4) == 4, "and a window still rounds up to the pad")

    var a = List[Int]()
    t.alloc_run(0, 0, 5, a)
    suite.check(t.top == 5, "the frontier follows the highest cell taken")
    suite.check(t.window(4) == 8, "and the window rounds it up")
    suite.check(t.window(1) == 5, "a pad of one is the frontier itself")
    suite.check(t.window(32) == 16, "and a pad past the pool is the pool")

    # A release at the top pulls the frontier back with it, which is what keeps
    # a finished conversation from being read forever by the ones after it.
    suite.check(t.release(0, 3, -1) == 2, "releasing the tail frees two")
    suite.check(t.top == 3, "and the frontier comes back")
    suite.check(t.release(0, 0, 1) == 1, "releasing the front frees one")
    suite.check(
        t.top == 3, "and leaves the frontier where the highest cell still is"
    )

    t.reset()
    var b = List[Int]()
    t.alloc_run(0, 0, 3, b)
    var c = List[Int]()
    t.alloc_run(1, 0, 2, c)

    # The list is the whole pool and a step fills the prefix it is going to
    # read, so the tail has to come back untouched. That is what lets one
    # buffer be allocated at startup and written again every step.
    var win = List[Int32](length=t.size(), fill=Int32(99))
    t.held(0, t.top, win)
    suite.check(
        Int(win[0]) == 0 and Int(win[1]) == 1 and Int(win[2]) == 2,
        "each of a sequence's cells says which position it holds",
    )
    suite.check(
        Int(win[3]) == CELL_FREE and Int(win[4]) == CELL_FREE,
        "and another sequence's cells say nothing at all",
    )
    suite.check(Int(win[5]) == 99, "and nothing past the window is written")

    var other = List[Int32](length=t.size(), fill=Int32(0))
    t.held(1, t.top, other)
    suite.check(
        Int(other[3]) == 0
        and Int(other[4]) == 1
        and Int(other[0]) == CELL_FREE,
        "the same cells read the other way round for the other sequence",
    )

    # Two windows back to back in one list, which is how a batch of sequences
    # gets to share the buffer.
    var both = List[Int32](length=2 * t.top, fill=Int32(0))
    t.held(0, t.top, both)
    t.held(1, t.top, both, t.top)
    suite.check(
        Int(both[0]) == 0 and Int(both[t.top + 3]) == 0,
        "an offset puts the second window after the first",
    )

    # A shared prefix is in both windows at once, which is the property the
    # whole thing exists for and the one a copy would have hidden.
    _ = t.share(0, 2, 0, 2)
    var shared = List[Int32](length=t.size(), fill=Int32(0))
    t.held(2, t.top, shared)
    suite.check(
        Int(shared[0]) == 0
        and Int(shared[1]) == 1
        and Int(shared[2]) == CELL_FREE,
        "a shared prefix is in the sharer's window and the rest is not",
    )

    var failed = False
    try:
        var over = List[Int32](length=17, fill=Int32(0))
        t.held(0, 17, over)
    except:
        failed = True
    suite.check(failed, "a window past the end of the pool is refused")

    var cramped = False
    try:
        var small = List[Int32](length=4, fill=Int32(0))
        t.held(0, 5, small)
    except:
        cramped = True
    suite.check(cramped, "and a window that does not fit the list it is given")


def test_cells_ring(mut suite: Suite) raises:
    """A window model gives back what it has walked past.

    The property worth pinning is not that some number of cells was freed. It
    is that trimming and masking agree, because a cell freed here that
    attention would still have read is context lost in silence.
    """
    suite.group("cell ring")

    var t = CellTable(16)
    var a = List[Int]()
    t.alloc_run(0, 0, 8, a)

    suite.check(t.trim(0, 7, 0, 0) == 0, "a model with no window trims nothing")
    suite.check(t.held_by(0) == 8, "and keeps everything it has")

    suite.check(
        t.trim(0, 7, 4, 0) == 4, "a window of four drops the other four"
    )
    suite.check(t.held_by(0) == 4, "leaving the window itself")
    suite.check(
        t.cell_of(0, 4) != CELL_FREE and t.cell_of(0, 3) == CELL_FREE,
        "and the boundary is the position the window starts at",
    )

    # What the kernel would have read, asked of every cell that is left. The
    # spec is the same one attention masks with, so this is the two sides of
    # the same condition meeting.
    var spec = AttnSpec(1, 1, 4)
    spec.window = 4
    var agreed = True
    for p in range(8):
        var kept = t.cell_of(0, p) != CELL_FREE
        if kept != spec.sees(p, 7):
            agreed = False
    suite.check(agreed, "what is kept is exactly what a query at seven sees")

    t.reset()
    var b = List[Int]()
    t.alloc_run(0, 0, 8, b)
    suite.check(t.trim(0, 7, 4, 2) == 2, "sinks are pinned against the window")
    suite.check(
        t.cell_of(0, 0) != CELL_FREE and t.cell_of(0, 1) != CELL_FREE,
        "so the first positions stay whatever the window says",
    )
    suite.check(
        t.held_by(0) == 6, "and the sequence holds the window plus them"
    )

    # A sequence longer than the pool, one token at a time, which is the case
    # the ring exists for. It has to run to the end without filling up.
    t.reset()
    var live_max = 0
    for p in range(100):
        _ = t.alloc(0, p)
        _ = t.trim(0, p, 8, 2)
        if t.live > live_max:
            live_max = t.live
    suite.check(
        live_max <= 10, "a hundred tokens through a window of eight hold ten"
    )
    suite.check(
        t.cell_of(0, 0) != CELL_FREE and t.cell_of(0, 99) != CELL_FREE,
        "the sink and the newest token, at the two ends of the conversation",
    )

    # Another sequence's cells are not this one's to give back.
    t.reset()
    var c = List[Int]()
    t.alloc_run(0, 0, 4, c)
    var d = List[Int]()
    t.alloc_run(1, 0, 4, d)
    suite.check(t.trim(0, 3, 2, 0) == 2, "trimming one sequence frees its own")
    suite.check(t.held_by(1) == 4, "and leaves the other one alone")


def test_cells_eviction(mut suite: Suite) raises:
    """A finished turn keeps its cells until somebody needs them.

    Retiring rather than releasing is what makes a prefix worth looking for
    later, and eviction is the price of that: the pool hands the oldest one
    back when the next request does not fit.
    """
    suite.group("cell eviction")

    var t = CellTable(8)
    var a = List[Int]()
    t.alloc_run(0, 0, 4, a)
    suite.check(t.unreferenced() == 0, "a running sequence holds its own cells")

    t.retire(0)
    suite.check(
        t.unreferenced() == 4, "and a retired one holds them for anybody"
    )
    suite.check(t.live == 4 and t.free() == 4, "without giving anything back")
    suite.check(
        t.cell_of(0, 2) == a[2], "the positions are still where they were"
    )

    var b = List[Int]()
    t.alloc_run(1, 0, 4, b)
    t.retire(1)
    suite.check(t.free() == 0, "two retired turns can fill the pool")

    suite.check(t.evict(3) == 4, "evicting takes back the whole oldest turn")
    suite.check(
        t.held_by(0) == 0 and t.held_by(1) == 4,
        "the one that finished first, and only it",
    )
    suite.check(
        t.evict(4) == 0, "and a second ask that fits already does nothing"
    )

    # Running sequences are not eviction's to take, whatever the pressure.
    t.reset()
    var c = List[Int]()
    t.alloc_run(0, 0, 4, c)
    var d = List[Int]()
    t.alloc_run(1, 0, 4, d)
    t.retire(1)
    suite.check(t.evict(8) == 4, "eviction frees what it can")
    suite.check(t.held_by(0) == 4, "and stops at the sequence still running")
    suite.check(t.free() == 4, "which leaves the pool short of what was asked")

    # A retired sequence sharing with a running one is not free to take either,
    # and asking costs the ask rather than the cells.
    t.reset()
    var e = List[Int]()
    t.alloc_run(0, 0, 4, e)
    _ = t.share(0, 1, 0, -1)
    t.retire(0)
    suite.check(t.unreferenced() == 0, "a shared prefix is still referenced")
    suite.check(t.evict(8) == 0, "so eviction has nothing to take")
    suite.check(t.held_by(1) == 4, "and the sequence reading it keeps it")

    # Least recently used, which is the order turns finished in rather than the
    # order they started.
    t.reset()
    var f = List[Int]()
    t.alloc_run(0, 0, 2, f)
    var g = List[Int]()
    t.alloc_run(1, 0, 2, g)
    var h = List[Int]()
    t.alloc_run(2, 0, 2, h)
    t.retire(1)
    t.retire(2)
    t.retire(0)
    suite.check(t.evict(3) == 2, "the first turn to finish goes first")
    suite.check(t.held_by(1) == 0, "which is the one retired first")
    suite.check(
        t.held_by(2) == 2 and t.held_by(0) == 2, "and the rest are untouched"
    )

    # Coming back to a retired sequence puts it out of eviction's reach.
    t.reset()
    var i = List[Int]()
    t.alloc_run(0, 0, 4, i)
    t.retire(0)
    t.resume(0)
    suite.check(t.unreferenced() == 0, "resuming a turn references it again")
    suite.check(t.evict(8) == 0, "and eviction leaves it alone")


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
