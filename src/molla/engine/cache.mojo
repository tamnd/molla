"""The keys and values one sequence has accumulated.

`molla.nn.attention` reads keys with position on the outside, so token `t`, head
`h`, element `d` sits at `t * kv_heads * head_dim + h * head_dim + d`, and
`molla.nn.model.forward` already takes one flat list per layer plus a slot to
write into. So this is not a new layout. It is the thing that owns those lists
and answers the three questions the forward pass deliberately does not: how long
each one is, which slot a position goes in, and what happens when the context
fills.

One sequence, contiguous, no paging. Paging is M3 and it changes the answer to
the middle question and nothing else, which is why that question is a method
here rather than an expression at the call site. Today `slot_for` returns the
position it was given. The day it stops doing that is the day every caller that
had inlined the identity becomes a bug, and there are no such callers because
there is a method.

This one is float32 and stays there. It is the reference the device cache is
checked against, and a reference that rounds the same way the thing it is
checking rounds has stopped being one. The device cache holds float16 by default
and q8_0 when asked, and what that means to a byte is `CACHE_F16`, `CACHE_Q8` and
`cache_row` in `molla.nn.repack`, beside the weight layout they follow.

There are two caches now, this one and the device one in `molla.engine.device`,
and they differ only in which memory the floats are in. So the two questions
that are policy rather than storage are free functions here and both caches call
them. That is the same argument the paragraph above makes about `slot_for` being
a method, one level up: the day a slot stops being a position, one function
changes and both caches follow, rather than one of them being updated and the
other quietly staying correct for a while.

`CellTable` below is the start of the day that paragraph is about. It is stage
two of #31 and `docs/validation/paging.md` is the argument for its shape: the
pool is addressed by cell rather than by position, a cell holds one token, and
which cell a position went in is host side bookkeeping that no kernel walks.
Nothing calls it yet. It lands first on its own because a free list and an owner
set are the part of paging that can be tested to exhaustion without a card, and
because the stage after it wants to be a change to attention alone.
"""

comptime CELL_FREE = -1
"""The position held by a cell nothing has written.

Negative rather than a separate flag vector, which is what llama.cpp does, for
the same reason: the free question and the position question are then one load
rather than two, and a position can never be negative anyway.
"""

comptime MAX_SEQS = 64
"""How many sequences a pool can hold at once.

llama.cpp carries a `std::bitset<256>` a cell and so allows 256. One machine
word allows 64, costs eight bytes a cell rather than thirty two, and tests with
a shift instead of a loop. M3 asks for sixteen concurrent streams, so 64 is the
honest size until something asks for more, and the day it does this becomes a
short array of words with no caller above it noticing, because no caller above
it sees the word.
"""


def slot_of(pos: Int, context: Int) raises -> Int:
    """Where the key for `pos` goes in a cache with room for `context`.

    The identity, for now. See the module docstring for why it is not written
    out at the call sites.
    """
    if pos < 0:
        raise Error("a position cannot be negative")
    if pos >= context:
        raise Error(
            "position "
            + String(pos)
            + " is past the end of a cache with room for "
            + String(context)
        )
    return pos


def check_room(count: Int, room: Int, context: Int) raises:
    """Refuse a sequence that will not fit, before any of it is computed.

    A refusal and not a wrap. A cache that fills and starts overwriting slot
    zero gives a model that is still fluent and has forgotten its instructions,
    which is worse than an error because nobody reads it as one. What to drop
    when the context is full is a policy, it belongs with paging in M3, and until
    there is one the honest answer is that this sequence does not fit.
    """
    if count < 0:
        raise Error("cannot reserve a negative number of positions")
    if count > room:
        raise Error(
            "this sequence needs "
            + String(count)
            + " more positions and the cache has "
            + String(room)
            + " left of "
            + String(context)
        )


struct CellTable(Movable):
    """Which position each cell of a pool holds, and which sequences own it.

    Two parallel vectors over the pool, host side, never uploaded. A cell is a
    token, so there is no block size to choose, no partial block to copy when
    two sequences diverge, and no table for a kernel to walk. What reaches the
    device is a vector of cell indices computed once a step, and that is a later
    commit in this stage.

    The whole table is small enough not to think about. A pool big enough for
    an 8B to hold 147000 tokens is 147000 positions and 147000 owner words,
    which is 2.4 MiB beside 18 GiB of cells.

    Everything here is a range of positions rather than a range of cells,
    because a caller thinks in positions and only the table knows where they
    went. The one exception is `owns` and `position`, which are how a mask gets
    built and are asked cell by cell.
    """

    var pos: List[Int]
    """`CELL_FREE`, or the position this cell holds in its owners' sequences."""

    var owners: List[UInt64]
    """A bit a sequence. Zero exactly when `pos` is `CELL_FREE`."""

    var live: Int
    """How many cells are not free.

    Kept rather than counted. The fit question is asked once a step and the
    count would be over the whole pool, which is the one loop in here that
    would show up.
    """

    var head: Int
    """Where the next search starts.

    A hint and not state. Every search wraps and scans the whole pool before it
    gives up, so a head pointing at a taken cell costs a few loads and nothing
    else. llama.cpp keeps the same hint for the same reason: allocation is
    almost always at the end of what was allocated last, and starting from zero
    every time turns a decode step into a scan of the pool.
    """

    var top: Int
    """One past the highest live cell, which is how far attention has to read.

    Kept rather than found, because it is asked once a step and finding it is a
    scan back from the end of the pool. Allocation raises it and is the common
    case. A release only lowers it when what it freed was at the top, and then
    the scan back is over the cells that were just freed rather than over the
    pool.
    """

    def __init__(out self, cells: Int) raises:
        """A pool of `cells` cells, all free."""
        if cells <= 0:
            raise Error("a cell table needs at least one cell")
        self.pos = List[Int](length=cells, fill=CELL_FREE)
        self.owners = List[UInt64](length=cells, fill=0)
        self.live = 0
        self.head = 0
        self.top = 0

    def size(self) -> Int:
        return len(self.pos)

    def free(self) -> Int:
        return self.size() - self.live

    def reset(mut self):
        """Give every cell back, forgetting every sequence."""
        for i in range(self.size()):
            self.pos[i] = CELL_FREE
            self.owners[i] = 0
        self.live = 0
        self.head = 0
        self.top = 0

    def window(self, pad: Int) raises -> Int:
        """How many cells a step reads, `top` rounded up to `pad`.

        The rounding is llama.cpp's and the reason it gives is the reason to
        take it: a launch whose grid changes every token is a launch the driver
        cannot reuse anything about, and the cells between `top` and the round
        number are masked off by the window anyway because nothing owns them.
        What it costs is reading up to `pad - 1` cells that say nothing, which
        at a pad of 256 and a context of a few thousand is single digit per
        cent of the attention and buys a constant shape.
        """
        if pad <= 0:
            raise Error("a window has to round up to something positive")
        var n = (self.top + pad - 1) // pad * pad
        if n < pad:
            n = pad
        if n > self.size():
            n = self.size()
        return n

    def held(self, seq: Int, upto: Int, mut out: List[Int]) raises:
        """The position each of the first `upto` cells holds for `seq`.

        `CELL_FREE` for a cell `seq` does not own, whether that is because it is
        free or because it belongs to somebody else, so one vector answers both
        halves of the question attention asks. That is what makes this cheaper
        than the mask llama.cpp builds: it is one entry a cell rather than one
        entry a cell a token, so it does not grow with the batch, and the
        causality that would be baked into a two dimensional mask is a compare
        against the query's own position instead.

        Appended rather than assigned, so a caller building the window for a
        batch of sequences fills one list back to back.
        """
        if upto < 0 or upto > self.size():
            raise Error(
                "a window of "
                + String(upto)
                + " does not fit a pool of "
                + String(self.size())
            )
        var bit = _bit_of(seq)
        for i in range(upto):
            if (self.owners[i] & bit) != 0:
                out.append(self.pos[i])
            else:
                out.append(CELL_FREE)

    def position(self, cell: Int) raises -> Int:
        """What position `cell` holds, or `CELL_FREE`."""
        self._check_cell(cell)
        return self.pos[cell]

    def owns(self, cell: Int, seq: Int) raises -> Bool:
        """Whether `seq` may read `cell`."""
        self._check_cell(cell)
        return (self.owners[cell] & _bit_of(seq)) != 0

    def alloc(mut self, seq: Int, at: Int) raises -> Int:
        """Take a free cell for `seq` at position `at`, and say which one.

        No check that `seq` does not already hold `at`. A sequence that writes
        the same position twice has a bug one level up, and a table that
        searched for a duplicate on every token would be paying for that bug
        once a token forever.
        """
        if at < 0:
            raise Error("a position cannot be negative")
        var bit = _bit_of(seq)
        var n = self.size()
        for i in range(n):
            var cell = self.head + i
            if cell >= n:
                cell -= n
            if self.pos[cell] == CELL_FREE:
                self.pos[cell] = at
                self.owners[cell] = bit
                self.live += 1
                if cell + 1 > self.top:
                    self.top = cell + 1
                self.head = cell + 1 if cell + 1 < n else 0
                return cell
        raise Error("the cell pool is full at " + String(n) + " cells")

    def alloc_run(
        mut self, seq: Int, first: Int, count: Int, mut cells: List[Int]
    ) raises:
        """Take `count` cells for `seq` at `first` onward, appending each one.

        The room check comes first so that a run either happens or does not.
        Half a prompt written into the pool with the other half refused would
        leave cells owned by a sequence that is about to be thrown away, and
        the caller that got the error is the one caller not in a position to
        clean that up.
        """
        check_room(count, self.free(), self.size())
        for i in range(count):
            cells.append(self.alloc(seq, first + i))

    def share(mut self, src: Int, dst: Int, p0: Int, p1: Int) raises -> Int:
        """Give `dst` a second claim on the cells `src` holds in `[p0, p1)`.

        A negative `p1` means the rest of the sequence. Returns how many cells
        changed hands, which is what a caller reports as a prefix hit.

        No copy. That is the whole point of an owner set: two sequences sharing
        a prompt share its cells until one of them writes, and the write goes
        to a new cell because it is at a position neither holds yet.
        """
        var add = _bit_of(dst)
        var keep = _bit_of(src)
        var shared = 0
        for i in range(self.size()):
            if self.pos[i] == CELL_FREE:
                continue
            if (self.owners[i] & keep) == 0:
                continue
            if self.pos[i] < p0:
                continue
            if p1 >= 0 and self.pos[i] >= p1:
                continue
            self.owners[i] |= add
            shared += 1
        return shared

    def release(mut self, seq: Int, p0: Int, p1: Int) raises -> Int:
        """Drop `seq`'s claim on `[p0, p1)`, freeing what nobody else holds.

        A negative `p1` means the rest of the sequence. Returns how many cells
        became free, which is not how many were released, because a cell two
        sequences hold survives the first of them leaving.
        """
        var bit = _bit_of(seq)
        var freed = 0
        for i in range(self.size()):
            if self.pos[i] == CELL_FREE:
                continue
            if (self.owners[i] & bit) == 0:
                continue
            if self.pos[i] < p0:
                continue
            if p1 >= 0 and self.pos[i] >= p1:
                continue
            self.owners[i] &= ~bit
            if self.owners[i] == 0:
                self.pos[i] = CELL_FREE
                self.live -= 1
                freed += 1
        while self.top > 0 and self.pos[self.top - 1] == CELL_FREE:
            self.top -= 1
        return freed

    def release_all(mut self, seq: Int) raises -> Int:
        """Drop every claim `seq` has, which is what ending a session does."""
        return self.release(seq, 0, -1)

    def cell_of(self, seq: Int, at: Int) raises -> Int:
        """Which cell holds `at` for `seq`, or `CELL_FREE`.

        A scan, and it stays one. Nothing on the decode path asks this: a
        sequence keeps the cells it was handed, in order, and asks the table
        only when it wants a position it has not written. This is here for the
        checks that compare two routes position by position, and for a prefix
        lookup that already costs more than a scan.
        """
        var bit = _bit_of(seq)
        for i in range(self.size()):
            if self.pos[i] == at and (self.owners[i] & bit) != 0:
                return i
        return CELL_FREE

    def held_by(self, seq: Int) raises -> Int:
        """How many cells `seq` holds, shared or not."""
        var bit = _bit_of(seq)
        var count = 0
        for i in range(self.size()):
            if (self.owners[i] & bit) != 0:
                count += 1
        return count

    def _check_cell(self, cell: Int) raises:
        if cell < 0 or cell >= self.size():
            raise Error(
                "cell "
                + String(cell)
                + " is outside a pool of "
                + String(self.size())
            )


def _bit_of(seq: Int) raises -> UInt64:
    """The owner bit for a sequence, refusing one the word cannot hold."""
    if seq < 0 or seq >= MAX_SEQS:
        raise Error(
            "sequence "
            + String(seq)
            + " is outside the "
            + String(MAX_SEQS)
            + " a cell pool tracks"
        )
    return UInt64(1) << UInt64(seq)


struct KvCache(Movable):
    """One sequence's keys and values, one list per layer."""

    var keys: List[List[Float32]]
    """`layers` lists of `context * kv_width` floats."""

    var values: List[List[Float32]]
    """The same, and the same length, always."""

    var layers: Int
    var context: Int
    """How many positions there is room for. Not the model's trained context,
    which is an upper bound, but what this session asked for."""

    var kv_width: Int
    """`kv_heads * head_dim`, which is what one position occupies per layer."""

    var filled: Int
    """How many positions have been written. Also the next free slot, until
    something evicts."""

    def __init__(out self, layers: Int, context: Int, kv_width: Int) raises:
        """Allocate the whole thing up front.

        A cache that grows is a cache that reallocates in the middle of a
        decode, and a reallocation of several hundred megabytes between two
        tokens is a stall a user can see. The size is known the moment the
        context length is chosen, so it is taken then.
        """
        if layers <= 0:
            raise Error("a cache needs at least one layer")
        if context <= 0:
            raise Error("a cache needs room for at least one position")
        if kv_width <= 0:
            raise Error("a cache needs a positive key width")
        self.layers = layers
        self.context = context
        self.kv_width = kv_width
        self.filled = 0
        self.keys = List[List[Float32]]()
        self.values = List[List[Float32]]()
        var per = context * kv_width
        for _ in range(layers):
            var k = List[Float32]()
            var v = List[Float32]()
            for _ in range(per):
                k.append(0.0)
                v.append(0.0)
            self.keys.append(k^)
            self.values.append(v^)

    def bytes(self) -> Int:
        """What this occupies, which is worth reporting before allocating it."""
        return 2 * self.layers * self.context * self.kv_width * 4

    def reset(mut self):
        """Forget the sequence without giving back the memory.

        The stale floats are left where they are. Nothing reads past `filled`,
        and zeroing several hundred megabytes to make the unread bytes tidier
        is work done for an invariant that is already held elsewhere.
        """
        self.filled = 0

    def slot_for(self, pos: Int) raises -> Int:
        """Where the key for `pos` goes."""
        return slot_of(pos, self.context)

    def room(self) -> Int:
        return self.context - self.filled

    def reserve(mut self, count: Int) raises:
        """Refuse a sequence that will not fit, before any of it is computed."""
        check_room(count, self.room(), self.context)

    def advance(mut self, count: Int = 1) raises:
        """Record that `count` more positions have been written."""
        self.reserve(count)
        self.filled += count
