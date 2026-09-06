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

`CellTable` below is the start of the day that paragraph is about. It is #31 and
`docs/validation/paging.md` is the argument for its shape: the pool is addressed
by cell rather than by position, a cell holds one token, and which cell a
position went in is host side bookkeeping that no kernel walks. It holds the
free list, the owner set, the window a step reads, the bounded ring a window
model needs, and the eviction order. Nothing calls it yet, and that is the last
thing #31 wants: a session still reserves its whole context and `slot_for` is
still the identity, so the wiring is where #32 starts.
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

    var running: UInt64
    """A bit a sequence that still needs the cells it holds.

    A sequence that has finished its turn is retired rather than released, so
    its cells stay where they are and the next request that starts with the
    same tokens can share them instead of computing them again. A cell no
    running sequence owns is what #31 calls unreferenced, and it is what
    `evict` is allowed to take.
    """

    var stamp: List[Int]
    """When each sequence last touched the pool, on the `clock` below.

    One entry a sequence rather than one a cell, which is the whole reason
    eviction here is cheap. A retired sequence's cells were all written for the
    same turn and are all worth the same to the request after it, so ordering
    them individually would be an ordering over 147000 numbers to answer a
    question that has 64 possible answers.

    Zero means never touched, so a stamp of zero is skipped rather than sorted
    first. `clock` is incremented before it is read, so a real stamp is never
    zero.
    """

    var clock: Int
    """Counts touches, so that `stamp` has an order.

    Not a time. A wall clock would make eviction depend on how fast the machine
    is, and the only question being asked is which of two sequences was used
    more recently.
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
        self.running = 0
        self.stamp = List[Int](length=MAX_SEQS, fill=0)
        self.clock = 0
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
        for s in range(MAX_SEQS):
            self.stamp[s] = 0
        self.running = 0
        self.clock = 0
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

        Allocating counts as running and as a touch, so a caller never has to
        say that a sequence it is writing into is alive.
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
                self._touch(seq)
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

        Sharing counts as running and as a touch for `dst`, the same way
        allocating does, and it does so whether or not any cell changed hands.
        A caller that asks for a prefix is telling the table that `dst` is a
        sequence it is about to use, and whether the prefix was there is the
        answer rather than part of the question.
        """
        var add = _bit_of(dst)
        var keep = _bit_of(src)
        self._touch(dst)
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
            if self._drop(i, bit):
                freed += 1
        self._settle()
        return freed

    def release_all(mut self, seq: Int) raises -> Int:
        """Drop every claim `seq` has, and stop counting it as running.

        What throwing a session away does, as opposed to `retire`, which is
        what finishing a turn does. The difference is whether the cells are
        kept for somebody else to find.

        The stamp goes back to zero with the claims, because a sequence that
        holds nothing is not a sequence eviction should ever look at again, and
        zero is how `stamp` says never touched.
        """
        self.running &= ~_bit_of(seq)
        self.stamp[seq] = 0
        return self.release(seq, 0, -1)

    def trim(
        mut self, seq: Int, pos: Int, window: Int, sinks: Int
    ) raises -> Int:
        """Free what a query at `pos` will never read again, and say how many.

        The bounded ring. A window model's context stops growing here rather
        than in the kernel: the cells that have fallen out behind the window
        are dropped as the sequence walks past them, so a conversation of any
        length holds `window + sinks` cells and a long chat costs constant KV.
        The sinks are pinned by the same expression that makes attention read
        them, which is that a position below `sinks` is visible forever.

        The condition is `molla.nn.attention.AttnSpec.sees` negated term for
        term, and it is written that way on purpose. Every cell this frees is a
        cell attention would have masked, so trimming can change how much
        memory a sequence holds and cannot change a logit. The two have to
        agree, and the way to keep them agreeing is for one of them to be the
        other one read backwards.

        A model with no window trims nothing, because every position it has
        ever held stays visible, and answering that here means a caller does
        not have to ask whether its model has one.
        """
        if window <= 0:
            return 0
        if sinks < 0:
            raise Error("a sink count cannot be negative")
        var bit = _bit_of(seq)
        var freed = 0
        for i in range(self.size()):
            var held = self.pos[i]
            if held == CELL_FREE:
                continue
            if (self.owners[i] & bit) == 0:
                continue
            if held < sinks or held > pos - window:
                continue
            if self._drop(i, bit):
                freed += 1
        self._settle()
        return freed

    def retire(mut self, seq: Int) raises:
        """Stop counting `seq` as running, without giving its cells back.

        What a session does when a turn ends. The cells stay exactly as they
        are, holding the positions they held, so the next request that begins
        with the same tokens can take them with `share` rather than computing
        them again. That is the whole reason this is not `release_all`, and it
        is what #33's prefix cache is going to be built out of.

        Until somebody shares them they are the eviction budget. A pool under
        pressure takes them back in the order their sequences stopped being
        used, which is why retiring is a touch: the stamp a retired sequence
        carries is the moment it finished.
        """
        var bit = _bit_of(seq)
        self._touch(seq)
        self.running &= ~bit

    def resume(mut self, seq: Int) raises:
        """Count `seq` as running again, which a new turn on it does.

        Whatever `evict` already took is gone, so a caller that resumes asks
        the table what is left rather than assuming its cells survived.
        `held_by` is that question and `cell_of` is the finer one.
        """
        self._touch(seq)

    def unreferenced(self) -> Int:
        """How many live cells no running sequence holds.

        The eviction budget. Reported beside `free` rather than folded into it,
        because the difference matters to a caller: a free cell costs nothing
        and one of these costs a prefix that somebody may still come back for.
        """
        var n = 0
        for i in range(self.size()):
            if self.pos[i] == CELL_FREE:
                continue
            if (self.owners[i] & self.running) == 0:
                n += 1
        return n

    def evict(mut self, need: Int) raises -> Int:
        """Take back retired cells until `need` are free, and say how many.

        Least recently used, over sequences rather than over cells. A retired
        sequence's cells were all written for one turn and are worth the same
        to the request after it, so the whole sequence goes at once and the
        order is the order the sequences stopped. Picking cell by cell would be
        an ordering over the pool to answer a question with 64 answers.

        Nothing running is touched. Preempting a sequence that is mid stream is
        a scheduler's decision and not a table's, and when the scheduler makes
        it the way it says so is `release_all` and then recomputing, which is
        what #31 means by preferring recompute over swapping to host memory.
        There is no path here that moves a cell off the device.

        Returns what it freed, which can be less than was asked for. A caller
        that still does not fit gets its refusal from `check_room` or from
        `alloc`, in the same words it would have got them before, rather than
        from here.

        The loop cannot spin. A victim is released whole, which zeroes its
        stamp, and a zero stamp is not a candidate, so each turn of the loop
        either frees cells or takes a sequence out of the running. A retired
        sequence whose cells are all shared with one that is running frees
        nothing and is exactly that second case.
        """
        if need < 0:
            raise Error("cannot ask for a negative number of free cells")
        var freed = 0
        while self.free() < need:
            var victim = -1
            var oldest = 0
            for s in range(MAX_SEQS):
                if (self.running & (UInt64(1) << UInt64(s))) != 0:
                    continue
                if self.stamp[s] == 0:
                    continue
                if victim < 0 or self.stamp[s] < oldest:
                    victim = s
                    oldest = self.stamp[s]
            if victim < 0:
                return freed
            freed += self.release_all(victim)
        return freed

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

    def _touch(mut self, seq: Int) raises:
        """Mark `seq` running and record that it used the pool just now."""
        self.running |= _bit_of(seq)
        self.clock += 1
        self.stamp[seq] = self.clock

    def _drop(mut self, cell: Int, bit: UInt64) -> Bool:
        """Clear one owner of `cell`, saying whether that freed it.

        The one place a cell goes back on the free list, so that `release` and
        `trim` cannot disagree about what freeing means. They disagree about
        which cells to free, which is the whole difference between them.
        """
        self.owners[cell] &= ~bit
        if self.owners[cell] != 0:
            return False
        self.pos[cell] = CELL_FREE
        self.live -= 1
        return True

    def _settle(mut self):
        """Pull the frontier back over cells that were just freed."""
        while self.top > 0 and self.pos[self.top - 1] == CELL_FREE:
            self.top -= 1

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
