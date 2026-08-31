"""Two bounded queues of integers, neither of which takes a lock.

Both carry an `Int`, which in molla is either a descriptor or the address of
something the sender keeps alive. That is a deliberate floor. A queue of values
means a copy per push and a queue of owned Mojo objects means the ownership
rules have to hold across a thread boundary the compiler cannot see, and every
handoff molla actually has is one machine word.

## MpscQueue, many senders and one receiver

The ticket algorithm, which is Dmitry Vyukov's bounded queue. Each cell carries
a sequence number, and a producer may only write to the cell whose sequence
equals the position it claimed. Claiming is one compare and exchange on the
producer cursor, so N producers make progress without a lock and without any of
them being able to block the others: a producer that is descheduled between
claiming a cell and publishing it holds up only the consumer's view of that one
cell, and only until it runs again.

Full is a return value, not a wait. Every caller here has something better to
do than block, which for the reactor handoff means serving the connection on
the accepting thread instead.

The consumer side is single threaded on purpose. One consumer means the read
cursor is only ever written by the thread that owns the queue, so a pop is two
loads and two stores with no compare and exchange at all. The queue belongs to
one reactor and the reactor is one thread, so that precondition is structural
rather than something a caller has to remember.

## SpscRing, one sender and one receiver

For the path where a producer thread makes tokens and one writer thread frames
them, which is what M2 needs. With exactly one thread on each end there is no
race to lose, so there is no compare and exchange anywhere: the producer owns
the write cursor, the consumer owns the read cursor, and each reads the other's.

## The padding

Every cursor is alone on a cache line, and in the MPSC queue every cell is too.
The cursors are obvious, they are written by different threads on every
operation. The cells are less obvious and matter more here than in the usual
description of this algorithm, because molla's queues run close to empty. A
handoff queue with one item in it has the producer's cell and the consumer's
cell adjacent, so packing cells sixteen bytes apart would put both threads on
one line almost every time the queue is used. The memory is a page or two per
queue and the queue is created once at startup.
"""

from molla.sys.atomic import CACHE_LINE, AtomicBlock, AtomicRef
from molla.sys.mem import allocate, release

comptime VALUE_OFFSET = 8
"""Where a cell's payload sits, eight bytes past its sequence number and on the
same cache line. The sequence number is what makes the payload safe to read
with a plain load, so keeping them together is the point."""

comptime DEFAULT_CAPACITY = 1024


def round_up_pow2(n: Int) -> Int:
    """The next power of two at or above `n`, and at least 2.

    A power of two capacity turns the modulo in every push and pop into one
    AND. Rounding up rather than rejecting, because every caller's number is a
    guess at a good size and none of them care about the exact figure.
    """
    var size = 2
    while size < n:
        size <<= 1
    return size


def _value_ptr(cell: Int) -> Pointer[Int64, MutAnyOrigin]:
    return Pointer[Int64, MutAnyOrigin](unsafe_from_address=cell + VALUE_OFFSET)


struct MpscQueue(Movable):
    """A bounded queue any thread may push to and one thread may pop from."""

    var block: AtomicBlock
    """Two cursor lines followed by one line per cell."""

    var capacity: Int
    var mask: Int

    def __init__(out self, capacity: Int = DEFAULT_CAPACITY):
        self.capacity = round_up_pow2(capacity)
        self.mask = self.capacity - 1
        self.block = AtomicBlock(self.capacity + 2)
        if not self.block.is_valid():
            return
        # Cell i starts with sequence i, which is what makes the first lap of
        # the ring look exactly like every lap after it.
        for i in range(self.capacity):
            self.block.slot(i + 2).store(i)

    def is_valid(self) -> Bool:
        return self.block.is_valid()

    def _head(self) -> AtomicRef:
        """The producer cursor. Every push claims a position here."""
        return self.block.slot(0)

    def _tail(self) -> AtomicRef:
        """The consumer cursor. Written by the owning thread only."""
        return self.block.slot(1)

    def _cell(self, position: Int) -> Int:
        return self.block.address_of((position & self.mask) + 2)

    def push(mut self, value: Int) -> Bool:
        """Add one item. False means the queue is full.

        Safe from any thread. Full is reported rather than waited on, and the
        caller decides what that means.
        """
        if not self.block.is_valid():
            return False
        var pos = self._head().load()
        while True:
            var cell = self._cell(pos)
            var seq = AtomicRef(cell).load()
            var ahead = seq - pos
            if ahead == 0:
                if self._head().compare_exchange_update(pos, pos + 1):
                    _value_ptr(cell).unsafe_store(0, Int64(value))
                    # Publishing the sequence is what makes the value visible.
                    # Nothing reads the payload of a cell whose sequence has
                    # not moved, so this store is the whole handover.
                    AtomicRef(cell).store(pos + 1)
                    return True
                # Lost the race. `pos` now holds what the winner left.
            elif ahead < 0:
                # The cell still belongs to the consumer a lap behind us.
                return False
            else:
                pos = self._head().load()

    def pop(mut self, mut out_value: Int) -> Bool:
        """Take the oldest item. False means there is nothing ready.

        Only the owning thread may call this. False covers both an empty queue
        and a producer that has claimed the next cell and not published it yet,
        and the two are the same thing to a caller: come back later.
        """
        if not self.block.is_valid():
            return False
        var pos = self._tail().load()
        var cell = self._cell(pos)
        if AtomicRef(cell).load() != pos + 1:
            return False
        out_value = Int(_value_ptr(cell).unsafe_load(0))
        self._tail().store(pos + 1)
        # Hand the cell to the producer a lap ahead.
        AtomicRef(cell).store(pos + self.capacity)
        return True

    def depth(self) -> Int:
        """Roughly how many items are waiting.

        Two loads that are not taken together, so under load this is a reading
        rather than a fact. It is here for the SIGQUIT dump and for tests that
        have stopped every producer first, and nothing decides anything on it.
        """
        var depth = self._head().load() - self._tail().load()
        if depth < 0:
            return 0
        if depth > self.capacity:
            return self.capacity
        return depth

    def is_empty(self) -> Bool:
        return self.depth() == 0


struct SpscRing(Movable):
    """A bounded queue with one writing thread and one reading thread."""

    var cursors: AtomicBlock
    """Two lines. The producer owns the first, the consumer owns the second."""

    var slots: Int
    """The payloads, eight bytes each. Packed, because a producer and a
    consumer on this path run far enough apart that the cell they are on is
    rarely the same line, and the ring is sized in tokens rather than in
    connections."""

    var capacity: Int
    var mask: Int

    def __init__(out self, capacity: Int = DEFAULT_CAPACITY):
        self.capacity = round_up_pow2(capacity)
        self.mask = self.capacity - 1
        self.cursors = AtomicBlock(2)
        self.slots = 0
        if not self.cursors.is_valid():
            return
        self.slots = allocate(self.capacity * 8 + CACHE_LINE)

    def __deinit__(deinit self):
        release(self.slots)

    def is_valid(self) -> Bool:
        return self.cursors.is_valid() and self.slots != 0

    def _write(self) -> AtomicRef:
        return self.cursors.slot(0)

    def _read(self) -> AtomicRef:
        return self.cursors.slot(1)

    def _at(self, position: Int) -> Pointer[Int64, MutAnyOrigin]:
        return Pointer[Int64, MutAnyOrigin](
            unsafe_from_address=self.slots + (position & self.mask) * 8
        )

    def push(mut self, value: Int) -> Bool:
        """Add one item. Only the producing thread may call this."""
        if not self.is_valid():
            return False
        var head = self._write().load()
        if head - self._read().load() >= self.capacity:
            return False
        self._at(head).unsafe_store(0, Int64(value))
        # The value goes in before the cursor moves, so a consumer that sees
        # the new cursor sees the value that goes with it.
        self._write().store(head + 1)
        return True

    def pop(mut self, mut out_value: Int) -> Bool:
        """Take one item. Only the consuming thread may call this."""
        if not self.is_valid():
            return False
        var tail = self._read().load()
        if tail == self._write().load():
            return False
        out_value = Int(self._at(tail).unsafe_load(0))
        self._read().store(tail + 1)
        return True

    def depth(self) -> Int:
        var depth = self._write().load() - self._read().load()
        if depth < 0:
            return 0
        return depth

    def is_empty(self) -> Bool:
        return self.depth() == 0

    def is_full(self) -> Bool:
        return self.depth() >= self.capacity
