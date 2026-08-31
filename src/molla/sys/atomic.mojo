"""Atomic integers, and blocks of them that do not share a cache line.

`std.atomic.Atomic` is the arithmetic. What it is not is a place to put the
value. A Mojo local moves, and an atomic that two threads reach through
different copies of the same value is not shared at all, it is two counters
that both look right in a single threaded test. So this module keeps the same
shape as `molla.sys.thread`: the storage is a block this module allocates from
libc and frees itself, and a thread is given the address rather than the value.

`AtomicRef` is that address with the operations on it. It is copyable and
carries no ownership, which is what makes it something a worker can be handed
through the one integer a thread entry function gets.

The padding is the other half. Two counters that are eight bytes apart are one
cache line apart on every machine molla runs on, and two cores incrementing
them take turns owning that line whether or not they ever touch the same
counter. That is false sharing, it costs an order of magnitude, and it does not
show up as anything except the number being lower than it should be. So
`AtomicBlock` puts every counter alone on a 64 byte line, and the address it
allocates is rounded up to a line boundary rather than trusting what calloc
happened to return.

Every operation is sequentially consistent. Acquire and release are what the
queues actually need and would be measurably cheaper on arm64, but Mojo 1.0
exposes no ordering argument, and a fence that is stronger than necessary is a
performance question rather than a correctness one.
"""

from std.atomic import Atomic

from molla.sys.mem import allocate, release

comptime CACHE_LINE = 64
"""One line on x86-64 and on Apple silicon's L1. Apple's L2 line is 128 bytes,
so a pair of counters can still share one there, and doubling this to cover it
would double the memory every queue cell costs for a difference nobody has
measured on this workload."""


struct AtomicRef(Copyable, ImplicitlyCopyable, Movable):
    """One atomic integer, by address.

    A borrow rather than ownership. It never allocates and never frees, and a
    zero address means the block it came from could not be allocated, in which
    case every operation here is a no op that reports zero. Callers check
    `AtomicBlock.is_valid` once at startup instead of on every increment.
    """

    var address: Int

    def __init__(out self, address: Int):
        self.address = address

    def _cell(self) -> Pointer[Atomic[DType.int64], MutAnyOrigin]:
        return Pointer[Atomic[DType.int64], MutAnyOrigin](
            unsafe_from_address=self.address
        )

    def load(self) -> Int:
        if self.address == 0:
            return 0
        return Int(self._cell()[].load())

    def store(self, value: Int):
        if self.address == 0:
            return
        self._cell()[].store(Int64(value))

    def add(self, delta: Int) -> Int:
        """Add and return what was there before, like every fetch_add."""
        if self.address == 0:
            return 0
        return Int(self._cell()[].fetch_add(Int64(delta)))

    def sub(self, delta: Int) -> Int:
        if self.address == 0:
            return 0
        return Int(self._cell()[].fetch_sub(Int64(delta)))

    def compare_exchange(self, expected: Int, desired: Int) -> Bool:
        """Store `desired` if the value is still `expected`."""
        if self.address == 0:
            return False
        var slot = Int64(expected)
        return self._cell()[].compare_exchange(slot, Int64(desired))

    def compare_exchange_update(self, mut expected: Int, desired: Int) -> Bool:
        """Compare and exchange, and on failure say what was there instead.

        The queues need this shape. A producer that loses the race has to try
        again from the position the winner left, and reloading it would be a
        second trip to a line another core has just taken.
        """
        if self.address == 0:
            return False
        var slot = Int64(expected)
        var won = self._cell()[].compare_exchange(slot, Int64(desired))
        expected = Int(slot)
        return won

    def swap(self, value: Int) -> Int:
        """Store and return the old value.

        A compare and exchange loop rather than one instruction, because Mojo
        1.0 has no atomic exchange. Nothing on the request path uses this, so
        the loop is fine where it is."""
        if self.address == 0:
            return 0
        while True:
            var seen = self.load()
            if self.compare_exchange(seen, value):
                return seen


struct AtomicBlock(Movable):
    """`count` atomic integers, each alone on its own cache line.

    Not copyable, because it owns the storage. The address survives a move,
    which is what lets a reactor holding one be appended to a list before the
    threads start.

    Mojo destroys a local at its last use rather than at the end of the scope,
    and handing out an address is not a use it can see. A block whose last
    mention is the line that takes an `AtomicRef` out of it is freed before the
    threads that were given that reference have run, and what they increment is
    whatever the allocator handed out next. `molla.sys.mem.keep` after the
    joins is the fix, and every test here does it."""

    var address: Int
    """What calloc returned, and what has to be freed. Not the address the
    counters are at."""

    var base: Int
    """The first counter, rounded up to a cache line boundary."""

    var count: Int

    def __init__(out self, count: Int):
        self.count = count if count > 0 else 1
        # One extra line so there is room to round up. calloc gives 16 byte
        # alignment on both platforms and molla wants 64.
        self.address = allocate((self.count + 1) * CACHE_LINE)
        if self.address == 0:
            self.base = 0
            return
        self.base = (self.address + CACHE_LINE - 1) & ~(CACHE_LINE - 1)

    def __deinit__(deinit self):
        release(self.address)

    def is_valid(self) -> Bool:
        return self.base != 0

    def address_of(self, index: Int) -> Int:
        """Where counter `index` lives. Zero if there is no such counter, which
        turns into a no op `AtomicRef` rather than a wild write."""
        if self.base == 0 or index < 0 or index >= self.count:
            return 0
        return self.base + index * CACHE_LINE

    def slot(self, index: Int) -> AtomicRef:
        return AtomicRef(self.address_of(index))
