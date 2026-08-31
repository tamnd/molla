"""A bump allocator with a per request lifetime.

Everything a request needs that is not a slice into the read buffer goes here:
the header table, the decoded path, the small working strings the router builds.
Allocation is a bounds check and an add. Freeing is setting one integer back to
zero, which costs the same whether the request touched ten objects or ten
thousand.

There is no free for a single object and there will not be one. An arena that
can free one thing is a heap with worse fragmentation. The rule that makes it
safe is that the arena outlives everything pointing into it, and the lifetime is
one request, which is short enough to hold in your head.

Alignment is explicit rather than inferred, because Mojo cannot see the type
being stored behind an address. Ask for what the target needs. Getting it wrong
is a fault on arm64 for anything atomic, and merely slow elsewhere, which is the
kind of bug that only shows up on the platform you did not test on.

When the block is full, `alloc` returns 0 and does not grow. That is the honest
answer for a per request budget: growing means a request can take the process
down by asking for more, and a chain of blocks means the constant time reset
stops being constant time. The caller falls back to the heap and the counter
records it, which is exactly the signal issue #17 is looking for.

The high water mark is the number that tells you what the budget should be. It
is the largest the arena ever got across every request that used it, so a value
close to the capacity means the next unusual request will spill.
"""

from molla.sys.mem import (
    RawPtr,
    as_ptr,
    counted_allocate,
    counted_release,
    fill_bytes,
)

comptime DEFAULT_ARENA = 65536
"""64 kB per request. Enough for a large header table and the strings around it,
small enough that a thousand concurrent requests is 64 MB rather than a
surprise."""


struct Arena(Movable):
    """One block, an offset, and a reset."""

    var address: Int
    var capacity: Int
    var offset: Int
    var high_water: Int
    """The largest `offset` ever reached, across resets. What a budget is set
    from."""

    var spills: Int
    """How many allocations were refused because the block was full. Not zero
    means the budget is too small, and a caller fell back to the heap."""

    var counter: Int

    def __init__(out self, capacity: Int, counter: Int):
        var want = capacity if capacity > 0 else DEFAULT_ARENA
        self.address = counted_allocate(want, counter)
        self.capacity = want if self.address != 0 else 0
        self.offset = 0
        self.high_water = 0
        self.spills = 0
        self.counter = counter

    def __deinit__(deinit self):
        counted_release(self.address, self.counter)

    def is_valid(self) -> Bool:
        return self.address != 0

    def used(self) -> Int:
        return self.offset

    def available(self) -> Int:
        return self.capacity - self.offset

    def alloc(mut self, size: Int, align: Int) -> Int:
        """`size` bytes aligned to `align`, or 0 if the block is full.

        Zero is a real answer and every caller has to handle it. It is not an
        error the process should die from: a request that needs more than its
        budget is a request, not a bug."""
        if self.address == 0 or size <= 0:
            return 0
        var step = align if align > 0 else 1
        var start = self.offset
        var slack = start % step
        if slack != 0:
            start += step - slack
        if start + size > self.capacity:
            self.spills += 1
            return 0
        self.offset = start + size
        if self.offset > self.high_water:
            self.high_water = self.offset
        return self.address + start

    def alloc_bytes(mut self, size: Int) -> Int:
        """Unaligned bytes. What strings and byte buffers want."""
        return self.alloc(size, 1)

    def alloc_ints(mut self, count: Int) -> Int:
        """Room for `count` machine integers, aligned for them."""
        return self.alloc(count * 8, 8)

    def reset(mut self):
        """Free everything at once.

        Does not zero the block. Zeroing 64 kB per request to hide a use after
        free would cost more than the bug does, and the bug is prevented by the
        lifetime rule rather than by scrubbing. `reset_zeroed` is there for the
        arena that held a token."""
        self.offset = 0

    def reset_zeroed(mut self):
        """Same, wiping what was there. For anything that held a secret."""
        fill_bytes(self.address, 0, self.offset)
        self.offset = 0

    def ptr(self, address: Int) -> RawPtr:
        """An address from `alloc`, as a byte pointer."""
        return as_ptr(address)

    def contains(self, address: Int) -> Bool:
        """Whether an address came from this arena. For assertions in tests,
        and for the debug check that a span outliving its arena is caught near
        where it happened rather than three layers up."""
        return (
            self.address != 0
            and address >= self.address
            and address < self.address + self.capacity
        )
