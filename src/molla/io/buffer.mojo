"""An owned growable byte buffer with a capacity policy you can read.

This is the connection's read buffer and the response's write buffer. It owns
its block, the block does not move when the `Buffer` does, and every span handed
out points into it.

The growth policy is written down rather than emergent, because a server's
memory profile is the sum of its growth policies. Doubling up to 64 kB, then a
fixed 64 kB step after that. Doubling is right while the buffer is small,
because the copies are cheap and the alternative is a realloc per header. It is
wrong once the buffer is large, because doubling a 4 MB buffer asks for 8 MB to
hold 4 MB and one byte, and a server holding a thousand of those is what an
out of memory kill looks like.

A buffer never shrinks on its own. It is per connection and it gets reused
across every request on that connection, so shrinking after a large request just
means growing again for the next one. `reset_to` is there for the connection
pool to call when a socket is recycled, which is the point where the size of the
last request stops predicting anything.

`take` and `commit` are how a read from a socket lands here without a copy: ask
for the writable tail, hand that pointer to `recv`, then commit however many
bytes it actually wrote. The alternative is reading into a stack array and
copying, which is a memcpy per read per connection.
"""

from molla.sys.mem import (
    RawPtr,
    as_ptr,
    copy_bytes,
    counted_allocate,
    counted_release,
    fill_bytes,
)

comptime DOUBLE_UNTIL = 65536
"""Past this, grow by a fixed step instead of doubling."""

comptime GROW_STEP = 65536
comptime MIN_CAPACITY = 64


def _next_capacity(current: Int, needed: Int) -> Int:
    """The policy, in one place, so it can be tested rather than trusted."""
    var size = current
    if size < MIN_CAPACITY:
        size = MIN_CAPACITY
    while size < needed:
        if size < DOUBLE_UNTIL:
            size = size * 2
        else:
            size = size + GROW_STEP
    return size


struct Buffer(Movable):
    """Bytes molla owns, with room to grow at the end."""

    var address: Int
    """The block. Zero means the allocation failed and every write is refused
    rather than writing to a null."""

    var capacity: Int
    var length: Int
    """How much of the block is real data. Everything from here to `capacity` is
    the writable tail."""

    var counter: Int
    """Where to record allocations, or 0 for not counting."""

    def __init__(out self, capacity: Int, counter: Int):
        var want = capacity
        if want < MIN_CAPACITY:
            want = MIN_CAPACITY
        self.address = counted_allocate(want, counter)
        self.capacity = want if self.address != 0 else 0
        self.length = 0
        self.counter = counter

    def __deinit__(deinit self):
        counted_release(self.address, self.counter)

    def is_valid(self) -> Bool:
        """False if the allocation failed. Checked once at setup rather than on
        every write."""
        return self.address != 0

    def available(self) -> Int:
        """Writable bytes at the end before a grow is needed."""
        return self.capacity - self.length

    def base(self) -> Int:
        """The address of byte zero."""
        return self.address

    def tail(self) -> Int:
        """The address to hand to `recv`."""
        return self.address + self.length

    def ptr(self) -> RawPtr:
        return as_ptr(self.address)

    def tail_ptr(self) -> RawPtr:
        return as_ptr(self.address + self.length)

    def bytes(self) -> Span[UInt8, MutAnyOrigin]:
        """Everything written so far, as a span that does not own it."""
        return Span[UInt8, MutAnyOrigin](
            unsafe_ptr=self.ptr(), length=self.length
        )

    def at(self, index: Int) -> UInt8:
        return self.ptr().unsafe_load(index)

    def reserve(mut self, needed: Int) -> Bool:
        """Make room for `needed` more bytes. False if that could not be done.

        This is the only place a `Buffer` allocates after construction, so it is
        the only place the counter moves, which is what makes the zero
        allocation claim in issue #17 checkable."""
        if self.address == 0:
            return False
        if self.length + needed <= self.capacity:
            return True
        var want = _next_capacity(self.capacity, self.length + needed)
        var fresh = counted_allocate(want, self.counter)
        if fresh == 0:
            return False
        copy_bytes(fresh, self.address, self.length)
        counted_release(self.address, self.counter)
        self.address = fresh
        self.capacity = want
        return True

    def commit(mut self, count: Int):
        """Say that `count` bytes of the tail are now real data.

        What a caller does after `recv` wrote straight into the tail. Refuses to
        go past the capacity, because a wrong count here would hand the parser
        bytes nobody wrote."""
        if count <= 0:
            return
        if self.length + count > self.capacity:
            self.length = self.capacity
            return
        self.length += count

    def append(mut self, data: Span[UInt8, _]) -> Bool:
        """Copy bytes onto the end, growing if there is room to."""
        if len(data) == 0:
            return True
        if not self.reserve(len(data)):
            return False
        copy_bytes(self.tail(), Int(data.unsafe_ptr()), len(data))
        self.length += len(data)
        return True

    def append_byte(mut self, value: UInt8) -> Bool:
        if not self.reserve(1):
            return False
        self.tail_ptr().unsafe_store(0, value)
        self.length += 1
        return True

    def append_str(mut self, text: StringSpan) -> Bool:
        """The one that response building actually uses."""
        if text.byte_length() == 0:
            return True
        if not self.reserve(text.byte_length()):
            return False
        copy_bytes(self.tail(), Int(text.unsafe_ptr()), text.byte_length())
        self.length += text.byte_length()
        return True

    def consume(mut self, count: Int):
        """Drop the first `count` bytes and move the rest down.

        This is a memmove and it is on the read path, which is why the parser is
        written to take a whole buffer and report how much it used, so this gets
        called once per request rather than once per header. When the buffer is
        empty it is a length reset and no copy at all, which is the common case
        for a request that arrived in one read."""
        if count <= 0:
            return
        if count >= self.length:
            self.length = 0
            return
        copy_bytes(self.address, self.address + count, self.length - count)
        self.length -= count

    def clear(mut self):
        """Forget the contents. Keeps the block, because the next request on
        this connection is going to want it."""
        self.length = 0

    def reset_to(mut self, capacity: Int) -> Bool:
        """Drop back to a given capacity. For a connection being recycled.

        Refuses to shrink below what is currently in the buffer, since that
        would throw away live bytes to save memory, and returns False so the
        caller knows the size it asked for is not the size it got."""
        self.length = 0
        if capacity >= self.capacity:
            return True
        var want = capacity
        if want < MIN_CAPACITY:
            want = MIN_CAPACITY
        var fresh = counted_allocate(want, self.counter)
        if fresh == 0:
            return False
        counted_release(self.address, self.counter)
        self.address = fresh
        self.capacity = want
        return True

    def zero(mut self):
        """Wipe the whole block. For anything that held a secret."""
        fill_bytes(self.address, 0, self.capacity)
        self.length = 0
