"""Raw allocation, and the counter that proves the request path does not use it.

Everything above this asks for memory here, which is the only reason molla can
answer the question "did that request allocate" with a number instead of an
opinion. Mojo's own allocations do not go through this and cannot be counted,
so what this measures is molla's own heap traffic, which is the part molla can
do anything about.

The counter is a struct rather than a global because Mojo 1.0 has no globals at
all. It gets passed by address to whoever needs to record against it, and zero
means nobody is counting, which is the release build.

`calloc` rather than `malloc` throughout. A buffer full of whatever the last
owner left is a bad thing to hand to a parser, and the cost of zeroing shows up
once per allocation rather than once per request, because these blocks are
allocated at startup and reused.
"""

from std.ffi import c_size_t, external_call
from std.memory import unsafe_memcpy, unsafe_memset

comptime RawPtr = Pointer[UInt8, MutAnyOrigin]

comptime COUNTER_SIZE = 24
"""Three integers: live allocations, total allocations, total bytes."""

comptime OFF_LIVE = 0
comptime OFF_TOTAL = 1
comptime OFF_BYTES = 2


def allocate(size: Int) -> Int:
    """A zeroed block of `size` bytes, or 0 if there is no memory.

    Returns an address rather than a pointer on purpose. `Pointer` is non
    nullable, so a failed allocation has nowhere to go in the type, and every
    caller here has to handle failure anyway."""
    if size <= 0:
        return 0
    return Int(external_call["calloc", Int](c_size_t(1), c_size_t(size)))


def release(address: Int):
    """Give a block back. Freeing 0 is defined and does nothing."""
    external_call["free", NoneType](address)


def as_ptr(address: Int) -> RawPtr:
    """An address as a byte pointer. Only ever called on a checked address."""
    return RawPtr(unsafe_from_address=address)


def keep[T: AnyType](ref value: T):
    """Count as a use of `value`, so it is not destroyed before this point.

    Mojo destroys a local at its last use, not at the end of the scope. That is
    usually what you want and is a problem for everything in this codebase that
    holds an address rather than a reference: once a buffer's address has been
    handed to a parser, the buffer itself looks dead to the compiler, so it is
    freed while the parser is still reading it.

    A field of a longer lived struct has no such problem, which is why this
    almost never comes up in the server and comes up immediately in a test that
    keeps its buffer in a local. Put `keep(buf)` after the last read of anything
    pointing into `buf` and the lifetime covers the whole parse.

    It was found by running the JSON suite on x86_64 Linux, where the freed
    block gets reused straight away. On the M4 the same code passed, because
    the block still held the old bytes.
    """
    pass


def copy_bytes(dest: Int, src: Int, count: Int):
    """Copy `count` bytes. The regions must not overlap, which is true of every
    caller here: buffers copy into their own tail from somewhere else, and the
    ring never copies at all."""
    if count <= 0:
        return
    unsafe_memcpy(dest=as_ptr(dest), src=as_ptr(src), count=count)


def fill_bytes(dest: Int, value: UInt8, count: Int):
    """Set `count` bytes to `value`."""
    if count <= 0:
        return
    unsafe_memset(as_ptr(dest), value, count)


struct AllocCounter(Copyable, ImplicitlyCopyable, Movable):
    """Three numbers in a block this struct allocates, addressed by whoever
    needs it.

    A `Buffer` or an `Arena` holds the address, not the counter, so counting
    survives the owner being moved into a list and does not need the borrow
    checker to agree that a shared mutable counter is reasonable.

    There is no destructor here and that is deliberate, so here is the reason
    before someone helpfully adds one. Mojo destroys a value at its last use,
    not at the end of the scope. The last use of a counter is usually the last
    time a test or a log line reads `total()`, which happens while the buffers
    holding its address are still alive and still counting. A destructor would
    free the block right there, and the next allocation would add three integers
    to freed memory, and malloc would abort in whatever unrelated code allocated
    next. That is exactly the crash this shape produced.

    So the block is freed by `close`, called by the owner once everything it
    counts is gone. Forgetting it leaks twenty four bytes, once, in a debug
    build. That is the cheaper mistake of the two."""

    var address: Int

    def __init__(out self):
        self.address = allocate(COUNTER_SIZE)

    def close(mut self):
        """Give the block back. Nothing may hold the address after this, which
        in practice means calling it last."""
        release(self.address)
        self.address = 0

    def raw(self) -> Int:
        """The address to hand to a buffer or an arena. Zero if the allocation
        failed, which switches counting off rather than crashing."""
        return self.address

    def live(self) -> Int:
        """Blocks allocated and not yet released."""
        return _counter_read(self.address, OFF_LIVE)

    def total(self) -> Int:
        """Blocks allocated since the last reset. This is the number issue #17
        asserts is unchanged across a request."""
        return _counter_read(self.address, OFF_TOTAL)

    def bytes(self) -> Int:
        """Bytes allocated since the last reset."""
        return _counter_read(self.address, OFF_BYTES)

    def reset(mut self):
        """Zero the totals. Live count goes too, so reset while blocks are out
        makes the live number meaningless. Call it at a quiet point."""
        fill_bytes(self.address, 0, COUNTER_SIZE)


def _counter_read(address: Int, slot: Int) -> Int:
    if address == 0:
        return 0
    return Int(
        Pointer[Int, MutAnyOrigin](unsafe_from_address=address).unsafe_load(
            slot
        )
    )


def _counter_add(address: Int, slot: Int, delta: Int):
    if address == 0:
        return
    var p = Pointer[Int, MutAnyOrigin](unsafe_from_address=address)
    p.unsafe_store(slot, p.unsafe_load(slot) + delta)


def counted_allocate(size: Int, counter: Int) -> Int:
    """Allocate and record it against a counter, if there is one.

    Not atomic. Two threads counting against one counter will lose updates,
    which is the right trade for a debug counter: making it atomic would put a
    lock on the allocation path to measure a path that is not supposed to be
    allocating at all. Each connection gets its own counter."""
    var address = allocate(size)
    if address != 0:
        _counter_add(counter, OFF_LIVE, 1)
        _counter_add(counter, OFF_TOTAL, 1)
        _counter_add(counter, OFF_BYTES, size)
    return address


def counted_release(address: Int, counter: Int):
    """Release and decrement the live count."""
    if address != 0:
        _counter_add(counter, OFF_LIVE, -1)
    release(address)
