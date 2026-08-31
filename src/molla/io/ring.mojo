"""The per connection output ring, so a partial write never costs a memmove.

The problem this solves is small and it happens on every slow connection. The
kernel takes 8 kB of a 40 kB response and says it is full. With a plain buffer
the remaining 32 kB has to move to the front before the next write, or the read
offset has to be tracked and the buffer has to grow forever. Thirty two
kilobytes of memmove, per short write, per connection, is real work to do
nothing.

A ring has no front to move to. The read offset walks forward, the write offset
walks forward, and both wrap. The only cost is that the readable bytes can be in
two pieces when the data wraps the end, which is exactly what `writev` takes:
`readable_pieces` fills in two bases and two lengths, and one syscall writes
both. That is the reason `write_vectored` exists in `molla.sys.socket`.

The ring is a fixed size on purpose. A response that does not fit is written in
more than one pass rather than growing the buffer, because unbounded output
buffering is how a server with slow clients runs out of memory. The size is a
per connection budget, and how much is queued is `readable()`, which is what
backpressure reads.

`capacity - 1` usable bytes, and it is worth saying why. Full and empty both
have the read and write offsets equal, and telling them apart needs either a
count, a flag, or one byte given up. Giving up the byte keeps the two offsets
independent, which matters because the reader and the writer are the two ends of
the request path.
"""

from molla.sys.mem import (
    RawPtr,
    as_ptr,
    copy_bytes,
    counted_allocate,
    counted_release,
)


def _round_up_pow2(value: Int) -> Int:
    """The next power of two at or above `value`, minimum 64.

    A power of two turns the wrap into an `and` instead of a modulo. The
    difference is one instruction on the hot path, which would be nothing if it
    were not per write."""
    var size = 64
    while size < value:
        size = size * 2
    return size


struct Ring(Movable):
    """A fixed size circular byte buffer."""

    var address: Int
    var capacity: Int
    """A power of two, so masking replaces the modulo."""

    var mask: Int
    var read_at: Int
    """Total bytes read since the ring was made. Never wrapped, so the distance
    between the two offsets is the readable count without a special case for
    which one is ahead. Wrapping happens when an offset is turned into an index,
    which is the only place it can be got wrong."""

    var write_at: Int
    var counter: Int

    def __init__(out self, capacity: Int, counter: Int):
        var want = _round_up_pow2(capacity)
        self.address = counted_allocate(want, counter)
        self.capacity = want if self.address != 0 else 0
        self.mask = self.capacity - 1 if self.address != 0 else 0
        self.read_at = 0
        self.write_at = 0
        self.counter = counter

    def __deinit__(deinit self):
        counted_release(self.address, self.counter)

    def is_valid(self) -> Bool:
        return self.address != 0

    def readable(self) -> Int:
        """Bytes queued and not yet written to the socket. Backpressure reads
        this."""
        return self.write_at - self.read_at

    def writable(self) -> Int:
        """Room for more output. One byte less than the capacity when empty."""
        if self.address == 0:
            return 0
        return self.capacity - 1 - self.readable()

    def is_empty(self) -> Bool:
        return self.read_at == self.write_at

    def clear(mut self):
        self.read_at = 0
        self.write_at = 0

    def push(mut self, data: Span[UInt8, _]) -> Int:
        """Queue bytes. Returns how many were taken, which can be fewer than
        offered and can be zero.

        A short return is not an error, it is backpressure. The caller stops
        producing and waits for the socket to drain, which is the behaviour that
        keeps memory flat when a client stops reading."""
        var count = len(data)
        if count > self.writable():
            count = self.writable()
        if count <= 0:
            return 0

        var start = self.write_at & self.mask
        var first = self.capacity - start
        if first > count:
            first = count
        copy_bytes(self.address + start, Int(data.unsafe_ptr()), first)
        if count > first:
            copy_bytes(
                self.address, Int(data.unsafe_ptr()) + first, count - first
            )
        self.write_at += count
        return count

    def push_str(mut self, text: StringSpan) -> Int:
        """Same, for the parts of a response that are literals."""
        return self.push(
            Span[UInt8, MutAnyOrigin](
                unsafe_ptr=as_ptr(Int(text.unsafe_ptr())),
                length=text.byte_length(),
            )
        )

    def readable_pieces(
        mut self, mut bases: List[Int], mut lengths: List[Int]
    ) -> Int:
        """Fill in the one or two pieces the queued bytes live in.

        Straight into the shape `writev` wants. Two pieces at most, always, so
        the caller can use fixed storage. Returns the number of pieces, which is
        zero when there is nothing to write."""
        bases.clear()
        lengths.clear()
        var count = self.readable()
        if count <= 0:
            return 0

        var start = self.read_at & self.mask
        var first = self.capacity - start
        if first > count:
            first = count
        bases.append(self.address + start)
        lengths.append(first)
        if count > first:
            bases.append(self.address)
            lengths.append(count - first)
            return 2
        return 1

    def consume(mut self, count: Int):
        """Say that `count` queued bytes made it to the socket.

        The whole point of the type: this is two integer updates and no copy, no
        matter where the write stopped."""
        if count <= 0:
            return
        if count >= self.readable():
            self.read_at = self.write_at
            return
        self.read_at += count

    def peek(self, index: Int) -> UInt8:
        """The queued byte at `index`, counting from the read offset. For tests
        and for the framing code that has to look at what it queued."""
        return as_ptr(self.address).unsafe_load(
            (self.read_at + index) & self.mask
        )
