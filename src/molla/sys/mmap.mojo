"""Read only memory maps.

The point of mapping a model rather than reading it is that a 4 GB file costs
no heap and no copy. The kernel faults pages in as they are touched and evicts
them under pressure, so metadata at the front of the file can be walked without
the tensor data behind it ever being read.

Three libc details are worth writing down because each one is a trap.

`mmap` takes a hint address as its first argument and every caller passes NULL.
Pointers are non nullable in Mojo 1.0, so there is no NULL to pass. It goes over
as an integer instead, which is what the ABI does anyway on both x86_64 and
aarch64, and the result comes back as an integer so it can be compared against
MAP_FAILED before being turned into a pointer.

Size comes from `lseek` to the end rather than `fstat`. Both are in
`molla.sys.file` now and either would do, and `lseek` stays because it is one
call and one number rather than a struct with three layouts behind it.

Opening and closing go through `molla.sys.file` rather than being declared here
a second time. Two declarations of `openat` with different argument counts in
one build fail to lower, which is what happened the first time this module kept
its own.
"""

from std.ffi import c_int, c_size_t, external_call
from std.sys.info import CompilationTarget

from molla.sys.errno import errno_name, get_errno
from molla.sys.file import SEEK_END, SEEK_SET, close_fd, lseek, open_read

comptime RawPtr = Pointer[UInt8, MutAnyOrigin]

comptime PROT_READ = 0x01
comptime MAP_PRIVATE = 0x02
"""Same values on macOS and Linux. Unlike the open flags, these two agree."""

comptime MAP_FAILED = -1
"""mmap reports failure as (void *)-1, not NULL, because NULL is a legal
address to map at."""

comptime MADV_WILLNEED = 3
"""Tell the kernel these pages are about to be read. Also the same number on
both platforms, along with MADV_NORMAL, MADV_RANDOM and MADV_SEQUENTIAL."""

comptime PROT_NONE = 0x00
comptime MAP_FIXED = 0x10
"""Also the same on both platforms. `MAP_FIXED` replaces whatever is already at
the address instead of refusing, atomically, which is the whole trick behind
`drop_pages`."""


def _map_anon() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0x1000
    else:
        return 0x20


comptime MAP_ANONYMOUS = _map_anon()
"""One of the few flags the two platforms disagree about. macOS spells it
MAP_ANON and Linux MAP_ANONYMOUS, and the numbers are not the same."""


def _sc_pagesize() -> Int:
    comptime if CompilationTarget.is_macos():
        return 29
    else:
        return 30


comptime SC_PAGESIZE = _sc_pagesize()


def page_size() -> Int:
    """Bytes per page, or 4096 if the call fails.

    Worth asking rather than assuming, because the answer is 4096 on x86_64 and
    16384 on Apple silicon, and both a `madvise` alignment and a loop that
    touches one byte per page get four times too much or four times too little
    work from the wrong constant.
    """
    var n = Int(external_call["sysconf", Int](c_int(SC_PAGESIZE)))
    if n < 1:
        return 4096
    return n


def will_need(address: Int, length: Int) -> Bool:
    """Ask the kernel to start reading the pages at this address.

    This is a hint and it is allowed to do nothing, which is why it returns a
    bool rather than a `SysResult` and why nothing branches on the answer. What
    it buys is readahead. Without it a loop that touches one byte per page waits
    for one page at a time and gets a fraction of what the device can do, and
    with it the reads are already in flight by the time the loop arrives. The
    range is widened to whole pages because `madvise` rejects an address that is
    not page aligned, and widening is safe here because a neighbouring page of a
    file we are walking is a page we are about to want anyway.

    Takes a bare address rather than a `Mapping` so a worker thread can call it
    without holding a reference to the mapping, which it cannot do through a
    thin function entry point.
    """
    if length <= 0:
        return False
    var page = page_size()
    var begin = address - (address % page)
    var rc = Int(
        external_call["madvise", c_int](
            RawPtr(unsafe_from_address=begin),
            c_size_t(address + length - begin),
            c_int(MADV_WILLNEED),
        )
    )
    return rc == 0


def drop_pages(address: Int, length: Int) -> Bool:
    """Give back the pages at this address, for good.

    The other end of `will_need`, and the reason peak resident memory does not
    have to scale with the size of a model. A load walks a mapping once, hands
    each tensor to the card, and never looks at those bytes again, so without
    this the whole file ends up resident and stays resident. See
    [docs/validation/performance.md](../../../docs/validation/performance.md).

    The obvious way to write this is `madvise(MADV_DONTNEED)`, which is what it
    was first, and on macOS that does nothing at all. A clean page of a mapped
    file stays charged to the process until there is memory pressure, and
    MADV_DONTNEED, MADV_FREE and `msync(MS_INVALIDATE)` all return success and
    move the resident set by nothing. Measured on a 468 MiB file: 485 MiB
    resident before the call and 485 MiB after, for all three.

    So the range is replaced instead. An anonymous `PROT_NONE` mapping over the
    top with `MAP_FIXED` drops the file pages, on both platforms, and the same
    468 MiB file went from 485 MiB resident to 16 MiB. `MAP_FIXED` does that
    atomically, so there is no moment where the address is unmapped and some
    other allocation could land in it.

    `PROT_NONE` and not a readable mapping, and replaced rather than plain
    `munmap`, for the same reason: it keeps the hole. A read of a dropped
    address is a bug, and this way it is a fault at the address that caused it
    rather than zeros, or worse, whatever the next `mmap` in the process
    happened to put there. `Mapping.close()` still unmaps the whole original
    range afterwards, which is fine over a range with replacements in it.

    Narrowed to whole pages inside the range rather than widened to whole pages
    covering it, which is the opposite of what `will_need` does and for the
    opposite reason. A partial page at either end is shared with the tensor next
    door, and taking away a page somebody else is about to read is no longer a
    re-read, it is a crash. So the ends are left alone and a tensor smaller than
    a page drops nothing.
    """
    if length <= 0:
        return False
    var page = page_size()
    var begin = address + (page - 1)
    begin = begin - (begin % page)
    var end = address + length
    end = end - (end % page)
    if end <= begin:
        return False
    var got = Int(
        external_call["mmap", Int](
            begin,
            end - begin,
            c_int(PROT_NONE),
            c_int(MAP_PRIVATE | MAP_FIXED | MAP_ANONYMOUS),
            c_int(-1),
            Int64(0),
        )
    )
    return got == begin


struct Mapping(Movable):
    """A whole file mapped read only.

    Closed explicitly rather than in `__del__`, matching `Poller` and the
    servers. The descriptor is kept because closing it early is legal but makes
    the mapping harder to reason about in a debugger.
    """

    var address: Int
    """Held as an integer, not a pointer, because a struct field may not expose
    AnyOrigin and a mapping has no origin to borrow from. `base()` hands one
    out, which is a register move rather than a load."""

    var length: Int
    var fd: Int
    var mapped: Bool

    def __init__(out self):
        """A mapping of nothing.

        There is no file behind this and there never was. It exists so a struct
        that holds a mapping can be built before it knows whether the file it
        wants is there, which is the common case for the optional JSON files in
        a model directory. `close()` on one is a no op, and `base()` is not to
        be called on one: length is zero, so every loop over it does nothing.
        """
        self.address = 0
        self.length = 0
        self.fd = -1
        self.mapped = False

    def __init__(out self, path: StringSpan) raises:
        var opened = open_read(path)
        if opened.is_err():
            raise Error(
                "open failed for "
                + String(path)
                + ": "
                + errno_name(opened.err)
            )
        var fd = opened.value

        var measured = lseek(fd, 0, SEEK_END)
        if measured.is_err():
            _ = close_fd(fd)
            raise Error("lseek failed: " + errno_name(measured.err))
        var length = measured.value
        if length == 0:
            _ = close_fd(fd)
            raise Error("refusing to map an empty file: " + String(path))
        _ = lseek(fd, 0, SEEK_SET)

        # The hint address goes over as an integer because there is no null
        # pointer to pass, and the result comes back as one so it can be
        # checked before it becomes a pointer.
        var addr = Int(
            external_call["mmap", Int](
                Int(0),
                length,
                c_int(PROT_READ),
                c_int(MAP_PRIVATE),
                c_int(fd),
                Int64(0),
            )
        )
        if addr == MAP_FAILED:
            var code = get_errno()
            _ = close_fd(fd)
            raise Error("mmap failed: " + errno_name(code))

        self.address = addr
        self.length = length
        self.fd = fd
        self.mapped = True

    def base(self) -> RawPtr:
        return RawPtr(unsafe_from_address=self.address)

    def will_need(self, start: Int, length: Int) -> Bool:
        """Ask the kernel to start reading this range of the file now."""
        if not self.mapped or start < 0 or length <= 0:
            return False
        if start + length > self.length:
            return False
        return will_need(self.address + start, length)

    def close(mut self):
        if not self.mapped:
            return
        _ = external_call["munmap", c_int](self.base(), self.length)
        _ = close_fd(self.fd)
        self.mapped = False
        self.length = 0
        self.fd = -1
