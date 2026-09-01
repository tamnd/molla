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

from std.ffi import c_int, external_call

from molla.sys.errno import errno_name, get_errno
from molla.sys.file import SEEK_END, SEEK_SET, close_fd, lseek, open_read

comptime RawPtr = Pointer[UInt8, MutAnyOrigin]

comptime PROT_READ = 0x01
comptime MAP_PRIVATE = 0x02
"""Same values on macOS and Linux. Unlike the open flags, these two agree."""

comptime MAP_FAILED = -1
"""mmap reports failure as (void *)-1, not NULL, because NULL is a legal
address to map at."""


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

    def close(mut self):
        if not self.mapped:
            return
        _ = external_call["munmap", c_int](self.base(), self.length)
        _ = close_fd(self.fd)
        self.mapped = False
        self.length = 0
        self.fd = -1
