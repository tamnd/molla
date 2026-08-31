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

Size comes from `lseek` to the end rather than `fstat`. `struct stat` has a
different layout on macOS and Linux and a different one again per architecture,
so declaring it correctly means writing out three layouts and getting all three
right. `lseek` returns the same number with no struct at all.

The file is opened with `openat` rather than `open`, which looks like an odd
choice and is explained where the call is. Paths are also the first thing in
molla that needs a C string, and Mojo strings are not null terminated, so the
bytes get copied into a list with a zero on the end and that list has to
outlive the call.
"""

from std.ffi import c_int, external_call
from std.sys.info import CompilationTarget

from molla.sys.errno import errno_name, get_errno

comptime RawPtr = Pointer[UInt8, MutAnyOrigin]

comptime O_RDONLY = 0
comptime SEEK_SET = 0
comptime SEEK_END = 2


def _at_fdcwd() -> Int:
    comptime if CompilationTarget.is_macos():
        return -2
    else:
        return -100


comptime AT_FDCWD = _at_fdcwd()
"""Tells openat to resolve a relative path against the working directory, which
is what plain open does. One of the few constants where the two kernels picked
different numbers for the same idea."""

comptime PROT_READ = 0x01
comptime MAP_PRIVATE = 0x02
"""Same values on macOS and Linux. Unlike the open flags, these two agree."""

comptime MAP_FAILED = -1
"""mmap reports failure as (void *)-1, not NULL, because NULL is a legal
address to map at."""


def _c_string(path: StringSpan) -> List[UInt8]:
    """Copy a path into a null terminated buffer.

    The returned list owns the bytes and has to stay alive across the call that
    uses it. Taking `unsafe_ptr` of a temporary would dangle.
    """
    var out = List[UInt8]()
    out.reserve(path.byte_length() + 1)
    var p = path.unsafe_ptr()
    for i in range(path.byte_length()):
        out.append(p.unsafe_load(i))
    out.append(0)
    return out^


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

    def __init__(out self, path: StringSpan) raises:
        var buf = _c_string(path)
        # openat rather than open, and not for any of the reasons openat
        # usually exists. Declaring `open` here collides with the declaration
        # the standard library's own file API makes, the two signatures
        # disagree, and the compiler refuses the module with "existing function
        # with conflicting signature". openat has no such declaration anywhere
        # and AT_FDCWD makes it behave exactly like open.
        var fd = Int(
            external_call["openat", c_int](
                c_int(AT_FDCWD), buf.unsafe_ptr(), c_int(O_RDONLY)
            )
        )
        if fd < 0:
            raise Error(
                "open failed for "
                + String(path)
                + ": "
                + errno_name(get_errno())
            )

        var length = Int(
            external_call["lseek", Int64](c_int(fd), Int64(0), c_int(SEEK_END))
        )
        if length < 0:
            var code = get_errno()
            _ = external_call["close", c_int](c_int(fd))
            raise Error("lseek failed: " + errno_name(code))
        if length == 0:
            _ = external_call["close", c_int](c_int(fd))
            raise Error("refusing to map an empty file: " + String(path))
        _ = external_call["lseek", Int64](c_int(fd), Int64(0), c_int(SEEK_SET))

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
            _ = external_call["close", c_int](c_int(fd))
            raise Error("mmap failed: " + errno_name(code))

        self.address = addr
        self.length = length
        self.fd = fd
        self.mapped = True
        # `buf` dies here, after the open call that used it.

    def base(self) -> RawPtr:
        return RawPtr(unsafe_from_address=self.address)

    def close(mut self):
        if not self.mapped:
            return
        _ = external_call["munmap", c_int](self.base(), self.length)
        _ = external_call["close", c_int](c_int(self.fd))
        self.mapped = False
        self.length = 0
        self.fd = -1
