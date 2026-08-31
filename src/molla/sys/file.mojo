"""Files and directories over libc.

Everything here returns a `SysResult` and nothing here raises, which is the rule
for the whole `molla.sys` layer.

Three things about this module are not obvious and all three are compiler
constraints rather than choices.

`open`, `read` and `write` cannot be reached through `external_call`. The
standard library already declares symbols with those names and a second
declaration with a different signature fails to lower, so this module uses the
`at` family instead: `openat`, `unlinkat`, `mkdirat` and `renameat` with
`AT_FDCWD`, which behave exactly like the plain calls when the path is absolute
or relative to the working directory. Byte transfer goes through `pread` and
`pwrite` with an explicit offset. That is not a workaround so much as the
version molla wants anyway, because an offset in the call is one fewer piece of
per descriptor state to get wrong when two things read the same file.

`struct stat` has three layouts across the three targets molla builds for, and
they disagree on where `st_size` and `st_mode` are. The offsets are written out
per target and asserted against a file of known length in the test suite, which
is the only way to notice a wrong offset before it turns into a corrupt read.

`struct dirent` disagrees too, in the same way and for the same reason, so the
name and type offsets are also per platform.
"""

from std.ffi import c_int, c_size_t, c_ssize_t, external_call
from std.memory import stack_allocation
from std.sys.info import CompilationTarget

from molla.sys.cstr import c_string, from_c_string
from molla.sys.result import SysResult, checked, checked_ptr, ok

comptime O_RDONLY = 0
comptime O_WRONLY = 1
comptime O_RDWR = 2
"""The access modes are the same three numbers everywhere. Everything else in
the open flag space is not."""


def _at_fdcwd() -> Int:
    comptime if CompilationTarget.is_macos():
        return -2
    else:
        return -100


def _o_creat() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0x0200
    else:
        return 0o100


def _o_trunc() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0x0400
    else:
        return 0o1000


def _o_append() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0x0008
    else:
        return 0o2000


def _o_excl() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0x0800
    else:
        return 0o200


comptime AT_FDCWD = _at_fdcwd()
comptime O_CREAT = _o_creat()
comptime O_TRUNC = _o_trunc()
comptime O_APPEND = _o_append()
comptime O_EXCL = _o_excl()


def _at_removedir() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0x80
    else:
        return 0x200


comptime AT_REMOVEDIR = _at_removedir()

comptime SEEK_SET = 0
comptime SEEK_CUR = 1
comptime SEEK_END = 2

comptime MODE_644 = 0o644
"""What molla creates files with. The store writes blobs that other tools are
expected to read, per the second commitment in the README."""

comptime MODE_755 = 0o755

comptime MODE_600 = 0o600
"""What molla creates temporary files with. A spilled request body is one
client's data sitting in a world readable directory, so it is readable by the
user molla runs as and by nobody else."""


def open_at(path: StringSpan, flags: Int, mode: Int) -> SysResult:
    """Open a path. Returns the descriptor, or the errno that stopped it.

    `openat` is variadic: the mode argument only exists when O_CREAT is in the
    flags, and C passes it the variadic way. `num_fixed_args=3` is what tells
    the compiler where the fixed arguments stop, and without it the call is a
    four argument function to Mojo, which collides with the three argument
    declaration the standard library already made and fails to lower. The same
    rule caught `fcntl` in the socket spike, where the effect was worse: it
    compiled and set nothing."""
    var buf = c_string(path)
    var rc = checked(
        Int(
            external_call["openat", c_int, num_fixed_args=3](
                c_int(AT_FDCWD), buf.unsafe_ptr(), c_int(flags), c_int(mode)
            )
        )
    )
    _ = buf^
    return rc


def open_read(path: StringSpan) -> SysResult:
    """Open an existing file for reading."""
    return open_at(path, O_RDONLY, 0)


def create(path: StringSpan) -> SysResult:
    """Create or truncate a file for writing."""
    return open_at(path, O_WRONLY | O_CREAT | O_TRUNC, MODE_644)


def close_fd(fd: Int) -> SysResult:
    """Close a descriptor.

    Worth checking on a file that was written to. A failure here can be the
    first report of a write that never reached the disk."""
    return checked(Int(external_call["close", c_int](c_int(fd))))


def pread[
    o: MutOrigin
](fd: Int, buf: Pointer[UInt8, o], count: Int, offset: Int) -> SysResult:
    """Read at an offset without moving the descriptor's position.

    Returns the byte count, which can be short, and 0 at end of file."""
    return checked(
        Int(
            external_call["pread", c_ssize_t](
                c_int(fd), buf, c_size_t(count), Int64(offset)
            )
        )
    )


def pwrite[
    o: MutOrigin
](fd: Int, buf: Pointer[UInt8, o], count: Int, offset: Int) -> SysResult:
    """Write at an offset. Returns the byte count, which can be short."""
    return checked(
        Int(
            external_call["pwrite", c_ssize_t](
                c_int(fd), buf, c_size_t(count), Int64(offset)
            )
        )
    )


def pread_all[
    o: MutOrigin
](fd: Int, buf: Pointer[UInt8, o], count: Int, offset: Int) -> SysResult:
    """Read exactly `count` bytes, or fewer if the file ends first.

    A short read from a regular file is not an error and not rare, so the loop
    belongs here rather than in every caller. Returns how many bytes landed."""
    var done = 0
    while done < count:
        var r = pread(fd, buf.unsafe_offset(done), count - done, offset + done)
        if r.is_err():
            return SysResult(done, r.err)
        if r.value == 0:
            break
        done += r.value
    return ok(done)


def pwrite_all[
    o: MutOrigin
](fd: Int, buf: Pointer[UInt8, o], count: Int, offset: Int) -> SysResult:
    """Write exactly `count` bytes, looping over short writes."""
    var done = 0
    while done < count:
        var w = pwrite(fd, buf.unsafe_offset(done), count - done, offset + done)
        if w.is_err():
            return SysResult(done, w.err)
        if w.value == 0:
            break
        done += w.value
    return ok(done)


def lseek(fd: Int, offset: Int, whence: Int) -> SysResult:
    """Move or query the descriptor position. `lseek(fd, 0, SEEK_END)` is how
    this module gets a file length without a stat."""
    return checked(
        Int(
            external_call["lseek", Int64](
                c_int(fd), Int64(offset), c_int(whence)
            )
        )
    )


def ftruncate(fd: Int, length: Int) -> SysResult:
    """Set a file's length. Used to preallocate a blob before writing it."""
    return checked(
        Int(external_call["ftruncate", c_int](c_int(fd), Int64(length)))
    )


def fsync(fd: Int) -> SysResult:
    """Flush a file's data to the device.

    The store calls this before renaming a blob into place. Without it a crash
    can leave a file that has a name and no contents, which is worse than not
    having the file, because the name is what says the contents are good."""
    return checked(Int(external_call["fsync", c_int](c_int(fd))))


def unlink(path: StringSpan) -> SysResult:
    """Remove a file."""
    var buf = c_string(path)
    var rc = checked(
        Int(
            external_call["unlinkat", c_int](
                c_int(AT_FDCWD), buf.unsafe_ptr(), c_int(0)
            )
        )
    )
    _ = buf^
    return rc


def rmdir(path: StringSpan) -> SysResult:
    """Remove an empty directory."""
    var buf = c_string(path)
    var rc = checked(
        Int(
            external_call["unlinkat", c_int](
                c_int(AT_FDCWD), buf.unsafe_ptr(), c_int(AT_REMOVEDIR)
            )
        )
    )
    _ = buf^
    return rc


def mkdir(path: StringSpan, mode: Int) -> SysResult:
    """Create one directory. Does not create parents."""
    var buf = c_string(path)
    var rc = checked(
        Int(
            external_call["mkdirat", c_int](
                c_int(AT_FDCWD), buf.unsafe_ptr(), c_int(mode)
            )
        )
    )
    _ = buf^
    return rc


def rename(old_path: StringSpan, new_path: StringSpan) -> SysResult:
    """Rename within one filesystem.

    Atomic on both platforms, which is the whole reason the store writes to a
    temporary name and renames. Across filesystems it fails with EXDEV rather
    than copying, and that is the correct behaviour to expose."""
    var a = c_string(old_path)
    var b = c_string(new_path)
    var rc = checked(
        Int(
            external_call["renameat", c_int](
                c_int(AT_FDCWD),
                a.unsafe_ptr(),
                c_int(AT_FDCWD),
                b.unsafe_ptr(),
            )
        )
    )
    _ = a^
    _ = b^
    return rc


comptime STAT_BUF_SIZE = 256
"""Bigger than `struct stat` on every target molla builds for. macOS is 144
bytes, Linux x86-64 is 144 and Linux arm64 is 128. Over allocating on the stack
costs nothing and means a layout that grows cannot scribble past the buffer."""


def _stat_size_offset() -> Int:
    comptime if CompilationTarget.is_macos():
        return 96
    else:
        return 48


def _stat_mode_offset() -> Int:
    comptime if CompilationTarget.is_macos():
        return 4
    elif CompilationTarget.is_x86():
        return 24
    else:
        return 16


def _stat_mode_width() -> Int:
    comptime if CompilationTarget.is_macos():
        return 2
    else:
        return 4


def _stat_mtime_offset() -> Int:
    comptime if CompilationTarget.is_macos():
        return 48
    elif CompilationTarget.is_x86():
        return 88
    else:
        return 72


comptime STAT_SIZE_OFF = _stat_size_offset()
comptime STAT_MODE_OFF = _stat_mode_offset()
comptime STAT_MODE_WIDTH = _stat_mode_width()
comptime STAT_MTIME_OFF = _stat_mtime_offset()
"""Three layouts, one per target. macOS puts a 16 bit mode near the front and
pushes size out past four timespecs. Linux x86-64 has a 64 bit inode and nlink
ahead of a 32 bit mode. Linux arm64 orders the same fields differently again.
Every one of these offsets is checked in `tests/test_sys.mojo` against a file
whose length and type the test wrote itself."""

comptime S_IFMT = 0xF000
comptime S_IFREG = 0x8000
comptime S_IFDIR = 0x4000


struct FileInfo(Copyable, ImplicitlyCopyable, Movable):
    """The three fields of `struct stat` molla actually reads."""

    var size: Int
    var mode: Int
    var mtime: Int

    def __init__(out self):
        self.size = 0
        self.mode = 0
        self.mtime = 0

    def is_file(self) -> Bool:
        return (self.mode & S_IFMT) == S_IFREG

    def is_dir(self) -> Bool:
        return (self.mode & S_IFMT) == S_IFDIR

    def permissions(self) -> Int:
        return self.mode & 0o7777


def fstat(fd: Int, mut info: FileInfo) -> SysResult:
    """Read size, mode and modification time for an open descriptor."""
    var buf = stack_allocation[STAT_BUF_SIZE, UInt8]()
    for i in range(STAT_BUF_SIZE):
        buf.unsafe_store(i, UInt8(0))
    var rc = checked(Int(external_call["fstat", c_int](c_int(fd), buf)))
    if rc.is_err():
        return rc
    info.size = Int(
        buf.unsafe_offset(STAT_SIZE_OFF).unsafe_bitcast[Int64]().unsafe_load(0)
    )
    var mode_ptr = buf.unsafe_offset(STAT_MODE_OFF)
    comptime if STAT_MODE_WIDTH == 2:
        info.mode = Int(mode_ptr.unsafe_bitcast[UInt16]().unsafe_load(0))
    else:
        info.mode = Int(mode_ptr.unsafe_bitcast[UInt32]().unsafe_load(0))
    info.mtime = Int(
        buf.unsafe_offset(STAT_MTIME_OFF).unsafe_bitcast[Int64]().unsafe_load(0)
    )
    return rc


def stat_path(path: StringSpan, mut info: FileInfo) -> SysResult:
    """Same as `fstat` for a path that may not be open.

    Opening and closing to stat is a real cost and this is not on any hot path.
    It exists so the store can ask whether a blob is already there without
    inventing a second stat binding for `fstatat`, whose flag argument differs
    between the two platforms in a way that is easy to get wrong."""
    var fd = open_read(path)
    if fd.is_err():
        return fd
    var rc = fstat(fd.value, info)
    _ = close_fd(fd.value)
    return rc


comptime F_OK = 0
"""The `access` mode that asks about existence rather than permission."""


def exists(path: StringSpan) -> Bool:
    """Whether anything is at this path.

    `access` with F_OK, which is the existence question and not the permission
    question. Asking about permission and then acting on the answer is a race
    and molla never does it, but existence is the whole answer here.

    This started out as an open followed by a close, which is a fine test for a
    regular file and a wrong one for everything else: opening a unix socket for
    reading fails with ENXIO even though the path is plainly there, so a bound
    listener looked like a missing file."""
    var buf = c_string(path)
    var rc = Int(external_call["access", c_int](buf.unsafe_ptr(), c_int(F_OK)))
    _ = buf^
    return rc == 0


def _dirent_name_offset() -> Int:
    comptime if CompilationTarget.is_macos():
        return 21
    else:
        return 19


def _dirent_type_offset() -> Int:
    comptime if CompilationTarget.is_macos():
        return 20
    else:
        return 18


comptime DIRENT_NAME_OFF = _dirent_name_offset()
comptime DIRENT_TYPE_OFF = _dirent_type_offset()
"""macOS carries an eight byte seek offset and a two byte name length that
Linux does not, so the name starts two bytes later. Reading a Linux name at the
macOS offset gives a name missing its first two characters, which looks like a
corrupt directory rather than a wrong constant."""

comptime DT_DIR = 4
comptime DT_REG = 8
comptime DT_LNK = 10
comptime MAX_NAME = 255


struct DirEntry(Copyable, Movable):
    """One name from a directory, with the type the kernel already knew."""

    var name: String
    var kind: Int

    def __init__(out self, var name: String, kind: Int):
        self.name = name^
        self.kind = kind

    def is_dir(self) -> Bool:
        return self.kind == DT_DIR

    def is_file(self) -> Bool:
        return self.kind == DT_REG


def read_dir(path: StringSpan, mut out_entries: List[DirEntry]) -> SysResult:
    """List a directory, skipping `.` and `..`.

    Appends to `out_entries` rather than returning a list, because a list of a
    non trivial type cannot be moved out of a struct field or a tuple in Mojo
    1.0 without a fight, and an out parameter is the shape that always works.

    `d_type` comes back as DT_UNKNOWN on a few filesystems, notably some
    network mounts, and callers that care have to stat the name themselves.
    molla's store lives on a local disk where the field is filled in."""
    var buf = c_string(path)
    var dir = checked_ptr(Int(external_call["opendir", Int](buf.unsafe_ptr())))
    _ = buf^
    if dir.is_err():
        return dir

    var count = 0
    while True:
        var entry = Int(external_call["readdir", Int](dir.value))
        if entry == 0:
            # Null means end of directory, or an error with errno set. Reading
            # errno here is safe because nothing has run since the call.
            break
        var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=entry)
        var name = from_c_string(p.unsafe_offset(DIRENT_NAME_OFF), MAX_NAME)
        if name == "." or name == "..":
            continue
        var kind = Int(p.unsafe_load(DIRENT_TYPE_OFF))
        out_entries.append(DirEntry(name^, kind))
        count += 1

    _ = external_call["closedir", c_int](dir.value)
    return ok(count)
