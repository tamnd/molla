"""Reading errno across the FFI boundary.

Every libc call in molla reports failure the same way: a negative return and a
value in errno. errno is thread local, so it cannot be read as a global. Both
platforms expose it as a function returning a pointer to the calling thread's
slot, they just disagree on the name.

The numbers below differ between macOS and Linux and there is no way to ask the
C headers at runtime, so they are written out per platform. Getting one wrong is
the kind of bug that looks like a hang rather than an error, which is why the
test suite provokes real failures and checks the code that comes back.
"""

from std.ffi import c_int, external_call
from std.sys.info import CompilationTarget

comptime ErrnoPtr = Pointer[c_int, MutAnyOrigin]


def errno_location() -> ErrnoPtr:
    """Pointer to this thread's errno slot."""
    comptime if CompilationTarget.is_macos():
        return external_call["__error", ErrnoPtr]()
    else:
        return external_call["__errno_location", ErrnoPtr]()


def get_errno() -> Int:
    """Read errno. Only meaningful straight after a call that failed."""
    return Int(errno_location()[])


def clear_errno():
    """Zero errno so a later read cannot pick up a stale value."""
    errno_location()[] = c_int(0)


comptime EINTR = 4
"""Interrupted by a signal. Retry the call."""

comptime EBADF = 9
"""Bad file descriptor. Always our bug, never the peer's."""

comptime ENOENT = 2
comptime ENOMEM = 12
comptime EACCES = 13
comptime EBUSY = 16
comptime EEXIST = 17
comptime EXDEV = 18
comptime ENOTDIR = 20
comptime EISDIR = 21
comptime EINVAL = 22
comptime EMFILE = 24
comptime ENOSPC = 28
comptime EPIPE = 32
comptime ERANGE = 34
"""These carry the same number on macOS and Linux, which is not a rule so much
as an accident of both inheriting the early Unix table. The ones below did not
survive that inheritance and have to be selected per platform."""


def _eagain() -> Int:
    comptime if CompilationTarget.is_macos():
        return 35
    else:
        return 11


def _einprogress() -> Int:
    comptime if CompilationTarget.is_macos():
        return 36
    else:
        return 115


def _econnreset() -> Int:
    comptime if CompilationTarget.is_macos():
        return 54
    else:
        return 104


def _enotsup() -> Int:
    comptime if CompilationTarget.is_macos():
        return 45
    else:
        return 95


def _eaddrinuse() -> Int:
    comptime if CompilationTarget.is_macos():
        return 48
    else:
        return 98


def _etimedout() -> Int:
    comptime if CompilationTarget.is_macos():
        return 60
    else:
        return 110


def _econnrefused() -> Int:
    comptime if CompilationTarget.is_macos():
        return 61
    else:
        return 111


comptime EAGAIN = _eagain()
"""Would block. On both platforms EWOULDBLOCK has the same value, so one
constant covers both spellings."""

comptime EINPROGRESS = _einprogress()
"""A non blocking connect is underway. Not an error."""

comptime ECONNRESET = _econnreset()
"""The peer went away rudely. Expected traffic, not a fault."""

comptime ENOTSUP = _enotsup()
"""What a wrapper returns for something the platform genuinely cannot do, such
as pinning a thread to a core on macOS."""

comptime EADDRINUSE = _eaddrinuse()
comptime ETIMEDOUT = _etimedout()
comptime ECONNREFUSED = _econnrefused()


def errno_name(code: Int) -> String:
    """A short name for the codes we actually branch on.

    This is not strerror. It covers the handful of values the event loop treats
    specially, so a log line says EAGAIN rather than 35 and nobody has to look
    up which platform's table applies.
    """
    if code == 0:
        return "OK"
    if code == EINTR:
        return "EINTR"
    if code == EBADF:
        return "EBADF"
    if code == EAGAIN:
        return "EAGAIN"
    if code == EINPROGRESS:
        return "EINPROGRESS"
    if code == ECONNRESET:
        return "ECONNRESET"
    if code == ENOENT:
        return "ENOENT"
    if code == ENOMEM:
        return "ENOMEM"
    if code == EACCES:
        return "EACCES"
    if code == EBUSY:
        return "EBUSY"
    if code == EEXIST:
        return "EEXIST"
    if code == EXDEV:
        return "EXDEV"
    if code == EINVAL:
        return "EINVAL"
    if code == EMFILE:
        return "EMFILE"
    if code == ENOSPC:
        return "ENOSPC"
    if code == EPIPE:
        return "EPIPE"
    if code == ENOTSUP:
        return "ENOTSUP"
    if code == EADDRINUSE:
        return "EADDRINUSE"
    if code == ETIMEDOUT:
        return "ETIMEDOUT"
    if code == ECONNREFUSED:
        return "ECONNREFUSED"
    return "errno " + String(code)
