"""File descriptor operations.

Thin wrappers over libc. They return what libc returns, negative on failure,
because the event loop needs to look at errno and decide rather than have an
exception thrown at it on every EAGAIN. The one place that raises is
`set_nonblocking`, since a failure there is a setup bug and there is nothing
sensible to do but stop.

Byte transfer is not here. `read` and `write` cannot be reached through
`external_call` at all, because `std.ffi` already declares symbols with those
names and a second declaration with a different signature fails to lower. Socket
I/O uses `send` and `recv` instead, in `molla.sys.socket`.

`fcntl` is variadic, and on arm64 macOS that is not a detail. Variadic arguments
go on the stack there while fixed arguments go in registers, so calling it as an
ordinary three argument function puts the flags in x2 where libc never looks,
and libc reads whatever was on the stack instead. The call returns 0 and does
nothing, or sets a flag word out of thin air. `num_fixed_args=2` tells the
compiler where the fixed arguments stop, and every variadic libc function molla
touches has to carry it.
"""

from std.ffi import c_int, external_call
from std.sys.info import CompilationTarget

from molla.sys.errno import errno_name, get_errno

comptime F_GETFL = 3
comptime F_SETFL = 4


def _o_nonblock() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0x0004
    else:
        return 0o4000


comptime O_NONBLOCK = _o_nonblock()
"""macOS and Linux picked different bits for this. Linux uses 04000 octal,
macOS uses 0x4. Hardcoding either one silently does nothing on the other."""


def close(fd: Int) -> Int:
    """Close a descriptor. Returns 0, or -1 with errno set."""
    return Int(external_call["close", c_int](c_int(fd)))


def set_nonblocking(fd: Int) raises:
    """Put a descriptor into non blocking mode.

    Read modify write rather than a bare set, because an accepted socket
    inherits flags on some platforms and clobbering them would be a subtle bug.
    """
    var flags = get_flags(fd)
    if flags < 0:
        raise Error("fcntl F_GETFL failed: " + errno_name(get_errno()))
    var rc = Int(
        external_call["fcntl", c_int, num_fixed_args=2](
            c_int(fd), c_int(F_SETFL), c_int(flags | O_NONBLOCK)
        )
    )
    if rc < 0:
        raise Error("fcntl F_SETFL failed: " + errno_name(get_errno()))
    # fcntl reports success whether or not the flag took, so this reads it back.
    # That is not paranoia, it is the exact failure the variadic ABI produced.
    if not is_nonblocking(fd):
        raise Error("fcntl F_SETFL reported success but O_NONBLOCK is not set")


def get_flags(fd: Int) -> Int:
    """The descriptor's F_GETFL flag word, or -1."""
    return Int(
        external_call["fcntl", c_int, num_fixed_args=2](
            c_int(fd), c_int(F_GETFL), c_int(0)
        )
    )


def is_nonblocking(fd: Int) -> Bool:
    """Whether O_NONBLOCK is set. Used by the tests to prove `accept` results
    were switched over, since inheriting the listener's flags is platform
    dependent and easy to assume wrongly."""
    var flags = get_flags(fd)
    if flags < 0:
        return False
    return (flags & O_NONBLOCK) != 0
