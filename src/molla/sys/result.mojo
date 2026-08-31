"""The result type every OS wrapper returns.

libc has one convention and molla has one type for it. A call either produced a
value or it set errno, and both live in the same struct so a caller can branch
without a second trip to `get_errno`. That second trip is the bug this type
exists to prevent: errno is only meaningful straight after the call that set it,
and any allocation, print or `String` concatenation in between can overwrite it.
Capturing it at the call site is the only version that is always correct.

Nothing here raises and nothing here aborts. The event loop sees EAGAIN on most
of its reads and that is normal traffic rather than an error, so a wrapper that
threw would turn the common path into exception handling. Callers that do want
an exception ask for one with `unwrap`, which is the right shape for setup code
where there is nothing sensible to do but stop.
"""

from molla.sys.errno import errno_name, get_errno


struct SysResult(Copyable, ImplicitlyCopyable, Movable):
    """A libc return value, and the errno that came with it if it failed."""

    var value: Int
    """What the call returned. Meaningful only when `err` is zero, except for
    the calls that return a count and set errno on partial failure, and molla
    has none of those."""

    var err: Int
    """errno as captured immediately after the call, or 0 on success."""

    def __init__(out self, value: Int, err: Int):
        self.value = value
        self.err = err

    def is_ok(self) -> Bool:
        return self.err == 0

    def is_err(self) -> Bool:
        return self.err != 0

    def is_errno(self, code: Int) -> Bool:
        """Whether the call failed with this specific errno.

        Reads better than comparing `err` at the call site, and it means the
        EAGAIN check in the reactor says what it means."""
        return self.err == code

    def unwrap(self, what: StringSpan) raises -> Int:
        """The value, or an exception naming the operation and the errno.

        For setup paths. Binding a listener that fails is not something the
        caller can carry on from, and a raise there is shorter than a check."""
        if self.err != 0:
            raise Error(String(what) + " failed: " + errno_name(self.err))
        return self.value

    def describe(self, what: StringSpan) -> String:
        """One line for a log or an error message."""
        if self.err == 0:
            return String(what) + " ok, returned " + String(self.value)
        return String(what) + " failed: " + errno_name(self.err)


def checked(rc: Int) -> SysResult:
    """Wrap a libc return that signals failure with a negative value.

    Call this with the return value directly, as `checked(Int(external_call...))`,
    so nothing runs between the call and the errno read."""
    if rc < 0:
        return SysResult(rc, get_errno())
    return SysResult(rc, 0)


def checked_ptr(address: Int) -> SysResult:
    """Wrap a libc return that signals failure with a null pointer.

    `opendir`, `fdopendir` and the rest of the handle returning calls. mmap is
    not one of these, because it fails with (void *)-1 rather than null."""
    if address == 0:
        return SysResult(0, get_errno())
    return SysResult(address, 0)


def ok(value: Int) -> SysResult:
    """A success with a value nobody had to ask libc for."""
    return SysResult(value, 0)
