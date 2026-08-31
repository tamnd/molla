"""Signals and process identity.

Two signals matter to a server. SIGPIPE has to be ignored or writing to a peer
that hung up kills the process, and SIGINT and SIGTERM have to arrive somewhere
that can shut the thing down in order.

The plan was to block both in every thread and have one thread call `sigwait`,
which turns a signal into an ordinary function return on a thread of our
choosing with the whole language available. It does not work in a Mojo 1.0
binary. Blocking SIGTERM on the main thread and then sending it to ourselves
still kills the process, because the runtime has already started threads of its
own that do not have it blocked and a process directed signal goes to any thread
that will take it. There is no way to set another thread's mask, so `sigwait` is
only correct in a process whose threads molla created, and molla does not have
one. The wrappers are kept because they are the right primitive the day that
changes, and the docstrings say so rather than leaving a trap.

What works is the self pipe, and the awkward part of it is that Mojo 1.0 has no
globals, so a handler cannot be told where to write. The one thing a handler can
close over is a compile time constant, so molla reserves a descriptor number,
duplicates the writing end of a socketpair onto it at startup, and the handler
sends one byte containing the signal number. `send` is on the list of calls a
handler may make. The reading end is an ordinary descriptor the reactor can poll
alongside its sockets, which is what makes shutdown one more readable thing
rather than a special case.

A socketpair rather than a pipe, because `write` cannot be reached through
`external_call` at all and `pwrite` fails on a pipe with ESPIPE. `send` on a
Unix domain socket is the same idea and it is a symbol we can actually bind.

`sigset_t` is 4 bytes on macOS and 128 on Linux, so it lives in heap storage
sized for the larger one and gets filled in by libc rather than by hand.
"""

from std.ffi import c_int, c_size_t, c_ssize_t, external_call
from std.memory import stack_allocation
from std.sys.info import CompilationTarget

from molla.sys.errno import EBADF, EBUSY
from molla.sys.fd import close, get_flags, set_nonblocking
from molla.sys.result import SysResult, checked, ok
from molla.sys.socket import AF_UNIX, SOCK_STREAM

comptime SIGINT = 2
comptime SIGQUIT = 3
comptime SIGKILL = 9
comptime SIGPIPE = 13
comptime SIGALRM = 14
comptime SIGTERM = 15
"""The numbers molla uses are the same on macOS and Linux. The ones that differ,
SIGUSR1 and the job control signals, are not used here."""

comptime SIG_DFL = 0
comptime SIG_IGN = 1
comptime SIG_ERR = -1

"""A C symbol can only be declared once per binary, and every `external_call`
naming it has to agree on the signature. `signal` takes a function pointer and
also takes SIG_IGN, which is the integer 1 wearing a function pointer's clothes,
and Mojo sees those as two different declarations of the same symbol. The build
fails at lowering with "existing function with conflicting signature", well
after the point where the mistake looks like it is in the file that broke.

So the two shapes get two symbols. `signal` is only ever called with an integer
disposition here, and `sigset` is only ever called with a real handler. They are
the same call underneath, `sigset` additionally unblocks the signal it is
installing, which is what molla wants anyway."""

comptime SIGSET_SIZE = 128
"""Large enough for either platform's sigset_t. libc writes it, nothing here
reads its layout, so over allocating is free."""


def _sig_block() -> Int:
    comptime if CompilationTarget.is_macos():
        return 1
    else:
        return 0


def _sig_unblock() -> Int:
    comptime if CompilationTarget.is_macos():
        return 2
    else:
        return 1


def _sig_setmask() -> Int:
    comptime if CompilationTarget.is_macos():
        return 3
    else:
        return 2


comptime SIG_BLOCK = _sig_block()
comptime SIG_UNBLOCK = _sig_unblock()
comptime SIG_SETMASK = _sig_setmask()
"""These three do not agree across platforms and nothing warns you. macOS
numbers them 1, 2 and 3, Linux numbers them 0, 1 and 2, so Linux's SIG_BLOCK is
macOS's "no such how argument" and macOS's SIG_BLOCK is Linux's SIG_UNBLOCK.
Getting it wrong on Linux does not fail, it unblocks the signals you meant to
block and the process dies on the first SIGTERM instead of shutting down. It
was caught here because pthread_sigmask on macOS returned EINVAL for 0."""


def ignore_sigpipe() -> SysResult:
    """Stop a write to a closed peer from killing the process.

    Call this once at startup, before any socket exists. The alternative is
    MSG_NOSIGNAL on every send, which macOS spells SO_NOSIGPIPE and applies to
    the socket rather than the call, so one process wide setting is the version
    that is the same on both platforms."""
    var rc = Int(external_call["signal", Int](c_int(SIGPIPE), Int(SIG_IGN)))
    if rc == SIG_ERR:
        return checked(-1)
    return ok(rc)


def restore_default(signum: Int) -> SysResult:
    """Put a signal back to its default disposition."""
    var rc = Int(external_call["signal", Int](c_int(signum), Int(SIG_DFL)))
    if rc == SIG_ERR:
        return checked(-1)
    return ok(rc)


struct SignalSet(Movable):
    """A set of signals, in storage libc owns the layout of."""

    var storage: List[UInt8]

    def __init__(out self):
        self.storage = List[UInt8]()
        for _ in range(SIGSET_SIZE):
            self.storage.append(0)

    def raw(self) -> Int:
        return Int(self.storage.unsafe_ptr())

    def clear(mut self) -> SysResult:
        return checked(Int(external_call["sigemptyset", c_int](self.raw())))

    def fill(mut self) -> SysResult:
        return checked(Int(external_call["sigfillset", c_int](self.raw())))

    def add(mut self, signum: Int) -> SysResult:
        return checked(
            Int(external_call["sigaddset", c_int](self.raw(), c_int(signum)))
        )

    def remove(mut self, signum: Int) -> SysResult:
        return checked(
            Int(external_call["sigdelset", c_int](self.raw(), c_int(signum)))
        )

    def contains(mut self, signum: Int) -> Bool:
        return (
            Int(external_call["sigismember", c_int](self.raw(), c_int(signum)))
            == 1
        )


def block_signals(mut mask: SignalSet) -> SysResult:
    """Block a set of signals on the calling thread.

    A thread created after this inherits the mask, so calling it before
    spawning is what puts a thread in the same state. That is not enough to
    make `wait_for_signal` usable on its own, because the Mojo runtime's own
    threads exist before molla's `main` runs and nothing can change their
    masks. Useful for keeping a signal off a worker thread, which is what
    molla uses it for."""
    return _sigmask_result(
        Int(
            external_call["pthread_sigmask", c_int](
                c_int(SIG_BLOCK), mask.raw(), Int(0)
            )
        )
    )


def unblock_signals(mut mask: SignalSet) -> SysResult:
    """Unblock a set of signals on the calling thread."""
    return _sigmask_result(
        Int(
            external_call["pthread_sigmask", c_int](
                c_int(SIG_UNBLOCK), mask.raw(), Int(0)
            )
        )
    )


def _sigmask_result(rc: Int) -> SysResult:
    """pthread_sigmask reports through its return value like the rest of
    pthreads, not through errno."""
    if rc != 0:
        return SysResult(-1, rc)
    return ok(0)


def wait_for_signal(mut mask: SignalSet) -> SysResult:
    """Block until one of the signals in `mask` arrives, and return which.

    Correct only in a process where every thread blocks the signal, which a
    Mojo 1.0 binary is not, so molla's shutdown path uses `SignalChannel`
    instead. Kept because it is the primitive that should be used the day the
    runtime lets us mask its threads, and because a wrapper that exists and is
    documented is better than one that gets written from scratch under time
    pressure."""
    var slot = List[c_int]()
    slot.append(c_int(0))
    var rc = Int(external_call["sigwait", c_int](mask.raw(), slot.unsafe_ptr()))
    if rc != 0:
        return SysResult(-1, rc)
    return ok(Int(slot[0]))


def signal_name(signum: Int) -> String:
    """A short name for the signals molla handles."""
    if signum == SIGINT:
        return "SIGINT"
    if signum == SIGQUIT:
        return "SIGQUIT"
    if signum == SIGPIPE:
        return "SIGPIPE"
    if signum == SIGALRM:
        return "SIGALRM"
    if signum == SIGTERM:
        return "SIGTERM"
    if signum == SIGKILL:
        return "SIGKILL"
    return "signal " + String(signum)


def getpid() -> Int:
    """This process."""
    return Int(external_call["getpid", c_int]())


def getppid() -> Int:
    """The parent. Worth logging when molla is run under a supervisor."""
    return Int(external_call["getppid", c_int]())


def kill(pid: Int, signum: Int) -> SysResult:
    """Send a signal. `kill(pid, 0)` asks whether the process is still there
    without sending anything, which is how a supervisor check is written."""
    return checked(Int(external_call["kill", c_int](c_int(pid), c_int(signum))))


comptime SIGNAL_WAKE_FD = 900
"""The descriptor number the handler writes to.

A compile time constant is the only thing a signal handler can refer to in a
language with no globals, so molla reserves one. 900 is high enough to be clear
of anything a normal process opens and low enough to be under the default soft
limit everywhere. `open_signal_channel` checks it is free before taking it,
because `dup2` onto a live descriptor closes that descriptor without a word."""

comptime SIGNAL_BYTE_LIMIT = 64
"""How many pending signal bytes one drain reads. More than will ever be
waiting, since there are two signals molla catches and a handler that runs
faster than a person can press the key twice."""


def on_signal(signum: Int32) abi("C") -> None:
    """The handler. Writes the signal number to the reserved descriptor.

    Everything this does is on the list of what a handler may call. No
    allocation, no locks, no `String`, no logging. The byte carries the signal
    number so the reader can say which one arrived, and a failed send is
    dropped on purpose: a full buffer means a byte is already waiting and the
    reader will wake for that one."""
    var b = stack_allocation[1, UInt8]()
    b.unsafe_store(0, UInt8(Int(signum)))
    _ = external_call["send", c_ssize_t](
        c_int(SIGNAL_WAKE_FD), b, c_size_t(1), c_int(0)
    )


comptime SignalHandler = def(Int32) thin abi("C") -> None


struct SignalChannel(Movable):
    """Signals delivered as bytes on a descriptor the reactor can poll."""

    var read_fd: Int
    var open: Bool

    def __init__(out self):
        self.read_fd = -1
        self.open = False

    def take(mut self) -> SysResult:
        """One pending signal number, or EAGAIN when there is nothing waiting.

        Non blocking, so this is safe to call from the event loop after the
        reader reports readable."""
        if not self.open:
            return SysResult(-1, EBADF)
        var buf = stack_allocation[1, UInt8]()
        var n = checked(
            Int(
                external_call["recv", c_ssize_t](
                    c_int(self.read_fd), buf, c_size_t(1), c_int(0)
                )
            )
        )
        if n.is_err():
            return n
        if n.value == 0:
            return SysResult(-1, EBADF)
        return ok(Int(buf.unsafe_load(0)))

    def close(mut self):
        """Give the descriptors back and stop catching anything.

        The reserved number is closed too, so a second `open_signal_channel` in
        the same process finds it free again. That is what makes the test
        suite able to run this more than once."""
        if not self.open:
            return
        _ = restore_default(SIGINT)
        _ = restore_default(SIGTERM)
        _ = close(SIGNAL_WAKE_FD)
        _ = close(self.read_fd)
        self.read_fd = -1
        self.open = False


def open_signal_channel(mut channel: SignalChannel) -> SysResult:
    """Reserve the descriptor and wire the handler to it.

    Does not install any handler yet. `catch_signal` does that, one signal at a
    time, so a caller that only wants SIGINT does not silently get SIGTERM as
    well."""
    if channel.open:
        return SysResult(-1, EBUSY)
    if get_flags(SIGNAL_WAKE_FD) >= 0:
        # Something already holds the number. Refusing is the only safe answer,
        # because dup2 would close it and whatever owned it would fail later
        # somewhere unrelated.
        return SysResult(-1, EBUSY)

    var pair = stack_allocation[2, c_int]()
    var made = checked(
        Int(
            external_call["socketpair", c_int](
                c_int(AF_UNIX), c_int(SOCK_STREAM), c_int(0), pair
            )
        )
    )
    if made.is_err():
        return made

    var read_end = Int(pair.unsafe_load(0))
    var write_end = Int(pair.unsafe_load(1))

    var duped = checked(
        Int(
            external_call["dup2", c_int](
                c_int(write_end), c_int(SIGNAL_WAKE_FD)
            )
        )
    )
    _ = close(write_end)
    if duped.is_err():
        _ = close(read_end)
        return duped

    try:
        # Non blocking on both ends. The handler must never block, and the
        # reader is polled rather than waited on.
        set_nonblocking(SIGNAL_WAKE_FD)
        set_nonblocking(read_end)
    except:
        _ = close(SIGNAL_WAKE_FD)
        _ = close(read_end)
        return SysResult(-1, EBADF)

    channel.read_fd = read_end
    channel.open = True
    return ok(read_end)


def catch_signal(signum: Int) -> SysResult:
    """Route one signal to the channel.

    Call `open_signal_channel` first. Installing a handler that writes to a
    descriptor nobody reserved would send the byte into whatever happens to be
    open on that number."""
    var handler: SignalHandler = on_signal
    var rc = Int(external_call["sigset", Int](c_int(signum), handler))
    if rc == SIG_ERR:
        return checked(-1)
    return ok(rc)
