"""Threads, mutexes and condition variables over pthreads.

Mojo 1.0 has no threading module. There is nothing to spawn a thread with and
nothing to synchronise one with, which is the gap that left the third reversal
condition on D1 untestable at the M0 gate. This module closes it at the FFI
boundary. The queues and the shutdown protocol built on top are issue #15.

pthreads does not use errno. Every call returns the error number directly and
returns 0 on success, so the wrappers here put that number in `SysResult.err`
and leave `value` at -1. Reading errno after a failed `pthread_mutex_lock` gives
whatever the last unrelated syscall left there, which is the kind of wrong that
looks right in a log line.

Mutexes and condition variables live in storage this module allocates from libc
and frees itself, rather than on the stack or inside a struct's own bytes.
pthreads records the address of the object it initialised, so anything that
moves after `init` is broken in a way that shows up as a hang under contention
and never in a single threaded test. The first version of this used a `List` for
that storage, on the reasoning that a list's buffer stays put when the list
moves. It does, but the list itself is subject to Mojo's ownership rules, and a
worker holding the address is not a use the compiler can see. Owning the block
directly is the version where the address is stable because we made it stable.

Thread naming and CPU affinity are the two places the platforms genuinely
differ rather than just disagreeing on a constant. macOS can only name the
calling thread, and it has no CPU affinity API at all, so `set_affinity` there
reports ENOTSUP rather than pretending.
"""

from std.ffi import c_int, c_size_t, external_call
from std.sys.info import CompilationTarget

from molla.sys.cstr import c_string
from molla.sys.errno import ENOMEM, ENOTSUP
from molla.sys.result import SysResult, checked, ok

comptime ThreadFunc = def(Int) thin abi("C") -> Int
"""What pthread_create takes: one pointer argument, one pointer return, C ABI.

A Mojo function converts to this by assignment, `var f: ThreadFunc = entry`,
and only by assignment. Calling the alias as a constructor does not compile.
Everything a thread needs has to arrive through that single argument, so the
caller passes an integer or an address it keeps alive itself."""

comptime SYNC_STORAGE = 64
"""Room for a pthread_mutex_t or a pthread_cond_t on any target. macOS mutexes
are 64 bytes, Linux ones are 40 on x86-64 and 48 on arm64, and the condition
variables are 48 everywhere. One size covers all of it, and over allocating a
few bytes per lock is not a cost worth chasing."""

comptime MAX_THREAD_NAME = 15
"""Linux truncates at 15 characters plus the terminator and returns ERANGE if
you hand it more, so molla truncates rather than failing on a long name."""


def _pthread_result(rc: Int) -> SysResult:
    """Turn a pthreads return code into a result. Zero means success."""
    if rc != 0:
        return SysResult(-1, rc)
    return ok(0)


def _alloc_sync() -> Int:
    """A zeroed block for one pthreads object, at an address that will not move.

    `calloc` rather than `malloc` because pthreads reads the whole object during
    init on some platforms, and a block full of whatever was there before is a
    bad thing to hand it. Returns 0 if the allocation failed, which every caller
    turns into ENOMEM rather than dereferencing."""
    return Int(
        external_call["calloc", Int](c_size_t(1), c_size_t(SYNC_STORAGE))
    )


def _free_sync(address: Int):
    """Give the block back. Free of 0 is defined and does nothing, so the
    failed-allocation path does not need its own branch."""
    external_call["free", NoneType](address)


struct Thread(Movable):
    """One OS thread.

    Not `Copyable`. Two copies of a handle means two joins, and the second one
    is undefined behaviour rather than an error."""

    var handle: Int
    """pthread_t. Pointer sized on macOS and an unsigned long on Linux, opaque
    on both, so it travels as an integer and is never dereferenced here."""

    var live: Bool

    def __init__(out self):
        self.handle = 0
        self.live = False

    def join(mut self) -> SysResult:
        """Wait for the thread and collect what its entry function returned."""
        if not self.live:
            return ok(0)
        var slot = List[Int]()
        slot.append(0)
        var rc = Int(
            external_call["pthread_join", c_int](self.handle, slot.unsafe_ptr())
        )
        self.live = False
        if rc != 0:
            return SysResult(-1, rc)
        return ok(slot[0])

    def detach(mut self) -> SysResult:
        """Give up the right to join. The thread cleans itself up when it ends.
        """
        if not self.live:
            return ok(0)
        var rc = Int(external_call["pthread_detach", c_int](self.handle))
        self.live = False
        return _pthread_result(rc)


def spawn(entry: ThreadFunc, arg: Int, mut out_thread: Thread) -> SysResult:
    """Start a thread running `entry(arg)`.

    An out parameter rather than a return value, because a `Thread` is not
    copyable and returning it alongside a result would mean a tuple that cannot
    be taken apart in Mojo 1.0."""
    var slot = List[Int]()
    slot.append(0)
    var rc = Int(
        external_call["pthread_create", c_int](
            slot.unsafe_ptr(), Int(0), entry, arg
        )
    )
    if rc != 0:
        return SysResult(-1, rc)
    out_thread.handle = slot[0]
    out_thread.live = True
    return ok(0)


def self_id() -> Int:
    """This thread's pthread_t, for logging and for naming."""
    return Int(external_call["pthread_self", Int]())


def set_thread_name(name: StringSpan) -> SysResult:
    """Name the calling thread, for `top`, `ps -L` and crash reports.

    Only the calling thread, on purpose. macOS can only name itself and Linux
    can name any thread, so the intersection is the version that behaves the
    same on both, and every caller molla has names itself on entry anyway."""
    var trimmed = name
    if trimmed.byte_length() > MAX_THREAD_NAME:
        trimmed = name[byte=0:MAX_THREAD_NAME]
    var buf = c_string(trimmed)
    var rc: Int
    comptime if CompilationTarget.is_macos():
        rc = Int(external_call["pthread_setname_np", c_int](buf.unsafe_ptr()))
    else:
        rc = Int(
            external_call["pthread_setname_np", c_int](
                external_call["pthread_self", Int](), buf.unsafe_ptr()
            )
        )
    _ = buf^
    return _pthread_result(rc)


comptime CPU_SET_SIZE = 128
"""cpu_set_t is 1024 bits on Linux. Enough for any machine molla will meet."""


def set_affinity(cpu: Int) -> SysResult:
    """Pin the calling thread to one core.

    Linux only. macOS has no affinity API that a normal process can use, and
    the thread policy interface that exists is a hint the scheduler is free to
    ignore, so this reports ENOTSUP there instead of pretending it worked. The
    reactor treats a failure as advisory: one reactor per core is a performance
    choice, not a correctness one."""
    comptime if CompilationTarget.is_macos():
        return SysResult(-1, ENOTSUP)
    else:
        var mask = List[UInt8]()
        for _ in range(CPU_SET_SIZE):
            mask.append(0)
        var byte = cpu // 8
        if byte >= CPU_SET_SIZE:
            return SysResult(-1, ENOTSUP)
        mask[byte] = UInt8(1 << (cpu % 8))
        return checked(
            Int(
                external_call["sched_setaffinity", c_int](
                    c_int(0), c_size_t(CPU_SET_SIZE), mask.unsafe_ptr()
                )
            )
        )


def _sc_nprocessors_onln() -> Int:
    comptime if CompilationTarget.is_macos():
        return 58
    else:
        return 84


comptime SC_NPROCESSORS_ONLN = _sc_nprocessors_onln()


def cpu_count() -> Int:
    """Online cores, or 1 if the call fails.

    Falling back to 1 rather than raising, because every caller uses this to
    decide how many reactors to start and one reactor is a working server."""
    var n = Int(external_call["sysconf", Int](c_int(SC_NPROCESSORS_ONLN)))
    if n < 1:
        return 1
    return n


def sched_yield() -> SysResult:
    """Give up the rest of this time slice."""
    return checked(Int(external_call["sched_yield", c_int]()))


struct MutexRef(Copyable, ImplicitlyCopyable, Movable):
    """A mutex by address, for a thread that was handed one.

    A thread entry function takes a single integer, so a worker cannot be given
    a `Mutex` itself. It gets the address and wraps it in this, which is a
    borrow rather than ownership: it never initialises and never destroys."""

    var address: Int

    def __init__(out self, address: Int):
        self.address = address

    def lock(self) -> SysResult:
        return _pthread_result(
            Int(external_call["pthread_mutex_lock", c_int](self.address))
        )

    def try_lock(self) -> SysResult:
        return _pthread_result(
            Int(external_call["pthread_mutex_trylock", c_int](self.address))
        )

    def unlock(self) -> SysResult:
        return _pthread_result(
            Int(external_call["pthread_mutex_unlock", c_int](self.address))
        )


struct CondvarRef(Copyable, ImplicitlyCopyable, Movable):
    """A condition variable by address. Same reason as `MutexRef`."""

    var address: Int

    def __init__(out self, address: Int):
        self.address = address

    def wait(self, mutex: MutexRef) -> SysResult:
        return _pthread_result(
            Int(
                external_call["pthread_cond_wait", c_int](
                    self.address, mutex.address
                )
            )
        )

    def signal(self) -> SysResult:
        return _pthread_result(
            Int(external_call["pthread_cond_signal", c_int](self.address))
        )

    def broadcast(self) -> SysResult:
        return _pthread_result(
            Int(external_call["pthread_cond_broadcast", c_int](self.address))
        )


struct Mutex(Movable):
    """A pthread mutex in stable heap storage.

    Default attributes, which means not recursive and not error checking. A
    thread that locks the same mutex twice deadlocks rather than being told
    about it, and that is the standard behaviour molla codes against."""

    var address: Int
    """The block libc gave us. Zero means the allocation failed, and every
    method below reports ENOMEM rather than calling pthreads with a null."""

    var live: Bool

    def __init__(out self):
        self.address = _alloc_sync()
        self.live = False

    def __deinit__(deinit self):
        """Frees the block. A move takes the address with it and does not run
        this, so a `Mutex` that has been moved into a list is not freed twice.
        """
        if self.live:
            _ = external_call["pthread_mutex_destroy", c_int](self.address)
        _free_sync(self.address)

    def raw(self) -> Int:
        """The address pthreads knows this mutex by, and the one a worker gets.
        """
        return self.address

    def ref(self) -> MutexRef:
        """A handle another thread can be given."""
        return MutexRef(self.raw())

    def init(mut self) -> SysResult:
        if self.address == 0:
            return SysResult(-1, ENOMEM)
        var rc = Int(
            external_call["pthread_mutex_init", c_int](self.raw(), Int(0))
        )
        if rc == 0:
            self.live = True
        return _pthread_result(rc)

    def lock(mut self) -> SysResult:
        return _pthread_result(
            Int(external_call["pthread_mutex_lock", c_int](self.raw()))
        )

    def try_lock(mut self) -> SysResult:
        """Take the lock if it is free. Fails with EBUSY if it is not, which is
        an answer rather than an error."""
        return _pthread_result(
            Int(external_call["pthread_mutex_trylock", c_int](self.raw()))
        )

    def unlock(mut self) -> SysResult:
        return _pthread_result(
            Int(external_call["pthread_mutex_unlock", c_int](self.raw()))
        )

    def destroy(mut self) -> SysResult:
        if not self.live:
            return ok(0)
        self.live = False
        return _pthread_result(
            Int(external_call["pthread_mutex_destroy", c_int](self.raw()))
        )


struct Condvar(Movable):
    """A pthread condition variable, paired with a mutex the caller owns."""

    var address: Int
    var live: Bool

    def __init__(out self):
        self.address = _alloc_sync()
        self.live = False

    def __deinit__(deinit self):
        if self.live:
            _ = external_call["pthread_cond_destroy", c_int](self.address)
        _free_sync(self.address)

    def raw(self) -> Int:
        return self.address

    def ref(self) -> CondvarRef:
        """A handle another thread can be given."""
        return CondvarRef(self.raw())

    def init(mut self) -> SysResult:
        if self.address == 0:
            return SysResult(-1, ENOMEM)
        var rc = Int(
            external_call["pthread_cond_init", c_int](self.raw(), Int(0))
        )
        if rc == 0:
            self.live = True
        return _pthread_result(rc)

    def wait(mut self, mut mutex: Mutex) -> SysResult:
        """Release the mutex, sleep until signalled, take the mutex back.

        Spurious wakeups are allowed by the standard and do happen, so every
        caller loops on its own predicate rather than trusting the return."""
        return _pthread_result(
            Int(
                external_call["pthread_cond_wait", c_int](
                    self.raw(), mutex.raw()
                )
            )
        )

    def signal(mut self) -> SysResult:
        """Wake one waiter."""
        return _pthread_result(
            Int(external_call["pthread_cond_signal", c_int](self.raw()))
        )

    def broadcast(mut self) -> SysResult:
        """Wake every waiter. What shutdown uses."""
        return _pthread_result(
            Int(external_call["pthread_cond_broadcast", c_int](self.raw()))
        )

    def destroy(mut self) -> SysResult:
        if not self.live:
            return ok(0)
        self.live = False
        return _pthread_result(
            Int(external_call["pthread_cond_destroy", c_int](self.raw()))
        )


comptime TIMESPEC_SIZE = 16
comptime NANOS_PER_MILLI = 1000000
comptime NANOS_PER_SECOND = 1000000000


def sleep_ms(millis: Int) -> SysResult:
    """Sleep for a while. Interrupted sleeps return EINTR and the caller
    decides whether the remainder matters, which for molla it never does."""
    var ts = List[UInt8]()
    for _ in range(TIMESPEC_SIZE):
        ts.append(0)
    var p = ts.unsafe_ptr().unsafe_bitcast[Int64]()
    p.unsafe_store(0, Int64(millis // 1000))
    p.unsafe_store(1, Int64((millis % 1000) * NANOS_PER_MILLI))
    var rc = checked(
        Int(external_call["nanosleep", c_int](ts.unsafe_ptr(), Int(0)))
    )
    _ = ts^
    return rc
