"""Readiness polling: kqueue on macOS, epoll on Linux.

One `Poller` type, two implementations chosen at compile time. Nothing above
this module knows which one it got. That is the whole point of the file, because
the alternative is `comptime if` scattered through the event loop.

Both mechanisms are used in edge triggered mode. That means a readiness
notification arrives once when a socket goes from not ready to ready, and the
caller is obliged to read until EAGAIN before waiting again. Level triggered
would be more forgiving, but edge triggered is what keeps the wait call from
returning the same descriptor on every pass under load, and the discipline it
demands is the same discipline the HTTP layer needs anyway.

The struct layouts below are the fiddly part and there is no header to ask, so
they are written out. Two of them differ by platform in ways that are silent
when you get them wrong:

  kevent is 32 bytes on 64 bit macOS.
  epoll_event is packed to 12 bytes on x86_64 and 16 bytes everywhere else,
  because glibc only applies the packed attribute on x86_64 to stay compatible
  with the 32 bit ABI. Assuming 12 bytes on aarch64 reads the wrong half of
  every event and looks like random descriptors going ready.
"""

from std.ffi import c_int, external_call
from std.memory import stack_allocation
from std.sys.info import CompilationTarget

from molla.sys.errno import EINTR, errno_name, get_errno
from molla.sys.fd import close


def _is_kqueue() -> Bool:
    return CompilationTarget.is_macos()


comptime USES_KQUEUE = _is_kqueue()
"""True on macOS. Exposed so tests can name which backend they exercised."""

comptime EVFILT_READ: Int16 = -1
comptime EVFILT_WRITE: Int16 = -2
comptime EV_ADD: UInt16 = 0x0001
comptime EV_DELETE: UInt16 = 0x0002
comptime EV_CLEAR: UInt16 = 0x0020
comptime EV_ERROR: UInt16 = 0x4000
comptime EV_EOF: UInt16 = 0x8000

comptime EPOLL_CTL_ADD = 1
comptime EPOLL_CTL_DEL = 2
comptime EPOLL_CTL_MOD = 3
comptime EPOLLIN: UInt32 = 0x001
comptime EPOLLOUT: UInt32 = 0x004
comptime EPOLLERR: UInt32 = 0x008
comptime EPOLLHUP: UInt32 = 0x010
comptime EPOLLRDHUP: UInt32 = 0x2000
comptime EPOLLET: UInt32 = 0x80000000


def _event_size() -> Int:
    comptime if CompilationTarget.is_macos():
        return 32
    elif CompilationTarget.is_x86():
        return 12
    else:
        return 16


def _epoll_data_offset() -> Int:
    comptime if CompilationTarget.is_x86():
        return 4
    else:
        return 8


comptime EVENT_SIZE = _event_size()
comptime EPOLL_DATA_OFFSET = _epoll_data_offset()

comptime TIMESPEC_SIZE = 16
"""time_t plus long, both 8 bytes, on every platform molla targets."""


@fieldwise_init
struct Ready(Copyable, Movable):
    """One readiness notification, in terms both backends can express."""

    var fd: Int
    var readable: Bool
    var writable: Bool
    var hangup: Bool
    """The peer closed or the connection broke. The socket may still have
    buffered data to read, so a hangup is a reason to drain and then close, not
    a reason to close immediately."""


def _store_u64[
    o: MutOrigin
](buf: Pointer[UInt8, o], offset: Int, value: UInt64):
    for i in range(8):
        buf.unsafe_store(offset + i, UInt8((value >> (UInt64(i) * 8)) & 0xFF))


def _load_u64[o: MutOrigin](buf: Pointer[UInt8, o], offset: Int) -> UInt64:
    var value: UInt64 = 0
    for i in range(8):
        value |= UInt64(buf.unsafe_load(offset + i)) << (UInt64(i) * 8)
    return value


def _store_u32[
    o: MutOrigin
](buf: Pointer[UInt8, o], offset: Int, value: UInt32):
    for i in range(4):
        buf.unsafe_store(offset + i, UInt8((value >> (UInt32(i) * 8)) & 0xFF))


def _load_u32[o: MutOrigin](buf: Pointer[UInt8, o], offset: Int) -> UInt32:
    var value: UInt32 = 0
    for i in range(4):
        value |= UInt32(buf.unsafe_load(offset + i)) << (UInt32(i) * 8)
    return value


def _store_u16[
    o: MutOrigin
](buf: Pointer[UInt8, o], offset: Int, value: UInt16):
    buf.unsafe_store(offset, UInt8(value & 0xFF))
    buf.unsafe_store(offset + 1, UInt8((value >> 8) & 0xFF))


def _load_u16[o: MutOrigin](buf: Pointer[UInt8, o], offset: Int) -> UInt16:
    return UInt16(buf.unsafe_load(offset)) | (
        UInt16(buf.unsafe_load(offset + 1)) << 8
    )


struct Poller(Movable):
    """Edge triggered readiness for a set of descriptors.

    Not thread safe and not meant to be. The design in docs/design.md gives each
    I/O worker its own poller and its own set of connections, so no two threads
    ever touch the same one.
    """

    var fd: Int
    """kqueue or epoll descriptor. Negative once closed."""

    var max_events: Int
    var buffer: List[UInt8]
    var ready_count: Int

    def __init__(out self, max_events: Int) raises:
        self.max_events = max_events
        self.buffer = List[UInt8](length=max_events * EVENT_SIZE, fill=0)
        self.ready_count = 0

        comptime if USES_KQUEUE:
            self.fd = Int(external_call["kqueue", c_int]())
        else:
            self.fd = Int(external_call["epoll_create1", c_int](c_int(0)))

        if self.fd < 0:
            raise Error(
                "creating the poller failed: " + errno_name(get_errno())
            )

    def _kevent_change(mut self, fd: Int, filter: Int16, flags: UInt16) -> Int:
        """Submit one kqueue change with no event list. macOS only."""
        var change = stack_allocation[32, UInt8]()
        for i in range(32):
            change.unsafe_store(i, UInt8(0))
        _store_u64(change, 0, UInt64(fd))
        _store_u16(change, 8, UInt16(filter.cast[DType.uint16]()))
        _store_u16(change, 10, flags)

        # Zero this or lose an afternoon. `stack_allocation` does not
        # initialise, and kevent rejects the whole call with EINVAL if the
        # timespec it was handed has a tv_nsec outside 0 to 999999999. With
        # nevents at 0 the timeout is never waited on, so whatever garbage was
        # on the stack decides whether registering a descriptor works.
        var ts = stack_allocation[TIMESPEC_SIZE, UInt8]()
        for i in range(TIMESPEC_SIZE):
            ts.unsafe_store(i, UInt8(0))

        return Int(
            external_call["kevent", c_int](
                c_int(self.fd),
                change,
                c_int(1),
                stack_allocation[1, UInt8](),
                c_int(0),
                ts,
            )
        )

    def _epoll_ctl(mut self, op: Int, fd: Int, mask: UInt32) -> Int:
        """Submit one epoll change. Linux only."""
        var event = stack_allocation[16, UInt8]()
        for i in range(16):
            event.unsafe_store(i, UInt8(0))
        _store_u32(event, 0, mask)
        # The kernel hands this back untouched, so the descriptor rides in the
        # data field and there is no side table to keep in step.
        _store_u64(event, EPOLL_DATA_OFFSET, UInt64(fd))
        return Int(
            external_call["epoll_ctl", c_int](
                c_int(self.fd), c_int(op), c_int(fd), event
            )
        )

    def add_read(mut self, fd: Int) raises:
        """Watch a descriptor for readability, edge triggered."""
        comptime if USES_KQUEUE:
            if self._kevent_change(fd, EVFILT_READ, EV_ADD | EV_CLEAR) < 0:
                raise Error(
                    "kevent EV_ADD read on fd "
                    + String(fd)
                    + " failed: "
                    + errno_name(get_errno())
                )
        else:
            if (
                self._epoll_ctl(
                    EPOLL_CTL_ADD, fd, EPOLLIN | EPOLLRDHUP | EPOLLET
                )
                < 0
            ):
                raise Error(
                    "epoll_ctl ADD on fd "
                    + String(fd)
                    + " failed: "
                    + errno_name(get_errno())
                )

    def set_write_interest(mut self, fd: Int, wanted: Bool) raises:
        """Turn writability notifications on or off for a descriptor.

        This is what stops a short write from turning into a spin. When `send`
        cannot take everything, the remainder is buffered and write interest
        goes on. When the buffer drains it goes off again, because leaving it on
        means the poller reports a writable socket on every single pass.

        kqueue keeps read and write as separate filters, so this adds or deletes
        one. epoll has a single mask per descriptor, so this rewrites it.
        """
        comptime if USES_KQUEUE:
            var flags = (EV_ADD | EV_CLEAR) if wanted else EV_DELETE
            var rc = self._kevent_change(fd, EVFILT_WRITE, flags)
            # Deleting a filter that was never added returns ENOENT, which is
            # the state we wanted anyway.
            if rc < 0 and wanted:
                raise Error(
                    "kevent EV_ADD write on fd "
                    + String(fd)
                    + " failed: "
                    + errno_name(get_errno())
                )
        else:
            var mask = EPOLLIN | EPOLLRDHUP | EPOLLET
            if wanted:
                mask |= EPOLLOUT
            if self._epoll_ctl(EPOLL_CTL_MOD, fd, mask) < 0:
                raise Error(
                    "epoll_ctl MOD on fd "
                    + String(fd)
                    + " failed: "
                    + errno_name(get_errno())
                )

    def remove(mut self, fd: Int):
        """Stop watching a descriptor.

        Best effort on purpose. Closing a descriptor already drops it from both
        a kqueue and an epoll set, so the common path calls this and then close,
        and a failure here means it was already gone.
        """
        comptime if USES_KQUEUE:
            _ = self._kevent_change(fd, EVFILT_READ, EV_DELETE)
            _ = self._kevent_change(fd, EVFILT_WRITE, EV_DELETE)
        else:
            _ = self._epoll_ctl(EPOLL_CTL_DEL, fd, 0)

    def wait(mut self, timeout_ms: Int) raises -> Int:
        """Block until something is ready or the timeout expires.

        Returns how many events are available, which may be zero on timeout.
        EINTR is turned into zero rather than an error, because a signal
        arriving is not a reason to tear down a server.
        """
        var count: Int
        comptime if USES_KQUEUE:
            var ts = stack_allocation[TIMESPEC_SIZE, UInt8]()
            _store_u64(ts, 0, UInt64(timeout_ms // 1000))
            _store_u64(ts, 8, UInt64((timeout_ms % 1000) * 1000000))
            count = Int(
                external_call["kevent", c_int](
                    c_int(self.fd),
                    stack_allocation[1, UInt8](),
                    c_int(0),
                    self.buffer.unsafe_ptr(),
                    c_int(self.max_events),
                    ts,
                )
            )
        else:
            count = Int(
                external_call["epoll_wait", c_int](
                    c_int(self.fd),
                    self.buffer.unsafe_ptr(),
                    c_int(self.max_events),
                    c_int(timeout_ms),
                )
            )

        if count < 0:
            var code = get_errno()
            if code == EINTR:
                self.ready_count = 0
                return 0
            raise Error("waiting on the poller failed: " + errno_name(code))

        self.ready_count = count
        return count

    def event(mut self, index: Int) -> Ready:
        """Decode the event at `index` from the last `wait`."""
        var base = index * EVENT_SIZE
        var buf = self.buffer.unsafe_ptr()

        comptime if USES_KQUEUE:
            var fd = Int(_load_u64(buf, base + 0))
            var filter = _load_u16(buf, base + 8).cast[DType.int16]()
            var flags = _load_u16(buf, base + 10)
            # EV_ERROR means the change itself was rejected, and the reason sits
            # in data. Both that and EOF are reported as a hangup, since either
            # way the caller drains and closes.
            var bad = (flags & EV_ERROR) != 0
            var eof = (flags & EV_EOF) != 0
            return Ready(
                fd=fd,
                readable=filter == EVFILT_READ,
                writable=filter == EVFILT_WRITE,
                hangup=bad or eof,
            )
        else:
            var events = _load_u32(buf, base + 0)
            var fd = Int(_load_u64(buf, base + EPOLL_DATA_OFFSET))
            var gone = (events & (EPOLLHUP | EPOLLERR | EPOLLRDHUP)) != 0
            # A hangup counts as readable so the caller drains whatever is still
            # buffered and then sees the zero byte read for itself. epoll does
            # not always set EPOLLIN alongside EPOLLHUP.
            return Ready(
                fd=fd,
                readable=(events & EPOLLIN) != 0 or gone,
                writable=(events & EPOLLOUT) != 0,
                hangup=gone,
            )

    def shutdown(mut self):
        """Close the poller. Safe to call more than once."""
        if self.fd >= 0:
            _ = close(self.fd)
            self.fd = -1
