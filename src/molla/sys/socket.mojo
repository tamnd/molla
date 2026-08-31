"""IPv4 TCP sockets over libc.

There is no `std.net` in Mojo 1.0, so this is the bottom of molla's network
stack. It is deliberately small: enough to open a listener, accept, connect, and
find out what port the kernel gave us. Everything above this works in terms of
descriptors and never calls libc directly.

The awkward part is sockaddr_in. It is 16 bytes on both platforms but they do
not agree on the layout: BSD put a length byte at the front and shrank the
family field to one byte, Linux kept a two byte family. Writing the struct field
by field into a byte buffer, with the difference handled at compile time, is
uglier than declaring a struct but it is the version that is obviously correct
on both. Port and address are network byte order, which is big endian, so the
bytes go in most significant first regardless of what the host does.
"""

from std.ffi import c_int, c_size_t, c_ssize_t, external_call
from std.memory import stack_allocation
from std.sys.info import CompilationTarget

from molla.sys.errno import EINVAL, ERANGE, errno_name, get_errno
from molla.sys.fd import close, set_nonblocking
from molla.sys.result import SysResult, checked, ok

comptime AF_INET = 2
comptime SOCK_STREAM = 1
comptime SOCKADDR_IN_SIZE = 16


def _af_unix() -> Int:
    comptime if CompilationTarget.is_macos():
        return 1
    else:
        return 1


comptime AF_UNIX = _af_unix()
"""One on both platforms. Written out anyway, because every other address
family constant in this file needed a per platform answer and a reader should
not have to guess which ones were checked."""

comptime SOCKADDR_UN_PATH_OFF = 2
"""Where the path starts in a sockaddr_un. BSD spends the first two bytes on a
length and a one byte family, Linux spends them on a two byte family, so the
path lands at the same offset for opposite reasons."""

comptime SOCKADDR_UN_SIZE = 106
"""macOS allows 104 path bytes and Linux allows 108. Using the smaller one
everywhere means a path that works on macOS works on Linux, which is the
direction that matters, and it is still longer than any socket path molla
creates."""


def _sol_socket() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0xFFFF
    else:
        return 1


def _so_reuseaddr() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0x0004
    else:
        return 2


def _so_reuseport() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0x0200
    else:
        return 15


def _so_keepalive() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0x0008
    else:
        return 9


comptime IPPROTO_TCP = 6
comptime TCP_NODELAY = 1
"""Same numbers on macOS and Linux, unlike most of the socket option space."""

comptime SOL_SOCKET = _sol_socket()
comptime SO_REUSEADDR = _so_reuseaddr()
comptime SO_REUSEPORT = _so_reuseport()
comptime SO_KEEPALIVE = _so_keepalive()
"""8 on macOS and 9 on Linux. The two platforms agree on TCP_NODELAY and on
almost nothing else in this space."""

comptime INADDR_LOOPBACK: UInt32 = 0x7F000001
"""127.0.0.1 in host order. molla binds loopback by default, see D9."""

comptime INADDR_ANY: UInt32 = 0


def write_sockaddr_in[
    o: MutOrigin
](buf: Pointer[UInt8, o], addr_host_order: UInt32, port: UInt16):
    """Fill 16 bytes with an IPv4 sockaddr for the given address and port.

    `buf` must have room for SOCKADDR_IN_SIZE bytes. Everything is zeroed first
    so the trailing pad is clean, which matters because some kernels compare it.
    """
    for i in range(SOCKADDR_IN_SIZE):
        buf.unsafe_store(i, UInt8(0))

    comptime if CompilationTarget.is_macos():
        # BSD layout: one length byte, then a one byte family.
        buf.unsafe_store(0, UInt8(SOCKADDR_IN_SIZE))
        buf.unsafe_store(1, UInt8(AF_INET))
    else:
        # Linux layout: a two byte family in host order, which is little endian
        # on every platform we support, so the low byte goes first.
        buf.unsafe_store(0, UInt8(AF_INET))
        buf.unsafe_store(1, UInt8(0))

    # Port and address are big endian on the wire, on every host.
    buf.unsafe_store(2, UInt8((port >> 8) & 0xFF))
    buf.unsafe_store(3, UInt8(port & 0xFF))
    buf.unsafe_store(4, UInt8((addr_host_order >> 24) & 0xFF))
    buf.unsafe_store(5, UInt8((addr_host_order >> 16) & 0xFF))
    buf.unsafe_store(6, UInt8((addr_host_order >> 8) & 0xFF))
    buf.unsafe_store(7, UInt8(addr_host_order & 0xFF))


def read_port[o: MutOrigin](buf: Pointer[UInt8, o]) -> UInt16:
    """Pull the port back out of a sockaddr_in. Same offset on both platforms.
    """
    return (UInt16(buf.unsafe_load(2)) << 8) | UInt16(buf.unsafe_load(3))


def socket_tcp() raises -> Int:
    """Create a non blocking IPv4 TCP socket."""
    var fd = Int(
        external_call["socket", c_int](
            c_int(AF_INET), c_int(SOCK_STREAM), c_int(0)
        )
    )
    if fd < 0:
        raise Error("socket failed: " + errno_name(get_errno()))
    set_nonblocking(fd)
    return fd


def set_reuse(fd: Int, option: Int) raises:
    """Set a boolean SOL_SOCKET option."""
    var on = stack_allocation[1, c_int]()
    on.unsafe_store(0, c_int(1))
    var rc = Int(
        external_call["setsockopt", c_int](
            c_int(fd),
            c_int(SOL_SOCKET),
            c_int(option),
            on,
            c_int(4),
        )
    )
    if rc < 0:
        raise Error("setsockopt failed: " + errno_name(get_errno()))


def set_nodelay(fd: Int) raises:
    """Turn off Nagle's algorithm on an accepted socket.

    Nagle holds a small write back until the previous small segment has been
    acknowledged, and the peer's stack delays that acknowledgement hoping to
    piggyback it on data going the other way. For bulk transfer that trade is
    worth it. For a request and response server it stalls a path that otherwise
    answers in tens of microseconds.

    Measured on the M4 against the fixed body, with the buffer churn described
    in `http/server.mojo` already fixed, this is worth 39006 to 95102 requests
    per second at 64 connections and takes the p99 at one connection from
    14.81ms to 2.29ms. The first time it was measured the buffer churn was
    still there, it swamped this completely, and turning Nagle off looked like
    it did nothing. Worth remembering before concluding a knob does not matter.
    """
    var on = stack_allocation[1, c_int]()
    on.unsafe_store(0, c_int(1))
    var rc = Int(
        external_call["setsockopt", c_int](
            c_int(fd),
            c_int(IPPROTO_TCP),
            c_int(TCP_NODELAY),
            on,
            c_int(4),
        )
    )
    if rc < 0:
        raise Error("setsockopt TCP_NODELAY failed: " + errno_name(get_errno()))


def _so_sndbuf() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0x1001
    else:
        return 7


def _so_rcvbuf() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0x1002
    else:
        return 8


comptime SO_SNDBUF = _so_sndbuf()
comptime SO_RCVBUF = _so_rcvbuf()


def set_buffer_size(fd: Int, option: Int, bytes: Int) raises:
    """Fix the kernel's send or receive buffer for a socket.

    Normally worth leaving alone. Both platforms size these themselves and
    Linux grows them over the life of a connection, and setting either one by
    hand turns that off, so a number chosen today is the number a connection
    still has when the network it runs on is faster.

    It is here because a listening socket passes both down to every socket
    accepted from it, which makes it the one honest way to test the write path
    under backpressure. Without it a test has to push enough bytes to fill
    whatever the kernel decided to give it, that number is different on every
    machine, and the test either takes megabytes or quietly stops testing
    anything.
    """
    var value = stack_allocation[1, c_int]()
    value.unsafe_store(0, c_int(bytes))
    var rc = Int(
        external_call["setsockopt", c_int](
            c_int(fd),
            c_int(SOL_SOCKET),
            c_int(option),
            value,
            c_int(4),
        )
    )
    if rc < 0:
        raise Error("setsockopt buffer size failed: " + errno_name(get_errno()))


def set_keepalive(fd: Int) raises:
    """Ask the kernel to probe an idle connection.

    This is not the idle timeout, which the reactor does itself with a timing
    wheel and much shorter deadlines. Keepalive is for the case the reactor
    cannot see: a peer whose machine went away without sending anything, where
    the socket stays open and readable forever because nothing tells us
    otherwise. The default interval is two hours on both platforms, so this is
    a backstop against descriptor leaks over days and not a latency control.
    """
    set_reuse(fd, SO_KEEPALIVE)


def listen_tcp(
    addr_host_order: UInt32, port: UInt16, backlog: Int
) raises -> Int:
    """Open a listening socket. Returns the descriptor.

    Pass port 0 to let the kernel pick one, then ask for it with `local_port`.
    That is what the tests do, so two test runs on the same machine cannot
    collide on a fixed port.
    """
    return listen_tcp_shared(addr_host_order, port, backlog, False)


def listen_tcp_shared(
    addr_host_order: UInt32, port: UInt16, backlog: Int, reuse_port: Bool
) raises -> Int:
    """The same, with SO_REUSEPORT as a choice rather than a default.

    REUSEPORT is how each I/O worker on Linux gets its own listening socket on
    one port, with the kernel hashing new connections across them, which is
    what removes the single accept queue as a contention point. It is off
    unless asked for, because a stray REUSEPORT listener would let a second
    molla silently take half the traffic from a running one, and that is a
    confusing morning for whoever has to work out why half the requests are
    answered by yesterday's build.
    """
    var fd = socket_tcp()

    try:
        set_reuse(fd, SO_REUSEADDR)
        if reuse_port:
            set_reuse(fd, SO_REUSEPORT)
    except e:
        _ = close(fd)
        raise e

    var sa = stack_allocation[SOCKADDR_IN_SIZE, UInt8]()
    write_sockaddr_in(sa, addr_host_order, port)

    var rc = Int(
        external_call["bind", c_int](c_int(fd), sa, c_int(SOCKADDR_IN_SIZE))
    )
    if rc < 0:
        var code = get_errno()
        _ = close(fd)
        raise Error(
            "bind to port " + String(port) + " failed: " + errno_name(code)
        )

    rc = Int(external_call["listen", c_int](c_int(fd), c_int(backlog)))
    if rc < 0:
        var code = get_errno()
        _ = close(fd)
        raise Error("listen failed: " + errno_name(code))

    return fd


def local_port(fd: Int) raises -> UInt16:
    """The port a socket is actually bound to.

    Needed because binding to port 0 is the only way to get a port that is
    definitely free, and after that we have to ask the kernel which one it was.
    """
    var sa = stack_allocation[SOCKADDR_IN_SIZE, UInt8]()
    var len_out = stack_allocation[1, c_int]()
    len_out.unsafe_store(0, c_int(SOCKADDR_IN_SIZE))
    var rc = Int(external_call["getsockname", c_int](c_int(fd), sa, len_out))
    if rc < 0:
        raise Error("getsockname failed: " + errno_name(get_errno()))
    return read_port(sa)


def accept(fd: Int) -> Int:
    """Accept one pending connection, or return -1 with errno set.

    EAGAIN here is the normal way the accept loop learns the backlog is drained,
    so this does not raise. There is no accept4 on macOS, so the caller sets non
    blocking mode on the result itself.

    The peer address is read into a local buffer and thrown away. libc will take
    a null pointer here, but passing a real buffer costs 16 stack bytes and is
    what we want anyway once connections get logged.
    """
    var peer = stack_allocation[SOCKADDR_IN_SIZE, UInt8]()
    var peer_len = stack_allocation[1, c_int]()
    peer_len.unsafe_store(0, c_int(SOCKADDR_IN_SIZE))
    return Int(external_call["accept", c_int](c_int(fd), peer, peer_len))


comptime MSG_NONE = 0
"""No flags. molla does not use MSG_PEEK or out of band data anywhere."""


def recv[o: MutOrigin](fd: Int, buf: Pointer[UInt8, o], count: Int) -> Int:
    """Read up to `count` bytes. Returns bytes read, 0 at end of stream, or -1.

    Zero means the peer closed its side. That is a normal event, not an error,
    and callers have to tell it apart from -1 with EAGAIN.
    """
    return Int(
        external_call["recv", c_ssize_t](
            c_int(fd), buf, c_size_t(count), c_int(MSG_NONE)
        )
    )


def send[o: MutOrigin](fd: Int, buf: Pointer[UInt8, o], count: Int) -> Int:
    """Write up to `count` bytes. Returns bytes written, which can be fewer than
    asked for on a non blocking socket, or -1."""
    return Int(
        external_call["send", c_ssize_t](
            c_int(fd), buf, c_size_t(count), c_int(MSG_NONE)
        )
    )


def connect(fd: Int, addr_host_order: UInt32, port: UInt16) -> Int:
    """Start a connection. Returns 0, or -1 with errno EINPROGRESS while a non
    blocking connect is still in flight."""
    var sa = stack_allocation[SOCKADDR_IN_SIZE, UInt8]()
    write_sockaddr_in(sa, addr_host_order, port)
    return Int(
        external_call["connect", c_int](c_int(fd), sa, c_int(SOCKADDR_IN_SIZE))
    )


def _so_rcvtimeo() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0x1006
    else:
        return 20


def _so_sndtimeo() -> Int:
    comptime if CompilationTarget.is_macos():
        return 0x1005
    else:
        return 21


comptime SO_RCVTIMEO = _so_rcvtimeo()
comptime SO_SNDTIMEO = _so_sndtimeo()

comptime TIMEVAL_SIZE = 16
"""struct timeval is 16 bytes on both platforms, though not for the same reason.
Linux has two 8 byte fields. macOS has an 8 byte seconds field and a 4 byte
microseconds field with 4 bytes of padding. Writing seconds into the first 8
bytes and zeroing the rest is correct on both."""


def set_timeout(fd: Int, option: Int, seconds: Int) raises:
    """Put a wall clock limit on a blocking send or receive."""
    var tv = stack_allocation[TIMEVAL_SIZE, UInt8]()
    for i in range(TIMEVAL_SIZE):
        tv.unsafe_store(i, UInt8(0))
    tv.unsafe_bitcast[Int64]().unsafe_store(0, Int64(seconds))
    var rc = Int(
        external_call["setsockopt", c_int](
            c_int(fd),
            c_int(SOL_SOCKET),
            c_int(option),
            tv,
            c_int(TIMEVAL_SIZE),
        )
    )
    if rc < 0:
        raise Error("setsockopt timeout failed: " + errno_name(get_errno()))


def dial(
    addr_host_order: UInt32, port: UInt16, timeout_seconds: Int
) raises -> Int:
    """Open a blocking connected TCP socket, with send and receive timeouts.

    Blocking on purpose. The server side is non blocking because it holds
    thousands of connections in one thread, and none of that applies to a client
    that opens one connection, pulls a blob and closes it. A blocking socket also
    makes the TLS layer above simple, since neither OpenSSL nor Secure Transport
    then has to be driven through a retry loop for every partial record. The
    timeouts are what stops a dead peer hanging the process forever.
    """
    var fd = Int(
        external_call["socket", c_int](
            c_int(AF_INET), c_int(SOCK_STREAM), c_int(0)
        )
    )
    if fd < 0:
        raise Error("socket failed: " + errno_name(get_errno()))

    try:
        set_timeout(fd, SO_RCVTIMEO, timeout_seconds)
        set_timeout(fd, SO_SNDTIMEO, timeout_seconds)
        set_nodelay(fd)
    except e:
        _ = close(fd)
        raise e

    if connect(fd, addr_host_order, port) < 0:
        var code = get_errno()
        _ = close(fd)
        raise Error("connect failed: " + errno_name(code))
    return fd


comptime SHUT_RD = 0
comptime SHUT_WR = 1
comptime SHUT_RDWR = 2
"""Same three numbers on both platforms."""


def shutdown(fd: Int, how: Int) -> Int:
    """Half close a connection.

    `SHUT_WR` is how a server says "that is the whole response" on a connection
    it does not want to close yet, and it is the only one molla uses. Returns
    -1 with errno set, and ENOTCONN here is normal rather than a fault, because
    the peer may already have gone."""
    return Int(external_call["shutdown", c_int](c_int(fd), c_int(how)))


def get_peer_port(fd: Int) -> SysResult:
    """The port at the other end. For logging a connection."""
    var sa = stack_allocation[SOCKADDR_IN_SIZE, UInt8]()
    var len_out = stack_allocation[1, c_int]()
    len_out.unsafe_store(0, c_int(SOCKADDR_IN_SIZE))
    var rc = checked(
        Int(external_call["getpeername", c_int](c_int(fd), sa, len_out))
    )
    if rc.is_err():
        return rc
    return ok(Int(read_port(sa)))


def socket_pair(mut out_ends: List[Int]) -> SysResult:
    """A connected pair of Unix domain sockets.

    Both ends are already connected, so this is the cheapest way to get a
    channel between two threads that the reactor can poll. Appends the two
    descriptors to `out_ends`, reading end first."""
    var pair = stack_allocation[2, c_int]()
    var rc = checked(
        Int(
            external_call["socketpair", c_int](
                c_int(AF_UNIX), c_int(SOCK_STREAM), c_int(0), pair
            )
        )
    )
    if rc.is_err():
        return rc
    out_ends.append(Int(pair.unsafe_load(0)))
    out_ends.append(Int(pair.unsafe_load(1)))
    return ok(2)


def write_sockaddr_un[
    o: MutOrigin
](buf: Pointer[UInt8, o], path: StringSpan) -> SysResult:
    """Fill a sockaddr_un for a filesystem path.

    Fails with ERANGE rather than truncating. A truncated socket path binds to
    a different socket than the one that was asked for, silently, and the two
    processes then wait for each other on different files."""
    if path.byte_length() + 1 > SOCKADDR_UN_SIZE - SOCKADDR_UN_PATH_OFF:
        return SysResult(-1, ERANGE)

    for i in range(SOCKADDR_UN_SIZE):
        buf.unsafe_store(i, UInt8(0))

    comptime if CompilationTarget.is_macos():
        buf.unsafe_store(0, UInt8(SOCKADDR_UN_SIZE))
        buf.unsafe_store(1, UInt8(AF_UNIX))
    else:
        buf.unsafe_store(0, UInt8(AF_UNIX))
        buf.unsafe_store(1, UInt8(0))

    var p = path.unsafe_ptr()
    for i in range(path.byte_length()):
        buf.unsafe_store(SOCKADDR_UN_PATH_OFF + i, p.unsafe_load(i))
    return ok(path.byte_length())


def listen_unix(path: StringSpan, backlog: Int) raises -> Int:
    """Open a listening Unix domain socket at a path.

    The path has to be free. bind fails with EADDRINUSE on a leftover socket
    file, including one left by a process that crashed, and unlinking it here
    would race with a live server that is using it. That is the caller's
    decision to make with a lock file, not this layer's."""
    var fd = Int(
        external_call["socket", c_int](
            c_int(AF_UNIX), c_int(SOCK_STREAM), c_int(0)
        )
    )
    if fd < 0:
        raise Error("socket AF_UNIX failed: " + errno_name(get_errno()))
    set_nonblocking(fd)

    var sa = stack_allocation[SOCKADDR_UN_SIZE, UInt8]()
    var written = write_sockaddr_un(sa, path)
    if written.is_err():
        _ = close(fd)
        raise Error("socket path too long: " + String(path))

    var rc = Int(
        external_call["bind", c_int](c_int(fd), sa, c_int(SOCKADDR_UN_SIZE))
    )
    if rc < 0:
        var code = get_errno()
        _ = close(fd)
        raise Error("bind to " + String(path) + " failed: " + errno_name(code))

    rc = Int(external_call["listen", c_int](c_int(fd), c_int(backlog)))
    if rc < 0:
        var code = get_errno()
        _ = close(fd)
        raise Error("listen failed: " + errno_name(code))
    return fd


def connect_unix(path: StringSpan) raises -> Int:
    """Connect to a Unix domain socket. Blocking, for the CLI talking to a
    server on the same machine."""
    var fd = Int(
        external_call["socket", c_int](
            c_int(AF_UNIX), c_int(SOCK_STREAM), c_int(0)
        )
    )
    if fd < 0:
        raise Error("socket AF_UNIX failed: " + errno_name(get_errno()))

    var sa = stack_allocation[SOCKADDR_UN_SIZE, UInt8]()
    var written = write_sockaddr_un(sa, path)
    if written.is_err():
        _ = close(fd)
        raise Error("socket path too long: " + String(path))

    var rc = Int(
        external_call["connect", c_int](c_int(fd), sa, c_int(SOCKADDR_UN_SIZE))
    )
    if rc < 0:
        var code = get_errno()
        _ = close(fd)
        raise Error(
            "connect to " + String(path) + " failed: " + errno_name(code)
        )
    return fd


comptime IOVEC_SIZE = 16
"""struct iovec is a pointer and a length, so 16 bytes on both platforms."""

comptime MAX_IOV = 8
"""How many pieces one vectored write may have. Both platforms allow at least
1024, but molla's writer never has more than a status line, headers and a body,
and a fixed small number keeps the buffer on the stack."""


def write_vectored(fd: Int, bases: List[Int], lengths: List[Int]) -> SysResult:
    """Write several buffers in one call.

    This is what keeps a response from being copied into one buffer before it
    goes out. Headers live in one place and the body in another, and `writev`
    puts them on the wire in order without either of them moving.

    `writev` rather than `sendmsg`, because molla sends no control messages and
    passes no file descriptors, and the two do the same thing for plain bytes
    with `sendmsg` needing a `struct msghdr` whose layout differs across the
    platforms. Partial writes are normal on a non blocking socket, so the
    caller has to look at the count and resume."""
    var count = len(bases)
    if count != len(lengths):
        return SysResult(-1, EINVAL)
    if count == 0:
        return ok(0)
    if count > MAX_IOV:
        return SysResult(-1, EINVAL)

    var iov = stack_allocation[MAX_IOV * IOVEC_SIZE, UInt8]()
    var slots = iov.unsafe_bitcast[Int64]()
    for i in range(count):
        slots.unsafe_store(i * 2, Int64(bases[i]))
        slots.unsafe_store(i * 2 + 1, Int64(lengths[i]))

    return checked(
        Int(external_call["writev", c_ssize_t](c_int(fd), iov, c_int(count)))
    )
