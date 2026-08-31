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

from molla.sys.errno import errno_name, get_errno
from molla.sys.fd import close, set_nonblocking

comptime AF_INET = 2
comptime SOCK_STREAM = 1
comptime SOCKADDR_IN_SIZE = 16


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


comptime IPPROTO_TCP = 6
comptime TCP_NODELAY = 1
"""Same numbers on macOS and Linux, unlike most of the socket option space."""

comptime SOL_SOCKET = _sol_socket()
comptime SO_REUSEADDR = _so_reuseaddr()
comptime SO_REUSEPORT = _so_reuseport()

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


def listen_tcp(
    addr_host_order: UInt32, port: UInt16, backlog: Int
) raises -> Int:
    """Open a listening socket. Returns the descriptor.

    Pass port 0 to let the kernel pick one, then ask for it with `local_port`.
    That is what the tests do, so two test runs on the same machine cannot
    collide on a fixed port.
    """
    var fd = socket_tcp()

    # SO_REUSEADDR only, not SO_REUSEPORT. REUSEPORT would let a second molla
    # silently steal half the traffic from a running one, which is a confusing
    # failure mode to hand somebody. The listener sharing case can turn it on
    # explicitly when we get there.
    try:
        set_reuse(fd, SO_REUSEADDR)
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
