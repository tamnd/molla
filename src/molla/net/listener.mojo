"""Listening sockets, and the two ways to share one between workers.

There are two accept strategies here and which one runs is decided at compile
time, because the platforms genuinely differ and pretending otherwise costs
either correctness or throughput.

On Linux each worker opens its own listening socket on the same port with
SO_REUSEPORT, and the kernel hashes incoming connections across them. No shared
accept queue, no thundering herd, no handoff between threads, and a connection
is accepted by the thread that will serve it. This is the reason the option
exists and it is worth using.

On macOS SO_REUSEPORT exists and does something different. It lets several
sockets bind the same port, but it does not load balance: the last socket to
bind takes every connection. A server sharded that way on macOS would run all
its traffic on one worker and look fine in a test with one connection. So macOS
uses one listening socket, one thread accepting from it, and a round robin
handoff of accepted descriptors to the workers. That costs one queue push and
one wakeup byte per connection, which is real but small, and macOS is the
development platform rather than the deployment one.

The backlog is 1024. That is the queue of connections the kernel has completed
the handshake on and molla has not accepted yet, so it is the buffer that
absorbs a burst arriving faster than the accept loop drains it. Both platforms
silently clamp it to their own maximum, and asking for more than the system
allows is not an error, so this is a request rather than a guarantee.

Unix sockets go through the same reactor as TCP. The only differences are that
the path has to be removed before binding, since bind fails with EADDRINUSE on
a leftover file from a process that did not exit cleanly, and that there is no
Nagle to turn off.
"""

from std.sys.info import CompilationTarget

from molla.sys.fd import close, set_nonblocking
from molla.sys.file import exists, unlink
from molla.sys.socket import (
    INADDR_ANY,
    INADDR_LOOPBACK,
    listen_tcp_shared,
    listen_unix,
    local_port,
)

comptime BACKLOG = 1024
"""Pending connections the kernel may hold for us. Clamped by the system to
somn.max_syn_backlog on Linux and kern.ipc.somaxconn on macOS."""


def _shards_are_kernel_balanced() -> Bool:
    """Whether SO_REUSEPORT spreads connections rather than replacing the
    listener. True on Linux, false on macOS."""
    return not CompilationTarget.is_macos()


comptime SHARDED_ACCEPT = _shards_are_kernel_balanced()
"""True when each worker gets its own listening socket. False means one
acceptor and a round robin handoff, which is the macOS path."""

comptime LISTEN_TCP = 0
comptime LISTEN_UNIX = 1


struct ListenAddress(Copyable, ImplicitlyCopyable, Movable):
    """Where a server should listen. A tagged pair rather than a union,
    because Mojo 1.0 has no sum type worth the trouble for two cases."""

    var kind: Int
    var host: UInt32
    """Host order IPv4. Loopback unless something says otherwise, per D9."""

    var port: UInt16
    var path: String

    def __init__(out self, port: UInt16):
        self.kind = LISTEN_TCP
        self.host = INADDR_LOOPBACK
        self.port = port
        self.path = String("")

    def __init__(out self, host: UInt32, port: UInt16):
        self.kind = LISTEN_TCP
        self.host = host
        self.port = port
        self.path = String("")

    def __init__(out self, var path: String):
        self.kind = LISTEN_UNIX
        self.host = INADDR_ANY
        self.port = 0
        self.path = path^

    def describe(self) -> String:
        if self.kind == LISTEN_UNIX:
            return "unix:" + self.path
        return "tcp:" + String(self.port)


def open_listener(address: ListenAddress, shared: Bool) raises -> Int:
    """Open one listening socket, non blocking, ready for the poller.

    `shared` asks for SO_REUSEPORT, which is only meaningful where the kernel
    balances across the sharing sockets. It is ignored for unix sockets, where
    there is nothing to shard.
    """
    var fd: Int
    if address.kind == LISTEN_UNIX:
        # A path left behind by a process that died holds the address, and
        # bind fails with EADDRINUSE on a file nothing is listening to.
        if exists(address.path):
            _ = unlink(address.path)
        fd = listen_unix(address.path, BACKLOG)
    else:
        fd = listen_tcp_shared(address.host, address.port, BACKLOG, shared)

    try:
        set_nonblocking(fd)
    except e:
        _ = close(fd)
        raise e
    return fd


def bound_port(fd: Int) raises -> UInt16:
    """The port a TCP listener actually got. Binding to port 0 is the only way
    to be sure of a free one, and after that the kernel has to be asked."""
    return local_port(fd)
