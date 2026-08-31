"""What a server is told at startup, in one value that gets passed down.

Mojo 1.0 has no globals, which is usually an obstacle and here is the feature.
Every setting a server has arrives in this struct, the struct is made by
whoever starts the server, and a test that wants two servers with different
settings in one process makes two of them. There is no configuration a test has
to remember to put back, because there is nowhere for it to live.

Deliberately only what the server itself decides. The protocol's limits belong
to the protocol, and logging and metrics are #16. What is here is the shape of
the thread pool, how long a connection may idle, where allocations are counted,
and how long a shutdown may take.

Copyable, because every field is a number and passing a copy down to a reactor
is exactly what should happen. Nothing here is mutable state shared between
threads. The state that is shared lives in the reactor's atomic control block,
where it is one line per flag and obvious that it is shared.
"""

from molla.net.reactor import DEFAULT_IDLE_MS

comptime DEFAULT_DRAIN_MS = 10000
"""How long a graceful shutdown waits for connections to finish.

Ten seconds is what a deploy can spare. A request that has not finished in that
long is either a stream somebody is still watching or a client that has stopped
reading, and neither is worth holding the new process for."""


struct ServerContext(Copyable, ImplicitlyCopyable, Movable):
    """The settings one server runs with."""

    var workers: Int
    """I/O threads. Zero means one per core, within the cap."""

    var idle_timeout_ms: Int
    var counter: Int
    """Address of the allocation counter, or zero for not counting."""

    var drain_deadline_ms: Int
    """How long `Server.drain` waits before it starts cutting connections."""

    var send_buffer_bytes: Int
    """Kernel send buffer for accepted sockets, or zero to leave it alone.

    Zero is the right answer in production, where Linux sizes this per
    connection and does it better than a number written down once. The
    shutdown test sets it small on purpose, because a response that the kernel
    has already taken is not a response the drain has to flush."""

    def __init__(
        out self,
        workers: Int = 0,
        idle_timeout_ms: Int = DEFAULT_IDLE_MS,
        counter: Int = 0,
        drain_deadline_ms: Int = DEFAULT_DRAIN_MS,
        send_buffer_bytes: Int = 0,
    ):
        self.workers = workers
        self.idle_timeout_ms = idle_timeout_ms
        self.counter = counter
        self.drain_deadline_ms = drain_deadline_ms
        self.send_buffer_bytes = send_buffer_bytes
