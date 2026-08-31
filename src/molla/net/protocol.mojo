"""The reference protocol, which is echo.

Two reasons this exists rather than being test scaffolding. It is the smallest
complete implementation of the `Protocol` trait, so it is what to read when
writing the HTTP one in #11, and it is what the reactor tests drive, so the
loop is exercised by the same code shape a real protocol will use rather than
by a special path only tests take.

Echo is also the honest worst case for the write path. It queues exactly as
much as it reads, which means a client that writes faster than it reads fills
the ring and produces the backpressure case on purpose, and that is the case
that is hard to get right.

`molla.net.echo` is the M0 spike and is a different thing: a whole server in
one file, kept because it is the evidence behind D1. This is a protocol on the
real reactor.
"""

from molla.net.conn import Connection
from molla.net.reactor import Protocol
from molla.sys.mem import as_ptr


struct EchoProtocol(Movable, Protocol):
    """Write back what arrives, and count what happened."""

    var opened: Int
    var closed: Int
    var echoed: Int
    var stalled: Int
    """How many times the ring was too full to take everything read. Not an
    error, and the number tests assert backpressure with."""

    def __init__(out self):
        self.opened = 0
        self.closed = 0
        self.echoed = 0
        self.stalled = 0

    def on_open(mut self, mut conn: Connection):
        self.opened += 1

    def on_readable(mut self, mut conn: Connection) -> Bool:
        """Queue whatever is in the read buffer and consume what fit.

        The part worth copying into a real protocol is the consume. What the
        ring would not take stays in the input buffer, so the bytes are not
        lost and not duplicated, and the next pass tries again once the socket
        has drained. A protocol that queued blindly and cleared the buffer
        would silently drop a response under exactly the load that makes
        dropping one worst.
        """
        var available = conn.input.length
        if available == 0:
            return True
        # Through the address rather than through `conn.input.bytes()`, because
        # queueing borrows the whole connection mutably and the span would be a
        # second borrow of a field inside it. The ring and the buffer are
        # separate blocks, so there is no aliasing here to be careful about.
        var data = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(conn.input.base()), length=available
        )
        var took = conn.queue(data)
        if took < available:
            self.stalled += 1
        if took > 0:
            conn.input.consume(took)
            self.echoed += took
        return True

    def on_writable(mut self, mut conn: Connection) -> Bool:
        """The ring drained, so whatever did not fit before may fit now."""
        return self.on_readable(conn)

    def on_close(mut self, mut conn: Connection):
        self.closed += 1
