"""A TCP echo server on the raw syscall layer.

This exists to answer one question from M0: can a non blocking, single threaded,
edge triggered socket server be written in Mojo 1.0 at all, given there is no
async, no `std.net`, and no threading module. Everything here is throwaway as a
feature and load bearing as a shape. The HTTP server in the next milestone is
this loop with a parser in the middle, so the parts worth getting right are the
parts that are hard to retrofit: partial writes, edge triggered draining, and
knowing exactly when a descriptor may be closed.

What it deliberately does not do: no threads, no timeouts, no limit on how long
a connection may idle, no TLS. Those belong to the real server and each one is
its own issue.

Three things here are the actual findings, and they are the reason the code is
not shorter.

Edge triggered means read until EAGAIN. If a connection is drained with one
`recv` and 3 KB arrived, the remaining bytes sit there and no further event is
ever delivered for them, because readiness already fired. That is a hang, not a
slowdown, and it does not show up under light load.

A short write is normal, not an error. `send` on a non blocking socket returns
what fits in the kernel buffer. Retrying immediately in a loop burns a core
while the peer is slow. The remainder gets buffered on the connection and write
interest is registered instead, so the loop goes back to waiting.

A descriptor cannot be closed while there is still output pending, which means
end of stream is not the same as end of connection. A peer that sends a request
and shuts down its write side still wants the answer.
"""

from std.memory import stack_allocation

from molla.sys.errno import EAGAIN, ECONNRESET, EINTR, errno_name, get_errno
from molla.sys.fd import close, set_nonblocking
from molla.sys.poll import Poller, Ready
from molla.sys.socket import (
    INADDR_LOOPBACK,
    accept,
    listen_tcp,
    local_port,
    recv,
    send,
)

comptime READ_CHUNK = 4096
"""How much we try to read per syscall. Small enough that one slow connection
cannot hold a big buffer, large enough that a normal request is one call."""

comptime MAX_EVENTS = 64
comptime BACKLOG = 128


struct Connection(Movable):
    """One accepted socket, plus whatever we owe it.

    `pending` is output that `send` would not take. It is drained from
    `pending_at` rather than by removing from the front, because shifting a list
    on every partial write is quadratic on exactly the connections that are
    already struggling.
    """

    var fd: Int
    var pending: List[UInt8]
    var pending_at: Int
    var peer_done: Bool
    """The peer sent end of stream. We still owe it whatever is in `pending`."""

    var closed: Bool

    def __init__(out self, fd: Int):
        self.fd = fd
        self.pending = List[UInt8]()
        self.pending_at = 0
        self.peer_done = False
        self.closed = False

    def pending_bytes(self) -> Int:
        return len(self.pending) - self.pending_at

    def wants_write(self) -> Bool:
        return self.pending_bytes() > 0

    def is_finished(self) -> Bool:
        """Everything owed has been written and the peer is done talking."""
        return self.peer_done and not self.wants_write()


struct EchoServer(Movable):
    """The event loop. One listener, one poller, one connection table."""

    var listener: Int
    var poller: Poller
    var conns: List[Connection]
    var write_interest: List[Bool]
    """Whether write interest is currently registered for `conns[i]`. Tracked so
    the loop only calls the poller when the answer changes, since a register
    call per pass is a syscall per pass for nothing."""

    var accepted: Int
    var bytes_echoed: Int

    def __init__(out self, address: UInt32, port: UInt16) raises:
        self.listener = listen_tcp(address, port, BACKLOG)
        self.poller = Poller(MAX_EVENTS)
        self.conns = List[Connection]()
        self.write_interest = List[Bool]()
        self.accepted = 0
        self.bytes_echoed = 0
        try:
            self.poller.add_read(self.listener)
        except e:
            _ = close(self.listener)
            raise e

    def port(self) raises -> UInt16:
        """The port actually bound. Interesting when port 0 was requested."""
        return local_port(self.listener)

    def _index_of(self, fd: Int) -> Int:
        for i in range(len(self.conns)):
            if self.conns[i].fd == fd and not self.conns[i].closed:
                return i
        return -1

    def _accept_all(mut self) raises:
        """Drain the accept backlog.

        Edge triggered again: one readiness notification can cover several
        pending connections, so accepting once per event would leave the rest
        stuck until the next unrelated client showed up.
        """
        while True:
            var fd = accept(self.listener)
            if fd < 0:
                var code = get_errno()
                if code == EAGAIN or code == EINTR:
                    return
                raise Error("accept failed: " + errno_name(code))
            # macOS does not inherit O_NONBLOCK onto an accepted socket and
            # there is no accept4 to ask for it, so set it here on both.
            set_nonblocking(fd)
            self.poller.add_read(fd)
            self.conns.append(Connection(fd))
            self.write_interest.append(False)
            self.accepted += 1

    def _flush(mut self, index: Int) raises:
        """Write as much of the pending buffer as the kernel will take."""
        while self.conns[index].pending_bytes() > 0:
            var start = self.conns[index].pending_at
            var remaining = self.conns[index].pending_bytes()
            var wrote = send(
                self.conns[index].fd,
                self.conns[index].pending.unsafe_ptr().unsafe_offset(start),
                remaining,
            )
            if wrote < 0:
                var code = get_errno()
                if code == EAGAIN:
                    return
                if code == EINTR:
                    continue
                # The peer is gone. Dropping the output is the only option and
                # it is not an error worth failing the server over.
                self.conns[index].peer_done = True
                self.conns[index].pending.clear()
                self.conns[index].pending_at = 0
                return
            self.conns[index].pending_at += wrote
            self.bytes_echoed += wrote

        self.conns[index].pending.clear()
        self.conns[index].pending_at = 0

    def _drain(mut self, index: Int) raises:
        """Read until EAGAIN, queueing everything read for writing back."""
        var buf = stack_allocation[READ_CHUNK, UInt8]()
        while True:
            var got = recv(self.conns[index].fd, buf, READ_CHUNK)
            if got == 0:
                self.conns[index].peer_done = True
                return
            if got < 0:
                var code = get_errno()
                if code == EAGAIN:
                    return
                if code == EINTR:
                    continue
                if code == ECONNRESET:
                    self.conns[index].peer_done = True
                    self.conns[index].pending.clear()
                    self.conns[index].pending_at = 0
                    return
                raise Error("recv failed: " + errno_name(code))
            for i in range(got):
                self.conns[index].pending.append(buf.unsafe_load(i))

    def _sync_write_interest(mut self, index: Int) raises:
        var wanted = self.conns[index].wants_write()
        if wanted == self.write_interest[index]:
            return
        self.poller.set_write_interest(self.conns[index].fd, wanted)
        self.write_interest[index] = wanted

    def _close(mut self, index: Int):
        if self.conns[index].closed:
            return
        self.poller.remove(self.conns[index].fd)
        _ = close(self.conns[index].fd)
        self.conns[index].closed = True

    def _reap(mut self):
        """Drop closed connections from the table.

        Done in one pass after the events are handled rather than during, so no
        index can shift under a loop that is still using it. Backwards, so that
        removing one entry does not move the entries not yet looked at.
        """
        var i = len(self.conns) - 1
        while i >= 0:
            if self.conns[i].closed:
                _ = self.conns.pop(i)
                _ = self.write_interest.pop(i)
            i -= 1

    def poll_once(mut self, timeout_ms: Int) raises -> Int:
        """Wait once and handle whatever came back. Returns the event count.

        Split out from `serve` so tests can step the loop by hand and drive a
        client on the same thread. There is no second thread to put a client on.
        """
        var count = self.poller.wait(timeout_ms)

        for i in range(count):
            var ready = self.poller.event(i)
            if ready.fd == self.listener:
                self._accept_all()
                continue

            var index = self._index_of(ready.fd)
            if index < 0:
                continue

            if ready.readable:
                self._drain(index)
            if not self.conns[index].closed:
                self._flush(index)

        # Interest and lifetime are settled after the whole batch, because one
        # descriptor can appear twice in it, once readable and once writable.
        for i in range(len(self.conns)):
            if self.conns[i].closed:
                continue
            if self.conns[i].is_finished():
                self._close(i)
            else:
                self._sync_write_interest(i)

        self._reap()
        return count

    def serve(mut self, timeout_ms: Int) raises:
        """Run forever. Only `molla echo` calls this."""
        while True:
            _ = self.poll_once(timeout_ms)

    def shutdown(mut self):
        for i in range(len(self.conns)):
            self._close(i)
        self.conns.clear()
        self.write_interest.clear()
        self.poller.shutdown()
        if self.listener >= 0:
            _ = close(self.listener)
            self.listener = -1


def run_echo(port: UInt16) raises:
    """Entry point for `molla echo`. Binds loopback only, per D9."""
    var server = EchoServer(INADDR_LOOPBACK, port)
    print("echo server listening on 127.0.0.1:" + String(server.port()))
    print("this is an M0 spike, not a molla feature. ctrl-c to stop.")
    server.serve(1000)
