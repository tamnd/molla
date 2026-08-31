"""The shutdown test, as a command, so it can be run a hundred times.

Issue #15 says a shutdown under load has to drain cleanly every time in a
hundred runs, and every time is a number a unit test cannot reach. So the whole
thing is one command that exits zero or one, and `scripts/drain-loop.sh` runs
it in a loop. A flake at one in fifty is invisible in CI and obvious here.

What it sets up is the case that is actually hard. Each client pipelines a
batch of requests and then reads almost none of the answers, so when the signal
arrives every connection has requests the server has read and not answered,
answers the server has written and the kernel has not delivered, and answers
still sitting in the connection's write ring with nowhere to go. A shutdown
that closes descriptors when it is asked to loses all three, and the client
sees a truncated response rather than an error, which is the failure worth
catching.

Making that load real rather than nominal took some measuring. The server asks
for a small send buffer on every accepted socket through the context, which
holds a connection to eight kilobytes in the kernel and leaves the rest of its
answers in the write ring, and each client asks for far more than that. With
the numbers below every connection is sitting on around thirty kilobytes of
queued response and a batch of unread requests behind it when the signal
lands, which is what the drain is supposed to be for. The client's receive
buffer is asked to be small as well. Linux honours that and macOS grows it
again over the life of the connection, so it helps rather than being the
mechanism. The first version of this leaned on the receive buffer alone, and it
reported a clean drain in zero milliseconds because there was nothing left to
drain.

The reader thread is the other half. Somebody has to keep taking bytes off
those sockets while the drain runs, because a drain that is flushing into a
buffer nobody empties is a drain that hits its deadline. It is a thread rather
than a loop on the main one because the main one is inside the signal wait and
then inside `Server.drain`, which is exactly the window the responses have to
move in.

The last piece is the handler that raises. One connection asks for `/boom`
before the load starts, has to get a 500, and the server has to still be
serving afterwards.
"""

from std.memory import stack_allocation

from molla.http.protocol import HttpProtocol
from molla.net.context import ServerContext
from molla.net.listener import ListenAddress
from molla.net.server import Server
from molla.net.supervisor import SignalWatcher, serve_until_signal
from molla.sys.clock import monotonic_ms
from molla.sys.fd import close
from molla.sys.mem import keep
from molla.sys.signal import SIGTERM, getpid, ignore_sigpipe, kill
from molla.sys.socket import (
    INADDR_LOOPBACK,
    SO_RCVBUF,
    connect,
    recv,
    send,
    set_buffer_size,
    socket_tcp,
)
from molla.sys.thread import Thread, set_thread_name, sleep_ms, spawn

comptime REQUESTS_PER_CONNECTION = 1024
"""Pipelined requests each connection has outstanding when the signal lands.

Around a hundred and sixty kilobytes of answers per connection, against a send
buffer of eight, so the great majority of it is still molla's problem when the
drain starts. The requests themselves come to under sixty kilobytes, which
keeps them below the header limit that would turn a half sent batch into a
431."""

comptime SERVER_SNDBUF = 8192
comptime CLIENT_RCVBUF = 4096
"""Small on purpose. See the module docstring: the point is that the kernel
cannot hold the whole batch, so the drain has to."""

comptime READ_MS = 5000
"""How long the reader waits for what the clients are owed before giving up.
Long, since the point of the exercise is that the answers do arrive."""

comptime SETUP_MS = 4000
comptime CHUNK = 4096


def _client(port: UInt16) raises -> Int:
    var fd = socket_tcp()
    # Before connect, because the window is negotiated during the handshake and
    # a receive buffer shrunk afterwards is a receive buffer the peer has
    # already been told it can fill.
    set_buffer_size(fd, SO_RCVBUF, CLIENT_RCVBUF)
    _ = connect(fd, INADDR_LOOPBACK, port)
    return fd


def _send_text(fd: Int, text: StringSpan) -> Int:
    var n = text.byte_length()
    var p = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(text.unsafe_ptr())
    )
    var sent = 0
    while sent < n:
        var wrote = send(fd, p.unsafe_offset(sent), n - sent)
        if wrote <= 0:
            _ = sleep_ms(1)
            continue
        sent += wrote
    return sent


def _read_more(fd: Int, mut into: List[UInt8], deadline: Int) -> Bool:
    """Read whatever is there, waiting until the deadline for the first byte.

    False means the peer closed, which after a drain is the normal ending and
    not a failure.
    """
    var buf = stack_allocation[CHUNK, UInt8]()
    while monotonic_ms() < deadline:
        var got = recv(fd, buf, CHUNK)
        if got > 0:
            for i in range(got):
                into.append(buf.unsafe_load(i))
            return True
        if got == 0:
            return False
        _ = sleep_ms(1)
    return True


struct ReadJob(Movable):
    """Every client socket, and everything read from it so far.

    Handed to the reader thread by address, since a thread entry point takes
    one integer. The main thread does not touch any of it between the spawn and
    the join, which is the whole of the sharing rule here.
    """

    var fds: List[Int]
    var received: List[List[UInt8]]
    var open: List[Bool]
    var closed: Int
    var deadline: Int

    def __init__(out self):
        self.fds = List[Int]()
        self.received = List[List[UInt8]]()
        self.open = List[Bool]()
        self.closed = 0
        self.deadline = 0

    def add(mut self, fd: Int):
        self.fds.append(fd)
        self.received.append(List[UInt8]())
        self.open.append(True)


def _reader(arg: Int) abi("C") -> Int:
    """Keep every client socket empty until the server closes them all.

    Ends on the last EOF, which is the server saying the drain finished, or on
    the deadline, which is the run failing and needing to say so rather than
    hang.
    """
    var job = Pointer[ReadJob, MutAnyOrigin](unsafe_from_address=arg)
    _ = set_thread_name("molla-reader")
    var buf = stack_allocation[CHUNK, UInt8]()
    while job[].closed < len(job[].fds) and monotonic_ms() < job[].deadline:
        var moved = False
        for i in range(len(job[].fds)):
            if not job[].open[i]:
                continue
            var got = recv(job[].fds[i], buf, CHUNK)
            if got > 0:
                for k in range(got):
                    job[].received[i].append(buf.unsafe_load(k))
                moved = True
            elif got == 0:
                job[].open[i] = False
                job[].closed += 1
        if not moved:
            _ = sleep_ms(1)
    return 0


def _count(data: List[UInt8], needle: StringSpan) -> Int:
    """How many times `needle` appears. Small strings, small buffers, and a
    plain scan is the right amount of machinery for both."""
    var n = needle.byte_length()
    if n == 0 or len(data) < n:
        return 0
    var p = needle.unsafe_ptr()
    var found = 0
    for start in range(len(data) - n + 1):
        var same = True
        for i in range(n):
            if data[start + i] != p.unsafe_load(i):
                same = False
                break
        if same:
            found += 1
    return found


def _request(target: StringSpan) -> String:
    return (
        String("GET ")
        + String(target)
        + " HTTP/1.1\r\nHost: molla\r\nConnection: keep-alive\r\n\r\n"
    )


def run_drain(connections: Int, deadline_ms: Int) raises -> Int:
    """Start a server, load it, signal it, and check nothing was cut off."""
    _ = ignore_sigpipe()

    print("drain", connections, "connections,", deadline_ms, "ms deadline")

    # Armed before the server starts, so the signal cannot arrive in the gap.
    var watcher = SignalWatcher()
    watcher.arm()

    var context = ServerContext(0, 60000, 0, deadline_ms, SERVER_SNDBUF)
    var server = Server[HttpProtocol](ListenAddress(UInt16(0)), context)
    for i in range(len(server.reactors)):
        server.reactors[i].proto.configure_fault(True)
    var port = server.port
    print("  workers       ", server.workers)
    print("  port          ", port)
    server.start()

    # A handler that raises, before anything else, so the rest of the run is
    # also evidence that it did not take the server with it.
    var boom = _client(port)
    _ = _send_text(boom, _request("/boom"))
    var boom_reply = List[UInt8]()
    var boom_deadline = monotonic_ms() + SETUP_MS
    while (
        _count(boom_reply, "\r\n\r\n") == 0
        and monotonic_ms() < boom_deadline
        and _read_more(boom, boom_reply, boom_deadline)
    ):
        pass
    var boom_ok = _count(boom_reply, "HTTP/1.1 500") == 1
    print("  handler raise ", "500" if boom_ok else "no 500")
    _ = close(boom)

    var after = _client(port)
    _ = _send_text(after, _request("/healthz"))
    var after_reply = List[UInt8]()
    var after_deadline = monotonic_ms() + SETUP_MS
    while (
        _count(after_reply, "\r\n\r\n") == 0
        and monotonic_ms() < after_deadline
        and _read_more(after, after_reply, after_deadline)
    ):
        pass
    var alive_ok = _count(after_reply, "HTTP/1.1 200") == 1
    print("  still serving ", "yes" if alive_ok else "no")
    _ = close(after)

    var batch = String("")
    for _ in range(REQUESTS_PER_CONNECTION):
        batch += _request("/")

    var job = ReadJob()
    for _ in range(connections):
        job.add(_client(port))
    for i in range(len(job.fds)):
        _ = _send_text(job.fds[i], batch)

    # One answer each, on this thread, before anything else happens. It is the
    # cheap way to know the server has really started on every connection, and
    # a run that signals a server which has not yet noticed the load is a run
    # that proves nothing.
    var first_ok = 0
    for i in range(len(job.fds)):
        var deadline = monotonic_ms() + SETUP_MS
        while monotonic_ms() < deadline:
            if _count(job.received[i], "HTTP/1.1 200") >= 1:
                first_ok += 1
                break
            if not _read_more(job.fds[i], job.received[i], deadline):
                break
    print("  in flight     ", first_ok, "of", connections, "connections")

    # What the clients have taken so far, against what they asked for. Printed
    # because it is the one number that says whether the run stressed the
    # shutdown or sailed past it, and checked below for the same reason: a
    # future change to the buffer sizes that lets the whole batch through
    # before the signal should fail here rather than pass quietly.
    var delivered = 0
    var wanted = connections * REQUESTS_PER_CONNECTION
    for i in range(len(job.fds)):
        delivered += _count(job.received[i], "HTTP/1.1 200 OK")
    print("  before signal ", delivered, "of", wanted, "answers")

    # The signal goes first and the reader starts after it, which is the wrong
    # way round until you try it the other way. A reader running before the
    # signal empties the sockets as fast as the workers fill them, the workers
    # finish the whole batch during the moment it takes the supervisor to wake
    # up, and the drain arrives to find every connection idle. Sending first
    # means the only thing that can move those answers is the drain itself.
    _ = kill(getpid(), SIGTERM)

    job.deadline = monotonic_ms() + deadline_ms + READ_MS
    var reader = Thread()
    var rc = spawn(_reader, Int(Pointer(to=job)), reader)
    if not rc.is_ok():
        raise Error(rc.describe("could not start the reader thread"))

    var outcome = serve_until_signal(server, watcher, deadline_ms)
    print("  shutdown      ", outcome.describe())

    _ = reader.join()

    var complete = 0
    var truncated = 0
    var still_open = 0
    for i in range(len(job.fds)):
        var answers = _count(job.received[i], "HTTP/1.1 200 OK")
        var bodies = _count(job.received[i], "Hello from molla.\n")
        if answers == REQUESTS_PER_CONNECTION and bodies == answers:
            complete += 1
        elif answers != bodies:
            truncated += 1
        if job.open[i]:
            still_open += 1
        _ = close(job.fds[i])

    print("  answered      ", complete, "of", connections, "connections")
    print("  truncated     ", truncated)

    var ok = (
        boom_ok
        and alive_ok
        and first_ok == connections
        and complete == connections
        and truncated == 0
        and still_open == 0
        and delivered < wanted
        and outcome.report.clean
    )
    print("  result        ", "pass" if ok else "fail")
    # The reader thread held this by address, and Mojo would otherwise be free
    # to drop it at its last mention above.
    keep(job)
    return 0 if ok else 1
