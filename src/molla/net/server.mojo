"""N reactors, N threads, one address.

This is the piece that turns an event loop into a server. Everything hard about
it is either the platform difference in how connections get spread across
workers, which lives in `listener.mojo`, or the fact that Mojo 1.0 will happily
free something a running thread is still using, which lives here.

The lifetime rule is the whole of the danger. A worker thread is given the
address of its reactor, and an address is not a use the compiler can see, so
the reactors have to live somewhere that outlives every thread by construction
rather than by luck. Two things make that true here. The reactor list is
created with its full capacity up front and never appended to after a thread
has started, because appending can move every reactor to a new block while a
worker is mid pass on the old one. And the server joins its threads in
`stop`, which the destructor also calls, so a server that goes out of scope
early stops its workers rather than leaving them running against memory that is
about to be reused.

Shutdown is cooperative and comes in two strengths. `stop` sets a flag on each
reactor and writes one byte to its wakeup socket, so a reactor sitting in the
poller returns immediately rather than after its timeout, and everything still
queued for a client is lost. `drain` asks the same reactors to stop accepting
and then close each connection at the first moment it owes that client nothing,
with a deadline after which what is left is cut. `drain` is what a signal
turns into and what a deploy should use. `stop` is what a test uses when it has
already checked what it came to check.

The report `drain` returns is the whole point of doing it this way. A shutdown
that says it dropped four connections after nine seconds is a shutdown you can
do something about, and a process that just exits is not.
"""

from molla.net.context import ServerContext
from molla.net.listener import (
    LISTEN_TCP,
    SHARDED_ACCEPT,
    ListenAddress,
    bound_port,
    open_listener,
)
from molla.net.reactor import Protocol, Reactor, state_name
from molla.sys.clock import monotonic_ms
from molla.sys.thread import Thread, ThreadFunc, cpu_count, spawn

comptime MAX_WORKERS = 64
"""A ceiling on I/O threads. More reactors than cores is worse than fewer,
because each one costs a poller and a wakeup socket and none of them are
waiting on anything a context switch would help with."""


def default_workers() -> Int:
    """One reactor per core, within reason.

    Not one per hyperthread pair and not one per NUMA node, because the request
    path is not the thing that will saturate this machine. The engine is, and
    it gets the cores that matter. Two here is the floor, since one reactor
    cannot overlap a slow client with a fast one across cores at all.
    """
    var cores = cpu_count()
    if cores < 2:
        return 2
    if cores > MAX_WORKERS:
        return MAX_WORKERS
    return cores


struct DrainReport(Copyable, ImplicitlyCopyable, Movable):
    """What a graceful shutdown actually did."""

    var clean: Bool
    """True when every connection finished what it owed before the deadline."""

    var dropped: Int
    """Connections cut because the deadline passed."""

    var open_at_end: Int
    var elapsed_ms: Int
    var accepted: Int

    def __init__(out self):
        self.clean = True
        self.dropped = 0
        self.open_at_end = 0
        self.elapsed_ms = 0
        self.accepted = 0

    def describe(self) -> String:
        """One line for a log or for the end of a test run."""
        return (
            "drained "
            + ("cleanly" if self.clean else "with cuts")
            + " in "
            + String(self.elapsed_ms)
            + "ms, "
            + String(self.accepted)
            + " connections served, "
            + String(self.dropped)
            + " dropped"
        )


def _worker[P: Protocol](arg: Int) abi("C") -> Int:
    """A thread's whole life. Runs one reactor until it is told to stop.

    The argument is the reactor's address, because a thread entry point takes
    one integer and there is nowhere else to put anything. Errors are swallowed
    into a return code: a reactor that raises has already lost its connections,
    and taking the process down with it would lose everybody else's.
    """
    var reactor = Pointer[Reactor[P], MutAnyOrigin](unsafe_from_address=arg)
    try:
        reactor[].run()
    except:
        return 1
    return 0


struct Server[P: Protocol](Movable):
    """One listening address served by several reactors."""

    var reactors: List[Reactor[Self.P]]
    var threads: List[Thread]
    var address: ListenAddress
    var port: UInt16
    """The port actually bound, which is the interesting one when the address
    asked for port 0."""

    var workers: Int
    var running: Bool
    var context: ServerContext
    """The settings this server was made with, kept so `drain` knows its own
    deadline and a caller does not have to remember what it asked for."""

    def __init__(
        out self, var address: ListenAddress, context: ServerContext
    ) raises:
        var count = (
            context.workers if context.workers > 0 else default_workers()
        )
        if count > MAX_WORKERS:
            count = MAX_WORKERS
        self.workers = count
        self.context = context
        self.address = address^
        self.port = 0
        self.running = False
        self.threads = List[Thread]()

        # Full capacity up front. A reallocation after the first thread starts
        # would move a reactor a worker is in the middle of running.
        self.reactors = List[Reactor[Self.P]](capacity=count)
        for i in range(count):
            self.reactors.append(
                Reactor[Self.P](
                    Self.P(), context.idle_timeout_ms, context.counter
                )
            )
            self.reactors[i].index = i
            self.reactors[i].send_buffer_bytes = context.send_buffer_bytes

        # Unix sockets are never sharded, whatever the platform does with TCP.
        # There is one path in the filesystem, binding it twice would mean the
        # second listener unlinking the first one's address, and a handoff
        # costs almost nothing next to the kind of traffic that arrives on a
        # unix socket.
        var sharded = SHARDED_ACCEPT and self.address.kind == LISTEN_TCP

        if sharded:
            # Linux. Every reactor gets its own listening socket on the same
            # port and the kernel spreads connections across them.
            var first = open_listener(self.address, True)
            self.port = bound_port(first)
            self.address.port = self.port
            self.reactors[0].add_listener(first)
            for i in range(1, count):
                self.reactors[i].add_listener(open_listener(self.address, True))
        else:
            # One listener, and the reactor that owns it hands accepted
            # connections round the others including itself.
            var only = open_listener(self.address, False)
            if self.address.kind == LISTEN_TCP:
                self.port = bound_port(only)
                self.address.port = self.port
            self.reactors[0].add_listener(only)
            for i in range(count):
                var peer = Int(Pointer(to=self.reactors[i]))
                self.reactors[0].add_peer(peer)

    def __deinit__(deinit self):
        """Stop the workers before anything they use goes away.

        A safety net rather than the normal path. Mojo destroys a value at its
        last use, so a server whose last mention is halfway down a function
        would otherwise have its reactors freed while its threads are still in
        them."""
        for i in range(len(self.reactors)):
            self.reactors[i].stop()
        for i in range(len(self.threads)):
            _ = self.threads[i].join()
        for i in range(len(self.reactors)):
            self.reactors[i].shutdown()

    def start(mut self) raises:
        """Put every reactor on its own thread and return.

        Returns as soon as the threads exist, not once they are in the poller,
        so a client connecting immediately after this may land in the backlog
        for a moment. That is what the backlog is for.
        """
        if self.running:
            return
        var entry: ThreadFunc = _worker[Self.P]
        for i in range(self.workers):
            var address = Int(Pointer(to=self.reactors[i]))
            var thread = Thread()
            var rc = spawn(entry, address, thread)
            if not rc.is_ok():
                raise Error(rc.describe("could not start an I/O worker"))
            self.threads.append(thread^)
        self.running = True

    def stop(mut self):
        """Tell every reactor to finish and wait for the threads to end."""
        if not self.running:
            return
        for i in range(len(self.reactors)):
            self.reactors[i].stop()
        for i in range(len(self.threads)):
            _ = self.threads[i].join()
        self.threads.clear()
        self.running = False
        for i in range(len(self.reactors)):
            self.reactors[i].shutdown()

    def drain(mut self, deadline_ms: Int = 0) raises -> DrainReport:
        """Stop accepting, let what is in flight finish, then stop.

        One deadline for the whole server rather than one per reactor. Each
        reactor is given the same monotonic timestamp, so a worker whose thread
        is slow to be scheduled does not get a fresh ten seconds of its own and
        the shutdown takes as long as it says it will.

        Returns once every worker thread has ended, which is what makes it safe
        for the caller to free anything the workers were using.
        """
        var started = monotonic_ms()
        var report = DrainReport()
        var budget = (
            deadline_ms if deadline_ms > 0 else self.context.drain_deadline_ms
        )

        if not self.running:
            # Nothing to wait for, but the listeners and the poller still have
            # to go back.
            for i in range(len(self.reactors)):
                self.reactors[i].shutdown()
            report.elapsed_ms = monotonic_ms() - started
            report.accepted = self.accepted()
            return report^

        var deadline = started + budget
        for i in range(len(self.reactors)):
            self.reactors[i].begin_drain(deadline)
        for i in range(len(self.threads)):
            _ = self.threads[i].join()
        self.threads.clear()
        self.running = False

        # Read after the joins, so these are final rather than a snapshot of a
        # reactor that is still working.
        for i in range(len(self.reactors)):
            report.dropped += self.reactors[i].dropped()
            report.open_at_end += self.reactors[i].connection_count()
            report.accepted += self.reactors[i].accepted
        report.clean = report.dropped == 0

        for i in range(len(self.reactors)):
            self.reactors[i].shutdown()
        report.elapsed_ms = monotonic_ms() - started
        return report^

    def dump(self) -> String:
        """Every worker's state, thread, connection count and queue depth.

        What SIGQUIT prints. Read from another thread while the workers are
        running, so everything in it comes from the atomic control block rather
        than from walking a table a worker is in the middle of changing. The
        numbers are a reading taken at slightly different moments, which is the
        honest thing for a diagnostic to be and worth knowing before anybody
        adds them up.
        """
        var out = String("molla: ") + String(self.workers) + " workers\n"
        for i in range(len(self.reactors)):
            out += (
                "  worker "
                + String(i)
                + " "
                + state_name(self.reactors[i].state())
                + " thread "
                + String(self.reactors[i].thread_id())
                + " open "
                + String(self.reactors[i].open_published())
                + " accepted "
                + String(self.reactors[i].accepted)
                + " handoff "
                + String(self.reactors[i].handoff_depth())
                + "\n"
            )
        return out^

    def accepted(self) -> Int:
        """Connections taken across every reactor."""
        var total = 0
        for i in range(len(self.reactors)):
            total += self.reactors[i].accepted
        return total

    def open_connections(self) -> Int:
        var total = 0
        for i in range(len(self.reactors)):
            total += self.reactors[i].connection_count()
        return total

    def spread(self) -> List[Int]:
        """How many connections each reactor took. What a test asserts on to
        show the work is going more than one place."""
        var counts = List[Int]()
        for i in range(len(self.reactors)):
            counts.append(self.reactors[i].accepted)
        return counts^
