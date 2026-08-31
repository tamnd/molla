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

Shutdown is cooperative. `stop` sets a flag on each reactor and writes one byte
to its wakeup socket, so a reactor sitting in the poller returns immediately
rather than after its timeout. Each reactor then finishes the pass it is in,
closes its connections through the protocol, and returns from `run`.
"""

from molla.net.listener import (
    LISTEN_TCP,
    SHARDED_ACCEPT,
    ListenAddress,
    bound_port,
    open_listener,
)
from molla.net.reactor import DEFAULT_IDLE_MS, Protocol, Reactor
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

    def __init__(
        out self,
        var address: ListenAddress,
        workers: Int,
        idle_timeout_ms: Int,
        counter: Int,
    ) raises:
        var count = workers if workers > 0 else default_workers()
        if count > MAX_WORKERS:
            count = MAX_WORKERS
        self.workers = count
        self.address = address^
        self.port = 0
        self.running = False
        self.threads = List[Thread]()

        # Full capacity up front. A reallocation after the first thread starts
        # would move a reactor a worker is in the middle of running.
        self.reactors = List[Reactor[Self.P]](capacity=count)
        for _ in range(count):
            self.reactors.append(
                Reactor[Self.P](Self.P(), idle_timeout_ms, counter)
            )

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
