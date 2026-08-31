"""Tests for atomics, the queues, thread naming, and graceful shutdown.

Real threads, because none of this is worth testing without them. An atomic
counter that only one thread ever touches is a counter, a queue that is filled
and drained by the same thread is a list, and a shutdown that has nothing in
flight is a close. So every test here that can race does race, and the ones
that check a number check an exact number rather than a plausible one: four
threads adding twenty thousand each is eighty thousand or it is a bug.

The lifetime rule shows up in every one of them. A thread is given an address,
Mojo destroys a local at its last use rather than at the end of the scope, and
handing out an address is not a use it can see. Everything shared here is kept
alive across the joins with `keep`, and a test that forgets it does not fail
cleanly, it increments memory that belongs to somebody else.

The shutdown tests are the small version of what `molla drain` does end to end.
The command is the real check, since the acceptance criterion is a hundred runs
and a test suite cannot spend that. What is here is the part that fits: an
idle pool closes at once, a connection mid request is served to the end, and a
handler that raises turns into a 500 rather than a dead server.
"""

from std.memory import stack_allocation

from harness import Suite

from molla.http.protocol import HttpProtocol
from molla.net.context import ServerContext
from molla.net.listener import ListenAddress
from molla.net.protocol import EchoProtocol
from molla.net.server import Server
from molla.sys.atomic import CACHE_LINE, AtomicBlock, AtomicRef
from molla.sys.clock import monotonic_ms
from molla.sys.fd import close
from molla.sys.mem import keep
from molla.sys.queue import MpscQueue, SpscRing, round_up_pow2
from molla.sys.socket import INADDR_LOOPBACK, connect, recv, send, socket_tcp
from molla.sys.thread import (
    Once,
    Thread,
    self_id,
    set_thread_name,
    sleep_ms,
    spawn,
    thread_name,
)

comptime BUMP_THREADS = 4
comptime BUMP_ROUNDS = 20000
"""Eighty thousand increments across four threads. Large enough that a lost
update is almost certain if the add is not atomic, small enough that the test
costs milliseconds."""

comptime ONCE_RACERS = 8
comptime PRODUCERS = 3
comptime PER_PRODUCER = 500
comptime QUEUE_CAPACITY = 64
"""Deliberately smaller than one producer's share, so the queue fills, push
returns false, and the retry path is what is being tested."""

comptime WAIT_MS = 4000
"""How long a test waits for something running on another thread. Generous,
because a shared CI runner can leave a thread unscheduled for a while and that
is not a bug."""


def _bump(arg: Int) abi("C") -> Int:
    """Add one, twenty thousand times, to the counter at this address."""
    var counter = AtomicRef(arg)
    for _ in range(BUMP_ROUNDS):
        _ = counter.add(1)
    return 0


def _once_body(arg: Int) abi("C") -> Int:
    """The initialiser. Runs exactly once however many racers call it."""
    var counter = AtomicRef(arg)
    _ = counter.add(1)
    return 0


struct OnceJob(Movable):
    """What the racing threads share: the `Once`, the thing it initialises, and
    a count of how many of them believed they were the one that ran it."""

    var gate: Once
    var body_runs: AtomicRef
    var winners: AtomicRef

    def __init__(out self, body_runs: AtomicRef, winners: AtomicRef):
        self.gate = Once()
        self.body_runs = body_runs
        self.winners = winners


def _once_racer(arg: Int) abi("C") -> Int:
    var job = Pointer[OnceJob, MutAnyOrigin](unsafe_from_address=arg)
    var rc = job[].gate.call(_once_body, job[].body_runs.address)
    if rc.is_ok() and rc.value == 1:
        _ = job[].winners.add(1)
    return 0


struct PushJob(Movable):
    """One producer's share of the values, and what it managed to push."""

    var queue: Int
    """Address of the queue. Shared with every other producer, which is the
    point of an MPSC queue."""

    var first: Int
    var count: Int
    var pushed: Int
    var retries: Int

    def __init__(out self, queue: Int, first: Int, count: Int):
        self.queue = queue
        self.first = first
        self.count = count
        self.pushed = 0
        self.retries = 0


def _producer(arg: Int) abi("C") -> Int:
    """Push a block of distinct values, retrying while the queue is full."""
    var job = Pointer[PushJob, MutAnyOrigin](unsafe_from_address=arg)
    var queue = Pointer[MpscQueue, MutAnyOrigin](
        unsafe_from_address=job[].queue
    )
    for i in range(job[].count):
        while not queue[].push(job[].first + i):
            job[].retries += 1
            _ = sleep_ms(1)
        job[].pushed += 1
    return 0


def _ring_producer(arg: Int) abi("C") -> Int:
    """The single writer of an SPSC ring, sending a run of values in order."""
    var ring = Pointer[SpscRing, MutAnyOrigin](unsafe_from_address=arg)
    for i in range(PER_PRODUCER):
        while not ring[].push(i):
            _ = sleep_ms(1)
    return 0


def _named(arg: Int) abi("C") -> Int:
    """Name this thread and report whether the name reads back."""
    var result = AtomicRef(arg)
    _ = set_thread_name("molla-test")
    if thread_name() == "molla-test":
        result.store(1)
    else:
        result.store(2)
    return 0


def _client(port: UInt16) raises -> Int:
    var fd = socket_tcp()
    _ = connect(fd, INADDR_LOOPBACK, port)
    return fd


def _send_text(fd: Int, text: StringSpan) -> Int:
    var n = text.byte_length()
    var p = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(text.unsafe_ptr())
    )
    var sent = 0
    var deadline = monotonic_ms() + WAIT_MS
    while sent < n and monotonic_ms() < deadline:
        var wrote = send(fd, p.unsafe_offset(sent), n - sent)
        if wrote <= 0:
            _ = sleep_ms(1)
            continue
        sent += wrote
    return sent


def _read_until(fd: Int, needle: StringSpan, mut into: List[UInt8]) -> Bool:
    """Read until the answer contains `needle` or the deadline passes."""
    var buf = stack_allocation[4096, UInt8]()
    var deadline = monotonic_ms() + WAIT_MS
    while monotonic_ms() < deadline:
        if _contains(into, needle):
            return True
        var got = recv(fd, buf, 4096)
        if got > 0:
            for i in range(got):
                into.append(buf.unsafe_load(i))
            continue
        if got == 0:
            return _contains(into, needle)
        _ = sleep_ms(1)
    return _contains(into, needle)


def _contains(data: List[UInt8], needle: StringSpan) -> Bool:
    var n = needle.byte_length()
    if n == 0 or len(data) < n:
        return False
    var p = needle.unsafe_ptr()
    for start in range(len(data) - n + 1):
        var same = True
        for i in range(n):
            if data[start + i] != p.unsafe_load(i):
                same = False
                break
        if same:
            return True
    return False


def _check_atomics(mut suite: Suite) raises:
    suite.group("sys.atomic")

    var block = AtomicBlock(4)
    suite.check(block.is_valid(), "a block of four counters allocates")
    suite.check(
        block.address_of(0) % CACHE_LINE == 0,
        "and the first counter is on a cache line boundary",
    )
    suite.check(
        block.address_of(1) - block.address_of(0) == CACHE_LINE,
        "with one whole line between neighbours",
    )
    suite.check(
        block.address_of(4) == 0 and block.address_of(-1) == 0,
        "an index outside the block is an address of zero",
    )

    var counter = block.slot(0)
    suite.check(counter.load() == 0, "a fresh counter reads zero")
    counter.store(7)
    suite.check(counter.load() == 7, "a store is visible to a load")
    suite.check(counter.add(3) == 7, "add returns the value from before it")
    suite.check(counter.load() == 10, "and leaves the sum behind")
    suite.check(counter.sub(4) == 10, "sub also returns the old value")
    suite.check(counter.load() == 6, "and subtracts")

    suite.check(
        counter.compare_exchange(6, 11), "compare and exchange takes the value"
    )
    suite.check(counter.load() == 11, "and stores the new one")
    suite.check(
        not counter.compare_exchange(6, 99),
        "and refuses when the value moved on",
    )
    suite.check(counter.load() == 11, "leaving it alone when it refuses")

    var seen = 6
    suite.check(
        not counter.compare_exchange_update(seen, 12),
        "the updating form also refuses",
    )
    suite.check(seen == 11, "and reports what was really there")
    suite.check(
        counter.compare_exchange_update(seen, 12),
        "so the retry with that value wins",
    )
    suite.check(counter.swap(20) == 12, "swap returns the old value")
    suite.check(counter.load() == 20, "and leaves the new one")

    var nowhere = AtomicRef(0)
    suite.check(nowhere.load() == 0, "a null reference loads zero")
    suite.check(nowhere.add(5) == 0, "adds nothing")
    nowhere.store(9)
    suite.check(
        nowhere.load() == 0 and not nowhere.compare_exchange(0, 1),
        "and never writes anywhere",
    )

    # Four threads on one counter. The number is exact or the add is not
    # atomic, and there is no third possibility.
    var shared = block.slot(2)
    var threads = List[Thread](capacity=BUMP_THREADS)
    var started = 0
    for _ in range(BUMP_THREADS):
        var thread = Thread()
        if spawn(_bump, shared.address, thread).is_ok():
            started += 1
        threads.append(thread^)
    for i in range(len(threads)):
        _ = threads[i].join()
    suite.check(started == BUMP_THREADS, "four threads start")
    suite.check(
        shared.load() == BUMP_THREADS * BUMP_ROUNDS,
        "and eighty thousand increments all land",
    )
    suite.check(
        block.slot(1).load() == 0 and block.slot(3).load() == 0,
        "with the neighbouring counters untouched",
    )

    # The block outlives the threads because of this, and only because of it.
    keep(block)


def _check_once(mut suite: Suite) raises:
    suite.group("sys.thread once")

    var block = AtomicBlock(2)
    var job = OnceJob(block.slot(0), block.slot(1))
    suite.check(not job.gate.is_done(), "a fresh gate has not run")

    var address = Int(Pointer(to=job))
    var threads = List[Thread](capacity=ONCE_RACERS)
    for _ in range(ONCE_RACERS):
        var thread = Thread()
        _ = spawn(_once_racer, address, thread)
        threads.append(thread^)
    for i in range(len(threads)):
        _ = threads[i].join()

    suite.check(job.gate.is_done(), "the gate is done after the race")
    suite.check(block.slot(0).load() == 1, "the body ran exactly once")
    suite.check(block.slot(1).load() == 1, "and exactly one racer ran it")

    var again = job.gate.call(_once_body, block.slot(0).address)
    suite.check(
        again.is_ok() and again.value == 0, "a later call does not run it"
    )
    suite.check(block.slot(0).load() == 1, "and the body still ran once")

    keep(job)
    keep(block)


def _check_mpsc(mut suite: Suite) raises:
    suite.group("sys.queue mpsc")

    suite.check(round_up_pow2(1) == 2, "a capacity of one rounds up to two")
    suite.check(round_up_pow2(64) == 64, "a power of two is left alone")
    suite.check(round_up_pow2(65) == 128, "and anything else rounds up")

    var small = MpscQueue(4)
    suite.check(small.is_valid(), "a queue allocates")
    suite.check(small.is_empty(), "and starts empty")
    var out = 0
    suite.check(not small.pop(out), "an empty queue pops nothing")
    for i in range(4):
        suite.check(small.push(i), "the queue takes four items")
    suite.check(not small.push(99), "and refuses the fifth")
    suite.check(small.depth() == 4, "the depth is what was pushed")
    for i in range(4):
        suite.check(
            small.pop(out) and out == i, "items come back in the order sent"
        )
    suite.check(small.is_empty(), "and the queue is empty again")
    suite.check(small.push(5) and small.pop(out) and out == 5, "and reusable")

    # Three producers, one consumer, and a queue too small to hold what they
    # are sending, so every producer meets a full queue and has to wait.
    var queue = MpscQueue(QUEUE_CAPACITY)
    var total = PRODUCERS * PER_PRODUCER
    var jobs = List[PushJob](capacity=PRODUCERS)
    for t in range(PRODUCERS):
        jobs.append(
            PushJob(Int(Pointer(to=queue)), t * PER_PRODUCER, PER_PRODUCER)
        )

    var threads = List[Thread](capacity=PRODUCERS)
    for t in range(PRODUCERS):
        var thread = Thread()
        _ = spawn(_producer, Int(Pointer(to=jobs[t])), thread)
        threads.append(thread^)

    var seen = List[Bool]()
    for _ in range(total):
        seen.append(False)
    var taken = 0
    var duplicates = 0
    var deadline = monotonic_ms() + WAIT_MS
    while taken < total and monotonic_ms() < deadline:
        var value = 0
        if queue.pop(value):
            if value < 0 or value >= total:
                duplicates += 1
            elif seen[value]:
                duplicates += 1
            else:
                seen[value] = True
                taken += 1
            continue
        _ = sleep_ms(1)

    for i in range(len(threads)):
        _ = threads[i].join()

    suite.check(taken == total, "every value pushed comes out exactly once")
    suite.check(duplicates == 0, "and nothing comes out twice")
    suite.check(queue.is_empty(), "the queue is empty when they are done")

    var filled = 0
    for t in range(len(jobs)):
        if jobs[t].pushed == PER_PRODUCER:
            filled += 1
    suite.check(filled == PRODUCERS, "every producer sent its whole share")

    keep(queue)
    keep(jobs)


def _check_spsc(mut suite: Suite) raises:
    suite.group("sys.queue spsc")

    var small = SpscRing(4)
    suite.check(small.is_valid(), "a ring allocates")
    suite.check(small.is_empty() and not small.is_full(), "and starts empty")
    var out = 0
    suite.check(not small.pop(out), "an empty ring pops nothing")
    for i in range(4):
        suite.check(small.push(i * 10), "the ring takes four items")
    suite.check(small.is_full() and not small.push(1), "and then it is full")
    suite.check(small.pop(out) and out == 0, "the oldest comes out first")
    suite.check(small.push(40), "which makes room for one more")
    suite.check(small.depth() == 4, "the depth follows both cursors")

    # One thread on each end, which is the only configuration this ring is
    # correct for, and enough values that the ring wraps many times.
    var ring = SpscRing(32)
    var thread = Thread()
    _ = spawn(_ring_producer, Int(Pointer(to=ring)), thread)

    var next = 0
    var wrong_order = 0
    var deadline = monotonic_ms() + WAIT_MS
    while next < PER_PRODUCER and monotonic_ms() < deadline:
        var value = 0
        if ring.pop(value):
            if value != next:
                wrong_order += 1
            next += 1
            continue
        _ = sleep_ms(1)
    _ = thread.join()

    suite.check(next == PER_PRODUCER, "the reader sees every value")
    suite.check(wrong_order == 0, "and sees them in the order they were sent")
    suite.check(ring.is_empty(), "with nothing left in the ring")

    keep(ring)


def _check_thread_names(mut suite: Suite) raises:
    suite.group("sys.thread naming")

    suite.check(self_id() != 0, "a thread can name itself")
    var block = AtomicBlock(1)
    var slot = block.slot(0)
    var thread = Thread()
    _ = spawn(_named, slot.address, thread)
    _ = thread.join()
    suite.check(slot.load() == 1, "a spawned thread's name reads back")

    # The main thread is not renamed on purpose. On macOS the name of the first
    # thread is what the process shows in Activity Monitor, and taking it over
    # would be a lie about what the process is.
    keep(block)


def _check_drain(mut suite: Suite) raises:
    suite.group("net.server drain")

    var server = Server[EchoProtocol](
        ListenAddress(UInt16(0)), ServerContext(2, 60000, 0, 2000)
    )
    var port = server.port
    server.start()

    var clients = List[Int]()
    for _ in range(8):
        clients.append(_client(port))
    var deadline = monotonic_ms() + WAIT_MS
    while server.accepted() < 8 and monotonic_ms() < deadline:
        _ = sleep_ms(1)
    suite.check(server.accepted() == 8, "eight idle connections are open")

    var dump = server.dump()
    suite.check(
        dump.find("worker") >= 0 and dump.find("running") >= 0,
        "the dump names each worker and says what it is doing",
    )

    var report = server.drain()
    suite.check(report.clean, "an idle pool drains cleanly")
    suite.check(report.dropped == 0, "with nothing dropped")
    suite.check(report.open_at_end == 0, "and nothing left open")
    suite.check(report.accepted == 8, "the report counts what was served")
    suite.check(
        report.elapsed_ms < 2000, "and it does not wait for the deadline"
    )
    suite.check(
        report.describe().byte_length() > 0, "the report says so in one line"
    )

    var buf = stack_allocation[64, UInt8]()
    var closed = 0
    for i in range(len(clients)):
        var seen = monotonic_ms() + WAIT_MS
        while monotonic_ms() < seen:
            var got = recv(clients[i], buf, 64)
            if got == 0:
                closed += 1
                break
            if got > 0:
                continue
            _ = sleep_ms(1)
        _ = close(clients[i])
    suite.check(closed == 8, "and every client sees the connection close")


def _check_handler_error(mut suite: Suite) raises:
    suite.group("http handler error")

    var server = Server[HttpProtocol](
        ListenAddress(UInt16(0)), ServerContext(2, 60000, 0, 2000)
    )
    for i in range(len(server.reactors)):
        server.reactors[i].proto.configure_fault(True)
    var port = server.port
    server.start()

    var boom = _client(port)
    _ = _send_text(boom, "GET /boom HTTP/1.1\r\nHost: molla\r\n\r\n")
    var reply = List[UInt8]()
    _ = _read_until(boom, "\r\n\r\n", reply)
    suite.check(
        _contains(reply, "HTTP/1.1 500"), "a handler that raises returns a 500"
    )
    _ = close(boom)

    var after = _client(port)
    _ = _send_text(after, "GET /healthz HTTP/1.1\r\nHost: molla\r\n\r\n")
    var alive = List[UInt8]()
    _ = _read_until(after, "\r\n\r\n", alive)
    suite.check(
        _contains(alive, "HTTP/1.1 200"),
        "and the server is still serving afterwards",
    )
    _ = close(after)

    var report = server.drain()
    suite.check(report.clean, "and it still drains cleanly")


def run(mut suite: Suite) raises:
    _check_atomics(suite)
    _check_once(suite)
    _check_mpsc(suite)
    _check_spsc(suite)
    _check_thread_names(suite)
    _check_drain(suite)
    _check_handler_error(suite)
