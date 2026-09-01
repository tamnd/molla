"""The zero allocation claim, checked on every commit.

Issue #17. The README says the request path allocates nothing in steady state.
That is the kind of sentence which is true when it is written and quietly false
two months later, because nothing about adding a `String` to a handler tells you
that you have just put a malloc on the hot path. So the claim gets an assertion,
and the assertion runs here rather than only in a benchmark somebody remembers
to look at.

The shape is a warm up and a steady pass. Everything molla allocates for a
connection is allocated the first time it needs it: the read buffer, the write
ring, the reactor's slot table, the responses the writer builds. The load runs
until one pass of it costs nothing, the identical load runs once more, and that
last reading has to be zero. There is no tolerance, because a tolerance is a
budget and a budget gets spent. A load that never stops costing runs out of
warm up passes and fails saying that, which is what a real regression looks
like against this.

`molla allocs` is the same measurement with more connections and more rounds,
wired into CI as its own step. What is here is the version that fits in a test
suite, which is small enough to be quick and is still the thing that would have
caught the bug this issue found: the reactor was rebuilding a `Connection` for a
slot it already had, so every accept freed and re-calloc'd a buffer, and every
first request on a connection grew it again.

The unit level check is here too, because the fix has a property worth stating
on its own. `Connection.reuse` has to keep the buffers, and a future edit that
turns it back into a fresh `Connection` would still pass every functional test
in this repo.
"""

from harness import Suite

from molla.net.allocs import measure_allocs
from molla.net.conn import Connection
from molla.sys.mem import AllocCounter, keep

comptime TEST_CONNECTIONS = 4
comptime TEST_ROUNDS = 2
"""Small. The load is a pipelined batch of thirteen requests per connection per
round, so this is a hundred and four requests through the whole thing, which is
a second or so and is plenty for the second pass to have nothing left to
allocate."""


def _check_conn_reuse(mut suite: Suite):
    suite.group("net.conn reuse")

    var counter = AllocCounter()
    if counter.raw() == 0:
        suite.check(False, "the counter allocates")
        return

    var conn = Connection(7, 0, 1, 1024, 1024, counter.raw(), 100)
    suite.check(
        conn.input.capacity >= 1024, "a new connection has a read buffer"
    )

    # Whatever the connection did with its buffers, the slot keeps.
    var grown = conn.input.capacity
    var after_build = counter.total()

    conn.bytes_in = 500
    conn.bytes_out = 900
    conn.short_writes = 3
    conn.peer_done = True
    conn.closing = True
    conn.closed = True
    conn.write_interest = True
    conn.producing = True
    conn.timer = 42

    conn.reuse(9, 2, 250)

    suite.check(
        counter.total() == after_build,
        "reusing a slot does not allocate, which is the whole point of it",
    )
    suite.check(
        conn.input.capacity == grown,
        "and the read buffer the last connection grew is still there",
    )
    suite.check(conn.fd == 9, "the new socket is in place")
    suite.check(
        conn.generation == 2, "with a generation the old one cannot use"
    )
    suite.check(conn.last_active_ms == 250, "the idle clock starts again")
    suite.check(conn.input.length == 0, "nothing is left in the read buffer")
    suite.check(conn.output.is_empty(), "nor in the write ring")
    suite.check(conn.timer == -1, "the timer is unarmed")
    suite.check(
        not conn.peer_done and not conn.closing and not conn.closed,
        "and none of the closing flags survive into the next connection",
    )
    suite.check(
        not conn.write_interest and not conn.producing,
        "neither does the writability interest or a half done stream",
    )
    suite.check(
        conn.bytes_in == 0 and conn.bytes_out == 0 and conn.short_writes == 0,
        "the byte counters are per connection and start again",
    )
    keep(counter)


def _check_steady_state_allocates_nothing(mut suite: Suite) raises:
    suite.group("net.allocs")

    var report = measure_allocs(TEST_CONNECTIONS, TEST_ROUNDS)

    suite.check(
        report.warm_bytes_read > 0,
        "the warm up pass got answers back, so there was something to measure",
    )
    suite.check(
        report.steady_bytes_read == report.warm_bytes_read,
        "the second pass read back exactly what the first one did",
    )
    suite.check(
        report.steady_allocations == 0,
        (
            "and a mixed load of GET, HEAD, 404, a body, chunked, pipelined and"
            " two streams allocated nothing at all the second time round"
        ),
    )
    suite.check(
        report.steady_bytes_grown == 0,
        "so the heap is the size it was before the pass",
    )
    suite.check(report.drained, "and the server shut down clean afterwards")
    if report.steady_allocations != 0:
        print(
            "  steady state allocated",
            report.steady_allocations,
            "times for",
            report.steady_bytes_grown,
            "bytes. Run `molla allocs` for the full numbers.",
        )


def run(mut suite: Suite) raises:
    _check_conn_reuse(suite)
    _check_steady_state_allocates_nothing(suite)
