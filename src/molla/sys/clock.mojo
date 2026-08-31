"""Wall clock time from libc.

`std.time.monotonic` is the right clock for measuring durations and the wrong
one for anything a client sees, because it counts from an arbitrary point. HTTP
needs a real date on the response, so this reaches for `time`.

The other function here is `monotonic_ms`, which is not FFI at all. It is
`std.time.monotonic` divided down to milliseconds, and it exists so that the
timing wheel and the reactor have one obvious thing to call instead of each
picking their own unit. Milliseconds because that is what a poll timeout takes
and what an idle timeout is configured in.

The rule for which clock to reach for is worth stating once. A duration is
always measured on the monotonic clock, because a wall clock that jumps when
ntp corrects it turns a two millisecond request into a two second one or into a
negative one. The wall clock is only for something a human or a client reads: a
Date header, a log line timestamp. Nothing subtracts two realtime readings, and
every duration molla exports is an integer count of nanoseconds rather than a
float of seconds, so that no reader has to guess what unit it is in and no
precision is lost on the way out.
"""

from std.ffi import external_call
from std.memory import stack_allocation
from std.time import monotonic


def unix_time() -> Int:
    """Seconds since the epoch.

    `time` takes an optional out pointer and returns the same value. Pointers
    are non nullable in Mojo 1.0, so rather than fighting to express `NULL` we
    hand it a stack slot and ignore it.
    """
    var slot = stack_allocation[1, Int64]()
    slot.unsafe_store(0, Int64(0))
    return Int(external_call["time", Int64](slot))


comptime CLOCK_REALTIME = 0
"""Zero on Linux and on macOS, which is the only reason this file does not have
a platform branch in it."""


def realtime_ns() -> Int:
    """Nanoseconds since the epoch, for something a person will read.

    `time` only gives seconds, which is too coarse for a log line, so this goes
    through `clock_gettime`. A `timespec` is two 64 bit words on every target
    molla builds for, so the stack slot below is the struct.

    Never subtract two of these. It is the clock that moves when somebody fixes
    the machine's idea of the date.
    """
    var slot = stack_allocation[2, Int64]()
    slot.unsafe_store(0, Int64(0))
    slot.unsafe_store(1, Int64(0))
    var rc = external_call["clock_gettime", Int32](Int32(CLOCK_REALTIME), slot)
    if Int(rc) != 0:
        return unix_time() * 1000000000
    return Int(slot.unsafe_load(0)) * 1000000000 + Int(slot.unsafe_load(1))


def monotonic_ns() -> Int:
    """Nanoseconds on a clock that only goes forward.

    What every exported duration is measured in. `monotonic_ms` is this divided
    down, and it stays because a poll timeout is in milliseconds, but anything
    that ends up in a metric or a log field comes through here so that a
    sub millisecond request does not round to zero.
    """
    return Int(monotonic())


def monotonic_ms() -> Int:
    """Milliseconds on a clock that only goes forward.

    Counted from an arbitrary point, so the number means nothing on its own and
    everything as a difference. Every timeout in the reactor is measured with
    this, because a wall clock that jumps when ntp corrects it would either fire
    every timer at once or hold connections open for hours.
    """
    return Int(monotonic() // 1000000)
