"""Wall clock time from libc.

`std.time.monotonic` is the right clock for measuring durations and the wrong
one for anything a client sees, because it counts from an arbitrary point. HTTP
needs a real date on the response, so this reaches for `time`.

There is exactly one function here on purpose. Anything that wants to measure
how long something took should keep using `monotonic`, which does not need FFI
and cannot jump backwards when someone corrects the clock.
"""

from std.ffi import external_call
from std.memory import stack_allocation


def unix_time() -> Int:
    """Seconds since the epoch.

    `time` takes an optional out pointer and returns the same value. Pointers
    are non nullable in Mojo 1.0, so rather than fighting to express `NULL` we
    hand it a stack slot and ignore it.
    """
    var slot = stack_allocation[1, Int64]()
    slot.unsafe_store(0, Int64(0))
    return Int(external_call["time", Int64](slot))
