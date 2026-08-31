"""Wall clock time from libc.

`std.time.monotonic` is the right clock for measuring durations and the wrong
one for anything a client sees, because it counts from an arbitrary point. HTTP
needs a real date on the response, so this reaches for `time`.

The other function here is `monotonic_ms`, which is not FFI at all. It is
`std.time.monotonic` divided down to milliseconds, and it exists so that the
timing wheel and the reactor have one obvious thing to call instead of each
picking their own unit. Milliseconds because that is what a poll timeout takes
and what an idle timeout is configured in.
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


def monotonic_ms() -> Int:
    """Milliseconds on a clock that only goes forward.

    Counted from an arbitrary point, so the number means nothing on its own and
    everything as a difference. Every timeout in the reactor is measured with
    this, because a wall clock that jumps when ntp corrects it would either fire
    every timer at once or hold connections open for hours.
    """
    return Int(monotonic() // 1000000)
