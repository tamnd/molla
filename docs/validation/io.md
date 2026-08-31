# The memory layer

`molla.io` is where a request's bytes live. A read buffer per connection, an output ring per connection, an arena per request, and a set of span helpers that let the parser answer questions about bytes without copying them. This records what the policies actually are, what was measured rather than assumed, and the one bug that took a while to find because it did not fail anywhere near where it was caused.

## What is covered

| Module | What it is |
| --- | --- |
| `sys.mem` | Raw allocation, and the counter every other module records against. |
| `buffer` | An owned growable buffer with a written down growth policy, and a tail a socket can read straight into. |
| `ring` | The per connection output ring. A short write costs two integer updates, and the queued bytes come out as the one or two pieces `writev` takes. |
| `arena` | A bump allocator with a per request lifetime, freed in constant time, refusing rather than growing when it is full. |
| `bytes` | Compare, search, trim and parse over spans. Nothing here allocates except `to_string`, which says so. |

## The policies, and why they are these numbers

**Buffers double to 64 kB and then grow by a fixed 64 kB.** Doubling is right while the buffer is small, because a realloc per header is worse than a copy of a few hundred bytes. It is wrong once the buffer is large, because doubling a 4 MB buffer asks the allocator for 8 MB to hold 4 MB and one byte, and a thousand connections doing that is what an out of memory kill looks like from the outside.

**A buffer never shrinks on its own.** It belongs to a connection and gets reused by every request on it, so shrinking after a large request only means growing again for the next one. `reset_to` exists for the connection pool, which is the one place where the size of the last request has stopped predicting anything.

**The ring is fixed size and gives up one byte.** Growing an output buffer because a client is slow is how a server with slow clients runs out of memory, so a response that does not fit is written in more than one pass and `readable()` is what backpressure reads. Full and empty both have the two offsets equal, and telling them apart costs either a count, a flag, or one byte of capacity. The byte is the cheapest of the three because it keeps the reader and the writer independent.

**The arena refuses instead of growing.** A per request budget that grows on demand is not a budget, and a chain of blocks would make the constant time reset stop being constant time. A refused allocation returns zero, the caller falls back to the heap, and the counter records it, which is exactly the signal issue #17 is looking for. The high water mark is the number a budget gets set from.

## What was run

`tests/test_io.mojo` is 116 checks, and it runs as part of the same binary as the rest of the suite.

| Machine | Kernel | Arch | Checks |
| --- | --- | --- | --- |
| macbook, M4 | Darwin 24.6 | arm64 | 415 passed |
| server1, doge-01 | Linux 6.8, Ubuntu 24.04, glibc 2.39 | x86-64 | 415 passed |
| CI, ubuntu-24.04 and ubuntu-24.04-arm | Linux, Ubuntu 24.04 | x86-64 and arm64 | 415 passed |

The count is the whole suite, because `molla.io` sits on `molla.sys` and there is no useful way to run one without the other.

The checks worth naming are the wrap and the counter. The ring is filled to 120 of its 127 usable bytes, drained to 5, and pushed again, so the next write has to straddle the end of the block. That is where an off by one produces a response body with ten bytes of the previous response in the middle of it, which is a correctness bug that looks like a networking bug. And the counter is asserted on directly: a buffer is one allocation, writing inside the capacity is none, growing past it is one more, and a ring is one allocation no matter how much goes through it.

## What went wrong

**A counter with a destructor frees itself while the buffers are still counting.** The allocation counter is three integers in a block, and buffers hold its address rather than the counter itself, so that counting survives a buffer being moved into a list. Giving that block a destructor looks obviously right and is obviously wrong: Mojo destroys a value at its last use, not at the end of the scope, and the last use of a counter is the last time something reads `total()`. That happens while every buffer holding the address is still alive. The block was freed there, the next buffer to grow added three integers to freed memory, and the process aborted inside malloc in a completely unrelated test that happened to allocate next. The counter is closed by hand now, and the reason is written above `AllocCounter` so nobody helpfully adds the destructor back.

That failure mode is the argument for the counter existing at all. Memory bugs in this layer do not fail where they are caused, and a number that says how many allocations a request made is the difference between noticing that in a test and noticing it in production.

## What is not covered

The zero allocation claim. This layer builds the counter and asserts the counter is honest. Whether a request that parses and responds actually allocates nothing is issue #17, and it cannot be answered until there is a request path to measure.

Concurrency. The counter is deliberately not atomic, because putting a lock on the allocation path to measure a path that is not supposed to allocate would be measuring the instrument. Each connection gets its own counter, and two threads sharing one will lose updates.

Fragmentation over time. Every test here runs for milliseconds. How these blocks behave after a day of traffic is a load question, which is issue #15's.
