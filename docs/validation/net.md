# The reactor

`molla.net` is the event loop every request will arrive on. One reactor per worker thread, each owning a poller, a connection table and a timing wheel, with no shared state on the request path. This records what it does, the two places it deliberately does something other than what issue #10 asked for, the bug the design exists to prevent and the one it did not prevent, and what was measured rather than assumed.

The M0 spike in `molla.net.echo` is still in the tree and is a different thing. It is one file that proved a socket server was possible in Mojo at all, and it is the evidence behind D1. This is the one that has to hold up.

## What is covered

| Module | What it is |
| --- | --- |
| `wheel` | Four levels of 64 slots at 100 ms. An idle connection costs one slab entry and no descriptor. |
| `conn` | One accepted socket, its read buffer, its output ring, and the four states a non blocking connection can be in. |
| `listener` | Opening a listening socket, and the platform split over how connections get spread across workers. |
| `reactor` | The loop. Accept, drain, service, flush, expire, and the `Protocol` trait everything above will be written against. |
| `protocol` | Echo, as a reference implementation of that trait and as what the tests drive. |
| `server` | N reactors on N threads behind one address, TCP or unix. |
| `soak_net` | `molla netsoak`, the acceptance test. |

## The two deliberate deviations from issue #10

**The poller stays edge triggered on both platforms.** The issue asked for edge triggered on Linux and level triggered on macOS. Running kqueue level triggered means the two platforms need different drain rules, and the drain rule is the part that is easy to get subtly wrong: with level triggering you may read once and be told again, with edge triggering you must read until EAGAIN or the rest of the bytes sit in the kernel with nothing scheduled to come back for them. Two rules means the rule that is wrong is only wrong on one platform, and it is the platform nobody develops on. So both are edge triggered, `fill` reads until EAGAIN, and every test exercises the same path on both machines. The cost is that kqueue reports readiness edges molla may not have drained yet, and the answer to that is the same EAGAIN loop.

**Unix sockets are never sharded.** The issue said SO_REUSEPORT sharded accept on Linux, and that is what TCP does. A unix socket is one path in the filesystem, so binding it from several reactors would mean each new listener unlinking the previous one's address, and the traffic that arrives on a unix socket does not need the accept queue split four ways. One acceptor and a handoff.

## The platform split that is not optional

On Linux each worker opens its own listening socket on the same port with SO_REUSEPORT and the kernel hashes incoming connections across them. No shared accept queue, no thundering herd, no cross thread handoff, and the thread that accepts a connection is the thread that serves it.

On macOS SO_REUSEPORT exists and means something else. Several sockets may bind the same port, and the last one to bind gets every connection rather than the kernel balancing between them. A server sharded that way on macOS would run all its traffic on one worker and would look completely fine in a test with one connection, which is the worst kind of wrong. So macOS uses one listening socket, one thread accepting from it, and a round robin handoff of accepted descriptors to the workers over a mutex and a one byte wakeup socket. That is one queue push and one wakeup per connection, which is real and is not on the request path, and macOS is the development platform rather than the deployment one.

`SHARDED_ACCEPT` is the compile time constant that decides, and both paths are covered by `tests/test_reactor.mojo`, which asserts a different thing on each: that every reactor got its own listener where the kernel balances, and that round robin actually used every reactor where it does not.

## The timing wheel, and why not one timer per connection

A connection that connects and says nothing has to be closed eventually. That is not a hypothetical, it is the cheapest denial of service there is: the attacker spends one socket and the server spends a descriptor until it runs out. So every connection needs a deadline.

The obvious implementation is one timer descriptor per connection, and it does not survive contact with a thousand connections. The wheel is four levels of 64 slots at 100 ms per tick, so a deadline costs an integer in a slab and a link into a slot, arming is constant time, and the whole structure is walked one slot per tick regardless of how many timers are in it. A deadline far in the future sits in a coarse level and is moved down at most once per level, so its amortised cost is constant no matter how far out it was set.

Firing is a question rather than an answer. A timer that comes due means "this connection may have been idle for its whole allowance", and the reactor checks the connection's last activity before closing it. That is what makes activity free: touching a connection is one store to a field, not a timer cancel and a rearm, and the correction happens at most once per timeout period on the connections that were actually busy.

## Servicing a connection is a loop, and that is not decoration

The first version of `_service` handled one readiness event and returned, which is the shape every tutorial has. It deadlocks, and the test caught it.

A connection can read more in one pass than its output ring holds. The leftover stays in the read buffer waiting for room, which is correct. What is not correct is what happens next: the socket has already been drained, so no read edge is coming, and if the ring drained fully in the same pass then write interest goes off, so no write edge is coming either. The connection sits there with a response half written and nothing scheduled to finish it. Under light load it never happens, because the ring is never full. Under the load that makes it worst it happens every time.

So `_service` loops while it is making progress, measured as bytes actually moving out of the read buffer or out of the ring, up to a budget of eight rounds. A connection that still has work when the budget runs out goes on a list, and a pass with a non empty list does not block in the poller. That bounds how long one busy connection can hold the loop while a thousand others wait, without ever leaving work with nothing to wake it.

## What was run

`tests/test_reactor.mojo` is 61 checks on macOS and 62 on Linux, because the sharded accept path has one more thing to assert than the handoff path does, and it runs as part of the same binary as the rest of the suite. The counts below are the whole suite, since that is what the runner prints.

| Machine | Kernel | Arch | Suite |
| --- | --- | --- | --- |
| macbook, M4 | Darwin 24.6 | arm64 | 474 passed |
| server1, doge-01 | Linux 6.8, Ubuntu 24.04, glibc 2.39 | x86-64 | 475 passed |
| server2, doge-02 | Linux 6.8, Ubuntu 24.04, glibc 2.39 | x86-64 | 475 passed |
| CI, ubuntu-24.04 | Linux, Ubuntu 24.04 | x86-64 | 475 passed |
| CI, ubuntu-24.04-arm | Linux, Ubuntu 24.04 | arm64 | 475 passed |
| CI, macos-15 | Darwin | arm64 | 474 passed |

The wheel is tested against an explicit clock rather than by sleeping, because a test that waits for a real deadline is a test that fails on a loaded CI runner. The reactor tests drive `poll_once` by hand and act as the client between passes, which is the only way to be both ends on one thread. The server tests use real threads, so the server side runs where it will really run.

The backpressure test pins the socket buffers at both ends to 8 kB rather than pushing bytes until the kernel happens to fill. Left alone, both platforms size those buffers themselves and Linux grows them over the life of a connection, so the amount of data needed to provoke a short write is different on every machine. The first version pushed half a megabyte, which was enough on macOS and not enough on Linux, so the test passed on the development machine and failed on the deployment one. An accepted socket inherits the listener's buffer sizes, which is what makes this settable from outside the reactor.

## The soak

Issue #10 is done when the reactor holds a thousand connections with mixed idle and active traffic for an hour without leaking descriptors and without latency drifting. `molla netsoak 1000 3600` runs exactly that and checks itself.

Mixed means one connection in eight sends and the rest connect and stay silent, which is what a keep alive pool looks like between requests and is the shape that finds the bug where a reactor scans its whole table per pass. Drift is measured by cutting the run into ten segments and comparing the p99 of the last against the first. Descriptors are checked by opening a socket after teardown and reading the number the kernel hands back, which is the lowest free one on both platforms, so a run that leaked even one comes back high. Memory is `getrusage` peak, which is weaker than it sounds: it can show that nothing grew, not that nothing leaked and was reused.

| Machine | Conns | Round trips | Mismatched | p99 first | p99 last | Probe fd | Peak rss | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macbook, M4 | 1000 | 657195369 | 0 | 1024 us | 1024 us | 3 before, 3 after | 90288 to 94352 kB | passed |
| server1, doge-01 | 1000 | 59256042 | 0 | 65536 us | 65536 us | 3 before, 3 after | 26240 to 31232 kB | passed |

Both ran the full hour with all thousand connections still held at the end, and the p99 is flat to the bucket across all ten segments on both, which is the thing being measured. The absolute latencies are not comparable between the two machines and are not meant to be: the M4 gave the soak ten workers and had nothing else to do, server1 has four cores and a desktop session on it, and the client is in the same process as the server on both. The soak is asking whether an hour changes anything, and on both machines the answer is that it does not.

One caveat on provenance. The M4 soak ran the binary from the commit this document ships with. The server1 soak was already an hour into its run when the socket buffer fix landed, so its binary is one commit older, and that commit differs only in `tests/test_reactor.mojo`, `src/molla/sys/socket.mojo` and documentation. Nothing under `src/molla/net/` changed between them, so both soaks exercised the same reactor.

The rss numbers are the honest weak point. Peak grew by about 4 MB on the M4 and about 5 MB on server1, both in the first segment as the wheel's slabs and the connection table reached their working size, and neither moved after that. Peak rss cannot tell the difference between memory that was freed and memory that leaked and was handed back out, so it is evidence that nothing is growing without bound rather than evidence that nothing leaks. Issue #17's allocation counter is what turns that into a number.

## What is not covered

Throughput. This layer is about whether the loop stays correct and stays the same size. How many requests per second it does is issue #12's, and there is no request on it yet to measure.

The handoff under real contention. The macOS path pushes an accepted descriptor onto another reactor's queue under a mutex, and the soak exercises it with a thousand connections arriving as fast as one thread can open them. That is a burst, not sustained contention from several acceptors, and there is only ever one acceptor on that path by construction.

aarch64 Linux. No machine in the fleet is one, so the `epoll_event` layout for that target is only run by the CI job, which runs the unit suite and not the soak. The same gap the socket layer has, for the same reason.

Descriptor exhaustion. The accept loop treats EMFILE and ENFILE as a pause and counts them, which is the right behaviour, and nothing yet drives a server into that state on purpose.
