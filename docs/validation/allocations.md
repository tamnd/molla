# Proving the request path allocates nothing

Issue #17. The design says molla allocates nothing per request in steady state. That claim was true when it was written and there was nothing stopping it becoming false, because nothing about adding a string to a handler tells you that you have just put a malloc on the hot path. So it has an assertion now, and the assertion runs on every commit on all three platforms.

The criterion was that the assertion runs on every commit. It does, twice: a small version inside the test suite and a bigger one as its own CI step.

## What it measures

`molla allocs` runs a mixed load against a real server on a real socket until one run of it costs nothing, then runs it once more and requires that last one to cost nothing too. Not nearly nothing. There is no tolerance, because a tolerance is a budget and a budget gets spent.

The warm up is where every buffer grows to the size the traffic needs, every connection slot in the reactor gets built, and every response the writer will ever produce is produced once.

The number of reactor slots is the awkward part, and it took two goes to make it deterministic. A slot is built the first time the reactor needs one and reused afterwards, so the slot count is the high water mark of connections open at once, and that mark is not the connection count: a round of the load closes its connections and the next round opens its own before the reactor has necessarily reaped the last ones, so whether one round overlaps the one before it by a connection is a scheduling question. A warm up that peaked one slot short leaves that slot for the steady pass to build, and the run fails for a reason that has nothing to do with the request path. That is not hypothetical, it is what the macOS CI runner caught twice while this was being written, both times at exactly three allocations, which is one slot.

So a primer pass opens twice the connection count at once before anything is measured. A pass never has more than the connection count open, plus at most the previous round's, so twice it is a ceiling the load cannot reach. After that the slot table is a property of the load rather than of the scheduler. The warm up then runs the real load until a pass of it costs nothing, up to eight passes, which covers everything else that is paid for once.

Neither of those is a loosening of the check. A real per request allocation never stops costing, so it runs out of warm up passes and fails, and it fails saying it ran out rather than quoting a number that looks like a rounding error.

The load is mixed on purpose, because the interesting allocation is the one on a path a simpler test does not take. One connection sends this batch, pipelined, in a single write:

| Request | Why it is in there |
| --- | --- |
| `GET /` | the ordinary case |
| `HEAD /` | the headers without the body, which is a different write path |
| `GET /healthz` | a second route, so the answer is not one cached string |
| `GET /nowhere` | a 404, built by the error path rather than the handler path |
| `POST /` with Content-Length | a body that arrives with a known length |
| `POST /` chunked | a body that arrives without one, through the chunked decoder |
| eight pipelined `GET /` | a batch in one segment, which takes the parse loop round again with no new readiness event |
| `GET /stream/ndjson` | a streaming response, several events per request |
| `GET /stream/sse` | the other framing, which frames the same events differently |

Anything that allocates on the fifth kind of request is invisible to a test that only sends the first.

The client ordering matters more than the load does, and it took three tries to get right.

Every connection in a round is opened before any of them is written to, all of them are written to before any of them is read, and every one of them has been answered before any of them is asked to close. The first version of the client opened and finished one connection at a time, which let the server accept, answer and free the same slot every time, so it proved that one slot gets reused and nothing about the other sixty three. It reported three allocations of warm up at sixty four connections, which should have been the giveaway.

Opening all of them first fixed that and left a flake behind. The batch ended with `Connection: close`, so the server was closing the early connections while the late ones were still in the accept queue, and the reactor served the whole load out of fifty five or sixty or sixty four slots depending on how the scheduler felt. A slot costs three allocations the first time and nothing afterwards, so a warm up that happened to build fifty five of them left nine for the steady pass to build, and the run failed about half the time. The closing request is now sent separately, after every connection has been answered once, and the slot count is the connection count on every run.

That is worth spelling out because the failure mode is the bad one: an assertion that fails half the time gets an exception added to it, and then it is not an assertion.

The byte count read back is compared between the two passes as well. A pass that quietly answered nothing would allocate nothing too, and would otherwise be the best result this command could produce.

## What it found

A per connection allocation, which is to say the claim was already false.

The reactor built a fresh `Connection` when it reused a slot. That freed and re-calloc'd the read buffer and the write ring on every accept, and then the read buffer had to grow again the first time a request did not fit in its starting size. Three allocations per connection. Against a client that opens a connection per request, which is most of them, that is a per request allocation on a server whose whole pitch is not having one.

The fix is `Connection.reuse`, which sets everything back to what a new connection would have and leaves the buffers alone. They keep whatever size the last connection grew them to, which is the point: after a warm up the traffic has already paid for the size it needs.

Before, at sixty four connections:

```text
  warm up        6528 allocations in 8 passes
  steady state   768 allocations, 152576 bytes read
  heap grew by   29360128 bytes
  the load never stopped allocating, which is the thing this looks for
  result         fail
```

After:

```console
$ molla allocs 64 4
allocs 64 connections, 4 rounds
  workers        1
  warm up        384 allocations in 1 pass
  steady state   0 allocations, 152576 bytes read
  heap grew by   0 bytes
  result         pass
```

Three allocations per connection, two hundred and fifty six connections, and twenty eight megabytes of churn to serve a hundred and fifty kilobytes of answers. Afterwards the warm up number is the one that should be there, which is three allocations for each of a hundred and twenty eight slots, paid once.

## What it cannot see

Mojo's own allocations. `List`, `String` and anything the standard library does internally go to the system allocator directly and not through `molla.sys.mem`, so what is counted is molla's heap traffic rather than the process's. A handler that builds a `String` would not be caught by this. That is a real gap and the honest answer is that closing it needs an allocator hook that Mojo 1.0 does not offer, so what is here is the part that can be measured today.

It runs one worker. The counter is not atomic, and two workers counting against one counter would lose updates, which is the direction that hides a bug rather than the direction that invents one. The per connection work is identical on every worker, so a second one would add confidence about the counter and nothing about the request path.

It says nothing about how much is allocated at startup, or about the arena. The arena has its own tests in `docs/validation/io.md`. This is about the steady state only.

## Where it runs

The suite has a small version, four connections and two rounds, which is a hundred and four requests. That is the one that runs on every `pixi run test`, so it fails in the pull request that broke it.

CI runs `molla allocs 16 4` as its own step on linux-x86_64, linux-aarch64 and macos-aarch64, after the suite. Bigger, and separate, so the failure says which thing broke without anybody reading a test name.

There is a unit level check on `Connection.reuse` too, because the fix has a property worth stating on its own: a future edit that turns it back into a fresh `Connection` would pass every functional test in this repo.

## What was run

| Machine | Kernel | Cores | Suite | `allocs 64 4` |
| --- | --- | --- | --- | --- |
| macbook, M4 | Darwin 24.6, macOS 15.8 | 10 | 1035 passed | 0 allocations |
| server1, EPYC | Linux 6.8, Ubuntu 24.04 | 4 | 1036 passed | 0 allocations |
| server2, EPYC | Linux 6.8, Ubuntu 24.04 | 6 | 1036 passed | 0 allocations |
| gpc, i9-13900K | Linux 6.18 on WSL2, Ubuntu 26.04 | 32 | 1035 passed, 1 failed | 0 allocations |

The suite is one check longer on Linux than on macOS, because sharded accept only exists there.

The one failure on gpc is issue #87, the reactor backpressure test on WSL2, which fails on the commit before this one too.
