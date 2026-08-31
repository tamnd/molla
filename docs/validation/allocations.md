# Proving the request path allocates nothing

Issue #17. The design says molla allocates nothing per request in steady state. That claim was true when it was written and there was nothing stopping it becoming false, because nothing about adding a string to a handler tells you that you have just put a malloc on the hot path. So it has an assertion now, and the assertion runs on every commit on all three platforms.

The criterion was that the assertion runs on every commit. It does, twice: a small version inside the test suite and a bigger one as its own CI step.

## What it measures

`molla allocs` runs a mixed load twice against a real server on a real socket.

The first pass is a warm up. Every buffer grows to the size the traffic needs, every connection slot in the reactor gets built, every response the writer will ever produce is produced once. The allocation counter is read after that, the identical load runs again, and the second reading has to equal the first exactly. Not nearly. There is no tolerance because a tolerance is a budget and a budget gets spent.

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
| `GET /stream/sse` | the other framing, and the one that asks the server to close |

Anything that allocates on the fifth kind of request is invisible to a test that only sends the first.

Every connection in a round is opened before any of them is written to, and all of them are written to before any of them is read. That ordering is the difference between exercising the reactor's whole slot table and exercising one slot. Opening and finishing one connection at a time lets the server accept, answer and free the same slot every time, which proves that one slot is reused and nothing about the other sixty three. This was not a hypothetical either, it is how the first version of the command was written, and it reported three allocations of warm up at sixty four connections, which should have been the giveaway.

The byte count read back is compared between the two passes as well. A pass that quietly answered nothing would allocate nothing too, and would otherwise be the best result this command could produce.

## What it found

A per connection allocation, which is to say the claim was already false.

The reactor built a fresh `Connection` when it reused a slot. That freed and re-calloc'd the read buffer and the write ring on every accept, and then the read buffer had to grow again the first time a request did not fit in its starting size. Three allocations per connection. Against a client that opens a connection per request, which is most of them, that is a per request allocation on a server whose whole pitch is not having one.

The fix is `Connection.reuse`, which sets everything back to what a new connection would have and leaves the buffers alone. They keep whatever size the last connection grew them to, which is the point: after a warm up the traffic has already paid for the size it needs.

Before, at sixty four connections:

```text
  warm up        48 allocations, 114432 bytes
  steady state   48 allocations, 114432 bytes read
```

After:

```console
$ molla allocs 64 3
allocs 64 connections, 3 rounds
  workers        1
  warm up        192 allocations, 114432 bytes read
  steady state   0 allocations, 114432 bytes read
  heap grew by   0 bytes
  result         pass
```

Warm up went up because the slot table is now genuinely exercised, and that number is the one that should be there: three allocations for each of sixty four slots, paid once.

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
