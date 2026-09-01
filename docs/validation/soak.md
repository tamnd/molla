# The hour long soak on the systems layer

Issue #18. Everything below the HTTP layer has its own tests and the reactor has its own soak, and none of that answers the question this one asks: does the whole stack stay the same size and the same speed for an hour while five kinds of client that annoy it in different ways are all connected at once.

The criterion was that the soak is green on macOS and Linux and runs nightly. It is, and it does.

## What it runs

`molla httpsoak [connections] [seconds]`, a thousand connections for an hour by default, which is what the nightly workflow runs with no arguments.

One process is both ends. The server runs on its own worker threads and every client runs on the main thread, which is what makes the round trip time worth measuring: a real client handing a request to a real event loop on another core and waiting for the answer to come back.

The five kinds of client are the point, and they run together rather than one after another. A server that survives each of them separately is not the thing anybody runs.

| Client | What it does | What it is there to exercise |
| --- | --- | --- |
| Keep alive | Alternates `GET /` with a `POST /healthz` carrying a body, thousands of times down one connection | The parser and the writer reused without a fresh connection to hide state behind, and the body reader on the hot path rather than in a test of its own |
| Streaming | Alternates `/stream/ndjson` and `/stream/sse`, reading each response whole | Chunked framing, both flavours, and a response the server produces in several passes |
| Slow reader | Asks for a stream and then reads sixty four bytes every hundred milliseconds | The write ring full and held full, which is the only way to keep the backpressure path warm for an hour |
| Abrupt | Sends a request and closes two milliseconds later without reading a byte | The write to a socket whose peer has gone, tens of thousands of times |
| Oversized | Announces a body of a megabyte against a limit of four kilobytes, gets a 413, closes | The error path with a close on the end of it, and the limit doing its job every time rather than most of the time |

Half the connections are keep alive, because that is what most traffic is and because the latency numbers should come mostly from the ordinary case. The other four kinds get an eighth each, which at a thousand connections is a hundred and twenty five of each, enough that all five things are happening continuously rather than occasionally.

The operations surface is on, with the log ring at warn and metrics enabled, because a soak of the systems layer that leaves out the parts an operator would have running is a soak of a server nobody is going to run.

## What it watches

Four things, and they are the four ways a server dies slowly rather than loudly.

Resident memory. Sampled per segment on Linux, which publishes the current figure in `/proc/self/statm`, and as a peak from `getrusage` on both platforms. macOS has no equivalent file and the mach call that answers the same question is a different kind of dependency, so on macOS the current figure is reported as not readable rather than invented, and the gate there is the peak. Saying which platform gets the stronger check is better than pretending both do.

Descriptors. A socket is opened after teardown and the number the kernel hands back is compared with the same probe taken before the run. Descriptors are allocated lowest free first on both platforms, so a leak comes back high. This soak reconnects constantly, tens of thousands of times over an hour, so a leak of one descriptor per connection would be obvious and a leak of one in a thousand would still show.

Queues. The log ring and the reactor's connection table. A ring that ends the run with bytes still in it is a flush that stopped keeping up, and connections that survive the clients closing are a reap that stopped happening.

Latency drift. The run is cut into ten segments, every round trip lands in the segment that was current when it finished, and the last segment is compared against the first. Latency that is flat is the claim. Latency that climbs means something is accumulating, and the number says how fast.

The slow readers are deliberately left out of the latency numbers. Their round trip time is their own doing and including it would drown the signal in a delay the test itself is causing.

The gate on drift is four times, which is wide on purpose. The histogram is powers of two microseconds, so two neighbouring buckets are already a factor of two apart and a run that crosses one boundary is noise rather than drift. The histogram lives in `molla.net.latency` and is shared with the reactor soak, so the gate means the same thing in both. It used to be two copies of the same code, which is two chances for the one number this rests on to mean something slightly different in each.

There is a correctness gate underneath all four. Every answer is checked against the status that kind of client should be getting, per kind rather than as a range, because a run where the oversized clients started getting 200 and the keep alive clients started getting 413 would have exactly the same totals as a clean one.

## What it found

A metrics bug, on the first run that got far enough to print a report.

`HttpProtocol._error` answers a 413, a 414 or a 431 from the parse path without going through `_write_default`, which is where the status accounting lived. So a run that sent a hundred thousand oversized bodies and got a hundred thousand 413s reported `molla_http_responses_4xx_total 0`. Those are the answers an operator most wants a graph of, and they were the ones that were never counted. The soak caught it because it checks the server's own count of 4xx answers against what the clients read back, which is the kind of cross check a unit test of the error path would not have.

The fix is one line, `_error` now calls `_account` the same way `_write_default` does.

It also found something about the client rather than the server, which is worth recording because it looked like a server failure for an hour. Every socket in molla is non blocking from the moment it is created, so `connect` returns before the handshake finishes and the first write on a fresh socket fails until it does. The client's first version treated that as a send failure after two hundred tight retries, and reported a couple of hundred failures per run against a server that was fine. A socket that is not ready now says so and the caller comes back on the next pass, with a five second budget before it counts as stuck. A loopback handshake finishes in microseconds, so anything near five seconds is a socket that will never take a byte.

## What it cannot see

It runs one process, so it says nothing about a server under a real client that lives somewhere else, and nothing about a network that drops or reorders anything. Loopback does neither.

It does not measure throughput and the numbers it prints should not be quoted as though it did. The client is one thread pacing itself with think times, so the request rate is what the client chose and not what the server could do.

The memory gate on macOS is a peak rather than a level, which is weaker than it sounds. What it can prove is the useful direction: if the peak at the end of an hour is the peak from the moment the connections came up, nothing grew while the loop ran.

Allocation counting is deliberately not in here. `AllocCounter` is explicitly not atomic and the soak runs several workers, so a count taken across them would be a number that looks precise and is not. The zero allocation claim has its own assertion in `molla allocs`, on one worker, where the count means something. See `docs/validation/allocations.md`.

## Where it runs

`.github/workflows/soak.yml`, at 03:17 UTC every night, on `ubuntu-24.04` and `macos-15`, with a ninety minute timeout and a `workflow_dispatch` trigger that takes a connection count and a duration. It raises the descriptor limit first, because one process holding a thousand connections is two thousand descriptors plus the listener, and the macOS default of 256 would fail on the first batch.

It is not in CI. An hour is longer than anybody will wait on a pull request, and the thing it is looking for needs the hour to show up at all.

The test suite runs a short version, sixteen connections for three seconds, which checks every gate the long run checks on numbers too small to prove anything about an hour. That is the division: the suite says the soak works, the nightly says the server does. The suite also covers the client side framing on responses assembled by hand, because a `Wire` that finds the end of a response too early counts two answers where there was one, and a run like that passes while measuring a server that was falling apart.

## What was run
