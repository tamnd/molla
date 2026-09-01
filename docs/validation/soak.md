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

Five things, and they are the ways a server dies slowly rather than loudly.

Resident memory. Sampled per segment on Linux, which publishes the current figure in `/proc/self/statm`, and as a peak from `getrusage` on both platforms. macOS has no equivalent file and the mach call that answers the same question is a different kind of dependency, so on macOS the current figure is reported as not readable rather than invented, and the gate there is the peak. Saying which platform gets the stronger check is better than pretending both do.

The claim is not that memory never grows. A server that has just accepted a thousand connections has not yet grown the buffers those connections need, and the first segments climb on every machine in the fleet, by a factor of three on the smaller ones. The claim is that it stops, so the end of the run is judged against the halfway mark rather than against where it started, with a tenth or four megabytes of slack. A leak does not level off: the one this soak found put a third more memory on in the second half of every run it was in.

The size of the busiest timing wheel. One timer is armed per connection, so the slab should settle at the number of connections a reactor is holding, and a slab in the millions is a wheel that is keeping timers it has finished with. This gate exists because that is what the first four runs turned out to be measuring.

Descriptors. A socket is opened after teardown and the number the kernel hands back is compared with the same probe taken before the run. Descriptors are allocated lowest free first on both platforms, so a leak comes back high. This soak reconnects constantly, tens of thousands of times over an hour, so a leak of one descriptor per connection would be obvious and a leak of one in a thousand would still show.

Queues. The log ring and the reactor's connection table. A ring that ends the run with bytes still in it is a flush that stopped keeping up, and connections that survive the clients closing are a reap that stopped happening.

Latency drift. The run is cut into ten segments, every round trip lands in the segment that was current when it finished, and the last segment is compared against the first. Latency that is flat is the claim. Latency that climbs means something is accumulating, and the number says how fast.

The slow readers are deliberately left out of the latency numbers. Their round trip time is their own doing and including it would drown the signal in a delay the test itself is causing.

Drift is only judged on a run of at least a minute. Ten segments of three seconds each, which is what the version in the test suite is, are three hundred milliseconds apiece, and one scheduling hiccup on a shared CI runner moves the tail three buckets. That is what the gate said the first time it ran on a macOS runner, and a gate that fails on jitter gets an exception written into it and then it is not a gate. The short run proves the soak works and checks everything that does not need the hour.

The gate on drift is four times, which is wide on purpose. The histogram is powers of two microseconds, so two neighbouring buckets are already a factor of two apart and a run that crosses one boundary is noise rather than drift. The histogram lives in `molla.net.latency` and is shared with the reactor soak, so the gate means the same thing in both. It used to be two copies of the same code, which is two chances for the one number this rests on to mean something slightly different in each.

There is a correctness gate underneath all four. Every answer is checked against the status that kind of client should be getting, per kind rather than as a range, because a run where the oversized clients started getting 200 and the keep alive clients started getting 413 would have exactly the same totals as a clean one.

## What it found

A memory leak in the timing wheel, which is the one that needed the hour.

Every one of the first four runs, on all four machines, grew resident memory in every segment and never stopped. On the laptop it went from 127 MB to 2.3 GB, on the Linux boxes from 46 MB to about 350 MB, on the Windows machine from 60 MB to 1.4 GB. The shape was the same everywhere: a straight line, and the slope in proportion to how many connections the machine had managed to churn through.

The wheel cancelled lazily. A cancelled timer was marked dead and left in its slot, to be freed when that slot was next walked, and the module said in as many words why that was fine: nothing here creates enough dead timers for the cost of a doubly linked list to be worth paying. Every connection that closes cancels its idle timer, so there was never a version of this server for which that was true. A slot is not walked until time gets close to the deadline it holds, and this soak sets the idle timeout past the end of the run so that a slow reader pausing is never mistaken for an idle connection, which meant no slot was ever walked and the wheel ended the run holding one dead entry for every connection the run had ever made. Three and a half million per reactor on the laptop.

Slot lists are doubly linked now and cancel unlinks and releases in constant time. At a thousand connections for five minutes the busiest slab went from 257,388 entries to 115, which is the number of connections that reactor was holding. The soak gained a gate on it, so a future edit that goes back to lazy cancelling fails the run with the reason rather than with a memory number somebody has to bisect.

The reason this took an hour to find and not a minute is worth being clear about. The leak is one 32 byte slab entry per connection, which is invisible at any scale a test suite would run at, and it is bounded in principle: with the default minute long idle timeout a server holds a minute of closed connections rather than all of them. It is still a server that gets steadily larger the busier it is, and nothing short of a long run with real churn was ever going to show it.

Then a metrics bug, on the first run that got far enough to print a report.

`HttpProtocol._error` answers a 413, a 414 or a 431 from the parse path without going through `_write_default`, which is where the status accounting lived. So a run that sent a hundred thousand oversized bodies and got a hundred thousand 413s reported `molla_http_responses_4xx_total 0`. Those are the answers an operator most wants a graph of, and they were the ones that were never counted. The soak caught it because it checks the server's own count of 4xx answers against what the clients read back, which is the kind of cross check a unit test of the error path would not have.

The fix is one line, `_error` now calls `_account` the same way `_write_default` does.

It also found two things about the client rather than the server, both of which looked like server failures for a while.

Every socket in molla is non blocking from the moment it is created, so `connect` returns before the handshake finishes and the first write on a fresh socket fails until it does. The client's first version treated that as a send failure after two hundred tight retries, and reported a couple of hundred failures per run against a server that was fine. A socket that is not ready now says so and the caller comes back on the next pass.

The second is what happens when it still is not ready five seconds later, which on the fastest machine in the fleet happened to about one connection in two thousand. That is not a stuck socket, it is the accept backlog full: this client reconnects roughly ten thousand times a second and the kernel drops the SYN when the queue is behind, so the handshake waits out a retransmit or two. A connection nobody has accepted yet and one the kernel turned away with a refusal are the same thing seen from two sides, so they are counted together and reported as connects the backlog would not take. The run fails only if more attempts were turned away than got through, which is a listener that has stopped working rather than one that is busy. What still fails the run outright is a request that went out half written on an established socket, because that is nobody's backpressure.

## What it cannot see

It runs one process, so it says nothing about a server under a real client that lives somewhere else, and nothing about a network that drops or reorders anything. Loopback does neither.

It does not measure throughput and the numbers it prints should not be quoted as though it did. The client is one thread pacing itself with think times, so the request rate is what the client chose and not what the server could do.

The memory gate on macOS is a peak rather than a level, which is weaker than it sounds. A peak only ever goes up, so it can say that nothing grew after the halfway mark and it cannot say that anything was given back. Linux gets the stronger check and the two platforms agreed on the leak, which is the most a peak can be asked for.

Allocation counting is deliberately not in here. `AllocCounter` is explicitly not atomic and the soak runs several workers, so a count taken across them would be a number that looks precise and is not. The zero allocation claim has its own assertion in `molla allocs`, on one worker, where the count means something. See `docs/validation/allocations.md`.

## Where it runs

`.github/workflows/soak.yml`, at 03:17 UTC every night, on `ubuntu-24.04` and `macos-15`, with a ninety minute timeout and a `workflow_dispatch` trigger that takes a connection count and a duration. It raises the descriptor limit first, because one process holding a thousand connections is two thousand descriptors plus the listener, and the macOS default of 256 would fail on the first batch.

It is not in CI. An hour is longer than anybody will wait on a pull request, and the thing it is looking for needs the hour to show up at all.

The test suite runs a short version, sixteen connections for three seconds, which checks every gate the long run checks except the one about drift, on numbers too small to prove anything about an hour. That is the division: the suite says the soak works, the nightly says the server does. The suite also covers the client side framing on responses assembled by hand, because a `Wire` that finds the end of a response too early counts two answers where there was one, and a run like that passes while measuring a server that was falling apart.

## What was run
