# Changelog

Notable changes per release. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [0.2.0] - 2026-09-01

M1 is done. Everything Mojo 1.0 does not ship and a server cannot do without: syscall wrappers, buffers and arenas, a reactor, HTTP/1.1, SSE and NDJSON, a JSON scanner, client TLS, threads and queues and a shutdown that finishes what it started, config and logging and metrics, and two commands that assert the properties the rest of it claims.

The milestone was to write eight layers a normal server project gets for free, and the point of ending it here is that there is now something to build on that has been measured rather than asserted. The server answers HTTP on Linux, macOS and Windows under WSL2, it holds a thousand connections through an hour without leaking, and it allocates nothing on the request path. Those are three claims and each of them has a command that fails if it stops being true.

This release adds the last of those, the hour long soak, and fixes the leak it found. The timing wheel cancelled lazily on a written assumption that nothing would ever create enough dead timers to matter, and every connection that closes cancels its idle timer, so the assumption was never true for any server anybody would run. It took an hour of real churn to become visible: on the laptop the run went from 127 MB to 2.3 GB. The sign off round was four machines, an hour each, a thousand connections, seven hundred million answers, no 5xx and no wrong statuses, and on every one of them the busiest timing wheel ended holding exactly as many entries as its reactor had connection slots.

There is still no model, no routing beyond the handful of built in routes, and no server side TLS. Those are M2 and later. What is here is the floor.

### Added

- `molla httpsoak`, an hour long soak on the systems layer. Five kinds of client at once against a real server: keep alive, streaming, slow readers that fill the write ring and hold it full, abrupt disconnects that never read the answer, and oversized bodies that get a 413 and a close. It watches resident memory, descriptors, the log ring and the connection table, the size of the busiest timing wheel, and latency drift across ten segments of the run. Runs nightly on Linux and macOS through `.github/workflows/soak.yml`, and a short version runs in the test suite. See [docs/validation/soak.md](docs/validation/soak.md).
- `molla.net.latency`, the segmented latency histogram both soaks now share, so the drift gate means the same thing in each.

### Fixed

- The timing wheel cancelled lazily, marking a timer dead and leaving it in its slot to be freed when that slot was next walked. Every connection that closes cancels its idle timer, and a slot is not walked until time gets close to the deadline it holds, so a busy server accumulated a dead slab entry per connection for the length of its idle timeout and never gave the memory back. The hour long soak grew from 127 MB to 2.3 GB on macOS and from 60 MB to 1.4 GB on Windows before this. Slot lists are doubly linked now and cancel unlinks and releases in constant time.
- `HttpProtocol._error` answered a 413, a 414 or a 431 without going through the handler path, which is where the status accounting lived, so those responses were never counted. A server that refused a hundred thousand oversized bodies reported `molla_http_responses_4xx_total 0`. Found by the soak on its first complete run.

## [0.1.7] - 2026-09-01

A per request allocation on a server whose whole pitch is not having one, found by the assertion added to stop exactly this.

The design has always said the request path allocates nothing in steady state. Nothing checked it, and it had already stopped being true: the reactor rebuilt a `Connection` when it reused a slot, so every accept freed and re-allocated the read buffer and the write ring, and the read buffer then grew again on the first request that did not fit. Three allocations per connection, which against a client that opens a connection per request is a per request allocation.

`molla allocs` is the check that will not let that happen again. It runs a mixed load until one pass of it costs nothing and then requires the next pass to cost nothing too, with no tolerance, because a tolerance is a budget and a budget gets spent. It runs in CI on every commit on all three platforms and a smaller version runs in the test suite.

### Added

- `molla allocs`, which runs a mixed load against a real server until a pass of it costs nothing and fails if the pass after that allocates anything. The load is a pipelined batch of a GET, a HEAD, a 404, a Content-Length body, a chunked body, eight pipelined GETs and both streaming routes, so an allocation on the fifth kind of request is not invisible. Runs in CI on every commit on all three platforms, and a smaller version runs in the test suite. See [docs/validation/allocations.md](docs/validation/allocations.md).

### Fixed

- The reactor rebuilt a `Connection` when it reused a slot, which freed and re-allocated the read buffer and the write ring on every accept and then grew the read buffer again on the first request. That is three allocations per connection, and against a client that opens a connection per request it is a per request allocation. `Connection.reuse` now takes over the new socket and keeps the buffers at whatever size the traffic already paid for.

## [0.1.6] - 2026-09-01

The operations surface. Config, structured logging, Prometheus metrics, and three `/molla` routes, none of which makes molla faster and all of which is what makes molla debuggable by somebody who is not holding the source open.

Config carries where each value came from rather than only the value. The precedence is flag over environment over file over default, and `molla config get` prints both halves, because the value on its own is not the question anybody has at three in the morning.

Logging is a byte ring per worker, written by that worker and read by the housekeeping thread, which makes it single producer single consumer and means nothing on the request path waits for anything. Records are built straight in the ring and become visible only when the length header is written last. The criterion was no allocations at a disabled level, and the cost turned out to be one atomic load and a comparison.

Metrics are one set of counters per worker on its own cache line, summed only at scrape time, so a request never touches a line another core owns. Statuses are bucketed by class rather than per code, and durations are integer nanoseconds beside a count, with the HELP text admitting there is no histogram rather than exporting a number that looks like a quantile.

The three admin routes share the main port. A second listener is the safer answer and is also a second thing to configure, expose in a container, and forget, and everything they return is already printed on startup.

### Added

- `molla.ops.config`, settings with a precedence of flag over environment over file over default, and every setting carrying where its value came from.
- `molla config get [key]`, which prints the effective value and the source, because the value on its own is not the question anybody has.
- `molla.ops.log`, structured logfmt logging on a byte ring per worker, written by the worker and flushed by the housekeeping thread. A disabled level costs one atomic load and no allocations at all.
- `molla.ops.metrics`, Prometheus counters with one set per worker on its own cache line, summed only when somebody scrapes them. Durations are exported as integer nanoseconds.
- `GET /molla/version`, `/molla/health` and `/molla/metrics`, served on the main port and off unless a caller turns them on.
- `sys.clock.monotonic_ns` and `sys.clock.realtime_ns`, so durations and timestamps stop sharing a clock.
- `tests/test_ops.mojo`, including the check that a thousand log calls at a disabled level allocate exactly nothing.
- `docs/validation/ops.md`.

### Changed

- `molla drain` now runs with logging, metrics and the admin routes on, asks for all three routes before the load starts, and prints the counters after the drain.

## [0.1.5] - 2026-09-01

The concurrency layer, and a shutdown that finishes what it started. Also the TLS policy work, which decides per host whether a certificate has to check out.

One rule runs through all of it: anything two threads touch lives at an address rather than in a value. A Mojo value moves, so two threads reaching a counter through two copies of it are incrementing two counters, and that looks entirely correct in a single threaded test. Every shared thing here is allocated once, kept at a fixed address, and handed to a thread as the one integer a thread entry point gets.

Three pieces are not built the way the design said they would be, and each of them is a Mojo 1.0 fact rather than a preference. `Once` cannot be `pthread_once`, whose callback takes no argument, because an initialiser would have nowhere to leave its result except a global and there are no globals. Signals cannot arrive on signalfd, which needs the signal blocked in every thread of the process, because the runtime starts threads before `main` and offers no way to reach their masks. And the signal has to be armed before the server starts, which the first version got wrong in a way that only shows up when something signals faster than a person can press Ctrl-C.

Draining means closing the listeners and then closing each connection at the first moment it owes the client nothing, cutting and counting what is left at the deadline. Writing the test for it found a real bug: the poller is edge triggered, so a request that arrived between two drain passes has already spent its edge, and a connection about to ask for something looked exactly like one that was asleep. The difference is a request the client sent and never got an answer to.

The acceptance test is a command rather than a unit test, because every time in a hundred runs is not something a unit test reaches. A hundred runs of thirty two connections are clean on all four machines in the fleet, and the only failing check anywhere is the WSL2 backpressure one that was already failing.

There is still no routing and no config file. Both are M1 and neither is here.

### Added

- `molla.sys.atomic`, with `AtomicRef` for one atomic integer reached by address and `AtomicBlock` for several of them each alone on a cache line. Everything two threads share in molla lives at an address rather than in a value, because a Mojo value moves and two threads reaching a counter through two copies are incrementing two counters, which looks correct in a single threaded test.
- `molla.sys.queue`, a bounded ticket based MPSC queue with padded cells and an SPSC ring with no compare and exchange in it at all. The cell padding matters more here than in the usual description of the algorithm, since molla's queues run close to empty and a queue holding one item has the producer's cell and the consumer's cell next to each other.
- `Once` in `molla.sys.thread`, which is not `pthread_once`. The callback `pthread_once` takes has no argument, so an initialiser has nowhere to leave what it made except a global, and Mojo 1.0 has no globals. This one takes the same function and integer a spawned thread takes, and tells the caller that ran the body apart from the callers that waited.
- `molla.net.context`, a `ServerContext` holding every setting a server has, made by the caller and passed down. It is what makes a server testable in process, and it is the reason there is no configuration global to remove later.
- `molla.net.supervisor`, signals delivered as a readable descriptor through a self pipe, and `serve_until_signal`. Not signalfd, which needs the signal blocked in every thread and Mojo starts threads before `main` with no way to reach their masks, and not EVFILT_SIGNAL, which works but is macOS only and would leave the shutdown path a different mechanism on each platform. SIGTERM and SIGINT drain, SIGQUIT dumps every worker's state, thread, connection count and queue depth first.
- `Server.drain` and the `DrainReport` it returns. A draining reactor closes its listeners and then closes each connection at the first moment that connection owes the client nothing, with a deadline after which what is left is cut and counted. A shutdown that reports it dropped four connections after nine seconds is one you can act on, and a process that just exits is not.
- `molla drain [connections] [deadline_ms]` and `scripts/drain-loop.sh`, which is issue #15's acceptance test. Each run loads every connection with a pipelined batch, sends the process SIGTERM, and fails unless every client received every answer whole. A hundred runs of 32 connections is clean in 3s on the M4.
- `ServerContext.send_buffer_bytes`, which sets a small kernel send buffer on accepted sockets. Zero in production, where Linux sizes this per connection and does it better than a number written down once. The drain test sets it low so most of a connection's answers are still molla's problem when the signal lands, and it goes on the accepted socket rather than the listener because a listener does pass the option down and macOS then autotunes the inherited buffer back up.
- `tests/test_concurrency.mojo`, 80 checks that all use real threads. Four threads and twenty thousand increments each against exactly eighty thousand, eight threads racing one `Once` for exactly one winner, three producers against a queue too small to hold one producer's share so every one of them meets a full queue, and a check that each of the 1500 values comes out exactly once.
- `docs/validation/threading.md`, with the sharing rule the layer is built on, the three places the obvious version does not work in Mojo 1.0, and the measurements behind the drain test.
- `molla.tls.policy`, which decides whether a certificate has to check out, by host name. There is no global insecure flag and there will not be one: a pull is not one connection, since ghcr.io answers a blob request with a redirect to a signed URL on a host it names, so a process wide switch would turn verification off for a host chosen by the response. `molla pull --insecure` and `molla tls --insecure` cover the host on the command line and nothing else, and every connection that skips verification says so on stdout.
- `probe` in `molla.tls.client` and a `tls` line in `molla version`, saying which backend loaded and how high it can negotiate, or why there is none. TLS is dlopened, so a machine without it runs everything except HTTPS, and that is now a line of output rather than a promise in a document. It also puts the macOS TLS 1.2 cap on the screen of the machine it applies to.
- `MOLLA_SECURITY` and `MOLLA_COREFOUNDATION` overrides for the two macOS framework paths, matching `MOLLA_LIBSSL` on the other platform. Nobody moves Security.framework, so these exist to point the loader at something that does not load, which is the only way to test what molla does on a host with no TLS library.
- `tests/test_tls.mojo`, 17 checks over the policy and the probe, including the missing library case on every machine that runs the suite. The one that matters is negative: naming a registry insecure must leave the CDN it redirects to verified.

### Fixed

- A drain no longer treats a connection with an unread request on it as idle. The poller is edge triggered, so a request that arrived between the last pass and this one has already spent its edge and there is no second one coming, and a connection about to ask for something looked exactly like one that was asleep. The difference is a request the client sent and never got an answer to. The drain now reads every live connection once more before deciding it is finished, which costs one recv per idle connection per drain pass and is only paid during a shutdown.
- `molla version` prints the version it was built at. It said 0.1.2 for the 0.1.3 and 0.1.4 releases, because the release process bumps `pixi.toml` and the changelog and nothing was checking that `VERSION` in `build_info.mojo` came with them. `scripts/check-version.sh` now fails CI when the three disagree, which is cheaper than noticing it in a bug report six months from now.

## [0.1.4] - 2026-09-01

JSON, in both directions, at a bit over 2 GB/s on the M4. Two modes over one SIMD scanner: a pull loop for request bodies, which is nearly all of the traffic, and a small tree for config and manifests. Object keys keep the order they arrived in, which matters because a tool call's arguments came from a model and the order is part of what it said.

Numbers are the half of a JSON library that is either right or nearly right, and nearly right shows up months later as one customer whose floats come back different. There is no `strtod` here, so no locale and no copy to get a NUL terminated string, and the conversion is exact for every input including the ones written specifically to break converters.

Running the suite on four machines was worth more than the parser was. It passed on the M4 and failed six checks on both EPYC boxes, because Mojo destroys a local at its last use and the reader holds its document as an address, so the buffer was freed while the parse was still reading it. The M4 passed because the freed block still held the bytes. That is a bug nothing about x86 caused and one machine would have shipped.

There is still no routing. That is M2.

### Added

- `molla.json`, a JSON parser and serializer with two modes over one SIMD scanner. `scan.mojo` classifies bytes a vector at a time and finds the quote, the backslash and any raw control byte with one mask, so validating a string costs nothing extra. `reader.mojo` is streaming mode, `dom.mojo` is DOM mode, `serialize.mojo` writes into a buffer with no intermediate allocation and keeps object keys in the order they arrived.
- Exact integers and correctly rounded doubles with no `strtod` and no locale, in `number.mojo` and `decimal.mojo`. Three paths: an integer that fits in 64 bits stays an integer, a double with 53 bits of digits and a small exponent goes through Clinger's fast path, and anything else goes through an exact decimal expansion. Printing goes back through the same struct and gives the shortest form that reads back as the same double, with the two cutoffs JavaScript uses.
- `molla jsonbench [kb] [n]`, the acceptance test for #13 as a command. On the M4 a 100 kB chat body parses at 2283 MB/s with zero allocations, against a gate of 1 GB/s. DOM mode is 1920 MB/s and five allocations for a 1234 node document.
- `tests/test_json.mojo`, 154 checks over the scanner, both number directions, the reader, the DOM and the writer, including the inputs that break converters and a round trip over four thousand doubles built from random bit patterns.
- `keep` in `molla.sys.mem`, which counts as a use of a value and does nothing else. Mojo destroys a local at its last use, so handing a buffer's address to a reader is the last thing the compiler sees using it and the buffer is freed while the parse is still reading it. It passed on the M4, where the freed block still held the bytes, and failed six checks on x86_64 Linux, where the allocator hands the block straight to someone else.
- `docs/validation/json.md`, with the numbers, the three places the design departs from what issue #13 describes, and the lifetime bug the fleet run found.

## [0.1.3] - 2026-09-01

The request path. A parser, bodies, multipart, a serializer, and the two streaming writers a completion needs, all on the reactor from 0.1.2.

The parser stops at the blank line and the body is read separately after it, which sounds like a detail and is the reason a body larger than the read buffer is no longer a contradiction. Content-Length, chunked and multipart all go through one call that says how much it took and whether it is done, and over a megabyte the body spills to a file, so an upload costs bounded memory rather than its own size.

The streaming writers are the half of the request path an inference server actually spends its time in. An event is flushed on its own as soon as it exists and events are only combined into one chunk when the socket cannot take them. Writing them found a real hole in 0.1.2: the reactor only called `on_writable` when the poller reported a socket writable, which is correct for a request and a response and leaves a stream stopped after one ring's worth of bytes, because a client that sent one request and is waiting produces no read edge and a drained ring produces no write edge. Nothing before this produced without being asked, so nothing had caught it.

There is still no routing and no JSON. Both are M1 work and neither is here.

### Added

- `molla.http` gets the parser the request path will actually run on. `scan.mojo` finds delimiters sixteen bytes at a time, `request.mojo` parses a request line and a header block into zero copy spans and stops at the blank line, `body.mojo` reads the body after that, `serialize.mojo` writes responses without allocating, `multipart.mojo` streams multipart/form-data, and `protocol.mojo` puts all of it on the reactor from #10.
- Bodies are read separately from headers, so a body larger than the read buffer is no longer a contradiction. Content-Length, chunked and multipart all go through one call that says how much it took and whether it is done. Over a megabyte the body spills to a file opened `O_EXCL` with mode 600, so peak memory for an upload is bounded by the threshold rather than by the upload.
- A hostile input corpus in `tests/test_http.mojo`, one case per refusal, asserting the status and not just the rejection. Bare LF, bare CR, Content-Length with Transfer-Encoding, duplicate framing headers, a Transfer-Encoding that is not chunked, whitespace before a colon, obsolete line folding, zero or two Host headers, control characters in a field, an Expect we cannot answer, and the 8 KiB, 64 KiB and 128 header bounds.
- An assertion that two thousand responses allocate nothing, measured after a warmup so the number does not hide the first few responses growing the buffer.
- `MODE_600` in `molla.sys.file`.
- `molla.http.stream`, the SSE and NDJSON writers, over chunked transfer encoding. An event is flushed on its own as soon as it exists and events are only combined into one chunk when the socket cannot take them, which falls out of holding framed payload and chunk framed bytes in separate buffers. Past the staging limit a producer gets `STREAM_FULL` rather than a bigger buffer. Event names, ids and NDJSON records are refused if they contain a line break, since either one silently desynchronises the client rather than failing.
- The SSE heartbeat, as `heartbeat_due` and `heartbeat` taking the current time rather than reading a clock, so it fits a protocol trait with no tick in it and so the test is deterministic. NDJSON deliberately has none, because it has no comment syntax and a blank line is not portably ignored.
- `Connection.produce`, one bit saying the protocol has more to write on its own initiative, and `/stream/sse` and `/stream/ndjson` on the M1 protocol.
- `tests/test_stream.mojo`, 59 checks over framing, validation, backpressure, the heartbeat, a reader taking one byte at a time against pinned 8 kB socket buffers and a 2 kB output ring, and a client that hangs up mid stream.

### Changed

- The reactor calls `on_writable` for a connection whose protocol says it is producing, not only for one the poller reported writable, and counts bytes leaving the output ring as progress for the service loop. Without both, a streaming response stops after one ring's worth: the client is not going to send anything else, so there is no read edge, and once the ring drains there is no write edge either. Nothing before #12 produced without being asked, so nothing had exercised it.
- `write_decimal` is a free function in `molla.http.serialize` rather than a method, since the streaming writers need the same non allocating decimal into their own buffers.
- `molla.http.server`, the M0 spike, answers 501 to a request with a body instead of half reading one. The parser it calls no longer consumes bodies, and the spike is kept as the evidence behind the M0 throughput numbers rather than as something to build on. Benchmark anything other than a bare GET against `molla.http.protocol`.

## [0.1.2] - 2026-09-01

The event loop the request path will run on. One reactor per worker thread, each owning its poller, its connection table and its timers, with nothing shared on the request path.

There is still no request. This is the layer HTTP gets written against, and the reason it is tagged on its own is that it is the last piece that can be validated purely against a kernel, before anything above it can hide a bug in it.

Three things came out of building it. Servicing a connection cannot be a single pass over a readiness event, which is the shape every event loop tutorial has, because a connection that reads more than its output ring holds ends up with no read edge and no write edge coming and a response half written. It never happens under light load and happens every time under the load that makes it worst. The backpressure test that catches it was only honest on one platform until the socket buffers were pinned at both ends, since Linux starts them larger and grows them, so the amount of data needed to provoke a short write is different on every machine. And a test that waits for another thread has to wait in milliseconds rather than in loop iterations, because the two convert at a rate that depends on how loaded the machine is, and the budget runs out soonest on the machine that needed it to last longest.

A thousand connections held for an hour on the M4 and on server1, no mismatched payloads across 657195369 and 59256042 round trips, no descriptor growth, and a p99 flat to the bucket across all ten segments of both runs.

### Added

- `molla.net` gets a real reactor. One event loop per worker thread, each owning its poller, its connection table and its timers, with no shared state on the request path. `reactor.mojo` is the loop and the four call protocol trait everything above it will be written against. `conn.mojo` is one connection and the four states a non blocking socket can be in. `listener.mojo` decides how connections are spread, which is SO_REUSEPORT sharding on Linux and a round robin handoff on macOS because macOS gives the last binder every connection instead of balancing. `server.mojo` is N reactors on N threads behind one address, TCP or unix.
- `molla.net.wheel`, a four level timing wheel at 100 ms resolution. An idle connection costs one entry in a slab and no timer descriptor, which is what makes a thousand idle keep alive connections free rather than a thousand syscalls per second.
- `molla netsoak`, the acceptance test for issue #10. A thousand connections with mixed idle and active traffic, latency compared across ten segments of the run so drift shows up as a number, and descriptors and peak memory checked at the end.
- `tests/test_reactor.mojo`, 61 checks on macOS and 62 on Linux over the wheel against an explicit clock, the reactor stepped by hand, backpressure, idle timeouts, unix sockets and the threaded server.
- `set_keepalive`, `listen_tcp_shared` and `set_buffer_size` in `molla.sys.socket`, and `monotonic_ms` in `molla.sys.clock`. `set_buffer_size` exists because an accepted socket inherits its send and receive buffers from the listener, which is the only way to make a backpressure test hit the same wall on both platforms.

### Fixed

- The threaded server tests waited for a worker thread by counting empty non blocking reads rather than by watching a clock, so on a loaded runner the whole budget could burn through before the thread that owed the answer was scheduled at all. Both waits now run against a monotonic deadline and sleep between attempts instead of spinning on the core the worker needs.

## [0.1.1] - 2026-08-31

Two layers of the standard library Mojo 1.0 does not have. The OS boundary, and the memory the request path will live in. Nothing above them exists yet, so nothing here changes what molla can do, and both are the kind of thing that is much cheaper to get right before there is a server on top of it.

The sys layer is every OS call molla makes, in one module, returning one result type that carries errno from the call site. Files, threads, mutexes, condition variables and signals, all tested against a real kernel on three machines and both architectures.

The io layer is buffers, rings and arenas, with the growth policy of each written down next to the code rather than left to be inferred, and an allocation counter underneath so the zero allocation claim in M1 can be a number instead of a promise.

### Added

- `molla.sys` grows the rest of the OS boundary. `result.mojo` holds the one type every wrapper returns, carrying a value and the errno captured at the call site. `file.mojo` covers open, read, write, seek, truncate, sync, stat, unlink, rename and directory listing. `thread.mojo` covers pthreads, mutexes and condition variables, which is what Mojo 1.0 has no threading module for. `signal.mojo` covers dispositions, masks and a self pipe that turns a signal into a readable descriptor.
- `tests/test_sys.mojo`, which runs every wrapper against the real OS. FFI mistakes show up as memory corruption somewhere else entirely, so they are caught at the boundary or not at all.
- `access` behind `exists`, `writev` behind `write_vectored`, `socketpair`, unix domain sockets and `shutdown` in `molla.sys.socket`.
- `docs/validation/sys.md`, which records what the boundary covers, what ran green on which machine, and the four platform traps that cost a session each.
- `molla.io`, the memory layer the request path is built on. `buffer.mojo` is an owned growable buffer with a written down growth policy, doubling to 64 kB and then a fixed step. `ring.mojo` is the per connection output ring, so a short write costs two integer updates instead of a memmove, and it hands `writev` its one or two pieces directly. `arena.mojo` is a bump allocator with a per request lifetime, freed in constant time. `bytes.mojo` compares, searches, trims and parses spans without allocating.
- `molla.sys.mem`, which is where every allocation in molla goes, and the allocation counter that makes "this request allocated nothing" a number rather than a claim. Issue #17 is what the counter is for.
- `tests/test_io.mojo`, 116 checks over the growth policy, the ring wrap, the arena and the byte helpers.

### Changed

- `molla.sys.mmap` opens and closes through `molla.sys.file` instead of declaring `openat` a second time. Two declarations of one C symbol with different argument counts in the same build fail to lower, and the error points at the standard library rather than at either file that caused it.

## [0.1.0] - 2026-08-31

M0 is done. The question it existed to answer was whether Mojo 1.0 can hold a socket, parse HTTP fast enough, map a model file, call a kernel on a GPU and reach a TLS library, and the answer is yes on all five, with numbers rather than opinions behind each one. Two decisions were taken at the gate and both are recorded with the measurements next to them.

D1 holds. The network edge stays in Mojo, and the Rust fallback stays documented and untaken. On the M4 a trivial handler runs between 43705 and 249896 requests per second depending on how loaded the machine was, against a gate of 5000. A thousand concurrent connections held for sixty seconds on kqueue and on epoll with flat memory. The TLS binding pulls the same blob from ghcr.io through three different libraries on four machines. One condition, the multi threaded one, cannot be tested because Mojo 1.0 has no threading module, so it moves to the M1 gate rather than being rounded up.

D6 does not. `max/kernels` needs proprietary `max-core` at runtime, its CPU kernels included, so the promise of an optional MAX runtime was describing a seam that does not exist. molla accepts the dependency, which means running molla means installing a proprietary runtime, and the README says so instead of claiming otherwise.

This release is still a foundation. It does not serve a model. M2 is the first one that does.

### Added

- `docs/adr/`, for decisions taken at a gate against measurements, with `0001-network-edge-stays-in-mojo.md` and `0002-accept-max-core.md`, the two M0 gate records

### Changed

- D1 in `docs/design.md` records the M0 gate outcome. The network edge stays in Mojo. The multi threaded half of the third reversal condition moves to the M1 gate, because Mojo 1.0 has no threading module to test it with.
- D6 is rewritten. `max-core` is a required dependency at runtime rather than an optional backend, because `max/kernels` does not run without it and its CPU kernels do not either. Running molla now means installing a proprietary runtime, and the README names both proprietary packages and what each is needed for.
- D7 is marked load bearing. `max/kernels` has no quantized matmul that will launch on an Apple GPU below an M5, so every Apple GPU kernel is one molla writes, and D7's per target numerics tests are what keep the portability claim honest.

## [0.0.3] - 2026-08-31

The M0 kernel spike ran, and it changed what the README is allowed to claim. The TLS spike ran after it, and molla can now pull a blob from ghcr.io over HTTPS on macOS and Linux.

### Added

- `molla.tls`, client TLS over OpenSSL 3.x and 1.1.1 on Linux and Secure Transport on macOS, both loaded with dlopen so a machine without a TLS library still runs molla and only loses HTTPS
- `molla.http.client`, a GET only HTTPS client with redirect following and chunked bodies, and `molla.registry.ghcr`, enough of the OCI distribution protocol to fetch a blob and check its digest
- `molla tls <host>` prints the backend, protocol, cipher and certificate chain, and `molla pull <ref>` pulls a blob from ghcr.io and verifies it
- `molla.sys.dns` for `getaddrinfo`, `molla.sys.sha256`, `molla.sys.cstr`, and `dial` in `molla.sys.socket` for a blocking socket with timeouts
- `MOLLA_LIBSSL` and `MOLLA_LIBCRYPTO` to point at a specific OpenSSL, which is also how the 1.1 fallback gets tested on a machine that has 3.x
- `docs/validation/tls.md` with the results from four machines and three TLS libraries
- `spikes/qmatmul/`, the kernel spike for issue #5, with its own pixi manifest so its proprietary dependency stays out of the root build
- `docs/validation/kernels.md` with the licence audit, the numbers from six machines, and the three options for what molla does next
- Numerics tolerances for Q4_K matmul on CPU and on GPU, which D7 asked for and never gave

### Changed

- The README no longer claims there is no proprietary dependency in the stack, because there is, and it names it
- D6 in `docs/design.md` is marked under review, since `max/kernels` does not build or run without proprietary `max-core`, its CPU kernels included
- D7 in `docs/design.md` is marked achievable but not inherited, since one source did compile to Metal and sm_89 with byte identical output, but `max/kernels` is not organised that way

### Known issues

- Carried over from 0.0.2 unchanged: the compiler is proprietary, so releases are source only. Build with `pixi run build`.
- `max/kernels` has no quantized matmul that will launch on an Apple GPU below an M5. On an M4 it raises at launch. The spike wrote its own kernel to get a Metal number at all.
- What molla actually does about the licence finding is not decided here. That is issue #7.
- TLS on macOS caps at 1.2. Secure Transport has no TLS 1.3, and the framework that does is built on Objective-C blocks, which Mojo cannot emit. Linux gets 1.3 through OpenSSL.
- The HTTPS client is IPv4 only, opens a connection per request, and reads bodies into memory whole. None of that is suitable for pulling a model and M3 replaces it.

## [0.0.2] - 2026-08-31

Four of the seven M0 spikes are done. molla can now map a model file and read what is in it, though it still cannot read a tensor.

### Added

- `molla.sys.mmap`, a read only whole file memory map
- `molla.model.gguf`, a GGUF v2 and v3 metadata reader that walks the header, the key value block and the tensor directory in place without copying the file, and `molla gguf <path>` to dump one
- `docs/validation/gguf.md` with the comparison against `gguf-dump` on four models covering bert, llama, gemma3 and qwen2, and what the zero copy read is actually worth

### Known issues

- Carried over from 0.0.1 unchanged: the compiler is proprietary, so releases are source only. Build with `pixi run build`.
- Nothing reads a tensor. The GGUF reader records where each one is and what type it is, and stops there.
- Metadata arrays are measured and skipped rather than decoded, so there is no way to read a tokenizer vocabulary yet.

## [0.0.1] - 2026-08-31

First tagged release. Three of the seven M0 spikes are done: the toolchain is pinned across the fleet, sockets and the event loop work on epoll and kqueue, and HTTP/1.1 parse and respond clears the throughput gate. Nothing serves a model yet.

### Added

- Pixi workspace with the Mojo toolchain pinned to 1.0.0, locked for macOS arm64, Linux x86_64, and Linux arm64
- A `molla` binary with `version` and `help`, reporting the toolchain and detected host
- A test runner, since Mojo 1.0 has no `mojo test`
- CI builds and tests on all three platforms for real, and smoke tests the binary
- `docs/validation/toolchain.md` recording the pin, the machines validated so far, and what Mojo 1.0 actually looks like against the release notes
- `molla.sys`, the libc boundary: errno, descriptors, IPv4 TCP sockets, and one `Poller` over kqueue and epoll with read and write interest
- `molla.net.echo`, a non blocking edge triggered TCP echo server, and `molla echo` to run it
- `molla soak`, which holds a thousand connections for sixty seconds and checks for descriptor and memory leaks
- `docs/validation/sockets.md` with the soak results on all three platforms and what the spike says about D1
- `molla.http`, a zero copy HTTP/1.1 request parser and a prebuilt response with an in place `Date` field, and `molla http` to run the throughput spike
- `docs/validation/http.md` with the M0 throughput measurements, the fleet results, and the two allocation and socket problems that cost more than the parser did
- Design document, roadmap, and milestone plan
- CI with docs linting, workflow linting, CodeQL on workflow definitions, OpenSSF Scorecard, and dependency review
- Release pipeline with SBOM, build provenance attestation, and keyless signing
- `scripts/check-action-pins.sh`, run in CI, which fails if an action is pinned to an annotated tag object rather than a commit

### Changed

- The toolchain version lives only in `pixi.toml` now, rather than also in a CI environment variable

### Fixed

- Six actions were pinned to annotated tag object SHAs instead of commit SHAs, which made the OpenSSF Scorecard workflow fail on publish with an imposter commit error even though the scan itself succeeded

### Known issues

- The Mojo compiler we build with comes from Modular's conda channel under a proprietary license, so the build is not yet Apache-2.0 end to end even though the source is. See `docs/design.md`.
- Releases are source only for the same reason. A Mojo 1.0 binary links `libKGENCompilerRTShared` and two other runtime libraries that ship only as shared objects under `LicenseRef-Modular-Proprietary`, and the linker bakes RUNPATH to the pixi directory that built it, so a bare binary does not start anywhere else. Publishing a working tarball would mean redistributing Modular's runtime inside molla's own artifacts. Build with `pixi run build` until that changes.

molla answers HTTP requests as of the M0 spike, but every path returns the same fixed body. The first milestone that serves a model is M2.
