# HTTP/1.1 and the throughput gate

The M0 spike for the request path. Issue #3 asks for a request line and header parser, a fixed body response, and a measured requests per second figure on the M4 for a trivial handler, with a gate of 5000. This records what was built, what it measures, and two things that turned out to matter far more than the parser did.

## What was built

`molla.http.request` is the parser. It is incremental and zero copy: it records offsets into the connection's read buffer as `Span` values rather than building strings, so a request costs no allocation at all. One `Request` is reused for every request on every connection, because parsing is synchronous and finishes before the next one starts. A request split across reads returns need more and is retried when more arrives, which costs nothing beyond reparsing the bytes already seen.

`molla.http.response` builds the response once at startup and every request is a `memcpy` of it. Two are built, one for keep alive and one with `Connection: close`, so choosing between them is a branch rather than a formatter. The only field that changes is `Date`, which is patched in place at a known offset when the second rolls over. That is deliberate. A server that drops the date header to win a benchmark is not measuring the thing anyone cares about, and refreshing it once per pass of the event loop is close to free.

`molla.http.server` is the echo loop from #2 with the parser in the middle. The event loop, the edge triggered draining, the pending output buffer and the reaping pass are unchanged, which was the point of shaping the echo server that way.

`molla http [port]` runs it. Every path returns the same body.

## What it rejects

The parser refuses the RFC 9112 request smuggling shapes rather than picking a winner: `Content-Length` together with `Transfer-Encoding`, a duplicate `Content-Length`, whitespace between a header name and its colon, and obsolete line folding. All four are 400. `Transfer-Encoding: chunked` is 501 because decoding it is M1 work and answering as though the body were empty would be the smuggling bug.

A well formed but unsupported HTTP version is 505 and not 400. RFC 9110 section 15.6.6 is explicit about this and the first version of the parser got it wrong, returning 400 for `HTTP/2.0` because it matched the literal `HTTP/1.` prefix. The test caught it. Shape and value are now checked separately: a malformed version token is 400, a well formed version we do not speak is 505.

## The two things that mattered

Neither was the parser.

The read buffer was being resized to 16 KB before each read and back down to the byte count afterwards. That is a malloc, a memset of 16 KB and a free on every read, and there are two reads per request because the loop reads until EAGAIN. It was most of the p99. The buffer is now grown to fit and never shrunk, with an explicit length field saying how much of it is real. On the M4 at one connection this took the run from 14077 to 28280 requests per second, the p90 from 7.62ms to 66us and the p99 from 40.31ms to 2.29ms. At 64 connections it took 11943 to 95102.

Nagle's algorithm was the other, and the order these were found in is worth recording because it nearly went the wrong way. Nagle was the first hypothesis for the tail, `TCP_NODELAY` went in, and the p99 got slightly worse. That looked like a clean refutation. It was not. The buffer churn was so much larger that it swamped the measurement completely. Once the churn was gone the same A/B showed `TCP_NODELAY` worth 39006 to 95102 requests per second at 64 connections, and the p99 at one connection dropping from 14.81ms to 2.29ms. A knob that measures as doing nothing may just be behind a bigger problem.

| M4, 64 connections | Requests/sec | p99 |
| --- | --- | --- |
| As first written | 11943 | 87.53ms |
| Buffer grown and never shrunk, Nagle on | 39006 | 75.03ms |
| Both fixed | 95102 | 19.72ms |

## The gate

D1 names the M4 as the reference machine and the gate is 5000 requests per second. It is cleared by a wide margin. The margin is easier to state than the number is, because no machine available to measure on was idle.

The M4 was carrying between 10 and 22 load average on 10 cores throughout, from unrelated compiler and benchmark work belonging to the person who owns the laptop. At c=32 with a fresh server per run, repeated measurements ranged from 43705 to 249896 requests per second. The low end of that range is the machine at load average 21 and it is still 8.7 times the gate. The high end is the machine at a quieter moment.

| M4, c=32, fresh server each run | Requests/sec | p50 | p99 |
| --- | --- | --- | --- |
| Quieter machine, five runs | 249896, 214350, 208084, 183295, 180161 | 107 to 134us | 236us to 11.46ms |
| Load average around 20, five runs | 86514, 60144, 56465, 50990, 43705 | 250 to 354us | 45.73 to 84.69ms |

At one connection, which is a pure round trip measurement with no queueing, the best observed was 71724 requests per second at a p50 of 12us and a p99 of 51us. Peak RSS stayed between 2.2 and 4.8 MB across everything above.

The honest summary is that every measurement taken clears the gate by at least 8x and the quiet ones clear it by roughly 50x, and that a single headline figure for the M4 would need a machine nobody was using.

## The fleet

Per D8 the server was built and run on all four Linux machines, which exercises the epoll path rather than kqueue. It works everywhere. The throughput numbers below are not benchmarks and should not be read as any.

| Machine | Threads | Backend | Peak req/sec | Best p99 | Peak RSS | Load average, 15 min | Other work on the box |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpc, i9-13900K on WSL2 | 32 | epoll | 260234 at c=128 | 212us at c=1 | 16012 kB | 3.61 | valkey and memtier_benchmark |
| server3, EPYC | 8 | epoll | 20808 at c=64 | 53.64ms at c=1 | 12440 kB | 11.65 | chrome |
| server1, EPYC | 4 | epoll | 15433 at c=128 | 61.29ms at c=8 | 11668 kB | 22.24 | three python3 jobs |
| server2, EPYC | 6 | epoll | 11010 at c=64 | 97.76ms at c=32 | 11848 kB | 8.13 | postgres |

gpc is the only one of the four with cores to spare, and it is the only one with a clean curve. It scales monotonically from 10509 at one connection to 260234 at 128 and its p99 never goes above 1.51ms. That is the shape a single threaded event loop should have, and it is the strongest evidence that the tails seen elsewhere are not coming from the server.

The three EPYC instances top out between 11k and 21k with p99 between 53ms and 183ms. Three things were checked before blaming the machines. CPU steal is zero on all of them. TCP retransmits during a 58416 request run on server3 numbered four, so the 100ms scale tails are not retransmission timeouts. And server2 and server3 sit 40 to 60 percent idle while producing a fraction of gpc's throughput, which means they are waiting rather than working.

What they are waiting on is partly the kernel. A `close(-1)`, which always traps and cannot be served from userspace, costs 79ns on gpc, 181ns on the M4, and 334ns, 433ns and 519ns on server3, server2 and server1. That is a four to six times tax on every syscall, and the request path takes several. The rest is that each box has between 4 and 8 threads which the load generator and the server are sharing with somebody else's postgres, chrome or python.

This matches what the socket soak in #2 already found, where server3 managed 7184 passes against gpc's 123764 on the same code. It is the same machines being slow in the same way, and it is a reason to keep D1's choice of the M4 as the reference machine.

Note that `getpid` is useless as a syscall probe on macOS, where it is served from the commpage and times at 1.16ns. The numbers above use `close(-1)` for that reason.

## What is not covered by the spike

There is no request body handling beyond counting `Content-Length` and skipping the bytes, no chunked decoding, no compression, no routing and no TLS. Each is M1 work. The handler is a constant, so none of the numbers above include anything an application would do.

There are still no timeouts and no connection limit, carried over from #2 unchanged. A client that opens a connection and says nothing is held forever, and a client that opens a request line one byte at a time is bounded only by `MAX_REQUEST_BYTES`, which produces a 431 at 64 KB. That is a slowloris with a ceiling rather than a defence.

`_index_of` is still a linear scan of the connection table and is still the wrong data structure above a few thousand connections. It did not show up in any measurement here because the concurrency levels tested are small, which is exactly why it is worth writing down rather than forgetting.

The p99 numbers on the M4 are not a property of the server and should not be quoted as one. On the only machine with spare cores the p99 is 1.51ms at 128 connections. Everywhere else it is the load generator and the server taking turns on the same cores as other people's work.

## What this says about D1

The request path is not where the difficulty is. The parser is a few hundred lines, it is zero copy, it holds up under the smuggling cases, and it never appeared in a profile. Both of the things that actually cost throughput were memory and socket behaviour, and both were fixed by writing less code rather than more.

The gate wanted 5000 requests per second. The worst measurement taken on a heavily loaded laptop was 43705 and the best was 249896. This is not close, and D1 should not be decided on throughput grounds.

## M1: the parser that has to hold up

Everything above is the spike. Issue #11 replaced the parser and built the four things that go around it, and the difference in intent is worth stating plainly. The spike parser had to be fast enough to measure the socket. This one has to be correct against a client that is trying to get something past it.

### Where the message boundary is decided

The single largest change is that `parse` now stops at the blank line and does not wait for the body. It records how the body is framed, in `body_kind` and `content_length`, and returns. `BodyReader` reads the body after that.

This is not a tidying. A parser that consumes the whole message cannot handle a body larger than the buffer it is parsing out of, which means either a cap on upload size set by the read buffer or a read buffer the size of the largest upload. An inference server is sent images and audio, so both of those are wrong. Splitting the two makes the memory cost of a body a property of `BodyReader` rather than of the connection.

### What the parser refuses now, and why each one

Every rule below is something two implementations can disagree about. That disagreement is the whole of request smuggling: a front end reads a byte stream as one request, molla reads it as two, and the second one is attributable to whoever's connection it lands in.

| Input | Answer | Why |
| --- | --- | --- |
| Bare LF anywhere, in the request line or a header | 400 | RFC 9112 says a recipient MAY treat it as a terminator. Both readings are legal, which is exactly what makes it useful to an attacker. molla takes neither. |
| Bare CR inside a field | 400 | Same argument, and a lone CR is also how a value gets read as the start of a new header. |
| `Content-Length` with `Transfer-Encoding` | 400 | The original smuggling pair. Preferring one is legal and is what the front end on the other side also did, differently. |
| Duplicate `Content-Length` | 400 | Even with equal values, because agreeing to read the second one is a rule the other hop may not have. |
| Duplicate `Transfer-Encoding` | 400 | Same. |
| `Transfer-Encoding` that is not exactly `chunked` | 501 | `gzip, chunked` is legal and molla does not decode it. Reading a compressed stream as chunk headers is the failure mode being avoided. |
| Space or tab before the colon | 400 | `Content-Length : 5` is a header to some parsers and not to others. |
| Obsolete line folding, a field starting with SP or HTAB | 400 | Deprecated by RFC 9112 and a reliable way to hide a second header inside a first. |
| HTTP/1.1 with zero or two `Host` headers | 400 | Zero is what a smuggled request looks like after a front end strips the outer one. Two is a request two hops route to different places. |
| A control character or NUL anywhere in a field | 400 | Header injection, and the NUL case specifically splits a C string parser from a length based one. |
| `Expect` that is not `100-continue` | 417 | The client is blocked waiting for an answer. Ignoring it hangs them until a timeout. |
| More than 8 KiB of request line, 64 KiB of headers, or 128 headers | 431 | Bounds, so a client cannot dribble headers forever. |
| A well formed version we do not speak | 505 | RFC 9110 section 15.6.6, carried over from the spike. |

The hostile corpus in `tests/test_http.mojo` is one case per row, and it asserts the status as well as the refusal. A 400 where a 501 belongs is a smaller bug than accepting the input, but it is still a wrong answer and the test says so.

### SIMD scanning

`molla.http.scan` finds delimiters sixteen bytes at a time. A scalar loop over a header block is a dependent branch per byte, which is the pattern a branch predictor is worst at, and header parsing is nothing but that loop.

Two Mojo details are worth recording because both cost time. `simd_width_of` lives in `std.sys.info` and not in `std.sys`. And `chunk == target` on two SIMD values returns a plain `Bool`, not a mask: the mask comes from `.eq()`, `.ne()`, `.lt()` and `.gt()`. Writing `chunk < space` fails at compile time with a message that says so, which is the good case. Writing `chunk == target` compiles and gives an answer about the whole vector, which is not.

The tail is a plain byte loop rather than a masked load. A masked load needs the mask built per call and buys back at most fifteen bytes of a scan that is already bounded by the header block, and being able to read the tail handling is worth more here than the instructions are.

### Bodies

`BodyReader` handles the three framings behind one call: hand it what arrived, it says how much it took and whether the body is finished. Bytes past the end of the body are left alone, which is what makes a pipelined request behind a body work at all.

Under a megabyte the body accumulates in memory. Over it, everything collected so far is written to a file and every byte after goes straight there, so peak memory for a body is bounded by the threshold and not by the body. The file is opened `O_EXCL` with mode 600 and retried on a collision, because a thousand connections on four threads can reach that line in the same millisecond with the same byte count behind them, and any name built from what the reader knows about itself is a name another reader can build too. The file is unlinked when the reader is destroyed. When the content addressed store lands in M3 the spill target becomes a store blob, which is the same write with a digest running over it, and nothing above `BodyReader` changes.

Chunked framing is a small state machine and a large attack surface. The size line is hex and nothing else: no leading plus, no whitespace, no `0x`, and at most sixteen digits so the accumulator cannot wrap. Chunk extensions are skipped and never interpreted. The trailer section is read and discarded rather than merged into the header set, because a trailer that lands in the headers is how a header a front end already checked gets replaced after the check.

### multipart

`molla.http.multipart` is a streaming parser fed decoded body bytes. It holds back the last `len(boundary) + 6` bytes of every feed and prepends them to the next, so a boundary landing across two reads is still found. Each part spills on the same threshold a whole body does.

It allocates, and that is deliberate. A part's headers become owned strings because a part header can be split across feeds and a span into a consumed buffer points at nothing. A request that reaches this code is already doing megabytes of I/O and a few short strings are not what makes it slow. The zero allocation claim is about the JSON request path, and this is not it.

Nested multipart, `multipart/byteranges`, and the base64 and quoted-printable content transfer encodings are all refused rather than read wrongly. The last one matters: reading a part as raw bytes when the sender said it was base64 hands a handler the wrong content with no sign that it did.

### The response path allocates nothing

`ResponseWriter` assembles a response into a buffer that belongs to the connection and is reused for the life of it. After the first few requests the buffer has grown to whatever that client's responses need and stops growing.

The test warms up eight responses and then measures two thousand more, and asserts the allocation counter is unchanged. Stating it that way rather than measuring from the first request is the honest version: the first few responses on a connection do allocate, once, and a number that hid that would depend on the initial capacity rather than on the code.

Getting there needed one specific thing. Writing a decimal number by building a list of digits and reversing it allocates once per number, and a response carries at least a status and a Content-Length. `write_int` computes the width and writes the digits into place instead.

`Date` is formatted at most once a second and held as bytes. HEAD is decided once, at `start`, and `body` then counts the bytes without writing them, so no call site has to remember. A HEAD that sends a body does not produce a wrong page, it desynchronises the connection, and the next request on it is read out of the middle of the body.

### On the reactor

`molla.http.protocol` implements the four calls in `Protocol` and gets the event loop, the sharded accept, the idle timer and the write ring from #9. Keep alive with a request cap, pipelining, `Expect: 100-continue` answered before the body is read, HEAD, and a framing error that closes the connection instead of reading what is behind it.

One thing about the reactor boundary is worth writing down because the first version got it wrong and the tests caught it. A protocol that wants to close after answering must call `conn.finish()` and return True, not return False. Returning False tells the reactor to stop servicing the connection, and it stops before the flush, so the response saying the connection is closing never leaves the ring and the client waits for a reply that is sitting in a buffer. `finish` says the same thing without cutting the write short: the reactor drains what is queued and closes after. `EchoProtocol` never closes, so nothing had exercised this path before.

The other is per connection state. The reactor holds one protocol object per worker thread, so anything that outlives a call is indexed by `conn.slot`, and `on_open` resets the entry rather than trusting what the last connection in that slot left behind.

Header spans are offsets into the read buffer, so they are valid only until it is consumed. A request with no body, which is every GET and HEAD, consumes nothing before the response is written and copies nothing at all. A request with a body has to consume as the body arrives or the read buffer grows to the size of the upload, so the method and target are copied into a per connection buffer first. That buffer is reused like the writer is, so it is a memcpy of a few dozen bytes.

### Where it was run

`tests/test_http.mojo` grew from 472 lines to roughly 1370 and the suite from 474 checks to 595. The whole suite was run on the M4 and on three Linux machines.

| Machine | Result |
| --- | --- |
| M4, macOS, kqueue | 595 passed, 0 failed |
| server1, EPYC, epoll | 596 passed, 0 failed |
| server2, EPYC, epoll | 596 passed, 0 failed |
| gpc, i9-13900K on WSL2, epoll | 595 passed, 1 failed |

Linux runs one more check than macOS because the reactor suite has a Linux only case for SO_REUSEPORT sharding. The gpc failure is `net.reactor backpressure, and the ring filled up behind it`, it is not in this issue's code, and it fails the same way on main at 474 passed and 1 failed. WSL2 does not honour the 8 kB SO_SNDBUF the test pins, so the kernel takes the whole write and the ring never fills. That is a test portability bug against WSL2 rather than a reactor bug, and it is filed as #87.

### What is still not covered

No routing, no compression, no TLS termination, no HTTP/2. Routing arrives with the API routes in M2, `molla.tls` is a client and has nothing to do with terminating TLS on the listener, and HTTP/2 is refused by the issue for reasons that have not changed: it buys nothing for a loopback inference server with one long stream per request and costs HPACK, flow control and a settings state machine.

Streaming responses are #12. `ResponseWriter` can already write a chunked body, so SSE and NDJSON are framing on top of what is here rather than another pass over the response path.

The connection cap is a count of requests and not a byte budget, so a client can hold a slot for a long time within it. The idle timeout is the reactor's and covers a silent connection, not a slow one.

`molla.http.server`, the M0 spike, now answers 501 to a request with a body rather than half reading one, because the parser no longer consumes the body. It is kept as the evidence behind the throughput numbers above and should not be used for anything else.
