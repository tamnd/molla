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

## What is not covered

There is no request body handling beyond counting `Content-Length` and skipping the bytes, no chunked decoding, no compression, no routing and no TLS. Each is M1 work. The handler is a constant, so none of the numbers above include anything an application would do.

There are still no timeouts and no connection limit, carried over from #2 unchanged. A client that opens a connection and says nothing is held forever, and a client that opens a request line one byte at a time is bounded only by `MAX_REQUEST_BYTES`, which produces a 431 at 64 KB. That is a slowloris with a ceiling rather than a defence.

`_index_of` is still a linear scan of the connection table and is still the wrong data structure above a few thousand connections. It did not show up in any measurement here because the concurrency levels tested are small, which is exactly why it is worth writing down rather than forgetting.

The p99 numbers on the M4 are not a property of the server and should not be quoted as one. On the only machine with spare cores the p99 is 1.51ms at 128 connections. Everywhere else it is the load generator and the server taking turns on the same cores as other people's work.

## What this says about D1

The request path is not where the difficulty is. The parser is a few hundred lines, it is zero copy, it holds up under the smuggling cases, and it never appeared in a profile. Both of the things that actually cost throughput were memory and socket behaviour, and both were fixed by writing less code rather than more.

The gate wanted 5000 requests per second. The worst measurement taken on a heavily loaded laptop was 43705 and the best was 249896. This is not close, and D1 should not be decided on throughput grounds.
