# 0001: the network edge stays in Mojo

Status: accepted, 2026-08-31. Decides the M0 gate, issue #7, on D1 only. The kernel half of M0 is a separate question and this record does not touch it.

## The decision

D1 holds. The HTTP server, the socket layer, the TLS bindings and the registry client stay in Mojo. The Rust fallback documented in D1 is not taken and stays documented.

One of the three reversal conditions cannot be fully tested on Mojo 1.0 and moves to the M1 gate, stated below rather than quietly dropped.

## What D1 said would reverse it

D1 names three conditions, any one of which moves the network edge to Rust:

1. Sustained HTTP throughput on the M4 under 5000 requests per second for a trivial handler.
2. The TLS FFI binding unstable across macOS and Linux.
3. Memory safety diagnostics making a non blocking multi threaded socket server impractical in Mojo 1.0.

The M0 spikes exist to test exactly these. Each is taken in turn.

## Condition 1: throughput

Not triggered, by a wide margin. Details in [validation/http.md](../validation/http.md).

The gate is 5000. On the M4 at 32 connections with a fresh server per run, five runs on a loaded machine gave 43705 to 86514 requests per second and five runs on a quieter one gave 180161 to 249896. The worst measurement anybody took is 8.7 times the gate. At one connection, which is a round trip with no queueing, the best was 71724 at a p50 of 12 microseconds. Peak RSS across all of it stayed between 2.2 and 4.8 MB.

There is no single headline number and there should not be one, because no machine available was idle. The honest statement is the range, and the bottom of the range clears the gate.

Two things about how that number was reached matter more than the number.

The first version of the parser ran at 11943 requests per second at 64 connections. It ended at 95102, and the parser was not touched in between. The whole difference was a read buffer being resized to 16 KB and back on every read, and Nagle. That is a warning about where to look for time in this codebase, not a reason to doubt the language.

The tails on four of the five fleet machines are the load generator and the server sharing cores with somebody else's postgres, chrome and python. gpc is the only machine with cores to spare, and it is the only one with the curve a single threaded event loop should have: monotonic from 10509 at one connection to 260234 at 128, p99 never above 1.51ms.

## Condition 2: TLS FFI stability

Not triggered. Details in [validation/tls.md](../validation/tls.md).

`molla pull` fetches the same 1744 byte blob from ghcr.io and verifies its SHA-256 on four machines through three TLS libraries: Secure Transport on macOS 15.8, OpenSSL 3.6.4 on three Ubuntu 24.04 servers, and OpenSSL 1.1.1f on one of them with the fallback forced. The 204 check suite passes on all four and in CI on macos-aarch64, linux-x86_64 and linux-aarch64.

Stability was the specific worry and this is the strongest part of the answer, because the macOS binding is not a simple one. Secure Transport does not own the socket. It calls back into molla for bytes, which means Mojo has to hand C a function pointer and be called from C on C's stack. That works. Both directions of the boundary hold, in a deprecated Apple API with a fiddly partial transfer contract, across three libraries and two operating systems.

dlopen rather than link was the right call and it is measurable rather than a claim. A binary with no usable libssl still starts and still runs every command that is not HTTPS, and the two that are print one line naming what to install. A linked binary would not have started at all.

One limit came out of this and it is not a reversal condition. Secure Transport has no TLS 1.3, and the Apple framework that does is built on Objective-C blocks, which Mojo cannot emit. macOS negotiates TLS 1.2 where Linux negotiates 1.3. Nothing on the public internet refuses 1.2 today. It is a floor that will rise and it is an M1 item.

## Condition 3: memory safety, and the part that cannot be tested

Not triggered on the evidence available, and the evidence available is incomplete. Details in [validation/sockets.md](../validation/sockets.md).

The tested half. A thousand concurrent connections held for sixty seconds on macOS with kqueue and on Linux with epoll, every byte echoed back in order, every connection reaped, and peak RSS after sixty seconds of traffic identical to peak RSS once the connections were up. Roughly ten kilobytes per live connection.

The borrow checker was not the problem, which was the specific worry. Without coroutines a connection has to be a struct mutated in place rather than a suspended stack, and `List[Connection]` with in place mutation through the index works. The one rule is not to remove entries while indices are live, which is why closing is marked during the event pass and compaction runs after it. That is a normal constraint, not an impractical one.

Three real problems turned up in M0 and none of them were about ownership. `read` and `write` cannot go through `external_call` because `std.ffi` already declares those names, so socket I/O uses `send` and `recv`. `fcntl` is variadic, and on arm64 macOS variadic arguments go on the stack while fixed ones go in registers, so calling it as an ordinary three argument function set nothing, returned success, and left the socket blocking until the first `recv` hung the process. `stack_allocation` does not zero, and `kevent` rejects a whole batch if the `timespec` it is handed has an out of range `tv_nsec`, so whether registering a descriptor worked depended on stack residue. All three are reasons to verify every libc signature and to prefer a test that provokes the real failure over one that asserts a return code. None is a reason to reverse D1.

The untested half, stated plainly. **Mojo 1.0 has no threading module.** There is nothing to spawn a thread with and nothing to synchronise one with. The condition asks about a "non blocking multi threaded socket server" and only the non blocking part has been tested. The soak runs client and server on one thread taking turns, which is a correctness and stability measurement rather than a concurrency one.

So the honest position is that condition 3 is half answered. The half that was testable passed. The half that was not is deferred to the M1 gate, where threading is in scope and where a multi threaded server either exists or D1 gets revisited with real evidence instead of an absence of it.

Deferring rather than reversing is the right call because reversing now would trade a measured result for a guess. Nothing in M1 is cheaper to change if the network edge is written in Rust first, and D1 already draws the module boundaries so the swap costs weeks rather than a rewrite. The cost of being wrong is bounded and known. The cost of moving to Rust on a condition nobody could test is a rewrite of everything M0 built, on no evidence.

## What this commits to

M1 builds the systems layer in Mojo: `sys`, `io`, `net`, `http`, `json`, `tls`, threading, config, logging and metrics.

Foreign code stays inside the D1 allowances. Two of the three are now exercised. libc through `molla.sys`, and platform TLS through `molla.tls`, both through the C ABI, with no C, C++, Rust or Go in the build.

## What this does not decide

The kernel and licensing question is open and this record does not touch it. [validation/kernels.md](../validation/kernels.md) found that `max/kernels` does not build or run without proprietary `max-core`, its CPU kernels included, and that the molla binary has linked a proprietary runtime library from `mojo-compiler` since the first commit. That is a D6 question and it gets its own record.

The two are separable, which is the reason for deciding one without the other. D1 is about whether Mojo can hold the network edge, and the answer does not depend on where kernels come from.

## What would still reverse this

At the M1 gate: no workable threading story, or a multi threaded server that the compiler makes impractical, or sustained throughput that collapses under real mixed traffic rather than a fixed body benchmark.

Later than that: a TLS floor rising to 1.3 with no way to reach Network.framework from Mojo would break HTTPS on macOS specifically, which is a smaller reversal than moving the whole edge and should be treated as one.
