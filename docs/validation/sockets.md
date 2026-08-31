# Sockets and the event loop

The M0 spike for the network edge. It asks one question: can a non blocking, edge triggered socket server be written in Mojo 1.0, given no async, no `std.net`, and no threading module. The answer is yes, and this records what it cost and what it proved.

## What was built

`molla.sys` is the libc boundary and the only place in the network path that calls `external_call`. `errno` reads the thread local slot through `__error` on macOS and `__errno_location` on Linux and carries the per platform numbers for the codes the loop branches on. `fd` has close and the non blocking flag. `socket` has listen, accept, connect, send, recv, and a `sockaddr_in` writer. `poll` is one `Poller` type over kqueue and epoll with read and write interest, edge triggered on both.

`molla.net.echo` is the loop, `molla echo` runs it, and `molla soak` is the acceptance test.

Nothing above `molla.sys` knows which polling backend it got. That boundary is the point of the exercise, because the HTTP server in M1 is this loop with a parser in the middle.

## The soak

Issue #2 asks for a thousand concurrent connections held for sixty seconds on macOS and Linux with no descriptor or memory leaks. `molla soak 1000 60` runs exactly that and checks itself.

| Machine | Backend | Passes | Round trips | Bytes echoed | Peak RSS | Probe fd |
| --- | --- | --- | --- | --- | --- | --- |
| macbook, M4 | kqueue | 54955 | 3517120 | 28144448 | 10112 kB, unchanged | 3 to 3 |
| server3, EPYC | epoll | 7184 | 459776 | 3685696 | 9856 kB, unchanged | 3 to 3 |
| gpc, i9-13900K on WSL2 | epoll | 123764 | 7920840 | 63374144 | 13448 kB, unchanged | 3 to 3 |

All three held all thousand connections for the full minute, echoed every byte back in order, reaped every connection when the clients closed, and finished with peak memory exactly where it started. Ten megabytes for a thousand live connections is about ten kilobytes each, which is the read buffer and the pending output list and not much else.

Two caveats on how this was measured, because the numbers are weaker than they look at first glance.

Both sides run on one thread. There is no threading module in Mojo 1.0, so a pass is: let every client write, turn the server loop once, let every client read. Real clients would not be politely taking turns with the server, so this is a correctness and stability measurement rather than a performance one. The 5000 requests per second gate is issue #3 and needs a separate client.

Memory is `getrusage`, which reports a high water mark rather than current usage. It cannot show that memory was returned. What it does show is the useful direction: peak memory after sixty seconds of traffic is identical to peak memory once the connections were up, so nothing grew while the loop ran. `ru_maxrss` is in bytes on macOS and kilobytes on Linux, same struct field, no flag to ask which.

The pass counts differ by seventeen times between server3 and gpc. Do not read that as a benchmark. Each pass is a couple of thousand syscalls, so this measures syscall cost, and server3 is a small virtualised instance where syscalls are expensive. It is worth knowing before anyone picks a machine to measure the M1 throughput gate on. D1 names the M4 for that and this is a reason to keep it there.

## Descriptor leaks

The check is to open a socket after teardown and look at the number the kernel hands back. Both kernels allocate the lowest free descriptor, so a run that leaked even one would come back high. All three machines went from 3 to 3.

This is cheap and it is not complete. It catches leaks in the paths the soak exercises, which is accept, normal close, and peer close. It would not catch a leak on an error path that the soak never takes.

## What is not covered

The aarch64 Linux `epoll_event` layout is not exercised by any machine in the fleet. `epoll_event` is packed to 12 bytes on x86_64 and 16 bytes everywhere else, because glibc only applies the packed attribute on x86_64 for compatibility with the 32 bit ABI. Every Linux machine we own is x86_64, so the 16 byte path is only run by the `linux-aarch64` job in CI. That job builds and runs the unit suite, which does open real sockets and does drive the poller, so the layout is checked. It does not run the soak. If molla ever grows an aarch64 Linux server, the soak should run there first.

There are no timeouts. A connection that opens and says nothing is held forever. There is no limit on how many connections may be accepted, no TLS, and no threads. Each of those is its own M1 issue and none of them belong in a spike.

`EchoServer` finds a connection by scanning the table, which is fine at a thousand and is not fine at the scale M1 targets. The fix is a descriptor indexed slot table with a free list, which also removes the compaction pass. It was left alone here because the spike is about whether the shape works, and swapping the lookup does not change the shape.

## What this says about D1

The borrow checker was not the problem. The specific worry was that connection state machines would be impractical, since without coroutines a connection has to be a struct that is mutated in place rather than a suspended stack. In practice `List[Connection]` with in place mutation through the index works, and the one rule that has to be respected is not to remove entries while indices are live, which is why closing is marked during the event pass and compaction happens after it.

Three real problems turned up and none of them were about ownership.

`read` and `write` cannot be called through `external_call` at all. `std.ffi` already declares symbols with those names, and a second declaration with a different signature fails to lower with a conflicting signature error out of the standard library rather than out of our code. Socket I/O uses `send` and `recv`, which is the better call for a socket anyway. This is worth remembering because the GGUF reader in M2 will want `pread`.

`fcntl` is variadic, and on arm64 macOS that is not a detail. Variadic arguments are passed on the stack there while fixed arguments go in registers, so calling `fcntl` as an ordinary three argument function put the flags in a register libc never reads and libc read whatever happened to be on the stack. It returned success and set nothing. The socket stayed blocking, and the first `recv` that ran out of data hung the process with no error anywhere. `external_call` takes `num_fixed_args` for exactly this. `set_nonblocking` now reads the flag back rather than trusting the return code, because a call that reports success while doing nothing is the failure mode worth a regression test.

`stack_allocation` does not zero. `kevent` rejects an entire batch with EINVAL if the `timespec` it is handed has a `tv_nsec` outside the legal range, and with zero events requested that timeout is never waited on, so whether registering a descriptor worked depended on what was left on the stack. It worked in an early probe and failed once the same code was called from a struct with a different frame layout.

None of these are reasons to reverse D1. They are reasons to treat every libc signature as something to check rather than assume, and to prefer a test that provokes the real failure over one that asserts the return code.

## Reproducing

```console
pixi run build
./build/molla soak 1000 60
```

On Linux raise the descriptor limit first, since the default soft limit of 1024 is not enough for a thousand connections on both sides:

```console
ulimit -n 8192
```

`molla echo` runs the server on its own if you want to point something else at it. It binds loopback only, per D9.
