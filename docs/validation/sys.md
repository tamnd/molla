# The OS boundary

`molla.sys` is the only module that calls `external_call` for OS services. Everything above it sees typed Mojo and a result that carries errno. This records what the wrappers cover, what was run against a real kernel, and the four things that went wrong on the way, because each of them is the kind of mistake that does not fail where you made it.

## What is covered

| Module | What it wraps |
| --- | --- |
| `result` | The one type every wrapper returns. A value, and the errno captured at the call site. |
| `errno` | The thread local slot, and the codes whose numbers differ between the two platforms. |
| `fd` | Close, and the non blocking flag. |
| `socket` | Listen, accept, connect, send, recv, `writev`, `shutdown`, socketpair, unix domain sockets. |
| `poll` | One `Poller` over kqueue and epoll, edge triggered on both. |
| `mmap` | Read only file mappings, `madvise`, `msync`. |
| `file` | Open, `pread`, `pwrite`, seek, truncate, `fsync`, `fstat`, unlink, rename, mkdir, rmdir, directory listing. |
| `thread` | pthreads, mutexes, condition variables, thread naming, CPU affinity, `sched_yield`, `nanosleep`. |
| `signal` | Dispositions, masks, signal sets, and a self pipe that turns a signal into a readable descriptor. |

## What was run

`tests/test_sys.mojo` exercises every wrapper against the real kernel. Not a mock anywhere: it creates files under `/tmp`, reads its own directory entries back, starts four threads that fight over a mutex, sends itself a SIGTERM, and binds a unix socket.

| Machine | Kernel | Arch | Checks |
| --- | --- | --- | --- |
| macbook, M4 | Darwin 24.6 | arm64 | 299 passed |
| server1, doge-01 | Linux 6.8, Ubuntu 24.04, glibc 2.39 | x86-64 | 299 passed |

The counting tests are the ones that matter. Four threads times two thousand increments under one mutex has to come to exactly eight thousand, and it did on both. A condition variable waiter has to see the value that was published before the broadcast, and it did. One signal has to produce exactly one byte on the channel, and it did.

## Four things that went wrong

**SIG_BLOCK is not the same number on both platforms.** macOS numbers block, unblock and setmask 1, 2 and 3, and Linux numbers them 0, 1 and 2. macOS caught it by returning EINVAL for 0. Linux would not have: passing macOS's SIG_BLOCK there means SIG_UNBLOCK, which quietly unblocks the signals you meant to block, and the failure arrives much later as a process that dies on the first SIGTERM instead of shutting down.

**`sigwait` cannot work in a Mojo 1.0 binary.** The design was to block SIGINT and SIGTERM everywhere and have one thread call `sigwait`, which turns a signal into an ordinary function return with the whole language available. Blocking SIGTERM on the main thread and then sending it to ourselves still killed the process. The Mojo runtime starts threads before `main` runs, those threads do not have the signal blocked, and a process directed signal goes to any thread that will take it. Nothing can set another thread's mask. The wrappers are kept, because they are the right primitive the day molla owns all of its threads, and the docstrings say plainly that they do not work yet.

What works is the self pipe. Mojo 1.0 has no globals of any kind, so a handler cannot be told where to write, and the only thing it can close over is a compile time constant. molla reserves descriptor number 900, duplicates the writing end of a socketpair onto it at startup, and the handler sends one byte carrying the signal number. The reading end is an ordinary descriptor the reactor polls next to its sockets, so shutdown is one more readable thing rather than a special case.

**A C symbol can only be declared once per binary.** `openat` was declared with three arguments in `mmap.mojo` and four in `file.mojo`, because the mode argument is variadic and only exists with O_CREAT. Both are correct on their own and together they fail to lower, with an error that points at a line in the standard library rather than at either file that caused it. `signal` failed the same way, taking a function pointer in one place and SIG_IGN in another, which is why installing a handler goes through `sigset` and only integer dispositions go through `signal`.

**A worker thread holding an address is not a use the compiler can see.** The mutex started out backed by a `List[UInt8]`, on the reasoning that a list's buffer stays put when the list moves. The buffer does stay put, but the list is still subject to ownership rules, and a list whose last visible use has passed gets released while four threads are still writing through the address it used to own. The symptom was `pthread_mutex_lock` returning EINVAL from workers and a counter that came up short. The sync objects own a `calloc` block now and free it themselves, so the address is stable because this module made it stable rather than because the optimiser happened not to notice.

## What is not covered

Linux on arm64. The `struct stat` field offsets differ per architecture as well as per platform, and the arm64 Linux numbers are written from the headers and have never been run. Every machine in the fleet is x86-64 or Apple silicon, so the first arm64 Linux box to run this suite is doing real work, and `sys.file` is where it will fail if the numbers are wrong.

`sendmsg` and control messages. molla passes no descriptors between processes, so `writev` covers the vectored case and `struct msghdr`, whose layout differs between the platforms, never has to be described.

Windows. There is no Windows path in `molla.sys` at all. The gaming PC runs this through WSL2, which is Linux.
