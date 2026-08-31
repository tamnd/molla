# Threads, queues, and shutdown

Issue #15 is the systems layer under everything else: how molla starts a thread, how two threads share a number, how work moves between them without a lock, and how the process stops without cutting a client off mid answer. This records what was built, the three places where the obvious version does not work in Mojo 1.0 and what replaced it, and what was actually run.

## What is covered

| Module | What it is |
| --- | --- |
| `sys.atomic` | `AtomicRef`, one atomic integer by address, and `AtomicBlock`, several of them each alone on a cache line. |
| `sys.queue` | `MpscQueue`, a bounded ticket queue any thread may push to, and `SpscRing`, one writer and one reader with no compare and exchange at all. |
| `sys.thread` | Spawn and join, mutexes, condition variables, `Once`, thread naming, and reading a name back. |
| `net.context` | `ServerContext`, every setting a server has, made by the caller and passed down. |
| `net.reactor` | The drain: stop accepting, close each connection the moment it owes nothing, cut what is left at the deadline. |
| `net.server` | `Server.drain` and the `DrainReport` it returns, and `Server.dump` for SIGQUIT. |
| `net.supervisor` | Signals as a readable descriptor, and `serve_until_signal`. |
| `net.drain` | `molla drain`, the acceptance test, run a hundred times by `scripts/drain-loop.sh`. |
| `http.protocol` | A handler that raises turns into a 500 on that connection and nothing else. |

## Shared state lives at an address, not in a value

This is the rule the whole layer is built on, and it comes from one property of the language rather than from taste. A Mojo value moves. Two threads that reach an atomic through two copies of the same value are not sharing a counter, they are incrementing two counters, and both of them look right in a single threaded test.

So anything two threads touch is allocated from libc, kept at a fixed address, and handed to a thread as the one integer a thread entry function gets. `Mutex` already worked this way before #15 and `AtomicBlock` follows it. `AtomicRef` is that address plus the operations on it: copyable, owning nothing, safe to put in a struct that a worker is given a pointer to.

The trap next to it is that Mojo destroys a local at its last use, not at the end of its scope, and handing out an address is not a use the compiler can see. The first threaded probe written for this issue asked four threads for twenty thousand increments each and got 99662056985664, because the line that took a reference out of the block was the last mention of the block, the block was freed before any thread ran, and four threads spent their time incrementing memory the allocator had already handed to somebody else. `molla.sys.mem.keep` after the joins is the fix, it is documented on `AtomicBlock` itself, and every test in `tests/test_concurrency.mojo` ends with it.

Every operation is sequentially consistent. Acquire and release are what the queues actually need and would be cheaper on arm64, but Mojo 1.0's `std.atomic.Atomic` takes no ordering argument, and a fence stronger than necessary is a performance question rather than a correctness one.

## Padding is not decoration

Two counters eight bytes apart are on one cache line on every machine molla runs on, and two cores incrementing them take turns owning that line whether or not they ever touch the same counter. That is false sharing, it costs an order of magnitude, and it shows up as nothing at all except a number being lower than it should be.

`AtomicBlock` puts every counter alone on a 64 byte line and rounds the address it allocates up to a line boundary rather than trusting what calloc returned. The MPSC queue pads its cells too, which matters more here than in the usual description of the algorithm: molla's queues run close to empty, so a handoff queue with one item in it has the producer's cell and the consumer's cell adjacent, and packing cells sixteen bytes apart would put both threads on one line almost every time the queue is used.

## Three things that had to be built differently

**`Once` is not `pthread_once`.** The callback `pthread_once` takes has no argument, so the only place for an initialiser to leave its result is a global, and Mojo 1.0 has no globals at all. The initialiser would have nowhere to put what it made. So `Once` is three states in one atomic, takes the same `ThreadFunc` and integer argument a spawned thread takes, and returns 1 to the caller that ran the body and 0 to everybody else. Eight racing threads produce exactly one body run and exactly one winner.

**Signals arrive on a descriptor, and it is a self pipe rather than signalfd.** signalfd needs the signal blocked in every thread of the process, or the default action fires on whichever thread did not block it and the process dies before the descriptor is ever read. Mojo 1.0 starts runtime threads before `main` and gives no way to set their masks, so the precondition cannot be met. EVFILT_SIGNAL on macOS works without a mask but is macOS only, and a shutdown path that is a different mechanism on each platform is a shutdown path that is only ever tested on one of them. The self pipe is the same on both, and the property that made signalfd attractive, a signal arriving as a readable descriptor in a poller, is the property it has. The handler writes one byte and returns, which is all a handler is allowed to do.

**Arming the signal comes before starting the server.** The first version armed inside the wait. It passed by hand, because a person takes a second to press Ctrl-C, and it died with exit 143 the first time anything sent the signal quickly: SIGTERM arrived before the handler was installed and the default disposition ran. `SignalWatcher.arm` is now separate from `SignalWatcher.wait`, and `molla drain` arms before `Server.start`.

## What draining means

A reactor that is draining has closed its listeners, so nothing new arrives, and closes each connection at the first moment that connection owes the client nothing: no bytes in its output ring, nothing half read in its input, no stream in flight. An idle keep alive pool closes instantly. A connection with a request in flight is served to the end. When the deadline passes, whatever is left is closed and counted, because a shutdown that waits forever for one stuck client is not a shutdown.

Deciding it that way is what keeps the `Protocol` trait at four calls. A fifth call asking the protocol whether it is finished would put the same knowledge in two places, and the reactor already has it.

One thing had to be added to make that safe. The poller is edge triggered, so a request that arrived between the last pass and this one has already spent its edge and there is no second one coming. A connection that is about to ask for something looks exactly like one that is asleep, and the difference is a request the client sent and never got an answer to. So the drain reads every live connection once more before deciding it is finished. It costs one recv per idle connection per drain pass, which is a cost only paid during a shutdown.

`DrainReport` is the other half. A shutdown that says it dropped four connections after nine seconds is one you can do something about, and a process that just exits is not.

## The acceptance test

Issue #15 is done when a shutdown under load drains cleanly every time in a hundred runs, and every time is a number a unit test cannot reach. So it is a command, `molla drain [connections] [deadline_ms]`, which exits zero or one, and `scripts/drain-loop.sh` runs it a hundred times and stops at the first failure.

Each run starts a server on a fresh port, checks that `/boom` returns 500 and the server is still serving afterwards, opens N connections, sends each one a pipelined batch of 1024 requests, reads a single answer on each, sends itself SIGTERM, and then checks that every client received all 1024 answers with a body on every one of them.

Making the load real took measuring. The first version reported a clean drain in zero milliseconds, which is not a fast drain, it is a drain with nothing to do: the answers had all been written into socket buffers before the signal was sent. Two changes fixed it. `ServerContext.send_buffer_bytes` sets a small send buffer on every accepted socket, which holds a connection to eight kilobytes in the kernel and leaves the rest of its answers in the write ring, and the signal is now sent before the reader thread starts rather than after. With a reader running first, the workers finish the whole batch during the moment it takes the supervisor to wake up and the drain again finds every connection idle.

The buffer had to go on the accepted socket rather than on the listener, which is where the socket layer's own documentation says to put it. A listener does pass SO_SNDBUF down, and macOS then autotunes the inherited buffer back up over the life of the connection, so a test that asked for eight kilobytes got a hundred and sixty. Measured with a probe: with the option set on the listener, macOS loopback took 65328 bytes before it said no, and with it set after accept, exactly 8192.

With that in place, a run of 32 connections on the M4 has every connection holding around 31 kB of queued response and a batch of unread requests behind it when the signal lands, and delivers 31936 of its 32768 answers after the signal rather than before it.

## What was run

| Machine | Kernel | Arch | Suite | drain-loop 100 |
| --- | --- | --- | --- | --- |
| macbook, M4 | Darwin 24.6 | arm64 | 907 passed | 100 clean, 3s |
| server1, doge-01 | Linux 6.8, Ubuntu 24.04, glibc 2.39 | x86-64 | | |
| server2, doge-02 | Linux 6.8, Ubuntu 24.04, glibc 2.39 | x86-64 | | |
| gamingpc, WSL2 | Linux 6.6, Ubuntu 24.04 | x86-64 | | |

`tests/test_concurrency.mojo` is 80 of those checks and every one of them that can race does. Four threads and twenty thousand increments each is checked against exactly eighty thousand rather than against something plausible. The MPSC queue is filled by three producer threads with a capacity smaller than one producer's share, so every producer meets a full queue and the retry path is what is under test, and the consumer checks that each of the 1500 distinct values comes out exactly once. The SPSC ring is checked for order as well as for count, since order is the property it exists to have.

## What is not covered

Contention as a number. Everything here is checked for correctness under threads, and nothing measures what a contended push costs against an uncontended one. The queue is on the accept path rather than the request path, so the number that matters is not this one, and #12 is where throughput gets measured.

A real panic. `_write_default` catches an error raised by a handler and turns it into a 500 on that connection, which is what a handler that raises does. A genuine crash, a null dereference or an out of bounds write in a handler, still takes the process down, and nothing in Mojo 1.0 offers to catch that.

The queues at their limits over time. The MPSC queue's ticket sequence numbers are 64 bit and wrap after more pushes than the machine will ever do, so that is fine by construction rather than by test.

Thread naming on anything but Linux and macOS. Both spell the getter the same way and both are checked by reading the name back inside a spawned thread, which is the only way to know the setter did anything.
