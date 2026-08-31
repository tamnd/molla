"""Tests for the OS boundary.

These run against the real operating system rather than a fake one, on purpose.
An FFI wrapper has two ways to be wrong. It can call the wrong thing, which a
fake would catch, or it can have the right name and the wrong struct layout, the
wrong constant or the wrong calling convention, which only the kernel can tell
you about. The second kind is the one that corrupts memory somewhere else
entirely three modules away, so every check here provokes the real call.

Every temporary file lives under a directory named after the process, so two
runs on one machine cannot collide and a crashed run leaves something obvious
behind rather than something that looks like the next run's state.
"""

from std.memory import stack_allocation

from harness import Suite

from molla.sys.errno import EBADF, EEXIST, ENOENT, ENOTSUP, errno_name
from molla.sys.file import (
    DirEntry,
    FileInfo,
    MODE_755,
    O_CREAT,
    O_EXCL,
    O_RDWR,
    SEEK_END,
    close_fd,
    create,
    exists,
    fstat,
    fsync,
    ftruncate,
    lseek,
    mkdir,
    open_at,
    open_read,
    pread,
    pread_all,
    pwrite_all,
    read_dir,
    rename,
    rmdir,
    stat_path,
    unlink,
)
from molla.sys.result import SysResult, ok
from molla.sys.signal import (
    SIGINT,
    SIGTERM,
    SignalChannel,
    SignalSet,
    catch_signal,
    getpid,
    getppid,
    ignore_sigpipe,
    kill,
    open_signal_channel,
    signal_name,
)
from molla.sys.socket import (
    SHUT_WR,
    connect_unix,
    listen_unix,
    recv,
    send,
    shutdown,
    socket_pair,
    write_vectored,
)
from molla.sys.thread import (
    Condvar,
    CondvarRef,
    Mutex,
    MutexRef,
    Thread,
    ThreadFunc,
    cpu_count,
    sched_yield,
    set_affinity,
    set_thread_name,
    sleep_ms,
    spawn,
)


def _scratch() -> String:
    """A directory this process owns."""
    return "/tmp/molla-sys-" + String(getpid())


def _fill(buf: List[UInt8], mut into: List[UInt8]):
    for b in buf:
        into.append(b)


def run(mut suite: Suite) raises:
    _check_result(suite)
    _check_files(suite)
    _check_directories(suite)
    _check_threads(suite)
    _check_sync(suite)
    _check_signals(suite)
    _check_sockets(suite)


def _check_result(mut suite: Suite) raises:
    suite.group("sys.result")

    var good = ok(3)
    suite.check(good.is_ok(), "a success is ok")
    suite.check(not good.is_err(), "a success is not an error")
    suite.check(good.value == 3, "a success carries its value")

    # A close on a descriptor that was never open is the cheapest real failure
    # available, and it proves errno was captured rather than read later.
    var bad = close_fd(-1)
    suite.check(bad.is_err(), "closing a bad descriptor fails")
    suite.check(bad.is_errno(EBADF), "and it fails with EBADF")
    suite.check(
        bad.describe("close").endswith("EBADF"), "describe names the errno"
    )

    var raised = False
    try:
        _ = bad.unwrap("close")
    except:
        raised = True
    suite.check(raised, "unwrap raises on an error")
    suite.check(good.unwrap("noop") == 3, "unwrap returns the value otherwise")


def _check_files(mut suite: Suite) raises:
    suite.group("sys.file")

    var dir = _scratch()
    _ = mkdir(dir, MODE_755)
    var path = dir + "/data.bin"
    _ = unlink(path)

    var fd = create(path)
    suite.check(fd.is_ok(), "create opens a new file")

    var payload = List[UInt8]()
    for i in range(256):
        payload.append(UInt8(i))
    var wrote = pwrite_all(fd.value, payload.unsafe_ptr(), 256, 0)
    suite.check(
        wrote.is_ok() and wrote.value == 256, "pwrite_all writes it all"
    )
    suite.check(fsync(fd.value).is_ok(), "fsync succeeds on a real file")
    suite.check(close_fd(fd.value).is_ok(), "close succeeds")

    var info = FileInfo()
    var st = stat_path(path, info)
    suite.check(st.is_ok(), "stat finds the file")
    suite.check(info.size == 256, "stat reports the size we wrote")
    suite.check(info.is_file(), "stat says it is a regular file")
    suite.check(not info.is_dir(), "stat does not say it is a directory")
    suite.check(info.permissions() == 0o644, "created with mode 644")
    suite.check(info.mtime > 1700000000, "mtime is a plausible unix time")

    var back = open_read(path)
    suite.check(back.is_ok(), "reopen for reading")
    var buf = List[UInt8]()
    for _ in range(256):
        buf.append(0)
    var got = pread_all(back.value, buf.unsafe_ptr(), 256, 0)
    suite.check(got.is_ok() and got.value == 256, "pread_all reads it all")
    var same = True
    for i in range(256):
        if buf[i] != payload[i]:
            same = False
    suite.check(same, "every byte came back unchanged")

    var tail = pread(back.value, buf.unsafe_ptr(), 16, 250)
    suite.check(tail.value == 6, "a read past the end is short, not an error")
    var eof = pread(back.value, buf.unsafe_ptr(), 16, 256)
    suite.check(eof.is_ok() and eof.value == 0, "reading at the end gives zero")
    suite.check(
        lseek(back.value, 0, SEEK_END).value == 256, "lseek finds the length"
    )
    _ = close_fd(back.value)

    var writable = open_at(path, O_RDWR, 0)
    suite.check(ftruncate(writable.value, 10).is_ok(), "ftruncate shortens it")
    var after = FileInfo()
    _ = fstat(writable.value, after)
    suite.check(after.size == 10, "and the size follows")
    _ = close_fd(writable.value)

    var moved = dir + "/renamed.bin"
    _ = unlink(moved)
    suite.check(rename(path, moved).is_ok(), "rename moves the file")
    suite.check(not exists(path), "the old name is gone")
    suite.check(exists(moved), "the new name is there")

    suite.check(
        open_read(dir + "/absent").is_errno(ENOENT), "missing is ENOENT"
    )
    var exclusive = open_at(moved, O_RDWR | O_CREAT | O_EXCL, 0o644)
    suite.check(exclusive.is_errno(EEXIST), "O_EXCL on an existing file fails")

    suite.check(unlink(moved).is_ok(), "unlink removes it")
    suite.check(unlink(moved).is_errno(ENOENT), "unlinking it twice fails")


def _check_directories(mut suite: Suite) raises:
    suite.group("sys.file directories")

    var dir = _scratch() + "/listing"
    _ = rmdir(dir)
    suite.check(mkdir(dir, MODE_755).is_ok(), "mkdir makes a directory")
    suite.check(mkdir(dir, MODE_755).is_errno(EEXIST), "twice is EEXIST")

    var info = FileInfo()
    _ = stat_path(dir, info)
    suite.check(info.is_dir(), "stat says a directory is a directory")

    var names = List[String]()
    names.append("alpha")
    names.append("beta")
    names.append("gamma")
    for name in names:
        var fd = create(dir + "/" + name)
        _ = close_fd(fd.value)
    _ = mkdir(dir + "/nested", MODE_755)

    var entries = List[DirEntry]()
    var listed = read_dir(dir, entries)
    suite.check(listed.is_ok(), "read_dir succeeds")
    suite.check(listed.value == 4, "four entries, with dot and dotdot skipped")

    var found_files = 0
    var found_dirs = 0
    var found_alpha = False
    for e in entries:
        if e.is_file():
            found_files += 1
        if e.is_dir():
            found_dirs += 1
        if e.name == "alpha":
            found_alpha = True
    suite.check(found_files == 3, "three of them are files")
    suite.check(found_dirs == 1, "one of them is a directory")
    suite.check(found_alpha, "names survive the dirent offset")

    for name in names:
        _ = unlink(dir + "/" + name)
    _ = rmdir(dir + "/nested")
    suite.check(rmdir(dir).is_ok(), "rmdir removes an empty directory")
    suite.check(read_dir(dir, entries).is_errno(ENOENT), "and it is gone")


def _adder(arg: Int) abi("C") -> Int:
    """Add to a shared counter without a lock, from one thread only."""
    var p = Pointer[Int, MutAnyOrigin](unsafe_from_address=arg)
    for _ in range(1000):
        p[] = p[] + 1
    return 42


def _locked_adder(arg: Int) abi("C") -> Int:
    """Add to a shared counter under a mutex, from several threads.

    `arg` points at two integers: the address of the counter and the address of
    the mutex. One argument is all pthreads gives you, so anything wider than a
    pointer travels as a small array the caller keeps alive."""
    var args = Pointer[Int, MutAnyOrigin](unsafe_from_address=arg)
    var counter = Pointer[Int, MutAnyOrigin](
        unsafe_from_address=args.unsafe_load(0)
    )
    var lock = MutexRef(args.unsafe_load(1))
    for _ in range(2000):
        _ = lock.lock()
        counter[] = counter[] + 1
        _ = lock.unlock()
    return 0


def _waiter(arg: Int) abi("C") -> Int:
    """Wait for a predicate under a mutex, the way every condvar user must.

    `arg` points at the counter, the mutex and the condition variable. The loop
    around the wait is not decoration: a condition variable is allowed to wake
    a waiter that nobody signalled."""
    var args = Pointer[Int, MutAnyOrigin](unsafe_from_address=arg)
    var ready = Pointer[Int, MutAnyOrigin](
        unsafe_from_address=args.unsafe_load(0)
    )
    var lock = MutexRef(args.unsafe_load(1))
    var cond = CondvarRef(args.unsafe_load(2))
    _ = lock.lock()
    while ready[] == 0:
        _ = cond.wait(lock)
    var seen = ready[]
    _ = lock.unlock()
    return seen


def _check_threads(mut suite: Suite) raises:
    suite.group("sys.thread")

    suite.check(cpu_count() >= 1, "cpu_count returns at least one core")
    suite.check(set_thread_name("molla-test").is_ok(), "naming a thread works")
    suite.check(
        set_thread_name("a-name-far-longer-than-fifteen").is_ok(),
        "a long name is truncated rather than refused",
    )
    suite.check(sched_yield().is_ok(), "sched_yield works")
    suite.check(sleep_ms(1).is_ok(), "nanosleep works")

    var pinned = set_affinity(0)
    suite.check(
        pinned.is_ok() or pinned.is_errno(ENOTSUP),
        "affinity either pins or says the platform cannot",
    )

    # Stack storage rather than a `List`, and it matters. A worker holds the
    # address, which is not a use the compiler can see, so a heap buffer whose
    # owner looks dead is free to be released and handed to the next allocation
    # while four threads are still writing to it. This frame outlives every
    # thread it starts because it joins them all before returning.
    var counter = stack_allocation[1, Int]()
    counter.unsafe_store(0, 0)
    var t = Thread()
    var entry: ThreadFunc = _adder
    var started = spawn(entry, Int(counter), t)
    suite.check(started.is_ok(), "a thread starts")
    var joined = t.join()
    suite.check(joined.is_ok(), "and joins")
    suite.check(joined.value == 42, "its return value comes back")
    suite.check(
        counter.unsafe_load(0) == 1000, "it really ran and shared our memory"
    )
    suite.check(t.join().is_ok(), "joining twice is harmless")


def _check_sync(mut suite: Suite) raises:
    suite.group("sys.thread mutex and condvar")

    var lock = Mutex()
    suite.check(lock.init().is_ok(), "a mutex initialises")
    suite.check(lock.lock().is_ok(), "and locks")
    suite.check(lock.unlock().is_ok(), "and unlocks")
    suite.check(lock.try_lock().is_ok(), "try_lock takes a free mutex")
    suite.check(lock.unlock().is_ok(), "and gives it back")

    var counter = stack_allocation[1, Int]()
    counter.unsafe_store(0, 0)
    var args = stack_allocation[2, Int]()
    args.unsafe_store(0, Int(counter))
    args.unsafe_store(1, lock.raw())

    var workers = List[Thread]()
    var entry: ThreadFunc = _locked_adder
    for _ in range(4):
        var w = Thread()
        _ = spawn(entry, Int(args), w)
        workers.append(w^)
    for i in range(len(workers)):
        _ = workers[i].join()
    suite.check(
        counter.unsafe_load(0) == 8000,
        "four threads and a mutex lose no increments",
    )

    var cond = Condvar()
    suite.check(cond.init().is_ok(), "a condition variable initialises")

    var ready = stack_allocation[1, Int]()
    ready.unsafe_store(0, 0)
    var wait_args = stack_allocation[3, Int]()
    wait_args.unsafe_store(0, Int(ready))
    wait_args.unsafe_store(1, lock.raw())
    wait_args.unsafe_store(2, cond.raw())

    var sleeper = Thread()
    var waiting: ThreadFunc = _waiter
    _ = spawn(waiting, Int(wait_args), sleeper)
    _ = sleep_ms(20)
    _ = lock.lock()
    ready.unsafe_store(0, 9)
    _ = lock.unlock()
    _ = cond.broadcast()
    var woke = sleeper.join()
    suite.check(woke.is_ok(), "the waiter joins")
    suite.check(woke.value == 9, "it saw the value that was set before signal")

    suite.check(cond.destroy().is_ok(), "the condition variable destroys")
    suite.check(lock.destroy().is_ok(), "the mutex destroys")


def _check_signals(mut suite: Suite) raises:
    suite.group("sys.signal")

    suite.check(getpid() > 0, "getpid returns something")
    suite.check(getppid() > 0, "getppid returns something")
    suite.check(ignore_sigpipe().is_ok(), "SIGPIPE can be ignored")
    suite.check(kill(getpid(), 0).is_ok(), "signal zero probes a live process")
    suite.check(signal_name(SIGTERM) == "SIGTERM", "signals have names")

    var set = SignalSet()
    _ = set.clear()
    _ = set.add(SIGTERM)
    suite.check(set.contains(SIGTERM), "a signal set holds what was added")
    suite.check(not set.contains(SIGINT), "and not what was not")
    _ = set.remove(SIGTERM)
    suite.check(not set.contains(SIGTERM), "removing works")

    var channel = SignalChannel()
    var opened = open_signal_channel(channel)
    suite.check(opened.is_ok(), "the signal channel reserves its descriptor")
    suite.check(catch_signal(SIGTERM).is_ok(), "SIGTERM is routed to it")

    var empty = channel.take()
    suite.check(empty.is_err(), "nothing is waiting before a signal")

    suite.check(kill(getpid(), SIGTERM).is_ok(), "send ourselves a SIGTERM")
    _ = sleep_ms(20)
    var got = channel.take()
    suite.check(got.is_ok(), "the handler wrote to the channel")
    suite.check(got.value == SIGTERM, "and it says which signal arrived")
    suite.check(channel.take().is_err(), "one signal produces exactly one byte")

    channel.close()
    var again = SignalChannel()
    suite.check(
        open_signal_channel(again).is_ok(),
        "the reserved descriptor is free again after close",
    )
    again.close()


def _check_sockets(mut suite: Suite) raises:
    suite.group("sys.socket")

    var ends = List[Int]()
    suite.check(socket_pair(ends).is_ok(), "socketpair makes a connected pair")
    suite.check(len(ends) == 2, "and hands back two descriptors")

    var message = List[UInt8]()
    for b in "hello".as_bytes():
        message.append(b)
    suite.check(send(ends[1], message.unsafe_ptr(), 5) == 5, "five bytes go in")

    var buf = List[UInt8]()
    for _ in range(16):
        buf.append(0)
    suite.check(recv(ends[0], buf.unsafe_ptr(), 16) == 5, "five bytes come out")
    suite.check(
        buf[0] == UInt8(ord("h")) and buf[4] == UInt8(ord("o")),
        "and they match",
    )

    var head = List[UInt8]()
    for b in "HTTP/1.1 200 OK\r\n\r\n".as_bytes():
        head.append(b)
    var body = List[UInt8]()
    for b in "body".as_bytes():
        body.append(b)
    var bases = List[Int]()
    bases.append(Int(head.unsafe_ptr()))
    bases.append(Int(body.unsafe_ptr()))
    var lengths = List[Int]()
    lengths.append(len(head))
    lengths.append(len(body))
    var vectored = write_vectored(ends[1], bases, lengths)
    suite.check(vectored.is_ok(), "a vectored write succeeds")
    suite.check(
        vectored.value == len(head) + len(body),
        "and reports both pieces as written",
    )

    var joined = List[UInt8]()
    for _ in range(64):
        joined.append(0)
    var read_back = recv(ends[0], joined.unsafe_ptr(), 64)
    suite.check(
        read_back == len(head) + len(body),
        "the reader sees one stream, not two writes",
    )
    suite.check(
        joined[19] == UInt8(ord("b")),
        "the body follows the headers in order",
    )

    suite.check(shutdown(ends[1], SHUT_WR) == 0, "half close succeeds")
    suite.check(
        recv(ends[0], joined.unsafe_ptr(), 64) == 0,
        "and the reader sees end of stream",
    )
    _ = close_fd(ends[0])
    _ = close_fd(ends[1])

    var dir = _scratch()
    _ = mkdir(dir, MODE_755)
    var path = dir + "/sock"
    _ = unlink(path)
    var listener = listen_unix(path, 8)
    suite.check(listener > 0, "a unix socket binds to a path")
    suite.check(exists(path), "and the path exists after binding")
    var client = connect_unix(path)
    suite.check(client > 0, "a client connects to it")
    _ = close_fd(client)
    _ = close_fd(listener)
    _ = unlink(path)
    _ = rmdir(dir)
