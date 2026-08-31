"""Structured logs, written by the worker that has the news and flushed by
somebody else.

The shape is one byte ring per worker, written only by that worker and read
only by the housekeeping thread. That is a single producer single consumer
queue, so the whole of the synchronisation is two positions that only ever go
up, and nothing on the request path ever waits for anything. A worker that
finds its ring full drops the record and counts the drop, which is the right
answer for a log: a server that stalls a request to write a line about the
request has its priorities backwards, and a drop counter says so out loud.

Records are written straight into the ring rather than into a scratch buffer
and copied. The worker is the only writer, and the reader only ever looks below
the published write position, so the bytes of a half built record are invisible
until the length header is filled in and the position is published. That is
what makes the hot path allocate nothing at all rather than nearly nothing.

Which brings up the acceptance criterion, which is that logging at a disabled
level costs no allocations. It costs less than that: one atomic load and a
comparison. `begin` returns an entry with `ok` false and every method on it
returns immediately, so the call site reads the same whether the level is on or
off and compiles to a branch when it is off.

The format is logfmt, not JSON. Both are structured and only one of them can be
read by a person at three in the morning with grep and no tools. Control bytes
in a value become dots, since a newline in a log line means the next line is a
forgery, and the message is quoted because it is the one field expected to have
spaces in it.

The flush allocates, once per batch, on the housekeeping thread. `print` wants
a `String` and there is no way to hand libc a span in Mojo 1.0. It is off the
request path, it happens at most a few hundred times a second, and pretending
otherwise in a comment would be worse than the allocation.
"""

from std.sys import stderr

from molla.io.buffer import Buffer
from molla.ops.config import (
    LEVEL_DEBUG,
    LEVEL_ERROR,
    LEVEL_INFO,
    LEVEL_OFF,
    LEVEL_WARN,
    level_name,
)
from molla.sys.atomic import AtomicBlock, AtomicRef
from molla.sys.clock import realtime_ns
from molla.sys.mem import allocate, as_ptr, release
from molla.sys.queue import round_up_pow2
from molla.sys.thread import Thread, set_thread_name, sleep_ms, spawn

comptime HEADER_BYTES = 4
"""How many bytes of length go in front of a record. Four, so a record can be
up to four gigabytes, which it will not be, and so the header never costs more
than the shortest useful line."""

comptime MIN_RING_BYTES = 4096
comptime MAX_RECORD_BYTES = 4096
"""A single record longer than this is truncated rather than dropped, because a
line that says most of what happened is worth more than a drop counter going
up. The truncation is visible: the record ends with ` truncated=true`."""

comptime _DOT: UInt8 = 46
comptime _QUOTE: UInt8 = 34
comptime _APOS: UInt8 = 39
comptime _SPACE: UInt8 = 32
comptime _EQUALS: UInt8 = 61
comptime _MINUS: UInt8 = 45
comptime _ZERO: UInt8 = 48
comptime _NEWLINE: UInt8 = 10
comptime _DEL: UInt8 = 127

comptime SLOTS_PER_STREAM = 3
comptime SLOT_WRITE = 0
comptime SLOT_READ = 1
comptime SLOT_DROPS = 2
comptime SLOT_LEVEL = 0
"""Slot zero of the block is the level, shared by every worker, and the per
stream slots start after it. The level is atomic rather than copied into each
`Logger` so that it can be turned up on a running process without restarting
it, which is the situation a log level exists for."""


struct Logger(Copyable, ImplicitlyCopyable, Movable):
    """One worker's end of the log. Addresses only, so it can be copied into a
    reactor and handed across a thread boundary.

    Owns nothing. The storage belongs to the `LogSink` that made it, which must
    outlive every worker, and does, because the sink is started before the
    threads and stopped after them.
    """

    var data: Int
    """The ring's bytes. Zero means the sink could not allocate, in which case
    every call here is a no op."""

    var mask: Int
    var capacity: Int
    var write: AtomicRef
    var read: AtomicRef
    var drops: AtomicRef
    var level: AtomicRef
    var stream: Int

    def __init__(out self):
        """A logger that discards. What a server started without a sink gets,
        so that nothing has to check whether logging exists before logging."""
        self.data = 0
        self.mask = 0
        self.capacity = 0
        self.write = AtomicRef(0)
        self.read = AtomicRef(0)
        self.drops = AtomicRef(0)
        self.level = AtomicRef(0)
        self.stream = 0

    def enabled(self, level: Int) -> Bool:
        """One atomic load and a comparison, which is the whole cost of a log
        line that is not going to be written."""
        if self.data == 0:
            return False
        return level >= self.level.load()

    def begin(self, level: Int) -> Entry:
        return Entry(self, level)

    def debug(self, message: StringSpan) -> Bool:
        var e = self.begin(LEVEL_DEBUG)
        e.message(message)
        return e.end()

    def info(self, message: StringSpan) -> Bool:
        var e = self.begin(LEVEL_INFO)
        e.message(message)
        return e.end()

    def warn(self, message: StringSpan) -> Bool:
        var e = self.begin(LEVEL_WARN)
        e.message(message)
        return e.end()

    def error(self, message: StringSpan) -> Bool:
        var e = self.begin(LEVEL_ERROR)
        e.message(message)
        return e.end()


struct Entry(Movable):
    """One record, being written directly into the ring.

    Everything is a no op when `ok` is false, which covers a disabled level, a
    missing sink and a full ring with the same branch. So a call site can write
    the fields it wants without asking first, and the fields it wrote cost
    nothing when nobody is listening.
    """

    var log: Logger
    var start: Int
    var at: Int
    var limit: Int
    """One past the last position that can be written without overtaking the
    reader. Read once at `begin`, because the reader only ever moves it
    forwards and a stale value is conservative."""

    var ok: Bool
    var truncated: Bool

    def __init__(out self, log: Logger, level: Int):
        self.log = log
        self.start = 0
        self.at = 0
        self.limit = 0
        self.ok = False
        self.truncated = False
        if not log.enabled(level):
            return
        var w = log.write.load()
        var r = log.read.load()
        self.start = w
        self.at = w + HEADER_BYTES
        self.limit = r + log.capacity
        if self.at + MIN_RING_BYTES // 8 > self.limit:
            # Not enough room even for a short record. Count it now rather than
            # writing half of one and finding out.
            _ = log.drops.add(1)
            return
        self.ok = True
        self._preamble(level)

    def _put(mut self, value: UInt8):
        if not self.ok:
            return
        if self.at >= self.limit or self.at - self.start >= MAX_RECORD_BYTES:
            self.truncated = True
            return
        as_ptr(self.log.data).unsafe_store(self.at & self.log.mask, value)
        self.at += 1

    def _raw(mut self, text: StringSpan):
        """Bytes as they are, for names and punctuation this module controls."""
        if not self.ok:
            return
        var p = text.unsafe_ptr()
        for i in range(text.byte_length()):
            self._put(p.unsafe_load(i))

    def _clean(mut self, text: StringSpan):
        """Bytes with anything that would break a line taken out.

        A control byte becomes a dot and a double quote becomes a single one. A
        log line is a record with a delimiter, and a value that can contain the
        delimiter is a value that can forge a record.
        """
        if not self.ok:
            return
        var p = text.unsafe_ptr()
        for i in range(text.byte_length()):
            var b = p.unsafe_load(i)
            if b < _SPACE or b == _DEL:
                self._put(_DOT)
            elif b == _QUOTE:
                self._put(_APOS)
            else:
                self._put(b)

    def _decimal(mut self, value: Int):
        if not self.ok:
            return
        if value == 0:
            self._put(_ZERO)
            return
        var n = value
        if n < 0:
            self._put(_MINUS)
            n = -n
        # Twenty digits covers every Int, and going through a fixed array keeps
        # this off the heap, which is the whole point of the module.
        var digits = InlineArray[UInt8, 20](fill=0)
        var count = 0
        while n > 0:
            digits[count] = _ZERO + UInt8(n % 10)
            n //= 10
            count += 1
        for i in range(count):
            self._put(digits[count - 1 - i])

    def _preamble(mut self, level: Int):
        self._raw("ts=")
        self._decimal(realtime_ns())
        self._raw(" level=")
        self._raw(level_name(level))
        self._raw(" worker=")
        self._decimal(self.log.stream)

    def message(mut self, text: StringSpan):
        """The one free text field, quoted because it is the one with spaces."""
        self._raw(' msg="')
        self._clean(text)
        self._put(_QUOTE)

    def field(mut self, name: StringSpan, value: StringSpan):
        self._put(_SPACE)
        self._raw(name)
        self._put(_EQUALS)
        self._clean(value)

    def field_int(mut self, name: StringSpan, value: Int):
        self._put(_SPACE)
        self._raw(name)
        self._put(_EQUALS)
        self._decimal(value)

    def field_bool(mut self, name: StringSpan, value: Bool):
        self.field(name, "true" if value else "false")

    def end(mut self) -> Bool:
        """Fill in the length and publish. False if nothing was written.

        The length goes in last on purpose. Until this store the reader has no
        idea the record exists, so a worker that is halfway through building
        one when the housekeeping thread runs is not a problem that has to be
        locked against.
        """
        if not self.ok:
            return False
        if self.truncated:
            self._raw(" truncated=true")
        var length = self.at - self.start - HEADER_BYTES
        var base = as_ptr(self.log.data)
        for i in range(HEADER_BYTES):
            base.unsafe_store(
                (self.start + i) & self.log.mask,
                UInt8((length >> (8 * i)) & 0xFF),
            )
        self.log.write.store(self.at)
        self.ok = False
        return True


struct LogSink(Movable):
    """Every worker's ring, the shared level, and the drain.

    Not copyable, because it owns the rings. The addresses survive a move,
    which is what lets a server hold one and hand `Logger` copies to reactors
    that are already in a list.
    """

    var block: AtomicBlock
    var rings: List[Int]
    var capacity: Int
    var streams: Int

    def __init__(out self, streams: Int, ring_bytes: Int, level: Int):
        self.streams = streams if streams > 0 else 1
        var wanted = ring_bytes if ring_bytes > MIN_RING_BYTES else (
            MIN_RING_BYTES
        )
        self.capacity = round_up_pow2(wanted)
        self.block = AtomicBlock(1 + self.streams * SLOTS_PER_STREAM)
        self.rings = List[Int](capacity=self.streams)
        for _ in range(self.streams):
            self.rings.append(allocate(self.capacity))
        self.block.slot(SLOT_LEVEL).store(level)

    def __deinit__(deinit self):
        for i in range(len(self.rings)):
            release(self.rings[i])

    def is_valid(self) -> Bool:
        if not self.block.is_valid():
            return False
        for i in range(len(self.rings)):
            if self.rings[i] == 0:
                return False
        return True

    def _slot(self, stream: Int, which: Int) -> AtomicRef:
        return self.block.slot(1 + stream * SLOTS_PER_STREAM + which)

    def level(self) -> Int:
        return self.block.slot(SLOT_LEVEL).load()

    def set_level(mut self, level: Int):
        """Change the level on a running process. One store, seen by every
        worker on its next log call."""
        self.block.slot(SLOT_LEVEL).store(level)

    def logger(self, stream: Int) -> Logger:
        """The end a worker holds. An out of range stream gets the discarding
        logger rather than somebody else's ring."""
        var out = Logger()
        if not self.is_valid() or stream < 0 or stream >= self.streams:
            return out^
        out.data = self.rings[stream]
        out.capacity = self.capacity
        out.mask = self.capacity - 1
        out.write = self._slot(stream, SLOT_WRITE)
        out.read = self._slot(stream, SLOT_READ)
        out.drops = self._slot(stream, SLOT_DROPS)
        out.level = self.block.slot(SLOT_LEVEL)
        out.stream = stream
        return out^

    def dropped(self) -> Int:
        var total = 0
        for i in range(self.streams):
            total += self._slot(i, SLOT_DROPS).load()
        return total

    def pending(self) -> Int:
        """Bytes written and not yet flushed, across every worker."""
        var total = 0
        for i in range(self.streams):
            total += (
                self._slot(i, SLOT_WRITE).load()
                - self._slot(i, SLOT_READ).load()
            )
        return total

    def drain(mut self, mut out: Buffer) -> Int:
        """Move every complete record into `out`, one per line.

        Round robin over the workers rather than draining one at a time, so a
        busy worker cannot starve a quiet one out of the log. Returns how many
        records were taken.
        """
        if not self.is_valid():
            return 0
        var taken = 0
        for i in range(self.streams):
            taken += self._drain_one(i, out)
        return taken

    def _drain_one(mut self, stream: Int, mut out: Buffer) -> Int:
        var write = self._slot(stream, SLOT_WRITE)
        var read = self._slot(stream, SLOT_READ)
        var data = self.rings[stream]
        var mask = self.capacity - 1
        var r = read.load()
        var w = write.load()
        var taken = 0
        var base = as_ptr(data)
        while w - r >= HEADER_BYTES:
            var length = 0
            for i in range(HEADER_BYTES):
                length |= Int(base.unsafe_load((r + i) & mask)) << (8 * i)
            if length <= 0 or w - r < HEADER_BYTES + length:
                break
            if not out.reserve(length + 1):
                break
            for i in range(length):
                _ = out.append_byte(
                    base.unsafe_load((r + HEADER_BYTES + i) & mask)
                )
            _ = out.append_byte(_NEWLINE)
            r += HEADER_BYTES + length
            taken += 1
        read.store(r)
        return taken


comptime FLUSH_INTERVAL_MS = 20
"""How often the housekeeping thread looks. Twenty milliseconds is under what
anybody notices tailing a log and is fifty wakeups a second on an idle server,
which is nothing next to what the pollers are already doing."""

comptime FLUSH_BUFFER_BYTES = 65536


struct LogPump(Movable):
    """The housekeeping thread, and the flag that stops it.

    Given to the thread by address, like everything else here. `stop` is an
    atomic rather than a Bool because it is written by the main thread and read
    by the pump, and a plain Bool shared between two threads is the bug this
    codebase keeps writing comments about.
    """

    var sink: Int
    """Address of the `LogSink`. Not owned."""

    var control: AtomicBlock
    var thread: Thread
    var running: Bool

    def __init__(out self, sink: Int):
        self.sink = sink
        self.control = AtomicBlock(2)
        self.thread = Thread()
        self.running = False

    def _stop_flag(self) -> AtomicRef:
        return self.control.slot(0)

    def flushed(self) -> Int:
        """Records written out since the pump started."""
        return self.control.slot(1).load()

    def start(mut self) raises:
        if self.running:
            return
        var rc = spawn(_pump, Int(Pointer(to=self)), self.thread)
        if not rc.is_ok():
            raise Error(rc.describe("could not start the log flush thread"))
        self.running = True

    def stop(mut self):
        """Ask the pump to finish and wait for it.

        It flushes once more after seeing the flag, which is the difference
        between a crash report that reaches the log and one that was still in a
        ring when the process exited.
        """
        if not self.running:
            return
        self._stop_flag().store(1)
        _ = self.thread.join()
        self.running = False


def _pump(arg: Int) abi("C") -> Int:
    var pump = Pointer[LogPump, MutAnyOrigin](unsafe_from_address=arg)
    _ = set_thread_name("molla-log")
    var sink = Pointer[LogSink, MutAnyOrigin](unsafe_from_address=pump[].sink)
    var out = Buffer(FLUSH_BUFFER_BYTES, 0)
    var stop = pump[].control.slot(0)
    var counted = pump[].control.slot(1)
    while True:
        var last = stop.load() != 0
        out.clear()
        var taken = sink[].drain(out)
        if taken > 0:
            _ = counted.add(taken)
            write_out(out.bytes())
        if last:
            return 0
        if taken == 0:
            _ = sleep_ms(FLUSH_INTERVAL_MS)


def write_out(data: Span[UInt8, MutAnyOrigin]):
    """Put a flush batch on stderr.

    stderr rather than stdout because a log is not output, and unbuffered so
    that a line written before a crash is a line somebody sees. The `String`
    here is the one allocation in the module and it is on the housekeeping
    thread, once per batch.
    """
    if len(data) == 0:
        return
    print(String(StringSlice(unsafe_from_utf8=data)), file=stderr, end="")
