"""Reading a request body without deciding in advance that it fits in memory.

A body is either a known number of bytes, a chunked stream whose length is not
known until it ends, or absent. `BodyReader` handles all three behind one call:
hand it whatever arrived, it tells you how much of that it took and whether the
body is finished.

The part that matters is where the bytes go. Under `spill_threshold` they
accumulate in a buffer, because a body that fits in a page and gets read once
should not cost a file. Over it, everything collected so far is written to a
file and every byte after that goes straight there, so peak memory for a body
is bounded by the threshold no matter how large the body is. An inference
server gets sent whole images and audio files, and the difference between
bounded and unbounded here is the difference between a slow request and an
out of memory kill.

The spill file is a temporary that is unlinked when the reader is destroyed.
When the content addressed store lands in M3 the spill target becomes a store
blob instead, which is the same write with a digest running over it and a name
at the end, and this is the seam that changes. Nothing above `BodyReader` needs
to know which it was.

## On chunked

Chunked framing is a small state machine and a large attack surface. The size
line is hex, and the parser rejects anything that is not: no leading plus, no
whitespace, no `0x`, and a bound on how many digits so the value cannot wrap.
A chunk extension after a semicolon is skipped and never interpreted, because
nothing legitimate uses them and interpreting them means a second grammar.

The trailer section is read and discarded. Accepting trailers into the header
set is how a header that a front end already checked gets replaced after the
check, so molla parses them only far enough to find the end of the message.
"""

from std.memory import Pointer

from molla.http.scan import find_byte
from molla.io.buffer import Buffer
from molla.sys.clock import monotonic_ms
from molla.sys.file import (
    MODE_600,
    O_CREAT,
    O_EXCL,
    O_WRONLY,
    close_fd,
    open_at,
    pwrite_all,
    unlink,
)
from molla.sys.mem import as_ptr
from molla.sys.signal import getpid

comptime BODY_NEED_MORE = 0
comptime BODY_DONE = 1
comptime BODY_FAILED = 2

comptime SPILL_THRESHOLD = 1 << 20
"""A megabyte in memory before a body goes to a file. Large enough that a JSON
request or a prompt never touches the disk, small enough that a thousand
connections all uploading at once cannot add up to more than a gigabyte."""

comptime MAX_CHUNK_DIGITS = 16
"""Hex digits allowed in a chunk size line. Sixteen is a full 64 bit value, and
anything longer is a sender trying to overflow the accumulator."""

comptime MAX_TRAILER_BYTES = 8192
"""Trailer section bound. It is discarded, so this only exists to stop a sender
streaming trailers forever after the last chunk."""

comptime _CR: UInt8 = 13
comptime _LF: UInt8 = 10
comptime _SEMI: UInt8 = 59

# Chunked states.
comptime _SIZE = 0
comptime _DATA = 1
comptime _DATA_CRLF = 2
comptime _TRAILER = 3
comptime _FINISHED = 4


def _hex_value(c: UInt8) -> Int:
    if c >= 48 and c <= 57:
        return Int(c - 48)
    if c >= 97 and c <= 102:
        return Int(c - 97) + 10
    if c >= 65 and c <= 70:
        return Int(c - 65) + 10
    return -1


struct BodyReader(Movable):
    """Collects one request body, in memory or on disk."""

    var kind: Int
    """`BODY_NONE`, `BODY_LENGTH` or `BODY_CHUNKED` from `request.mojo`."""

    var remaining: Int
    """Bytes left for `BODY_LENGTH`, or bytes left in the current chunk."""

    var state: Int
    """Where the chunked machine is. Unused for the other two kinds."""

    var held: Buffer
    """Body bytes so far, while they are still in memory."""

    var spilled: Bool
    var spill_fd: Int
    var spill_path: String
    var spill_written: Int

    var total: Int
    """Body bytes decoded so far, wherever they ended up."""

    var limit: Int
    """Largest body accepted. 0 means no limit."""

    var threshold: Int
    var error_status: Int
    var finished: Bool

    var _trailer_bytes: Int
    var _consumed: Int
    """Input bytes taken by the last `feed`. Read through `last_consumed`."""

    def __init__(
        out self,
        kind: Int,
        content_length: Int,
        limit: Int,
        counter: Int,
        threshold: Int = SPILL_THRESHOLD,
    ):
        self.kind = kind
        self.remaining = content_length if content_length > 0 else 0
        self.state = _SIZE
        self.held = Buffer(0, counter)
        self.spilled = False
        self.spill_fd = -1
        self.spill_path = String("")
        self.spill_written = 0
        self.total = 0
        self.limit = limit
        self.threshold = threshold
        self.error_status = 0
        self.finished = False
        self._trailer_bytes = 0
        self._consumed = 0

    def __deinit__(deinit self):
        if self.spill_fd >= 0:
            _ = close_fd(self.spill_fd)
        if self.spill_path.byte_length() > 0:
            _ = unlink(self.spill_path)

    def is_done(self) -> Bool:
        return self.finished

    def in_memory(self) -> Bool:
        return not self.spilled

    def bytes(self) -> Span[UInt8, MutAnyOrigin]:
        """The body, when it stayed in memory. Empty once it has spilled, and
        the caller is expected to check `in_memory` first rather than to read
        this and wonder where the body went."""
        return self.held.bytes()

    def _fail(mut self, status: Int) -> Int:
        self.error_status = status
        return BODY_FAILED

    def _open_spill(mut self) -> Bool:
        """Move what is held to a file and switch to writing through.

        Opened with O_EXCL and retried on a collision rather than trusting a
        name to be unique. A thousand connections on four threads can reach
        this line at the same millisecond with the same byte count behind
        them, so any name built out of what this object knows about itself is
        a name another reader can also build. O_EXCL makes the kernel settle
        it, and the retry is what turns losing that race into a different
        file rather than into two readers writing over each other.
        """
        var attempt = monotonic_ms()
        var tries = 0
        while tries < 64:
            var path = String("/tmp/molla-body-")
            path += String(getpid())
            path += "-"
            path += String(attempt + tries)
            path += ".tmp"
            var rc = open_at(path, O_WRONLY | O_CREAT | O_EXCL, MODE_600)
            if rc.is_ok():
                self.spill_fd = rc.value
                self.spill_path = path
                break
            tries += 1
        if self.spill_fd < 0:
            return False
        var carried = self.held.length
        if carried > 0:
            var wrote = pwrite_all(
                self.spill_fd, as_ptr(self.held.base()), carried, 0
            )
            if not wrote.is_ok():
                return False
            self.spill_written = carried
        self.held.clear()
        # Give the memory back rather than holding a megabyte per spilled body
        # for the life of the connection.
        _ = self.held.reset_to(0)
        self.spilled = True
        return True

    def _store[
        o: MutOrigin
    ](mut self, buf: Pointer[UInt8, o], start: Int, count: Int) -> Bool:
        """Take `count` decoded body bytes, spilling first if they would push
        the in memory total over the threshold."""
        if count == 0:
            return True
        if not self.spilled and self.held.length + count > self.threshold:
            if not self._open_spill():
                return False
        if self.spilled:
            var wrote = pwrite_all(
                self.spill_fd,
                Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=Int(buf) + start
                ),
                count,
                self.spill_written,
            )
            if not wrote.is_ok():
                return False
            self.spill_written += count
            return True
        var data = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=Pointer[UInt8, MutAnyOrigin](
                unsafe_from_address=Int(buf) + start
            ),
            length=count,
        )
        return self.held.append(data)

    def feed[
        o: MutOrigin
    ](mut self, buf: Pointer[UInt8, o], start: Int, length: Int) -> Int:
        """Take as much of `[start, length)` as belongs to this body.

        Returns `BODY_DONE`, `BODY_NEED_MORE` or `BODY_FAILED`, and leaves the
        number of input bytes it took in `consumed`, which the caller reads
        with `last_consumed`. Bytes past the end of the body are left alone,
        which is what makes a pipelined request after a body work.
        """
        self._consumed = 0
        if self.finished:
            return BODY_DONE
        if self.kind == 0:
            self.finished = True
            return BODY_DONE
        if self.kind == 1:
            return self._feed_length(buf, start, length)
        return self._feed_chunked(buf, start, length)

    def last_consumed(self) -> Int:
        return self._consumed

    def _feed_length[
        o: MutOrigin
    ](mut self, buf: Pointer[UInt8, o], start: Int, length: Int) -> Int:
        # A declared length over the limit is refused on the first feed rather
        # than after reading up to the limit and then giving up. The client
        # told us how big it was, so believing it costs nothing and saves
        # reading a gigabyte in order to say no to it.
        if self.limit > 0 and self.total + self.remaining > self.limit:
            return self._fail(413)
        var available = length - start
        var take = min(available, self.remaining)
        if take > 0:
            if self.limit > 0 and self.total + take > self.limit:
                return self._fail(413)
            if not self._store(buf, start, take):
                return self._fail(500)
            self.total += take
            self.remaining -= take
            self._consumed = take
        if self.remaining == 0:
            self.finished = True
            return BODY_DONE
        return BODY_NEED_MORE

    def _feed_chunked[
        o: MutOrigin
    ](mut self, buf: Pointer[UInt8, o], start: Int, length: Int) -> Int:
        var at = start
        while True:
            if self.state == _SIZE:
                # The size line has to arrive whole. Holding out for the LF
                # rather than accumulating digits across calls keeps the parse
                # in one place, and the line is bounded so waiting is safe.
                var lf = find_byte(buf, at, length, _LF)
                if lf < 0:
                    if length - at > MAX_CHUNK_DIGITS + 64:
                        self._consumed = at - start
                        return self._fail(400)
                    self._consumed = at - start
                    return BODY_NEED_MORE
                if lf == at or buf.unsafe_load(lf - 1) != _CR:
                    self._consumed = at - start
                    return self._fail(400)
                var line_end = lf - 1

                var size = 0
                var digits = 0
                var j = at
                while j < line_end:
                    var c = buf.unsafe_load(j)
                    if c == _SEMI:
                        break
                    var v = _hex_value(c)
                    if v < 0:
                        self._consumed = at - start
                        return self._fail(400)
                    digits += 1
                    if digits > MAX_CHUNK_DIGITS:
                        self._consumed = at - start
                        return self._fail(400)
                    size = size * 16 + v
                    j += 1
                if digits == 0:
                    self._consumed = at - start
                    return self._fail(400)
                # Whatever follows a semicolon is a chunk extension. Skipped
                # without being read, because nothing uses them and parsing
                # them would be a second grammar to get wrong.

                at = lf + 1
                if size == 0:
                    self.state = _TRAILER
                    continue
                if self.limit > 0 and self.total + size > self.limit:
                    self._consumed = at - start
                    return self._fail(413)
                self.remaining = size
                self.state = _DATA
                continue

            if self.state == _DATA:
                var take = min(length - at, self.remaining)
                if take > 0:
                    if not self._store(buf, at, take):
                        self._consumed = at - start
                        return self._fail(500)
                    self.total += take
                    self.remaining -= take
                    at += take
                if self.remaining > 0:
                    self._consumed = at - start
                    return BODY_NEED_MORE
                self.state = _DATA_CRLF
                continue

            if self.state == _DATA_CRLF:
                if at + 2 > length:
                    self._consumed = at - start
                    return BODY_NEED_MORE
                if buf.unsafe_load(at) != _CR or buf.unsafe_load(at + 1) != _LF:
                    self._consumed = at - start
                    return self._fail(400)
                at += 2
                self.state = _SIZE
                continue

            if self.state == _TRAILER:
                # Trailer fields, read only far enough to find the blank line
                # that ends the message. Never added to the request's headers.
                var lf = find_byte(buf, at, length, _LF)
                if lf < 0:
                    self._trailer_bytes += length - at
                    if self._trailer_bytes > MAX_TRAILER_BYTES:
                        self._consumed = at - start
                        return self._fail(431)
                    self._consumed = at - start
                    return BODY_NEED_MORE
                if lf == at or buf.unsafe_load(lf - 1) != _CR:
                    self._consumed = at - start
                    return self._fail(400)
                self._trailer_bytes += lf + 1 - at
                if self._trailer_bytes > MAX_TRAILER_BYTES:
                    self._consumed = at - start
                    return self._fail(431)
                var blank = lf - 1 == at
                at = lf + 1
                if blank:
                    self.state = _FINISHED
                    self.finished = True
                    self._consumed = at - start
                    return BODY_DONE
                continue

            self._consumed = at - start
            return BODY_DONE
