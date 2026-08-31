"""multipart/form-data, read as a stream rather than out of a whole body.

This sits after `BodyReader` decodes framing and before anything looks at what
was uploaded. It is fed decoded body bytes as they arrive and produces parts,
each with its own small header block and its own content, and each part's
content spills to a file on the same threshold a whole body does. A four
gigabyte file arriving in a form post therefore costs the same memory as a four
kilobyte one.

## Why the carry buffer exists

A boundary is a fixed string that can land across two reads. Feed a parser
`\\r\\n--bound` in one call and `ary\\r\\n` in the next and a scanner that only
looks at what it was just handed will miss it, write the first half into the
part it was collecting and then fail to find the end of anything. So the reader
holds back the last `len(boundary) + 4` bytes of every feed and prepends them
to the next one. That is the whole trick, and the bound on how much is held is
what stops it being a place where memory grows.

## Where this allocates, and why that is allowed here

A part's headers become owned strings rather than spans, because a part header
can be split across feeds and a span into a buffer that has already been
consumed points at nothing. Multipart is a file upload path, not the inference
path: a request that gets here is already doing megabytes of I/O and a handful
of small strings is not what makes it slow. The zero allocation claim in #17 is
about the JSON request path, and the parts of it that run per token, and this
is neither.

## What is deliberately not supported

Nested multipart, which nothing sends to an API. `multipart/byteranges`, which
is a response type. Base64 or quoted-printable content transfer encodings,
which are email conventions that HTTP form posts do not use. A part carrying
one of them is an error rather than something read wrongly.
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

comptime MULTIPART_NEED_MORE = 0
comptime MULTIPART_DONE = 1
comptime MULTIPART_FAILED = 2

comptime MAX_PARTS = 64
"""Parts in one message. A form with more than this is not a form."""

comptime MAX_PART_HEADER_BYTES = 4096
"""Header block of one part. Content-Disposition and Content-Type and nothing
else is what real senders emit."""

comptime PART_SPILL_THRESHOLD = 1 << 20

comptime _CR: UInt8 = 13
comptime _LF: UInt8 = 10
comptime _DASH: UInt8 = 45

# Reader states.
comptime _PREAMBLE = 0
comptime _PART_HEADERS = 1
comptime _PART_BODY = 2
comptime _EPILOGUE = 3


def boundary_of(content_type: StringSpan) -> String:
    """Pull the boundary out of a Content-Type value, or return empty.

    Handles the quoted form, because a boundary containing a colon or a space
    has to be quoted and some clients quote unconditionally. Everything else in
    the parameter grammar is ignored, since boundary is the only parameter that
    changes how the body is read.
    """
    var text = content_type
    var n = text.byte_length()
    var p = text.unsafe_ptr()
    var i = 0
    while i + 9 <= n:
        # Case insensitive "boundary=" preceded by a delimiter, so a parameter
        # called "xboundary" does not match.
        var c = p.unsafe_load(i)
        var lower = c + 32 if c >= 65 and c <= 90 else c
        if lower == UInt8(98):
            var target = "boundary="
            var tp = target.unsafe_ptr()
            var matched = True
            for k in range(9):
                var b = p.unsafe_load(i + k)
                var bl = b + 32 if b >= 65 and b <= 90 else b
                if bl != tp.unsafe_load(k):
                    matched = False
                    break
            if matched and (i == 0 or _is_delimiter(p.unsafe_load(i - 1))):
                var at = i + 9
                var quoted = at < n and p.unsafe_load(at) == UInt8(34)
                if quoted:
                    at += 1
                var out = String("")
                while at < n:
                    var v = p.unsafe_load(at)
                    if quoted:
                        if v == UInt8(34):
                            break
                    elif v == UInt8(59) or v == UInt8(32) or v == UInt8(9):
                        break
                    out += chr(Int(v))
                    at += 1
                return out
        i += 1
    return String("")


def _is_delimiter(c: UInt8) -> Bool:
    return c == UInt8(59) or c == UInt8(32) or c == UInt8(9)


struct Part(Movable):
    """One part of a form post."""

    var name: String
    """The `name` parameter of Content-Disposition, which is what a handler
    looks a field up by."""

    var filename: String
    """The `filename` parameter, empty for an ordinary field. Never used as a
    path: it is whatever the client sent and the client is not trusted."""

    var content_type: String
    var held: Buffer
    var spilled: Bool
    var spill_fd: Int
    var spill_path: String
    var size: Int

    def __init__(out self, counter: Int):
        self.name = String("")
        self.filename = String("")
        self.content_type = String("")
        self.held = Buffer(0, counter)
        self.spilled = False
        self.spill_fd = -1
        self.spill_path = String("")
        self.size = 0

    def __deinit__(deinit self):
        if self.spill_fd >= 0:
            _ = close_fd(self.spill_fd)
        if self.spill_path.byte_length() > 0:
            _ = unlink(self.spill_path)

    def in_memory(self) -> Bool:
        return not self.spilled

    def bytes(self) -> Span[UInt8, MutAnyOrigin]:
        return self.held.bytes()


def _open_spill(mut fd: Int, mut path: String) -> Bool:
    """Same O_EXCL and retry as `body.mojo`, for the same reason."""
    var attempt = monotonic_ms()
    var tries = 0
    while tries < 64:
        var candidate = String("/tmp/molla-part-")
        candidate += String(getpid())
        candidate += "-"
        candidate += String(attempt + tries)
        candidate += ".tmp"
        var rc = open_at(candidate, O_WRONLY | O_CREAT | O_EXCL, MODE_600)
        if rc.is_ok():
            fd = rc.value
            path = candidate
            return True
        tries += 1
    return False


struct MultipartReader(Movable):
    """Splits a form post into parts as its bytes arrive."""

    var boundary: String
    """The delimiter without the leading dashes, as it came off Content-Type."""

    var parts: List[Part]
    var state: Int
    var carry: Buffer
    """Tail of the previous feed, held back so a boundary split across two
    feeds is still found."""

    var scratch: Buffer
    """Carry and the new bytes, joined, which is what is actually scanned."""

    var counter: Int
    var threshold: Int
    var error_status: Int
    var finished: Bool
    var header_bytes: Int
    """Bytes of the current part's header block seen so far, for the bound."""

    def __init__(
        out self,
        boundary: String,
        counter: Int,
        threshold: Int = PART_SPILL_THRESHOLD,
    ):
        self.boundary = boundary
        self.parts = List[Part]()
        self.state = _PREAMBLE
        self.carry = Buffer(0, counter)
        self.scratch = Buffer(0, counter)
        self.counter = counter
        self.threshold = threshold
        self.error_status = 0
        self.finished = False
        self.header_bytes = 0

    def is_done(self) -> Bool:
        return self.finished

    def _fail(mut self, status: Int) -> Int:
        self.error_status = status
        return MULTIPART_FAILED

    def _hold(self) -> Int:
        """How many bytes of the tail to keep back.

        The longest thing that has to be matched whole is CRLF, two dashes, the
        boundary, and either two more dashes or a CRLF, so holding the boundary
        plus six is enough and holding less is a bug that only shows up when a
        read lands in the wrong place.
        """
        return self.boundary.byte_length() + 6

    def feed[
        o: MutOrigin
    ](mut self, buf: Pointer[UInt8, o], start: Int, length: Int) -> Int:
        """Take decoded body bytes. Everything handed in is consumed."""
        if self.finished:
            return MULTIPART_DONE
        if self.boundary.byte_length() == 0:
            return self._fail(400)

        # Join the held tail with the new bytes and work over the result. The
        # scratch buffer is reused, so this is a copy and not an allocation
        # after the first feed that needed one.
        self.scratch.clear()
        if self.carry.length > 0:
            var held = Span[UInt8, MutAnyOrigin](
                unsafe_ptr=as_ptr(self.carry.base()), length=self.carry.length
            )
            if not self.scratch.append(held):
                return self._fail(500)
        if length > start:
            var incoming = Span[UInt8, MutAnyOrigin](
                unsafe_ptr=as_ptr(Int(buf) + start), length=length - start
            )
            if not self.scratch.append(incoming):
                return self._fail(500)
        self.carry.clear()

        var data = as_ptr(self.scratch.base())
        var total = self.scratch.length
        var at = 0

        while at < total:
            if self.state == _PREAMBLE:
                var first = self._find_boundary(data, at, total)
                if first < 0:
                    return self._carry(
                        data, max(at, total - self._hold()), total
                    )
                at = first
                if self._is_final(data, at, total):
                    self.finished = True
                    self.state = _EPILOGUE
                    return MULTIPART_DONE
                var opened = self._after_boundary(data, at, total)
                if opened < 0:
                    return self._carry(data, at, total)
                at = opened
                if len(self.parts) >= MAX_PARTS:
                    return self._fail(413)
                self.parts.append(Part(self.counter))
                self.header_bytes = 0
                self.state = _PART_HEADERS
                continue

            if self.state == _PART_HEADERS:
                var end = find_byte(data, at, total, _LF)
                if end < 0:
                    self.header_bytes += total - at
                    if self.header_bytes > MAX_PART_HEADER_BYTES:
                        return self._fail(431)
                    return self._carry(data, at, total)
                if end == at or data.unsafe_load(end - 1) != _CR:
                    return self._fail(400)
                self.header_bytes += end + 1 - at
                if self.header_bytes > MAX_PART_HEADER_BYTES:
                    return self._fail(431)
                var field_end = end - 1
                if field_end == at:
                    at = end + 1
                    self.state = _PART_BODY
                    continue
                if not self._take_part_header(data, at, field_end):
                    return self._fail(400)
                at = end + 1
                continue

            if self.state == _PART_BODY:
                var hit = self._find_boundary(data, at, total)
                if hit < 0:
                    # Everything except the held tail belongs to this part.
                    var keep = total - self._hold()
                    if keep > at:
                        if not self._store(data, at, keep - at):
                            return self._fail(500)
                    return self._carry(data, max(at, keep), total)
                if not self._store(data, at, hit - at):
                    return self._fail(500)
                at = hit
                if self._is_final(data, at, total):
                    self.finished = True
                    self.state = _EPILOGUE
                    return MULTIPART_DONE
                var after = self._after_boundary(data, at, total)
                if after < 0:
                    return self._carry(data, at, total)
                at = after
                if len(self.parts) >= MAX_PARTS:
                    return self._fail(413)
                self.parts.append(Part(self.counter))
                self.header_bytes = 0
                self.state = _PART_HEADERS
                continue

            return MULTIPART_DONE

        return MULTIPART_NEED_MORE

    def _carry[
        o: MutOrigin
    ](mut self, data: Pointer[UInt8, o], from_: Int, total: Int) -> Int:
        """Hold `[from_, total)` back for the next feed."""
        var start = max(0, from_)
        if total > start:
            var tail = Span[UInt8, MutAnyOrigin](
                unsafe_ptr=as_ptr(Int(data) + start), length=total - start
            )
            if not self.carry.append(tail):
                return self._fail(500)
        return MULTIPART_NEED_MORE

    def _find_boundary[
        o: MutOrigin
    ](self, data: Pointer[UInt8, o], start: Int, total: Int) -> Int:
        """Index of the CRLF that precedes a delimiter line, or of the two
        dashes when the delimiter is the very first thing in the body.

        Returns -1 when no complete delimiter is present in what is here. A
        partial one at the tail is what the carry buffer is for.
        """
        var blen = self.boundary.byte_length()
        var bp = self.boundary.unsafe_ptr()
        var at = start
        while at < total:
            var dash = find_byte(data, at, total, _DASH)
            if dash < 0:
                return -1
            if dash + 1 < total and data.unsafe_load(dash + 1) == _DASH:
                var body_at = dash + 2
                if body_at + blen > total:
                    return -1
                var matched = True
                for i in range(blen):
                    if data.unsafe_load(body_at + i) != bp.unsafe_load(i):
                        matched = False
                        break
                if matched:
                    # A delimiter is preceded by CRLF except at the very start
                    # of the body, and that CRLF belongs to the delimiter
                    # rather than to the part before it.
                    if (
                        dash >= 2
                        and data.unsafe_load(dash - 1) == _LF
                        and data.unsafe_load(dash - 2) == _CR
                    ):
                        return dash - 2
                    return dash
            at = dash + 1
        return -1

    def _delimiter_at[
        o: MutOrigin
    ](self, data: Pointer[UInt8, o], hit: Int, total: Int) -> Int:
        """Index just past the boundary token at a hit from `_find_boundary`."""
        var at = hit
        if at + 1 < total and data.unsafe_load(at) == _CR:
            at += 2
        return at + 2 + self.boundary.byte_length()

    def _is_final[
        o: MutOrigin
    ](self, data: Pointer[UInt8, o], hit: Int, total: Int) -> Bool:
        """Whether the delimiter at `hit` is the closing one, which ends with
        two more dashes."""
        var at = self._delimiter_at(data, hit, total)
        if at + 2 > total:
            return False
        return (
            data.unsafe_load(at) == _DASH and data.unsafe_load(at + 1) == _DASH
        )

    def _after_boundary[
        o: MutOrigin
    ](self, data: Pointer[UInt8, o], hit: Int, total: Int) -> Int:
        """Index of the first byte after a non final delimiter line, or -1 when
        the line has not fully arrived."""
        var at = self._delimiter_at(data, hit, total)
        # Transport padding, which the grammar allows and nothing sends.
        while at < total:
            var c = data.unsafe_load(at)
            if c == UInt8(32) or c == UInt8(9):
                at += 1
            else:
                break
        if at + 2 > total:
            return -1
        if data.unsafe_load(at) != _CR or data.unsafe_load(at + 1) != _LF:
            return -1
        return at + 2

    def _take_part_header[
        o: MutOrigin
    ](mut self, data: Pointer[UInt8, o], start: Int, end: Int) -> Bool:
        """Read one part header, keeping only the three that mean anything."""
        var colon = find_byte(data, start, end, UInt8(58))
        if colon < 0:
            return False
        var name = _lower_string(data, start, colon)
        var value_start = colon + 1
        while value_start < end:
            var c = data.unsafe_load(value_start)
            if c == UInt8(32) or c == UInt8(9):
                value_start += 1
            else:
                break
        var value = _string(data, value_start, end)
        var index = len(self.parts) - 1
        if index < 0:
            return False
        if name == "content-disposition":
            self.parts[index].name = _parameter(value, "name")
            self.parts[index].filename = _parameter(value, "filename")
        elif name == "content-type":
            self.parts[index].content_type = value
        elif name == "content-transfer-encoding":
            # An email convention. HTTP form posts do not use it, and reading a
            # part as raw bytes when the sender said it was base64 would hand a
            # handler the wrong content without any sign that it had.
            var folded = _lower(value)
            if folded != "binary" and folded != "8bit" and folded != "7bit":
                return False
        return True

    def _store[
        o: MutOrigin
    ](mut self, data: Pointer[UInt8, o], start: Int, count: Int) -> Bool:
        if count <= 0:
            return True
        var index = len(self.parts) - 1
        if index < 0:
            return False
        if (
            not self.parts[index].spilled
            and self.parts[index].held.length + count > self.threshold
        ):
            var fd = -1
            var path = String("")
            if not _open_spill(fd, path):
                return False
            var carried = self.parts[index].held.length
            if carried > 0:
                var wrote = pwrite_all(
                    fd, as_ptr(self.parts[index].held.base()), carried, 0
                )
                if not wrote.is_ok():
                    _ = close_fd(fd)
                    return False
            self.parts[index].spill_fd = fd
            self.parts[index].spill_path = path
            self.parts[index].held.clear()
            _ = self.parts[index].held.reset_to(0)
            self.parts[index].spilled = True
        if self.parts[index].spilled:
            var wrote = pwrite_all(
                self.parts[index].spill_fd,
                as_ptr(Int(data) + start),
                count,
                self.parts[index].size,
            )
            if not wrote.is_ok():
                return False
            self.parts[index].size += count
            return True
        var chunk = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(Int(data) + start), length=count
        )
        if not self.parts[index].held.append(chunk):
            return False
        self.parts[index].size += count
        return True

    def part(self, name: StringSpan) -> Int:
        """Index of the part with this field name, or -1."""
        for i in range(len(self.parts)):
            if self.parts[i].name == name:
                return i
        return -1


def _string[
    o: MutOrigin
](data: Pointer[UInt8, o], start: Int, end: Int) -> String:
    var out = String("")
    for i in range(start, end):
        out += chr(Int(data.unsafe_load(i)))
    return out


def _lower_string[
    o: MutOrigin
](data: Pointer[UInt8, o], start: Int, end: Int) -> String:
    var out = String("")
    for i in range(start, end):
        var c = data.unsafe_load(i)
        if c >= 65 and c <= 90:
            c += 32
        out += chr(Int(c))
    return out


def _lower(text: String) -> String:
    var out = String("")
    var p = text.unsafe_ptr()
    for i in range(text.byte_length()):
        var c = p.unsafe_load(i)
        if c >= 65 and c <= 90:
            c += 32
        out += chr(Int(c))
    return out


def _parameter(value: String, want: StringSpan) -> String:
    """A parameter out of a Content-Disposition value, quoted or not.

    A backslash escape inside the quoted form is honoured, because a filename
    containing a quote is legal and a parser that stops at the first one takes
    a truncated name.
    """
    var n = value.byte_length()
    var p = value.unsafe_ptr()
    var wlen = want.byte_length()
    var wp = want.unsafe_ptr()
    var i = 0
    while i + wlen + 1 <= n:
        var matched = True
        for k in range(wlen):
            var c = p.unsafe_load(i + k)
            var lower = c + 32 if c >= 65 and c <= 90 else c
            if lower != wp.unsafe_load(k):
                matched = False
                break
        if (
            matched
            and p.unsafe_load(i + wlen) == UInt8(61)
            and (i == 0 or _is_delimiter(p.unsafe_load(i - 1)))
        ):
            var at = i + wlen + 1
            var out = String("")
            if at < n and p.unsafe_load(at) == UInt8(34):
                at += 1
                while at < n:
                    var c = p.unsafe_load(at)
                    if c == UInt8(92) and at + 1 < n:
                        out += chr(Int(p.unsafe_load(at + 1)))
                        at += 2
                        continue
                    if c == UInt8(34):
                        break
                    out += chr(Int(c))
                    at += 1
                return out
            while at < n:
                var c = p.unsafe_load(at)
                if c == UInt8(59) or c == UInt8(32) or c == UInt8(9):
                    break
                out += chr(Int(c))
                at += 1
            return out
        i += 1
    return String("")
