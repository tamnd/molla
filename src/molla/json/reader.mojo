"""Streaming mode. Events over a buffer, with no document in between.

A chat request is parsed once, read once, and thrown away. Building a tree of
nodes for it means allocating a node per value, copying every string out of the
read buffer, walking the tree once, and freeing all of it, to get at a handful of
fields that were already sitting in the buffer the socket filled. The tree is a
materialised intermediate for something read exactly once.

So this is the mode the request path uses. It walks the document and hands back
one event at a time, and the caller writes the values it cares about straight
into its own struct as they go by. Strings are spans into the read buffer, so a
model name or a message body costs nothing to look at, and only a string with a
backslash in it is decoded, into a scratch buffer that is reused for the life of
the reader.

## Pull rather than push

The issue asks for events into a handler. This hands them back instead, and the
reason is what the handler would have had to look like. A push handler is called
with an event and no idea where it is, so it carries a state machine of its own
to know that this string is the model name rather than a message role, which is
the parser's state duplicated into every caller. A pull loop lets the caller be
ordinary code: read the key, switch on it, read the value. The events are the
same events either way and the scanner underneath is the same scanner. Building
a push handler on top of this is a loop and a virtual call. Getting the state
back out of a push handler is not.

## Not resumable, on purpose

The events come from a buffer that holds the whole document. A parser that could
be fed one network read at a time would have to hold a partial token across the
gap, which means copying it somewhere, which is the allocation this whole layer
exists to avoid. `body.mojo` already buffers a request body or spills it to a
file past a megabyte, so a complete buffer is what the layer above has anyway.

## Scratch has a lifetime

`text()` after a string event points either into the source buffer or into the
reader's scratch. The scratch is reused by the next string that needs decoding,
so a caller that wants to keep a decoded string past the next `next()` has to
copy it. That is the same rule the header table follows, and it is the rule that
makes the common case free.
"""

from std.memory import Pointer

from molla.io.buffer import Buffer
from molla.json.number import NUM_INT, Number, parse_number
from molla.json.scan import (
    STR_CONTROL,
    STR_END,
    STR_ESCAPE,
    STR_SHORT,
    scan_string,
    skip_ws,
    validate_utf8,
)
from molla.sys.mem import as_ptr

comptime EV_ERROR = -1
comptime EV_END = 0
comptime EV_NULL = 1
comptime EV_BOOL = 2
comptime EV_NUMBER = 3
comptime EV_STRING = 4
comptime EV_KEY = 5
comptime EV_ARRAY_BEGIN = 6
comptime EV_ARRAY_END = 7
comptime EV_OBJECT_BEGIN = 8
comptime EV_OBJECT_END = 9

comptime JSON_OK = 0
comptime JSON_SYNTAX = 1
comptime JSON_DEPTH = 2
comptime JSON_SHORT = 3
comptime JSON_ESCAPE = 4
comptime JSON_UTF8 = 5
comptime JSON_TRAILING = 6
comptime JSON_NO_SPACE = 7

comptime MAX_DEPTH = 128
"""The hard cap on nesting, which is a bound on the stack this reader keeps and
not a matter of taste. A document nested a million deep is four bytes per level
of someone else's memory, and refusing it is the whole defence."""

comptime DEFAULT_DEPTH = 64
"""What a caller gets unless it says otherwise. An OpenAI chat request nests
four deep and a tool schema maybe eight, so this is already generous."""

comptime _IN_OBJECT: UInt8 = 1
comptime _SEEN: UInt8 = 2


def error_text(code: Int) -> StaticString:
    if code == JSON_OK:
        return "ok"
    if code == JSON_SYNTAX:
        return "syntax error"
    if code == JSON_DEPTH:
        return "nested too deeply"
    if code == JSON_SHORT:
        return "input ended inside a value"
    if code == JSON_ESCAPE:
        return "bad escape sequence"
    if code == JSON_UTF8:
        return "invalid UTF-8 in a string"
    if code == JSON_TRAILING:
        return "trailing content after the document"
    if code == JSON_NO_SPACE:
        return "out of scratch space"
    return "unknown error"


def _hex(c: UInt8) -> Int:
    if c >= 48 and c <= 57:
        return Int(c) - 48
    if c >= 97 and c <= 102:
        return Int(c) - 87
    if c >= 65 and c <= 70:
        return Int(c) - 55
    return -1


struct Reader(Movable):
    """A position in a buffer, a stack of containers, and one event."""

    var base: Int
    var length: Int
    var at: Int

    var event: Int
    var error: Int
    var error_at: Int
    """Byte offset the error was found at. A parse failure with no offset is a
    support ticket, and a parse failure with one is a two minute look at the
    body."""

    var depth: Int
    var max_depth: Int
    var stack: InlineArray[UInt8, MAX_DEPTH]

    var started: Bool
    var done: Bool
    var pending_value: Bool

    var str_at: Int
    var str_len: Int
    var str_decoded: Bool
    """The bytes are in the scratch buffer rather than in the source, because
    the string had an escape in it."""

    var scratch: Buffer
    var number: Number
    var bool_value: Bool

    var strings: Int
    """How many strings were seen."""

    var decoded: Int
    """How many of them had to be decoded. A ratio close to zero is the whole
    argument for spans, and a ratio close to one on real traffic would be the
    argument against."""

    def __init__(out self, counter: Int, scratch_capacity: Int = 1024):
        self.base = 0
        self.length = 0
        self.at = 0
        self.event = EV_END
        self.error = JSON_OK
        self.error_at = 0
        self.depth = 0
        self.max_depth = DEFAULT_DEPTH
        self.stack = InlineArray[UInt8, MAX_DEPTH](fill=0)
        self.started = False
        self.done = False
        self.pending_value = False
        self.str_at = 0
        self.str_len = 0
        self.str_decoded = False
        self.scratch = Buffer(scratch_capacity, counter)
        self.number = Number()
        self.bool_value = False
        self.strings = 0
        self.decoded = 0

    def configure_depth(mut self, depth: Int):
        var want = depth
        if want < 1:
            want = 1
        if want > MAX_DEPTH:
            want = MAX_DEPTH
        self.max_depth = want

    def begin(mut self, data: Span[UInt8, _]):
        """Point at a document.

        The buffer has to outlive the reader, which is the same rule the header
        table follows. Only the address is kept, so this call is the last thing
        the compiler sees using the buffer: a buffer held in a local is freed
        here unless the caller ends the scope with `keep(buf)` from
        `molla.sys.mem`. A buffer that is a field of a connection has no such
        problem, which is why the server never hits it and a test does.
        """
        self.base = Int(data.unsafe_ptr())
        self.length = len(data)
        self.at = 0
        self.event = EV_END
        self.error = JSON_OK
        self.error_at = 0
        self.depth = 0
        self.started = False
        self.done = False
        self.pending_value = False
        self.str_at = 0
        self.str_len = 0
        self.str_decoded = False
        self.scratch.clear()
        self.strings = 0
        self.decoded = 0

    def ok(self) -> Bool:
        return self.error == JSON_OK

    def text(self) -> Span[UInt8, MutAnyOrigin]:
        """The bytes of the last string or key event.

        Valid until the next `next()` when the string needed decoding, and for
        as long as the source buffer lives when it did not.
        """
        if self.str_decoded:
            return Span[UInt8, MutAnyOrigin](
                unsafe_ptr=as_ptr(self.scratch.base() + self.str_at),
                length=self.str_len,
            )
        return Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(self.base + self.str_at), length=self.str_len
        )

    def _fail(mut self, code: Int, offset: Int) -> Int:
        if self.error == JSON_OK:
            self.error = code
            self.error_at = offset
        self.event = EV_ERROR
        return EV_ERROR

    def _push(mut self, in_object: Bool) -> Bool:
        if self.depth >= self.max_depth:
            return False
        self.stack[self.depth] = _IN_OBJECT if in_object else 0
        self.depth += 1
        return True

    def _string(mut self) -> Bool:
        """Read a string starting at the opening quote, leaving `at` past the
        closing one. Sets the span, decoding only if it has to."""
        var p = as_ptr(self.base)
        var start = self.at + 1
        var i = start
        var code = scan_string(p, start, self.length, i)
        if code == STR_END:
            # The common case. Nothing to decode, so the value is a span into
            # the buffer the socket filled.
            self.str_at = start
            self.str_len = i - start
            self.str_decoded = False
            self.at = i + 1
            self.strings += 1
            if not validate_utf8(p, start, i):
                _ = self._fail(JSON_UTF8, start)
                return False
            return True
        if code == STR_SHORT:
            _ = self._fail(JSON_SHORT, start)
            return False
        if code == STR_CONTROL:
            # A raw newline inside a string is the usual cause, and it comes
            # from a writer that concatenated strings instead of encoding them.
            _ = self._fail(JSON_SYNTAX, i)
            return False
        return self._decode(start, i)

    def _decode(mut self, start: Int, first_escape: Int) -> Bool:
        """The uncommon case. Copy into scratch, undoing escapes as they come.

        Everything before the first backslash is copied in one go rather than
        byte at a time, since a string with one escape at the end is still
        mostly a memcpy.
        """
        var p = as_ptr(self.base)
        var mark = self.scratch.length
        var i = first_escape
        if not self.scratch.append(
            Span[UInt8, MutAnyOrigin](
                unsafe_ptr=as_ptr(self.base + start),
                length=first_escape - start,
            )
        ):
            _ = self._fail(JSON_NO_SPACE, start)
            return False

        while True:
            if i >= self.length:
                _ = self._fail(JSON_SHORT, i)
                return False
            var c = p.unsafe_load(i)
            if c == 34:
                self.str_at = mark
                self.str_len = self.scratch.length - mark
                self.str_decoded = True
                self.at = i + 1
                self.strings += 1
                self.decoded += 1
                var bytes = self.scratch.bytes()
                if not validate_utf8(
                    bytes.unsafe_ptr(), self.str_at, self.scratch.length
                ):
                    _ = self._fail(JSON_UTF8, start)
                    return False
                return True
            if c == 92:
                if not self._escape(i):
                    return False
                i = self.at
                continue
            if c < 32:
                _ = self._fail(JSON_SYNTAX, i)
                return False
            # A run of ordinary bytes between two escapes, copied in one call.
            var run = i
            var stop = i
            var code = scan_string(p, run, self.length, stop)
            if code == STR_SHORT:
                _ = self._fail(JSON_SHORT, stop)
                return False
            if not self.scratch.append(
                Span[UInt8, MutAnyOrigin](
                    unsafe_ptr=as_ptr(self.base + run), length=stop - run
                )
            ):
                _ = self._fail(JSON_NO_SPACE, run)
                return False
            i = stop

    def _escape(mut self, at: Int) -> Bool:
        """Decode one escape starting at the backslash, leaving `self.at` past
        it. `\\u` is the only one that is more than two bytes wide."""
        var p = as_ptr(self.base)
        if at + 1 >= self.length:
            _ = self._fail(JSON_SHORT, at)
            return False
        var c = p.unsafe_load(at + 1)
        var out: UInt8
        if c == 34 or c == 92 or c == 47:
            out = c
        elif c == 98:
            out = 8
        elif c == 102:
            out = 12
        elif c == 110:
            out = 10
        elif c == 114:
            out = 13
        elif c == 116:
            out = 9
        elif c == 117:
            return self._escape_unicode(at)
        else:
            # `\x41` and `\'` are JavaScript, not JSON, and a parser that takes
            # them will read a document some other parser refuses.
            _ = self._fail(JSON_ESCAPE, at)
            return False
        if not self.scratch.append_byte(out):
            _ = self._fail(JSON_NO_SPACE, at)
            return False
        self.at = at + 2
        return True

    def _read_u16(mut self, at: Int) -> Int:
        """The four hex digits after a `\\u`, or -1."""
        if at + 4 > self.length:
            return -1
        var p = as_ptr(self.base)
        var value = 0
        for i in range(4):
            var d = _hex(p.unsafe_load(at + i))
            if d < 0:
                return -1
            value = value * 16 + d
        return value

    def _escape_unicode(mut self, at: Int) -> Bool:
        """`\\uXXXX`, and the surrogate pair that follows it when the code point
        is above the basic plane.

        A lone surrogate is refused rather than replaced. Replacing it produces
        a string that is valid UTF-8 and is not what was sent, and the caller
        has no way to tell. Refusing it is a 400 with a byte offset in it.
        """
        var first = self._read_u16(at + 2)
        if first < 0:
            _ = self._fail(JSON_ESCAPE, at)
            return False
        var code = first
        var width = 6
        if first >= 0xD800 and first <= 0xDBFF:
            # A high surrogate, which is only meaningful with its partner.
            var p = as_ptr(self.base)
            if (
                at + 7 >= self.length
                or p.unsafe_load(at + 6) != 92
                or p.unsafe_load(at + 7) != 117
            ):
                _ = self._fail(JSON_ESCAPE, at)
                return False
            var second = self._read_u16(at + 8)
            if second < 0xDC00 or second > 0xDFFF:
                _ = self._fail(JSON_ESCAPE, at)
                return False
            code = 0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
            width = 12
        elif first >= 0xDC00 and first <= 0xDFFF:
            _ = self._fail(JSON_ESCAPE, at)
            return False

        var ok: Bool
        if code < 0x80:
            ok = self.scratch.append_byte(UInt8(code))
        elif code < 0x800:
            ok = self.scratch.append_byte(UInt8(0xC0 | (code >> 6)))
            ok = ok and self.scratch.append_byte(UInt8(0x80 | (code & 0x3F)))
        elif code < 0x10000:
            ok = self.scratch.append_byte(UInt8(0xE0 | (code >> 12)))
            ok = ok and self.scratch.append_byte(
                UInt8(0x80 | ((code >> 6) & 0x3F))
            )
            ok = ok and self.scratch.append_byte(UInt8(0x80 | (code & 0x3F)))
        else:
            ok = self.scratch.append_byte(UInt8(0xF0 | (code >> 18)))
            ok = ok and self.scratch.append_byte(
                UInt8(0x80 | ((code >> 12) & 0x3F))
            )
            ok = ok and self.scratch.append_byte(
                UInt8(0x80 | ((code >> 6) & 0x3F))
            )
            ok = ok and self.scratch.append_byte(UInt8(0x80 | (code & 0x3F)))
        if not ok:
            _ = self._fail(JSON_NO_SPACE, at)
            return False
        self.at = at + width
        return True

    def _literal(mut self, text: StaticString, event: Int, value: Bool) -> Int:
        var p = as_ptr(self.base)
        var n = text.byte_length()
        if self.at + n > self.length:
            return self._fail(JSON_SHORT, self.at)
        var q = text.unsafe_ptr()
        for i in range(n):
            if p.unsafe_load(self.at + i) != q.unsafe_load(i):
                return self._fail(JSON_SYNTAX, self.at)
        self.at += n
        self.bool_value = value
        self.event = event
        return event

    def _value(mut self) -> Int:
        """Parse one value at `at`, which is already past any whitespace."""
        if self.at >= self.length:
            return self._fail(JSON_SHORT, self.at)
        var p = as_ptr(self.base)
        var c = p.unsafe_load(self.at)

        if c == 123:
            if not self._push(True):
                return self._fail(JSON_DEPTH, self.at)
            self.at += 1
            self.event = EV_OBJECT_BEGIN
            return EV_OBJECT_BEGIN
        if c == 91:
            if not self._push(False):
                return self._fail(JSON_DEPTH, self.at)
            self.at += 1
            self.event = EV_ARRAY_BEGIN
            return EV_ARRAY_BEGIN
        if c == 34:
            if not self._string():
                return EV_ERROR
            self.event = EV_STRING
            return EV_STRING
        if c == 116:
            return self._literal("true", EV_BOOL, True)
        if c == 102:
            return self._literal("false", EV_BOOL, False)
        if c == 110:
            return self._literal("null", EV_NULL, False)
        if c == 45 or (c >= 48 and c <= 57):
            var end = parse_number(p, self.at, self.length, self.number)
            if end < 0:
                return self._fail(JSON_SYNTAX, self.at)
            self.at = end
            self.event = EV_NUMBER
            return EV_NUMBER
        return self._fail(JSON_SYNTAX, self.at)

    def next(mut self) -> Int:
        """The next event, or `EV_END` at the end and `EV_ERROR` on a bad
        document. Once either of those comes back it keeps coming back."""
        if self.error != JSON_OK:
            return EV_ERROR
        if self.done:
            self.event = EV_END
            return EV_END

        var p = as_ptr(self.base)
        if self.pending_value:
            # The colon after a key has already been eaten.
            self.pending_value = False
            self.at = skip_ws(p, self.at, self.length)
            return self._value()

        self.at = skip_ws(p, self.at, self.length)

        if self.depth == 0:
            if self.started:
                if self.at != self.length:
                    return self._fail(JSON_TRAILING, self.at)
                self.done = True
                self.event = EV_END
                return EV_END
            self.started = True
            return self._value()

        if self.at >= self.length:
            return self._fail(JSON_SHORT, self.at)
        var top = self.stack[self.depth - 1]
        var in_object = (top & _IN_OBJECT) != 0
        var seen = (top & _SEEN) != 0
        var c = p.unsafe_load(self.at)

        var close = UInt8(125) if in_object else UInt8(93)
        if c == close:
            self.at += 1
            self.depth -= 1
            self.event = EV_OBJECT_END if in_object else EV_ARRAY_END
            return self.event

        if seen:
            if c != 44:
                return self._fail(JSON_SYNTAX, self.at)
            self.at += 1
            self.at = skip_ws(p, self.at, self.length)
            if self.at >= self.length:
                return self._fail(JSON_SHORT, self.at)
            c = p.unsafe_load(self.at)
            if c == close:
                # A trailing comma. Legal in JavaScript, not in JSON, and the
                # two disagreeing is how a document parses on one end and not
                # on the other.
                return self._fail(JSON_SYNTAX, self.at)
        self.stack[self.depth - 1] = top | _SEEN

        if not in_object:
            return self._value()

        if c != 34:
            return self._fail(JSON_SYNTAX, self.at)
        if not self._string():
            return EV_ERROR
        self.at = skip_ws(p, self.at, self.length)
        if self.at >= self.length or p.unsafe_load(self.at) != 58:
            return self._fail(JSON_SYNTAX, self.at)
        self.at += 1
        self.pending_value = True
        self.event = EV_KEY
        return EV_KEY

    def skip_value(mut self) -> Bool:
        """Step over the value the last event opened.

        For a scalar that is nothing. For a container it runs to the matching
        close. This is what lets a caller filling a typed struct ignore a field
        it does not know without knowing how big it is, which is most of what
        makes forward compatibility cheap.
        """
        if self.event != EV_ARRAY_BEGIN and self.event != EV_OBJECT_BEGIN:
            return self.error == JSON_OK
        var target = self.depth - 1
        while self.depth > target:
            var e = self.next()
            if e == EV_ERROR or e == EV_END:
                return False
        return True

    def skip_next_value(mut self) -> Bool:
        """Read the next value and step over it, which is what a caller does
        with the value belonging to a key it does not recognise."""
        var e = self.next()
        if e == EV_ERROR or e == EV_END:
            return False
        return self.skip_value()

    def key_is(self, name: StringSpan) -> Bool:
        """Whether the last key event matches, compared as bytes."""
        var text = self.text()
        if len(text) != name.byte_length():
            return False
        var p = name.unsafe_ptr()
        for i in range(len(text)):
            if text[i] != p.unsafe_load(i):
                return False
        return True

    def finish(mut self) -> Bool:
        """Run to the end of the document and say whether it was well formed.

        A caller that stops as soon as it has the fields it wants has not
        checked that the rest of the body parses, and a body that is truncated
        or has garbage after it is a bad request whether or not the interesting
        part came first.
        """
        while True:
            var e = self.next()
            if e == EV_END:
                return True
            if e == EV_ERROR:
                return False
