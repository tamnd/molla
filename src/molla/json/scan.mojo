"""Finding the bytes that matter, a vector at a time.

Nearly all of a JSON document is bytes that a parser has nothing to do with.
The inside of a string, the whitespace an encoder left behind, the digits of a
number. A scalar parser walks all of them one at a time with a switch on each,
which is the shape that stalls a pipeline: every byte is a compare and a branch
and the predictor has nothing to learn, because the length of a message content
field is arbitrary.

These find the next byte a parser has to make a decision about, and skip
everything in between at vector width. The parser above is still an ordinary
recursive descent state machine, it just steps between interesting positions
rather than between bytes.

## Why there is no tape

simdjson's stage one writes the index of every structural character into a tape
and stage two walks the tape. That is the right design for its problem, which is
parsing a document you are going to query repeatedly, and it costs a pass over
the input and an array as large as the number of structurals.

The problem here is different. A request body is parsed once, into a typed
struct, and thrown away. Everything the tape would remember is consumed
immediately by the state machine that asked for it, so writing it down first is
a materialised intermediate for a value read exactly once. The classification is
the same idea and the same instructions. The tape is what is missing.

## What each one does

`skip_ws` steps over the four bytes JSON calls whitespace. Anything else, control
characters included, stops it, because a control byte between values is an error
and not something to walk past quietly.

`scan_string` is the one that earns its keep, since string content is most of a
chat body. One mask per vector covers all three things that can end the skip: the
closing quote, a backslash that means the next byte is not what it looks like,
and a raw control byte, which RFC 8259 forbids inside a string and which a
careless writer produces from an unescaped newline. Finding all three with the
same compare means validation is free rather than an extra pass.

All of them take an exclusive `limit` and return an absolute index, or -1, so a
result can go straight back in as the next `start`.
"""

from std.memory import Pointer
from std.sys.info import simd_width_of

comptime W = simd_width_of[DType.uint8]()
"""Bytes per vector on this target. 16 with NEON or SSE2, 32 with AVX2."""

comptime SPACE: UInt8 = 32
comptime TAB: UInt8 = 9
comptime LF: UInt8 = 10
comptime CR: UInt8 = 13
comptime QUOTE: UInt8 = 34
comptime BACKSLASH: UInt8 = 92

comptime STR_END = 0
"""The closing quote was found."""

comptime STR_ESCAPE = 1
"""A backslash was found, so the caller has to decode from here."""

comptime STR_CONTROL = 2
"""A raw control byte, which is not allowed inside a JSON string."""

comptime STR_SHORT = 3
"""The buffer ran out before the string ended."""


def skip_ws[
    o: MutOrigin
](buf: Pointer[UInt8, o], start: Int, limit: Int) -> Int:
    """Index of the first byte in `[start, limit)` that is not JSON whitespace,
    or `limit`.

    JSON has exactly four: space, tab, line feed and carriage return. Not the
    fourteen a C `isspace` under some locales will agree to, and not a vertical
    tab or a form feed, both of which some encoders emit and which are an error
    here rather than something to be lenient about.
    """
    var at = start
    var sp = SIMD[DType.uint8, W](SPACE)
    var tab = SIMD[DType.uint8, W](TAB)
    var lf = SIMD[DType.uint8, W](LF)
    var cr = SIMD[DType.uint8, W](CR)
    while at + W <= limit:
        var chunk = buf.unsafe_load[width=W](at)
        var ws = chunk.eq(sp) | chunk.eq(tab) | chunk.eq(lf) | chunk.eq(cr)
        var stop = ~ws
        if stop.reduce_or():
            for i in range(W):
                if stop[i]:
                    return at + i
        at += W
    while at < limit:
        var c = buf.unsafe_load(at)
        if c != SPACE and c != TAB and c != LF and c != CR:
            return at
        at += 1
    return limit


def scan_string[
    o: MutOrigin
](buf: Pointer[UInt8, o], start: Int, limit: Int, mut at: Int) -> Int:
    """Walk string content from `start` until something needs a decision.

    `start` is the first byte after the opening quote. Sets `at` to where it
    stopped and returns one of the `STR_` codes. The quote, the backslash and
    the control byte check share one mask, so a string with no escapes in it
    costs one pass with nothing else layered on top.
    """
    var i = start
    var quote = SIMD[DType.uint8, W](QUOTE)
    var slash = SIMD[DType.uint8, W](BACKSLASH)
    var space = SIMD[DType.uint8, W](SPACE)
    while i + W <= limit:
        var chunk = buf.unsafe_load[width=W](i)
        var hit = chunk.eq(quote) | chunk.eq(slash) | chunk.lt(space)
        if hit.reduce_or():
            for k in range(W):
                if hit[k]:
                    at = i + k
                    var c = buf.unsafe_load(at)
                    if c == QUOTE:
                        return STR_END
                    if c == BACKSLASH:
                        return STR_ESCAPE
                    return STR_CONTROL
        i += W
    while i < limit:
        var c = buf.unsafe_load(i)
        if c == QUOTE:
            at = i
            return STR_END
        if c == BACKSLASH:
            at = i
            return STR_ESCAPE
        if c < SPACE:
            at = i
            return STR_CONTROL
        i += 1
    at = limit
    return STR_SHORT


def find_quote[
    o: MutOrigin
](buf: Pointer[UInt8, o], start: Int, limit: Int) -> Int:
    """Next unescaped quote, for skipping a string whose contents are not
    wanted. Used when a reader is asked to step over a value."""
    var at = start
    while True:
        var pos = at
        var code = scan_string(buf, pos, limit, at)
        if code == STR_END:
            return at
        if code == STR_ESCAPE:
            at += 2
            continue
        return -1


def all_ascii[
    o: MutOrigin
](buf: Pointer[UInt8, o], start: Int, limit: Int) -> Bool:
    """Whether every byte in `[start, limit)` is below 0x80.

    The whole point of asking is that a string of plain ASCII needs no UTF-8
    checking at all, and that is nearly every key and most values in an API
    request. One compare per vector answers it.
    """
    var at = start
    var high = SIMD[DType.uint8, W](0x80)
    while at + W <= limit:
        if buf.unsafe_load[width=W](at).ge(high).reduce_or():
            return False
        at += W
    while at < limit:
        if buf.unsafe_load(at) >= 0x80:
            return False
        at += 1
    return True


def validate_utf8[
    o: MutOrigin
](buf: Pointer[UInt8, o], start: Int, limit: Int) -> Bool:
    """Whether `[start, limit)` is well formed UTF-8.

    Rejects the three things a lenient decoder lets through, all of which are
    security bugs rather than pedantry. An overlong encoding lets `..%c0%af`
    become `../` after a path check has already looked at it. A surrogate half
    is not a character and round trips into a different string through anything
    that speaks UTF-16. And anything above U+10FFFF is not Unicode at all.

    The ASCII check first is not an optimisation for the benchmark, it is the
    real distribution: a JSON key is always ASCII and a model name usually is.
    """
    if all_ascii(buf, start, limit):
        return True
    var at = start
    while at < limit:
        var c = buf.unsafe_load(at)
        if c < 0x80:
            at += 1
            continue
        var need: Int
        var code: Int
        if (c & 0xE0) == 0xC0:
            need = 1
            code = Int(c & 0x1F)
        elif (c & 0xF0) == 0xE0:
            need = 2
            code = Int(c & 0x0F)
        elif (c & 0xF8) == 0xF0:
            need = 3
            code = Int(c & 0x07)
        else:
            return False
        if at + need >= limit:
            return False
        for i in range(1, need + 1):
            var cc = buf.unsafe_load(at + i)
            if (cc & 0xC0) != 0x80:
                return False
            code = (code << 6) | Int(cc & 0x3F)
        if need == 1 and code < 0x80:
            return False
        if need == 2 and code < 0x800:
            return False
        if need == 3 and code < 0x10000:
            return False
        if code > 0x10FFFF:
            return False
        if code >= 0xD800 and code <= 0xDFFF:
            return False
        at += need + 1
    return True


def count_escapes[
    o: MutOrigin
](buf: Pointer[UInt8, o], start: Int, limit: Int) -> Int:
    """How many backslashes are in `[start, limit)`.

    Answering "this string needs no decoding" without a byte loop is what lets
    a value be a span into the read buffer instead of a copy, which is most of
    what makes the parse allocate nothing.
    """
    var at = start
    var total = 0
    var slash = SIMD[DType.uint8, W](BACKSLASH)
    while at + W <= limit:
        var hit = buf.unsafe_load[width=W](at).eq(slash)
        total += Int(hit.cast[DType.uint8]().reduce_add())
        at += W
    while at < limit:
        if buf.unsafe_load(at) == BACKSLASH:
            total += 1
        at += 1
    return total
