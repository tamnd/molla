"""Reading bytes without copying them.

The parser above this works entirely in spans, so everything it needs to ask
about a header name or a path has to be answerable without building a `String`.
Comparing, searching, trimming and parsing an integer are the four things it
does, and all four are here.

Case insensitive comparison is ASCII only and deliberately so. HTTP field names
are ASCII by the grammar, and a Unicode aware fold would be both slower and
wrong here: it would make `HTTP/1.1` equal to something that is not `HTTP/1.1`
under a Turkish locale, which is a real bug that real servers have had.

Nothing in this file allocates. `to_string` does, obviously, and it is the one
function whose name says so, kept apart from the rest for that reason.
"""

from molla.io.arena import Arena
from molla.sys.mem import RawPtr, as_ptr, copy_bytes

comptime UPPER_A: UInt8 = 65
comptime UPPER_Z: UInt8 = 90
comptime LOWER_A: UInt8 = 97
comptime DIGIT_0: UInt8 = 48
comptime DIGIT_9: UInt8 = 57
comptime SPACE: UInt8 = 32
comptime TAB: UInt8 = 9

comptime MAX_UINT_DIGITS = 19
"""Longest decimal that certainly fits in a signed 64 bit integer. A
Content-Length longer than this is refused rather than wrapped, because a
wrapped length is how a request smuggles a second request."""


def lower_ascii(value: UInt8) -> UInt8:
    """ASCII fold, one byte. Bytes outside A to Z come back unchanged."""
    if value >= UPPER_A and value <= UPPER_Z:
        return value + (LOWER_A - UPPER_A)
    return value


def equals(left: Span[UInt8, _], right: Span[UInt8, _]) -> Bool:
    if len(left) != len(right):
        return False
    for i in range(len(left)):
        if left[i] != right[i]:
            return False
    return True


def equals_str(left: Span[UInt8, _], right: StringSpan) -> Bool:
    """Compare against a literal, which is what the caller usually has."""
    if len(left) != right.byte_length():
        return False
    var p = right.unsafe_ptr()
    for i in range(len(left)):
        if left[i] != p.unsafe_load(i):
            return False
    return True


def equals_ignore_case(left: Span[UInt8, _], right: StringSpan) -> Bool:
    """ASCII case insensitive compare against a literal. Header names."""
    if len(left) != right.byte_length():
        return False
    var p = right.unsafe_ptr()
    for i in range(len(left)):
        if lower_ascii(left[i]) != lower_ascii(p.unsafe_load(i)):
            return False
    return True


def starts_with(data: Span[UInt8, _], prefix: StringSpan) -> Bool:
    if len(data) < prefix.byte_length():
        return False
    var p = prefix.unsafe_ptr()
    for i in range(prefix.byte_length()):
        if data[i] != p.unsafe_load(i):
            return False
    return True


def index_of_byte(data: Span[UInt8, _], value: UInt8) -> Int:
    """Where `value` first appears, or -1.

    A plain loop. The SIMD version belongs in the JSON scanner where the strings
    are long and the answer is on the hot path. Header values are tens of bytes
    and the branch predictor eats this."""
    for i in range(len(data)):
        if data[i] == value:
            return i
    return -1


def index_of(data: Span[UInt8, _], needle: StringSpan) -> Int:
    """Where `needle` first appears, or -1. Empty needle matches at 0."""
    var size = needle.byte_length()
    if size == 0:
        return 0
    if len(data) < size:
        return -1
    var p = needle.unsafe_ptr()
    var first = p.unsafe_load(0)
    for start in range(len(data) - size + 1):
        if data[start] != first:
            continue
        var hit = True
        for i in range(1, size):
            if data[start + i] != p.unsafe_load(i):
                hit = False
                break
        if hit:
            return start
    return -1


def is_space(value: UInt8) -> Bool:
    """Optional whitespace in HTTP is space and horizontal tab, nothing else.
    Not a newline, which is a terminator, and not a form feed."""
    return value == SPACE or value == TAB


def trim(data: Span[UInt8, _]) -> Span[UInt8, MutAnyOrigin]:
    """Drop leading and trailing spaces and tabs. Still no copy: the result
    points into the same bytes."""
    var start = 0
    var stop = len(data)
    while start < stop and is_space(data[start]):
        start += 1
    while stop > start and is_space(data[stop - 1]):
        stop -= 1
    return Span[UInt8, MutAnyOrigin](
        unsafe_ptr=as_ptr(Int(data.unsafe_ptr()) + start), length=stop - start
    )


def slice(
    data: Span[UInt8, _], start: Int, length: Int
) -> Span[UInt8, MutAnyOrigin]:
    """A window into the same bytes. Clamped rather than raising, because every
    caller here has already bounds checked and a raise would put an exception
    edge on the parse path."""
    var from_index = start
    if from_index < 0:
        from_index = 0
    if from_index > len(data):
        from_index = len(data)
    var size = length
    if size < 0:
        size = 0
    if from_index + size > len(data):
        size = len(data) - from_index
    return Span[UInt8, MutAnyOrigin](
        unsafe_ptr=as_ptr(Int(data.unsafe_ptr()) + from_index), length=size
    )


def parse_uint(data: Span[UInt8, _]) -> Int:
    """A non negative decimal, or -1 if it is not exactly one.

    Strict. No sign, no whitespace, no leading plus, at least one digit, and a
    refusal past nineteen digits rather than a silent wrap. Content-Length is
    parsed with this, and every one of those rules is a smuggling shape that
    some parser somewhere accepted."""
    if len(data) == 0 or len(data) > MAX_UINT_DIGITS:
        return -1
    var total = 0
    for i in range(len(data)):
        var digit = data[i]
        if digit < DIGIT_0 or digit > DIGIT_9:
            return -1
        total = total * 10 + Int(digit - DIGIT_0)
    return total


def parse_hex(data: Span[UInt8, _]) -> Int:
    """A hexadecimal number, or -1. What chunked framing sizes are written in.

    Chunk extensions are the caller's problem: this takes the digits and stops
    at the first byte that is not one, which means a caller that hands over a
    whole chunk header with an extension gets -1 and has to split it first."""
    if len(data) == 0 or len(data) > 16:
        return -1
    var total = 0
    for i in range(len(data)):
        var value = lower_ascii(data[i])
        var digit = -1
        if value >= DIGIT_0 and value <= DIGIT_9:
            digit = Int(value - DIGIT_0)
        elif value >= LOWER_A and value <= LOWER_A + 5:
            digit = Int(value - LOWER_A) + 10
        if digit < 0:
            return -1
        total = total * 16 + digit
    return total


def copy_into_arena(
    data: Span[UInt8, _], mut arena: Arena
) -> Span[UInt8, MutAnyOrigin]:
    """Put a copy in the arena and return a span over it.

    For the few things that genuinely have to outlive the read buffer, which is
    anything the connection keeps across a `consume`. An empty span comes back
    if the arena is full, which the caller checks with `len`."""
    if len(data) == 0:
        return Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(arena.address), length=0
        )
    var address = arena.alloc_bytes(len(data))
    if address == 0:
        return Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(arena.address), length=0
        )
    copy_bytes(address, Int(data.unsafe_ptr()), len(data))
    return Span[UInt8, MutAnyOrigin](
        unsafe_ptr=as_ptr(address), length=len(data)
    )


def to_string(data: Span[UInt8, _]) -> String:
    """The one function here that allocates, named so you can see it in a diff.

    Nothing on the request path should call this. It is for log lines, error
    messages and tests, where a `String` is what the destination takes."""
    var out = String("")
    for i in range(len(data)):
        out += chr(Int(data[i]))
    return out
