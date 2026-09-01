"""UTF-8, decoded strictly and encoded back.

Strictly means the four rules a decoder is allowed to forget and then cannot
un-forget: an overlong form is not the character it spells, a surrogate is not
a character at all, nothing above U+10FFFF is a character, and a truncated
sequence at the end of a buffer is not the same thing as a bad one in the
middle. Model output arrives a byte at a time, so the last of those is the one
that does real work: `decode` says how many bytes it would need before it says
the bytes are wrong, and the streaming decoder in the tokenizer holds back on
that answer rather than emitting a replacement character a client would then
have to see.

Everything here works on `Span[UInt8, _]` and code points are `Int`, because the
callers are a normalizer that rewrites, a regex that compares, and a decoder
that reassembles, and none of them wants a `String` in the middle.
"""

comptime REPLACEMENT = 0xFFFD
"""U+FFFD, what a byte that cannot start a character decodes to."""

comptime INVALID = -1
"""Not a character, and no more bytes will change that."""

comptime INCOMPLETE = -2
"""Not a character yet. The sequence runs off the end of the span."""

comptime MAX_CODE_POINT = 0x10FFFF
"""The last code point there is. Nothing above this encodes."""


def encoded_length(cp: Int) -> Int:
    """How many bytes `cp` takes in UTF-8, or zero if it is not a code point.

    Surrogates answer zero. They are code points in the sense that they have
    numbers, and they are not characters, and UTF-8 has no encoding for them.
    """
    if cp < 0:
        return 0
    if cp < 0x80:
        return 1
    if cp < 0x800:
        return 2
    if cp < 0x10000:
        if cp >= 0xD800 and cp <= 0xDFFF:
            return 0
        return 3
    if cp <= MAX_CODE_POINT:
        return 4
    return 0


def encode(cp: Int, mut out: List[UInt8]):
    """Append `cp` to `out` as UTF-8.

    A code point that does not encode appends the replacement character, which
    is what the callers want: a normalizer that produced something impossible
    should leave a mark in the text rather than silently drop a character or
    raise into the middle of a hot loop.
    """
    var width = encoded_length(cp)
    if width == 0:
        out.append(0xEF)
        out.append(0xBF)
        out.append(0xBD)
    elif width == 1:
        out.append(UInt8(cp))
    elif width == 2:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif width == 3:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


struct Decoded(Copyable, ImplicitlyCopyable, Movable):
    """One step of a decode: what was there and how far it went."""

    var code: Int
    """The code point, or `INVALID`, or `INCOMPLETE`."""

    var width: Int
    """Bytes consumed. One on a bad byte, so a loop always makes progress.

    On `INCOMPLETE` this is how many bytes the sequence is short by, which is
    the number a streaming decoder needs to decide whether to wait.
    """

    def __init__(out self, code: Int, width: Int):
        self.code = code
        self.width = width


def _continuation(b: UInt8) -> Bool:
    return (b & 0xC0) == 0x80


def decode(data: Span[UInt8, _], at: Int) -> Decoded:
    """Decode the character starting at `at`.

    Returns `INVALID` with a width of one for anything malformed in the middle
    of the span, so the caller can substitute one replacement character per bad
    byte, which is the substitution of maximal subparts everybody else does.
    Returns `INCOMPLETE` when the span ends inside a sequence that is still
    well formed as far as it goes, with the width set to the bytes still
    wanted.
    """
    var length = len(data)
    if at >= length:
        return Decoded(INCOMPLETE, 1)

    var first = data[at]
    if first < 0x80:
        return Decoded(Int(first), 1)
    if first < 0xC2:
        # 0x80 to 0xBF is a continuation with nothing in front of it, and 0xC0
        # and 0xC1 only ever start an overlong two byte form of ASCII.
        return Decoded(INVALID, 1)

    var width: Int
    var code: Int
    if first < 0xE0:
        width = 2
        code = Int(first & 0x1F)
    elif first < 0xF0:
        width = 3
        code = Int(first & 0x0F)
    elif first < 0xF5:
        width = 4
        code = Int(first & 0x07)
    else:
        # 0xF5 upwards would start a code point above U+10FFFF.
        return Decoded(INVALID, 1)

    var have = length - at
    var take = width
    if have < width:
        take = have
    for i in range(1, take):
        if not _continuation(data[at + i]):
            return Decoded(INVALID, 1)
        code = (code << 6) | Int(data[at + i] & 0x3F)

    if have < width:
        return Decoded(INCOMPLETE, width - have)

    # The two checks the byte pattern cannot make on its own. A three byte
    # sequence can spell a surrogate and a two byte sequence can spell ASCII,
    # and neither is the character it looks like.
    if width == 3 and code < 0x800:
        return Decoded(INVALID, 1)
    if width == 4 and code < 0x10000:
        return Decoded(INVALID, 1)
    if code >= 0xD800 and code <= 0xDFFF:
        return Decoded(INVALID, 1)
    if code > MAX_CODE_POINT:
        return Decoded(INVALID, 1)
    return Decoded(code, width)


def is_valid(data: Span[UInt8, _]) -> Bool:
    """True when every byte of `data` is part of a well formed character."""
    var at = 0
    while at < len(data):
        var step = decode(data, at)
        if step.code < 0:
            return False
        at += step.width
    return True


def count_code_points(data: Span[UInt8, _]) -> Int:
    """Characters in `data`, counting anything malformed as one each."""
    var at = 0
    var total = 0
    while at < len(data):
        at += decode(data, at).width
        total += 1
    return total


def to_code_points(data: Span[UInt8, _]) -> List[Int]:
    """Decode the whole span, substituting for anything that does not decode.

    Both failures become one replacement character, including the truncated
    tail, because this is the whole string form: there is no more input coming,
    so an incomplete sequence at the end is simply wrong.
    """
    var out = List[Int]()
    var at = 0
    while at < len(data):
        var step = decode(data, at)
        if step.code < 0:
            out.append(REPLACEMENT)
            at += 1
        else:
            out.append(step.code)
            at += step.width
    return out^


def from_code_points(points: List[Int]) -> List[UInt8]:
    """Encode a list of code points back to UTF-8."""
    var out = List[UInt8]()
    for i in range(len(points)):
        encode(points[i], out)
    return out^


def boundary_before(data: Span[UInt8, _], at: Int) -> Int:
    """The start of the character that contains or ends at `at`.

    Walks back over continuation bytes, at most three of them, so a broken
    buffer cannot make it walk to the front of a megabyte. Used by the
    streaming decoder to find the last place it is safe to cut.
    """
    var i = at
    if i > len(data):
        i = len(data)
    var steps = 0
    while i > 0 and steps < 4:
        i -= 1
        if not _continuation(data[i]):
            return i
        steps += 1
    return i
