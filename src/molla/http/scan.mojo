"""Finding delimiters in a header block, a vector at a time.

An HTTP header block is a search problem before it is a parsing problem. Nearly
all of the work is locating the next line feed and the next colon, and checking
that nothing between them is a byte that is not allowed to be there. A scalar
parser does that one byte at a time with a branch on each, which is the shape
that stalls a pipeline: every byte is a compare and a conditional jump, and the
predictor has nothing to learn because header lengths are arbitrary.

These read a vector at a time, produce a mask, and only fall back to a byte
loop inside the one vector that contains the answer. The scalar loop still
exists and still runs, it just runs over sixteen or thirty two bytes instead of
over the whole block.

The width is the native one for the target rather than a fixed sixteen. On the
M4 that is sixteen bytes, on a machine with AVX2 it is thirty two, and the code
does not change either way. What does change is the tail: the last partial
vector is handled with a plain loop rather than with a masked load, because a
masked load past the end of the buffer is only safe if you know the allocation
extends that far and the caller here does not promise that.

All four take an exclusive `limit` and return an absolute index into the
buffer, or -1 for not found, so a caller can pass a result straight back in as
the next `start` without arithmetic.
"""

from std.memory import Pointer
from std.sys.info import simd_width_of

comptime W = simd_width_of[DType.uint8]()
"""Bytes per vector on this target. 16 with NEON or SSE2, 32 with AVX2."""

comptime CR: UInt8 = 13
comptime LF: UInt8 = 10
comptime HTAB: UInt8 = 9
comptime SP: UInt8 = 32
comptime DEL: UInt8 = 127


def find_byte[
    o: MutOrigin
](buf: Pointer[UInt8, o], start: Int, limit: Int, value: UInt8) -> Int:
    """First occurrence of `value` in `[start, limit)`, or -1."""
    var at = start
    var target = SIMD[DType.uint8, W](value)
    while at + W <= limit:
        var hit = buf.unsafe_load[width=W](at).eq(target)
        if hit.reduce_or():
            for i in range(W):
                if hit[i]:
                    return at + i
        at += W
    while at < limit:
        if buf.unsafe_load(at) == value:
            return at
        at += 1
    return -1


def find_ctl[
    o: MutOrigin
](buf: Pointer[UInt8, o], start: Int, limit: Int) -> Int:
    """First byte in `[start, limit)` that may not appear in a header value.

    That is any control character except horizontal tab, plus DEL. The reason
    this is a separate scan rather than a check inside the delimiter scan is
    that it is the check people leave out: a value carrying a raw CR or a NUL
    is how a header gets split or truncated by whichever hop is less careful,
    and the parser that finds the delimiter without validating the bytes before
    it will happily pass one along.

    Two compares and an or per vector, which is cheap enough that there is no
    argument for making it optional.
    """
    var at = start
    var space = SIMD[DType.uint8, W](SP)
    var tab = SIMD[DType.uint8, W](HTAB)
    var del_ = SIMD[DType.uint8, W](DEL)
    while at + W <= limit:
        var chunk = buf.unsafe_load[width=W](at)
        # Below space and not a tab, or exactly DEL. Anything at or above 0x80
        # is left alone, since obs-text is allowed to appear in a value.
        var bad = (chunk.lt(space) & chunk.ne(tab)) | chunk.eq(del_)
        if bad.reduce_or():
            for i in range(W):
                if bad[i]:
                    return at + i
        at += W
    while at < limit:
        var c = buf.unsafe_load(at)
        if (c < SP and c != HTAB) or c == DEL:
            return at
        at += 1
    return -1


def find_crlf[
    o: MutOrigin
](buf: Pointer[UInt8, o], start: Int, limit: Int) -> Int:
    """First CRLF pair in `[start, limit)`, returning the index of the CR.

    A CR that is not followed by an LF is skipped rather than reported, because
    the caller wants a line terminator and a lone CR is not one. Whether a lone
    CR is an error is the parser's decision and it makes it separately.
    """
    var at = start
    while True:
        var cr = find_byte(buf, at, limit, CR)
        if cr < 0:
            return -1
        if cr + 1 >= limit:
            return -1
        if buf.unsafe_load(cr + 1) == LF:
            return cr
        at = cr + 1


def count_byte[
    o: MutOrigin
](buf: Pointer[UInt8, o], start: Int, limit: Int, value: UInt8) -> Int:
    """How many times `value` appears in `[start, limit)`.

    Used on the boundary scan in multipart, where knowing there is no candidate
    at all in a whole buffer is the common case and is worth answering without
    a byte loop.
    """
    var at = start
    var total = 0
    var target = SIMD[DType.uint8, W](value)
    while at + W <= limit:
        var hit = buf.unsafe_load[width=W](at).eq(target)
        total += Int(hit.cast[DType.uint8]().reduce_add())
        at += W
    while at < limit:
        if buf.unsafe_load(at) == value:
            total += 1
        at += 1
    return total
