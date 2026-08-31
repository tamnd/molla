"""Tests for the memory layer.

Three things are worth testing here and the rest is bookkeeping. That the growth
policy does what the docstring says, because a server's memory profile is the
sum of its growth policies. That the ring wraps correctly, because an off by one
in the wrap is a corrupted response body rather than a crash. And that the
allocation counter tells the truth, because issue #17 is going to assert on it
and a counter that undercounts would make that assertion meaningless.
"""

from harness import Suite

from molla.io.arena import DEFAULT_ARENA, Arena
from molla.io.buffer import (
    DOUBLE_UNTIL,
    GROW_STEP,
    MIN_CAPACITY,
    Buffer,
    _next_capacity,
)
from molla.io.bytes import (
    copy_into_arena,
    equals,
    equals_ignore_case,
    equals_str,
    index_of,
    index_of_byte,
    parse_hex,
    parse_uint,
    slice,
    starts_with,
    to_string,
    trim,
)
from molla.io.ring import Ring
from molla.sys.mem import AllocCounter, allocate, release


def _span(text: StringSpan) -> Span[UInt8, MutAnyOrigin]:
    """A span over a literal, for feeding the byte helpers."""
    return Span[UInt8, MutAnyOrigin](
        unsafe_ptr=Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(text.unsafe_ptr())
        ),
        length=text.byte_length(),
    )


def _check_mem(mut suite: Suite) raises:
    suite.group("io.mem")

    var block = allocate(128)
    suite.check(block != 0, "allocate returns a block")
    var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=block)
    var zeroed = True
    for i in range(128):
        if p.unsafe_load(i) != 0:
            zeroed = False
    suite.check(zeroed, "and it is zeroed")
    release(block)
    suite.check(allocate(0) == 0, "a zero sized allocation is refused")

    var counter = AllocCounter()
    suite.check(counter.raw() != 0, "a counter allocates its own storage")
    suite.check(counter.total() == 0, "and starts at zero")

    var scratch = Buffer(256, counter.raw())
    suite.check(counter.total() == 1, "a buffer counts as one allocation")
    suite.check(counter.bytes() == 256, "and records its size")
    suite.check(counter.live() == 1, "and is live")
    _ = scratch.append_str("x")
    suite.check(counter.total() == 1, "writing inside the capacity is free")
    counter.reset()
    suite.check(counter.total() == 0, "reset zeroes the totals")
    counter.close()


def _check_growth(mut suite: Suite) raises:
    suite.group("io.buffer growth policy")

    suite.check(
        _next_capacity(0, 1) == MIN_CAPACITY, "a tiny ask gets the floor"
    )
    suite.check(_next_capacity(64, 65) == 128, "doubling while small")
    suite.check(
        _next_capacity(DOUBLE_UNTIL, DOUBLE_UNTIL + 1)
        == DOUBLE_UNTIL + GROW_STEP,
        "a fixed step once large",
    )
    suite.check(
        _next_capacity(64, 5000) == 8192,
        "one call reaches a capacity many doublings away",
    )
    suite.check(
        _next_capacity(1024, 100) == 1024,
        "an ask that already fits changes nothing",
    )


def _check_buffer(mut suite: Suite) raises:
    suite.group("io.buffer")

    var counter = AllocCounter()
    var buf = Buffer(MIN_CAPACITY, counter.raw())
    suite.check(buf.is_valid(), "a buffer allocates")
    suite.check(buf.length == 0, "and starts empty")
    suite.check(buf.available() == MIN_CAPACITY, "with a writable tail")

    suite.check(buf.append_str("GET / HTTP/1.1"), "append takes a literal")
    suite.check(buf.length == 14, "and the length follows")
    suite.check(
        equals_str(buf.bytes(), "GET / HTTP/1.1"), "and the bytes match"
    )

    suite.check(buf.append_byte(33), "append_byte works")
    suite.check(buf.at(14) == 33, "and lands at the end")
    buf.consume(14)
    suite.check(buf.length == 1, "consume drops the front")
    suite.check(buf.at(0) == 33, "and moves the rest down")
    buf.clear()
    suite.check(buf.length == 0, "clear empties it")
    suite.check(buf.capacity == MIN_CAPACITY, "and keeps the block")

    var allocations = counter.total()
    var long_text = String("")
    for _ in range(20):
        long_text += "0123456789"
    _ = buf.append_str(long_text)
    suite.check(buf.length == 200, "a write past the capacity grows")
    suite.check(buf.capacity >= 200, "and the capacity follows")
    suite.check(
        counter.total() > allocations, "and the growth is counted, not hidden"
    )
    suite.check(
        equals_str(slice(buf.bytes(), 0, 10), "0123456789"),
        "the contents survived the move",
    )

    # A read straight into the tail, which is how the reactor fills this.
    var before = buf.length
    var tail = buf.tail_ptr()
    tail.unsafe_store(0, 65)
    tail.unsafe_store(1, 66)
    buf.commit(2)
    suite.check(buf.length == before + 2, "commit accepts written bytes")
    suite.check(buf.at(before) == 65, "and they are where they were written")
    buf.commit(1000000)
    suite.check(
        buf.length == buf.capacity, "a wrong commit clamps rather than escapes"
    )

    _ = buf.reset_to(MIN_CAPACITY)
    suite.check(buf.capacity == MIN_CAPACITY, "reset_to shrinks a recycled one")
    suite.check(buf.length == 0, "and empties it")

    # After the buffer, always. The counter's block outlives everything that
    # holds its address, and the buffer above is destroyed at its last use just
    # over this line.
    counter.close()


def _check_ring(mut suite: Suite) raises:
    suite.group("io.ring")

    var counter = AllocCounter()
    var ring = Ring(100, counter.raw())
    suite.check(ring.is_valid(), "a ring allocates")
    suite.check(ring.capacity == 128, "and rounds up to a power of two")
    suite.check(
        ring.writable() == 127, "with one byte given up to tell full from empty"
    )
    suite.check(ring.is_empty(), "and starts empty")

    suite.check(ring.push_str("hello") == 5, "push takes bytes")
    suite.check(ring.readable() == 5, "and they are queued")
    suite.check(ring.peek(0) == UInt8(ord("h")), "in order")
    suite.check(ring.peek(4) == UInt8(ord("o")), "to the end")

    var bases = List[Int]()
    var lengths = List[Int]()
    suite.check(
        ring.readable_pieces(bases, lengths) == 1,
        "unwrapped output is one piece",
    )
    suite.check(lengths[0] == 5, "and it is all of it")

    ring.consume(2)
    suite.check(ring.readable() == 3, "consume advances the read offset")
    suite.check(ring.peek(0) == UInt8(ord("l")), "and the front moves")

    # Fill it, drain it, and fill it again, so the data has to wrap.
    ring.clear()
    var filler = String("")
    for _ in range(12):
        filler += "0123456789"
    suite.check(ring.push_str(filler) == 120, "a large push fits")
    ring.consume(115)
    suite.check(ring.readable() == 5, "and drains")
    suite.check(
        ring.push_str("abcdefghij") == 10, "a push after draining wraps"
    )
    suite.check(ring.readable() == 15, "and everything is queued")
    suite.check(
        ring.readable_pieces(bases, lengths) == 2,
        "wrapped output is two pieces",
    )
    suite.check(
        lengths[0] + lengths[1] == 15, "and the pieces cover every byte"
    )
    suite.check(ring.peek(5) == UInt8(ord("a")), "the wrapped bytes read back")
    suite.check(ring.peek(14) == UInt8(ord("j")), "all the way to the end")

    var wrote = ring.push_str(filler)
    suite.check(wrote < 120, "a push past the capacity is short, not an error")
    suite.check(ring.writable() == 0, "and the ring is full")
    suite.check(ring.push_str("x") == 0, "a full ring takes nothing")
    ring.consume(1000000)
    suite.check(ring.is_empty(), "consuming more than is there empties it")
    suite.check(counter.total() == 1, "a ring never grows")
    _ = ring^
    counter.close()


def _check_arena(mut suite: Suite) raises:
    suite.group("io.arena")

    var counter = AllocCounter()
    var arena = Arena(1024, counter.raw())
    suite.check(arena.is_valid(), "an arena allocates")
    suite.check(arena.used() == 0, "and starts empty")

    var first = arena.alloc_bytes(10)
    suite.check(first != 0, "alloc returns an address")
    suite.check(arena.contains(first), "inside the block")
    suite.check(arena.used() == 10, "and the offset moves")

    var aligned = arena.alloc(8, 8)
    suite.check(aligned % 8 == 0, "an aligned request is aligned")
    suite.check(aligned > first, "and comes after the last one")

    var ints = arena.alloc_ints(4)
    suite.check(ints % 8 == 0, "integers are aligned for the target")
    var p = Pointer[Int, MutAnyOrigin](unsafe_from_address=ints)
    for i in range(4):
        p.unsafe_store(i, i * 100)
    suite.check(p.unsafe_load(3) == 300, "and the memory is usable")

    var mark = arena.high_water
    suite.check(mark >= arena.used(), "the high water mark tracks the offset")
    arena.reset()
    suite.check(arena.used() == 0, "reset frees everything at once")
    suite.check(
        arena.high_water == mark, "and the high water mark survives the reset"
    )
    suite.check(counter.total() == 1, "an arena allocates once and only once")

    suite.check(arena.alloc_bytes(4096) == 0, "an oversized request is refused")
    suite.check(arena.spills == 1, "and counted as a spill")
    suite.check(arena.used() == 0, "and takes no space")

    var text = _span("keep me")
    var kept = copy_into_arena(text, arena)
    suite.check(len(kept) == 7, "a copy into the arena keeps its length")
    suite.check(equals_str(kept, "keep me"), "and its bytes")
    suite.check(
        arena.contains(Int(kept.unsafe_ptr())), "and lives in the arena"
    )

    var small = Arena(64, counter.raw())
    var used_up = small.alloc_bytes(60)
    suite.check(used_up != 0, "a small arena still allocates")
    var overflow = copy_into_arena(_span("this will not fit"), small)
    suite.check(len(overflow) == 0, "a copy that does not fit comes back empty")
    _ = small^
    _ = arena^
    counter.close()


def _check_bytes(mut suite: Suite) raises:
    suite.group("io.bytes")

    var line = _span("Content-Length: 1234")
    suite.check(equals(line, _span("Content-Length: 1234")), "equals matches")
    suite.check(
        not equals(line, _span("Content-Length: 123")), "and length counts"
    )
    suite.check(equals_str(line, "Content-Length: 1234"), "equals_str matches")
    suite.check(
        equals_ignore_case(_span("CONTENT-LENGTH"), "content-length"),
        "header names compare without case",
    )
    suite.check(
        not equals_ignore_case(_span("content_length"), "content-length"),
        "but only case, nothing else",
    )
    suite.check(starts_with(line, "Content"), "starts_with matches a prefix")
    suite.check(
        not starts_with(_span("Con"), "Content"), "and refuses a short span"
    )

    suite.check(index_of_byte(line, 58) == 14, "index_of_byte finds the colon")
    suite.check(index_of_byte(line, 63) == -1, "and reports a miss")
    suite.check(index_of(line, "Length") == 8, "index_of finds a needle")
    suite.check(index_of(line, "Width") == -1, "and reports a miss")
    suite.check(index_of(line, "") == 0, "an empty needle matches at the front")
    suite.check(
        index_of(_span("aaab"), "aab") == 1, "a false start does not stop it"
    )

    suite.check(
        equals_str(trim(_span("  value  ")), "value"), "trim takes both ends"
    )
    suite.check(
        equals_str(trim(_span("\tvalue")), "value"), "tabs count as space"
    )
    suite.check(len(trim(_span("   "))) == 0, "all space trims to nothing")
    suite.check(
        equals_str(trim(_span("a b")), "a b"), "and the middle is left alone"
    )

    suite.check(
        equals_str(slice(line, 0, 7), "Content"), "slice takes a window"
    )
    suite.check(len(slice(line, 18, 100)) == 2, "and clamps past the end")
    suite.check(len(slice(line, 100, 2)) == 0, "and past the start")

    suite.check(parse_uint(_span("0")) == 0, "zero parses")
    suite.check(parse_uint(_span("1234")) == 1234, "a length parses")
    suite.check(parse_uint(_span("")) == -1, "an empty string does not")
    suite.check(parse_uint(_span("-1")) == -1, "a sign does not")
    suite.check(parse_uint(_span("12a")) == -1, "trailing rubbish does not")
    suite.check(parse_uint(_span(" 12")) == -1, "leading space does not")
    suite.check(
        parse_uint(_span("99999999999999999999")) == -1,
        "and a number that would wrap is refused",
    )

    suite.check(parse_hex(_span("0")) == 0, "a zero chunk size parses")
    suite.check(parse_hex(_span("ff")) == 255, "lowercase hex parses")
    suite.check(parse_hex(_span("FF")) == 255, "uppercase hex parses")
    suite.check(parse_hex(_span("1a2b")) == 6699, "a longer one parses")
    suite.check(parse_hex(_span("1;x=y")) == -1, "an extension is not our job")
    suite.check(parse_hex(_span("g")) == -1, "and a non digit is refused")

    suite.check(
        to_string(_span("round trip")) == "round trip", "to_string works"
    )


def run(mut suite: Suite) raises:
    _check_mem(suite)
    _check_growth(suite)
    _check_buffer(suite)
    _check_ring(suite)
    _check_arena(suite)
    _check_bytes(suite)
