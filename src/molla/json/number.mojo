"""Numbers, in both directions, with no strtod and no locale.

Calling `strtod` from a server is two problems. It reads the C locale, so the
same request body parses differently depending on an environment variable set by
whoever started the process, which is the kind of bug that only appears in
production in one region. And it takes a NUL terminated string, so a zero copy
parser has to copy the number out of the read buffer to call it, which is an
allocation on the path issue #17 says allocates nothing.

So the conversion is here. Three paths, in the order they are tried.

An integer with no fraction and no exponent that fits in 64 bits is exact and is
kept as an integer. That is most numbers in a chat request: token counts, seeds,
indices, and the two or three settings that are whole. A double that has been
through a JSON round trip and come back as `1024.0` is a worse thing to hand a
token limit than the integer it started as.

A double whose significant digits fit in 53 bits and whose exponent is between
-22 and 22 is one multiply or one divide away from the answer. Both operands are
exact in binary, so the single rounding IEEE performs is the correctly rounded
result by definition. This is Clinger's fast path and it takes nearly every
double a real request contains.

Anything else goes through `decimal.mojo`, which holds the exact decimal
expansion and shifts it by powers of two until the mantissa falls out. Slower,
and exactly right for every input including the ones written specifically to
break converters.

## Printing

Printing goes the other way through the same struct. The exact decimal expansion
of a double is finite, so a double can be written out with no error at all, and
the only question is how few digits are enough to read back as the same double.
That is decided by expanding the two midpoints to the neighbouring doubles and
walking the three digit strings together until the digits separate, which is the
`round_shortest` port below. It gives the shortest form that round trips, so
`0.1` prints as `0.1` rather than as `0.10000000000000001`.
"""

from std.memory import Pointer, bitcast

from molla.json.decimal import Decimal

comptime NUM_INT = 0
comptime NUM_DOUBLE = 1

comptime MANT_BITS = 52
comptime EXP_BITS = 11
comptime BIAS = -1023

comptime _POW10: InlineArray[Float64, 23] = [
    1e0,
    1e1,
    1e2,
    1e3,
    1e4,
    1e5,
    1e6,
    1e7,
    1e8,
    1e9,
    1e10,
    1e11,
    1e12,
    1e13,
    1e14,
    1e15,
    1e16,
    1e17,
    1e18,
    1e19,
    1e20,
    1e21,
    1e22,
]
"""Exact in binary up to 1e22. 1e23 is not, which is why the fast path stops
there rather than at some round number."""

comptime _POWTAB: InlineArray[Int, 9] = [1, 3, 6, 9, 13, 16, 19, 23, 26]
"""How far to shift when scaling a decimal into range, indexed by how far out it
is. Roughly log2(10) times the gap, so the loop converges in a few passes
instead of one bit at a time."""


struct Number(Copyable, ImplicitlyCopyable, Movable):
    """A parsed number, still knowing whether it was written as an integer."""

    var kind: Int
    var i: Int
    var d: Float64

    def __init__(out self):
        self.kind = NUM_INT
        self.i = 0
        self.d = 0.0

    def as_double(self) -> Float64:
        if self.kind == NUM_INT:
            return Float64(self.i)
        return self.d

    def as_int(self) -> Int:
        if self.kind == NUM_INT:
            return self.i
        return Int(self.d)


def _digit(c: UInt8) -> Bool:
    return c >= 48 and c <= 57


def _bits_to_double(mant: UInt64, exp: Int, negative: Bool) -> Float64:
    var bits = mant & ((UInt64(1) << MANT_BITS) - 1)
    bits |= UInt64((exp - BIAS) & ((1 << EXP_BITS) - 1)) << MANT_BITS
    if negative:
        bits |= UInt64(1) << (MANT_BITS + EXP_BITS)
    return bitcast[DType.float64](bits)


def decimal_to_double(mut dec: Decimal) -> Float64:
    """The slow path, exact for every input.

    Scale the decimal by powers of two until it sits in [0.5, 1), which puts the
    binary exponent in hand, then shift it 53 more bits so the mantissa is the
    whole part. Denormals are the one wrinkle: below the smallest normal
    exponent the mantissa gets fewer bits, so the shift is cut short first and
    the leading bit comes out zero on its own.
    """
    var mant = UInt64(0)
    var exp = 0
    var overflow = False

    if dec.nd == 0:
        return _bits_to_double(0, BIAS, dec.negative)
    if dec.dp > 310:
        overflow = True
    elif dec.dp < -330:
        return _bits_to_double(0, BIAS, dec.negative)

    if not overflow:
        var powtab = materialize[_POWTAB]()
        while True:
            if dec.dp > 0:
                var n = 27 if dec.dp >= 9 else powtab[dec.dp]
                dec.shift(-n)
                exp += n
                continue
            if dec.dp < 0 or (dec.dp == 0 and dec.d[0] < 5):
                var n = 27 if -dec.dp >= 9 else powtab[-dec.dp]
                dec.shift(n)
                exp -= n
                continue
            break

        # The scaling loop lands in [0.5, 1) and a float mantissa lives in
        # [1, 2), so the exponent is one less than the count of shifts.
        exp -= 1

        if exp < BIAS + 1:
            # Denormal. Shift the value down to the smallest exponent there is
            # and let the mantissa lose bits off the bottom.
            var n = BIAS + 1 - exp
            dec.shift(-n)
            exp += n

        if exp - BIAS >= (1 << EXP_BITS) - 1:
            overflow = True

    if not overflow:
        dec.shift(1 + MANT_BITS)
        mant = dec.rounded_integer()
        if mant == (UInt64(2) << MANT_BITS):
            # Rounding carried into a new bit.
            mant >>= 1
            exp += 1
            if exp - BIAS >= (1 << EXP_BITS) - 1:
                overflow = True

    if overflow:
        return _bits_to_double(0, (1 << EXP_BITS) - 1 + BIAS, dec.negative)

    if (mant & (UInt64(1) << MANT_BITS)) == 0:
        # The implicit bit is not set, so this is a denormal and the exponent
        # field is the reserved zero rather than a shifted one.
        exp = BIAS
    return _bits_to_double(mant, exp, dec.negative)


def _fill_decimal[
    o: MutOrigin
](
    buf: Pointer[UInt8, o],
    start: Int,
    stop: Int,
    exp10: Int,
    negative: Bool,
    mut dec: Decimal,
):
    """Build the exact decimal from the number text, which is walked a second
    time here. That second walk only happens on the slow path, which is why the
    first walk does not pay for it."""
    dec.clear()
    dec.negative = negative
    var saw_dot = False
    var at = start
    while at < stop:
        var c = buf.unsafe_load(at)
        if c == 46:
            saw_dot = True
            dec.dp = dec.nd
            at += 1
            continue
        if not _digit(c):
            break
        if c == 48 and dec.nd == 0:
            # A leading zero is not a significant digit, it moves the point.
            dec.dp -= 1
            at += 1
            continue
        dec.push(c - 48)
        at += 1
    if not saw_dot:
        dec.dp = dec.nd
    dec.dp += exp10


def parse_number[
    o: MutOrigin
](buf: Pointer[UInt8, o], start: Int, limit: Int, mut out: Number) -> Int:
    """Parse one JSON number, returning the index after it or -1.

    Strict about the grammar rather than about what looks like a number. A
    leading zero, a leading plus, a bare `.5`, a trailing `1.`, `Infinity` and
    `NaN` are all refused, because RFC 8259 has none of them and a parser that
    accepts them is a parser that disagrees with whatever is on the other end.
    """
    var at = start
    var negative = False
    if at < limit and buf.unsafe_load(at) == 45:
        negative = True
        at += 1
    if at >= limit or not _digit(buf.unsafe_load(at)):
        return -1

    var mant = UInt64(0)
    var digits = 0
    var exp10 = 0
    var truncated = False

    if buf.unsafe_load(at) == 48:
        at += 1
        # A second digit after a leading zero is not a number, it is two.
        if at < limit and _digit(buf.unsafe_load(at)):
            return -1
    else:
        while at < limit and _digit(buf.unsafe_load(at)):
            if digits < 19:
                mant = mant * 10 + UInt64(buf.unsafe_load(at) - 48)
                digits += 1
            else:
                # Past the point where the mantissa can hold it, further integer
                # digits only move the exponent.
                exp10 += 1
                if buf.unsafe_load(at) != 48:
                    truncated = True
            at += 1

    var is_double = False
    if at < limit and buf.unsafe_load(at) == 46:
        is_double = True
        at += 1
        if at >= limit or not _digit(buf.unsafe_load(at)):
            return -1
        while at < limit and _digit(buf.unsafe_load(at)):
            if digits < 19:
                if not (
                    mant == 0 and digits == 0 and buf.unsafe_load(at) == 48
                ):
                    mant = mant * 10 + UInt64(buf.unsafe_load(at) - 48)
                    digits += 1
                exp10 -= 1
            else:
                if buf.unsafe_load(at) != 48:
                    truncated = True
            at += 1

    var number_stop = at
    if at < limit and (buf.unsafe_load(at) == 101 or buf.unsafe_load(at) == 69):
        is_double = True
        at += 1
        var exp_negative = False
        if at < limit and (
            buf.unsafe_load(at) == 43 or buf.unsafe_load(at) == 45
        ):
            exp_negative = buf.unsafe_load(at) == 45
            at += 1
        if at >= limit or not _digit(buf.unsafe_load(at)):
            return -1
        var written = 0
        while at < limit and _digit(buf.unsafe_load(at)):
            # Clamped rather than wrapped. Ten thousand is already far past
            # overflow in both directions and the conversion handles it.
            if written < 10000:
                written = written * 10 + Int(buf.unsafe_load(at) - 48)
            at += 1
        exp10 += -written if exp_negative else written

    if not is_double and exp10 == 0 and not truncated:
        # An integer, if it fits. 19 digits can still overflow, so the check is
        # on the value and not only on the count, and a non zero `exp10` here
        # means digits ran past the mantissa and the value is not in hand.
        if digits < 19 or (
            digits == 19
            and (
                mant < UInt64(9223372036854775808)
                or (negative and mant == UInt64(9223372036854775808))
            )
        ):
            out.kind = NUM_INT
            if negative and mant == UInt64(9223372036854775808):
                out.i = -9223372036854775807 - 1
            else:
                out.i = -Int(mant) if negative else Int(mant)
            out.d = 0.0
            return at

    out.kind = NUM_DOUBLE
    out.i = 0

    if (
        not truncated
        and mant < (UInt64(1) << 53)
        and exp10 >= -22
        and exp10 <= 22
    ):
        # Clinger's fast path. Both operands are exact and there is one
        # rounding, which is the correctly rounded answer by definition.
        var m = Float64(mant)
        var table = materialize[_POW10]()
        var value: Float64
        if exp10 >= 0:
            value = m * table[exp10]
        else:
            value = m / table[-exp10]
        out.d = -value if negative else value
        return at

    var dec = Decimal()
    # `_fill_decimal` recovers the point from the digits themselves, so the only
    # thing it needs from here is the explicit exponent, which it is cheaper to
    # read again than to carry out of a loop that almost never needs it.
    _fill_decimal(
        buf,
        start + 1 if negative else start,
        number_stop,
        _exponent_of(buf, number_stop, at),
        negative,
        dec,
    )
    out.d = decimal_to_double(dec)
    return at


def _exponent_of[
    o: MutOrigin
](buf: Pointer[UInt8, o], start: Int, limit: Int) -> Int:
    """Re-read the `e` part for the slow path, which is cheaper than threading
    it out of the first walk and only runs when the slow path runs."""
    var at = start
    if at >= limit:
        return 0
    if buf.unsafe_load(at) != 101 and buf.unsafe_load(at) != 69:
        return 0
    at += 1
    var negative = False
    if at < limit and (buf.unsafe_load(at) == 43 or buf.unsafe_load(at) == 45):
        negative = buf.unsafe_load(at) == 45
        at += 1
    var value = 0
    while at < limit and _digit(buf.unsafe_load(at)):
        if value < 10000:
            value = value * 10 + Int(buf.unsafe_load(at) - 48)
        at += 1
    return -value if negative else value


def double_to_decimal(value: Float64, mut dec: Decimal):
    """The exact decimal expansion of a double, then the shortest form of it
    that reads back as the same double."""
    var bits = bitcast[DType.uint64](value)
    dec.clear()
    dec.negative = (bits >> (MANT_BITS + EXP_BITS)) != 0
    var exp = Int((bits >> MANT_BITS) & ((UInt64(1) << EXP_BITS) - 1))
    var mant = bits & ((UInt64(1) << MANT_BITS) - 1)
    if exp == 0:
        # Denormal, so there is no implicit leading bit.
        exp = 1 + BIAS
    else:
        mant |= UInt64(1) << MANT_BITS
        exp += BIAS
    if mant == 0:
        return
    dec.assign(mant)
    dec.shift(exp - MANT_BITS)
    _round_shortest(dec, mant, exp)


def _round_shortest(mut dec: Decimal, mant: UInt64, exp: Int):
    """Cut the exact expansion down to the fewest digits that still read back.

    A decimal reads back as this double exactly when it falls between the
    midpoints to the two neighbouring doubles. So expand both midpoints, walk
    the three digit strings together, and stop at the first position where the
    value can be rounded either down away from the lower bound or up away from
    the upper one. This is the algorithm from Go's `strconv`, which is the
    clearest statement of it, and it is exact rather than a heuristic with a
    fallback.
    """
    if mant == 0:
        dec.nd = 0
        return

    var min_exp = BIAS + 1
    if exp > min_exp and 332 * (dec.dp - dec.nd) >= 100 * (exp - MANT_BITS):
        # Already as short as it can be.
        return

    var upper = Decimal()
    upper.assign(mant * 2 + 1)
    upper.shift(exp - MANT_BITS - 1)

    var mant_lo: UInt64
    var exp_lo: Int
    if mant > (UInt64(1) << MANT_BITS) or exp == min_exp:
        mant_lo = mant - 1
        exp_lo = exp
    else:
        # Dropping the leading bit halves the gap below, so the midpoint is
        # taken at one exponent finer.
        mant_lo = mant * 2 - 1
        exp_lo = exp - 1
    var lower = Decimal()
    lower.assign(mant_lo * 2 + 1)
    lower.shift(exp_lo - MANT_BITS - 1)

    # A bound is only reachable when the mantissa is even, since that is when
    # round to even would come back to this double rather than to the neighbour.
    var inclusive = (mant % 2) == 0

    var upper_delta = 0
    var ui = 0
    while True:
        # The three have their points in different places and upper is the
        # longest, so upper drives the walk and the other two are offset.
        var mi = ui - upper.dp + dec.dp
        if mi >= dec.nd:
            break
        var li = ui - upper.dp + lower.dp
        var l = UInt8(0)
        if li >= 0 and li < lower.nd:
            l = lower.d[li]
        var m = UInt8(0)
        if mi >= 0:
            m = dec.d[mi]
        var u = UInt8(0)
        if ui < upper.nd:
            u = upper.d[ui]

        var ok_down = l != m or (inclusive and li + 1 == lower.nd)

        if upper_delta == 0 and m + 1 < u:
            upper_delta = 2
        elif upper_delta == 0 and m != u:
            upper_delta = 1
        elif upper_delta == 1 and (m != 9 or u != 0):
            upper_delta = 2

        var ok_up = upper_delta > 0 and (
            inclusive or upper_delta > 1 or ui + 1 < upper.nd
        )

        if ok_down and ok_up:
            dec.round_to(mi + 1)
            return
        if ok_down:
            dec.round_down(mi + 1)
            return
        if ok_up:
            dec.round_up(mi + 1)
            return
        ui += 1
