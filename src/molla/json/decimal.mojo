"""An exact decimal, used in both directions between text and a double.

Converting decimal text to a binary double correctly, and printing a double back
as the shortest text that reads as the same double, are usually two separate
piles of code. They are the same problem seen from two sides, and both reduce to
holding an exact decimal expansion and shifting it by powers of two. A double is
a mantissa times a power of two, so `12.375` as a decimal shifted right three
bits is `1.546875`, exactly, with no rounding anywhere, and both directions fall
out of that one operation.

So this is one struct with a digit array, a decimal point position, and a shift.
`parse.mojo` uses it as the slow path when the fast one cannot promise a
correctly rounded answer, and `serialize.mojo` uses it to get the exact decimal
expansion of a double before rounding that to the shortest form that round
trips.

The shifting is a port of the algorithm in Go's `strconv/decimal.go`, which is
the clearest version of it in any standard library. Two differences. Shifting
left writes into scratch from the right hand end and takes the digit count from
where the write pointer stopped, rather than looking up how many digits the
result will gain in a table, which drops the table and the string comparison
that goes with it at the cost of one copy. And the digits are held as values
rather than as ASCII, because nothing here prints them directly.

## Why there is a limit and what truncation means

The array holds 800 digits. That is enough for the exact expansion of any finite
double, which is at most 767 significant digits, so printing is never truncated.
Parsing can be: a caller is allowed to send a number with ten thousand digits.
When digits are dropped, `truncated` is set, and it means the real value is
strictly greater than what the digits say. That is exactly the information
rounding needs. A value that looks like a tie but is truncated is not a tie, it
is above one, so it rounds up rather than to even. Losing that flag is how a
converter gets the last bit wrong on one input in a billion and passes every
test anyone thought to write.
"""

comptime DIGITS = 800
"""Digits held. A finite double's exact decimal expansion is at most 767, so
printing never truncates and only a hostile input makes parsing truncate."""

comptime SHIFT_MAX = 57
"""Bits per shift pass. The inner loops carry a digit and a shifted digit in a
UInt64, so the bound is what keeps `9 << k` plus a carry inside 64 bits with
room to spare. Larger shifts are split into passes."""


struct Decimal(Movable):
    """Digits, where the point sits, and whether anything was dropped."""

    var d: InlineArray[UInt8, DIGITS]
    """Significant digits as values 0 to 9, most significant first."""

    var nd: Int
    """How many of them are in use."""

    var dp: Int
    """Where the decimal point sits relative to `d[0]`. A `dp` of 2 with digits
    1, 2, 5 is 12.5, a `dp` of 0 is 0.125, and a `dp` of -1 is 0.0125."""

    var negative: Bool

    var truncated: Bool
    """Digits were dropped, so the value is strictly larger than the digits
    say. Rounding needs this and nothing else does."""

    def __init__(out self):
        self.d = InlineArray[UInt8, DIGITS](fill=0)
        self.nd = 0
        self.dp = 0
        self.negative = False
        self.truncated = False

    def clear(mut self):
        self.nd = 0
        self.dp = 0
        self.negative = False
        self.truncated = False

    def assign(mut self, value: UInt64):
        """Set to a whole number, which is where printing a double starts."""
        var tmp = InlineArray[UInt8, 24](fill=0)
        var n = 0
        var v = value
        while v > 0:
            var q = v // 10
            tmp[n] = UInt8(v - q * 10)
            n += 1
            v = q
        self.nd = 0
        self.truncated = False
        var i = n - 1
        while i >= 0:
            self.d[self.nd] = tmp[i]
            self.nd += 1
            i -= 1
        self.dp = self.nd
        self.trim()

    def copy_from(mut self, other: Self):
        for i in range(other.nd):
            self.d[i] = other.d[i]
        self.nd = other.nd
        self.dp = other.dp
        self.negative = other.negative
        self.truncated = other.truncated

    def is_zero(self) -> Bool:
        return self.nd == 0

    def push(mut self, digit: UInt8):
        """Append one significant digit, dropping it if the array is full.

        A dropped digit that is not zero sets `truncated`, and a dropped zero
        does not, because trailing zeros carry no information about whether the
        value sits above a tie.
        """
        if self.nd < DIGITS:
            self.d[self.nd] = digit
            self.nd += 1
        elif digit != 0:
            self.truncated = True

    def trim(mut self):
        """Drop trailing zeros, which are never significant here."""
        while self.nd > 0 and self.d[self.nd - 1] == 0:
            self.nd -= 1
        if self.nd == 0:
            self.dp = 0

    def _shift_right(mut self, k: Int):
        """Divide by two to the k, exactly, with digits produced in place."""
        var kk = UInt64(k)
        var r = 0
        var w = 0
        var n = UInt64(0)
        # Take in leading digits until there is something to divide. Each digit
        # taken without producing one moves the point left by one.
        while (n >> kk) == 0:
            if r >= self.nd:
                if n == 0:
                    self.nd = 0
                    self.dp = 0
                    return
                while (n >> kk) == 0:
                    n *= 10
                    r += 1
                break
            n = n * 10 + UInt64(self.d[r])
            r += 1

        self.dp -= r - 1
        var mask = (UInt64(1) << kk) - 1
        while r < self.nd:
            var c = UInt64(self.d[r])
            var digit = n >> kk
            n &= mask
            self.d[w] = UInt8(digit)
            w += 1
            n = n * 10 + c
            r += 1
        # Whatever is left of the remainder keeps producing digits.
        while n > 0:
            var digit = n >> kk
            n &= mask
            if w < DIGITS:
                self.d[w] = UInt8(digit)
                w += 1
            elif digit > 0:
                self.truncated = True
            n *= 10
        self.nd = w
        self.trim()

    def _shift_left(mut self, k: Int):
        """Multiply by two to the k, exactly.

        Written into scratch from the right, so the number of digits gained
        comes out of where the write pointer stopped instead of out of a table.
        The scratch is a little larger than the digit array because one pass can
        add at most eighteen digits at this shift width.
        """
        var tmp = InlineArray[UInt8, DIGITS + 32](fill=0)
        var kk = UInt64(k)
        var w = DIGITS + 32
        var n = UInt64(0)
        var r = self.nd - 1
        while r >= 0:
            n += UInt64(self.d[r]) << kk
            var q = n // 10
            w -= 1
            tmp[w] = UInt8(n - q * 10)
            n = q
            r -= 1
        while n > 0:
            var q = n // 10
            w -= 1
            tmp[w] = UInt8(n - q * 10)
            n = q

        var produced = DIGITS + 32 - w
        self.dp += produced - self.nd
        var keep = produced
        if keep > DIGITS:
            # The tail falls off the end. Only a non zero digit among the
            # dropped ones changes what rounding should do.
            for i in range(DIGITS, produced):
                if tmp[w + i] != 0:
                    self.truncated = True
                    break
            keep = DIGITS
        for i in range(keep):
            self.d[i] = tmp[w + i]
        self.nd = keep
        self.trim()

    def shift(mut self, k: Int):
        """Multiply by two to the k, for a k of either sign and any size."""
        if self.nd == 0:
            return
        var left = k
        while left > SHIFT_MAX:
            self._shift_left(SHIFT_MAX)
            left -= SHIFT_MAX
        while left < -SHIFT_MAX:
            self._shift_right(SHIFT_MAX)
            left += SHIFT_MAX
            if self.nd == 0:
                return
        if left > 0:
            self._shift_left(left)
        elif left < 0:
            self._shift_right(-left)

    def _should_round_up(self, at: Int) -> Bool:
        """Whether rounding away everything from `at` onwards rounds up.

        Ties go to even, except that a truncated value is not a tie: digits were
        dropped, so the real number sits above the halfway point.
        """
        if at < 0 or at >= self.nd:
            return False
        if self.d[at] == 5 and at + 1 == self.nd:
            if self.truncated:
                return True
            return at > 0 and (self.d[at - 1] & 1) != 0
        return self.d[at] >= 5

    def round_down(mut self, digits: Int):
        """Keep `digits` significant digits and drop the rest."""
        if digits < 0 or digits >= self.nd:
            return
        self.nd = digits
        self.truncated = False
        self.trim()

    def round_up(mut self, digits: Int):
        """Keep `digits` significant digits and add one to the last."""
        if digits < 0 or digits >= self.nd:
            return
        # The carry is a walk left over the nines.
        var i = digits - 1
        while i >= 0:
            if self.d[i] < 9:
                self.d[i] += 1
                self.nd = i + 1
                self.truncated = False
                self.trim()
                return
            i -= 1
        # All nines, so the result is a one and the point moved.
        self.d[0] = 1
        self.nd = 1
        self.dp += 1
        self.truncated = False

    def round_to(mut self, digits: Int):
        """Keep `digits` significant digits, rounding the rest to nearest."""
        if digits < 0 or digits >= self.nd:
            return
        if self._should_round_up(digits):
            self.round_up(digits)
        else:
            self.round_down(digits)

    def rounded_integer(self) -> UInt64:
        """The value rounded to a whole number, which is how a mantissa comes
        out once the decimal has been shifted so the point sits after it."""
        if self.dp > 20:
            return UInt64(0xFFFFFFFFFFFFFFFF)
        var i = 0
        var n = UInt64(0)
        while i < self.dp and i < self.nd:
            n = n * 10 + UInt64(self.d[i])
            i += 1
        while i < self.dp:
            n *= 10
            i += 1
        if self._should_round_up(self.dp):
            n += 1
        return n
