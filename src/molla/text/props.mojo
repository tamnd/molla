"""Unicode character properties, decoded from the generated tables.

A `Unicode` is the tables in memory. Building one decodes about eleven thousand
varints and costs well under a millisecond, and everything that needs a
property holds a reference to one rather than reaching for a global, because
Mojo 1.0 has no global to reach for. A server builds one when it loads a
tokenizer and keeps it.

The properties here are the ones a tokenizer actually asks for: the general
category, the canonical combining class, the decompositions, the compositions
that are allowed to run backwards, and the lowercase mapping. Whitespace and
the regex shorthand classes are written out below rather than generated,
because they are short and a list you can read is a list you can check.
"""

from molla.text.tables import (
    CATEGORY_COUNT,
    CATEGORY_DATA,
    COMBINING_COUNT,
    COMBINING_DATA,
    DECOMPOSITION_COUNT,
    DECOMPOSITION_DATA,
    LOWERCASE_COUNT,
    LOWERCASE_DATA,
)

comptime CAT_LU = 0
comptime CAT_LL = 1
comptime CAT_LT = 2
comptime CAT_LM = 3
comptime CAT_LO = 4
comptime CAT_MN = 5
comptime CAT_MC = 6
comptime CAT_ME = 7
comptime CAT_ND = 8
comptime CAT_NL = 9
comptime CAT_NO = 10
comptime CAT_PC = 11
comptime CAT_PD = 12
comptime CAT_PS = 13
comptime CAT_PE = 14
comptime CAT_PI = 15
comptime CAT_PF = 16
comptime CAT_PO = 17
comptime CAT_SM = 18
comptime CAT_SC = 19
comptime CAT_SK = 20
comptime CAT_SO = 21
comptime CAT_ZS = 22
comptime CAT_ZL = 23
comptime CAT_ZP = 24
comptime CAT_CC = 25
comptime CAT_CF = 26
comptime CAT_CS = 27
comptime CAT_CO = 28
comptime CAT_CN = 29
"""Unassigned. Anything not in the table is this."""

comptime CATEGORY_NAMES = (
    "LuLlLtLmLoMnMcMeNdNlNoPcPdPsPePiPfPoSmScSkSoZsZlZpCcCfCsCoCn"
)
"""The two letter names in table order, so a category index can be printed."""

comptime HANGUL_S_BASE = 0xAC00
comptime HANGUL_L_BASE = 0x1100
comptime HANGUL_V_BASE = 0x1161
comptime HANGUL_T_BASE = 0x11A7
comptime HANGUL_L_COUNT = 19
comptime HANGUL_V_COUNT = 21
comptime HANGUL_T_COUNT = 28
comptime HANGUL_N_COUNT = 588
comptime HANGUL_S_COUNT = 11172


def category_name(index: Int) -> String:
    """The two letter name of a category index, for a message or a report."""
    if index < 0 or index > CAT_CN:
        return String("??")
    var bytes = CATEGORY_NAMES.as_bytes()
    var out = List[UInt8]()
    out.append(bytes[index * 2])
    out.append(bytes[index * 2 + 1])
    return String(StringSpan(unsafe_from_utf8=out))


def category_index(name: StringSpan) -> Int:
    """The index of a two letter category name, or minus one.

    A one letter name is not accepted here. The regex layer wants `\\p{L}` to
    mean five categories rather than one, so it expands the major categories
    itself and this stays a straight lookup.
    """
    if name.byte_length() != 2:
        return -1
    var bytes = CATEGORY_NAMES.as_bytes()
    var p = name.unsafe_ptr()
    for i in range(CAT_CN + 1):
        if bytes[i * 2] == p.unsafe_load(0) and bytes[
            i * 2 + 1
        ] == p.unsafe_load(1):
            return i
    return -1


def is_whitespace(cp: Int) -> Bool:
    """The White_Space property, which is what a regex `\\s` means.

    Written out rather than generated. There are twenty five of them, the list
    has not changed since Unicode 4, and a reader can check this against the
    standard in about a minute. Note what is not here: the four ASCII
    separators at 0x1C to 0x1F, which Python calls space and Unicode does not.
    """
    if cp < 0x2000:
        if cp >= 0x09 and cp <= 0x0D:
            return True
        return cp == 0x20 or cp == 0x85 or cp == 0xA0 or cp == 0x1680
    if cp <= 0x200A:
        return True
    return (
        cp == 0x2028
        or cp == 0x2029
        or cp == 0x202F
        or cp == 0x205F
        or cp == 0x3000
    )


def is_ascii_whitespace(cp: Int) -> Bool:
    """Space, tab, newline, carriage return, form feed, vertical tab."""
    return cp == 0x20 or (cp >= 0x09 and cp <= 0x0D)


def _digit(b: UInt8) -> Int:
    """The value of one base 64 digit, in the standard alphabet."""
    if b >= 65 and b <= 90:
        return Int(b) - 65
    if b >= 97 and b <= 122:
        return Int(b) - 97 + 26
    if b >= 48 and b <= 57:
        return Int(b) - 48 + 52
    if b == 43:
        return 62
    return 63


def _varint(data: Span[UInt8, _], mut at: Int) -> Int:
    """Read one varint: five payload bits per digit, 0x20 says another follows.
    """
    var value = 0
    var shift = 0
    while at < len(data):
        var d = _digit(data[at])
        at += 1
        value |= (d & 0x1F) << shift
        if (d & 0x20) == 0:
            return value
        shift += 5
    return value


comptime UNICODE_MAX = 0x110000
comptime BMP_MAX = 0x10000


struct Unicode(Movable):
    """The Unicode tables, decoded once.

    The three range tables are parallel lists searched by bisection. The two
    mapping tables are a sorted list of code points and an arena of the code
    points they map to. The composition map is built here rather than
    generated, by running the canonical decompositions backwards and dropping
    the ones the exclusion bit says do not compose.

    On top of the ranges there are four flat tables, built at startup, that
    turn the four hot lookups into one load each. Bisection over four thousand
    category ranges is twelve unpredictable branches, and a tokenizer asks for
    the category of every character it reads, several times over, so those
    twelve branches were most of the cost of encoding a string. The flat
    tables cost about three megabytes and they are worth it here.
    """

    var cat_start: List[Int]
    var cat_end: List[Int]
    var cat_value: List[UInt8]

    var ccc_start: List[Int]
    var ccc_end: List[Int]
    var ccc_value: List[UInt8]

    var cat_direct: List[UInt8]
    """Category per code point, the whole range, `CAT_CN` where unassigned."""

    var ccc_direct: List[UInt8]
    """Combining class per code point, the whole range, zero for a starter."""

    var lower_index: List[Int]
    """Index into `lower_cp` plus one, or zero, for the basic plane only."""

    var decomp_index: List[Int]
    """Index into `decomp_cp` plus one, or zero, for the basic plane only."""

    var decomp_cp: List[Int]
    var decomp_header: List[Int]
    var decomp_at: List[Int]
    var decomp_points: List[Int]

    var lower_cp: List[Int]
    var lower_at: List[Int]
    var lower_points: List[Int]

    var comp_key: List[Int]
    var comp_value: List[Int]
    var comp_mask: Int

    var canonical_floor: Int
    """The lowest code point canonical normalization can do anything to.

    Below it every character decomposes to itself, has a combining class of
    zero, and is not a Hangul syllable, so a run of characters under this
    value is already in NFC and NFD both. Worked out from the tables rather
    than written down, so it stays right when the tables are regenerated.
    """

    var compatibility_floor: Int
    """The same for the K forms, which take a wider set of characters apart."""

    def __init__(out self):
        self.cat_start = List[Int]()
        self.cat_end = List[Int]()
        self.cat_value = List[UInt8]()
        self.ccc_start = List[Int]()
        self.ccc_end = List[Int]()
        self.ccc_value = List[UInt8]()
        self.cat_direct = List[UInt8]()
        self.ccc_direct = List[UInt8]()
        self.lower_index = List[Int]()
        self.decomp_index = List[Int]()
        self.decomp_cp = List[Int]()
        self.decomp_header = List[Int]()
        self.decomp_at = List[Int]()
        self.decomp_points = List[Int]()
        self.lower_cp = List[Int]()
        self.lower_at = List[Int]()
        self.lower_points = List[Int]()
        self.comp_key = List[Int]()
        self.comp_value = List[Int]()
        self.comp_mask = 0
        self.canonical_floor = 0
        self.compatibility_floor = 0

        self._read_ranges(
            CATEGORY_DATA.as_bytes(),
            CATEGORY_COUNT,
            self.cat_start,
            self.cat_end,
            self.cat_value,
        )
        self._read_ranges(
            COMBINING_DATA.as_bytes(),
            COMBINING_COUNT,
            self.ccc_start,
            self.ccc_end,
            self.ccc_value,
        )
        self._read_decompositions()
        self._read_lowercase()
        self._build_compositions()
        self._build_direct()

    def _build_direct(mut self):
        """Flatten the ranges and the two mapping tables into direct lookups."""
        self.cat_direct = List[UInt8](length=UNICODE_MAX, fill=UInt8(CAT_CN))
        for i in range(len(self.cat_start)):
            var end = self.cat_end[i]
            if end >= UNICODE_MAX:
                end = UNICODE_MAX - 1
            for cp in range(self.cat_start[i], end + 1):
                self.cat_direct[cp] = self.cat_value[i]

        self.ccc_direct = List[UInt8](length=UNICODE_MAX, fill=UInt8(0))
        for i in range(len(self.ccc_start)):
            var end = self.ccc_end[i]
            if end >= UNICODE_MAX:
                end = UNICODE_MAX - 1
            for cp in range(self.ccc_start[i], end + 1):
                self.ccc_direct[cp] = self.ccc_value[i]

        # The two mapping tables only get a flat index for the basic plane.
        # Everything they hold above it is a handful of recent scripts and some
        # mathematical alphabets, and bisection is fine for those.
        self.lower_index = List[Int](length=BMP_MAX, fill=0)
        for i in range(len(self.lower_cp)):
            if self.lower_cp[i] < BMP_MAX:
                self.lower_index[self.lower_cp[i]] = i + 1

        self.decomp_index = List[Int](length=BMP_MAX, fill=0)
        for i in range(len(self.decomp_cp)):
            if self.decomp_cp[i] < BMP_MAX:
                self.decomp_index[self.decomp_cp[i]] = i + 1

        self.canonical_floor = BMP_MAX
        self.compatibility_floor = BMP_MAX
        for cp in range(BMP_MAX):
            var found = self._decomposition_of(cp)
            if found < 0 and self.ccc_direct[cp] == 0:
                continue
            if self.compatibility_floor == BMP_MAX:
                self.compatibility_floor = cp
            if found < 0 or (self.decomp_header[found] & 1) == 0:
                self.canonical_floor = cp
                break
        if self.canonical_floor > HANGUL_S_BASE:
            self.canonical_floor = HANGUL_S_BASE
        if self.compatibility_floor > HANGUL_S_BASE:
            self.compatibility_floor = HANGUL_S_BASE

        # A character below the floor could also turn up as the second half of
        # a composition, and then the character in front of it is not safe to
        # hand back untouched either. No pair in the current tables does that,
        # and if one ever appears the floor drops to meet it rather than the
        # fast path going quietly wrong.
        for i in range(len(self.decomp_cp)):
            var header = self.decomp_header[i]
            if (header & 3) != 0 or (header >> 2) != 2:
                continue
            var second = self.decomp_points[self.decomp_at[i] + 1]
            if second < self.canonical_floor:
                self.canonical_floor = second
            if second < self.compatibility_floor:
                self.compatibility_floor = second

    @staticmethod
    def _read_ranges(
        data: Span[UInt8, _],
        count: Int,
        mut starts: List[Int],
        mut ends: List[Int],
        mut values: List[UInt8],
    ):
        starts.reserve(count)
        ends.reserve(count)
        values.reserve(count)
        var at = 0
        var previous = -1
        for _ in range(count):
            var start = previous + 1 + _varint(data, at)
            var end = start + _varint(data, at)
            var value = _varint(data, at)
            starts.append(start)
            ends.append(end)
            values.append(UInt8(value))
            previous = end

    def _read_decompositions(mut self):
        var data = DECOMPOSITION_DATA.as_bytes()
        self.decomp_cp.reserve(DECOMPOSITION_COUNT)
        self.decomp_header.reserve(DECOMPOSITION_COUNT)
        self.decomp_at.reserve(DECOMPOSITION_COUNT + 1)
        var at = 0
        var previous = -1
        for _ in range(DECOMPOSITION_COUNT):
            var cp = previous + 1 + _varint(data, at)
            var header = _varint(data, at)
            self.decomp_cp.append(cp)
            self.decomp_header.append(header)
            self.decomp_at.append(len(self.decomp_points))
            for _ in range(header >> 2):
                self.decomp_points.append(_varint(data, at))
            previous = cp
        self.decomp_at.append(len(self.decomp_points))

    def _read_lowercase(mut self):
        var data = LOWERCASE_DATA.as_bytes()
        self.lower_cp.reserve(LOWERCASE_COUNT)
        self.lower_at.reserve(LOWERCASE_COUNT + 1)
        var at = 0
        var previous = -1
        for _ in range(LOWERCASE_COUNT):
            var cp = previous + 1 + _varint(data, at)
            var count = _varint(data, at)
            self.lower_cp.append(cp)
            self.lower_at.append(len(self.lower_points))
            for _ in range(count):
                self.lower_points.append(_varint(data, at))
            previous = cp
        self.lower_at.append(len(self.lower_points))

    def _build_compositions(mut self):
        """Every canonical two character decomposition that composes back.

        Open addressed, power of two, kept under half full. The key packs the
        two code points into one integer, which fits because a code point is
        twenty one bits and there are sixty four to put them in.
        """
        var size = 64
        while size < DECOMPOSITION_COUNT * 2:
            size *= 2
        self.comp_mask = size - 1
        for _ in range(size):
            self.comp_key.append(-1)
            self.comp_value.append(0)

        for i in range(len(self.decomp_cp)):
            var header = self.decomp_header[i]
            if (header & 1) != 0 or (header & 2) != 0:
                continue
            if (header >> 2) != 2:
                continue
            var at = self.decomp_at[i]
            var key = (self.decomp_points[at] << 21) | self.decomp_points[
                at + 1
            ]
            var slot = self._slot(key)
            self.comp_key[slot] = key
            self.comp_value[slot] = self.decomp_cp[i]

    def _slot(self, key: Int) -> Int:
        """Where a key lives, or the first free slot after where it would."""
        # Knuth's multiplicative hash on the packed pair. The keys are dense in
        # the low bits and clustered by script, so the low bits on their own
        # collide badly.
        var h = (key * 0x9E3779B1) & 0x7FFFFFFFFFFFFFFF
        var slot = (h >> 13) & self.comp_mask
        while self.comp_key[slot] != -1 and self.comp_key[slot] != key:
            slot = (slot + 1) & self.comp_mask
        return slot

    @staticmethod
    def _find(starts: List[Int], ends: List[Int], cp: Int) -> Int:
        """The range containing `cp`, or minus one."""
        var low = 0
        var high = len(starts) - 1
        while low <= high:
            var mid = (low + high) >> 1
            if cp < starts[mid]:
                high = mid - 1
            elif cp > ends[mid]:
                low = mid + 1
            else:
                return mid
        return -1

    def category(self, cp: Int) -> Int:
        """The general category of `cp`, `CAT_CN` if it is not assigned."""
        if cp < 0 or cp >= UNICODE_MAX:
            return CAT_CN
        return Int(self.cat_direct[cp])

    def combining(self, cp: Int) -> Int:
        """The canonical combining class of `cp`, zero for a starter."""
        if cp < 0 or cp >= UNICODE_MAX:
            return 0
        return Int(self.ccc_direct[cp])

    def is_letter(self, cp: Int) -> Bool:
        var c = self.category(cp)
        return c >= CAT_LU and c <= CAT_LO

    def is_mark(self, cp: Int) -> Bool:
        var c = self.category(cp)
        return c >= CAT_MN and c <= CAT_ME

    def is_number(self, cp: Int) -> Bool:
        var c = self.category(cp)
        return c >= CAT_ND and c <= CAT_NO

    def is_punctuation(self, cp: Int) -> Bool:
        var c = self.category(cp)
        return c >= CAT_PC and c <= CAT_PO

    def is_symbol(self, cp: Int) -> Bool:
        var c = self.category(cp)
        return c >= CAT_SM and c <= CAT_SO

    def is_separator(self, cp: Int) -> Bool:
        var c = self.category(cp)
        return c >= CAT_ZS and c <= CAT_ZP

    def is_other(self, cp: Int) -> Bool:
        var c = self.category(cp)
        return c >= CAT_CC and c <= CAT_CN

    def is_word(self, cp: Int) -> Bool:
        """What a regex `\\w` means: alphabetic, a mark, a digit, a connector
        or a joiner.

        Written the way the Rust regex crate writes it, since that is the
        engine the tokenizer files were tested against. Alphabetic is not the
        letter categories. It is wider by the letter numbers, which is where
        the ideographic zero and the Roman numerals live, and by the circled
        and squared Latin letters, which are symbols. It is also narrower by
        about a thousand marks, but `\\w` takes every mark anyway so that end
        does not matter. The two extra pieces do, because a pre-tokenizer that
        calls the ideographic zero a symbol cuts a Chinese word in half.
        """
        if cp == 0x200C or cp == 0x200D:
            return True
        var c = self.category(cp)
        if c >= CAT_LU and c <= CAT_LO:
            return True
        if c >= CAT_MN and c <= CAT_ME:
            return True
        if c == CAT_ND or c == CAT_PC or c == CAT_NL:
            return True
        if c != CAT_SO:
            return False
        return (
            (cp >= 0x24B6 and cp <= 0x24E9)
            or (cp >= 0x1F130 and cp <= 0x1F149)
            or (cp >= 0x1F150 and cp <= 0x1F169)
            or (cp >= 0x1F170 and cp <= 0x1F189)
        )

    def decomposition(self, cp: Int, compatibility: Bool) -> Int:
        """The index of the decomposition of `cp`, or minus one.

        Compatibility decompositions are only used when asked for.
        Canonical ones are used either way, because NFKD is NFD plus more.
        Hangul is not in the table and callers handle it themselves.
        """
        var found = self._decomposition_of(cp)
        if found < 0:
            return -1
        if not compatibility and (self.decomp_header[found] & 1) != 0:
            return -1
        return found

    def _decomposition_of(self, cp: Int) -> Int:
        if cp >= 0 and cp < BMP_MAX:
            return self.decomp_index[cp] - 1
        var low = 0
        var high = len(self.decomp_cp) - 1
        while low <= high:
            var mid = (low + high) >> 1
            if cp < self.decomp_cp[mid]:
                high = mid - 1
            elif cp > self.decomp_cp[mid]:
                low = mid + 1
            else:
                return mid
        return -1

    def _lowercase_of(self, cp: Int) -> Int:
        if cp >= 0 and cp < BMP_MAX:
            return self.lower_index[cp] - 1
        var low = 0
        var high = len(self.lower_cp) - 1
        while low <= high:
            var mid = (low + high) >> 1
            if cp < self.lower_cp[mid]:
                high = mid - 1
            elif cp > self.lower_cp[mid]:
                low = mid + 1
            else:
                return mid
        return -1

    def decomposition_length(self, index: Int) -> Int:
        return self.decomp_header[index] >> 2

    def decomposition_at(self, index: Int, offset: Int) -> Int:
        return self.decomp_points[self.decomp_at[index] + offset]

    def compose(self, first: Int, second: Int) -> Int:
        """The single character `first` and `second` compose to, or zero.

        Hangul composes by arithmetic. Everything else is the table, which
        already had the exclusions taken out of it when it was built.
        """
        var l_index = first - HANGUL_L_BASE
        if l_index >= 0 and l_index < HANGUL_L_COUNT:
            var v_index = second - HANGUL_V_BASE
            if v_index >= 0 and v_index < HANGUL_V_COUNT:
                return (
                    HANGUL_S_BASE
                    + (l_index * HANGUL_V_COUNT + v_index) * HANGUL_T_COUNT
                )
        var s_index = first - HANGUL_S_BASE
        if s_index >= 0 and s_index < HANGUL_S_COUNT:
            if s_index % HANGUL_T_COUNT == 0:
                var t_index = second - HANGUL_T_BASE
                if t_index > 0 and t_index < HANGUL_T_COUNT:
                    return first + t_index
            return 0

        var slot = self._slot((first << 21) | second)
        if self.comp_key[slot] == -1:
            return 0
        return self.comp_value[slot]

    def lowercase(self, cp: Int, mut out: List[Int]):
        """Append the lowercase of `cp`, which is sometimes two characters.

        Capital I with a dot above lowercases to i and a combining dot, and a
        normalizer that returned one character would be deleting the dot rather
        than lowercasing the letter.
        """
        var at = self._lowercase_of(cp)
        if at < 0:
            out.append(cp)
            return
        for i in range(self.lower_at[at], self.lower_at[at + 1]):
            out.append(self.lower_points[i])

    def lowercase_one(self, cp: Int) -> Int:
        """The lowercase of `cp` when it is a single character, else `cp`.

        The regex layer uses this. A case insensitive literal is one character
        against one character, so the two character mappings cannot take part
        and leaving them alone is the honest answer.
        """
        var at = self._lowercase_of(cp)
        if at < 0:
            return cp
        if self.lower_at[at + 1] - self.lower_at[at] != 1:
            return cp
        return self.lower_points[self.lower_at[at]]

    def uppercase_one(self, cp: Int) -> Int:
        """The character whose lowercase is `cp`, or `cp` when there is none.

        There is no uppercase table. Case insensitive matching needs to get
        from a lowercase letter back to its uppercase, and building that from
        the lowercase table costs one pass at startup, so this walks instead
        and the regex compiler calls it once per literal rather than once per
        byte of input.
        """
        for i in range(len(self.lower_cp)):
            if self.lower_at[i + 1] - self.lower_at[i] != 1:
                continue
            if self.lower_points[self.lower_at[i]] == cp:
                return self.lower_cp[i]
        return cp
