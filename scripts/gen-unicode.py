#!/usr/bin/env python3
"""Generate src/molla/text/tables.mojo from the Unicode database.

Run it with the Python that ships the Unicode version you want:

    python3 scripts/gen-unicode.py

It writes five tables and nothing else. General category ranges, canonical
combining class ranges, character decompositions, the lowercase mapping, and
the grapheme cluster break property. Everything else the text layer needs is
derived from those at runtime, and the properties that are short enough to read
are written out by hand in props.mojo rather than generated, because a list of
twenty five space characters is easier to check by eye than by diff.

The grapheme table is the one that needs the network. Python's `unicodedata`
does not carry Grapheme_Cluster_Break, Extended_Pictographic or
Indic_Conjunct_Break, so those three come from the Unicode files themselves,
fetched at the version the running Python reports. That keeps all five tables
on one version, and it means this script does not run offline.

The tables are strings rather than arrays. A Mojo array literal with twelve
thousand elements in it is a long compile for data that never changes, and a
string of the same data decodes in well under a millisecond. The encoding is a
base 64 varint: five payload bits per character, low to high, with 0x20 set on
every character but the last of a number, and the digits are the standard base
64 alphabet so the result is greppable and diffable and contains nothing that
needs escaping.

Hangul is not in the decomposition table. Eleven thousand of the thirteen
thousand canonical decompositions in the database are Hangul syllables, and
they follow a formula that is four lines of arithmetic, so props.mojo does them
that way and this script drops them.
"""

import sys
import unicodedata as ud
import urllib.request

MAX = 0x110000
HANGUL_BASE = 0xAC00
HANGUL_COUNT = 11172

DIGITS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

CATEGORIES = [
    "Lu", "Ll", "Lt", "Lm", "Lo",
    "Mn", "Mc", "Me",
    "Nd", "Nl", "No",
    "Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po",
    "Sm", "Sc", "Sk", "So",
    "Zs", "Zl", "Zp",
    "Cc", "Cf", "Cs", "Co", "Cn",
]


def varint(n):
    """Encode one non negative number as base 64 varint digits."""
    if n < 0:
        raise ValueError("varint of a negative number: %d" % n)
    out = []
    while True:
        part = n & 0x1F
        n >>= 5
        out.append(DIGITS[part | (0x20 if n else 0)])
        if not n:
            return "".join(out)


def ranges(value_of):
    """Walk the whole code space and yield (start, end, value) runs."""
    start = 0
    current = value_of(0)
    for cp in range(1, MAX):
        value = value_of(cp)
        if value != current:
            yield start, cp - 1, current
            start = cp
            current = value
    yield start, MAX - 1, current


def encode_ranges(runs):
    """Ranges as deltas from the end of the range before, so most are tiny."""
    out = []
    previous = -1
    for start, end, value in runs:
        out.append(varint(start - previous - 1))
        out.append(varint(end - start))
        out.append(varint(value))
        previous = end
    return "".join(out)


def category_table():
    def value_of(cp):
        return CATEGORIES.index(ud.category(chr(cp)))

    unassigned = CATEGORIES.index("Cn")
    runs = [r for r in ranges(value_of) if r[2] != unassigned]
    return encode_ranges(runs), len(runs)


def combining_table():
    def value_of(cp):
        return ud.combining(chr(cp))

    runs = [r for r in ranges(value_of) if r[2] != 0]
    return encode_ranges(runs), len(runs)


def decomposition_table():
    """Every decomposition, canonical and compatibility, minus Hangul.

    Each entry is the code point as a delta from the one before, then a header
    holding the length and two flags, then the code points it decomposes to.
    The compatibility flag says which of the two normal forms uses it. The
    excluded flag says the pair does not compose back, which is the composition
    exclusion list, worked out here by asking the database rather than parsing
    DerivedNormalizationProps: if NFC of the two halves is not the character
    itself then the character is excluded, and that is the definition.
    """
    out = []
    previous = -1
    count = 0
    for cp in range(MAX):
        if HANGUL_BASE <= cp < HANGUL_BASE + HANGUL_COUNT:
            continue
        raw = ud.decomposition(chr(cp))
        if not raw:
            continue
        parts = raw.split()
        compat = parts[0].startswith("<")
        if compat:
            parts = parts[1:]
        points = [int(p, 16) for p in parts]

        excluded = True
        if not compat and len(points) == 2:
            joined = "".join(chr(p) for p in points)
            excluded = ud.normalize("NFC", joined) != chr(cp)

        header = len(points) << 2
        if compat:
            header |= 1
        if excluded:
            header |= 2

        out.append(varint(cp - previous - 1))
        out.append(varint(header))
        for p in points:
            out.append(varint(p))
        previous = cp
        count += 1
    return "".join(out), count


def lowercase_table():
    """The lowercase mapping, full rather than simple.

    Full means the one character that lowercases to two stays two: capital I
    with a dot above is i followed by a combining dot, and a normalizer that
    dropped the dot would be changing the text rather than lowercasing it.
    """
    out = []
    previous = -1
    count = 0
    for cp in range(MAX):
        ch = chr(cp)
        low = ch.lower()
        if low == ch:
            continue
        points = [ord(c) for c in low]
        out.append(varint(cp - previous - 1))
        out.append(varint(len(points)))
        for p in points:
            out.append(varint(p))
        previous = cp
        count += 1
    return "".join(out), count


BASE = "https://www.unicode.org/Public/%s/ucd/%s"

GRAPHEME_CLASSES = [
    "Other", "CR", "LF", "Control", "Extend", "ZWJ", "Regional_Indicator",
    "Prepend", "SpacingMark", "L", "V", "T", "LV", "LVT",
]

INDIC_CLASSES = ["", "Consonant", "Extend", "Linker"]


def ucd(name):
    """One file out of the Unicode database at the version Python carries."""
    url = BASE % (ud.unidata_version, name)
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def properties(text):
    """A UCD property file as a list of (fields, first, last)."""
    out = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split(";")]
        span = fields[0]
        if ".." in span:
            first, last = span.split("..")
        else:
            first = last = span
        out.append((fields[1:], int(first, 16), int(last, 16)))
    return out


def grapheme_table():
    """Everything the grapheme cluster boundary rules need, packed per range.

    Four low bits for the break class, then one bit for Extended_Pictographic,
    then two bits for Indic_Conjunct_Break. They go in one table because the
    rules read all three at once and three bisections per character to answer
    one question is two too many.

    Extended_Pictographic is only needed for GB11 and Indic_Conjunct_Break only
    for GB9c, and neither rule can change what the SentencePiece charsmap does,
    since both need a cluster longer than the five bytes that lookup is capped
    at. They are here because a grapheme iterator that is right for the easy
    cases and wrong for emoji is worse than not having one.
    """
    breaks = bytearray(MAX)
    for fields, first, last in properties(ucd("auxiliary/GraphemeBreakProperty.txt")):
        if fields[0] not in GRAPHEME_CLASSES:
            raise ValueError("unknown break class %s" % fields[0])
        value = GRAPHEME_CLASSES.index(fields[0])
        for cp in range(first, last + 1):
            breaks[cp] = value

    for fields, first, last in properties(ucd("emoji/emoji-data.txt")):
        if fields[0] != "Extended_Pictographic":
            continue
        for cp in range(first, last + 1):
            breaks[cp] |= 1 << 4

    for fields, first, last in properties(ucd("DerivedCoreProperties.txt")):
        if fields[0] != "InCB":
            continue
        value = INDIC_CLASSES.index(fields[1])
        for cp in range(first, last + 1):
            breaks[cp] |= value << 5

    def value_of(cp):
        return breaks[cp]

    runs = [r for r in ranges(value_of) if r[2] != 0]
    return encode_ranges(runs), len(runs)


HEADER = '''"""Unicode tables, generated by scripts/gen-unicode.py.

Do not edit this by hand. Regenerate it, and say in the commit which Unicode
version the Python that generated it was carrying. This one is {version}.

The encoding is a base 64 varint, five payload bits per digit, low bits first,
0x20 set on every digit but the last of a number. The reader is in props.mojo
and it is fifteen lines. Range tables hold a delta from the end of the range
before, then the length minus one, then the value. Mapping tables hold a delta
from the code point before, then a header, then the code points.

Hangul syllables are not in the decomposition table. They decompose by formula
and props.mojo does it that way.
"""

comptime UNICODE_VERSION = "{version}"
'''


def main():
    category, category_count = category_table()
    combining, combining_count = combining_table()
    decomposition, decomposition_count = decomposition_table()
    lowercase, lowercase_count = lowercase_table()
    grapheme, grapheme_count = grapheme_table()

    with open("src/molla/text/tables.mojo", "w") as f:
        f.write(HEADER.format(version=ud.unidata_version))
        f.write('\n"""The Unicode version of the database these came out of."""\n')

        f.write("\ncomptime CATEGORY_COUNT = %d\n" % category_count)
        f.write('"""Ranges in the general category table. Unassigned is not in it."""\n')
        f.write("\ncomptime CATEGORY_DATA = \"%s\"\n" % category)
        f.write('"""Start, length and category index per range, as varints."""\n')

        f.write("\ncomptime COMBINING_COUNT = %d\n" % combining_count)
        f.write('"""Ranges in the combining class table. Class zero is not in it."""\n')
        f.write("\ncomptime COMBINING_DATA = \"%s\"\n" % combining)
        f.write('"""Start, length and combining class per range, as varints."""\n')

        f.write("\ncomptime DECOMPOSITION_COUNT = %d\n" % decomposition_count)
        f.write('"""Characters with a decomposition, Hangul excluded."""\n')
        f.write("\ncomptime DECOMPOSITION_DATA = \"%s\"\n" % decomposition)
        f.write(
            '"""Code point delta, then length shifted left two with the'
            " compatibility bit at 1 and the composition exclusion bit at 2,"
            ' then the code points."""\n'
        )

        f.write("\ncomptime LOWERCASE_COUNT = %d\n" % lowercase_count)
        f.write('"""Characters that lowercase to something else."""\n')
        f.write("\ncomptime LOWERCASE_DATA = \"%s\"\n" % lowercase)
        f.write('"""Code point delta, then length, then the code points."""\n')

        f.write("\ncomptime GRAPHEME_COUNT = %d\n" % grapheme_count)
        f.write('"""Ranges in the grapheme break table. Plain text is not in it."""\n')
        f.write("\ncomptime GRAPHEME_DATA = \"%s\"\n" % grapheme)
        f.write(
            '"""Start, length and packed break property per range, as varints.'
            " Four bits of break class, then the pictographic bit, then two bits"
            ' of Indic conjunct class."""\n'
        )

    sys.stderr.write(
        "unicode %s: %d category ranges, %d combining ranges, "
        "%d decompositions, %d lowercase mappings, %d grapheme ranges, "
        "%d bytes\n"
        % (
            ud.unidata_version,
            category_count,
            combining_count,
            decomposition_count,
            lowercase_count,
            grapheme_count,
            len(category)
            + len(combining)
            + len(decomposition)
            + len(lowercase)
            + len(grapheme),
        )
    )


if __name__ == "__main__":
    main()
