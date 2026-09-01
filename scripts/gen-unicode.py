#!/usr/bin/env python3
"""Generate src/molla/text/tables.mojo from the Unicode database.

Run it with the Python that ships the Unicode version you want:

    python3 scripts/gen-unicode.py

It writes four tables and nothing else. General category ranges, canonical
combining class ranges, character decompositions, and the lowercase mapping.
Everything else the text layer needs is derived from those at runtime, and the
properties that are short enough to read are written out by hand in props.mojo
rather than generated, because a list of twenty five space characters is easier
to check by eye than by diff.

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

    sys.stderr.write(
        "unicode %s: %d category ranges, %d combining ranges, "
        "%d decompositions, %d lowercase mappings, %d bytes\n"
        % (
            ud.unidata_version,
            category_count,
            combining_count,
            decomposition_count,
            lowercase_count,
            len(category) + len(combining) + len(decomposition) + len(lowercase),
        )
    )


if __name__ == "__main__":
    main()
