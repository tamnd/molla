"""Tests for `molla.text`.

The interesting part of testing this layer is that almost all of it has an
external answer key. Normalization is defined by a file in the Unicode
database, and the regex behaviour that matters is whatever the Rust crates
behind Hugging Face do. So the values below are not invented, they were taken
from Python's `unicodedata` and from the `regex` module, and a full
differential run against both lives outside the suite because it takes minutes
and needs a Python interpreter.

What is here instead is the set of cases that the differential run would find
too late: the ones a rewrite is most likely to break, chosen because each of
them separates a correct implementation from a plausible one. A decoder that
accepts an overlong encoding passes every round trip test and still lets
`\\xc0\\x80` through as a NUL. A composer that does not know about the
exclusion list turns U+0915 U+093C into U+0958 and quietly changes what the
model reads. A canonical orderer that uses an unstable sort agrees with the
answer key on every string that has one mark in it.

Two regex behaviours are pinned here because they are choices rather than
bugs, and both are documented at the check that pins them: what happens after
an empty match, and what happens when a pattern goes exponential.
"""

from harness import Suite

from molla.text.graphemes import cluster_ends
from molla.text.normalize import (
    canonical_order,
    normalize,
    strip_combining,
    strip_marks,
)
from molla.text.props import (
    CAT_CC,
    CAT_CN,
    CAT_LL,
    CAT_LU,
    CAT_MN,
    CAT_ND,
    CAT_PO,
    CAT_SM,
    CAT_ZS,
    Unicode,
    category_name,
    is_whitespace,
)
from molla.text.regex import Regex, Scratch
from molla.text.utf8 import (
    INCOMPLETE,
    INVALID,
    boundary_before,
    count_code_points,
    decode,
    encode,
    encoded_length,
    from_code_points,
    is_valid,
    to_code_points,
)


def _points(text: StringSpan) -> List[Int]:
    return to_code_points(text.as_bytes())


def _hex(points: List[Int]) -> String:
    """Code points as space separated hex, which is how the Unicode files
    write them and how a mismatch is readable."""
    var digits = String("0123456789ABCDEF")
    var out = String("")
    for i in range(len(points)):
        if i > 0:
            out += " "
        var cp = points[i]
        var one = String("")
        while cp > 0:
            var nibble = cp & 0xF
            one = digits[byte = nibble : nibble + 1] + one
            cp >>= 4
        if one.byte_length() == 0:
            one = "0"
        out += one
    return out^


def _form(
    tables: Unicode, text: StringSpan, compatibility: Bool, composed: Bool
) -> String:
    return _hex(normalize(tables, _points(text), compatibility, composed))


def _form_of(
    tables: Unicode, points: List[Int], compatibility: Bool, composed: Bool
) -> String:
    """The same thing for an input written as code points.

    Anything with a combining mark in it is written that way below, because a
    mark in a source file is invisible and two of them are worse.
    """
    return _hex(normalize(tables, points, compatibility, composed))


def _bytes(values: List[Int]) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(len(values)):
        out.append(UInt8(values[i]))
    return out^


def _text(points: List[Int], start: Int, end: Int) -> String:
    """Part of a code point list back as text, so a failing check reads as the
    piece that came out rather than as a list of numbers."""
    var out = String("")
    for i in range(start, end):
        out += chr(points[i])
    return out^


def _split(
    pattern: StringSpan, text: StringSpan, tables: Unicode
) raises -> String:
    """Every match of `pattern` in `text`, joined by a pipe.

    The pieces themselves rather than their offsets, because a pre-tokenizer
    bug shows up as the wrong text far more legibly than as the wrong number.
    """
    var expression = Regex(pattern, tables)
    var scratch = Scratch()
    var points = _points(text)
    var found = expression.find_all(points, scratch)
    var out = String("")
    for i in range(len(found)):
        if i > 0:
            out += "|"
        out += _text(points, found[i].start, found[i].end)
    return out^


def _matches(
    pattern: StringSpan, text: StringSpan, tables: Unicode
) raises -> Bool:
    var expression = Regex(pattern, tables)
    var scratch = Scratch()
    return expression.is_match(_points(text), scratch)


def _utf8(mut suite: Suite):
    suite.group("text/utf8")

    suite.check(encoded_length(0x24) == 1, "one byte length")
    suite.check(encoded_length(0xA2) == 2, "two byte length")
    suite.check(encoded_length(0x20AC) == 3, "three byte length")
    suite.check(encoded_length(0x10348) == 4, "four byte length")

    # The four widths and the boundary on each side of every one of them. A
    # decoder with an off by one in its width table gets the middle of a range
    # right and both ends wrong.
    var edges: List[Int] = [
        0x0,
        0x7F,
        0x80,
        0x7FF,
        0x800,
        0xD7FF,
        0xE000,
        0xFFFF,
        0x10000,
        0x10FFFF,
    ]
    var round_trip = True
    for i in range(len(edges)):
        var one = List[UInt8]()
        encode(edges[i], one)
        var back = decode(Span(one), 0)
        if back.code != edges[i] or back.width != len(one):
            round_trip = False
    suite.check(round_trip, "encode and decode agree at every width boundary")

    var euro = _bytes([0xE2, 0x82, 0xAC])
    var read = decode(Span(euro), 0)
    suite.check(read.code == 0x20AC and read.width == 3, "decode the euro sign")

    # Overlong forms. Each of these encodes a code point that has a shorter
    # form, and each has been a real hole in a real parser: the two byte NUL
    # slips past a scanner looking for a zero byte, and the three byte solidus
    # slips past a path check looking for a slash.
    var overlong_nul = _bytes([0xC0, 0x80])
    var overlong_slash = _bytes([0xE0, 0x80, 0xAF])
    var overlong_max = _bytes([0xF0, 0x8F, 0xBF, 0xBF])
    suite.check(
        decode(Span(overlong_nul), 0).code == INVALID, "reject overlong NUL"
    )
    suite.check(
        decode(Span(overlong_slash), 0).code == INVALID,
        "reject overlong solidus",
    )
    suite.check(
        decode(Span(overlong_max), 0).code == INVALID,
        "reject overlong U+FFFF",
    )

    # Surrogates are not characters, and CESU-8 and WTF-8 both exist because
    # something encoded them anyway.
    var high = _bytes([0xED, 0xA0, 0x80])
    var low = _bytes([0xED, 0xBF, 0xBF])
    suite.check(decode(Span(high), 0).code == INVALID, "reject high surrogate")
    suite.check(decode(Span(low), 0).code == INVALID, "reject low surrogate")

    var too_big = _bytes([0xF4, 0x90, 0x80, 0x80])
    var five_byte = _bytes([0xF8, 0x88, 0x80, 0x80, 0x80])
    var stray = _bytes([0x80])
    suite.check(
        decode(Span(too_big), 0).code == INVALID, "reject beyond U+10FFFF"
    )
    suite.check(
        decode(Span(five_byte), 0).code == INVALID, "reject five byte form"
    )
    suite.check(
        decode(Span(stray), 0).code == INVALID, "reject a lone continuation"
    )

    var missing = _bytes([0xE2, 0x82])
    var short_read = decode(Span(missing), 0)
    suite.check(
        short_read.code == INCOMPLETE and short_read.width == 1,
        "a truncated sequence says how many bytes are still wanted",
    )

    var mixed = String("aé€𐍈").as_bytes()
    var again = from_code_points(to_code_points(mixed))
    var round_bytes = len(again) == len(mixed)
    for i in range(len(again)):
        if round_bytes and again[i] != mixed[i]:
            round_bytes = False
    suite.check(round_bytes, "bytes to code points and back are the same bytes")

    suite.check(is_valid(mixed), "a mixed width string is valid")
    suite.check(count_code_points(mixed) == 4, "count code points not bytes")
    suite.check(
        not is_valid(Span(overlong_nul)), "an overlong string is not valid"
    )

    # Where a decoder has to restart when it is handed the middle of a
    # character, which is the whole of incremental decoding in one function.
    suite.check(
        boundary_before(mixed, 3) == 1, "back up to the start of e acute"
    )
    suite.check(
        boundary_before(mixed, 1) == 0,
        "the character ending at one starts at zero",
    )
    suite.check(boundary_before(mixed, 9) == 6, "back up into a four byte one")


def _props(mut suite: Suite, tables: Unicode):
    suite.group("text/props")

    suite.check(tables.category(0x41) == CAT_LU, "A is an uppercase letter")
    suite.check(tables.category(0x61) == CAT_LL, "a is a lowercase letter")
    suite.check(tables.category(0x31) == CAT_ND, "1 is a decimal number")
    suite.check(tables.category(0x20) == CAT_ZS, "space is a separator")
    suite.check(tables.category(0x21) == CAT_PO, "bang is punctuation")
    suite.check(tables.category(0x2B) == CAT_SM, "plus is a math symbol")
    suite.check(tables.category(0x00) == CAT_CC, "NUL is a control")
    suite.check(tables.category(0x301) == CAT_MN, "U+0301 is a mark")
    suite.check(tables.category(0xE00) == CAT_CN, "U+0E00 is unassigned")
    suite.check(
        tables.category(0x110000) == CAT_CN, "past the end is unassigned"
    )
    suite.check(category_name(CAT_LU) == "Lu", "category names round trip")

    suite.check(tables.combining(0x301) == 230, "acute is class 230")
    suite.check(tables.combining(0x327) == 202, "cedilla is class 202")
    suite.check(tables.combining(0x323) == 220, "dot below is class 220")
    suite.check(tables.combining(0x41) == 0, "a letter has no class")

    suite.check(tables.is_letter(0x41) and tables.is_word(0x41), "A is a word")
    suite.check(tables.is_number(0x31), "1 is a number")
    suite.check(not tables.is_letter(0x31), "1 is not a letter")
    suite.check(tables.is_word(0x5F), "underscore is a word character")

    # What `\w` means is Alphabetic and not the letter categories, and the two
    # differ in both directions. These are the code points where it shows, and
    # every answer is what the Rust regex crate gives, because that is the
    # engine the tokenizer files were tested against. The ideographic zero is
    # the one that costs real ids: a pre-tokenizer that calls it a symbol cuts
    # a Chinese word in half.
    suite.check(tables.is_word(0x3007), "the ideographic zero is a word")
    suite.check(tables.is_word(0x2160), "and so is a roman numeral")
    suite.check(tables.is_word(0x24B6), "and so is a circled letter")
    suite.check(tables.is_word(0x1F130), "and so is a squared one")
    suite.check(tables.is_word(0x200D), "and so is the joiner")
    suite.check(tables.is_word(0x0663), "and so is an arabic digit")
    suite.check(not tables.is_word(0x2603), "a snowman is not a word")
    suite.check(not tables.is_word(0xA9), "and neither is a copyright sign")
    suite.check(tables.is_punctuation(0x21), "bang is punctuation")

    suite.check(is_whitespace(0x20), "space is whitespace")
    suite.check(is_whitespace(0x0A), "newline is whitespace")
    suite.check(is_whitespace(0x00A0), "no break space is whitespace")
    suite.check(not is_whitespace(0x200B), "zero width space is not")

    suite.check(tables.lowercase_one(0xC9) == 0xE9, "E acute lowercases")
    suite.check(tables.uppercase_one(0xE9) == 0xC9, "e acute uppercases")
    suite.check(tables.lowercase_one(0x41) == 0x61, "A lowercases")

    # The full lowercase mapping is not one to one. Turkish capital I with dot
    # becomes two characters, and a simple mapping table cannot say so.
    var folded = List[Int]()
    tables.lowercase(0x130, folded)
    suite.check(
        _hex(folded) == "69 307", "dotted capital I lowercases to two points"
    )

    suite.check(tables.compose(0x65, 0x301) == 0xE9, "e and acute compose")
    suite.check(
        tables.compose(0x1100, 0x1161) == 0xAC00, "Hangul composes by sum"
    )
    suite.check(tables.compose(0x41, 0x41) == 0, "two letters do not compose")
    # U+0958 decomposes to these two and is on the composition exclusion list,
    # so NFC must leave the pair alone. Running the decomposition table
    # backwards without reading that list is the classic way to get this wrong.
    suite.check(
        tables.compose(0x915, 0x93C) == 0, "an excluded pair does not compose"
    )


def _normalize(mut suite: Suite, tables: Unicode):
    suite.group("text/normalize")

    suite.check(_form(tables, "é", False, False) == "65 301", "NFD splits")
    suite.check(_form(tables, "é", False, True) == "E9", "NFC joins")
    suite.check(
        _form(tables, "ﬁ", False, False) == "FB01", "NFD leaves a ligature"
    )
    suite.check(
        _form(tables, "ﬁ", True, True) == "66 69", "NFKC breaks a ligature"
    )
    suite.check(
        _form(tables, "½", True, True) == "31 2044 32", "NFKC on a half"
    )
    suite.check(
        _form(tables, "ＡＢ", True, True) == "41 42", "NFKC folds wide forms"
    )
    suite.check(
        _form(tables, "가", False, False) == "1100 1161", "NFD splits Hangul"
    )
    suite.check(_form(tables, "가", False, True) == "AC00", "NFC joins Hangul")
    suite.check(
        _form(tables, "각", False, False) == "1100 1161 11A8",
        "NFD splits a Hangul syllable with a trailer",
    )
    suite.check(
        _form(tables, "Ω", False, False) == "3A9", "the ohm sign is already NFD"
    )

    # Two marks on one base, written as code points because they are invisible
    # otherwise. Dot above arrives first and dot below second, and the answer
    # key puts dot below first, because 220 sorts before 230. Then NFC takes
    # only the first mark, because U+1E0D exists and U+1E0D U+0307 has no
    # further composition.
    var d_dots: List[Int] = [0x64, 0x307, 0x323]
    suite.check(
        _form_of(tables, d_dots, False, False) == "64 323 307",
        "canonical order sorts marks by combining class",
    )
    suite.check(
        _form_of(tables, d_dots, False, True) == "1E0D 307",
        "NFC composes the first mark and keeps the second",
    )
    # The same shape on a base with no precomposed form, so nothing joins and
    # all that is left of NFC is the reordering.
    var q_dots: List[Int] = [0x71, 0x307, 0x323]
    suite.check(
        _form_of(tables, q_dots, False, True) == "71 323 307",
        "NFC on a base with no precomposed form only reorders",
    )
    # Equal classes must not be reordered. Acute and diaeresis are both class
    # 230, and an unstable sort is free to swap them and still look sorted.
    var e_marks: List[Int] = [0x65, 0x301, 0x308]
    suite.check(
        _form_of(tables, e_marks, False, False) == "65 301 308",
        "marks of equal class keep the order they arrived in",
    )
    suite.check(
        _form_of(tables, e_marks, False, True) == "E9 308",
        "NFC composes the first of two equal class marks",
    )

    var dotted_i: List[Int] = [0x49, 0x307]
    suite.check(
        _form_of(tables, dotted_i, False, True) == "130",
        "I and dot above compose to the Turkish capital",
    )

    var already = _points("hello world")
    suite.check(
        _hex(normalize(tables, already, False, True)) == _hex(already),
        "ASCII is unchanged by every form",
    )

    var marks = _points("ḍ̇")
    canonical_order(tables, marks)
    suite.check(_hex(marks) == "64 323 307", "canonical_order sorts in place")

    var accented = normalize(tables, _points("Café"), False, False)
    suite.check(
        _hex(strip_marks(tables, accented)) == "43 61 66 65",
        "strip_marks drops what a decomposition separated",
    )
    suite.check(
        _hex(strip_combining(tables, accented)) == "43 61 66 65",
        "strip_combining drops the same non spacing marks",
    )

    # The two are different for the other two mark categories, and the
    # difference is the whole reason both exist. A spacing mark and an
    # enclosing mark survive the BERT flag and do not survive a StripAccents
    # step written on its own.
    var visarga: List[Int] = [0x61, 0x903]
    suite.check(
        _hex(strip_marks(tables, visarga)) == "61 903",
        "strip_marks keeps a spacing mark",
    )
    suite.check(
        _hex(strip_combining(tables, visarga)) == "61",
        "strip_combining drops a spacing mark",
    )
    var keycap: List[Int] = [0x31, 0x20E3]
    suite.check(
        _hex(strip_marks(tables, keycap)) == "31 20E3",
        "strip_marks keeps an enclosing mark",
    )
    suite.check(
        _hex(strip_combining(tables, keycap)) == "31",
        "strip_combining drops an enclosing mark",
    )


def _clusters(tables: Unicode, points: List[Int]) -> String:
    """Cluster lengths in code points, joined by a slash.

    Lengths rather than the text itself, because every string worth testing
    here is a base and something invisible hanging off it and printing them
    back would show one character either way.
    """
    var ends = cluster_ends(tables, points)
    var out = String("")
    var start = 0
    for i in range(len(ends)):
        if i > 0:
            out += "/"
        out += String(ends[i] - start)
        start = ends[i]
    return out^


def _graphemes(mut suite: Suite, tables: Unicode) raises:
    """Extended grapheme cluster boundaries, UAX #29.

    Every answer here came from the `regex` module's `\\X`, which is the same
    thing the Rust crate behind the reference tokenizer implements. The cases
    are one per rule that can be got wrong on its own rather than a sweep,
    because a sweep over the whole of Unicode lives outside the suite.
    """
    suite.group("text/graphemes")

    suite.check(_clusters(tables, _points("")) == "", "nothing has no cluster")
    suite.check(
        _clusters(tables, _points("abc")) == "1/1/1", "ASCII is one each"
    )

    var e_acute: List[Int] = [0x65, 0x301]
    suite.check(
        _clusters(tables, e_acute) == "2", "a mark joins the letter before it"
    )

    var crlf: List[Int] = [0x0D, 0x0A]
    suite.check(_clusters(tables, crlf) == "2", "CR and LF are one cluster")
    var lfcr: List[Int] = [0x0A, 0x0D]
    suite.check(_clusters(tables, lfcr) == "1/1", "and LF and CR are two")
    var line: List[Int] = [0x61, 0x0D, 0x0A, 0x62]
    suite.check(
        _clusters(tables, line) == "1/2/1", "a line break breaks either side"
    )
    var tabbed: List[Int] = [0x61, 0x09]
    suite.check(
        _clusters(tables, tabbed) == "1/1", "a control character stands alone"
    )

    var hangul: List[Int] = [0x1100, 0x1161, 0x11A8]
    suite.check(
        _clusters(tables, hangul) == "3", "a Hangul syllable spelled in jamo"
    )

    # GB12 and GB13. Regional indicators pair up from the left, so an odd one
    # at the end is a cluster on its own rather than joining the pair.
    var flag: List[Int] = [0x1F1E6, 0x1F1E7]
    suite.check(_clusters(tables, flag) == "2", "two regional indicators pair")
    var flags: List[Int] = [0x1F1E6, 0x1F1E7, 0x1F1E8]
    suite.check(
        _clusters(tables, flags) == "2/1", "and a third starts a new cluster"
    )

    # GB11, the emoji rule. The join only holds because the run before the
    # zero width joiner started with a pictograph.
    var joined: List[Int] = [0x1F468, 0x200D, 0x1F469]
    suite.check(
        _clusters(tables, joined) == "3", "a zero width joiner joins emoji"
    )
    var not_joined: List[Int] = [0x61, 0x200D, 0x1F469]
    suite.check(
        _clusters(tables, not_joined) == "2/1",
        "and does not join a letter to one",
    )
    var toned: List[Int] = [0x1F469, 0x1F3FD]
    suite.check(_clusters(tables, toned) == "2", "a skin tone is an extender")

    # GB9c, the Indic conjunct rule. Consonant, linker, consonant is one
    # cluster, and the linker is what makes it one.
    var conjunct: List[Int] = [0x915, 0x94D, 0x915]
    suite.check(
        _clusters(tables, conjunct) == "3", "a virama joins two consonants"
    )
    var separate: List[Int] = [0x915, 0x915]
    suite.check(
        _clusters(tables, separate) == "1/1",
        "and two consonants without one are two clusters",
    )
    var spacing: List[Int] = [0x915, 0x93E]
    suite.check(
        _clusters(tables, spacing) == "2", "a spacing mark joins as well"
    )
    var prepended: List[Int] = [0x600, 0x627]
    suite.check(
        _clusters(tables, prepended) == "2",
        "and a prepending character joins forwards",
    )


def _regex(mut suite: Suite, tables: Unicode) raises:
    suite.group("text/regex")

    suite.check(_split("a", "banana", tables) == "a|a|a", "a literal")
    suite.check(_split("ab|a", "ab", tables) == "ab", "alternation order")
    # Leftmost first, not leftmost longest. Given a choice at the same starting
    # position the earlier branch wins even though the later one is longer,
    # which is what Perl and fancy-regex do and what the tokenizer files were
    # written against. So this splits into two pieces, not one.
    suite.check(
        _split("a|ab", "ab", tables) == "a", "the earlier branch wins a tie"
    )

    suite.check(_split("a+", "aaa b aa", tables) == "aaa|aa", "greedy plus")
    suite.check(_split("a+?", "aaa", tables) == "a|a|a", "lazy plus")
    suite.check(_split("a*+b", "aaab", tables) == "aaab", "possessive star")
    suite.check(
        _split("a{2,3}", "aaaaa", tables) == "aaa|aa", "a counted repeat"
    )
    suite.check(_split("a{2}", "aaa", tables) == "aa", "an exact repeat")
    suite.check(_split("ab?", "ab a", tables) == "ab|a", "an optional")

    suite.check(_split("[abc]+", "xxabcxx", tables) == "abc", "a class")
    suite.check(_split("[^ ]+", "a bc", tables) == "a|bc", "a negated class")
    suite.check(_split("[a-c]+", "abcd", tables) == "abc", "a range")
    suite.check(_split(r"\d+", "a12b345", tables) == "12|345", "digits")
    suite.check(_split(r"\s+", "a  b\tc", tables) == "  |\t", "whitespace")
    suite.check(_split(r"\w+", "a_1 b", tables) == "a_1|b", "word characters")
    suite.check(
        _split(r"\p{L}+", "ab12cd", tables) == "ab|cd", "a Unicode category"
    )
    suite.check(
        _split(r"\p{N}+", "ab12cd", tables) == "12", "the number category"
    )
    suite.check(
        _split(r"\p{Lu}+", "aBCd", tables) == "BC", "a two letter category"
    )
    suite.check(
        _split(r"[^\p{L}\p{N}]+", "ab!? cd", tables) == "!? ",
        "a class of negated categories",
    )
    suite.check(
        _split(".+", "ab\ncd", tables) == "ab|cd", "dot stops at a line"
    )

    # Caret is the start of the text and dollar is the end of it, not the
    # start and end of a line. Perl and Python read them the other way round,
    # the Rust crates read them this way, and `(?m)` asks for the other one.
    suite.check(_split("^a", "ab\nab", tables) == "a", "caret is text start")
    suite.check(_split("b$", "ab\nab", tables) == "b", "dollar is text end")
    suite.check(
        _split("(?m)^a", "ab\nab", tables) == "a|a", "and (?m) makes it a line"
    )
    suite.check(
        _split("(?m)b$", "ab\nab", tables) == "b|b", "for dollar as well"
    )
    suite.check(_split(r"\bfoo\b", "a foo bar", tables) == "foo", "a boundary")
    suite.check(_split(r"\Bar", "bar car", tables) == "ar|ar", "a non boundary")
    suite.check(_split(r"\Aa", "aa", tables) == "a", "backslash A")

    suite.check(_split(r"a(?=b)", "ab ac", tables) == "a", "positive lookahead")
    suite.check(_split(r"a(?!b)", "ab ac", tables) == "a", "negative lookahead")
    suite.check(
        _split(r"\s+(?!\S)|\s+", "a  \n  b", tables) == "  \n | ",
        "the trailing whitespace idiom from the GPT-2 pattern",
    )
    suite.check(_split("(?i)ab", "AB ab Ab", tables) == "AB|ab|Ab", "a flag")
    suite.check(_split("(?i:a)b", "Ab", tables) == "Ab", "a scoped flag")
    suite.check(_split("(?i)[a-c]+", "ABCD", tables) == "ABC", "a folded range")
    suite.check(_split(r"a\.b", "a.b axb", tables) == "a.b", "an escaped dot")
    suite.check(_split(r"\x41+", "AAb", tables) == "AA", "a hex escape")
    suite.check(_split("é+", "aéé", tables) == "éé", "a literal above ASCII")
    suite.check(
        _split(r"a{2,", "a{2, a", tables) == "a{2,",
        "an unclosed counted repeat is a literal brace",
    )
    suite.check(_split("(ab)+", "abab", tables) == "abab", "a group repeats")

    suite.check(_matches("^abc$", "abc", tables), "is_match says yes")
    suite.check(not _matches("^abc$", "abcd", tables), "is_match says no")

    # The class `\w` compiles here and the same question gets answered again by
    # `Unicode.is_word` when a boundary is being decided, so the two have to
    # agree. These are the code points where Alphabetic and the letter
    # categories part company, and the splits came from Python's `regex`
    # module.
    suite.check(
        _split(r"\w+", "龥鿿〇々 ☃", tables) == "龥鿿〇々",
        r"\w takes the letter numbers, so a word keeps its zero",
    )
    suite.check(
        _split(r"\w+", "ⒶⒷ©", tables) == "ⒶⒷ",
        r"and the circled letters, and stops at a symbol that is not one",
    )
    suite.check(
        _split(r"\w+", "🄰🄱☃", tables) == "🄰🄱",
        r"and the squared letters above the basic plane",
    )

    # The real thing. This is the GPT-2 pre-tokenizer pattern, and the expected
    # split came from Python's `regex` module, not from us.
    var gpt2 = (
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|"
        r" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    )
    suite.check(
        _split(gpt2, "Hello, world! It's 42 tokens.\n\n  ", tables)
        == "Hello|,| world|!| It|'s| 42| tokens|.|\n\n  ",
        "the GPT-2 pre-tokenizer pattern",
    )

    # An empty match moves the cursor on by one character. Rust's regex crate
    # does that, Python's re retries at the same position looking for a non
    # empty match, and the two give different answers here. Hugging Face was
    # built against the Rust rule, so this is the Rust answer: three empty
    # matches at the three positions of a two character string, then nothing.
    var empties = Regex("a*", tables)
    var scratch = Scratch()
    var found = empties.find_all(_points("xx"), scratch)
    suite.check(
        len(found) == 3
        and found[0].start == 0
        and found[0].end == 0
        and found[2].start == 2,
        "an empty match advances one character, the Rust rule",
    )

    # Catastrophic backtracking is bounded rather than prevented. A pattern out
    # of a tokenizer.json is a pattern from a file somebody downloaded, so the
    # engine has to fail loudly on a bad one instead of hanging.
    var bomb = Regex("(a+)+b", tables)
    var bomb_scratch = Scratch()
    var hay = String("")
    for _ in range(30):
        hay += "a"
    var raised = False
    try:
        _ = bomb.is_match(_points(hay), bomb_scratch)
    except:
        raised = True
    suite.check(raised, "an exponential pattern raises instead of hanging")

    # Lookbehind is rejected at compile time rather than half supported. No
    # tokenizer pattern uses one, and a partly working one is worse than none.
    var rejected = False
    try:
        _ = Regex("(?<=a)b", tables)
    except:
        rejected = True
    suite.check(rejected, "lookbehind is rejected at compile time")

    # A pattern comes out of a file somebody downloaded, so every one of these
    # has to be an error at compile time rather than a surprise at match time.
    # The last two are the ones an engine is most likely to accept quietly:
    # `a**` reads as a repeat of a repeat, which is how you write an
    # exponential pattern by accident.
    var broken: List[String] = ["[a", "(a", "[z-a]", "\\q", "*a", "a**"]
    var bad = 0
    for i in range(len(broken)):
        try:
            _ = Regex(broken[i], tables)
        except:
            bad += 1
    suite.check(bad == len(broken), "malformed patterns raise")

    # One compiled pattern, many searches, one scratch space. The engine keeps
    # its backtracking stacks in the caller's scratch so that a tokenizer
    # encoding a million strings allocates once, and a stale stack from the
    # previous call would show up as a wrong answer on the next one.
    var reused = Regex(r"\p{L}+", tables)
    var shared = Scratch()
    var stable = True
    for _ in range(3):
        if len(reused.find_all(_points("ab cd ef"), shared)) != 3:
            stable = False
    suite.check(stable, "a scratch space is safe to reuse")


def run(mut suite: Suite) raises:
    var tables = Unicode()
    _utf8(suite)
    _props(suite, tables)
    _normalize(suite, tables)
    _graphemes(suite, tables)
    _regex(suite, tables)
