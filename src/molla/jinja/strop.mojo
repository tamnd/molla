"""String operations, spelled the way Python spells them.

Every function here exists because a chat template calls the Python method of
the same name, and the only thing that matters about any of them is that the
answer is the one Python gives. So the awkward corners are implemented rather
than approximated: `split` with no separator splits on runs of whitespace and
drops the empty pieces at both ends, `split` with a separator does not, negative
indices count from the end, and `strip` strips what Python calls space rather
than what Unicode does, which differ by four characters nobody has typed since
1975 and which would still be a difference.

ASCII is the fast path everywhere and is the only path a chat template normally
takes. The case functions come in pairs for that reason: the ASCII one handles a
string with no high bytes in it without touching the Unicode tables, and the
full one is called only when there is something in the string that needs it.
"""

from molla.text.props import Unicode, is_whitespace
from molla.text.utf8 import from_code_points, to_code_points


def py_space(cp: Int) -> Bool:
    """What `str.isspace` calls space.

    Unicode's White_Space, plus the four ASCII separators at 0x1C to 0x1F,
    which Python counts and Unicode does not. `strip` and the no separator
    `split` both use this, so the four characters would show up in both.
    """
    if cp >= 0x1C and cp <= 0x1F:
        return True
    return is_whitespace(cp)


def needs_unicode(s: String) -> Bool:
    """Whether the string has anything above ASCII in it."""
    var data = s.as_bytes()
    for i in range(len(data)):
        if data[i] >= 0x80:
            return True
    return False


def _from(data: Span[UInt8, _], start: Int, end: Int) -> String:
    var raw = List[UInt8]()
    for i in range(start, end):
        raw.append(data[i])
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def _space_at(data: Span[UInt8, _], at: Int) -> Int:
    """How many bytes of whitespace start at `at`, or zero.

    Returns a byte count rather than a bool so the caller can step over a
    multi byte space without decoding twice.
    """
    var c = data[at]
    if c < 0x80:
        return 1 if py_space(Int(c)) else 0
    var points = to_code_points(_from(data, at, len(data)).as_bytes())
    if len(points) == 0:
        return 0
    if not py_space(points[0]):
        return 0
    if points[0] < 0x800:
        return 2
    if points[0] < 0x10000:
        return 3
    return 4


def _is_space_byte(data: Span[UInt8, _], at: Int) -> Bool:
    var c = data[at]
    if c < 0x80:
        return py_space(Int(c))
    return _space_at(data, at) > 0


def strip(s: String, chars: String, left: Bool, right: Bool) -> String:
    """`strip`, `lstrip` and `rstrip`, with or without a character set.

    With a character set Python removes any of those characters, one at a time,
    and it compares them as characters and not as bytes. The set is short, so
    it is decoded once and each candidate is checked against it.
    """
    var data = s.as_bytes()
    var strip_set = List[Int]()
    if chars.byte_length() > 0:
        strip_set = to_code_points(chars.as_bytes())

    var from_ = 0
    var to = len(data)
    if left:
        while from_ < to:
            var step = _match_strip(data, from_, strip_set)
            if step == 0:
                break
            from_ += step
    if right:
        while to > from_:
            var back = _match_strip_back(data, from_, to, strip_set)
            if back == 0:
                break
            to -= back
    return _from(data, from_, to)


def _match_strip(data: Span[UInt8, _], at: Int, set: List[Int]) -> Int:
    if len(set) == 0:
        return 1 if _is_space_byte(data, at) else 0
    var here = to_code_points(_from(data, at, len(data)).as_bytes())
    if len(here) == 0:
        return 0
    for i in range(len(set)):
        if set[i] == here[0]:
            return _encoded_size(here[0])
    return 0


def _match_strip_back(
    data: Span[UInt8, _], from_: Int, to: Int, set: List[Int]
) -> Int:
    var start = to - 1
    while start > from_ and (data[start] & 0xC0) == 0x80:
        start -= 1
    if len(set) == 0:
        return (to - start) if _is_space_byte(data, start) else 0
    var here = to_code_points(_from(data, start, to).as_bytes())
    if len(here) == 0:
        return 0
    for i in range(len(set)):
        if set[i] == here[0]:
            return to - start
    return 0


def _encoded_size(cp: Int) -> Int:
    if cp < 0x80:
        return 1
    if cp < 0x800:
        return 2
    if cp < 0x10000:
        return 3
    return 4


def split_whitespace(s: String, maxsplit: Int) -> List[String]:
    """`split()` with no separator: runs of space, and no empty pieces."""
    var data = s.as_bytes()
    var out = List[String]()
    var at = 0
    while at < len(data):
        while at < len(data) and _is_space_byte(data, at):
            at += _space_at(data, at)
        if at >= len(data):
            break
        if maxsplit >= 0 and len(out) == maxsplit:
            out.append(_from(data, at, len(data)))
            return out^
        var start = at
        while at < len(data) and not _is_space_byte(data, at):
            at += 1
        out.append(_from(data, start, at))
    return out^


def _find_bytes(
    data: Span[UInt8, _], needle: Span[UInt8, _], from_: Int
) -> Int:
    if len(needle) == 0:
        return from_
    var last = len(data) - len(needle)
    var at = from_
    while at <= last:
        var same = True
        for i in range(len(needle)):
            if data[at + i] != needle[i]:
                same = False
                break
        if same:
            return at
        at += 1
    return -1


def split(s: String, sep: String, maxsplit: Int) -> List[String]:
    """`split(sep)`: exact pieces, empties kept, at most `maxsplit` cuts."""
    var data = s.as_bytes()
    var needle = sep.as_bytes()
    var out = List[String]()
    var at = 0
    while True:
        if maxsplit >= 0 and len(out) == maxsplit:
            break
        var hit = _find_bytes(data, needle, at)
        if hit < 0:
            break
        out.append(_from(data, at, hit))
        at = hit + len(needle)
    out.append(_from(data, at, len(data)))
    return out^


def rsplit(s: String, sep: String, maxsplit: Int) -> List[String]:
    """`rsplit(sep)`, which only differs from `split` when `maxsplit` bites."""
    var pieces = split(s, sep, -1)
    if maxsplit < 0 or len(pieces) <= maxsplit + 1:
        return pieces^
    var keep = len(pieces) - maxsplit
    var head = String("")
    for i in range(keep):
        if i > 0:
            head += sep
        head += pieces[i]
    var out = List[String]()
    out.append(head)
    for i in range(keep, len(pieces)):
        out.append(pieces[i])
    return out^


def splitlines(s: String) -> List[String]:
    """Lines, without their endings, and no trailing empty piece.

    Only the ASCII line endings, which is what a template splitting model
    output is looking for, and `\\r\\n` counts as one.
    """
    var data = s.as_bytes()
    var out = List[String]()
    var at = 0
    var start = 0
    while at < len(data):
        if data[at] == 0x0D:
            out.append(_from(data, start, at))
            at += 1
            if at < len(data) and data[at] == 0x0A:
                at += 1
            start = at
        elif data[at] == 0x0A:
            out.append(_from(data, start, at))
            at += 1
            start = at
        else:
            at += 1
    if start < len(data):
        out.append(_from(data, start, len(data)))
    return out^


def replace(s: String, old: String, new: String, count: Int) -> String:
    """`replace`, and an empty needle inserts between every character.

    The empty needle case is not a curiosity. `'abc'.replace('', '-')` gives
    `-a-b-c-` in Python, and a template doing it by accident with a variable
    that turned out empty would otherwise silently do nothing here and
    something there.
    """
    var data = s.as_bytes()
    var needle = old.as_bytes()
    var out = String("")
    var at = 0
    var done = 0
    if len(needle) == 0:
        out += new
        while at < len(data):
            var step = 1
            while at + step < len(data) and (data[at + step] & 0xC0) == 0x80:
                step += 1
            out += _from(data, at, at + step)
            at += step
            if count < 0 or done + 1 < count:
                out += new
                done += 1
        return out
    while True:
        if count >= 0 and done == count:
            break
        var hit = _find_bytes(data, needle, at)
        if hit < 0:
            break
        out += _from(data, at, hit)
        out += new
        at = hit + len(needle)
        done += 1
    out += _from(data, at, len(data))
    return out


def starts_with(s: String, prefix: String) -> Bool:
    var data = s.as_bytes()
    var want = prefix.as_bytes()
    if len(want) > len(data):
        return False
    for i in range(len(want)):
        if data[i] != want[i]:
            return False
    return True


def ends_with(s: String, suffix: String) -> Bool:
    var data = s.as_bytes()
    var want = suffix.as_bytes()
    if len(want) > len(data):
        return False
    var base = len(data) - len(want)
    for i in range(len(want)):
        if data[base + i] != want[i]:
            return False
    return True


def contains(s: String, needle: String) -> Bool:
    return _find_bytes(s.as_bytes(), needle.as_bytes(), 0) >= 0


def find(s: String, needle: String) -> Int:
    """The character index of the first match, or -1.

    Characters rather than bytes, because Python counts in characters and a
    template feeding the answer back into a slice would otherwise cut a
    multi byte character in half.
    """
    var hit = _find_bytes(s.as_bytes(), needle.as_bytes(), 0)
    if hit < 0:
        return -1
    var data = s.as_bytes()
    var n = 0
    for i in range(hit):
        if (data[i] & 0xC0) != 0x80:
            n += 1
    return n


def count_of(s: String, needle: String) -> Int:
    """Non overlapping occurrences, which is what `str.count` counts."""
    if needle.byte_length() == 0:
        var data = s.as_bytes()
        var n = 1
        for i in range(len(data)):
            if (data[i] & 0xC0) != 0x80:
                n += 1
        return n
    var data = s.as_bytes()
    var want = needle.as_bytes()
    var at = 0
    var n = 0
    while True:
        var hit = _find_bytes(data, want, at)
        if hit < 0:
            return n
        n += 1
        at = hit + len(want)


def repeat(s: String, times: Int) -> String:
    var out = String("")
    for _ in range(times):
        out += s
    return out


def _ascii_lower(c: UInt8) -> UInt8:
    return c + 32 if c >= 0x41 and c <= 0x5A else c


def _ascii_upper(c: UInt8) -> UInt8:
    return c - 32 if c >= 0x61 and c <= 0x7A else c


def lower_ascii(s: String) -> String:
    var data = s.as_bytes()
    var raw = List[UInt8]()
    for i in range(len(data)):
        raw.append(_ascii_lower(data[i]))
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def upper_ascii(s: String) -> String:
    var data = s.as_bytes()
    var raw = List[UInt8]()
    for i in range(len(data)):
        raw.append(_ascii_upper(data[i]))
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def lower_full(s: String, uni: Unicode) -> String:
    """Full lowercase, so a character that lowercases to two produces two."""
    var points = to_code_points(s.as_bytes())
    var out = List[Int]()
    for i in range(len(points)):
        uni.lowercase(points[i], out)
    var raw = from_code_points(out)
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def upper_full(s: String, uni: Unicode) -> String:
    """Uppercase, one code point at a time.

    Simple mappings only. The full ones are the handful where a character
    uppercases to more than one, German sharp s being the famous one, and the
    tables the tokenizer generates do not carry them because nothing in
    normalization needs them. A template uppercasing German prose would see the
    difference. Nothing in the corpus does, and this is written down rather than
    hidden so the day one does there is something to point at.
    """
    var points = to_code_points(s.as_bytes())
    var out = List[Int]()
    for i in range(len(points)):
        out.append(uni.uppercase_one(points[i]))
    var raw = from_code_points(out)
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def _is_alpha_ascii(c: UInt8) -> Bool:
    return (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A)


def title_python(s: String) -> String:
    """`str.title`, where a word starts after anything that is not a letter."""
    var data = s.as_bytes()
    var raw = List[UInt8]()
    var starting = True
    for i in range(len(data)):
        var c = data[i]
        if not _is_alpha_ascii(c):
            raw.append(c)
            if c < 0x80:
                starting = True
            continue
        raw.append(_ascii_upper(c) if starting else _ascii_lower(c))
        starting = False
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


comptime _TITLE_BREAKS = String("-\t\n\r\x0b\x0c ({[<")
"""What the `title` filter treats as the start of a word.

Jinja's filter is not `str.title`. It splits on runs of whitespace, hyphen, and
the three opening brackets and the less than sign, keeps the separators, and
then uppercases the first character of each piece and lowercases the rest. So
`don't` titles to `Don'T` under `str.title` and to `Don't` under the filter, and
the filter is the one a template gets.
"""


def title_filter(s: String) -> String:
    var data = s.as_bytes()
    var breaks = _TITLE_BREAKS.as_bytes()
    var raw = List[UInt8]()
    var starting = True
    for i in range(len(data)):
        var c = data[i]
        var is_break = False
        for j in range(len(breaks)):
            if c == breaks[j]:
                is_break = True
                break
        if is_break:
            raw.append(c)
            starting = True
            continue
        raw.append(_ascii_upper(c) if starting else _ascii_lower(c))
        starting = False
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def capitalize_python(s: String) -> String:
    """`str.capitalize`: first character up, everything after it down."""
    var data = s.as_bytes()
    var raw = List[UInt8]()
    for i in range(len(data)):
        raw.append(_ascii_upper(data[i]) if i == 0 else _ascii_lower(data[i]))
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def indent_lines(s: String, prefix: String, first: Bool, blank: Bool) -> String:
    """The `indent` filter, which is fussier than it looks.

    It does not indent the first line unless asked, it does not indent blank
    lines unless asked, and it leaves a trailing newline where it found one
    rather than indenting the empty piece after it.
    """
    var lines = splitlines(s)
    var trailing = ends_with(s, String("\n")) or ends_with(s, String("\r"))
    var out = String("")
    for i in range(len(lines)):
        if i > 0:
            out += "\n"
        var empty = lines[i].byte_length() == 0
        var wanted = (i > 0 or first) and (not empty or blank)
        if wanted:
            out += prefix
        out += lines[i]
    if trailing:
        out += "\n"
    return out
