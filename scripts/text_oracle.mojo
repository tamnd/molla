"""Prints what `molla.text` does, so a script can compare it against Python.

Not part of the library and not part of the test suite. `scripts/check-text.py`
writes the cases, runs this, and diffs the two answers. It lives here rather
than in `tests/` because it needs a file of a quarter of a million cases that
nobody wants in a repository, and because the answer key is a Python
interpreter rather than a value anyone typed.

Everything crossing the boundary is written as hex code points, one case per
line, because a file of normalization cases is a file of text that is designed
to look like other text, and comparing two of those by eye is not a thing that
works.
"""

from std.sys import argv, exit

from molla.sys.clock import monotonic_ns
from molla.sys.mmap import Mapping
from molla.text.normalize import normalize
from molla.text.props import Unicode
from molla.text.regex import Regex, Scratch
from molla.text.utf8 import from_code_points, to_code_points


def _hex(value: Int) -> String:
    var digits = "0123456789ABCDEF".as_bytes()
    if value == 0:
        return String("0")
    var out = List[UInt8]()
    var rest = value
    while rest > 0:
        out.append(digits[rest & 15])
        rest >>= 4
    var reversed = List[UInt8]()
    for i in range(len(out)):
        reversed.append(out[len(out) - 1 - i])
    return String(StringSpan(unsafe_from_utf8=reversed))


def _lines(path: String) raises -> List[List[Int]]:
    """A file of hex code points, one case per line, mapped rather than read.

    An empty line is an empty case and has to survive as one, because the empty
    string is where a matcher is most likely to differ from another matcher.
    """
    var mapping = Mapping(path)
    var data = Span[UInt8, MutAnyOrigin](
        unsafe_ptr=mapping.base(), length=mapping.length
    )
    var out = List[List[Int]]()
    var line = List[Int]()
    var value = 0
    var have = False
    for i in range(len(data)):
        var b = Int(data[i])
        if b == 32 or b == 10:
            if have:
                line.append(value)
            value = 0
            have = False
            if b == 10:
                out.append(line^)
                line = List[Int]()
        else:
            have = True
            value = value * 16 + (b - 48 if b <= 57 else b - 55)
    return out^


def _forms(path: String) raises:
    """Every case in all four normal forms, one line each, pipe separated."""
    var tables = Unicode()
    var cases = _lines(path)
    var out = String("")
    for c in range(len(cases)):
        for form in range(4):
            var result = normalize(tables, cases[c], form >= 2, form % 2 == 1)
            for i in range(len(result)):
                if i > 0:
                    out += " "
                out += _hex(result[i])
            out += "|"
        out += "\n"
        if out.byte_length() > 1 << 16:
            print(out, end="")
            out = String("")
    print(out, end="")


def _splits(pattern_path: String, text_path: String) raises:
    """Every pattern against every text, as the offsets of the matches.

    Offsets rather than the matched text, because the difference that matters
    is where an engine decided a piece ended, and two engines that disagree
    about that often produce two pieces that look the same.
    """
    var tables = Unicode()
    var scratch = Scratch()
    var patterns = _lines(pattern_path)
    var texts = _lines(text_path)
    var out = String("")
    for p in range(len(patterns)):
        var source = String(
            StringSpan(unsafe_from_utf8=from_code_points(patterns[p]))
        )
        var expression = Regex(source, tables)
        for t in range(len(texts)):
            var found = expression.find_all(texts[t], scratch)
            for i in range(len(found)):
                if i > 0:
                    out += ";"
                out += String(found[i].start) + "," + String(found[i].end)
            out += "\n"
        print(out, end="")
        out = String("")


def _throughput(path: String) raises:
    """How fast the GPT-2 pre-tokenizer pattern splits a file of real text.

    The split alone, with the file already decoded to code points, because the
    number this is here to defend is the one in the issue: encode throughput,
    of which this is the part that a regular expression engine decides. Three
    runs, because the first one is also paying for the page faults.
    """
    var tables = Unicode()
    var scratch = Scratch()
    var expression = Regex(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+"
        + "|\\s+(?!\\S)|\\s+",
        tables,
    )
    var mapping = Mapping(path)
    var data = Span[UInt8, MutAnyOrigin](
        unsafe_ptr=mapping.base(), length=mapping.length
    )
    var points = to_code_points(data)
    print("bytes", len(data), "code points", len(points))
    for _ in range(3):
        var started = monotonic_ns()
        var found = expression.find_all(points, scratch)
        var took = monotonic_ns() - started
        var rate = 0
        if took > 0:
            rate = (len(data) * 1000) // took
        print("pieces", len(found), "ms", took // 1000000, "MB/s", rate)


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(len(raw)):
        args.append(String(raw[i]))
    if len(args) < 3:
        print("usage: text_oracle normalize <cases>")
        print("       text_oracle split <patterns> <texts>")
        print("       text_oracle throughput <utf8 file>")
        exit(2)
    if args[1] == "normalize":
        _forms(args[2])
    elif args[1] == "split":
        if len(args) < 4:
            print("split wants a pattern file and a text file")
            exit(2)
        _splits(args[2], args[3])
    elif args[1] == "throughput":
        _throughput(args[2])
    else:
        print("unknown mode: ", args[1], sep="")
        exit(2)
