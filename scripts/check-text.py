#!/usr/bin/env python3
"""Check molla.text against Python, which is the only answer key there is.

Normalization is defined by a file in the Unicode database and Python ships a
decoder for it, so the four normal forms of every assigned code point can be
compared one for one. Regular expressions have no such file, but Hugging Face
runs the Rust `regex` and `fancy-regex` crates, and the `regex` module on PyPI
agrees with them everywhere that matters for a tokenizer pattern, so that is
the second oracle.

Run it as `python3 scripts/check-text.py`. It builds `scripts/text_oracle.mojo`,
writes the cases into a temporary directory, runs both sides and diffs them. It
takes a couple of minutes, almost all of it Python building the expected
answers for a quarter of a million normalization cases, which is why it is a
script somebody runs rather than a test in the suite.

Two known differences from Python are not bugs and are not compared here. An
empty match advances by one character rather than being retried at the same
place, and `$` means the end of the text rather than the end of the text or
just before a trailing newline. Both are what the Rust crates do, and the
tokenizer files were written against those. `tests/test_text.mojo` pins both.
"""

import os
import random
import subprocess
import sys
import tempfile
import unicodedata

try:
    import regex
except ImportError:
    regex = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The three real ones first. These are the pre-tokenizer patterns out of a
# GPT-2, a Qwen and a Llama 3 tokenizer.json, and they are the reason any of
# this exists.
TOKENIZER_PATTERNS = [
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
]

# And then one of each feature, because a differential run only tells you about
# the constructs you thought to write down.
GENERIC_PATTERNS = [
    "a+", "a*", "a?", "a{2}", "a{2,}", "a{2,4}", "a+?", "a{2,4}?",
    "a++", "a*+", "a?+", "a{1,3}+",
    "(ab)+", "(a|b)+c", "a|ab", "ab|a", "(?:foo|foobar)!",
    "[a-c]+", "[^a-c]+", r"[\d]+", r"\d+", r"\D+", r"\w+", r"\W+",
    r"\s+", r"\S+", r"\p{L}+", r"\p{Lu}+", r"\P{L}+", r"\pN+",
    "^a+", r"\Aab", r"ab\z", r"\ba\w*", r"\Ba\w*",
    "a(?=b)", "a(?!b)", r"\s+(?!\S)", r"[^\r\n\p{L}\p{N}]?\p{L}+",
    ".+", ".", "x.y", "(?i:abc)", "(?i)abc", "(?i:[a-z]+)",
    r"\.\+\*", r"[\]\-a]+", "a{1,3}b{2}",
]

HAND_WRITTEN_TEXTS = [
    "",
    "a",
    "aa",
    "aaa",
    "aaaab",
    "ab",
    "abc",
    "ABC",
    "abcabc",
    "foobar!",
    "foo!",
    "x.y",
    "x\ny",
    ".+*",
    "]-a]",
    "a b\tc\n d",
    "123 456",
    "héllo wörld",
    "Ünïcödé",
    "  \n\n  ",
    "a1!b2?",
    "café",
    "ЖУК жук",
    "混合text123",
    "a" * 30,
    "]",
    "-a-",
    "It's",
    "IT'S",
    "isn't",
    "\xa0nbsp\xa0",
    "tab\tsep",
    "Hello world! It's 2026, isn't it?  Fine.\n\n  ok",
    "def main():\n    print('hi')\n\treturn 0\n",
    "Крокодил ест 3 яблока, и не 42!",
    "こんにちは世界。これはテストです。",
    "café CAFÉ Café naïvé é",
    "  leading and trailing   ",
    "\n\n\n",
    "\r\n\r\nwindows\r\n",
    "emoji 👍🏽 family 👨‍👩‍👧‍👦 flag 🇻🇳 end",
    "a1b2c3 100 1000 10000 3.14159 -7",
    "«guillemets» ‹single› „german“ 中文标点。",
    "tabs\tand\tspaces      and nbsp\xa0em",
    "I'll say it's O'Brien's, y'all've won't",
    "ALLCAPS lower MiXeD _under_score_ __dunder__",
    "https://example.com/path?a=1&b=2#frag",
    "x" * 300,
    " " * 40 + "end",
    "กำเนิด ගිණුම",
    "𝓯𝓪𝓷𝓬𝔂 𝟏𝟐𝟑 math",
    "mixed中英文mixed 数字123 punctuation！？",
]

# The alphabet the random texts are drawn from. Every character in it is there
# because some branch of some pattern cares about it: two scripts, a combining
# mark, both kinds of space, the quote that starts a contraction.
ALPHABET = list("aabbccXYZ0123 \t\n.,!?'") + ["é", "中", "́", "ć"]
RANDOM_TEXTS = 80
RANDOM_SEED = 20260901


def random_texts():
    rng = random.Random(RANDOM_SEED)
    out = []
    for _ in range(RANDOM_TEXTS):
        length = rng.randint(0, 60)
        out.append("".join(rng.choice(ALPHABET) for _ in range(length)))
    return out


# The throughput file. Words rather than random characters, in more than one
# script, because the pattern has a branch per character class and a file of
# one class only exercises one of them.
BENCH_WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
    "numbers", "123", "4567", "punctuation,", "like:", "this;", "that!",
    "café", "naïve", "текст", "中文", "something", "and", "then",
]
BENCH_BYTES = 4 * 1000 * 1000


def bench_text():
    rng = random.Random(RANDOM_SEED)
    out = []
    size = 0
    line = []
    while size < BENCH_BYTES:
        word = rng.choice(BENCH_WORDS)
        line.append(word)
        size += len(word.encode()) + 1
        if len(line) >= rng.randint(4, 20):
            out.append(" ".join(line))
            line = []
    return "\n".join(out) + "\n"


def normalization_cases():
    """Every assigned code point on its own, then marks piled onto bases.

    The single code points are the bulk of it and they catch a wrong table. The
    combinations are the ones that catch a wrong algorithm: reordering only
    shows up when two marks of different classes meet, and the composition
    exclusions only show up when a pair that decomposes refuses to compose.
    """
    cases = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        if unicodedata.category(chr(cp)) == "Cn":
            continue
        cases.append([cp])

    bases = [0x61, 0x64, 0x71, 0x41, 0x49, 0x4F, 0x915, 0x931, 0x3B1, 0x1100]
    marks = [0x300, 0x301, 0x307, 0x308, 0x323, 0x327, 0x334, 0x5B0, 0x93C]
    for base in bases:
        for first in marks:
            cases.append([base, first])
            for second in marks:
                cases.append([base, first, second])
                cases.append([base, second, first])

    # Hangul from both ends, since it is arithmetic rather than a table.
    for syllable in range(0xAC00, 0xD7A4, 271):
        cases.append([syllable])
    for lead in range(0x1100, 0x1113):
        for vowel in range(0x1161, 0x1176, 3):
            cases.append([lead, vowel])
            cases.append([lead, vowel, 0x11A8])

    # A few longer strings, because a per character loop can be right on every
    # character and still lose track of where the last starter was.
    cases.append([0x1E0A, 0x323, 0x41, 0x30A, 0x1100, 0x1161, 0x11A8])
    cases.append([0xFB01, 0x301, 0x323])
    cases.append([0x2126, 0x301, 0x64, 0x307, 0x323, 0x71])
    return cases


def to_hex_lines(cases):
    return "".join(" ".join("%X" % cp for cp in case) + "\n" for case in cases)


def expected_forms(cases):
    out = []
    for case in cases:
        text = "".join(chr(cp) for cp in case)
        line = ""
        for form in ("NFD", "NFC", "NFKD", "NFKC"):
            done = unicodedata.normalize(form, text)
            line += " ".join("%X" % ord(c) for c in done) + "|"
        out.append(line)
    return "\n".join(out) + "\n"


def expected_splits(patterns, texts):
    out = []
    for pattern in patterns:
        compiled = regex.compile(pattern)
        for text in texts:
            spans = []
            for found in compiled.finditer(text):
                spans.append("%d,%d" % (found.start(), found.end()))
            out.append(";".join(spans))
    return "\n".join(out) + "\n"


def diff(name, expected, got):
    if expected == got:
        print("%s: %d cases, identical" % (name, len(expected.splitlines())))
        return True
    expected_lines = expected.splitlines()
    got_lines = got.splitlines()
    shown = 0
    print("%s: DIFFERENT" % name)
    if len(expected_lines) != len(got_lines):
        print("  %d lines expected, %d produced" % (len(expected_lines), len(got_lines)))
    for i in range(min(len(expected_lines), len(got_lines))):
        if expected_lines[i] != got_lines[i]:
            print("  line %d" % (i + 1))
            print("    python: %s" % expected_lines[i])
            print("    molla:  %s" % got_lines[i])
            shown += 1
            if shown == 10:
                print("  and more")
                break
    return False


def main():
    work = tempfile.mkdtemp(prefix="molla-text-")
    binary = os.path.join(work, "text_oracle")
    print("building the oracle")
    subprocess.run(
        ["pixi", "run", "mojo", "build", "-I", "src",
         "scripts/text_oracle.mojo", "-o", binary],
        cwd=ROOT, check=True,
    )

    ok = True

    cases = normalization_cases()
    print("normalization: %d cases" % len(cases))
    case_file = os.path.join(work, "normalize.txt")
    with open(case_file, "w") as handle:
        handle.write(to_hex_lines(cases))
    got = subprocess.run([binary, "normalize", case_file],
                         capture_output=True, text=True, check=True).stdout
    ok = diff("normalization", expected_forms(cases), got) and ok

    if regex is None:
        print("regex: skipped, `pip install regex` for this half")
    else:
        patterns = TOKENIZER_PATTERNS + GENERIC_PATTERNS
        texts = HAND_WRITTEN_TEXTS + random_texts()
        print("regex: %d patterns over %d texts" % (len(patterns), len(texts)))
        pattern_file = os.path.join(work, "patterns.txt")
        text_file = os.path.join(work, "texts.txt")
        with open(pattern_file, "w") as handle:
            handle.write(to_hex_lines([[ord(c) for c in p] for p in patterns]))
        with open(text_file, "w") as handle:
            handle.write(to_hex_lines([[ord(c) for c in t] for t in texts]))
        got = subprocess.run([binary, "split", pattern_file, text_file],
                             capture_output=True, text=True, check=True).stdout
        ok = diff("regex", expected_splits(patterns, texts), got) and ok

    bench_file = os.path.join(work, "bench.txt")
    with open(bench_file, "w") as handle:
        handle.write(bench_text())
    print("throughput, the GPT-2 pattern over %s" % bench_file)
    subprocess.run([binary, "throughput", bench_file], check=True)

    print("work kept in %s" % work)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
