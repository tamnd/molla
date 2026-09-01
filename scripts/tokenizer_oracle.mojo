"""Runs the conformance corpus and prints what molla makes of it.

Not part of the library and not part of the test suite. `scripts/tokenizers.tsv`
names a few hundred real `tokenizer.json` files by repository and revision, and
records for each one the sha256 of the answer the Hugging Face `tokenizers`
library gives for every case in `scripts/tokenizer_cases.txt`. This loads each
file, produces the same answer in the same spelling, hashes it, and compares.

The answer for one case is one line: the case number, the ids with special
tokens, the ids without them, and the bytes the ids decode back to, written as
hex. Tab separated, newline terminated, and the sha256 is taken over the whole
run of them. A digest rather than the answers themselves because the answers
are forty megabytes and the repository does not want them, and because a digest
that matches is the only thing anybody reads.

One thing in here has no reference to check against. The beginning of text
token is written twice when a chat template puts one in the text and the post
processor puts another in front of it, and molla drops the second. The Hugging
Face library has no such rule, so there is nothing to compare against and the
check is a property instead: encoding text that already opens with the token
gives the same ids as encoding it the ordinary way, with exactly one of them
gone. That runs over every file in the corpus too.

`scripts/check-tokenizer.py` is the other half. It fetches the files, runs the
reference implementation, and writes the digests this checks against.
"""

from std.sys import argv, exit

from molla.sys.mmap import Mapping
from molla.sys.sha256 import Sha256, hex_digest
from molla.text.utf8 import from_code_points, to_code_points
from molla.tokenizer.tokenizer import Session, Tokenizer

comptime STATUS_OK = 0
comptime STATUS_REFUSED = 1


def _read(path: String) raises -> List[UInt8]:
    var mapping = Mapping(path)
    var out = List[UInt8]()
    out.reserve(mapping.length)
    var data = Span[UInt8, MutAnyOrigin](
        unsafe_ptr=mapping.base(), length=mapping.length
    )
    for i in range(len(data)):
        out.append(data[i])
    mapping.close()
    return out^


def _lines(data: List[UInt8]) -> List[String]:
    """Split on newlines, dropping a trailing empty line and nothing else."""
    var out = List[String]()
    var one = List[UInt8]()
    for i in range(len(data)):
        if data[i] == 10:
            out.append(String(StringSpan(unsafe_from_utf8=Span(one))))
            one.clear()
        else:
            one.append(data[i])
    if len(one) > 0:
        out.append(String(StringSpan(unsafe_from_utf8=Span(one))))
    return out^


def _fields(line: String) -> List[String]:
    var out = List[String]()
    var one = List[UInt8]()
    var data = line.as_bytes()
    for i in range(len(data)):
        if data[i] == 9:
            out.append(String(StringSpan(unsafe_from_utf8=Span(one))))
            one.clear()
        else:
            one.append(data[i])
    out.append(String(StringSpan(unsafe_from_utf8=Span(one))))
    return out^


def _unhex(text: String) -> List[UInt8]:
    var data = text.as_bytes()
    var out = List[UInt8]()
    var i = 0
    while i + 1 < len(data):
        out.append(UInt8((_digit(data[i]) << 4) | _digit(data[i + 1])))
        i += 2
    return out^


def _digit(b: UInt8) -> Int:
    var v = Int(b)
    if v >= 97:
        return v - 87
    if v >= 65:
        return v - 55
    return v - 48


def _hex(data: Span[UInt8, _]) -> String:
    var digits = "0123456789abcdef".as_bytes()
    var out = List[UInt8]()
    for i in range(len(data)):
        out.append(digits[Int(data[i]) >> 4])
        out.append(digits[Int(data[i]) & 15])
    return String(StringSpan(unsafe_from_utf8=Span(out)))


def _ids(ids: List[Int]) -> String:
    var out = String("")
    for i in range(len(ids)):
        if i > 0:
            out += " "
        out += String(ids[i])
    return out^


def _skipped(field: String, count: Int) -> List[Bool]:
    """The excluded case numbers, as a flag per case.

    A dash means none, which is what almost every row says. The ones that do
    exclude a case name it here rather than dropping the case for everybody,
    because a case that one tokenizer disagrees about is still worth running
    against the other three hundred.
    """
    var out = List[Bool](length=count, fill=False)
    if field == "-":
        return out^
    var data = field.as_bytes()
    var value = 0
    var have = False
    for i in range(len(data)):
        if data[i] == 44:
            if have and value < count:
                out[value] = True
            value = 0
            have = False
            continue
        value = value * 10 + (Int(data[i]) - 48)
        have = True
    if have and value < count:
        out[value] = True
    return out^


def _answer(
    tokenizer: Tokenizer,
    cases: List[List[UInt8]],
    skip: List[Bool],
    mut session: Session,
    mut body: String,
) raises:
    for c in range(len(cases)):
        if skip[c]:
            continue
        var text = StringSpan(unsafe_from_utf8=Span(cases[c]))
        var with_special = List[Int]()
        tokenizer.encode(text, True, session, with_special)
        var plain = List[Int]()
        tokenizer.encode(text, False, session, plain)
        # Decoding gives bytes, and a run of ids can spell bytes that are not
        # text: a byte fallback token for a lone continuation byte, or an
        # unknown token standing in for half a character. The reference hands
        # back a string, so its answer already has a replacement character
        # wherever that happened. Putting ours through the same substitution is
        # what makes the two comparable, and it is the only place in here that
        # changes an answer rather than reading one.
        var back = from_code_points(
            to_code_points(Span(tokenizer.decode_bytes(plain, False)))
        )
        body += String(c)
        body += "\t"
        body += _ids(with_special)
        body += "\t"
        body += _ids(plain)
        body += "\t"
        body += _hex(Span(back))
        body += "\n"


def _same(a: List[Int], b: List[Int], without: Int) -> Bool:
    """Whether `a` with the id at `without` taken out is `b`.

    A negative index means take nothing out, which is the other half of the
    rule and the case almost every file lands in.
    """
    var expected = len(a) if without < 0 else len(a) - 1
    if len(b) != expected:
        return False
    var j = 0
    for i in range(len(a)):
        if i == without:
            continue
        if a[i] != b[j]:
            return False
        j += 1
    return True


def _bos(
    tokenizer: Tokenizer, repo: String, mut session: Session
) raises -> Int:
    """The beginning of text rule, checked against the file's own answers.

    Encoding nothing gives whatever the post processor writes around an empty
    sequence, and encoding one letter gives the same thing with the letter in
    the middle. The ids the two agree on at the front are the opening special,
    and the last of them is the id the rule is about, because that is the one
    sitting next to the text.

    Then the same text is encoded twice, once as ordinary text and once as text
    a chat template rendered. Feeding it the opening special spelled out makes
    the id appear twice, next to itself, and the rendered call has to come back
    with one of them gone and nothing else changed. When the id does not appear
    twice there is nothing to drop and the two calls have to agree exactly,
    which is the half of the rule that catches dropping too eagerly.

    Returns 1 if the file was in a position to say anything and 0 if it was
    not, so the run can report how much of the corpus the rule was read off.
    """
    var empty = List[Int]()
    tokenizer.encode("", True, session, empty)
    var letter = List[Int]()
    tokenizer.encode("x", True, session, letter)
    var lead = 0
    while (
        lead < len(empty) and lead < len(letter) and empty[lead] == letter[lead]
    ):
        lead += 1
    if lead == 0:
        return 0

    var one = List[Int]()
    one.append(empty[lead - 1])
    var spelled = tokenizer.decode_bytes(one, False)
    if len(spelled) == 0:
        return 0
    var text = String(StringSpan(unsafe_from_utf8=Span(spelled)))
    text += "hello world"

    var plain = List[Int]()
    tokenizer.encode(text, True, session, plain)
    var rendered = List[Int]()
    tokenizer.encode_rendered(text, session, rendered)

    var without = -1
    if len(plain) > lead and plain[lead - 1] == plain[lead]:
        without = lead - 1
    if not _same(plain, rendered, without):
        print("BOS", repo, "plain", _ids(plain), "rendered", _ids(rendered))
        raise Error("the beginning of text rule did not hold")
    return 1


def main() raises:
    var args = argv()
    if len(args) < 4:
        print(
            "usage: tokenizer_oracle CASES MANIFEST DIR [TIER] [--print REPO]"
        )
        exit(2)
    var cases_path = String(args[1])
    var manifest_path = String(args[2])
    var directory = String(args[3])
    var tier = String("all")
    if len(args) > 4:
        tier = String(args[4])
    var only = String("")
    if len(args) > 6 and String(args[5]) == "--print":
        only = String(args[6])

    var cases = List[List[UInt8]]()
    for line in _lines(_read(cases_path)):
        cases.append(_unhex(line))

    var checked = 0
    var refused = 0
    var failed = 0
    var bos = 0

    for line in _lines(_read(manifest_path)):
        if line.startswith("#") or line.startswith("repo\t"):
            continue
        var row = _fields(line)
        if len(row) < 8:
            continue
        var repo = row[0]
        var row_tier = row[2]
        var status = STATUS_OK if row[5] == "ok" else STATUS_REFUSED
        var want = row[6]
        var skip = _skipped(row[7], len(cases))
        # The quick tier is part of the full one rather than beside it, so
        # asking for everything means every row and asking for the quick tier
        # is the only thing that leaves any out.
        if tier == "quick" and row_tier != "quick":
            continue
        if only != "" and repo != only:
            continue

        var path = directory + "/" + _flat(repo) + ".json"
        if status == STATUS_REFUSED:
            var opened = True
            try:
                _ = Tokenizer(path, 0)
            except:
                opened = False
            if opened:
                print("LOADED", repo, "but the manifest says it is refused")
                failed += 1
            else:
                refused += 1
            continue

        var body = String("")
        var session = Session()
        try:
            var tokenizer = Tokenizer(path, 0)
            _answer(tokenizer, cases, skip, session, body)
            bos += _bos(tokenizer, repo, session)
        except e:
            print("ERROR", repo, String(e))
            failed += 1
            continue
        if only != "":
            print(body, end="")
            continue
        var digest = Sha256()
        digest.update(body.as_bytes())
        var got = hex_digest(digest.digest())
        if got != want:
            print("MISMATCH", repo, "want", want, "got", got)
            failed += 1
        else:
            checked += 1

    if only != "":
        return
    print(
        "checked",
        checked,
        "refused",
        refused,
        "failed",
        failed,
        "cases",
        len(cases),
        "bos",
        bos,
    )
    if failed > 0:
        exit(1)


def _flat(repo: String) -> String:
    """The file name a repository is saved under, which is its name with the
    slash doubled up into underscores so it fits in one directory."""
    var out = List[UInt8]()
    var data = repo.as_bytes()
    for i in range(len(data)):
        if data[i] == 47:
            out.append(UInt8(95))
            out.append(UInt8(95))
        else:
            out.append(data[i])
    return String(StringSpan(unsafe_from_utf8=Span(out)))
