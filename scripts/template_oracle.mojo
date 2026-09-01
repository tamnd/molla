"""Runs the chat template corpus and prints what molla makes of it.

Not part of the library and not part of the test suite. `scripts/templates.tsv`
names a few hundred real chat templates by repository and revision, and records
for each one the sha256 of the answer `transformers` gives for every case in
`scripts/template_cases.txt`. This compiles each template, produces the same
answer in the same spelling, hashes it, and compares.

The answer for one case is one line: the case number, whether it rendered or
raised, and the bytes it rendered to written as hex. Tab separated, newline
terminated, and the sha256 is taken over the whole run of them. A digest rather
than the answers themselves because the answers are twenty megabytes and the
repository does not want them, and because a digest that matches is the only
thing anybody reads.

A case that raises carries no bytes. The two implementations word their errors
differently on purpose, ours naming the line and the column, so comparing the
wording would be a test of the error messages rather than of the output. What is
compared is that both sides refused the same case of the same template.

A row whose status is `refused` is a template molla will not compile at all,
because it uses one of the constructs the engine excludes. The `note` column
says which one. This checks that the template really is still refused, since a
refusal that quietly turned into a render is the interesting direction, and it
prints the list at the end. That list is the backlog for the engine and it gets
read every release.

A row whose status is `disputed` is a template that renders on both sides and
does not agree, where the difference is the reference doing something that is
Python rather than Jinja. There are two of them and the `note` column says what
each one is. They are still compiled and still rendered, so a crash in one is
still a failure, and they are printed at the end next to the refusals, because a
list of the places we knowingly differ is worth as much as the count of the
places we do not.

The clock is pinned to 2025-01-01 00:00:00 UTC, because `strftime_now` is in
every Llama template and a corpus whose answer changes at midnight is not a
corpus. Both halves are given the same second and both run under `TZ=UTC`.

`scripts/check-template.py` is the other half. It fetches the templates, runs
`transformers` on them, and writes the digests this checks against.
"""

from std.sys import argv, exit

from molla.jinja.env import Limits
from molla.jinja.template import Template
from molla.sys.mmap import Mapping
from molla.sys.sha256 import Sha256, hex_digest

comptime PINNED = 1735689600
"""The second `strftime_now` reads, which is 2025-01-01 00:00:00 UTC.

The same number `scripts/check-template.py` pins, and it has to stay the same
number, because every answer digest in the manifest was taken with it.
"""


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


def _hex(data: Span[UInt8, _]) -> String:
    var digits = "0123456789abcdef".as_bytes()
    var out = List[UInt8]()
    for i in range(len(data)):
        out.append(digits[Int(data[i]) >> 4])
        out.append(digits[Int(data[i]) & 15])
    return String(StringSpan(unsafe_from_utf8=Span(out)))


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


def _answer(template: Template, cases: List[String]) raises -> String:
    """One line per case, in the spelling the Python side uses."""
    var limits = Limits()
    var body = String("")
    for c in range(len(cases)):
        var vars = cases[c]
        body += String(c)
        try:
            var text = template.render_object(vars, limits, PINNED)
            body += "\tok\t"
            body += _hex(text.as_bytes())
        except:
            body += "\traise\t"
        body += "\n"
    return body^


def main() raises:
    var args = argv()
    if len(args) < 4:
        print("usage: template_oracle CASES MANIFEST DIR [TIER] [--print REPO]")
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

    # A case is a name and the variables as JSON, and only the JSON is rendered.
    # The name is there so that a mismatch on case eleven can be read as the
    # tool result case rather than as the number eleven.
    var names = List[String]()
    var cases = List[String]()
    for line in _lines(_read(cases_path)):
        if line.byte_length() == 0:
            continue
        var row = _fields(line)
        names.append(row[0])
        cases.append(row[1])

    var checked = 0
    var refused = 0
    var disputed = 0
    var failed = 0
    var notes = List[String]()

    for line in _lines(_read(manifest_path)):
        if line.startswith("#") or line.startswith("repo\t"):
            continue
        var row = _fields(line)
        if len(row) < 9:
            continue
        var repo = row[0]
        var row_tier = row[2]
        var status = row[6]
        var want = row[7]
        var note = row[8]
        # The quick tier is part of the full one rather than beside it, so
        # asking for everything means every row and asking for the quick tier
        # is the only thing that leaves any out.
        if tier == "quick" and row_tier != "quick":
            continue
        if only != "" and repo != only:
            continue

        var path = directory + "/" + _flat(repo) + ".jinja"
        var source = String(StringSpan(unsafe_from_utf8=Span(_read(path))))
        if status == "refused":
            var compiled = True
            try:
                _ = Template(source)
            except:
                compiled = False
            if compiled:
                print("COMPILED", repo, "but the manifest says it is refused")
                failed += 1
            else:
                refused += 1
                notes.append("refused " + repo + "\t" + note)
            continue

        var body = String("")
        try:
            var template = Template(source)
            body += _answer(template, cases)
        except e:
            print("ERROR", repo, String(e))
            failed += 1
            continue
        if only != "":
            print(body, end="")
            continue
        if status == "disputed":
            disputed += 1
            notes.append("disputed " + repo + "\t" + note)
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
    # The refusal list is printed on every run rather than kept in a file that
    # nobody opens. It is short, and it is the backlog for the engine.
    for i in range(len(notes)):
        print(notes[i])
    print(
        "checked",
        checked,
        "refused",
        refused,
        "disputed",
        disputed,
        "failed",
        failed,
        "cases",
        len(cases),
    )
    if failed > 0:
        exit(1)
