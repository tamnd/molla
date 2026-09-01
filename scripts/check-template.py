#!/usr/bin/env python3
"""The reference half of the chat template conformance corpus.

`scripts/template_oracle.mojo` compiles every template in the corpus, renders
every case in `scripts/template_cases.txt`, and hashes the answers. This does
the same thing with `transformers`, which is the implementation every model
author tested their template against. The two digests either match or they do
not.

The rendering here goes through `transformers.utils.chat_template_utils
.render_jinja_template`, which is the function `apply_chat_template` calls to
turn a conversation into a string. So this is not a reimplementation of what
`transformers` does with a template, it is the thing itself, one call below the
tokenizer so that no model weights have to be downloaded to run it.

Two things have to be pinned or the same input gives two answers. The clock,
because a Llama template writes today's date into the system prompt, so
`strftime_now` is replaced with one reading a fixed second and both sides are
run under `TZ=UTC`. And the shape of the variables, because `render_jinja_
template` always binds `messages`, `tools`, `documents` and
`add_generation_prompt` whether the caller passed them or not, so every case in
the case file spells all four out, with a null where the case does not use one.
A template asking `tools is defined` gets the same answer from both sides that
way.

Three things to run:

    check-template.py --dir DIR

Recompute the reference digests and compare them against the manifest. This
needs the `transformers` package. It is not what continuous integration runs on
every commit, because the point of the digests is that the everyday check needs
no Python at all. Run it when the reference version moves.

    check-template.py --dir DIR --refresh

The same, but write the manifest back with the fresh digests. The `status` and
`note` columns are kept as they are, because they are statements about molla
rather than about the reference, and nothing here can work them out.

    check-template.py --dir DIR --explain REPO

Print the reference answer for one repository, one line per case, so it can be
diffed against the same repository out of the Mojo side's `--print`. This is how
a mismatch turns into a case number.

Python is a test time oracle and nothing else. It is not in the build, it is not
in the server, and it does not run on every commit.
"""

import argparse
import datetime
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

PINNED = 1735689600
"""The second `strftime_now` reads, which is 2025-01-01 00:00:00 UTC.

A Wednesday in a month whose name is not its abbreviation, in a year that is not
a leap year, on a day whose number is one, which between them make a wrong month
table, a wrong weekday and a missing zero pad all show up as a difference. The
Mojo side is given the same number and both sides run under `TZ=UTC`.
"""


def flat(repo):
    """The file name a repository is saved under, spelled the same way the
    fetcher and the Mojo side spell it."""
    return repo.replace("/", "__")


def read_manifest(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("repo\t"):
                continue
            f = line.split("\t")
            rows.append(
                {
                    "repo": f[0],
                    "revision": f[1],
                    "tier": f[2],
                    "bytes": f[3],
                    "sha256": f[4],
                    "source": f[5],
                    "status": f[6],
                    "answer": f[7],
                    "note": f[8],
                }
            )
    return rows


def write_manifest(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "repo\trevision\ttier\tbytes\tfile_sha256\tsource\tstatus\tanswer_sha256\tnote\n"
        )
        for r in rows:
            handle.write(
                "\t".join(
                    [
                        r["repo"],
                        r["revision"],
                        r["tier"],
                        r["bytes"],
                        r["sha256"],
                        r["source"],
                        r["status"],
                        r["answer"],
                        r["note"],
                    ]
                )
                + "\n"
            )


def read_cases(path):
    """One case per line, a name and the variables as JSON, tab separated."""
    import json

    cases = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            name, body = line.split("\t", 1)
            cases.append((name, json.loads(body)))
    return cases


def strftime_pinned(fmt):
    return datetime.datetime.fromtimestamp(PINNED).strftime(fmt)


def render(text, case):
    """One case, through the same call `apply_chat_template` makes."""
    from transformers.utils.chat_template_utils import render_jinja_template

    rest = dict(case)
    messages = rest.pop("messages")
    tools = rest.pop("tools", None)
    documents = rest.pop("documents", None)
    add_generation_prompt = rest.pop("add_generation_prompt", False)
    rendered, _ = render_jinja_template(
        conversations=[messages],
        tools=tools,
        documents=documents,
        chat_template=text,
        add_generation_prompt=add_generation_prompt,
        strftime_now=strftime_pinned,
        **rest,
    )
    return rendered[0]


def answer(text, cases):
    """One line per case, in the spelling the Mojo side uses.

    The case number, whether it rendered or raised, and the bytes it rendered
    to written as hex. A case that raises carries no bytes, because the two
    implementations word their errors differently on purpose: ours names the
    line and the column, and a corpus that compared the wording would be a test
    of the error messages rather than of the output.
    """
    lines = []
    for i, (_, case) in enumerate(cases):
        try:
            out = render(text, case)
        except Exception:  # noqa: BLE001
            lines.append("%d\traise\t" % i)
            continue
        lines.append("%d\tok\t%s" % (i, out.encode("utf-8").hex()))
    return "\n".join(lines) + "\n" if lines else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=os.path.join(HERE, "templates.tsv"))
    parser.add_argument("--cases", default=os.path.join(HERE, "template_cases.txt"))
    parser.add_argument("--dir", default="corpus/templates")
    parser.add_argument("--tier", default="full", choices=["quick", "full"])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--explain", default="")
    args = parser.parse_args()

    # Both halves read the clock through the same zone, and the zone has to be
    # set before anything formats a date. Setting it here rather than asking the
    # caller to means the two run the same way from a shell and from CI.
    os.environ["TZ"] = "UTC"
    if hasattr(__import__("time"), "tzset"):
        __import__("time").tzset()

    rows = read_manifest(args.manifest)
    cases = read_cases(args.cases)

    if args.explain:
        picked = [r for r in rows if r["repo"] == args.explain]
        if not picked:
            print("no such repo in the manifest: %s" % args.explain, file=sys.stderr)
            return 2
        path = os.path.join(args.dir, flat(picked[0]["repo"]) + ".jinja")
        with open(path, encoding="utf-8") as handle:
            sys.stdout.write(answer(handle.read(), cases))
        return 0

    wanted = rows if args.tier == "full" else [r for r in rows if r["tier"] == "quick"]
    agreed = 0
    moved = []
    missing = []
    for row in wanted:
        path = os.path.join(args.dir, flat(row["repo"]) + ".jinja")
        if not os.path.exists(path):
            missing.append(row["repo"])
            continue
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        try:
            body = answer(text, cases)
        except Exception as error:  # noqa: BLE001
            moved.append((row["repo"], "the reference refused it: %s" % str(error)[:80]))
            continue
        got = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if got == row["answer"]:
            agreed += 1
        else:
            moved.append((row["repo"], "%s to %s" % (row["answer"][:16], got[:16])))
            row["answer"] = got

    for repo, why in moved:
        print("moved %s %s" % (repo, why))
    for repo in missing:
        print("missing %s" % repo, file=sys.stderr)

    if args.refresh and moved:
        write_manifest(args.manifest, rows)
        print("wrote %s" % args.manifest)

    print(
        "rows %d agreed %d moved %d missing %d cases %d"
        % (len(wanted), agreed, len(moved), len(missing), len(cases))
    )
    if missing:
        return 1
    if moved and not args.refresh:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
