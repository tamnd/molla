#!/usr/bin/env python3
"""The reference half of the tokenizer conformance corpus.

`scripts/tokenizer_oracle.mojo` loads every `tokenizer.json` in the corpus,
encodes every case in `scripts/tokenizer_cases.txt`, and hashes the answers.
This does the same thing with the Hugging Face `tokenizers` library, which is
the implementation everybody else's ids come from. The two digests either match
or they do not.

Three things to run:

    check-tokenizer.py --dir DIR

Recompute the reference digests and compare them against the manifest. This is
the check that the manifest still says what the reference says, and it needs the
`tokenizers` package. It is not what continuous integration runs on every commit,
because the point of the digests is that the everyday check needs no Python at
all. Run it when the reference version moves.

    check-tokenizer.py --dir DIR --refresh

The same, but write the manifest back with the fresh digests. The `status` and
`skip` columns are kept as they are, because they are statements about molla
rather than about the reference, and nothing here can work them out.

    check-tokenizer.py --dir DIR --explain REPO

Print the reference answer for one repository, one line per case, so it can be
diffed against the same repository out of the Mojo side's `--print`. This is how
a mismatch turns into a case number.

Python is a test time oracle and nothing else. It is not in the build, it is not
in the server, and it does not run on every commit.
"""

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


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
                    "status": f[5],
                    "answer": f[6],
                    "skip": f[7],
                }
            )
    return rows


def write_manifest(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("repo\trevision\ttier\tbytes\tfile_sha256\tstatus\tanswer_sha256\tskip\n")
        for r in rows:
            handle.write(
                "\t".join(
                    [
                        r["repo"],
                        r["revision"],
                        r["tier"],
                        r["bytes"],
                        r["sha256"],
                        r["status"],
                        r["answer"],
                        r["skip"],
                    ]
                )
                + "\n"
            )


def read_cases(path):
    """One case per line, written as hex.

    Hex rather than the text itself because the corpus is full of things a text
    editor would ruin: lone control characters, a byte order mark, trailing
    spaces, and the empty string. The empty string is case zero and it is an
    empty line, so blank lines are cases here and not padding.
    """
    with open(path, encoding="utf-8") as handle:
        return [bytes.fromhex(line.rstrip("\n")) for line in handle]


def skipped(field):
    if field == "-":
        return set()
    return set(int(x) for x in field.split(",") if x)


def answer(tokenizer, cases, skip):
    """One line per case, in the spelling the Mojo side uses.

    The case number, the ids with special tokens, the ids without them, and the
    bytes the ids decode back to written as hex. Tab separated and newline
    terminated, and the digest is taken over the whole run of them.
    """
    lines = []
    for i, case in enumerate(cases):
        if i in skip:
            continue
        text = case.decode("utf-8")
        with_special = tokenizer.encode(text, add_special_tokens=True).ids
        plain = tokenizer.encode(text, add_special_tokens=False).ids
        back = tokenizer.decode(plain, skip_special_tokens=False).encode("utf-8")
        lines.append(
            "%d\t%s\t%s\t%s"
            % (
                i,
                " ".join(map(str, with_special)),
                " ".join(map(str, plain)),
                back.hex(),
            )
        )
    return "\n".join(lines) + "\n" if lines else ""


def load(path):
    """Open a file with padding and truncation turned off.

    A `tokenizer.json` can carry both, and about one in ten of the corpus does.
    Padding makes the reference return a fixed length row of ids with the pad
    token filling the end, and truncation makes it cut a long case short or
    refuse a pair outright. Neither is a property of the tokenizer, they are
    settings a training script left behind, and molla does not read them at all.
    Turning them off is what makes the two sides answer the same question.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(path)
    try:
        tokenizer.no_padding()
    except Exception:  # noqa: BLE001
        pass
    try:
        tokenizer.no_truncation()
    except Exception:  # noqa: BLE001
        pass
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=os.path.join(HERE, "tokenizers.tsv"))
    parser.add_argument("--cases", default=os.path.join(HERE, "tokenizer_cases.txt"))
    parser.add_argument("--dir", default="corpus/tokenizers")
    parser.add_argument("--tier", default="full", choices=["quick", "full"])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--explain", default="")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    cases = read_cases(args.cases)

    if args.explain:
        picked = [r for r in rows if r["repo"] == args.explain]
        if not picked:
            print("no such repo in the manifest: %s" % args.explain, file=sys.stderr)
            return 2
        row = picked[0]
        path = os.path.join(args.dir, flat(row["repo"]) + ".json")
        sys.stdout.write(answer(load(path), cases, skipped(row["skip"])))
        return 0

    wanted = rows if args.tier == "full" else [r for r in rows if r["tier"] == "quick"]
    agreed = 0
    moved = []
    missing = []
    for row in wanted:
        path = os.path.join(args.dir, flat(row["repo"]) + ".json")
        if not os.path.exists(path):
            missing.append(row["repo"])
            continue
        try:
            body = answer(load(path), cases, skipped(row["skip"]))
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
