#!/usr/bin/env python3
"""Downloads the `tokenizer.json` files the conformance corpus is built from.

The corpus is a few hundred real tokenizers and about a gigabyte of them, so
they are not in the repository. `scripts/tokenizers.tsv` names each one by
repository and revision and records its sha256, and this fetches them into a
directory and checks every digest. A file that is already there with the right
digest is left alone, which is what makes this cheap to run twice and what lets
continuous integration cache the directory.

The revision is a commit hash rather than a branch, so what comes back today is
what came back the day the manifest was written. A repository that has been
deleted or made private answers with an error and is reported at the end rather
than stopping the run, because one missing file should not hide the state of the
other three hundred.

Nothing here is used at runtime. Read the head of `scripts/tokenizer_oracle.mojo`
for what the corpus is for.
"""

import argparse
import concurrent.futures
import hashlib
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://huggingface.co/%s/resolve/%s/tokenizer.json"


def flat(repo):
    """The file name a repository is saved under.

    The slash doubled up into underscores, so every file sits in one directory
    and the name still reads as the repository it came from. The Mojo side
    spells it the same way.
    """
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
                    "bytes": int(f[3]),
                    "sha256": f[4],
                    "status": f[5],
                    "answer": f[6],
                    "skip": f[7],
                }
            )
    return rows


def digest_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(row, into, token):
    path = os.path.join(into, flat(row["repo"]) + ".json")
    if os.path.exists(path) and digest_of(path) == row["sha256"]:
        return (row["repo"], "have", "")
    url = BASE % (row["repo"], row["revision"])
    request = urllib.request.Request(url, headers={"User-Agent": "molla-corpus"})
    if token:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        return (row["repo"], "failed", "http %d" % error.code)
    except Exception as error:  # noqa: BLE001
        return (row["repo"], "failed", str(error)[:80])
    got = hashlib.sha256(body).hexdigest()
    if got != row["sha256"]:
        return (row["repo"], "failed", "sha256 %s want %s" % (got[:16], row["sha256"][:16]))
    tmp = path + ".part"
    with open(tmp, "wb") as handle:
        handle.write(body)
    os.replace(tmp, path)
    return (row["repo"], "fetched", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=os.path.join(HERE, "tokenizers.tsv"))
    parser.add_argument("--into", default="corpus/tokenizers")
    parser.add_argument("--tier", default="quick", choices=["quick", "full"])
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.tier == "quick":
        rows = [r for r in rows if r["tier"] == "quick"]
    os.makedirs(args.into, exist_ok=True)

    # The token is read from the environment because a few repositories answer
    # with a redirect that wants one. None of the corpus needs it today and the
    # run works without it, so it is a convenience rather than a requirement.
    token = os.environ.get("HF_TOKEN", "")

    counts = {"have": 0, "fetched": 0, "failed": 0}
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(fetch_one, r, args.into, token) for r in rows]
        for future in concurrent.futures.as_completed(futures):
            repo, what, why = future.result()
            counts[what] += 1
            if what == "failed":
                failures.append((repo, why))

    for repo, why in sorted(failures):
        print("failed %s %s" % (repo, why), file=sys.stderr)
    total = sum(r["bytes"] for r in rows)
    print(
        "tier %s rows %d have %d fetched %d failed %d bytes %.1f MB"
        % (
            args.tier,
            len(rows),
            counts["have"],
            counts["fetched"],
            counts["failed"],
            total / 1e6,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
