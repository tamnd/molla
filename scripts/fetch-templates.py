#!/usr/bin/env python3
"""Downloads the chat templates the conformance corpus is built from.

`scripts/templates.tsv` names a few hundred real model repositories by
revision, says which file the template came out of, and records the sha256 of
the template itself. This fetches them into a directory and checks every
digest. A file that is already there with the right digest is left alone, which
is what makes this cheap to run twice and what lets continuous integration
cache the directory.

A template lives in one of two places. Newer repositories ship
`chat_template.jinja`, which is the file as written. Older ones keep it as the
`chat_template` member of `tokenizer_config.json`, where it is a JSON string and
has to be decoded before it is a template. Either way what lands on disk and
what the digest covers is the template text, so the Mojo side reads one kind of
file and does not care where it came from.

The revision is a commit hash rather than a branch, so what comes back today is
what came back the day the manifest was written. A repository that has been
deleted or gated answers with an error and is reported at the end rather than
stopping the run, because one missing file should not hide the state of the
other three hundred.

Nothing here is used at runtime. Read the head of `scripts/template_oracle.mojo`
for what the corpus is for.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://huggingface.co/%s/resolve/%s/%s"


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
                    "source": f[5],
                    "status": f[6],
                    "answer": f[7],
                    "note": f[8],
                }
            )
    return rows


def digest_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(source, body):
    """The template text out of whatever file it arrived in.

    A handful of repositories put a list of named templates in
    `tokenizer_config.json` rather than one string, which is how a model ships
    a separate template for tool use. The default is the one everybody renders.
    """
    if source == "chat_template.jinja":
        return body
    config = json.loads(body)
    template = config.get("chat_template")
    if isinstance(template, list):
        for entry in template:
            if entry.get("name") == "default":
                return entry.get("template")
        return template[0].get("template") if template else None
    return template


def fetch_one(row, into, token):
    path = os.path.join(into, flat(row["repo"]) + ".jinja")
    if os.path.exists(path) and digest_of(path) == row["sha256"]:
        return (row["repo"], "have", "")
    url = BASE % (row["repo"], row["revision"], row["source"])
    request = urllib.request.Request(url, headers={"User-Agent": "molla-corpus"})
    if token:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return (row["repo"], "failed", "http %d" % error.code)
    except Exception as error:  # noqa: BLE001
        return (row["repo"], "failed", str(error)[:80])
    try:
        text = extract(row["source"], body)
    except Exception as error:  # noqa: BLE001
        return (row["repo"], "failed", "could not read it: %s" % str(error)[:60])
    if not text:
        return (row["repo"], "failed", "no chat template in %s" % row["source"])
    raw = text.encode("utf-8")
    got = hashlib.sha256(raw).hexdigest()
    if got != row["sha256"]:
        return (row["repo"], "failed", "sha256 %s want %s" % (got[:16], row["sha256"][:16]))
    tmp = path + ".part"
    with open(tmp, "wb") as handle:
        handle.write(raw)
    os.replace(tmp, path)
    return (row["repo"], "fetched", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=os.path.join(HERE, "templates.tsv"))
    parser.add_argument("--into", default="corpus/templates")
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
        "tier %s rows %d have %d fetched %d failed %d bytes %.1f kB"
        % (
            args.tier,
            len(rows),
            counts["have"],
            counts["fetched"],
            counts["failed"],
            total / 1e3,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
