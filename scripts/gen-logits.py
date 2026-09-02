#!/usr/bin/env python3
"""Write the logit conformance references by asking llama.cpp.

For each case this produces one small text file that records what llama.cpp
computed for a fixed list of token ids against a specific GGUF file. Two
different things come out of llama.cpp and both go in the file.

`llama-server` gives the whole final distribution. Asking it for `n_probs`
equal to the vocabulary size returns a log probability for every token, which
is the logits up to an additive constant, and the additive constant is the one
thing about a row of logits that carries no meaning. What lands in the file is
the top of that distribution plus an evenly spaced sample of the rest, because
the top is what a wrong kernel moves first and the tail is what a wrong kernel
that happens to keep the ranking still gets wrong.

`llama-eval-callback` gives one line per intermediate tensor with its sum and
six of its values. The ones worth keeping are `embd` and every `l_out-N`, which
are the residual stream on the way in and after each layer. That is what turns
a disagreement into a layer number.

Both run on whatever backend llama.cpp picks, which on this laptop is Metal.
That is deliberate and it is the opposite of what you would guess. llama.cpp's
CPU backend quantizes the activation vector to eight bits before every matmul,
so it sits further from exact arithmetic than molla does, and a reference taken
off it needs a tolerance around two per cent to pass. Its Metal backend keeps
activations in float and lands within a few parts in ten thousand of molla,
which is fifty times tighter and therefore fifty times more likely to notice a
real mistake. Pass `--cpu` to take the reference off the CPU backend instead,
and the file records which one it was.

Needs llama.cpp on PATH. Nothing else, no Python packages.

Usage:

    scripts/gen-logits.py scripts/logit_cases.txt scripts/logits corpus/logits
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

PORT = 8757
THREADS = 4
TOP = 64
"""Tokens of the head of the distribution to record."""

PROBES = 256
"""Evenly spaced tokens of the rest to record. Enough to notice a kernel that
got the ranking right and the spacing wrong, small enough that a case file
stays a few hundred lines."""

CPU_ONLY = False
"""Set by --cpu. Recorded in every file this run writes."""


def need(binary: str) -> str:
    found = shutil.which(binary)
    if found is None:
        sys.exit(f"{binary} is not on PATH, install llama.cpp first")
    return found


def build_version() -> str:
    out = subprocess.run(
        [need("llama-cli"), "--version"], capture_output=True, text=True
    )
    text = out.stderr + out.stdout
    m = re.search(r"build (\d+), commit ([0-9a-f]+)", text)
    if m:
        return f"b{m.group(1)} {m.group(2)}"
    return "unknown"


def quantize(source: pathlib.Path, dest: pathlib.Path, kind: str) -> None:
    """Requantize a GGUF into another type, if it is not already there.

    Requantizing from an already quantized file loses a little more than
    quantizing from the original weights would. That does not matter here.
    Nothing in this compares molla against the model the humans trained, it
    compares molla against llama.cpp reading the same bytes, and both sides
    read whatever came out of this.
    """
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  quantize {dest.name} from {source.name}", flush=True)
    args = [need("llama-quantize"), "--allow-requantize"]
    if kind != "F16":
        args.append("--pure")
    args += [str(source), str(dest), kind, "4"]
    run = subprocess.run(args, capture_output=True, text=True)
    if run.returncode != 0:
        sys.exit(run.stderr[-2000:])


def tensors(model: pathlib.Path, prompt: str) -> tuple[list[int], list[dict]]:
    """Token ids and one record per intermediate tensor."""
    run = subprocess.run(
        [
            need("llama-eval-callback"),
            "-m",
            str(model),
            "-p",
            prompt,
            "-n",
            "1",
            "-t",
            str(THREADS),
        ]
        + (["-ngl", "0"] if CPU_ONLY else []),
        capture_output=True,
        text=True,
    )
    text = run.stdout
    if "common_debug_cb_eval" not in text:
        sys.exit(f"llama-eval-callback said nothing useful:\n{run.stderr[-2000:]}")

    ids: list[int] = []
    lines = (run.stderr + "\n" + text).splitlines()
    for i, line in enumerate(lines):
        if "number of input tokens" in line:
            count = int(line.rsplit("=", 1)[1])
            for j in range(1, count + 1):
                ids.append(int(lines[i + j].split()[-1]))
            break
    if not ids:
        sys.exit("llama-eval-callback did not print the token ids")

    out: list[dict] = []
    body = text.splitlines()
    at = 0
    while at < len(body):
        line = body[at]
        if not line.startswith("common_debug_cb_eval:"):
            at += 1
            continue
        head = line.split(":", 1)[1].strip()
        name = head.split(" ", 1)[0]
        shape = re.findall(r"\{([^}]*)\}\s*$", head)
        dims = [int(x) for x in shape[0].split(",")] if shape else []
        rows: list[list[float]] = []
        total = None
        at += 1
        while at < len(body) and not body[at].startswith("common_debug_cb_eval:"):
            stripped = body[at].strip()
            if stripped.startswith("sum ="):
                total = float(stripped.split("=", 1)[1])
            elif re.match(r"^\[.*\],$", stripped):
                values = re.findall(r"-?\d+\.\d+", stripped)
                if values:
                    rows.append([float(v) for v in values])
            at += 1
        if total is not None:
            out.append({"name": name, "dims": dims, "rows": rows, "sum": total})
    return ids, out


def logprobs(model: pathlib.Path, ids: list[int], vocab: int) -> list[float]:
    """The whole final distribution, indexed by token id."""
    server = subprocess.Popen(
        [
            need("llama-server"),
            "-m",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "-c",
            "512",
            "-t",
            str(THREADS),
            "--no-webui",
        ]
        + (["-ngl", "0"] if CPU_ONLY else []),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 180
        while True:
            if time.time() > deadline:
                sys.exit("llama-server did not come up")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/health", timeout=2
                ) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                time.sleep(1)

        body = json.dumps(
            {
                "prompt": ids,
                "n_predict": 1,
                "temperature": 0,
                "n_probs": vocab,
                "post_sampling_probs": False,
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            answer = json.load(r)
    finally:
        server.terminate()
        server.wait(timeout=30)

    if answer["tokens_evaluated"] != len(ids):
        sys.exit(
            "llama-server evaluated "
            + str(answer["tokens_evaluated"])
            + " tokens for a prompt of "
            + str(len(ids))
            + ", so it added one of its own"
        )
    step = answer["completion_probabilities"][0]["top_logprobs"]
    out = [0.0] * vocab
    seen = [False] * vocab
    for entry in step:
        out[entry["id"]] = entry["logprob"]
        seen[entry["id"]] = True
    if not all(seen):
        sys.exit(
            "llama-server returned "
            + str(len(step))
            + " of "
            + str(vocab)
            + " log probabilities, so n_probs was capped"
        )
    return out


def write_case(
    path: pathlib.Path,
    name: str,
    model: pathlib.Path,
    prompt: str,
    version: str,
) -> None:
    ids, dumped = tensors(model, prompt)
    by_name = {t["name"]: t for t in dumped}
    if "embd" not in by_name:
        sys.exit("no embd tensor in the dump, llama.cpp renamed something")
    layers = 0
    while f"l_out-{layers}" in by_name:
        layers += 1
    if layers == 0:
        sys.exit("no l_out tensors in the dump, llama.cpp renamed something")
    width = by_name["embd"]["dims"][0]
    vocab = by_name["result_output"]["dims"][0]

    lines = [
        "# molla logit reference. Written by scripts/gen-logits.py.",
        "# Every number below came out of llama.cpp reading this exact file.",
        f"oracle llama.cpp {version}",
        "backend " + ("cpu" if CPU_ONLY else "default"),
        f"name {name}",
        f"model {model.name}",
        f"bytes {model.stat().st_size}",
        f"layers {layers}",
        f"width {width}",
        f"vocab {vocab}",
        f"prompt {prompt}",
        "tokens " + " ".join(str(i) for i in ids),
    ]

    # Snapshot k is the residual stream layer k was handed, so k of zero is the
    # embedding and k of layers is what the final norm reads. One more after
    # that holds the normed vector, which leaves the output head alone between
    # the last snapshot and the logits: a case whose layers all agree and whose
    # logits do not is then the head and nothing else. The row count goes in
    # because llama.cpp drops every position but the last once nothing
    # downstream needs them, and a sum over one row and a sum over five are not
    # the same number.
    for k in range(layers + 2):
        if k == 0:
            t = by_name["embd"]
        elif k <= layers:
            t = by_name[f"l_out-{k - 1}"]
        else:
            t = by_name["result_norm"]
        rows = t["dims"][1] if len(t["dims"]) > 1 else 1
        last = t["rows"][-1]
        if len(last) != 6:
            sys.exit(f"expected six sampled values in {t['name']}, got {len(last)}")
        sampled = " ".join(f"{v:.4f}" for v in last)
        lines.append(f"layer {k} {rows} {t['sum']:.6f} {sampled}")

    full = logprobs(model, ids, vocab)
    order = sorted(range(vocab), key=lambda i: full[i], reverse=True)
    for i in order[:TOP]:
        lines.append(f"top {i} {full[i]:.6f}")
    step = max(1, vocab // PROBES)
    for i in range(0, vocab, step):
        lines.append(f"probe {i} {full[i]:.6f}")

    path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {path} ({layers} layers, {len(ids)} tokens)", flush=True)


def main() -> None:
    global CPU_ONLY
    args = sys.argv[1:]
    if "--cpu" in args:
        CPU_ONLY = True
        args.remove("--cpu")
    sys.argv = [sys.argv[0]] + args
    if len(sys.argv) != 4:
        sys.exit(
            "usage: gen-logits.py <cases.txt> <reference-dir> <model-dir>"
        )
    cases = pathlib.Path(sys.argv[1])
    out_dir = pathlib.Path(sys.argv[2])
    model_dir = pathlib.Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    version = build_version()
    print(f"llama.cpp {version}")

    for raw in cases.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, source, kind, prompt = line.split("\t")
        source_path = pathlib.Path(source).expanduser()
        if not source_path.exists():
            print(f"  skip {name}, no {source_path}", flush=True)
            continue
        # The model file is named after the source and the type rather than
        # after the case, so several prompts against one file share it instead
        # of requantizing the same weights once per prompt.
        stem = source_path.stem
        if kind == "-":
            model = model_dir / f"{stem}.gguf"
            if not model.exists():
                model.symlink_to(source_path.resolve())
        else:
            model = model_dir / f"{stem}-{kind.lower()}.gguf"
            quantize(source_path, model, kind)
        print(f"case {name}", flush=True)
        write_case(out_dir / f"{name}.txt", name, model, prompt, version)


if __name__ == "__main__":
    main()
