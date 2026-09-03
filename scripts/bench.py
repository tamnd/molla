#!/usr/bin/env python3
"""Run the same work through molla, llama.cpp and Ollama and print one table.

D6 carries a reversal condition with a number in it and M7 has a gate with a
number in it, and neither is worth anything without a measurement taken the
same way twice. This is the thing that takes it. It starts now rather than at
M7 because a performance number first taken at the end is a number nobody can
bisect.

## What comparable means here

The same GGUF file, byte for byte, through all three. Not the same model at the
same quantization from three different downloads: the same file, with its
digest printed above the table, because a q4_K_M from one converter and a
q4_K_M from another are different numbers of bits in different places and the
difference is bigger than most of what this measures. Ollama gets the file
imported into its store with `ollama create`, which is the only way to make it
read a file somebody else chose.

The same prompt, and the same number of tokens out of it. The prompt is built
by repeating filler until molla's own tokenizer counts at least the number
asked for, then every engine is told that exact count. Prompt length changes
prefill throughput by a lot, so a table comparing 500 tokens against 512 would
be measuring the wrong thing quietly.

The same backend, named rather than defaulted. `--device` takes the same
spellings molla takes and each engine is given the matching flag, so a row that
says cuda is a row where all three were on the card.

## What each engine can and cannot report

Not every number is available from every engine, and a dash in the table is a
dash rather than a zero.

molla reports prefill and decode separately and this reads them off `molla
generate`. Peak memory comes from `wait4` on Linux, and on macOS from the
larger of `wait4` and `/usr/bin/time -l`, so it is the real maximum of the
process this script started rather than a figure the engine reports about
itself. The two platforms are asked different questions on purpose and `TIMED`
says why.

llama.cpp is measured with `llama-bench`, which runs prefill and decode as
separate passes and reports tokens per second for each. It does not report
time to first token, so that column is derived as the prompt divided by the
prefill rate, which is the same quantity measured a different way rather than a
second measurement.

Ollama is asked through `/api/generate` with `stream` off, which returns the
counts and the durations of both halves. Its prefill is the first run and no
other, because the server keeps the keys and values of the last prompt it saw
and this sends the same prompt every time, which turns the second run into a
cache hit reported as a throughput. The work happens inside a server this
script did not start, so peak resident bytes are not attributable and are left
empty. Running one is not a substitute: the server holds other models and the
number would be about the server rather than about the run.

## Usage

    scripts/bench.py MODEL.gguf TOKENIZER.json [options]

      --device=X        auto, cpu, metal, cuda, or an api and an index
      --prompt=N        prompt tokens, default 512
      --decode=N        tokens to generate, default 128
      --runs=N          repetitions per engine, median reported, default 3
      --molla=PATH      the molla binary, default build/molla
      --only=a,b        run only these engines of molla, llama.cpp, ollama
      --markdown        print the table as markdown for docs/validation

Needs no Python packages. Whichever of the three rivals is not installed is
reported as absent rather than skipped silently, because a table with two rows
in it and no explanation reads like the third one lost.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
FILLER = (
    "The history of computing is a history of people deciding what a machine"
    " should be allowed to forget. "
)


class Result:
    """One engine's numbers, with a dash for anything it cannot say."""

    def __init__(self, engine: str, backend: str):
        self.engine = engine
        self.backend = backend
        self.prefill_tps: float | None = None
        self.decode_tps: float | None = None
        self.ttft_ms: float | None = None
        self.peak_bytes: int | None = None
        self.note = ""

    def failed(self) -> bool:
        return self.decode_tps is None


def die(message: str) -> None:
    sys.exit("bench: " + message)


def digest(path: pathlib.Path) -> str:
    """The first twelve hex digits of the file's sha256.

    The whole hash of a five gigabyte file is a line nobody reads. Twelve digits
    is enough to tell two quantizations of one model apart, which is the thing
    this is here to catch.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:12]


def load_now() -> str:
    """The one minute load average, or a word saying it is not available.

    Every machine in this fleet is shared with something, and a run taken next
    to somebody else's build is not a measurement. The number belongs in the
    header beside the date, because a table that does not say what else was
    running is a table nobody can defend a year later.
    """
    try:
        return f"{os.getloadavg()[0]:.1f}"
    except (OSError, AttributeError):
        return "unavailable"


def peak_of(rusage: object) -> int:
    """Maximum resident bytes, from a unit that is not the same on both."""
    value = int(getattr(rusage, "ru_maxrss", 0))
    if sys.platform == "darwin":
        return value
    return value * 1024


TIMED = ["/usr/bin/time", "-l"] if sys.platform == "darwin" else []
"""What a child is wrapped in on macOS, and nothing anywhere else.

macOS has two answers to how much memory a process used and neither one is
right on its own, so the table takes the larger of the two and this says why.

`ru_maxrss`, which is what `wait4` returns, does not count the pages behind a
Metal buffer, and on Apple silicon those pages are host memory because the GPU
pool is the same RAM the process is in. A run of the 8B reported 5347 MiB for a
process that had just uploaded 9572 MiB of weights and 512 MiB of key value
cache.

`phys_footprint`, which `/usr/bin/time -l` prints as "peak memory footprint",
does count them. It reported 10212 MiB for that same run, which is the weights
plus the cache plus about 130 MiB of everything else, so it reconciles with
what the engine says it did. But it excludes clean file backed pages, on the
grounds that they are reclaimable, and llama.cpp holds its weights in a mapping
of the model file. On SmolLM2 llama.cpp is 236 MiB of resident set and 105 MiB
of footprint, and the 138 MiB between them is the model, which is really there
and really occupying RAM.

So one number hides molla's weights and the other hides llama.cpp's, and the
larger of the two is a lower bound that hides neither. That is not a perfect
measure of what a process cost the machine, and there is no perfect one, but it
is the only choice here that does not favour one engine's way of holding a
model over the other's.

Linux needs none of this. CUDA weights genuinely are not host memory, so the
resident set is the right number and the only number.
"""

TIME_L_HEAD = re.compile(r"^\s*[\d.]+ real\s", re.M)
TIME_L_PEAK = re.compile(r"^\s*(\d+)\s+peak memory footprint\s*$", re.M)


def split_timing(out: str) -> tuple[int, str]:
    """Peel `/usr/bin/time -l`'s report off the end of a child's output.

    The child's stdout and stderr are merged, and `time` writes its report to
    stderr after the child has exited, so the report is the tail of the merged
    text and nothing of the child's follows it. The last line matching "N real"
    is where it starts, last rather than first in case an engine printed
    something that looks like one.

    Returns zero and the text unchanged when there is no report, which is every
    platform that is not macOS and also a child that could not be started.
    """
    start = -1
    for m in TIME_L_HEAD.finditer(out):
        start = m.start()
    if start < 0:
        return 0, out
    found = TIME_L_PEAK.search(out, start)
    return (int(found.group(1)) if found else 0), out[:start]


def capture(args: list[str], env: dict | None = None) -> tuple[str, int, int]:
    """Run to completion and return its output, exit code and peak bytes.

    `wait4` rather than `getrusage`, because the process wide child maximum
    never goes back down and would report the largest engine of the run for
    every engine after it. On macOS `wait4` gives one of two incomplete answers
    and the child is wrapped to get the other, then the larger is taken.
    `TIMED` says why.

    Windows has no `wait4` and no rusage, so the peak comes back as zero there
    and the memory column is empty for the whole run. Reading it out of the job
    object would mean a package this script does not have, and a number that
    means something slightly different from the one the other two machines
    report is worse in a comparison table than no number.
    """
    proc = subprocess.Popen(
        [*TIMED, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    raw = proc.stdout.read() if proc.stdout else ""
    footprint, out = split_timing(raw)
    if not hasattr(os, "wait4"):
        return out, proc.wait(), footprint
    _, status, usage = os.wait4(proc.pid, 0)
    code = status >> 8 if status >= 256 else status
    return out, code, max(footprint, peak_of(usage))


def prompt_of(tokens: int, molla: str, tokenizer: str) -> tuple[str, int]:
    """Filler repeated until molla counts at least `tokens`, and the count.

    The count comes back from the engine rather than from an estimate here
    because the whole table depends on the three engines having been given the
    same amount of work, and a tokenizer disagreement of five per cent is a
    prefill throughput disagreement of five per cent.

    `molla tokenize` and not `molla generate`, because generate would have to
    load the weights and prefill the prompt to reach the line with the count on
    it, which on an eight gigabyte model is minutes per attempt for an answer
    the tokenizer alone can give in milliseconds.
    """
    text = FILLER
    for _ in range(40):
        out, code, _ = capture([molla, "tokenize", tokenizer, text])
        if code != 0:
            die("molla could not count the prompt:\n" + out[-800:])
        m = re.search(r"(\d+)", out)
        if m is None:
            die("molla printed no count:\n" + out[-800:])
        got = int(m.group(1))
        if got >= tokens:
            return text, got
        # Two tokens of slack per repeat, so a prompt that lands one short does
        # not spend another whole round trip getting there.
        want = tokens - got
        repeats = max(1, want * len(text) // max(1, got * len(FILLER)))
        text += FILLER * repeats
    die("could not build a prompt of that many tokens")
    return "", 0


def run_molla(
    molla: str,
    model: str,
    tokenizer: str,
    prompt: str,
    decode: int,
    device: str,
    runs: int,
) -> Result:
    out = Result("molla", device)
    prefills: list[float] = []
    decodes: list[float] = []
    ttfts: list[float] = []
    peak = 0
    for i in range(runs):
        text, code, bytes_used = capture(
            [
                molla,
                "generate",
                model,
                tokenizer,
                prompt,
                str(decode),
                "--device=" + device,
            ]
        )
        if code != 0:
            out.note = first_line(text)
            return out
        p = re.search(r"prefill:\s+(\d+) ms for (\d+) tokens", text)
        d = re.search(r"decode:\s+(\d+) ms for (\d+) tokens", text)
        if p is None or d is None:
            out.note = "molla printed no timing lines"
            return out
        p_ms, p_n = float(p.group(1)), int(p.group(2))
        d_ms, d_n = float(d.group(1)), int(d.group(2))
        if d_n == 0:
            out.note = "molla generated no tokens"
            return out
        if p_ms > 0:
            prefills.append(p_n * 1000.0 / p_ms)
        ttfts.append(p_ms)
        if d_ms > 0:
            decodes.append(d_n * 1000.0 / d_ms)
        # The first run is the one that faults the mapping in, so its peak is
        # the honest one and later runs read a warm page cache.
        if i == 0:
            peak = bytes_used
        peak = max(peak, bytes_used)
    if prefills:
        out.prefill_tps = statistics.median(prefills)
    if decodes:
        out.decode_tps = statistics.median(decodes)
    if ttfts:
        out.ttft_ms = statistics.median(ttfts)
    out.peak_bytes = peak
    return out


def llama_layers(device: str) -> str:
    """`-ngl`, which is llama.cpp's whole backend flag."""
    return "0" if device == "cpu" else "999"


def run_llama(
    model: str, prompt_tokens: int, decode: int, device: str, runs: int
) -> Result:
    out = Result("llama.cpp", device)
    binary = shutil.which("llama-bench")
    if binary is None:
        out.note = "llama-bench is not on PATH"
        return out
    text, code, peak = capture(
        [
            binary,
            "-m",
            model,
            "-p",
            str(prompt_tokens),
            "-n",
            str(decode),
            "-ngl",
            llama_layers(device),
            "-r",
            str(runs),
            "-o",
            "json",
        ]
    )
    if code != 0:
        out.note = first_line(text)
        return out
    rows = json.loads(text[text.index("[") : text.rindex("]") + 1])
    for row in rows:
        rate = float(row.get("avg_ts", 0))
        if int(row.get("n_prompt", 0)) > 0:
            out.prefill_tps = rate
        elif int(row.get("n_gen", 0)) > 0:
            out.decode_tps = rate
    if out.prefill_tps:
        out.ttft_ms = prompt_tokens * 1000.0 / out.prefill_tps
    out.peak_bytes = peak
    out.note = "time to first token derived from the prefill rate"
    return out


def ollama_up() -> bool:
    try:
        urllib.request.urlopen(f"http://{OLLAMA_HOST}/api/tags", timeout=2).read()
        return True
    except Exception:
        return False


def ollama_import(model: pathlib.Path, name: str) -> str | None:
    """Put this exact file in Ollama's store under a name of ours.

    `ollama create` rather than `ollama pull`, so the comparison is against the
    same bytes rather than against whatever the registry publishes under a
    similar name.
    """
    binary = shutil.which("ollama")
    if binary is None:
        return "ollama is not on PATH"
    listing = subprocess.run(
        [binary, "list"], capture_output=True, text=True
    ).stdout
    if name in listing:
        return None
    modelfile = pathlib.Path(f"/tmp/bench-{name}.Modelfile")
    modelfile.write_text(f"FROM {model.resolve()}\n")
    run = subprocess.run(
        [binary, "create", name, "-f", str(modelfile)],
        capture_output=True,
        text=True,
    )
    modelfile.unlink(missing_ok=True)
    if run.returncode != 0:
        return first_line(run.stderr + run.stdout)
    return None


def run_ollama(
    model: pathlib.Path, prompt: str, decode: int, device: str, runs: int
) -> Result:
    out = Result("ollama", device)
    name = "molla-bench-" + digest(model)
    problem = ollama_import(model, name)
    if problem is not None:
        out.note = problem
        return out
    if not ollama_up():
        out.note = f"nothing is listening on {OLLAMA_HOST}"
        return out

    options = {"num_predict": decode, "temperature": 0, "seed": 0}
    if device == "cpu":
        options["num_gpu"] = 0
    body = json.dumps(
        {
            "model": name,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "options": options,
        }
    ).encode()

    # One throwaway generation on a different prompt, so the model is resident
    # and the clocks are up before the run that counts, and so the prompt that
    # counts is not already in the server's cache when it is first asked.
    warm = json.dumps(
        {
            "model": name,
            "prompt": "warm",
            "stream": False,
            "raw": True,
            "options": {"num_predict": 1, "temperature": 0, "seed": 0},
        }
    ).encode()
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                f"http://{OLLAMA_HOST}/api/generate",
                data=warm,
                headers={"content-type": "application/json"},
            ),
            timeout=600,
        ).read()
    except Exception as e:
        out.note = str(e)
        return out

    decodes: list[float] = []
    prefill = None
    ttft = None
    for i in range(runs):
        request = urllib.request.Request(
            f"http://{OLLAMA_HOST}/api/generate",
            data=body,
            headers={"content-type": "application/json"},
        )
        try:
            answer = json.loads(urllib.request.urlopen(request, timeout=600).read())
        except Exception as e:
            out.note = str(e)
            return out
        p_n = int(answer.get("prompt_eval_count", 0))
        p_ns = int(answer.get("prompt_eval_duration", 0))
        d_n = int(answer.get("eval_count", 0))
        d_ns = int(answer.get("eval_duration", 0))
        # Prefill from the first run and no other. The server keeps the keys and
        # values of the last prompt it saw, and this sends the same prompt every
        # time, so the second run reports the full token count against almost no
        # time and comes out three or four times faster than the same engine
        # underneath llama.cpp. Decode is not cached and keeps its median.
        if i == 0 and p_ns > 0:
            prefill = p_n * 1e9 / p_ns
            ttft = p_ns / 1e6
        if d_ns > 0:
            decodes.append(d_n * 1e9 / d_ns)
    out.prefill_tps = prefill
    out.ttft_ms = ttft
    if decodes:
        out.decode_tps = statistics.median(decodes)
    out.note = (
        "prefill is the first run only, because the server caches the last"
        " prompt. Peak resident bytes belong to the server, not to the run"
    )
    return out


def first_line(text: str) -> str:
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()[:160]
    return "no output"


def molla_build(molla: str) -> str:
    out = subprocess.run([molla, "version"], capture_output=True, text=True)
    m = re.search(r"molla (\S+)", out.stdout)
    return m.group(1) if m else "unknown"


def llama_build() -> str:
    binary = shutil.which("llama-cli") or shutil.which("llama-bench")
    if binary is None:
        return "absent"
    out = subprocess.run([binary, "--version"], capture_output=True, text=True)
    m = re.search(r"build (\d+), commit ([0-9a-f]+)", out.stderr + out.stdout)
    return f"b{m.group(1)} {m.group(2)}" if m else "unknown"


def ollama_build() -> str:
    binary = shutil.which("ollama")
    if binary is None:
        return "absent"
    out = subprocess.run([binary, "--version"], capture_output=True, text=True)
    m = re.search(r"version is (\S+)", out.stdout + out.stderr)
    return m.group(1) if m else "unknown"


def number(value: float | None, places: int = 1) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def mib(value: int | None) -> str:
    return "-" if not value else str(value // (1 << 20))


def table(results: list[Result], markdown: bool) -> str:
    head = ["engine", "prefill tok/s", "decode tok/s", "ttft ms", "peak MiB"]
    rows = [
        [
            r.engine,
            number(r.prefill_tps),
            number(r.decode_tps),
            number(r.ttft_ms, 0),
            mib(r.peak_bytes),
        ]
        for r in results
    ]
    if markdown:
        lines = ["| " + " | ".join(head) + " |"]
        lines.append("| " + " | ".join("---" for _ in head) + " |")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)
    width = [
        max(len(head[i]), *(len(row[i]) for row in rows)) for i in range(len(head))
    ]
    lines = ["  ".join(head[i].ljust(width[i]) for i in range(len(head)))]
    for row in rows:
        lines.append("  ".join(row[i].ljust(width[i]) for i in range(len(head))))
    return "\n".join(lines)


def main() -> None:
    args = sys.argv[1:]
    positional = [a for a in args if not a.startswith("--")]
    flags = {a.split("=")[0]: a.split("=", 1)[-1] for a in args if a.startswith("--")}
    if len(positional) < 2:
        sys.exit(__doc__.split("## Usage")[1].strip())

    model = pathlib.Path(positional[0])
    tokenizer = positional[1]
    if not model.exists():
        die(f"no model at {model}")

    device = flags.get("--device", "auto")
    want_prompt = int(flags.get("--prompt", "512"))
    decode = int(flags.get("--decode", "128"))
    runs = int(flags.get("--runs", "3"))
    molla = flags.get("--molla", "build/molla")
    only = flags.get("--only", "molla,llama.cpp,ollama").split(",")
    markdown = "--markdown" in flags

    if not pathlib.Path(molla).exists() and shutil.which(molla) is None:
        die(f"no molla binary at {molla}, build it first")

    print(f"building a prompt of at least {want_prompt} tokens", file=sys.stderr)
    prompt, prompt_tokens = prompt_of(want_prompt, molla, tokenizer)

    # Before the run rather than after it, because after it is molla's own
    # load and not the load molla was competing with.
    before = load_now()

    # Resolved once, so every row says which card rather than saying auto.
    resolved = device
    if device == "auto":
        probe, code, _ = capture(
            [molla, "generate", str(model), tokenizer, "hi", "1"]
        )
        m = re.search(r"backend:\s+(\S+)", probe)
        if code == 0 and m:
            resolved = m.group(1)

    results = []
    if "molla" in only:
        print("molla", file=sys.stderr)
        results.append(
            run_molla(molla, str(model), tokenizer, prompt, decode, device, runs)
        )
    if "llama.cpp" in only:
        print("llama.cpp", file=sys.stderr)
        results.append(run_llama(str(model), prompt_tokens, decode, device, runs))
    if "ollama" in only:
        print("ollama", file=sys.stderr)
        results.append(run_ollama(model, prompt, decode, device, runs))

    stamp = time.strftime("%Y-%m-%d %H:%M %Z")
    print()
    # No host name. What matters about the machine for a number is the
    # operating system, the instruction set and how many cores it has, and the
    # name it answers to on this network is not any of that.
    cores = os.cpu_count() or 0
    print(
        f"machine   {platform.system()} {platform.machine()},"
        f" {cores} logical cores"
    )
    print(f"backend   {resolved}")
    print(f"model     {model.name} {digest(model)}")
    print(f"prompt    {prompt_tokens} tokens, {decode} generated, {runs} runs")
    print(f"builds    molla {molla_build(molla)}, llama.cpp {llama_build()},"
          f" ollama {ollama_build()}")
    print(f"load      {before} before the run, on {cores} logical cores")
    print(f"when      {stamp}")
    print()
    print(table(results, markdown))
    for r in results:
        if r.note:
            print(f"{r.engine}: {r.note}")


if __name__ == "__main__":
    main()
