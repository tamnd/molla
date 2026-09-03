# Speed against llama.cpp and Ollama

Issue #146. D6 carries a reversal condition with a number in it and M7 has a gate with a number in it, and neither is worth anything without a measurement taken the same way twice. `scripts/bench.py` is the thing that takes it. It runs now rather than at M7 because a performance number first taken at the end is a number nobody can bisect.

This is not a gate. The gate is at M7, where decode has to be within 1.5 times of llama.cpp on the same model, the same quantization and the same device. This page exists so the distance is known the whole way there.

## What comparable means here

The same GGUF file, byte for byte, through all three engines. Not the same model at the same quantization from three different downloads: the same file, with its digest in the table, because a q4_K_M from one converter and a q4_K_M from another are different numbers of bits in different places and the difference is bigger than most of what this measures. Ollama gets the file imported into its store with `ollama create`, which is the only way to make it read a file somebody else chose.

The same prompt and the same number of tokens out of it. The prompt is filler repeated until molla's own tokenizer counts at least the number asked for, and then every engine is told that exact count. Prompt length moves prefill throughput a long way, so a table comparing 500 tokens against 512 would be measuring the wrong thing quietly. Counting is `molla tokenize`, which needs no model file, because the obvious way to count was to run a generation and read its prompt line, and on an eight gigabyte model that costs a full prefill per attempt.

The same backend, named rather than defaulted. Every row of a table was produced with `--device` set to the same thing, and molla refuses a backend it cannot provide rather than falling back, which is the point of [backend.md](backend.md). A silent fallback here would produce a table of honest looking numbers about the wrong hardware.

## What each engine can and cannot say

A dash in a table is a dash and not a zero.

molla reports prefill and decode separately and the harness reads them off `molla generate`. Peak resident bytes come from `wait4` on the child, so they are the real maximum of the process the harness started, and not a process wide high water mark that would report the largest engine of the run for every engine after it.

llama.cpp is measured with `llama-bench`, which runs prefill and decode as separate passes and reports tokens per second for each. It does not report time to first token, so that column is the prompt divided by the prefill rate, which is the same quantity arrived at a different way rather than a second measurement.

Ollama is asked through `/api/generate` with `stream` off and `raw` on, which returns the counts and the durations of both halves. Its prefill is the first run and no other, because the server keeps the keys and values of the last prompt it saw and the harness sends the same prompt every time. Before that was fixed the second run reported the full token count against almost no time and Ollama came out three times faster at prefill than the same engine underneath llama.cpp. Decode is not cached and keeps its median. Peak resident bytes belong to a server the harness did not start, which also holds other models, so that cell is a dash rather than a number about the wrong process.

## gpc, the RTX 4090

An RTX 4090 with 24564 MiB, reached through WSL2 on a Windows machine, which is why the harness reports Linux x86_64 with 32 logical cores. It is a WSL2 CUDA result and not a Linux CUDA result, for the reason in [toolchain.md](toolchain.md): the only NVIDIA card in the fleet is in a Windows box. This is the quietest machine in the set and the one where a number is a measurement.

molla 0.3.3, llama.cpp b1 de8656b, Ollama 0.32.9, on 2026-09-03. All three on `--device=cuda`, 512 prompt tokens asked for, 128 generated, median, and the load average was between 0.1 and 0.6 before each run. These are the lengths M7 takes its gate at.

llama.cpp reports itself as build 1 on this machine because it was built there from a clone with no tags, and the build number comes from `git describe`. The commit is the identity that matters and it is in the line above.

SmolLM2 135M Instruct Q8_0, digest 5a1395716f79, 514 prompt tokens, 5 runs.

| engine | prefill tok/s | decode tok/s | ttft ms | peak MiB |
| --- | --- | --- | --- | --- |
| molla | 265.9 | 254.5 | 1933 | 1607 |
| llama.cpp | 29020.2 | 782.5 | 18 | 443 |
| ollama | 37824.7 | 651.9 | 14 | - |

Qwen 2.5 0.5B Instruct q4_K_M, digest 74a4da8c9fdb, 514 prompt tokens, 5 runs.

| engine | prefill tok/s | decode tok/s | ttft ms | peak MiB |
| --- | --- | --- | --- | --- |
| molla | 244.4 | 227.0 | 2103 | 2168 |
| llama.cpp | 42670.8 | 756.7 | 12 | 807 |
| ollama | 43229.6 | 517.4 | 12 | - |

Llama 3.1 8B Instruct q4_K_M, digest 7b064f5842bf, 515 prompt tokens, 3 runs.

| engine | prefill tok/s | decode tok/s | ttft ms | peak MiB |
| --- | --- | --- | --- | --- |
| molla | 75.6 | 66.5 | 6809 | 11066 |
| llama.cpp | 10609.9 | 161.2 | 49 | 5034 |
| ollama | 5578.9 | 144.1 | 92 | - |

## macbook, the M4

An M4 with 10 cores, and a shared working machine whose load average is regularly above sixty. These are indicative numbers and not measurements. They are here because they are the only Metal numbers in the set, and Metal is half of what M2b built.

molla 0.3.3, llama.cpp b10621 c1d0e7a00, Ollama not installed, on 2026-09-03. SmolLM2 135M Instruct Q8_0, digest 5a1395716f79, 134 prompt tokens, 32 generated, 3 runs, median.

On `--device=metal`.

| engine | prefill tok/s | decode tok/s | ttft ms | peak MiB |
| --- | --- | --- | --- | --- |
| molla | 25.7 | 24.7 | 5209 | 351 |
| llama.cpp | 8558.3 | 330.2 | 16 | 239 |
| ollama | - | - | - | - |

On `--device=cpu`, the same machine and the same file.

| engine | prefill tok/s | decode tok/s | ttft ms | peak MiB |
| --- | --- | --- | --- | --- |
| molla | 8.2 | 7.7 | 16409 | 261 |
| llama.cpp | 927.5 | 287.0 | 144 | 371 |
| ollama | - | - | - | - |

## server1, no accelerator

An AMD EPYC with four threads, no GPU, and neither rival installed. It is in the set to prove the harness reports an absent engine as absent rather than dropping the row, and to give the host path a second machine.

molla 0.3.3, on 2026-09-03. SmolLM2 135M Instruct Q8_0, digest 5a1395716f79, 134 prompt tokens, 32 generated, 3 runs, median, on `--device=cpu`.

| engine | prefill tok/s | decode tok/s | ttft ms | peak MiB |
| --- | --- | --- | --- | --- |
| molla | 1.5 | 1.7 | 91092 | 261 |
| llama.cpp | - | - | - | - |
| ollama | - | - | - | - |

llama.cpp and Ollama are reported as not on PATH, which is what a machine without them should produce.

1.7 tokens per second on four threads is the slowest host figure in the set, and the laptop's 7.7 is too noisy to divide by it and get anything meaningful. What the row is worth is a floor. Anybody running molla on a small cloud instance today gets roughly a token per second on a 135M model, and that is the number to hold this page against once the host path is worked on.

## What the numbers say

Five things, and only the first is about the gate.

Decode on CUDA is 3.1 times off llama.cpp on SmolLM2, 3.3 times off on Qwen and 2.4 times off on the 8B. The M7 gate is 1.5 times, so the distance is a factor of two beyond the gate rather than a rounding error.

The gap shrinks as the model grows, and that is the most useful line on this page. A cost that is a smaller fraction of a bigger model is a fixed cost per layer, not a slow kernel: a slow kernel would stay slow in proportion. molla launches about fifteen kernels per layer and 453 per token on SmolLM2, and an empty kernel on this machine costs 4.9 microseconds to launch, so 453 of them are 2.2 ms before any arithmetic happens at all, against the 3.9 ms a token actually takes. The 8B does more work per launch, so the same fixed cost buys a better ratio.

Prefill is not two times off, it is between 109 and 175 times off, and the reason is in the shape of molla's own numbers rather than in a comparison. On every row of every table molla's prefill rate and its decode rate are within a few per cent of each other. molla has no batched prefill: it runs the prompt through the same one token at a time path the decode uses, so a 514 token prompt costs 514 decodes. llama.cpp prefills the whole prompt as one matrix multiply. That is what puts 1933 ms of time to first token against 18 ms, and it is the single largest gap on this page.

Peak resident bytes on CUDA are 1607 MiB for a 138 MiB model, against llama.cpp's 443 MiB, and 11066 MiB for a 4.6 GiB model against 5034 MiB. The measurements fit `1312 MiB + 2.14x the model file size`, so there are two costs in there and only one of them is about the model. The host side of a device run holds the GGUF mapping and the planar repack cache mapping while the card holds a third copy, and none of it is freed once the weights are up.

The Metal path is much further behind than the CUDA path, and the laptop's load average means this page cannot say how much. Thirteen times off llama.cpp on a machine that is doing other work is not a number worth acting on. Getting a quiet Apple machine into the fleet is the way to make that one real.

## Running it

```sh
pixi run bench MODEL.gguf TOKENIZER.json --device=cuda --prompt=512 --decode=128
```

The flags are `--device`, `--prompt`, `--decode`, `--runs`, `--molla`, `--only` and `--markdown`, and `scripts/bench.py` with no arguments prints them. It needs no Python packages. Whichever rival is not installed is reported as absent rather than skipped, because a table with two rows in it and no explanation reads like the third one lost.

CI does not run it. A shared runner cannot produce a number worth writing down, and a benchmark that fails a build because somebody else's job was scheduled next to it is a benchmark people learn to ignore.

## What this does not cover

Not a long prompt on the machines that are not gpc. The gpc tables are the 512 and 128 the harness defaults to and are the ones to hold M7 against. The macbook and server1 tables are 134 and 32, because at those machines' rates the default lengths are a coffee break rather than a measurement.

Not concurrency. Every engine here is answering one sequence, and Ollama and llama.cpp both do more than that. molla refuses a second sequence with a 503, so there is nothing to compare yet.

Not power and not tokens per watt. Worth having and not measurable with what is attached to these machines.
