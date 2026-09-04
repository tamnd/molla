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

molla 0.4.5, llama.cpp b1 de8656b, Ollama 0.32.9, on 2026-09-04. All three on `--device=cuda`, 512 prompt tokens asked for, 128 generated, median, and the load average was between 2.9 and 8.4 before each run. These are the lengths M7 takes its gate at.

llama.cpp reports itself as build 1 on this machine because it was built there from a clone with no tags, and the build number comes from `git describe`. The commit is the identity that matters and it is in the line above.

There are two memory columns and the second one is the one that means anything here. The host column is resident set, which on a card holds whatever the engine happened to leave in host memory and nothing about the model, and it is not comparable between the two engines either, since llama.cpp keeps the model in a mapping whose pages are resident and molla streams the file to the card through a bounded arena. The card column is the most the card held while the engine ran, over what it held before. `CardWatch` in `scripts/bench.py` says what that does and does not measure.

SmolLM2 135M Instruct Q8_0, digest 5a1395716f79, 514 prompt tokens, 5 runs.

| engine | prefill tok/s | decode tok/s | ttft ms | host MiB | card MiB |
| --- | --- | --- | --- | --- | --- |
| molla | 10489.8 | 247.6 | 49 | 282 | 656 |
| llama.cpp | 32487.0 | 886.8 | 16 | 443 | 694 |
| ollama | 15667.9 | 684.4 | 33 | - | - |

Qwen 2.5 0.5B Instruct q4_K_M, digest 74a4da8c9fdb, 514 prompt tokens, 5 runs.

| engine | prefill tok/s | decode tok/s | ttft ms | host MiB | card MiB |
| --- | --- | --- | --- | --- | --- |
| molla | 5039.2 | 288.3 | 102 | 556 | 1110 |
| llama.cpp | 43338.7 | 772.6 | 12 | 806 | 1120 |
| ollama | 7320.8 | 495.1 | 70 | - | - |

Llama 3.1 8B Instruct q4_K_M, digest 7b064f5842bf, 515 prompt tokens, 3 runs.

| engine | prefill tok/s | decode tok/s | ttft ms | host MiB | card MiB |
| --- | --- | --- | --- | --- | --- |
| molla | 378.7 | 85.8 | 1360 | 1092 | 7404 |
| llama.cpp | 10548.6 | 161.6 | 49 | 5047 | 5198 |
| ollama | 5919.2 | 148.1 | 87 | - | - |

## macbook, the M4

An M4 with 10 cores, and a shared working machine whose load average is regularly above sixty. The run below caught it at 2.3, which is as quiet as it gets here, so these are better than the indicative numbers this section used to hold. They are still one machine on one afternoon and not a measurement anybody should bisect against. They are here because they are the only Metal numbers in the set, and Metal is half of what M2b built.

There is one memory column and not two. Unified memory means the card figure would be the host figure counted twice, and `nvidia-smi` is what the second column is read with anyway.

molla 0.4.5, llama.cpp b10621 c1d0e7a00, Ollama not installed, on 2026-09-04. 512 prompt tokens asked for, 128 generated, 3 runs, median, the same lengths as the gpc tables.

SmolLM2 135M Instruct Q8_0, digest 5a1395716f79, 514 prompt tokens, on `--device=metal`.

| engine | prefill tok/s | decode tok/s | ttft ms | host MiB |
| --- | --- | --- | --- | --- |
| molla | 472.0 | 48.9 | 1089 | 344 |
| llama.cpp | 9454.0 | 288.1 | 54 | 247 |
| ollama | - | - | - | - |

Qwen 2.5 0.5B Instruct q4_K_M, digest 74a4da8c9fdb, 514 prompt tokens, on `--device=metal`.

| engine | prefill tok/s | decode tok/s | ttft ms | host MiB |
| --- | --- | --- | --- | --- |
| molla | 169.4 | 23.9 | 3035 | 887 |
| llama.cpp | 3250.2 | 170.3 | 158 | 607 |
| ollama | - | - | - | - |

On `--device=cpu`, the same machine and the SmolLM2 file.

| engine | prefill tok/s | decode tok/s | ttft ms | host MiB |
| --- | --- | --- | --- | --- |
| molla | 8.3 | 6.5 | 62243 | 261 |
| llama.cpp | 635.8 | 220.6 | 808 | 391 |
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

Decode on CUDA is 3.6 times off llama.cpp on SmolLM2, 2.7 times off on Qwen and 1.9 times off on the 8B. The M7 gate is 1.5 times, so the 8B is close to it and the small models are not.

The decode gap shrinks as the model grows, which says the cost is fixed per layer rather than proportional to the work. molla launches about fifteen kernels per layer and 453 per token on SmolLM2, and an empty kernel on this machine costs 4.9 microseconds to launch, so 453 of them are 2.2 ms before any arithmetic happens. The 8B does more work per launch and gets a better ratio out of the same fixed cost. That reading is right about molla and it is not a reading anyone should generalize from, because llama.cpp on Metal issues about 580 dispatches per decoded token and does not care, which is in [engines.md](engines.md).

Prefill is 3.1 times off on SmolLM2, 8.6 on Qwen and 27.9 on the 8B. That is the only quantity on this page whose gap grows with the model, and a ratio that grows with model size is a scaling failure rather than a slow kernel. The cause is that the prefill matmul stages neither operand, so with a grid of one block per output row the whole activation tile is read from global memory `rows` times. #201 has the details and the fix.

Memory is much closer than this page used to say, and most of what it used to say was a reporting artefact. On the card molla holds 656 MiB against llama.cpp's 694 on SmolLM2 and 1110 against 1120 on Qwen, which is parity or better, and 7404 against 5198 on the 8B, which is 1.42 times. The 8B is the only real gap and it has two known causes: the repack cache is 6474 MiB against a 4693 MiB model file, which #182 and #183 are closing, and the KV cache is f32 where llama.cpp's is f16, which is #204.

Metal is further behind than CUDA and it is behind in a different shape. Prefill is 20.0 times off on SmolLM2 and 19.2 on Qwen, which is nearly the same ratio on two models an order of magnitude apart in size, where the CUDA gap went from 3.1 to 27.9 over the same pair. A flat ratio is a slow kernel and a growing one is a scaling failure, so the two backends are short for different reasons and the same fix will not close both. Decode is 5.9 and 7.1 times off, against 3.6 and 2.7 on the card, so the fixed per launch cost hurts more here.

The host path is the worst number on this page and nothing is currently pointed at it. On the same laptop and the same file, `--device=cpu` is 76.6 times off llama.cpp at prefill and 33.9 at decode, and llama.cpp on ten CPU cores beats molla on the GPU of the same machine at decode by a factor of four. That is a scalar loop against a NEON one and it is what a host path costs when every milestone so far has been about accelerators.

Both readings come from a run at load 2.3, which is the quietest this machine gets. Getting a dedicated Apple machine into the fleet is still the way to make a Metal number something to bisect against.

## Running it

```sh
pixi run bench MODEL.gguf TOKENIZER.json --device=cuda --prompt=512 --decode=128
```

The flags are `--device`, `--prompt`, `--decode`, `--runs`, `--molla`, `--only` and `--markdown`, and `scripts/bench.py` with no arguments prints them. It needs no Python packages. Whichever rival is not installed is reported as absent rather than skipped, because a table with two rows in it and no explanation reads like the third one lost.

CI does not run it. A shared runner cannot produce a number worth writing down, and a benchmark that fails a build because somebody else's job was scheduled next to it is a benchmark people learn to ignore.

## What this does not cover

Not a long prompt. Every table on this page is the 512 and 128 the harness defaults to, which is what M7 takes its gate at, and nothing here says what happens at eight thousand tokens of context. The tables used to differ in length between machines, which made the rows unreadable against each other, and they no longer do.

Not concurrency. Every engine here is answering one sequence, and Ollama and llama.cpp both do more than that. molla refuses a second sequence with a 503, so there is nothing to compare yet.

Not power and not tokens per watt. Worth having and not measurable with what is attached to these machines.
