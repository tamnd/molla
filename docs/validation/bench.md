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

molla 0.4.13, llama.cpp b1 de8656b, Ollama 0.32.9, on 2026-09-05. All three on `--device=cuda`, 512 prompt tokens asked for, 128 generated, five runs, median, and the load average was between 0.6 and 0.7 before each run. These are the lengths M7 takes its gate at.

All three rows of each table were taken in one sitting on this build. The last row of each is molla 0.4.11, measured the same way the day before, and that is the row the molla row should be read against.

What separates the two molla rows is #235, the decode attention split. It only reaches the 8B, because the two small models decode through the fused path and the fused kernel has cut a head's keys across blocks since 0.4.10. So the first two tables are the control and the third is the change, which is the reverse of the arrangement this page carried for #200.

Run to run spread on this harness is larger than the two small rows moved. Three molla only reruns of SmolLM2 in the same sitting read 467.2, 474.1 and 458.8 decode, a spread of 3.3 per cent, so the 472.3 in the table against 490.4 the day before is not a measurement of anything. The 8B row is 17 per cent and is far outside it.

That 17 per cent is smaller than the 30 per cent #235 measured at a 1121 token prompt, and the difference is the prompt. This page is 512 tokens, the term the split addresses grows with the context, and the ratio therefore grows with it too. Neither number contradicts the other and the longer one is in [budget.md](budget.md).

Ollama's prefill column in this sitting reads 124154 and 112472 tokens a second on the two small models, and those are not measurements. The Ollama server caches the last prompt, the harness takes the first run only for that reason, and on a warm server even the first run can hit the cache. The day before, the same two cells read 16138.7 and 6655.0. Treat the Ollama prefill cells as an upper bound with no lower bound under them, which is what the footnote the harness prints has always said and what these two numbers make impossible to ignore. The Ollama decode column does not have the problem, because 128 generated tokens are generated either way.

llama.cpp reports itself as build 1 on this machine because it was built there from a clone with no tags, and the build number comes from `git describe`. The commit is the identity that matters and it is in the line above.

There are two memory columns and the second one is the one that means anything here. The host column is resident set, which on a card holds whatever the engine happened to leave in host memory and nothing about the model, and it is not comparable between the two engines either, since llama.cpp keeps the model in a mapping whose pages are resident and molla streams the file to the card through a bounded arena. The card column is the most the card held while the engine ran, over what it held before. `CardWatch` in `scripts/bench.py` says what that does and does not measure.

SmolLM2 135M Instruct Q8_0, digest 5a1395716f79, 514 prompt tokens, 5 runs.

| engine | prefill tok/s | decode tok/s | ttft ms | host MiB | card MiB |
| --- | --- | --- | --- | --- | --- |
| molla | 10280.0 | 472.3 | 50 | 279 | 648 |
| llama.cpp | 37676.6 | 857.8 | 14 | 443 | 642 |
| ollama | 124154.6 | 703.9 | 4 | - | - |
| molla 0.4.11 | 10708.3 | 490.4 | 48 | 279 | 648 |

Qwen 2.5 0.5B Instruct q4_K_M, digest 74a4da8c9fdb, 514 prompt tokens, 5 runs.

| engine | prefill tok/s | decode tok/s | ttft ms | host MiB | card MiB |
| --- | --- | --- | --- | --- | --- |
| molla | 4990.3 | 371.0 | 103 | 549 | 918 |
| llama.cpp | 37256.0 | 749.6 | 14 | 807 | 1122 |
| ollama | 112472.6 | 517.0 | 5 | - | - |
| molla 0.4.11 | 5039.2 | 383.2 | 102 | 550 | 930 |

Llama 3.1 8B Instruct q4_K_M, digest 7b064f5842bf, 515 prompt tokens, 5 runs.

| engine | prefill tok/s | decode tok/s | ttft ms | host MiB | card MiB |
| --- | --- | --- | --- | --- | --- |
| molla | 378.7 | 111.9 | 1360 | 976 | 6084 |
| llama.cpp | 10475.2 | 161.1 | 49 | 5047 | 5198 |
| ollama | 6105.9 | 147.9 | 84 | - | - |
| molla 0.4.11 | 372.9 | 95.5 | 1381 | 1044 | 6084 |

## macbook, the M4

An M4 with 10 cores, and a shared working machine whose load average is regularly above sixty. The run below caught it at 2.3, which is as quiet as it gets here, so these are better than the indicative numbers this section used to hold. They are still one machine on one afternoon and not a measurement anybody should bisect against. They are here because they are the only Metal numbers in the set, and Metal is half of what M2b built.

There is one memory column and not two. Unified memory means the card figure would be the host figure counted twice, and `nvidia-smi` is what the second column is read with anyway.

molla 0.4.7, llama.cpp b10621 c1d0e7a00, Ollama not installed, on 2026-09-04. 512 prompt tokens asked for, 128 generated, 3 runs, median, the same lengths as the gpc tables.

All three Metal tables were retaken together after #203 and the last row of each is what molla did before it, from the same clone on the same afternoon. The load average moved between 3.2 and 10.6 over that sitting, which is why the llama.cpp rows are lower than the ones this page carried a version ago even though llama.cpp did not change. Read the last row against the first and not against anything on an earlier version of this page.

They understate 0.4.12 by more again. #229 narrows the matvec block on Metal from 128 threads a row to 32 and that is worth 1.4 times on SmolLM2, 1.8 on Qwen and 1.5 on the 8B, measured as both builds a minute apart. An attempt at retaking this section with it, on 2026-09-04 at a load average of 5, read llama.cpp at 133.4 tokens a second on SmolLM2 where a run twenty minutes earlier on the same machine and the same binary read 306.3. A rival column that moves by 2.3 times in twenty minutes is not a measurement, so nothing from that sitting is on this page. The before and after pair is in [layout.md](layout.md), where a pair belongs.

The molla rows here are 0.4.7 and they understate 0.4.8. #183 moves Metal decode a long way on the two models that have five or six bit weights in them, and the tables are not retaken with it because the machine has not been quiet since it landed. An attempt at 16 to 52 gave molla 2.8 tok/s on the 8B where the row below says 5.1, on a build that is faster than that row and not slower, which is the whole argument for not publishing it. What #183 is worth is in [layout.md](layout.md) instead, as both builds a minute apart on the same machine, which survives a load this one does not. The Metal section gets retaken whole at the next quiet window.

SmolLM2 135M Instruct Q8_0, digest 5a1395716f79, 514 prompt tokens, on `--device=metal`.

| engine | prefill tok/s | decode tok/s | ttft ms | host MiB |
| --- | --- | --- | --- | --- |
| molla | 1932.3 | 36.5 | 266 | 362 |
| llama.cpp | 8615.0 | 292.2 | 60 | 246 |
| ollama | - | - | - | - |
| molla before #203 | 1778.5 | 31.1 | 289 | 371 |

Qwen 2.5 0.5B Instruct q4_K_M, digest 74a4da8c9fdb, 514 prompt tokens, on `--device=metal`.

| engine | prefill tok/s | decode tok/s | ttft ms | host MiB |
| --- | --- | --- | --- | --- |
| molla | 513.5 | 23.9 | 1001 | 912 |
| llama.cpp | 2541.6 | 129.7 | 202 | 596 |
| ollama | - | - | - | - |
| molla before #203 | 508.4 | 18.1 | 1011 | 914 |

Llama 3.1 8B Instruct q4_K_M, digest 7b064f5842bf, 515 prompt tokens, on `--device=metal`. New to this page. It is the model M7 takes its gate on and it was missing from the Metal side, and it is also the only one of the three that is mostly q4_K, which is the type the decode matvec is written for.

| engine | prefill tok/s | decode tok/s | ttft ms | host MiB |
| --- | --- | --- | --- | --- |
| molla | 29.5 | 5.1 | 17430 | 7280 |
| llama.cpp | 115.2 | 13.7 | 4472 | 4918 |
| ollama | - | - | - | - |
| molla before #203 | 22.6 | 2.7 | 22792 | 7279 |

Both engines are slow on that one because a 4.7 GiB model on this laptop does not fit anywhere comfortable, so the row says more about the machine than about either engine. The ratio between the rows is still the thing to read and it is the tightest of the three.

On `--device=cpu`, the same machine and the SmolLM2 file.

| engine | prefill tok/s | decode tok/s | ttft ms | host MiB |
| --- | --- | --- | --- | --- |
| molla | 8.3 | 6.5 | 62243 | 261 |
| llama.cpp | 635.8 | 220.6 | 808 | 391 |
| ollama | - | - | - | - |

## server1, no accelerator

An AMD EPYC with four threads, no GPU, and neither rival installed. It is in the set to prove the harness reports an absent engine as absent rather than dropping the row, and to give the host path a second machine.

molla 0.4.5, on 2026-09-04. SmolLM2 135M Instruct Q8_0, digest 5a1395716f79, 514 prompt tokens, 128 generated, 3 runs, median, on `--device=cpu`, at the same lengths as every other table here.

| engine | prefill tok/s | decode tok/s | ttft ms | peak MiB |
| --- | --- | --- | --- | --- |
| molla | 2.4 | 1.8 | 211952 | 263 |
| llama.cpp | - | - | - | - |
| ollama | - | - | - | - |

llama.cpp and Ollama are reported as not on PATH, which is what a machine without them should produce.

The load average was 28.2 on four logical cores before the run, so this row is a ceiling on how bad it gets rather than a measurement. It is here for two things and neither needs precision. One is that the absent engines are reported absent. The other is a floor: anybody running molla on a small cloud instance today gets under two tokens a second on a 135M model, and a 514 token prompt takes three and a half minutes to answer. That is the number to hold this page against once the host path is worked on.

## What the numbers say

Decode on CUDA is 1.8 times off llama.cpp on SmolLM2, 2.0 times off on Qwen and 1.4 times off on the 8B, and 1.5, 1.4 and 1.3 times off Ollama. The M7 gate is 1.5 times the other way, so the whole of it is still ahead. The 8B is the closest of the three for the first time, having been the furthest, and it is now inside Ollama on that quantity by a fifth.

The version before this one said the decode gap shrank as the model grew, and read that as a fixed cost a layer rather than a cost proportional to the work: about fifteen kernels a layer and 453 a token on SmolLM2, at 4.9 microseconds a launch, is 2.2 ms of submission before any arithmetic happens. That reading was right and #200 collected most of what it promised. A layer is one launch now, a token is 33 rather than 453, and SmolLM2 decode went from 264.5 to 492.3 and Qwen from 304.8 to 367.8.

What is left is not launches. 33 launches is 0.16 ms of a 2.03 ms SmolLM2 token, so submission is now eight per cent of it and the 1.8 times gap is somewhere else. Two candidates have numbers behind them already. The fused grid is a few hundred blocks where the unfused matvec launches one a row, which is the whole reason the 8B is faster unfused. And the decode matvec still multiplies one weight at a time in float where llama.cpp multiplies four at a time in integers, which is #186 and #202 and is measured in [layout.md](layout.md) rather than here.

The 8B row moved because of the other end of the same argument. Attention over the context launched one block per query head, which is 32 on that model, and a 4090 with 128 multiprocessors was three quarters idle for 38 per cent of the token. Cutting the keys into slices with a block each is #235 and it is 17 per cent here and 30 at a 1121 token prompt. What it leaves is a token that is 73 per cent projections, which read 4.72 GiB at 749 GB/s against a card rated at 1008, and that is a kernel problem rather than a plumbing one. [budget.md](budget.md) splits the token three ways and measures each part.

Prefill is 3.7 times off on SmolLM2, 7.5 on Qwen and 27.7 on the 8B. That is the only quantity on this page whose gap grows with the model, and a ratio that grows with model size is a scaling failure rather than a slow kernel. The cause is that the prefill matmul stages neither operand, so with a grid of one block per output row the whole activation tile is read from global memory `rows` times. #201 has the details and the fix.

Memory on the card is parity or better on the two small models and 1.17 times on the 8B. molla holds 648 MiB against llama.cpp's 642 on SmolLM2, 918 against 1122 on Qwen and 6084 against 5198 on the 8B. The layout work of M2c is what closed that: the 8B was 7038 MiB before #182 and #183 and its repack cache is now 5151 MiB against a 4685 MiB file. What is left of the 8B gap is the KV cache, which is f32 here and f16 there, and that is #204.

The host column is under llama.cpp on all three models now that #222 is fixed: 279 MiB against 443 on SmolLM2, 549 against 807 on Qwen and around a thousand against 5047 on the 8B. It read 1468 and 1481 on the two fused rows one sitting ago, which was one pinned host allocation and not the model, and the paragraph above the tables says what that was.

Metal prefill was 20.0 times off on SmolLM2 and 19.2 on Qwen before #201, which is nearly the same ratio on two models an order of magnitude apart in size, where the CUDA gap went from 3.1 to 27.9 over the same pair. A flat ratio is a slow kernel and a growing one is a scaling failure, so the two backends were short for different reasons and one fix was never going to close both.

The slow kernel is the one #201 replaced, and the ratio is now 4.5 and 4.9, with the 8B at 3.9. What is left of it is not the instruction. A dense tile of this shape reaches 3.04 TFLOP/s on this M4 and llama.cpp's own prefill here works out at 2.55, so llama.cpp is at 84 per cent of the ceiling and molla is at about a fifth of it. Metal memory went up with it, 344 to 405 MiB and 887 to 915, which is the wider prefill chunk the tile wants and is the trade the numbers say to take.

Decode on Metal moved for the first time since the device path was written, and what moved it was how many values one thread takes between two reads of the group scale. That number was two. It is eight now on Metal and the three models gained 1.17, 1.32 and 1.89 times, in that order, which is the order of their sizes. #203 asked for a different change to the same loop, the nibble masks from the Metal q4_K matvec in llama.cpp, and `scripts/matvec_probe.mojo` says that one is worth nothing once the step width is held equal. The step width is the whole effect.

The gain growing with the model is the same shape as the launch cost argument two paragraphs up, read from the other side. A wider step only pays on rows long enough to have a loop, and the 8B has rows three and a half times wider than Qwen's and sixteen times wider than SmolLM2's.

What is left on Metal is 8.0, 5.4 and 2.7 times at decode. The 8B is inside three and the gate is one and a half.

Qwen is the odd one of the three and the reason is the file rather than the engine. It is named q4_K_M and 133 of its tensors are q5_0, against 14 that are q4_K. Every five and six bit type is widened to a byte a value in the planar layout today, so that model is read at eight bits a value on the card, which is 60 per cent more bytes a token than it needs and is most of why it holds 912 MiB against llama.cpp's 596. #182 and #183 are that, and this table is the first place the cost of it is visible as a decode number rather than a memory one.

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
