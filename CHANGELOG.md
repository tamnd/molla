# Changelog

Notable changes per release. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.15] - 2026-09-05

Prefill on a 4090 runs 1.75 times faster on the two small models and 1.34 times on an 8B, and decode 1.08 to 1.15 times, because a thread of the batched matmul was issuing one load for every multiply and a narrow matvec row was reduced through eight barriers. A 1121 token prompt prefills in 67 ms against 126 on SmolLM2 and 2118 against 2925 on the 8B, and time to first token on the 8B is 987 ms against 1319. The output is unchanged.

### Changed

- A prefill thread carries four output rows rather than one. `SPAN` already amortized a weight read over tokens and nothing was doing the other half, so a step of the column loop was `SPAN` activation loads feeding `SPAN` multiplies, one load an operation, on a card that issues four times as many multiplies a cycle as loads. All three models prefilled between three and six TFLOP/s against a card that does eighty two in float, whatever their size, which is the shape of a kernel waiting on its load unit. Hoisting the activations out of a loop over four rows makes a step sixty four multiplies for twenty four loads. Sixteen rows is slower than one, because the accumulators stop fitting in registers, and so was four until the row loop was made to unroll.
- The prefill chunk is 256 on both backends rather than 256 on Metal and 64 on CUDA. Widening it used to cost the 8B half its prefill, which was the row count above: a thread held one row, so a wider chunk bought nothing on the row axis while the extra token blocks contended for the same rows and the scratch grew four times for it. It now gains on every model. It keeps gaining past 256 and the scratch is why it stops there, since the scores are `chunk * heads * context` floats and a chunk of 1024 on a thirty two head 8B is 268 MiB of them.
- A matvec row goes to a warp when it is narrow and to a block when it is wide, rather than always to a block. A 576 wide row over a block of 128 gives a thread four and a half values to carry eight barriers, and the probe measured the two 576 by 1536 projections of a SmolLM2 layer at 124 GB/s against 454 for the 1536 by 576 beside them, which is the same weights in the other orientation. A warp per row takes SmolLM2's decode from 238 ms to 220 for 128 tokens and Qwen's from 317.7 to 275.3. Doing it to every row instead cost the 8B 21 per cent, because a 4096 wide row already gives a thread thirty two values and the barriers were being paid for, so both matvecs ask `row_takes_a_warp` which of the two they are running. They have to agree: the two mappings add a row up in different orders and the fused path is held to a bit for bit match against the unfused one.

### Fixed

- The prefill chunk no longer has to be one number per backend. That was a workaround for a cause that is now understood and removed.

## [0.4.14] - 2026-09-05

A SmolLM2 decode on a 4090 runs 1.20 times faster and Qwen2.5 0.5B 1.09 times, because the fold at the end of attention stopped running on five blocks out of three hundred and eighty four. SmolLM2 is 571.4 tokens a second against 476 and Qwen is 414.2 against 363, and both now decode faster than ollama. The output is byte identical.

### Changed

- The fold that joins attention's slices gives each answer a warp rather than a thread. It has only `heads * head_dim` answers in it, 576 on SmolLM2, so walking it a thread at a time left eighteen warps on five blocks running two serial passes over the slices while the other three hundred and seventy nine blocks waited at the barrier for them. The array being folded is about 74 KB and stays in L2, so this was latency and not traffic. Lanes stride the slices and `lane_group_max` and `lane_group_sum` finish the reduction, which takes the attention step from 17.15 microseconds a layer to 6.88 and a layer's total from 1761.5 to 1454.1. An 8B does not change, because its fold has 4096 answers and was already spread wide.
- A query meets one key with a warp rather than a thread. Threads a whole key apart meant every load in flight pulled a 32 byte sector to use 4 bytes of it, and on a head count smaller than the tile most of the block sat idle. The lanes of a warp walk one key with a stride of 32 instead. All three attention kernels share one `key_dot`, because the fused layer and the unfused one are held to a bit for bit match and `lane_group_sum` reduces in a tree, so the order the products of a key are added in has to be one constant and one function for both. This is worth 7.5 per cent of the attention record and 1.4 per cent of an 8B token, and nothing on the small models, whose whole key cache is 23 MB against 72 MB of L2 and so was never losing anything to sectors.

### Added

- `scripts/fused_probe.mojo` prices one record of a fused layer. It launches the record range in prefixes and differences them, which works because every prefix pays the same launch a layer and reads the same table, so both cancel. It takes the prefix over every layer rather than repeating one, because a layer is 3.8 MB and the card has 72 MB of L2. This is what found the two changes above, and it corrected the reading they were first attempted under: attention's 42 GB/s was a latency figure and not a bandwidth one, and the coalescing fix that reading predicted was worth a twentieth of what the fold fix turned out to be.

## [0.4.13] - 2026-09-05

An 8B decode on a 4090 runs 1.42 times faster because attention over the context stopped launching 32 blocks. The token is 9.16 ms against 13.03, which is 109.2 tokens a second against 86.0, and the output is byte identical.

### Changed

- Decode attention cuts the keys into slices, one block each, and joins them in a second kernel. `device_attend` launched one block per query head, which is 32 on an 8B, and `scripts/attend_probe.mojo` showed that reading the same 1185 keys with 64 or 128 blocks cost the same 112 microseconds it cost with 32. The card was doing four times the work for free and a decode had no way to ask for it, which is why the term was moving bytes at 63 GB/s while the projection kernel beside it moved them at 749. A slice subtracts its own maximum and writes its weighted value sum unnormalised alongside that maximum and its sum, and the join rescales each by the gap between its maximum and the row's, which is exact rather than approximate. That is 3.4 times on the term at 1185 tokens of context and 4.0 at 2048. The fused path has done this since 0.4.10 and the unfused one, which is what every model over a gibibyte a token uses, had not. See [docs/validation/budget.md](docs/validation/budget.md).
- A prefill chunk takes the single kernel unchanged, because it already launches hundreds of blocks, and so does any context short enough that the slices would not be worth having. Prefill measures the same either way, 3094 ms against 3087 for 1122 tokens on the 8B. How many slices comes from the grid width and from the room the caller gave, so a caller that hands over a buffer with room for one slice gets exactly the kernel it got before, which is what the probe uses to measure the two against each other in the same process.
- A target of 256 blocks is what ships. 1024 was measured and is worse at the contexts that matter, 35 microseconds at 1185 against 32 and 47 at 2048 against 45, because past the point where the card is full the only thing more slices buy is a longer join.

### Fixed

- A slice that a window has masked end to end is dropped rather than normalised. `NEG_INF` here is a finite float, so subtracting it from itself gives zero and every masked key would have come back weighted one. The single kernel never met this, because `device_attend` refuses a position that can see no keys at all, but a slice of a row that can see some is free to see none. Three cases at 512 keys cover it: a plain split, a window that empties the first three slices, and sinks that leave the middle two empty.
- The device matvec test wrote its group scales as float32 at a four byte stride into rows sized for float16. Host reference and device kernel read the same wrong bytes and agreed, so the test passed while checking nothing about scales, and the last row wrote three bytes past the pool.

### Removed

- The claim that the 8B decode reads weights at 630 GB/s where the card gives 901. It was never measured. It came out of a fit with two free parameters over two measured points, which will reproduce those two points whatever the parameters mean, and both of its terms are wrong in opposite directions: the fixed term came to 5.73 ms a token where `scripts/launch_probe.mojo` prices the same launches at about 1.9, and the bandwidth term came to 630 where `scripts/proj_probe.mojo` times the shipped kernel at the 8B's own ten shapes and gets 749. The largest of those shapes reads at 929. `docs/validation/layout.md` and `docs/validation/performance.md` carry the correction and the issue it came from is withdrawn.

## [0.4.12] - 2026-09-04

A Metal decode runs 1.4 to 1.8 times faster because the matvec stopped launching a block four times wider than the row it was reducing. Trying to remove the reduction underneath it as well is worth nothing, which is how this release also establishes that decode on the small models is bound by how many times a token crosses the driver and not by the kernel.

### Changed

- The matvec launches 32 threads a row on Metal rather than 128, which is worth 1.4 times on SmolLM2 135M, 1.8 on Qwen 2.5 0.5B and 1.5 on Llama 3.1 8B, measured as both builds a minute apart on an M4. 128 was never the answer to a question, it is the width every other kernel launches at and the matvec took it because it was there. Two things cost at 128 and both get worse as the row gets narrower: the reduction is a tree over the whole block, so what it costs is the block width and not the work in it, and a 576 column row at eight values a thread is 72 threads of work, so 56 of the 128 arrive at that reduction with nothing in them. The output head of a small model, 49152 rows of 576, was running at a tenth of what the same bytes read with no arithmetic on them. CUDA keeps 128 until the same sweep runs on a 4090. The logit corpus passes all thirteen cases on both widths and every figure in it moves in the fifth or sixth significant digit, because the additions happen in a different order.
- A shuffle reduction over that narrower block was written, measured and reverted. With the block at one simd group the tree through shared memory can be five `lane_group_sum` shuffles with no barriers, which is how the batched matmul has reduced since `MM_TILE` was measured, and on three alternating runs of each binary it is a wash on SmolLM2 and on Qwen. It is not free, since it changes the order the partial sums are added in, so it is not worth taking for nothing. What the negative result says is that the block width was never costing the reduction, it was costing the threads that arrived at a narrow row with no work in them, and that the matvec is no longer where a small model's decode goes. The three models fit a line of 29.4 ms a GiB with a 13.6 ms a token intercept that does not depend on model size, and a thirty layer token is around 450 launches at 19.6 us each. See [docs/validation/layout.md](docs/validation/layout.md).

## [0.4.11] - 2026-09-04

A fused session holds a fifth of the host memory it held yesterday. The 1.2 GiB it carried was one pinned allocation made to read three integers back from the grid barrier, not the compile it was blamed on, and reading them the ordinary way takes SmolLM2 on a 4090 from 1468 MiB to 279 and Qwen from 1481 to 550, both of which are under llama.cpp on the same file in the same sitting.

### Fixed

- A fused session holds 279 MiB of host memory on SmolLM2 rather than 1468. The 1.2 GiB was read as the price of compiling the fused kernel, which 0.4.10 says in as many words, and it was not: it was `enqueue_create_host_buffer`, which allocates pinned memory, and the first pinned allocation in a process costs 1.2 GiB whatever its size. The three words it was allocated for are the ones `fused_selftest` reads back to find out whether the grid reached a barrier, and an ordinary device to host copy into a list reads them for nothing. Compiling once rather than twice and cutting the six inlined quant forms down to one had both left the figure at 1468, which is what said the compile was not where it was going.

## [0.4.10] - 2026-09-04

A decode runs a layer in one launch on CUDA. A Llama shaped layer used to be twelve launches and a thirty layer token 363 of them, and it is 33 now. On a 4090 at a 512 token prompt and 128 generated, SmolLM2 135M goes from 264.5 to 492.3 tokens a second and Qwen 2.5 0.5B from 304.8 to 367.8, both against llama.cpp at 888.9 and 767.9 and Ollama at 682.0 and 490.8 in the same sitting.

It is off for a large model and off on Metal, and both of those are measurements rather than caution. Llama 3.1 8B decodes at 81.8 tokens a second fused against 93.7 unfused, because a few hundred blocks is narrower than the block a row the unfused matvec launches and a model that reads five gibibytes a token is bound by that rather than by submission.

### Added

- A decode runs a layer in one launch on the GPU. A Llama shaped layer used to be twelve launches and a thirty layer token 363 of them, which is 1.75 ms of pure submission on a 4090 and 7.30 on an M4 before any arithmetic happens. A layer is now a table of fixed width records in device memory, built once when the model binds, walked by one kernel with a grid wide barrier between the records that depend on each other, which takes a token to 33 launches. On a 4090 at 128 tokens decoded that is 483.0 tokens a second against 310.7 on SmolLM2 135M and 372.1 against 316.8 on Qwen 2.5 0.5B from a short prompt, and 460.4 against 254.0 and 367.8 against 268.9 from a 512 token one.
- The attention step splits a head's keys across blocks. A block used to own a whole head, so a grid of 384 blocks put at most thirty two of them on the one step whose cost grows with the context, and the fused path lost nine per cent on Qwen at a 512 token prompt while winning fourteen at eight tokens. A head is now cut into slices that reduce against their own maximum in the flash attention way, folded together after one extra barrier a layer, which is why the ratios above grow with the context instead of collapsing.
- The path is on by default for a model whose layer matrices come to a gibibyte a token or less, and off above that. Llama 3.1 8B reads 5151 MiB a token and decodes at 81.8 tokens a second fused against 93.7 unfused, because a few hundred blocks is narrower than the block a row the unfused matvec launches and a large model is bound by that rather than by submission. It is off on Metal whatever the model, where an M4 holds a fifth of the blocks a 4090 does. `MOLLA_FUSED` overrides the choice either way and `MOLLA_FUSED_BLOCKS` overrides the grid width.
- A session that is not going to use the fused path does not build a plan for one. Sizing the grid compiles the fused kernel, because the occupancy query is a question about a compiled function, and compiling it costs about 1.2 GiB of host memory. The 8B on gpc holds 900 MiB resident where it held 1493 with the plan built. What the compile costs a session that does use the path is #222.

### Fixed

- A repack that fails says why. A load that cannot write its cache put the reason in the report and the device path threw the report away, reopened the cache and raised with whatever was wrong with the file the repack did not replace. On a full disk with a stale cache beside the model that read as `the cache is an older weight layout`, which is a true sentence about the wrong file and sends the reader to a layout bug that is not there. The report now carries `repack_written` and the device path raises with the repack's own reason when nothing was published. ENOSPC and EACCES are also spelled out rather than numbered, because a five gibibyte write that fails with a bare 28 reads as a bug in the writer.

## [0.4.9] - 2026-09-04

The scale planes are float16. A planar row ends in one or two planes of group scales, and until now each of those was a float32, which was four bytes to hold a number that came out of the file as two. For q4_0, q4_1, q5_0, q5_1 and q8_0 the scale is a float16 out of the block, so storing it at float32 was widening a number on the way in and narrowing it back on the way out with nothing in between. For q4_K, q5_K and q6_K the plane holds a product of a float16 and a small integer, and that product is the one thing in the layout that now rounds.

It finishes the layout M2c set out to build. Llama 3.1 8B q4_K_M repacks to 5151 MiB against the 5153 that was predicted for it before any of the three pieces landed, down from 9574 where the milestone started. The file itself is 4685 MiB, so the whole of what molla holds above the weights is now 466 MiB rather than 4889.

The rounding is worth stating precisely, since it is the first lossy step in the layout. A float16 has eleven bits of mantissa, so one rounded scale is off by at most 2.4e-4 of itself and the corpus says the logits move less than that: 1.1e-6 on the summed row, 1.1e-5 on the head and 1.6e-5 on the tail, against the 1.75e-5 the cpu, Metal and CUDA backends already differ from each other by. No greedy token changed anywhere in the corpus.

### Changed

- The scale planes hold float16 rather than float32. `SCALE_BYTES` is 2, `planar_row_bytes` is the quant bytes plus `planes * groups * 2`, and every reader widens on load. `LAYOUT_VERSION` goes to 4, which invalidates existing repack caches.
- The repack cache for Llama 3.1 8B q4_K_M goes 6108 to 5151 MiB, Qwen 2.5 0.5B goes 512 to 468 and SmolLM2 135M goes 144 to 136. q4_0 and q8_0 now cost exactly what the tensor data costs in the file, because a q8_0 block is 32 bytes of quant and a float16 scale and that is what a planar row of it is. `tests/test_cache.mojo` asserts the two are equal rather than within a bound.
- On a 4090 at 512 prompt and 128 decoded, both builds in one sitting, the 8B holds 6082 MiB on the card against 7038 and decodes at 95.5 against 88.6 tokens a second. Prefill costs 1.4 per cent, 360.9 against 367.1, which is the widening load on the prefill path where the row is read once per output column rather than once per token.
- `tests/test_repack.mojo` splits its round trip check in two. The five non-k types are asserted bit for bit, because a float16 widened and narrowed lands on itself. The three k types are asserted within one float16 rounding of the blocks decoded directly. `scripts/kernel_oracle.mojo` does the same split and prints both tolerances in its banner.

## [0.4.8] - 2026-09-04

The five and six bit types are stored at five and six bits. Before this, `quant_form` widened q5_0, q5_1, q5_K and q6_K to a byte a value, so a file that held six bits was read at eight for the whole life of the process. A planar row for those types is now a nibble plane followed by a high plane of one or two bits a value, which is the shape ggml uses inside a block and the shape the four bit types already had here.

It is worth a quarter off decode on Llama 3.1 8B q4_K_M on Metal even though it makes the repack cache only 5.7 per cent smaller, and the two numbers are different for a reason worth knowing. The tensors it reaches in a q4_K_M file are the 33 q6_K ones, which are `ffn_down` and `attn_v`, and `ffn_down` is the widest matrix in a layer. The share of the bytes a token reads that moved from eight bits to six is much larger than the share of the file that did.

Every value the planes assemble is the value the byte held. The offset that makes q5_0 and q6_K signed comes off in the magic number subtraction the decode was already doing, so nothing about the terms the dot product sums changed, and the logit corpus is unchanged on both backends.

### Changed

- q5_0, q5_1, q5_K and q6_K repack to bit planes instead of a byte a value. A row is `5 * cols / 8` bytes at five bits and `3 * cols / 4` at six, both of which are a multiple of four for every row width molla accepts, so the scale planes stay aligned without padding. `LAYOUT_VERSION` goes to 3, which invalidates existing repack caches.
- On an M4 at 32 tokens decoded, three runs a side in one sitting, the repack cache for Llama 3.1 8B q4_K_M goes 6474 to 6108 MiB and decode goes 240, 243, 252 to 177, 159, 182 ms a token, and Qwen 2.5 0.5B goes 663 to 512 MiB and 38, 38, 38 to 29, 29, 29. A locally requantized pure q6_K 8B, which is the case the layout reaches hardest, goes 9572 to 7658 MiB and 393 to 266 ms a token.
- On a 4090 the 8B holds 366 MiB less on the card and 128 MiB less on the host, decode goes 85.8 to 89.1 and prefill goes 377.3 to 366.3. The prefill cost is real and not noise, and [docs/validation/layout.md](docs/validation/layout.md) says what it is: the wide path does two loads a pass where the nibble path does one. The q8_0 control moved 4 per cent between the same two builds in the same sitting, so the two small models are inside the noise on time and outside it on memory.
- The wide matvec and matmul step by the same `MATVEC_STEP` and 2 the nibble paths beside them use, rather than by a whole high byte. A whole high byte is the natural shape on Metal and costs occupancy on CUDA, where the first cut of this took 4090 prefill on Qwen from 4895 to 2937 tokens a second. Computing the high index and shift a pass constant folds back to the hoisted form wherever the step covers a whole byte, so both backends get the loop they want out of one body.
- `tests/test_repack.mojo` gains a device check that runs all eight repackable types through `device_matvec` and compares against `planar_row_dot` over the same host mapping, on whichever backend the suite is running.

## [0.4.7] - 2026-09-04

The decode matvec on Metal takes eight values a thread instead of two. That is the whole change and it is worth 1.17, 1.32 and 1.89 times on the three models in the bench set, in the order of their sizes. It came out of answering #203, which asked for something else.

What #203 asked for is the nibble masks from the Metal q4_K matvec in llama.cpp: read four values out of a `uint16`, let them come out sixteen and two hundred and fifty six and four thousand and ninety six times too large, and correct that in the scale so no shift is ever issued. The masks change two things at once, the extraction and how many values a thread carries, and a control that changes only the second one gets everything the masks get and more. So the shifts were never what the loop was paying for.

At eight values a thread the Metal decode matvec is within ten per cent of a variant that does the loads and no arithmetic at all, which is as far as this kernel goes. What is left on that backend is the number of bytes it reads and the number of times it is launched, and both of those are other issues with numbers on them now.

### Changed

- The decode matvec takes eight values a thread on Metal instead of two, and eight instead of one on the byte wide path. Nothing about the arithmetic changed. On an M4 at 512 prompt and 128 decode, with the version before it measured from the same clone on the same afternoon, decode goes 31.1 to 36.5 on SmolLM2 135M q8_0, 18.1 to 23.9 on Qwen 2.5 0.5B, and 2.7 to 5.1 on Llama 3.1 8B q4_K_M. The gain grows with the model because a wider step only pays on rows long enough to have a loop. CUDA keeps the step widths it had, because there a wider one is neutral on the four bit path and a tenth slower on the byte path, and both backends were rechecked on the hardware rather than assumed.
- [docs/validation/bench.md](docs/validation/bench.md) gains a Metal table for Llama 3.1 8B, which is the model M7 takes its gate on and was missing from that half of the page. All three Metal tables were retaken together and each carries a before row. The CUDA tables are unchanged and the page now says why, with the runs that checked it.

### Fixed

- #203 is answered rather than implemented. It asked for the nibble masks from the Metal q4_K matvec in llama.cpp, which read four values out of a `uint16` and correct the factors of sixteen and two hundred and fifty six in the scale so that no shift is issued. `scripts/matvec_probe.mojo` gains that variant and a control that changes the step width and nothing else, and against the control the masks are worth nothing on Metal and between nothing and a fifth on CUDA depending on the shape. The shifts were never what the loop was paying for.

## [0.4.6] - 2026-09-04

Prefill on Metal was multiplying one number at a time and the GPU has an instruction that multiplies an eight by eight matrix. Using it takes a 514 token prompt through SmolLM2 from 472 tokens a second to 2142, and through Qwen 2.5 0.5B from 169 to 613.

`_mma_apple_8x8` is the simdgroup matrix multiply accumulate every Apple GPU since the M1 has, and it is the instruction llama.cpp's `kernel_mul_mm` is built on. It is reachable from Mojo through `max.gpu.compute.arch.mma_apple` in the max-core release molla already pins. The measurement that made it worth doing is a dense tile of the shape `kernel_mul_mm` uses, timed on an M4: 0.30 TFLOP/s with ordinary multiplies, 3.04 through the instruction, against llama.cpp's own Metal prefill on the same machine working out at 2.55. So the instruction was both the whole gap and enough of it, and the tile is only the thing that feeds it.

The prefill against decode agreement is unchanged at a relative 2.4e-7, because both operands are staged as float. Half precision runs at the same rate on this hardware, 2.95 against 3.04, so it would buy threadgroup traffic and nothing else, while taking that agreement to 4.5e-4 against a gate of 2e-4. llama.cpp stages activations as half here and on this GPU that is not what the rate asks for.

### Added

- A matrix core prefill matmul on Metal. Sixty four output rows by thirty two tokens, four simdgroups two by two, a reduction step of thirty two because that is the group size of every type molla repacks, and the weight dequantized straight into the staged tile in transposed order since the instruction has no transposed form at this size. Measured on an M4 at 512 prompt and 128 decode with all three engines on the same afternoon: SmolLM2 135M q8_0 prefill 472.0 to 2141.7 against llama.cpp's 9341.0, and Qwen 2.5 0.5B q4_K_M 169.4 to 612.6 against 3238.0, so the gap goes from 20.0 to 4.4 and from 19.2 to 5.3. Decode is untouched. Memory goes 344 to 405 MiB and 887 to 915, which is the wider prefill chunk the tile wants.
- `write_epilogue`, one copy of the bias, gate, store or accumulate tail that all three matmul kernels had their own copy of. It is the only part of them that has to agree exactly, since a prefill and a decode of the same prompt are checked against each other.

### Changed

- `PREFILL_CHUNK` is 256 on Metal and stays 64 elsewhere. The tile covers thirty two tokens, so a chunk of sixty four gives the GPU two blocks of the token axis to fill ten cores with, and widening it is 1611 tokens a second against 2142. On CUDA the same widening with no kernel change at all goes the other way and by more, 4990 to 3545 on Qwen 2.5 0.5B and 378 to 181 on Llama 3.1 8B. That is not understood yet and it is filed.
- [docs/validation/bench.md](docs/validation/bench.md) is retaken against 0.4.5 on all three machines at the same 512 and 128, where the laptop and server1 tables used to be 134 and 32 and could not be read against the card. The readings that came out of it: decode on CUDA is a constant 3.6, 2.7 and 1.9 times off llama.cpp as the model grows, prefill was 3.1, 8.6 and 27.9 over the same three, and a ratio that grows with model size is a scaling failure rather than a slow kernel. Memory on the card is at parity except on the 8B. The worst number on the page is the host path, which is 76.6 times off at prefill and 33.9 at decode on the same laptop and the same file, and nothing is currently pointed at it.
- [docs/validation/engines.md](docs/validation/engines.md) records what llama.cpp and Ollama actually do, read at `llama.cpp` commit `f9f09f02cc44` and `ollama` commit `b68365a`, with a file and a line behind every claim. The correction that mattered most is what MAX covers: `layout.tensor_core.TensorCore` is float only, with no integer case and no `dp4a` and no `s8.s8.s32` anywhere in the tree, and `max.gpu.compute.arch.mma_apple` does exist and is what this release landed on. The previous reading had both halves the other way round.

### Known issues

- The same tile on CUDA loses to the kernel it replaces on two models out of three, so it is not in this release. An Apple simdgroup matrix is about ten times its own GPU's scalar rate and an NVIDIA half precision tensor core is two times its own fused multiply add rate, so the same tile buys ten times on one machine and two on the other before anything is spent on feeding it. What the numbers say is missing there is `ld_matrix` for the fragments, a swizzled shared layout and a K loop that stages ahead.

## [0.4.5] - 2026-09-04

A prompt is a matrix and molla was treating it as a run of decodes. It is a matrix now, and time to first token on a 514 token prompt falls by twenty five times on the smallest model of the three and by nearly four on the largest.

Sixty four prompt tokens go through the whole stack in one pass, so the kernel launches for a chunk are the launches for one token and a 514 token prompt is nine passes rather than 514. Every matvec on that path becomes a matmul, the embedding lookup and the norms and rope and attention all take a token count, and the keys and values for a whole chunk are written into the cache by the projection the way they already were in decode. Decode is untouched and takes the same path it always did.

The matmul is 91 per cent of a prefill pass, so that is where the work went. A block is several groups of a warp, each group carrying eight tokens on CUDA or sixteen on Metal in registers against one weight row, and the groups share the row rather than the registers. The tokens a block carries are a whole chunk, which means the weight matrix is read once a chunk rather than once a token, and that is the entire difference between prefill and a run of decodes.

### Added

- Batched prefill. A 514 token prompt and 64 decoded tokens on an RTX 4090, prefill before and after: SmolLM2 135M q8_0 from 358 tokens per second to 9018, Qwen 2.5 0.5B q4_K_M from 398 to 4673, Llama 3.1 8B q4_K_M from 100 to 369. Time to first token on the same runs: 1434 ms to 57, 1291 ms to 110, 5143 ms to 1397. Decode does not move on any of the three and neither does peak memory, which stays at 282 MiB against llama.cpp's 444 on SmolLM2 and about 1000 MiB against 4900 on the 8B. Greedy output is identical to 0.4.4 on all three models on both backends.
- [docs/validation/prefill.md](docs/validation/prefill.md) has the chunk arithmetic, the block geometry and the three things inside that block that were worth about a factor of five between them on CUDA. Two of the three are invisible in the shape of the code: the tail lanes of a short chunk read past the last token rather than clamping onto it, because a clamp costs a register per token held for the whole accumulation, and the reduction loop has to be written `comptime for`, because a plain loop indexes the accumulator array with a value the compiler will not fold and the whole array lands in local memory for the entire kernel. The third is a warp butterfly reduction in place of a tree through shared memory, which is worth 41 per cent on the 8B and nothing on the two models that fit in cache.
- A prefill agreement block in `tests/test_gpu_block.mojo`. The logit corpus does not cover any of this and cannot, because the oracle traces every layer and tracing needs one token's snapshot at a time, so the corpus feeds prompts through the token at a time path and always will. The new checks run a synthetic model both ways in one process: a chunk leaves the same logits and the same cache as the same prompt decoded a token at a time, greedy picks the same token off them, and a prompt split across two chunks reaches the same logits. None of the three runs is a whole number of blocks, because a run that divides evenly is the case that hides the tail.

## [0.4.4] - 2026-09-04

Three of the kernels a layer launches were doing one flop per element, and every one of them can be done by the kernel in front of it for free.

A matvec thread block owns one output row and has that row in a register when its reduction finishes. Anything that reads only that row, and nothing else the same launch is still writing, is work the block can finish itself. That is exactly what the projection bias, the residual add and the gate half of a gated MLP are, so all three are epilogues now rather than launches. The residual add is the best of the three, because folding it in also deletes the scratch vector the projection was landing in and the round trip through device memory that went with it.

A layer goes from fifteen launches to twelve on a model without qkv bias and from eighteen to twelve on one with it. It is exact, so the logit corpus is byte identical to the previous release on both backends and no tolerance moved.

### Changed

- The device matvec takes an epilogue argument and an auxiliary vector, and `device_mlp` and `device_attention` use it in place of the bias add, the residual add and the activation launches. Counted with nsys over five forward passes on an RTX 4090, SmolLM2 135M goes from 453 kernel launches a token to 363 and Qwen 2.5 0.5B goes from 435 to 291. Decode on the same card, eight interleaved repetitions: SmolLM2 135M q8_0 goes from 282.2 tokens per second to 310.8, Qwen 2.5 0.5B q4_K_M from 244.9 to 309.2, and Llama 3.1 8B q4_K_M from 99.8 to 101.4. The 8B barely moves because at 5.6 GiB a token its launches were already hidden behind its weight reads, which is the same split this project has been reporting from the byte side since 0.4.1.
- [docs/validation/performance.md](docs/validation/performance.md) has the launch counts, the throughput, the three things that fit in a matvec epilogue and the four that do not. It also records a variant that was measured and rejected, making the activation a compile time parameter rather than a runtime branch to keep an `exp`'s registers out of kernels that never run it, which came out a wash on two models in opposite directions and is not worth three instantiations of the hottest kernel in the program.

## [0.4.3] - 2026-09-04

The decode matvec spent more of its time turning integers into floats than multiplying them, and there is a way to do that with no conversion instruction in it.

`Float32(n)` for a small integer compiles to a convert. On NVIDIA a convert issues on the same unit as transcendentals, at a quarter of the rate of a multiply, so the four of them that a byte of two four bit weights costs are worth sixteen multiplies before any multiplying has happened, and the kernel only does four. A float32 whose exponent is 23 has a mantissa step of exactly one, which means every representable number between `2^23` and `2^24` is an integer and its bit pattern is `0x4B000000` plus that integer. Nothing carries out of the mantissa for a small value, so the addition is an or. Or the weight in, subtract `2^23` back off, and the conversion is an integer or and a floating point subtract, both of which run at full rate on both vendors. A type whose weights are centred gets its sign extension out of the same or for free, because the bias moves into the constant being subtracted.

It is exact rather than close, so the whole logit corpus is unchanged in every digit on both backends and no tolerance moved.

### Changed

- The device matvec and the device row dequantizer convert a weight to a float by bit pattern rather than by conversion instruction. Decode on Llama 3.1 8B q4_K_M goes from 72.9 tokens per second to 99.7 on an RTX 4090 and from 2.0 to 2.9 on an Apple M4, against llama.cpp's 163.7 on the same card and file. Microseconds a launch on the three shapes an 8B decode spends its matvec time in, Apple M4 before and after then RTX 4090 before and after: 3029 and 1924 then 93 and 42 on the 14336 by 4096 shape, 2509 and 1377 then 93 and 35 on the 4096 by 14336 one, 899 and 553 then 28 and 14 on the square one. The byte wide types are a wash on both cards and SmolLM2 135M q8_0 does not move, 223.8 to 226.5 on the 4090, because a signed byte load already sign extends in the hardware and there the change swaps one instruction for three rather than five for three. It was made in that loop anyway, because it costs nothing and it keeps the two readers of a planar row the same shape.
- `scripts/matvec_probe.mojo` prices the byte wide path as well as the four bit one, which is the only reason the paragraph above can say the two behave differently rather than assume they do not. Its baseline variant is also relabelled: what it called shipped was the divide that 0.4.2 deleted.
- [docs/validation/layout.md](docs/validation/layout.md) has the probe tables, the decode numbers and a section on the fix that is not landing yet. Multiplying weight by activation as integers is faster still, roughly a further 1.3 times, and it needs a 16 bit activation rather than the 8 bit one the research proposed, because at 8 bits the Metal corpus fails twelve of thirteen cases and narrowing the group does not rescue it. It also costs four kernel launches on a layer that has about fourteen, which is invisible on an 8B and takes SmolLM2 135M from 269.5 tokens per second to 212.3. That waits on the elementwise fusion work, which removes three of the four.

## [0.4.2] - 2026-09-04

One line of the decode matvec, and Metal is about twice as fast as it was.

The group index into the scale plane was written `i // group`, with `group` a compile time constant of 32 or 16. NVVM turns that into a shift and the Metal compiler does not, and a signed divide has to correct for a negative numerator whatever the divisor is, so on Metal more than half of the hottest kernel in the program was the compiler being careful about a sign that index has never had. It is a shift now. The change is worth nothing on CUDA and it is worth 0.8 to 2.1 tokens per second on Llama 3.1 8B on an M4.

The useful part is how it was found, which is a new probe rather than a profiler. Every performance number this project has published came off a 4090, and this one cost zero per cent there, so no amount of reading that profile would have turned it up.

### Added

- `scripts/matvec_probe.mojo`, which runs the decode matvec over a synthetic planar tensor of a real shape and prices six arrangements of its inner loop against each other. A question that was an hour of nsys against an 8B model is now a second against a tensor that does not have to be a model, and it runs on whichever backend the machine has, which is the point. Microseconds a launch on a 4096 by 4096 q4_K matvec, Apple M4 then RTX 4090: as shipped 1916 and 29, with the shift 867 and 29, with 32 bit loop indices 740 and 29, a thread to a group 639 and 64, with the scale plane read deleted 521 and 28, and with all the arithmetic deleted 199 and 10.

### Changed

- The device matvecs index the scale plane with a shift rather than a division. Decode on an M4 goes from 0.8 tokens per second to 2.1 on Llama 3.1 8B q4_K_M, 9.9 to 19.0 on Qwen 2.5 0.5B q4_K_M and 24.7 to 31.2 on SmolLM2 135M q8_0, and does not move on a 4090. The bigger the model the bigger the win, because the more of a token is that one kernel. It computes the same index, so it was checked with no tolerance: the whole logit corpus on Metal is identical in every digit to the same corpus from the commit before, and CUDA is green on all thirteen device cases. `group_shift` in `src/molla/nn/repack.mojo` is where the reasoning is, and `tests/test_repack.mojo` fails if a quant type is ever added whose group is not a power of two.
- [docs/validation/layout.md](docs/validation/layout.md) has the probe table, the three model decode table and a correction. The conclusion 0.4.1 drew from the packing measurement, that molla's matvec is limited by work per value rather than by bytes, is a fact about a 4090 and had been written as though it were a fact about both backends. It was not one, and half of the Metal kernel was something else entirely.

### Fixed

- The test suite builds without warnings. Five files had an unused variable or a loop index nothing read, and one check folded five constant comparisons that the compiler could answer without running the test. That one now compares every pair at runtime, which is both what it was trying to say and something the compiler cannot fold away.

## [0.4.1] - 2026-09-04

The first half of M2c, which is the memory half, plus one prediction that turned out to be wrong.

A device run used to carry about three times the model file in host memory: a 1.3 GiB arena nothing asked for, both file mappings held open for the life of the process, and a planar weight layout that spent a byte on every weight whatever the file spent. All three are gone or smaller. The Llama 3.1 8B on a 4090 went from 11066 MiB of peak resident memory to 9769 with the arena, then to a bounded window over the file rather than the whole file, and its repack cache went from 9573 MiB to 6474 with the packing. On the reporting side, every Metal memory number published before this release was understated by roughly the size of the weights, and that is fixed rather than explained away.

The prediction was that packing the weights would nearly double decode on that model. It did not, it cost four per cent, and the measurement that says why is the most useful thing in this release. The kernel was not waiting on memory, it was waiting on instructions, and #186 is the work that follows from it.

### Added

- [docs/validation/layout.md](docs/validation/layout.md), which is the research on what molla's weight layout costs and now also on what changing it was worth. One byte per weight makes a q4_K weight ten bits in molla against four and a half in the file, which is where the 8B's 9573 MiB cache came from. The page carries the prediction it made, the measurement that followed, and the correction, because a spec that quietly drops a prediction it got wrong is worth less than one that keeps it.
- [docs/validation/max.md](docs/validation/max.md), answering the three questions performance.md ended on. Graph capture is reachable from Mojo and worth nothing here, there is no grid wide barrier but the pieces to build one exist, and a Metal launch costs four times a CUDA one. Two of those change the plan: fusing a whole layer down to five kernels still leaves launch cost at 115 per cent of the CUDA budget and 203 per cent of the Metal one, so a persistent kernel stops being the fallback and becomes the thing the rest is arranged around.
- `scripts/mem_probe.mojo`, a standalone probe that reads a device buffer with the matvec's own one block per row access shape and reports what that shape is worth. It exists because the packed layout raised a question about the memory system that nothing in the repository could answer, and on a 4090 it answers it in a second: 945 GB/s a byte a thread at any row length from a kibibyte up, and less than that with wider loads.

### Changed

- The planar quant plane is packed at the type's own bit width. q4_0, q4_1 and q4_K store two values a byte, low nibble first, and the five, six and eight bit types keep their byte. `LAYOUT_VERSION` is 2, so every existing repack cache is rebuilt on next load. The Llama 3.1 8B cache is 6474 MiB against 9573, Qwen 2.5 0.5B is 663 against 688, and SmolLM2 q8_0 is unchanged, which is right for an eight bit type. The stored integers are the same integers, so this was checked with no tolerance at all: the whole logit corpus was run on Metal and on CUDA against a tree with the old layout and the logits were identical in every digit on both.
- Decode on the 8B went from 77.6 tok/s to 74.2 with that change, which is the opposite of what [docs/validation/layout.md](docs/validation/layout.md) predicted and worth stating plainly. nsys puts all of it in the q4_K matvec, which was 8.45 ms a forward pass moving 8.12 GB and is now 8.97 ms moving 4.87 GB. That is 961 GB/s falling to 543 while the values a second stay where they were, so the kernel was never being paid by the byte. Four arrangements of the inner loop, three block widths, four rows to a block and a probe with the activation read deleted entirely all land within noise of each other. What separates molla from llama.cpp on this model is work per value rather than bytes per value, and #186 is that.

### Fixed

- The first `map_to_host` call in a process reserves 1313 MiB and never gives it back or reuses it. molla hit it while binding a model, once per layer for the norm weights and again for the rope tables, and paid for it on the first one. Those and the per token logit read go through `enqueue_copy` now, which is what the weight loader always did. Peak resident memory on a 4090 at 512 prompt tokens and 128 generated went from 1607 MiB to 296 on SmolLM2, 2168 to 862 on Qwen and 11066 to 9769 on the 8B.
- A device load held the model file and the repack cache mapped for the whole run, so a process carried about twice the model file in pages nothing would read again. Both are closed before the first token, and the drain loop gives each matrix's pages back as it copies them, in a 64 MiB window, so peak is the window rather than the model. Giving pages back is an anonymous `PROT_NONE` mapping over the top and not `madvise`, because on macOS `MADV_DONTNEED`, `MADV_FREE` and `msync(MS_INVALIDATE)` all return success and all move the resident set by nothing.
- That window was not a bound. The transfer workers touched pages as fast as they could claim tensors and nothing made them wait for the drain loop, so on a warm page cache they walked an eight gigabyte file in under a second and peak was the whole file again. It only looked fixed because every measurement had been taken on a cold cache, where the disk holds the workers in step by accident. Warm cache peak on the 8B was 9719 MiB, which is what it cost before any of this. The read stage now waits for the window.
- The benchmark harness read `ru_maxrss` on macOS, which does not count the pages behind a Metal buffer, and on Apple silicon those pages are host memory. The 8B on the macbook reported 5347 MiB for a process that had just uploaded 9572 MiB of weights. Switching to `phys_footprint` alone would have been wrong in the other direction, because it excludes clean file backed pages and llama.cpp holds its weights in a mapping of the model file. The harness takes the larger of the two, which hides neither engine's weights. Linux is untouched and stays on `ru_maxrss`.

## [0.4.0] - 2026-09-03

M2b closes. molla runs on a GPU, and now there is a number for how well.

Every kernel a token needs is on the device, the KV cache lives there, the backend is chosen and printed rather than assumed, and the same fourteen logit cases agree with llama.cpp on the host, on an M4 and on a 4090 within one set of tolerances. That was the milestone. The last issue in it was the one that asked how fast any of this is, and the answer decides what happens next.

It is 2.4 to 3.3 times slower than llama.cpp on decode, more than a hundred times slower on prefill, and uses 2.2 to 3.6 times the host memory. M2b said up front that being faster was an M7 gate and that what this milestone owed was knowing the distance. It owes that now, and the distance turned out not to be a tuning problem. molla launches 453 kernels for one token and an empty kernel costs 4.9 microseconds to launch on a 4090, so a token cannot go below 2.2 ms of launch cost whatever the kernels do, which caps molla at about 450 tokens per second on that card against llama.cpp's 782.5. There is no version of the current architecture that reaches the 1.5 times gate, which is why M2c exists and why it is a milestone rather than a patch.

### Added

- molla is measured against llama.cpp and Ollama. `scripts/bench.py` puts the same GGUF file, byte for byte, with the same prompt and the same token count, through all three on one machine and prints one table of prefill tokens per second, decode tokens per second, time to first token and peak resident bytes. Every table records the machine, the backend, the digest of the model file, the build of each engine and the wall clock, so two numbers can be compared only when they are comparable. Results are in [docs/validation/bench.md](docs/validation/bench.md), one table per fleet machine, and `pixi run bench` is the task. Decode on a 4090 is 3.1 times off llama.cpp on SmolLM2, 3.3 times off on Qwen and 2.4 times off on Llama 3.1 8B, against the 1.5 times gate at M7. Prefill is between 109 and 175 times off, because molla runs a prompt through the one token at a time path and llama.cpp prefills it as one matrix multiply.
- The research behind those numbers, in [docs/validation/performance.md](docs/validation/performance.md). Decode is launch bound rather than kernel bound: molla launches 453 kernels per token and an empty kernel costs 4.9 microseconds to launch on a 4090, so a token cannot go below 2.2 ms and molla has a ceiling of about 450 tokens per second on that card whatever the kernels do. The 1.3 GiB of host memory that has nothing to do with the model is the first `map_to_host` call in the process, and the copy path never pays it. The page also says which parts of being two times faster than llama.cpp are reachable and which are above the hardware, because on an 8B llama.cpp is already at 79 per cent of the card's memory bandwidth and two times that does not exist. That work is M2c.
- `molla tokenize <tokenizer.json> "<prompt>" [--ids]` prints how many tokens a prompt is, and the ids when asked. No model file, because encoding does not need one. The benchmark harness builds a prompt to a target length and only knows the length once something has counted it, and doing that with `molla generate` cost a full prefill per attempt, which on an eight gigabyte model is minutes for an answer the tokenizer alone gives in milliseconds.

## [0.3.3] - 2026-09-03

The same fourteen logit cases, on the host, on Metal and on CUDA.

One issue, and what it produced is a number rather than a feature. The corpus that has been checking molla against llama.cpp since #30 now runs through the device kernels too, and all three backends land in the same place: one set of tolerances covers them, the worst case per target is identical to the digits printed, and the largest difference between any two of them on any case is 1.75e-5. Three implementations sharing no reduction order agreeing that closely is the strongest statement this repository can make about the device path being the same program.

It also cost one confusing failure, which is now a clear one. An unquantized model has no planar form of its weights and the device matvecs read nothing else, so an f16 file on a card used to fail from inside the repack with a message about a cache. It is refused up front and by name.

### Added

- The logit corpus runs on a device. `scripts/logit_oracle.mojo` takes `--device` with the same spellings every command takes, loads onto the card, traces the device forward pass and compares it against the same reference files the host run uses. One set of tolerances covers all three backends: the host, an M4 and a 4090 agree with llama.cpp to the digits printed, and the largest difference between any two of them on any case is 1.75e-5 against a tolerance of 2e-1. `pixi run conformance-logits-device` is the task. See [docs/validation/logits.md](docs/validation/logits.md).
- An unquantized model on a device is refused before it is loaded, and the message says why. The device matvecs read the planar form of a quantized weight and an f16 file has no planar form, so `--device=metal` on one is an error, `--device=auto` on one is a host run with the reason printed under the backend line, and the logit corpus reports its f16 case as a stated skip. Before this the failure came out of the repack as a cache that was written and still cannot be used, which said nothing about the model that caused it. `molla load` is exempt, because copying an f16 tensor to a card is a real thing to time and that command launches no kernels.

### Changed

- `molla generate --device` loads through the same code the server and the logit oracle load through. It had its own copy of the context, the repack and the fit check from before `molla.engine.device` existed, which is how the refusal above initially reached two of the three callers and not the third.

## [0.3.2] - 2026-09-03

A token goes in and a row of logits comes out without leaving the card.

Two issues, and together they are what M2b was for. The first is a whole forward pass in device memory rather than twenty one kernels that each have to come home, which on a 4090 is the difference between 5681 ms and 12 ms per token on an 8B. The second is being able to say which backend a run is on and to refuse the one that is not there, because the first is worth nothing to a benchmark that cannot tell which of the two it just measured.

The default changed with it. `molla generate` and `molla serve` with no flag now run on an accelerator when the model fits on one, and `--device=cpu` is the way back.

### Added

- A whole forward pass on the device. `molla.nn.gpu_block` is the twin of `block.mojo` and `model.mojo` together, operation for operation in the same order, and `molla.engine.device` is the twin of the session, holding the KV cache in device memory and the one context the process gets. A token goes in and a row of logits comes out with nothing in between crossing back to the host: the residual stream stays on the card from the embedding lookup to the final norm, and the keys and values are written where they will be read and never copied. On a 4090 an 8B Q4_K_M decodes at 12 ms/token against 5681 ms/token for the scalar host path, and both print the same text. See [docs/validation/device-decode.md](docs/validation/device-decode.md).
- `molla generate --device`, which runs that pass.
- The device matvec takes an offset, so a key projection lands in the layer cache at its slot with no kernel change. It is a pointer moved rather than an argument every kernel would have to apply and a bound it would have to be told, and the same call covers a bias over one run of a query and a per head norm over one head of a key where it already lies in the cache.
- `device_unpack_row`, the embedding lookup on the card. The one read of a weight in a forward pass that is not a matvec, so it was the last thing keeping a token's first operation on the host.
- `--device` is a backend every command agrees on. It takes `auto`, `cpu`, an api like `metal` or `cuda`, or an api and an index after a colon, and `molla generate`, `molla serve` and `molla load` all read the same values. Naming a backend that is absent, is at no such index, or is too small for the model is an error before the model file is opened, because a benchmark run against the wrong backend is worse than one that refused to start. Whether a model fits is asked of `plan_load` rather than estimated, which is the same function that will place the tensors a minute later. See [docs/validation/backend.md](docs/validation/backend.md).
- `molla serve --device`, and a backend line in `/molla/version` saying which one the running server is on. The server holds a host session or a device session and never both, so a host server allocates no device cache and a device server allocates no host cache.
- `molla load` prints where each class of weight ended up, in tensors and megabytes, split by embedding, attention, feed forward, norms and the output head. A load that put the attention matrices on the card and left the feed forward ones in the mapping is the interesting case and it was previously visible in nothing at all.
- The device forward pass is compared against the host one by the test suite on every machine that has a device. `tests/test_gpu_block.mojo` builds a two layer model twice out of the same planar bytes, once in host memory and once in a real pool, and checks the logits, the greedy pick, and the residual stream layer by layer. A kernel that is right about arithmetic and wrong about which buffer it was handed passes every test that looks at one kernel, which is why this one looks at all of them in order.

### Changed

- `molla generate` and `molla serve` run on an accelerator by default when there is one the model fits on. Both were host runs before this unless told otherwise, so the same command line is now faster on a machine with a card and unchanged on one without. `--device=cpu` is the way back, and it is honoured whatever is attached.
- `load` takes a device context the caller already owns. A CUDA process gets one and hangs on the first allocation against a second, so an engine that is going to run kernels has to hand its own over rather than let a load make one behind it.
- The two cache questions that are policy rather than storage, which slot a position goes in and what happens when the context fills, are free functions in `molla.engine.cache` and both caches call them. The day a slot stops being a position, one function changes and the host and device caches follow together.

## [0.3.1] - 2026-09-03

A unified machine can put weights where a device kernel can read them.

One change, and it is the last thing standing between M2b and a forward pass on the M4. A Metal kernel cannot read a mapped file where it lies, which #153 admitted and this release acts on, so a unified device now gets a pool and slots exactly as a card does. The copy is the same asynchronous one the discrete path already used, which was measured rather than assumed, and what it costs on the M4 is written down next to what it replaces.

### Added

- A unified machine can put weights where a device kernel can read them. `device_budget` gives a unified device a budget and `plan_load` gives it slots, both of which used to be skipped on the belief that an accelerator sharing the memory could read the mapping where it lies. It cannot, so a weight a Metal kernel is going to read needs a pool as much as one a CUDA kernel is going to read. The copy is the same asynchronous `enqueue_copy` the discrete path already used, which was measured on an M4 rather than assumed, so there is no second copy path. The reserve is a quarter of the machine rather than a tenth, because on a unified box the memory held back is memory the operating system is also drawing from.
- `molla load --host`, which leaves every weight in the mapping and asks for no pool. A machine with an accelerator has two loads worth timing and neither is the fallback of the other. On the M4 the same warm cache 8B is 4157 ms with `--host` and 5289 ms without, so a device address costs 27 per cent there, and almost none of that is the copy itself, which hides behind the read exactly as it does on a 4090. See [docs/validation/load.md](docs/validation/load.md).
- The device pool is checked by the test suite on the machines that have a device. `tests/test_gpu.mojo` fills a real `DevicePool` at the slots a plan would have chosen and reads both tensors back with a kernel, so the slot arithmetic and the asynchronous copy meet the way they do in a load. Two tensors rather than one, because a pool holding a single tensor cannot tell a correct base address from a correct offset.

### Fixed

- `DevicePool` can be handed a device context instead of always making one. A CUDA process gets one context and hangs on the first allocation against a second, so a pool that insists on its own is a hang waiting for the first caller that already has one.

## [0.3.0] - 2026-09-03

M2 closes and the GPU stops being a plan.

This is the release that closes M2, which is the milestone where a real model produces coherent text through an OpenAI compatible endpoint. The device half of M2's exit criteria was never inside its own scope and moved to M2b, which is where the rest of this release comes from: a quantized matvec on the GPU, then the whole of the rest of a transformer block, both compiled for Metal and CUDA out of one source and both checked against the host on an M4 and a 4090.

### Added

- A quantized matvec that runs on the GPU. `molla.nn.gpu` holds one kernel, compiled for Metal and for CUDA out of one source, reading the planar layout `molla.nn.repack` writes. One thread block per output row, `TILE` threads walking the row with a stride of `TILE`, then a tree reduction in shared memory. The group size and whether the type carries a minimum plane are compile time parameters, so there are three instantiations of one function and no target specific branch anywhere inside it.
- `pixi run conformance-kernels`, which multiplies every fixture in the quantization corpus three ways, ggml on the host, planar on the host and planar on the device, so a disagreement can be attributed to the repack or to the kernel rather than arriving with two suspects. Ten formats agree on both an M4 and a 4090 within 3.16e-07 of peak against a gate of 1e-5, and every figure it prints is identical on the two machines. See [docs/validation/kernels.md](docs/validation/kernels.md).
- A device matvec refuses a weight that is not in a device pool. A device kernel handed a host address does not fault, it reads zeros, so a model bound that way runs at full speed and answers with noise. The check is separate from the launch so it runs on the machines in the fleet with no accelerator, which is most of them.
- The rest of a transformer block runs on the GPU. `molla.nn.gpu_ops` holds the norm, rope, attention, the two gated feed forward shapes, the plain activations, the residual add, the scale and the argmax, all compiled for Metal and for CUDA out of one source with no target specific branch in any of them. They went across as a set rather than one at a time because at decode shapes a round trip between two kernels costs more than every small operation in a block put together. Attention scores, softmaxes and takes its weighted sum in one launch, so a head never writes its scores anywhere the host can see them.
- `pixi run conformance-block`, which compares every device operation against the host one it mirrors and prints the distance whether or not it passed. It needs no fixtures at all, so a machine with a GPU can run it straight after a clone. Twenty five cases agree on both an M4 and a 4090, worst case 1.91e-06 of peak against a gate of 1e-5. See [docs/validation/kernels.md](docs/validation/kernels.md).
- `molla.nn.rope.step_table`, which works out the frequency step of a rope spec on the host in float64 and hands the device the answer. It costs `dim / 2` floats per spec, of which a model has one or two, and it is for accuracy rather than speed: forming the step from a float32 exponential is about 1e-7 out, and the angle multiplies that by the position, so the error grew from 4e-06 of peak at position 137 to 1.5e-04 at position 4096.

### Fixed

- A rope angle no longer goes through the target's own argument reduction. CUDA's float32 `cos` and `sin` handle a large argument badly, and the angle at the fastest rotating pair is the position itself, so position 4096 is 652 whole turns and the 4090 was 4e-04 of peak away from the host where an M4 was within 1e-07 at the same point. The kernel now reduces the angle itself across three float32 constants before the call, which takes the 4090 to 2.6e-07 and leaves Metal where it was.
- A logit softcap no longer goes through the target's own `tanh`. Gemma 2 multiplies its `tanh` by 50, so the gap between CUDA's float32 one and the float64 the host takes arrived at the softmax fifty times larger, at 2.04e-05 of peak. Computing the same function through `exp`, arranged so the exponential lands in `(0, 1]` rather than overflowing on a large score, takes it to 1.9e-06.
- The test suite makes one device context for the process instead of one per module. A CUDA process gets one, and the second is constructed without complaint and then hangs on the first buffer allocated against it, with the GPU idle and every thread asleep on a futex, which reads like a kernel that will not finish and is not one. Metal does not mind, so this only ever showed on the 4090. See [docs/validation/kernels.md](docs/validation/kernels.md).
- `Tensor.device_address` refuses a unified weight. It used to let one through on the grounds that a machine with one pool of memory has one address, and that is false: a device buffer's own pointer segfaults when read from the host, `map_to_host` returns a different address for the same bytes with no copy, and a kernel handed the host one reads zeros without reporting anything. The memory is shared and the address space is not. Nothing measured changes, a unified load still allocates nothing and copies nothing, and what a unified placement promises is now about what the pool cost rather than about what a kernel can be given. Making it mean something to a kernel again is [#152](https://github.com/tamnd/molla/issues/152). See [docs/validation/load.md](docs/validation/load.md).

## [0.2.11] - 2026-09-03

molla checks itself against llama.cpp a layer at a time, and a weight knows which memory it is in.

Two things. The first is a conformance corpus: fourteen cases over three models at nine quantizations, comparing the residual stream after every layer and the whole final distribution against llama.cpp reading the same file, so a disagreement comes back as a layer number rather than as output that reads very slightly wrong. The second is the type system learning that a card has its own memory, which is the first piece of M2b and what every device kernel after it binds against.

### Added

- A logit conformance corpus. Fourteen cases over SmolLM2 135M, Qwen 2.5 0.5B and Llama 3.1 8B at nine quantizations, comparing the residual stream after every layer and the whole final distribution against llama.cpp reading the same file. `scripts/gen-logits.py` writes the references and needs llama.cpp on PATH, `scripts/logit_oracle.mojo` checks molla against them and needs nothing, and `pixi run conformance-logits` runs it. See [docs/validation/logits.md](docs/validation/logits.md).
- `Scratch` can record the residual stream. `tracing` is off by default and costs a bool test per layer, and with it on a model with n layers leaves n plus two snapshots per token: the embedding, one after each layer, and the final norm. A disagreement with llama.cpp comes back as a layer number rather than as a mismatch somewhere in a hundred million multiplies.
- A weight knows which memory it is in. A `Tensor` carries one of host, unified or device, `Tensor.base` refuses a weight in a device pool and names its shape rather than handing a host kernel an address it cannot follow, and `Tensor.device_address` refuses a host one. A unified weight passes both, which is what unified means. See [docs/validation/load.md](docs/validation/load.md).
- `Residency`, which is what a load hands the binder: which memory each tensor ended up in, by position in the file's directory, and where on the device if it moved. An empty one binds everything to the host, which is what every caller that loads with a device budget of zero gets.

### Changed

- `plan_load` can be told about the repack cache, and then it plans against the copy of each weight that `bind` is going to read. A cached tensor is sourced from the cache in the planar layout and one with no planar form is sourced from the file, so a warm load stops faulting in a copy of the model that nothing will touch, and a device pool holds the layout the device kernels want. `molla load`, `molla generate` and `molla serve` all pass the cache through.
- A load checks that the bytes it copied to the device add up to the bytes the plan placed there, and refuses to finish when they do not. A card missing one weight out of 292 produces text that is almost right.

## [0.2.10] - 2026-09-02

Weights get rewritten once and kept.

The ggml block layout is a compression format and a good one, and it is the wrong thing to read in an inner loop. molla now unpacks it once at load into a planar layout the kernels want, and writes the result beside the model so the next load maps it instead of doing the work again. A cached load of a 135M finishes in 4 ms against 192 ms for the load that writes the cache.

### Added

- `molla.nn.repack`, a planar weight layout and a transform into it from all eight quantized ggml types. A row becomes its int8 quants followed by float32 planes of scale, and a value is `dscale[g] * q[i] + mscale[g]`. Centring folds into the byte and a block minimum folds into a sign, so q4_0, q5_0, q6_k and q8_0 have no offset plane at all. A repacked row decodes to bit for bit the same float32 values as the blocks it came from, which the tests assert with no tolerance.
- `molla.model.repack`, a cache file beside the model holding the repacked weights. It is keyed on a digest of the model header, its directory and its length, plus the layout version and the target, and any of those changing is a miss rather than a wrong answer. The file is written under a temporary name and renamed, so a load that dies halfway leaves nothing to find, and deleting the cache costs one slow load and nothing else. See [docs/validation/repack.md](docs/validation/repack.md).
- The repack runs on the transfer pool the load already has, between the page touching loop and the ready queue push, so it reads the bytes a worker has just faulted in rather than reading the file a second time. A repack that fails records its errno, the load carries on, and the half written cache is thrown away.
- A hit and a miss are both reported. `molla load` says which it got and why, and `molla generate` and `molla serve` bind to the cache when there is one. A repack that silently reruns on every load is the thing the cache exists to prevent, and the only way to notice it is to be told.

## [0.2.9] - 2026-09-02

The model is behind a socket.

`molla serve` answers OpenAI chat and completion requests against a GGUF file on disk, streaming and not, and the OpenAI Python SDK talks to it with nothing configured but `base_url`. One worker and one sequence at a time, so a second request in flight gets a 503 rather than being queued behind a scheduler that does not exist yet.

### Added

- `molla.api.openai`, the OpenAI wire format: request parsing, the response and streaming chunk builders, and the error envelope. Everything under `molla.api` is somebody else's API written down in Mojo and holds no opinions of its own, which is decision D2. Fields this build does not implement are refused with a 400 that names them rather than accepted and ignored, because `tools` accepted and ignored looks exactly like the model deciding not to call one.
- `molla.engine.runner`, one loaded model plus the state of the one request using it. It owns the mapping, the weights, the tokenizer, the compiled chat template, the session and the sampler, and the protocol reaches it by address. Stop strings are held rather than searched for afterwards, since a stop string can straddle two tokens and half of one must not go out to a streaming client.
- `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/v1/models/{id}`, streaming and not, on the reactor that has been there since M0. The OpenAI Python SDK streams a conversation against it with nothing configured but `base_url`, and raises `NotFoundError` and `BadRequestError` from the error envelope rather than a generic `APIError`. See [docs/validation/openai.md](docs/validation/openai.md).
- `molla serve <model.gguf> <tokenizer.json>`, with `--host`, `--port` and `--ctx`. One worker and one sequence at a time: a second request arriving while one is running gets a 503 saying so, rather than being queued behind a scheduler that does not exist yet. A streaming request goes back to the event loop between tokens, so the admin routes stay answerable through one.
- `Connection.yield_now`, which a protocol calls when it has just done something expensive and wants the rest of the reactor looked at before it is asked for more. The reactor gives a connection eight rounds of read, produce and write per pass, which is right when a round is a memcpy and wrong when it is a forward pass through a language model. Without it a health check behind a streaming completion waited eight tokens, which was two seconds on a 135M model and would be a minute and a half on an 8B. It is one token now, and total stream time is unchanged.

## [0.2.8] - 2026-09-02

A model file goes in one end and English comes out the other.

This is the first release where molla runs a real model. `molla generate` reads a GGUF, prefills a prompt, decodes it and prints tokens as they arrive, with the sampling settings anyone would expect on the command line. There is no server yet and no comparison against llama.cpp, both of which are the rest of M2.

### Added

- `molla.engine.cache`, the keys and values one sequence has accumulated. Contiguous, one list per layer, float32, allocated once when the session is made. A sequence that would run past the end of the context is refused rather than wrapping onto slot zero, because a cache that wraps gives a model that is still fluent and has forgotten its instructions, and nobody reads that as an error.
- `molla.engine.session`, prefill and decode as one loop rather than two. The check that matters is that prefilling a prompt and then decoding leaves the cache byte for byte identical to feeding the same tokens one at a time, which is the statement that both routes agree about position. See [docs/validation/decode.md](docs/validation/decode.md).
- `molla.engine.bind`, which points the network at a file's bytes. By name, against the architecture table, with every shape checked before a token is computed rather than on the first forward pass.
- Qwen 2's bias on each of the three attention projections, which Qwen 3 dropped and Llama never had. Without it a qwen2 file loaded, passed every shape check and ran at full speed writing noise, because a missing bias is a constant vector absent from every head of every layer rather than a crash. A file and an architecture table that disagree about whether the biases are there is now refused in both directions. Qwen 2.5 0.5B generates English.
- q4_1, q5_0 and q5_1, the three block formats next to the ones molla already read. A Qwen 2.5 0.5B download named q4_k_m is q5_0 for every weight in it, and the type numbers in the directory are what molla believes rather than the name the file was published under. Decoded and fused, both checked against the `gguf` package, exactly. See [docs/validation/quant.md](docs/validation/quant.md).
- `molla.engine.sample`, the sampling pipeline: a grammar hook that M4 fills in, logit bias, repetition and frequency and presence penalties over a per sequence window, temperature, top-k, top-p, min-p and typical, in llama.cpp's order because that is the order every preset in circulation was tuned against. The randomness is counter based rather than a stream, so the same seed and the same prompt give the same tokens whatever else the server is doing, and greedy is the exact argmax rather than a temperature approaching zero. `molla generate` takes `--temp`, `--top-k`, `--top-p`, `--min-p`, `--typical`, `--repeat-penalty`, `--frequency-penalty`, `--presence-penalty`, `--repeat-last-n` and `--seed`, and no flags still means greedy. See [docs/validation/sampling.md](docs/validation/sampling.md).
- `molla generate <model.gguf> <tokenizer.json> "<prompt>" [n] [ctx]`, which is molla generating text for the first time. SmolLM2 135M at q8_0 and Llama 3.1 8B Instruct at q4_K_M both produce coherent English, the second of those through the precomputed rope frequency factors that Llama 3.1 ships in the file. It is scalar and single threaded, about 90 ms a token on the 135M and 5.5 seconds a token on the 8B, which is what issue #120 exists to change.

## [0.2.7] - 2026-09-02

The arithmetic a transformer is made of, checked against something outside molla.

Nothing in here runs a model. It decodes the weights, multiplies them, puts position into a query, scores it against the keys, and stacks that into layers, and every piece of it is checked against llama.cpp's numbers or against a second implementation. Loading a real file and getting tokens out is the next milestone step.

### Added

- `molla.nn.quant`, which decodes ggml blocks back to float32. f32, f16, bf16, q4_0, q8_0, q4_k, q5_k and q6_k, which covers everything a Llama 3 or Qwen 3 q4_K_M file contains. This is the slow obvious version and it stays that way: the fused kernels get checked against it and it gets checked against the reference implementation, so a fast path that drifts fails against something rather than making a model quietly worse. See [docs/validation/quant.md](docs/validation/quant.md).
- A quantization conformance corpus. `scripts/gen-quant.py` writes fixtures of random block bytes and the values the `gguf` package decodes them to, `scripts/quant_oracle.mojo` compares every one, and CI runs about half a million values per commit. The match is exact rather than within a tolerance, because both sides decode the same bytes in the same order and a tolerance would let a wrong nibble order through whenever the two nibbles happened to be close.
- `molla.nn.tensor`, a weight view that holds an address and a shape and owns nothing, and a float32 buffer for activations. Shape is ggml's, so `dims[0]` is the fast axis and a weight printed as `[4096, 14336]` is 14336 rows of 4096.
- `molla.nn.kernel`, the host arithmetic a transformer block is made of: a matvec that reads packed weights without dequantizing them first, rms_norm, softmax, silu and gelu and swiglu, the residual add, and argmax. Every fused path is checked against dequantizing the same bytes and taking a plain dot product. It is scalar and it is not fast yet, which is what issue #120 is for. See [docs/validation/kernel.md](docs/validation/kernel.md).
- `molla.nn.rope`, rotary position embedding with the four scaling schemes that turn up in the wild: none, linear, NTK aware, and YaRN, plus the per pair frequency factors that Llama 3.1 ships precomputed in the file. Both pairings are supported, adjacent for converted Llama and split half for Qwen, which is a property of the file rather than of the model and is the kind of mistake that produces fluent output that wanders. The expected values come from `scripts/rope_ref.py`, which is ggml's rope loop transcribed to Python. See [docs/validation/rope.md](docs/validation/rope.md).
- `molla.nn.attention`, which scores one query against a run of keys and mixes the values. Grouped query, multi head and multi query are one loop rather than three paths. Sliding windows, sink tokens, logit softcapping and a scale that is not one over the root of the head dimension are all fields on the spec, because each of them is one architecture's idea and none of them deserves its own code path. It holds no cache, which is issue #27.
- `molla.nn.block`, one transformer layer with the knobs every architecture turns: gated or plain mlps, silu or gelu, per head query and key norms, norms on a sublayer's output as well as its input, and the attention spec from above. Nothing in it allocates, because a decode step runs it once per layer per token. It is checked against a second implementation written with naive loops over plain lists, since a layer that calls the right kernels in the wrong order produces numbers in the right range every time. See [docs/validation/blocks.md](docs/validation/blocks.md).
- `molla.nn.arch`, a row per architecture saying what that family always does, and `block_spec`, which turns a row plus a file's `Geometry` into a layer. Head counts, widths, the rope base and the epsilon come off the file and are never guessed. Llama, Qwen 2 and Qwen 3 are marked supported; Gemma and Phi 3 are described without being claimed, because nobody has run one yet. See [docs/adding-an-architecture.md](docs/adding-an-architecture.md).
- `molla.nn.model`, the part above a layer: an embedding lookup, the stack, a final norm and an output head that is tied to the embedding or not according to the file. Gemma's embedding scale and output logit cap live here. It produces logits and stops, because turning logits into a token is issue #28 and holding a cache is #27.

## [0.2.6] - 2026-09-02

molla can now say what a machine has and put a model's weights on it.

### Added

- `max-core`, pinned to 26.5.0, as a required dependency. The `mojo` package ships the standard library and nothing else, and every device call lives in `max-core`, so there is no route to a GPU without it. This was accepted in [docs/adr/0002-accept-max-core.md](docs/adr/0002-accept-max-core.md) but had never been added to the manifest. It is a runtime dependency and not only a build time one, and it is proprietary. See [docs/validation/toolchain.md](docs/validation/toolchain.md).
- `molla.sys.device`, which reports what a machine can put a tensor on: the api, the name, the index, and free and total memory, with a CPU entry that is always first and is a real placement rather than a fallback. The unified flag says whether a mapped file is already visible to the device or has to cross a bus, and it is read off the reported api rather than guessed from the build target.
- `build_targets_gpu` and `build_target_arch`, because `max-core` resolves the device architecture at compile time. A build made on a machine with no GPU has no device code in it and reports no accelerators even on a machine that has one.
- `molla.model.load`, which gets weights from a mapped file to wherever the kernels will read them. A pool of transfer threads walks the mapping while the thread that owns the device drains a queue of finished tensors and enqueues their copies, so the card starts moving the first tensor while the pool is still reading the second. Every tensor is placed once, before any byte moves, as host, unified or device. Device tensors get an aligned slot in one pool buffer rather than an allocation each, and when the model does not fit the planner gives the card everything read once per token first and leaves the embedding behind. An 8B q4_K_M loads in 3746 ms on the M4 and 499 ms on the 4090, with a read and a copy that overlap almost completely. See [docs/validation/load.md](docs/validation/load.md).
- `molla load <path> [workers]`, which runs a load and reports each stage as it happens, because a load that takes half a minute and says nothing reads as a hang.
- `molla devices`, which lists what this machine can put a tensor on.
- `page_size` and `will_need` in `molla.sys.mmap`, for readahead on a range of the mapping. A page is 16384 bytes on Apple silicon and 4096 on x86, and `madvise` rejects an address that is not aligned to it, so the number is asked for rather than assumed.

## [0.2.5] - 2026-09-02

The tokenizer corpus has nothing left in it that molla refuses.

60 of the 338 files carried a `Precompiled` normalizer, which is a SentencePiece charsmap, and molla refused every one of them rather than loading it with the normalizer missing and producing ids that are quietly wrong. That is 18 per cent of the popular tokenizers on the hub and it is all of T5, mT5, XLM-R, ALBERT, DeBERTa-v3 and NLLB. They load now and the full tier reports no refusals and no mismatches.

### Added

- The `Precompiled` normalizer, which is the SentencePiece charsmap. It was 60 of the 338 files in the tokenizer conformance corpus, 18 per cent of them, and all of the T5, mT5, XLM-R, ALBERT, DeBERTa-v3 and NLLB families. Those 60 load now and the full corpus reports no refusals and no mismatches. See [docs/validation/tokenizer.md](docs/validation/tokenizer.md).
- `molla.text.graphemes`, UAX #29 extended grapheme clusters, with all of the rules including the emoji rule GB11 and the Indic conjunct rule GB9c. The charsmap needs it because SentencePiece looks text up one cluster at a time. The grapheme break property is a fifth generated table in `src/molla/text/tables.mojo`.
- `strip_combining` in `molla.text.normalize`, which drops spacing and enclosing marks as well as non spacing ones.

### Fixed

- A `StripAccents` normalizer written on its own dropped only non spacing marks. Hugging Face drops all three mark categories there and drops only the non spacing ones for the `strip_accents` flag inside a `BertNormalizer`, and molla was using one function for both. It lost the enclosing keycap off a digit and the vowel sign off a Devanagari syllable, and it was two files in the corpus.

## [0.2.4] - 2026-09-01

The template engine now answers to an oracle on every commit.

494 real chat templates from the hub, 20 conversation shapes each, compared for exact string equality against the function `transformers.apply_chat_template` calls to build its string. 9500 renders compared and 9500 identical. It found that `tojson` was ignoring every argument except `indent`, which was putting a space after every comma in the tool definitions of four models that had asked for the compact spelling.

Not one template in 494 uses a construct the engine refuses to compile, so the refusal list this was meant to produce is empty. That is a fact about what model authors write rather than a claim about the engine, and the machinery that records refusals runs on every commit so the first one gets noticed.

### Added

- The chat template conformance corpus. 494 real chat templates from the hub, pinned by commit hash, rendered against 20 conversation shapes and compared for exact string equality against `transformers.utils.chat_template_utils.render_jinja_template`, which is the function `apply_chat_template` calls. 9500 renders compared and 9500 identical, of which 750 are cases both sides refuse. It runs on every commit as the `Template conformance` job and a mismatch blocks the merge. See [docs/validation/jinja.md](docs/validation/jinja.md).
- `scripts/templates.tsv`, `scripts/fetch-templates.py`, `scripts/check-template.py` and `scripts/template_oracle.mojo`, laid out the same way the tokenizer corpus is. The manifest carries the sha256 of the reference answer per repository, so the everyday check needs no Python at all, and the Python half runs when the reference version moves.
- `strftime_now` reads a clock that can be pinned, so a template that stamps today's date into the system prompt has one answer rather than one a day. `Template.render` and `Template.render_object` take the second to read, and zero is the real clock.
- `{% generation %}` and `{% endgeneration %}` render their body. They are a `transformers` extension that marks the span a model was supposed to have produced, so a training script can build a loss mask, and they contribute nothing to the string. Seven templates in the corpus use them.

### Fixed

- `tojson` ignored every argument except `indent`. Four templates in the corpus write `tojson(separators=(',', ':'))` to get the compact spelling and were getting the spaced one, which put a space after every comma in their tool definitions. The filter now takes the signature transformers gives it, which is `ensure_ascii`, `indent`, `separators` and `sort_keys` in that order, so the first positional argument is `ensure_ascii` and not `indent`.
- `{'a': 1}.items()` printed its pairs as lists rather than as tuples, so a template writing a pair straight out got square brackets where Python writes round ones.
- `2 ** 3 ** 2` was 64 rather than 512. Exponentiation is the one operator in the language that associates to the right.

## [0.2.3] - 2026-09-01

A chat template renders to the same bytes Python produces, and anything it cannot render refuses to load.

This is the Jinja2 subset. It is bounded on purpose: seven constructs are excluded and each of them raises when the template is compiled, which is when a model loads, rather than at render time on somebody's request. A template that would misrender does not get to serve traffic. Four execution limits are on by default, because a chat template is code out of a repository anybody can publish.

The checking is the part that matters. 38 real chat templates from the hub, nine conversation shapes each, compared for exact string equality against Python `jinja2` in the environment `transformers.apply_chat_template` builds. 342 renders, 342 identical. It found four defects in code that had already passed its unit tests, three of which produced plausible looking output rather than an error, and the worst of them made every binary operator past a certain depth in a template silently evaluate to its left operand.

A 20 turn conversation renders in 59.9 microseconds on the Llama 3.1 template, against the 200 the milestone asks for, with the cost of parsing the request JSON inside that number rather than beside it.

### Added

- `molla.jinja`, a bounded Jinja2 subset for chat templates. Ten modules: a lexer with `trim_blocks` and `lstrip_blocks` and the explicit whitespace markers, a parser onto a flat node list, and an evaluator with 70 filters, 30 tests and the five globals `range`, `dict`, `namespace`, `strftime_now` and `raise_exception`. Statements are `if`, `for` with the loop object and `break` and `continue`, `set` in both forms, `macro`, `call` and `filter` blocks. Checked against Python `jinja2` in the environment `transformers.apply_chat_template` builds, over 38 real chat templates and nine conversation shapes each: 342 renders, 342 identical. See [docs/validation/jinja.md](docs/validation/jinja.md).
- Seven constructs are excluded on purpose and each is a named error at compile time, which is model load time rather than request time: `include`, `extends`, `import`, `from`, `do`, `autoescape` and the async forms. There is no template loader, because a chat template is one self contained string. A refusal carries the construct, the line, the column and a snippet with a caret under it.
- Four execution limits, because a template is untrusted input from a repository anybody can publish: a step budget, an output cap, a recursion depth and a wall clock deadline. The clock is read every 1024 steps rather than every step, which kept it out of the profile.
- `Template` and `Cache` in `molla.jinja.template`. Compiling and rendering are separate because a template is compiled when a model loads and rendered on every request, and the cache keys compiled trees by the SHA-256 of the source rather than by the model, since most forks of a model ship the same template bytes. Values go in as JSON, which is the shape a request body and a tokenizer config already have.
- `molla template <template> <vars> [rounds]`, which renders a template and times it. A 20 turn conversation is 59.9 us on the Llama 3.1 template against the 200 us the milestone asks for, with Qwen3 at 79.8, Mistral Small at 85.4 and Granite at 43.4, all on the M4 and all including the cost of parsing the variables JSON on every round.

## [0.2.2] - 2026-09-01

Text goes in and ids come out, and there is an oracle saying they are the right ids.

This is the tokenizer, and underneath it the whole character layer Mojo does not ship: UTF-8 that rejects what it should reject, Unicode categories and combining classes and decompositions and case mappings generated from the current database, the four normal forms, and a backtracking regular expression engine. On top of that the five stage pipeline a `tokenizer.json` describes, with BPE, WordPiece, Unigram and word level models, twelve normalizers, ten pre-tokenizers, eight decoders and four post processors.

None of that is worth anything without an independent implementation to check it against, because a tokenizer that is wrong produces output that still looks sensible. So most of the work here is the checking. 4560 differential cases across four real models, eight million bytes through each of them, and then the conformance corpus: 338 real `tokenizer.json` files from the hub and 355 pieces of text, every id and every decode round trip compared against Hugging Face `tokenizers` 0.23.1. It runs on every commit and a mismatch blocks the merge.

The corpus earned its keep immediately. It found six defects in code that had already passed 1430 unit checks and four models, and the beginning of text property found a seventh. Two of them were GPT-2 refusing to load at all, one was a Chinese word being cut in half at the ideographic zero, and one was Whisper getting two beginning of text tokens where it should get one. All seven are fixed and all seven have a fixture in the suite now.

It is also fast, between five and eight times the reference on six of the eight throughput rows, and seven of the eight clear the 20 MB/s the milestone asked for. The eighth is gemma on unwrapped documentation, which is a property of that file rather than of the code, and the reference is slower on it too.

What is still refused is a `Precompiled` normalizer, the SentencePiece charsmap, which is 60 of the 338 corpus files and covers the T5, XLM-R and NLLB families. The corpus asserts a clean refusal rather than a quiet wrong answer, and the reference digests are already recorded for the day it lands.

### Added

- `molla.text`, everything a tokenizer needs to know about characters before it can start: UTF-8 encoding and decoding that rejects overlong forms and surrogates and reports an incomplete sequence as one, Unicode categories, combining classes, decompositions and full lowercase mappings generated from the database by `scripts/gen-unicode.py`, the four normal forms, and a backtracking regular expression engine with Unicode categories, lookahead and possessive quantifiers. Checked against Python over 294552 normalization cases and 6732 regular expression cases, all identical. See [docs/validation/text.md](docs/validation/text.md). The regex engine now works out which characters a match starting at each instruction could begin with, which is what makes the seven way alternation at the front of every GPT-2 style pattern cheap, and normalization skips runs of characters below the floor where every form is the identity.
- `scripts/check-text.py` and `scripts/text_oracle.mojo`, the differential run behind that, which also reports the pre-tokenizer split throughput against a file it generates. The GPT-2 pattern over 4 MB of mixed text is 174ms on the M4, which is 23 MB/s.
- `molla.tokenizer`, a `tokenizer.json` reader and the five stage pipeline behind it: added tokens, normalizer, pre-tokenizer, model and post processor, with BPE, WordPiece and Unigram, and the decoders read back the other way. Checked against Hugging Face `tokenizers` 0.23.1 over 4560 differential cases on four real models, one known mismatch caused by the reference reading a Unicode 9 category table, and eight million bytes through each of the four producing exactly the same token counts. See [docs/validation/tokenizer.md](docs/validation/tokenizer.md).
- `Tokenizer.encode_rendered`, for text a chat template produced. A template writes the beginning of text token into the text and the post processor then writes a second one, which is not what the model was trained on and does not fail loudly. `encode` still reproduces that, because its job is to match the reference. This is the entry point that does not.
- `DecodeStream`, which decodes one id at a time and never hands back half a character, so a token that carries one third of a character produces nothing until the rest of it arrives.
- The tokenizer conformance corpus: 338 real `tokenizer.json` files from the hub and 355 pieces of text, every id and every decode round trip checked against Hugging Face `tokenizers` 0.23.1. `scripts/tokenizers.tsv` is the manifest, `scripts/fetch-tokenizers.py` downloads the files and verifies their digests, `scripts/check-tokenizer.py` is the reference half and `scripts/tokenizer_oracle.mojo` is ours. A quick tier of 59 files runs in CI as the `Tokenizer conformance` job, which the required check waits for, so a mismatch blocks the merge. 272 files identical, 6 identical apart from a case where the reference reads a Unicode 9 table, 60 refused as the manifest says they should be, zero mismatched. See [docs/validation/tokenizer.md](docs/validation/tokenizer.md).
- The beginning of text reconciliation rule is now asserted as a property over the whole corpus, since the reference has no such rule to diff against. 210 of the 278 loadable files have an opening special and all 210 hold.

### Fixed

- The model loader read `vocab` at the point it reached it, which needs the type, because the type says whether a vocabulary is an object or an array. 13 corpus files write `unk_token` or `dropout` ahead of the type and one of them is GPT-2. The spans of `vocab` and `merges` are noted and skipped on the way past now, and read once the object has closed.
- A model object with no `type` member at all was refused. GPT-2 is one: its model object has seven members and the type is not among them. The kind is read off the other members now, an array vocabulary being Unigram, a `merges` list being BPE and a word length limit being WordPiece, which leaves a plain word level vocabulary as the fourth model kind. That kind is new and is now implemented.
- Two wrong character lists in the Nmt normalizer. It used U+2000 through U+200F where SentencePiece uses U+200B through U+200F, so four space characters were turned into ordinary spaces that should have been left alone, and it never mapped tab, newline, form feed or carriage return to a space at all.
- `\w` in the regular expression engine meant the letter categories, and the crate the model files were tested against reads it as Alphabetic. The two differ by the letter numbers, where the ideographic zero and the Roman numerals live, and by the circled and squared Latin letters, which the database files as symbols. A pre-tokenizer that calls the ideographic zero a symbol cuts a Chinese word in half.
- A byte level pre-tokenizer with `add_prefix_space` behind a Bert pre-tokenizer, given a string of nothing but spaces, invented a token containing a space, because the prefix step made a piece to put the space in front of when the earlier step had correctly thrown everything away.
- The same step, given several pieces, prefixed only the first. The reference runs the prepend over every one of them, which is how `a  b` keeps the space in front of `b` that the splitting threw away.
- The beginning of text reconciliation in `encode_rendered` only fired when the post processor template was one run of specials followed by the sequence. Whisper writes three separate specials in front of the text, so the id that doubles up is the third and the rule never looked at it. It finds the sequence and takes the special before it now, wherever that is.

## [0.2.1] - 2026-09-01

The model plane starts. Two readers, one model spec, and nothing that touches a weight yet.

molla now reads a GGUF file and a Hugging Face directory and answers the same questions about either: what architecture it is, what shape it is, which tokenizer it wants, what the tensors add up to, and what this build could actually do with it. `molla spec <path>` prints that for both, and dispatches on what is at the path rather than on the extension.

The point of doing both formats before doing anything with the weights is that everything above this has to be written once. The tokenizer in #21, the weight loading in #25 and the architecture blocks in #26 all read a `ModelSpec` and none of them will care which file it came out of. The one place the formats genuinely disagree is shape order, GGUF writing the fastest varying dimension first and safetensors writing row major, and that is written down where the code that has to reconcile it will find it.

Both readers check rather than trust, since both formats hand you byte offsets out of a downloaded file. The GGUF tensor directory is recomputed from the block geometry of every type and required to match every offset in the file. Every safetensors range is checked against the mapping and then against the dtype and the shape. Neither reader allocates per tensor and neither reads past the header, so a 7.3 GB two shard repository costs twenty milliseconds and twelve megabytes of resident set.

The check that mattered most was reading the same four models both ways. bge-small, SmolLM2-135M, gemma-3-270m and Qwen2.5-0.5B agree on the architecture, every geometry field, the tokenizer algorithm and every special token id, and the four places the numbers differ are each a real difference between the two files rather than a bug in one of the readers.

### Added

- `molla.model.spec`, the mapping from a GGUF file to a `ModelSpec`: architecture id, geometry, tokenizer shape, the block geometry of every ggml tensor type, and the capabilities the file declares intersected with what this build can do with them. Nothing in it reads a weight. Checked field by field against what llama.cpp loads from the same four models, and the tensor directory is recomputed from the block sizes and required to match every offset in the file. See [docs/validation/spec.md](docs/validation/spec.md).
- `molla spec <path>`, which prints all of that for a model file. Seven milliseconds and fourteen megabytes of resident set against a 468 MB model.
- `Gguf.flt`, `float_or`, `bool_or`, `has`, `array_count`, `tensor_index` and `tensor_prefixed`, so the layer above can ask about floats, flags and tensor names without decoding anything it did not ask for.
- `molla.model.safetensors`, the safetensors container: the header, the tensor directory, and sharded repositories resolved through `model.safetensors.index.json`. Every byte range is checked against the mapping and against the dtype and the shape before it is kept, and index against shard disagreement is counted in both directions.
- `molla.model.repo`, the Hugging Face directory around it, producing the same `ModelSpec` the GGUF path produces. `config.json` for the geometry, the transformers class name for the architecture, and a streaming pass over `tokenizer.json` for the counts and the special token ids, so a 33 MB tokenizer costs no heap. Checked against the same four models read both ways, and against a 7.3 GB two shard repository. See [docs/validation/safetensors.md](docs/validation/safetensors.md).
- `molla safetensors <path>`, which prints the header and tensor directory of a file or a directory. `molla spec <path>` now takes either format and dispatches on what is actually there rather than on the extension.
- `ModelSpec.source`, which says which reader produced the spec, and `TokenizerSpec.embedding_rows`, which says how many rows the embedding matrix has when the model states it. Qwen2.5 has 151665 token ids in a matrix of 151936 rows, and gemma 3 has one id past the last row, so the two numbers are not the same question.
- `TokenizerSpec.model_source`, the tokenizer name as the file spelled it. `model` is now normalised across the two formats, so GGUF's gpt2 and `tokenizer.json`'s BPE both report as bpe.

## [0.2.0] - 2026-09-01

M1 is done. Everything Mojo 1.0 does not ship and a server cannot do without: syscall wrappers, buffers and arenas, a reactor, HTTP/1.1, SSE and NDJSON, a JSON scanner, client TLS, threads and queues and a shutdown that finishes what it started, config and logging and metrics, and two commands that assert the properties the rest of it claims.

The milestone was to write eight layers a normal server project gets for free, and the point of ending it here is that there is now something to build on that has been measured rather than asserted. The server answers HTTP on Linux, macOS and Windows under WSL2, it holds a thousand connections through an hour without leaking, and it allocates nothing on the request path. Those are three claims and each of them has a command that fails if it stops being true.

This release adds the last of those, the hour long soak, and fixes the leak it found. The timing wheel cancelled lazily on a written assumption that nothing would ever create enough dead timers to matter, and every connection that closes cancels its idle timer, so the assumption was never true for any server anybody would run. It took an hour of real churn to become visible: on the laptop the run went from 127 MB to 2.3 GB. The sign off round was four machines, an hour each, a thousand connections, seven hundred million answers, no 5xx and no wrong statuses, and on every one of them the busiest timing wheel ended holding exactly as many entries as its reactor had connection slots.

There is still no model, no routing beyond the handful of built in routes, and no server side TLS. Those are M2 and later. What is here is the floor.

### Added

- `molla httpsoak`, an hour long soak on the systems layer. Five kinds of client at once against a real server: keep alive, streaming, slow readers that fill the write ring and hold it full, abrupt disconnects that never read the answer, and oversized bodies that get a 413 and a close. It watches resident memory, descriptors, the log ring and the connection table, the size of the busiest timing wheel, and latency drift across ten segments of the run. Runs nightly on Linux and macOS through `.github/workflows/soak.yml`, and a short version runs in the test suite. See [docs/validation/soak.md](docs/validation/soak.md).
- `molla.net.latency`, the segmented latency histogram both soaks now share, so the drift gate means the same thing in each.

### Fixed

- The timing wheel cancelled lazily, marking a timer dead and leaving it in its slot to be freed when that slot was next walked. Every connection that closes cancels its idle timer, and a slot is not walked until time gets close to the deadline it holds, so a busy server accumulated a dead slab entry per connection for the length of its idle timeout and never gave the memory back. The hour long soak grew from 127 MB to 2.3 GB on macOS and from 60 MB to 1.4 GB on Windows before this. Slot lists are doubly linked now and cancel unlinks and releases in constant time.
- `HttpProtocol._error` answered a 413, a 414 or a 431 without going through the handler path, which is where the status accounting lived, so those responses were never counted. A server that refused a hundred thousand oversized bodies reported `molla_http_responses_4xx_total 0`. Found by the soak on its first complete run.

## [0.1.7] - 2026-09-01

A per request allocation on a server whose whole pitch is not having one, found by the assertion added to stop exactly this.

The design has always said the request path allocates nothing in steady state. Nothing checked it, and it had already stopped being true: the reactor rebuilt a `Connection` when it reused a slot, so every accept freed and re-allocated the read buffer and the write ring, and the read buffer then grew again on the first request that did not fit. Three allocations per connection, which against a client that opens a connection per request is a per request allocation.

`molla allocs` is the check that will not let that happen again. It runs a mixed load until one pass of it costs nothing and then requires the next pass to cost nothing too, with no tolerance, because a tolerance is a budget and a budget gets spent. It runs in CI on every commit on all three platforms and a smaller version runs in the test suite.

### Added

- `molla allocs`, which runs a mixed load against a real server until a pass of it costs nothing and fails if the pass after that allocates anything. The load is a pipelined batch of a GET, a HEAD, a 404, a Content-Length body, a chunked body, eight pipelined GETs and both streaming routes, so an allocation on the fifth kind of request is not invisible. Runs in CI on every commit on all three platforms, and a smaller version runs in the test suite. See [docs/validation/allocations.md](docs/validation/allocations.md).

### Fixed

- The reactor rebuilt a `Connection` when it reused a slot, which freed and re-allocated the read buffer and the write ring on every accept and then grew the read buffer again on the first request. That is three allocations per connection, and against a client that opens a connection per request it is a per request allocation. `Connection.reuse` now takes over the new socket and keeps the buffers at whatever size the traffic already paid for.

## [0.1.6] - 2026-09-01

The operations surface. Config, structured logging, Prometheus metrics, and three `/molla` routes, none of which makes molla faster and all of which is what makes molla debuggable by somebody who is not holding the source open.

Config carries where each value came from rather than only the value. The precedence is flag over environment over file over default, and `molla config get` prints both halves, because the value on its own is not the question anybody has at three in the morning.

Logging is a byte ring per worker, written by that worker and read by the housekeeping thread, which makes it single producer single consumer and means nothing on the request path waits for anything. Records are built straight in the ring and become visible only when the length header is written last. The criterion was no allocations at a disabled level, and the cost turned out to be one atomic load and a comparison.

Metrics are one set of counters per worker on its own cache line, summed only at scrape time, so a request never touches a line another core owns. Statuses are bucketed by class rather than per code, and durations are integer nanoseconds beside a count, with the HELP text admitting there is no histogram rather than exporting a number that looks like a quantile.

The three admin routes share the main port. A second listener is the safer answer and is also a second thing to configure, expose in a container, and forget, and everything they return is already printed on startup.

### Added

- `molla.ops.config`, settings with a precedence of flag over environment over file over default, and every setting carrying where its value came from.
- `molla config get [key]`, which prints the effective value and the source, because the value on its own is not the question anybody has.
- `molla.ops.log`, structured logfmt logging on a byte ring per worker, written by the worker and flushed by the housekeeping thread. A disabled level costs one atomic load and no allocations at all.
- `molla.ops.metrics`, Prometheus counters with one set per worker on its own cache line, summed only when somebody scrapes them. Durations are exported as integer nanoseconds.
- `GET /molla/version`, `/molla/health` and `/molla/metrics`, served on the main port and off unless a caller turns them on.
- `sys.clock.monotonic_ns` and `sys.clock.realtime_ns`, so durations and timestamps stop sharing a clock.
- `tests/test_ops.mojo`, including the check that a thousand log calls at a disabled level allocate exactly nothing.
- `docs/validation/ops.md`.

### Changed

- `molla drain` now runs with logging, metrics and the admin routes on, asks for all three routes before the load starts, and prints the counters after the drain.

## [0.1.5] - 2026-09-01

The concurrency layer, and a shutdown that finishes what it started. Also the TLS policy work, which decides per host whether a certificate has to check out.

One rule runs through all of it: anything two threads touch lives at an address rather than in a value. A Mojo value moves, so two threads reaching a counter through two copies of it are incrementing two counters, and that looks entirely correct in a single threaded test. Every shared thing here is allocated once, kept at a fixed address, and handed to a thread as the one integer a thread entry point gets.

Three pieces are not built the way the design said they would be, and each of them is a Mojo 1.0 fact rather than a preference. `Once` cannot be `pthread_once`, whose callback takes no argument, because an initialiser would have nowhere to leave its result except a global and there are no globals. Signals cannot arrive on signalfd, which needs the signal blocked in every thread of the process, because the runtime starts threads before `main` and offers no way to reach their masks. And the signal has to be armed before the server starts, which the first version got wrong in a way that only shows up when something signals faster than a person can press Ctrl-C.

Draining means closing the listeners and then closing each connection at the first moment it owes the client nothing, cutting and counting what is left at the deadline. Writing the test for it found a real bug: the poller is edge triggered, so a request that arrived between two drain passes has already spent its edge, and a connection about to ask for something looked exactly like one that was asleep. The difference is a request the client sent and never got an answer to.

The acceptance test is a command rather than a unit test, because every time in a hundred runs is not something a unit test reaches. A hundred runs of thirty two connections are clean on all four machines in the fleet, and the only failing check anywhere is the WSL2 backpressure one that was already failing.

There is still no routing and no config file. Both are M1 and neither is here.

### Added

- `molla.sys.atomic`, with `AtomicRef` for one atomic integer reached by address and `AtomicBlock` for several of them each alone on a cache line. Everything two threads share in molla lives at an address rather than in a value, because a Mojo value moves and two threads reaching a counter through two copies are incrementing two counters, which looks correct in a single threaded test.
- `molla.sys.queue`, a bounded ticket based MPSC queue with padded cells and an SPSC ring with no compare and exchange in it at all. The cell padding matters more here than in the usual description of the algorithm, since molla's queues run close to empty and a queue holding one item has the producer's cell and the consumer's cell next to each other.
- `Once` in `molla.sys.thread`, which is not `pthread_once`. The callback `pthread_once` takes has no argument, so an initialiser has nowhere to leave what it made except a global, and Mojo 1.0 has no globals. This one takes the same function and integer a spawned thread takes, and tells the caller that ran the body apart from the callers that waited.
- `molla.net.context`, a `ServerContext` holding every setting a server has, made by the caller and passed down. It is what makes a server testable in process, and it is the reason there is no configuration global to remove later.
- `molla.net.supervisor`, signals delivered as a readable descriptor through a self pipe, and `serve_until_signal`. Not signalfd, which needs the signal blocked in every thread and Mojo starts threads before `main` with no way to reach their masks, and not EVFILT_SIGNAL, which works but is macOS only and would leave the shutdown path a different mechanism on each platform. SIGTERM and SIGINT drain, SIGQUIT dumps every worker's state, thread, connection count and queue depth first.
- `Server.drain` and the `DrainReport` it returns. A draining reactor closes its listeners and then closes each connection at the first moment that connection owes the client nothing, with a deadline after which what is left is cut and counted. A shutdown that reports it dropped four connections after nine seconds is one you can act on, and a process that just exits is not.
- `molla drain [connections] [deadline_ms]` and `scripts/drain-loop.sh`, which is issue #15's acceptance test. Each run loads every connection with a pipelined batch, sends the process SIGTERM, and fails unless every client received every answer whole. A hundred runs of 32 connections is clean in 3s on the M4.
- `ServerContext.send_buffer_bytes`, which sets a small kernel send buffer on accepted sockets. Zero in production, where Linux sizes this per connection and does it better than a number written down once. The drain test sets it low so most of a connection's answers are still molla's problem when the signal lands, and it goes on the accepted socket rather than the listener because a listener does pass the option down and macOS then autotunes the inherited buffer back up.
- `tests/test_concurrency.mojo`, 80 checks that all use real threads. Four threads and twenty thousand increments each against exactly eighty thousand, eight threads racing one `Once` for exactly one winner, three producers against a queue too small to hold one producer's share so every one of them meets a full queue, and a check that each of the 1500 values comes out exactly once.
- `docs/validation/threading.md`, with the sharing rule the layer is built on, the three places the obvious version does not work in Mojo 1.0, and the measurements behind the drain test.
- `molla.tls.policy`, which decides whether a certificate has to check out, by host name. There is no global insecure flag and there will not be one: a pull is not one connection, since ghcr.io answers a blob request with a redirect to a signed URL on a host it names, so a process wide switch would turn verification off for a host chosen by the response. `molla pull --insecure` and `molla tls --insecure` cover the host on the command line and nothing else, and every connection that skips verification says so on stdout.
- `probe` in `molla.tls.client` and a `tls` line in `molla version`, saying which backend loaded and how high it can negotiate, or why there is none. TLS is dlopened, so a machine without it runs everything except HTTPS, and that is now a line of output rather than a promise in a document. It also puts the macOS TLS 1.2 cap on the screen of the machine it applies to.
- `MOLLA_SECURITY` and `MOLLA_COREFOUNDATION` overrides for the two macOS framework paths, matching `MOLLA_LIBSSL` on the other platform. Nobody moves Security.framework, so these exist to point the loader at something that does not load, which is the only way to test what molla does on a host with no TLS library.
- `tests/test_tls.mojo`, 17 checks over the policy and the probe, including the missing library case on every machine that runs the suite. The one that matters is negative: naming a registry insecure must leave the CDN it redirects to verified.

### Fixed

- A drain no longer treats a connection with an unread request on it as idle. The poller is edge triggered, so a request that arrived between the last pass and this one has already spent its edge and there is no second one coming, and a connection about to ask for something looked exactly like one that was asleep. The difference is a request the client sent and never got an answer to. The drain now reads every live connection once more before deciding it is finished, which costs one recv per idle connection per drain pass and is only paid during a shutdown.
- `molla version` prints the version it was built at. It said 0.1.2 for the 0.1.3 and 0.1.4 releases, because the release process bumps `pixi.toml` and the changelog and nothing was checking that `VERSION` in `build_info.mojo` came with them. `scripts/check-version.sh` now fails CI when the three disagree, which is cheaper than noticing it in a bug report six months from now.

## [0.1.4] - 2026-09-01

JSON, in both directions, at a bit over 2 GB/s on the M4. Two modes over one SIMD scanner: a pull loop for request bodies, which is nearly all of the traffic, and a small tree for config and manifests. Object keys keep the order they arrived in, which matters because a tool call's arguments came from a model and the order is part of what it said.

Numbers are the half of a JSON library that is either right or nearly right, and nearly right shows up months later as one customer whose floats come back different. There is no `strtod` here, so no locale and no copy to get a NUL terminated string, and the conversion is exact for every input including the ones written specifically to break converters.

Running the suite on four machines was worth more than the parser was. It passed on the M4 and failed six checks on both EPYC boxes, because Mojo destroys a local at its last use and the reader holds its document as an address, so the buffer was freed while the parse was still reading it. The M4 passed because the freed block still held the bytes. That is a bug nothing about x86 caused and one machine would have shipped.

There is still no routing. That is M2.

### Added

- `molla.json`, a JSON parser and serializer with two modes over one SIMD scanner. `scan.mojo` classifies bytes a vector at a time and finds the quote, the backslash and any raw control byte with one mask, so validating a string costs nothing extra. `reader.mojo` is streaming mode, `dom.mojo` is DOM mode, `serialize.mojo` writes into a buffer with no intermediate allocation and keeps object keys in the order they arrived.
- Exact integers and correctly rounded doubles with no `strtod` and no locale, in `number.mojo` and `decimal.mojo`. Three paths: an integer that fits in 64 bits stays an integer, a double with 53 bits of digits and a small exponent goes through Clinger's fast path, and anything else goes through an exact decimal expansion. Printing goes back through the same struct and gives the shortest form that reads back as the same double, with the two cutoffs JavaScript uses.
- `molla jsonbench [kb] [n]`, the acceptance test for #13 as a command. On the M4 a 100 kB chat body parses at 2283 MB/s with zero allocations, against a gate of 1 GB/s. DOM mode is 1920 MB/s and five allocations for a 1234 node document.
- `tests/test_json.mojo`, 154 checks over the scanner, both number directions, the reader, the DOM and the writer, including the inputs that break converters and a round trip over four thousand doubles built from random bit patterns.
- `keep` in `molla.sys.mem`, which counts as a use of a value and does nothing else. Mojo destroys a local at its last use, so handing a buffer's address to a reader is the last thing the compiler sees using it and the buffer is freed while the parse is still reading it. It passed on the M4, where the freed block still held the bytes, and failed six checks on x86_64 Linux, where the allocator hands the block straight to someone else.
- `docs/validation/json.md`, with the numbers, the three places the design departs from what issue #13 describes, and the lifetime bug the fleet run found.

## [0.1.3] - 2026-09-01

The request path. A parser, bodies, multipart, a serializer, and the two streaming writers a completion needs, all on the reactor from 0.1.2.

The parser stops at the blank line and the body is read separately after it, which sounds like a detail and is the reason a body larger than the read buffer is no longer a contradiction. Content-Length, chunked and multipart all go through one call that says how much it took and whether it is done, and over a megabyte the body spills to a file, so an upload costs bounded memory rather than its own size.

The streaming writers are the half of the request path an inference server actually spends its time in. An event is flushed on its own as soon as it exists and events are only combined into one chunk when the socket cannot take them. Writing them found a real hole in 0.1.2: the reactor only called `on_writable` when the poller reported a socket writable, which is correct for a request and a response and leaves a stream stopped after one ring's worth of bytes, because a client that sent one request and is waiting produces no read edge and a drained ring produces no write edge. Nothing before this produced without being asked, so nothing had caught it.

There is still no routing and no JSON. Both are M1 work and neither is here.

### Added

- `molla.http` gets the parser the request path will actually run on. `scan.mojo` finds delimiters sixteen bytes at a time, `request.mojo` parses a request line and a header block into zero copy spans and stops at the blank line, `body.mojo` reads the body after that, `serialize.mojo` writes responses without allocating, `multipart.mojo` streams multipart/form-data, and `protocol.mojo` puts all of it on the reactor from #10.
- Bodies are read separately from headers, so a body larger than the read buffer is no longer a contradiction. Content-Length, chunked and multipart all go through one call that says how much it took and whether it is done. Over a megabyte the body spills to a file opened `O_EXCL` with mode 600, so peak memory for an upload is bounded by the threshold rather than by the upload.
- A hostile input corpus in `tests/test_http.mojo`, one case per refusal, asserting the status and not just the rejection. Bare LF, bare CR, Content-Length with Transfer-Encoding, duplicate framing headers, a Transfer-Encoding that is not chunked, whitespace before a colon, obsolete line folding, zero or two Host headers, control characters in a field, an Expect we cannot answer, and the 8 KiB, 64 KiB and 128 header bounds.
- An assertion that two thousand responses allocate nothing, measured after a warmup so the number does not hide the first few responses growing the buffer.
- `MODE_600` in `molla.sys.file`.
- `molla.http.stream`, the SSE and NDJSON writers, over chunked transfer encoding. An event is flushed on its own as soon as it exists and events are only combined into one chunk when the socket cannot take them, which falls out of holding framed payload and chunk framed bytes in separate buffers. Past the staging limit a producer gets `STREAM_FULL` rather than a bigger buffer. Event names, ids and NDJSON records are refused if they contain a line break, since either one silently desynchronises the client rather than failing.
- The SSE heartbeat, as `heartbeat_due` and `heartbeat` taking the current time rather than reading a clock, so it fits a protocol trait with no tick in it and so the test is deterministic. NDJSON deliberately has none, because it has no comment syntax and a blank line is not portably ignored.
- `Connection.produce`, one bit saying the protocol has more to write on its own initiative, and `/stream/sse` and `/stream/ndjson` on the M1 protocol.
- `tests/test_stream.mojo`, 59 checks over framing, validation, backpressure, the heartbeat, a reader taking one byte at a time against pinned 8 kB socket buffers and a 2 kB output ring, and a client that hangs up mid stream.

### Changed

- The reactor calls `on_writable` for a connection whose protocol says it is producing, not only for one the poller reported writable, and counts bytes leaving the output ring as progress for the service loop. Without both, a streaming response stops after one ring's worth: the client is not going to send anything else, so there is no read edge, and once the ring drains there is no write edge either. Nothing before #12 produced without being asked, so nothing had exercised it.
- `write_decimal` is a free function in `molla.http.serialize` rather than a method, since the streaming writers need the same non allocating decimal into their own buffers.
- `molla.http.server`, the M0 spike, answers 501 to a request with a body instead of half reading one. The parser it calls no longer consumes bodies, and the spike is kept as the evidence behind the M0 throughput numbers rather than as something to build on. Benchmark anything other than a bare GET against `molla.http.protocol`.

## [0.1.2] - 2026-09-01

The event loop the request path will run on. One reactor per worker thread, each owning its poller, its connection table and its timers, with nothing shared on the request path.

There is still no request. This is the layer HTTP gets written against, and the reason it is tagged on its own is that it is the last piece that can be validated purely against a kernel, before anything above it can hide a bug in it.

Three things came out of building it. Servicing a connection cannot be a single pass over a readiness event, which is the shape every event loop tutorial has, because a connection that reads more than its output ring holds ends up with no read edge and no write edge coming and a response half written. It never happens under light load and happens every time under the load that makes it worst. The backpressure test that catches it was only honest on one platform until the socket buffers were pinned at both ends, since Linux starts them larger and grows them, so the amount of data needed to provoke a short write is different on every machine. And a test that waits for another thread has to wait in milliseconds rather than in loop iterations, because the two convert at a rate that depends on how loaded the machine is, and the budget runs out soonest on the machine that needed it to last longest.

A thousand connections held for an hour on the M4 and on server1, no mismatched payloads across 657195369 and 59256042 round trips, no descriptor growth, and a p99 flat to the bucket across all ten segments of both runs.

### Added

- `molla.net` gets a real reactor. One event loop per worker thread, each owning its poller, its connection table and its timers, with no shared state on the request path. `reactor.mojo` is the loop and the four call protocol trait everything above it will be written against. `conn.mojo` is one connection and the four states a non blocking socket can be in. `listener.mojo` decides how connections are spread, which is SO_REUSEPORT sharding on Linux and a round robin handoff on macOS because macOS gives the last binder every connection instead of balancing. `server.mojo` is N reactors on N threads behind one address, TCP or unix.
- `molla.net.wheel`, a four level timing wheel at 100 ms resolution. An idle connection costs one entry in a slab and no timer descriptor, which is what makes a thousand idle keep alive connections free rather than a thousand syscalls per second.
- `molla netsoak`, the acceptance test for issue #10. A thousand connections with mixed idle and active traffic, latency compared across ten segments of the run so drift shows up as a number, and descriptors and peak memory checked at the end.
- `tests/test_reactor.mojo`, 61 checks on macOS and 62 on Linux over the wheel against an explicit clock, the reactor stepped by hand, backpressure, idle timeouts, unix sockets and the threaded server.
- `set_keepalive`, `listen_tcp_shared` and `set_buffer_size` in `molla.sys.socket`, and `monotonic_ms` in `molla.sys.clock`. `set_buffer_size` exists because an accepted socket inherits its send and receive buffers from the listener, which is the only way to make a backpressure test hit the same wall on both platforms.

### Fixed

- The threaded server tests waited for a worker thread by counting empty non blocking reads rather than by watching a clock, so on a loaded runner the whole budget could burn through before the thread that owed the answer was scheduled at all. Both waits now run against a monotonic deadline and sleep between attempts instead of spinning on the core the worker needs.

## [0.1.1] - 2026-08-31

Two layers of the standard library Mojo 1.0 does not have. The OS boundary, and the memory the request path will live in. Nothing above them exists yet, so nothing here changes what molla can do, and both are the kind of thing that is much cheaper to get right before there is a server on top of it.

The sys layer is every OS call molla makes, in one module, returning one result type that carries errno from the call site. Files, threads, mutexes, condition variables and signals, all tested against a real kernel on three machines and both architectures.

The io layer is buffers, rings and arenas, with the growth policy of each written down next to the code rather than left to be inferred, and an allocation counter underneath so the zero allocation claim in M1 can be a number instead of a promise.

### Added

- `molla.sys` grows the rest of the OS boundary. `result.mojo` holds the one type every wrapper returns, carrying a value and the errno captured at the call site. `file.mojo` covers open, read, write, seek, truncate, sync, stat, unlink, rename and directory listing. `thread.mojo` covers pthreads, mutexes and condition variables, which is what Mojo 1.0 has no threading module for. `signal.mojo` covers dispositions, masks and a self pipe that turns a signal into a readable descriptor.
- `tests/test_sys.mojo`, which runs every wrapper against the real OS. FFI mistakes show up as memory corruption somewhere else entirely, so they are caught at the boundary or not at all.
- `access` behind `exists`, `writev` behind `write_vectored`, `socketpair`, unix domain sockets and `shutdown` in `molla.sys.socket`.
- `docs/validation/sys.md`, which records what the boundary covers, what ran green on which machine, and the four platform traps that cost a session each.
- `molla.io`, the memory layer the request path is built on. `buffer.mojo` is an owned growable buffer with a written down growth policy, doubling to 64 kB and then a fixed step. `ring.mojo` is the per connection output ring, so a short write costs two integer updates instead of a memmove, and it hands `writev` its one or two pieces directly. `arena.mojo` is a bump allocator with a per request lifetime, freed in constant time. `bytes.mojo` compares, searches, trims and parses spans without allocating.
- `molla.sys.mem`, which is where every allocation in molla goes, and the allocation counter that makes "this request allocated nothing" a number rather than a claim. Issue #17 is what the counter is for.
- `tests/test_io.mojo`, 116 checks over the growth policy, the ring wrap, the arena and the byte helpers.

### Changed

- `molla.sys.mmap` opens and closes through `molla.sys.file` instead of declaring `openat` a second time. Two declarations of one C symbol with different argument counts in the same build fail to lower, and the error points at the standard library rather than at either file that caused it.

## [0.1.0] - 2026-08-31

M0 is done. The question it existed to answer was whether Mojo 1.0 can hold a socket, parse HTTP fast enough, map a model file, call a kernel on a GPU and reach a TLS library, and the answer is yes on all five, with numbers rather than opinions behind each one. Two decisions were taken at the gate and both are recorded with the measurements next to them.

D1 holds. The network edge stays in Mojo, and the Rust fallback stays documented and untaken. On the M4 a trivial handler runs between 43705 and 249896 requests per second depending on how loaded the machine was, against a gate of 5000. A thousand concurrent connections held for sixty seconds on kqueue and on epoll with flat memory. The TLS binding pulls the same blob from ghcr.io through three different libraries on four machines. One condition, the multi threaded one, cannot be tested because Mojo 1.0 has no threading module, so it moves to the M1 gate rather than being rounded up.

D6 does not. `max/kernels` needs proprietary `max-core` at runtime, its CPU kernels included, so the promise of an optional MAX runtime was describing a seam that does not exist. molla accepts the dependency, which means running molla means installing a proprietary runtime, and the README says so instead of claiming otherwise.

This release is still a foundation. It does not serve a model. M2 is the first one that does.

### Added

- `docs/adr/`, for decisions taken at a gate against measurements, with `0001-network-edge-stays-in-mojo.md` and `0002-accept-max-core.md`, the two M0 gate records

### Changed

- D1 in `docs/design.md` records the M0 gate outcome. The network edge stays in Mojo. The multi threaded half of the third reversal condition moves to the M1 gate, because Mojo 1.0 has no threading module to test it with.
- D6 is rewritten. `max-core` is a required dependency at runtime rather than an optional backend, because `max/kernels` does not run without it and its CPU kernels do not either. Running molla now means installing a proprietary runtime, and the README names both proprietary packages and what each is needed for.
- D7 is marked load bearing. `max/kernels` has no quantized matmul that will launch on an Apple GPU below an M5, so every Apple GPU kernel is one molla writes, and D7's per target numerics tests are what keep the portability claim honest.

## [0.0.3] - 2026-08-31

The M0 kernel spike ran, and it changed what the README is allowed to claim. The TLS spike ran after it, and molla can now pull a blob from ghcr.io over HTTPS on macOS and Linux.

### Added

- `molla.tls`, client TLS over OpenSSL 3.x and 1.1.1 on Linux and Secure Transport on macOS, both loaded with dlopen so a machine without a TLS library still runs molla and only loses HTTPS
- `molla.http.client`, a GET only HTTPS client with redirect following and chunked bodies, and `molla.registry.ghcr`, enough of the OCI distribution protocol to fetch a blob and check its digest
- `molla tls <host>` prints the backend, protocol, cipher and certificate chain, and `molla pull <ref>` pulls a blob from ghcr.io and verifies it
- `molla.sys.dns` for `getaddrinfo`, `molla.sys.sha256`, `molla.sys.cstr`, and `dial` in `molla.sys.socket` for a blocking socket with timeouts
- `MOLLA_LIBSSL` and `MOLLA_LIBCRYPTO` to point at a specific OpenSSL, which is also how the 1.1 fallback gets tested on a machine that has 3.x
- `docs/validation/tls.md` with the results from four machines and three TLS libraries
- `spikes/qmatmul/`, the kernel spike for issue #5, with its own pixi manifest so its proprietary dependency stays out of the root build
- `docs/validation/kernels.md` with the licence audit, the numbers from six machines, and the three options for what molla does next
- Numerics tolerances for Q4_K matmul on CPU and on GPU, which D7 asked for and never gave

### Changed

- The README no longer claims there is no proprietary dependency in the stack, because there is, and it names it
- D6 in `docs/design.md` is marked under review, since `max/kernels` does not build or run without proprietary `max-core`, its CPU kernels included
- D7 in `docs/design.md` is marked achievable but not inherited, since one source did compile to Metal and sm_89 with byte identical output, but `max/kernels` is not organised that way

### Known issues

- Carried over from 0.0.2 unchanged: the compiler is proprietary, so releases are source only. Build with `pixi run build`.
- `max/kernels` has no quantized matmul that will launch on an Apple GPU below an M5. On an M4 it raises at launch. The spike wrote its own kernel to get a Metal number at all.
- What molla actually does about the licence finding is not decided here. That is issue #7.
- TLS on macOS caps at 1.2. Secure Transport has no TLS 1.3, and the framework that does is built on Objective-C blocks, which Mojo cannot emit. Linux gets 1.3 through OpenSSL.
- The HTTPS client is IPv4 only, opens a connection per request, and reads bodies into memory whole. None of that is suitable for pulling a model and M3 replaces it.

## [0.0.2] - 2026-08-31

Four of the seven M0 spikes are done. molla can now map a model file and read what is in it, though it still cannot read a tensor.

### Added

- `molla.sys.mmap`, a read only whole file memory map
- `molla.model.gguf`, a GGUF v2 and v3 metadata reader that walks the header, the key value block and the tensor directory in place without copying the file, and `molla gguf <path>` to dump one
- `docs/validation/gguf.md` with the comparison against `gguf-dump` on four models covering bert, llama, gemma3 and qwen2, and what the zero copy read is actually worth

### Known issues

- Carried over from 0.0.1 unchanged: the compiler is proprietary, so releases are source only. Build with `pixi run build`.
- Nothing reads a tensor. The GGUF reader records where each one is and what type it is, and stops there.
- Metadata arrays are measured and skipped rather than decoded, so there is no way to read a tokenizer vocabulary yet.

## [0.0.1] - 2026-08-31

First tagged release. Three of the seven M0 spikes are done: the toolchain is pinned across the fleet, sockets and the event loop work on epoll and kqueue, and HTTP/1.1 parse and respond clears the throughput gate. Nothing serves a model yet.

### Added

- Pixi workspace with the Mojo toolchain pinned to 1.0.0, locked for macOS arm64, Linux x86_64, and Linux arm64
- A `molla` binary with `version` and `help`, reporting the toolchain and detected host
- A test runner, since Mojo 1.0 has no `mojo test`
- CI builds and tests on all three platforms for real, and smoke tests the binary
- `docs/validation/toolchain.md` recording the pin, the machines validated so far, and what Mojo 1.0 actually looks like against the release notes
- `molla.sys`, the libc boundary: errno, descriptors, IPv4 TCP sockets, and one `Poller` over kqueue and epoll with read and write interest
- `molla.net.echo`, a non blocking edge triggered TCP echo server, and `molla echo` to run it
- `molla soak`, which holds a thousand connections for sixty seconds and checks for descriptor and memory leaks
- `docs/validation/sockets.md` with the soak results on all three platforms and what the spike says about D1
- `molla.http`, a zero copy HTTP/1.1 request parser and a prebuilt response with an in place `Date` field, and `molla http` to run the throughput spike
- `docs/validation/http.md` with the M0 throughput measurements, the fleet results, and the two allocation and socket problems that cost more than the parser did
- Design document, roadmap, and milestone plan
- CI with docs linting, workflow linting, CodeQL on workflow definitions, OpenSSF Scorecard, and dependency review
- Release pipeline with SBOM, build provenance attestation, and keyless signing
- `scripts/check-action-pins.sh`, run in CI, which fails if an action is pinned to an annotated tag object rather than a commit

### Changed

- The toolchain version lives only in `pixi.toml` now, rather than also in a CI environment variable

### Fixed

- Six actions were pinned to annotated tag object SHAs instead of commit SHAs, which made the OpenSSF Scorecard workflow fail on publish with an imposter commit error even though the scan itself succeeded

### Known issues

- The Mojo compiler we build with comes from Modular's conda channel under a proprietary license, so the build is not yet Apache-2.0 end to end even though the source is. See `docs/design.md`.
- Releases are source only for the same reason. A Mojo 1.0 binary links `libKGENCompilerRTShared` and two other runtime libraries that ship only as shared objects under `LicenseRef-Modular-Proprietary`, and the linker bakes RUNPATH to the pixi directory that built it, so a bare binary does not start anywhere else. Publishing a working tarball would mean redistributing Modular's runtime inside molla's own artifacts. Build with `pixi run build` until that changes.

molla answers HTTP requests as of the M0 spike, but every path returns the same fixed body. The first milestone that serves a model is M2.
