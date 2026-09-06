# Prefill

molla had no prefill. It had a decode loop that a prompt was also fed through, one token at a time, which is why molla's prefill rate and its decode rate were within a few per cent of each other on every row of every table in [bench.md](bench.md). A 514 token prompt was 514 forward passes and about 232000 kernel launches, and 1400 ms of time to first token on a 4090 against llama.cpp's 17 ms. Between 109 and 175 times off, and it was the largest single gap molla had against anything.

This is what was built, what it costs in scratch, and what each piece has to agree with.

## The shape

A prompt is a matrix. Every matvec on the prefill path is the same weight against many activation vectors, so it is a matmul with the prompt length as the free dimension, and the launch count for a whole prompt becomes the launch count for one token.

That last sentence is only true if the prompt is done in one pass, and it should not be, for a reason that has nothing to do with launches. Attention over a chunk of `T` tokens needs somewhere to put `T` rows of scores, and the score row for a token at position `p` is `p + 1` long. Sized for the whole context that is `T * heads * context` floats, which on a 4096 context 32 head model is 2 MiB a token. A 512 token prompt in one pass would be a gigabyte of scratch to save 7 ms of launch time.

So prefill is chunked. `PREFILL_CHUNK` tokens go through the stack together, the next chunk follows, and a 512 token prompt is two passes rather than 512. The chunk is 256, which bounds the scores at 134 MiB on an 8B at a 2048 context and 19 MiB on SmolLM2 135M. It was 64 when this was written, and what moved it is in `PREFILL_CHUNK`: giving a thread four output rows made a wider chunk a gain on every model where it used to be a loss on two of three.

## What the matmul has to do that the matvec does not

The obvious generalization is a block per output row per token, which is the existing kernel with a second grid dimension and no thinking. It is also the wrong one, because it reads the whole weight matrix once per token and prefill would be bandwidth bound at `T` times the weight size.

A block covers `SPAN` tokens instead. It reads and dequantizes each weight value once and multiplies it into `SPAN` accumulators against `SPAN` activations, so the dequantization work and the weight read are both amortized by `SPAN`. That is the whole point: decode is bandwidth bound at one value per weight byte and prefill should be compute bound, and `SPAN` is the knob that moves it.

`SPAN` is where the registers run out, so it is not the whole answer. Eight accumulators on CUDA and sixteen on Metal is as far as it goes before the accumulators spill, and a spilled accumulator undoes the change, so the rest of the amortization has to come from somewhere that is not registers. `MM_GROUPS` is that somewhere. A block is `MM_GROUPS` groups of `MM_TILE` threads, each group carrying its own `SPAN` tokens and all of them walking the same weight row, so the row is fetched from memory once for the block and out of the L1 for the groups behind the first. The groups share the weight and not the registers, which is why this goes past what `SPAN` can.

The two knobs multiply, and the product is the number that matters: a block covers `SPAN * MM_GROUPS` tokens, and the weight matrix is read `ceil(T / (SPAN * MM_GROUPS))` times for a chunk of `T`. Both backends want the same product, 64, which is what `MM_BLOCK_TOKENS` is and what `MM_GROUPS` is derived from. On a 4090 the 8B runs at 388 tokens a second with eight groups of eight and 262 with four of eight, and Metal falls off a cliff the other way, where half of every block is dead. The chunk was this number too when it was written and the two came apart later, so the grid now grows a block for every 64 tokens of the chunk.

`MM_TILE` is 32 on both backends, which is a warp on both. It is measured rather than assumed: 32 is the best of every width from sixteen to five hundred and twelve on both backends and on all three models, and the losses either side are large, since a matmul block reduces `SPAN` accumulators rather than one and past a certain width the reduction is more work than the dot product that fed it. Being a warp is what the reduction below then relies on.

The grid is `(ceil(T / (SPAN * MM_GROUPS)), rows)` and not the other way round, and that also matters. Blocks that share an output row have to be co-resident for the L2 to serve the weight row once, and the blocks that share a row are the ones that differ in the token index, so the token index is the fast axis.

The epilogue survives unchanged in meaning and changes in indexing. A bias reads `aux[r]`, the same element for every token in the block. A residual add reads and writes `o[t * rows + r]`. A gate reads `aux[t * rows + r]`. All three are still one thread's work at the end of a reduction.

## Three things the kernel cannot afford

The block reduces `SPAN` accumulators where the matvec reduced one, and everything that costs a register per token is now `SPAN` registers held for the length of the accumulation. Three of them were worth roughly a factor of five between them on CUDA, and all three are invisible in the shape of the code.

The tail lanes do not clamp. A group whose tokens run past the end of a short chunk reads past the last token rather than pinning onto it, because clamping needs a live token index per lane held in a register for the whole accumulation and reading slack is an affine offset the address unit folds in for free. What it computes is thrown away by the `t < live` at the end. The price is that every scratch vector a chunk uses is allocated rounded up to a whole block of tokens, which at the shipped chunk is exact and costs nothing.

The reduction is a warp butterfly and not a tree through shared memory. A group is a warp, so `lane_group_sum` reduces it in five shuffles and leaves the total in every lane, and lane `k` keeps the total for the token it is about to write. That removes the shared memory and all five barriers. It is worth 41 per cent on the 8B and nothing on the models that fit in cache, which is the shape of a change that buys occupancy.

The reduction loop has to be unrolled. This is the one that does not look like anything. A plain `for k in range(SPAN)` around the reduction indexes the accumulator array with a value the compiler will not treat as constant, and the whole array lands in local memory for the entire kernel, accumulation included. Writing it `comptime for` costs nothing and is worth 55 per cent on SmolLM2 and 64 per cent on Qwen. The loops inside the accumulation are the same shape for the same reason.

## The matrix core form

The kernel above is a good general matmul and it is not the fastest one either card can run, because both of them have an instruction that multiplies a small matrix by a small matrix in one go and neither is reachable from ordinary multiply and add. Apple has `simdgroup_matrix` at eight by eight and NVIDIA has `mma` at sixteen by eight by sixteen. There is one kernel for each, and they are two kernels rather than one parameterized one because everything below the tile disagrees: the fragment shape, which lane holds which element, whether the second operand needs transposing, and whether half precision is a choice.

What they share is worth writing down, because it is what makes them the same change twice. Both stage a tile of the activations and a tile of the dequantized weights into shared memory, both walk the reduction in steps of the staged depth, both hold the output in registers for the whole walk, and both end in the same epilogue as every other matmul in the file.

The tile is where the traffic is. A block covers `ROWS` output rows by `TOKENS` tokens and reads the whole reduction of its rows once, so wider in tokens is less weight traffic and wider in rows is more work for the same activation read. The token side cannot pass 64 on either backend, and not for a reason of speed: the tail block runs its dead lanes off the end of the chunk rather than branching, and the slack every prefill scratch vector is allocated with is exactly 64 rows. The row side has no such ceiling and was swept, and on a 4090 it wants 128 over eight warps, which is worth 1.20 times over 64 over four on the 8B.

Three things separate a tensor core kernel that is worth having from one that is not, and the CUDA one was rebuilt around all three after a first attempt reached five per cent of the card's peak. The fragments come from `ld_matrix`, which is one instruction a warp for a whole fragment where the obvious version issues twelve four byte loads for every multiply. The staged tiles are swizzled, so those loads take no bank conflicts: a staged row of 64 halves is 128 bytes, which is one full pass over the banks, so starting row `r` at chunk `r % 8` puts the eight rows a load phase reads on eight distinct chunks and all 32 banks once. And the tile is wide enough that the staging is amortized.

## Where the tile is not used

A block of the tile covers 8192 outputs on CUDA against the ordinary kernel's 256, which is thirty two times fewer blocks for the same matmul. That is the point of it on a large matrix and it is a loss on a small one, because a 4090 has 128 SMs and a matmul that does not have 128 blocks of work leaves whole SMs idle no matter how good the instruction inside them is.

Measured on gpc with the tile taken everywhere it fits, prefill tokens a second, molla against itself: Llama 3.1 8B at Q4_K_M goes 597 to 866, Qwen 2.5 0.5B goes 9345 to 8862 and SmolLM2 135M goes 18357 to 9885. Two of the three are regressions and the third is the only model with matrices large enough to fill the card.

So the dispatch asks for one block an SM before it takes the tile, which is `tokens * rows` against `128 * ROWS * TOKENS`. At a 256 token chunk the 8B is above that line on every matrix it has, the 0.5B is above it on the two wide feed forward matrices and below it on the rest, and the 135M is below it everywhere and never sees the tile. It is a property of the card rather than of the model, and it is the one number here that would want re measuring on a card with a different core count.

## What the tile agrees with, and how closely

The tile does not have to agree with the ordinary kernel exactly and cannot, so the gate is per backend and the difference is the hardware rather than the code.

Apple's `simdgroup_matrix` multiplies in float and accumulates in float, so the Metal tile is doing the same arithmetic as the kernel it replaces in a different order, and it holds to 2e-4 of the peak logit.

NVIDIA's tensor core takes half precision operands and there is no float shape to fall back to. The float shape on this hardware is tf32, which has the same ten bit mantissa at half the rate, so there is no accuracy to buy by staying wide. A half precision dot product agrees with a float one to a few times 1e-4 and no better, and that is the floor rather than a bug to find: the same tile at the same depth against a host reference lands there and stays there.

`tests/test_gpu.mojo` checks the tile directly against a host reference on a synthetic q8_0 weight wide enough to reach the dispatch, at 1e-5 where the backend multiplies in float and 2e-3 where it multiplies in half.

## What else has to be batched

The matmul is the interesting one and it is not the only one.

The embedding lookup becomes `T` rows in one launch. The norms become `T` independent reductions, one per block, which is what a block was already doing. Rope becomes a grid over `(T, heads)` with the position taken from the token index rather than passed in, and this is the one place where a batched kernel needs a per token scalar rather than a base and a stride, because the query and the key rotate against `pos0 + t`.

Attention is a grid over `(T, heads)`. The block for token `t` attends over `slot0 + t + 1` keys, which is where the causal mask comes from: there is no mask to apply because the count is the mask. Sliding window and sinks stay exactly as they are, since both are already expressed against `pos` and `pos` is now `pos0 + t`.

The keys and values are written straight into the cache by the projection, the way they already are in decode. A chunk of `T` tokens writes `T` contiguous slots, and because the cache is contiguous per slot the projection's output offset is `slot0 * kv_width` and its row stride is `kv_width`, which is what the matmul writes anyway.

Only the last token of the last chunk has logits worth anything, so the output head stays a matvec on one row. On a 49152 row head at a 64 token chunk that is sixty three sixty fourths of the largest single matmul in the pass, not done.

## What it has to agree with

The decode path and the prefill path have to produce the same logits for the same prompt, and that is a test rather than an argument, because the two do the same arithmetic in a different order and floating point addition is not associative.

The logit corpus does not cover this. The oracle traces every layer to compare against llama.cpp, tracing needs one token's snapshot at a time, and `device_forward` refuses to trace a chunk, so the corpus feeds prompts through the token at a time path and always will. It is still the thing that says the arithmetic is right, and it is not the thing that says prefill agrees with it.

What says that is `tests/test_gpu_block.mojo`, on a synthetic model small enough to run on every machine, against the decode path in the same process:

- A prompt through one chunk leaves the same logits as the same prompt decoded a token at a time, and greedy picks the same token off them.
- It leaves the same keys and values in the cache, checked against a cache that is not zeros.
- A chunk boundary is not special: the same prompt split across two chunks reaches the same logits.
- The chunk is not a whole chunk in any of the three, because a run that divides the block evenly is the case that hides the tail.

The model in that test carries the Gemma post norms on one layer and not on the other, so one pass covers both shapes. That path is the one that cannot ride a projection epilogue, and it is where a width derived by dividing the scratch by the run count goes wrong on a chunk that is not full.

Above the kernels there is one more thing to get right and it does not look like a kernel bug when it is wrong. The decode scratch and the chunk scratch each own a logits vector, a pass writes exactly one of them, and reading the other gives whatever the last decode left there. That is a silent wrong answer rather than an error, because the buffer is the right shape and full of real numbers from an earlier token. The session records which of the two the last pass wrote.

## What it is worth

A 514 token prompt and 64 decoded tokens on a 4090, three runs, molla before and after:

| model | prefill before | prefill after | ttft before | ttft after |
| --- | --- | --- | --- | --- |
| SmolLM2 135M Q8_0 | 358 tok/s | 9018 tok/s | 1434 ms | 57 ms |
| Qwen2.5 0.5B Q4_K_M | 398 tok/s | 4673 tok/s | 1291 ms | 110 ms |
| Llama 3.1 8B Q4_K_M | 100 tok/s | 369 tok/s | 5143 ms | 1397 ms |

Twenty five times, twelve times and under four times, and the same on time to first token because time to first token is what prefill is. Decode is unchanged, which is the point of a separate path. Greedy output is identical to the token at a time build on all three models on both backends.

It is also what makes the 8B testable at all. At the old rate a 512 token prompt against it was five seconds before the first token, which is why every 8B row in bench.md used a short prompt.

## What is left

llama.cpp on the same 4090 and the same prompt does 27000 to 44000 tokens a second on the two small models and about 10000 on the 8B. That was three times behind on SmolLM2, six times behind on Qwen and twenty seven times behind on the 8B when the ordinary kernel was all there was. Molla holds the memory side comfortably throughout, at 282 MiB against 444 on SmolLM2 and 1000 MiB against 4900 on the 8B.

The tile is what closed the 8B end of that. On gpc, best of three alternating rounds, a 512 token prompt, molla against itself:

| model | ordinary | tile |
| --- | --- | --- |
| Llama 3.1 8B Q4_K_M | 598.8 tok/s | 1086.5 tok/s |
| Qwen 2.5 0.5B Q4_K_M | 9017.5 tok/s | 11173.9 tok/s |
| SmolLM2 135M Q8_0 | 18357.1 tok/s | 18357.1 tok/s |

1.81 times on the 8B, 1.24 on Qwen, and parity on the 135M, which never reaches the tile and is the same kernel measured twice. Time to first token on the 8B goes from 860 ms to 474. Card memory is identical on all three, since the tile stages through shared memory and allocates nothing. Decode on the 8B is 111.1 against 111.6, which is unchanged as it has to be, since decode is a matvec and none of this is on that path. Decode on the two small models moves by less than the spread those models show between rounds of the same build.

The arithmetic rate says what is left. At 1086 tokens a second a chunk of 256 through the 8B is 235 ms for about 4.1 TFLOP of work and one pass over 4.6 GiB of weights, which is 17 TFLOP/s and 20 GB/s. The bandwidth is nowhere near the card and 17 is about a tenth of what a 4090 does in half precision with a float accumulator, so the kernel is still bound by neither the weights nor the peak of the instruction.

What it is bound by is that the staging and the multiplying take turns. A step stages a tile, waits on a barrier, multiplies it, waits again, and the two phases are close enough in cost that overlapping them is worth most of another factor. Doing that means a second set of shared buffers and a loop that stages step `n + 1` while multiplying step `n`, which is a change to this kernel rather than another kernel, and it is what is left open on the issue.
