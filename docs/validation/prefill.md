# Prefill

molla had no prefill. It had a decode loop that a prompt was also fed through, one token at a time, which is why molla's prefill rate and its decode rate were within a few per cent of each other on every row of every table in [bench.md](bench.md). A 514 token prompt was 514 forward passes and about 232000 kernel launches, and 1400 ms of time to first token on a 4090 against llama.cpp's 17 ms. Between 109 and 175 times off, and it was the largest single gap molla had against anything.

This is what was built, what it costs in scratch, and what each piece has to agree with.

## The shape

A prompt is a matrix. Every matvec on the prefill path is the same weight against many activation vectors, so it is a matmul with the prompt length as the free dimension, and the launch count for a whole prompt becomes the launch count for one token.

That last sentence is only true if the prompt is done in one pass, and it should not be, for a reason that has nothing to do with launches. Attention over a chunk of `T` tokens needs somewhere to put `T` rows of scores, and the score row for a token at position `p` is `p + 1` long. Sized for the whole context that is `T * heads * context` floats, which on a 4096 context 32 head model is 2 MiB a token. A 512 token prompt in one pass would be a gigabyte of scratch to save 7 ms of launch time.

So prefill is chunked. `PREFILL_CHUNK` tokens go through the stack together, the next chunk follows, and a 512 token prompt is eight passes rather than 512. The chunk is 64, which is 128 times fewer launches than a token at a time and bounds the scores at 33 MiB on an 8B and 4.7 MiB on SmolLM2 135M.

## What the matmul has to do that the matvec does not

The obvious generalization is a block per output row per token, which is the existing kernel with a second grid dimension and no thinking. It is also the wrong one, because it reads the whole weight matrix once per token and prefill would be bandwidth bound at `T` times the weight size.

A block covers `SPAN` tokens instead. It reads and dequantizes each weight value once and multiplies it into `SPAN` accumulators against `SPAN` activations, so the dequantization work and the weight read are both amortized by `SPAN`. That is the whole point: decode is bandwidth bound at one value per weight byte and prefill should be compute bound, and `SPAN` is the knob that moves it.

`SPAN` is where the registers run out, so it is not the whole answer. Eight accumulators on CUDA and sixteen on Metal is as far as it goes before the accumulators spill, and a spilled accumulator undoes the change, so the rest of the amortization has to come from somewhere that is not registers. `MM_GROUPS` is that somewhere. A block is `MM_GROUPS` groups of `MM_TILE` threads, each group carrying its own `SPAN` tokens and all of them walking the same weight row, so the row is fetched from memory once for the block and out of the L1 for the groups behind the first. The groups share the weight and not the registers, which is why this goes past what `SPAN` can.

The two knobs multiply, and the product is the number that matters: a block covers `SPAN * MM_GROUPS` tokens, and the weight matrix is read `ceil(T / (SPAN * MM_GROUPS))` times for a chunk of `T`. Both backends want that product to be the chunk exactly, which is what `MM_GROUPS` is defined as. On a 4090 the 8B runs at 388 tokens a second with eight groups of eight and 262 with four of eight, and Metal falls off a cliff the other way, at a product of twice the chunk, where half of every block is dead. At the product the grid is one block to an output row, and the weight matrix is read once a chunk.

`MM_TILE` is 32 on both backends, which is a warp on both. It is measured rather than assumed: 32 is the best of every width from sixteen to five hundred and twelve on both backends and on all three models, and the losses either side are large, since a matmul block reduces `SPAN` accumulators rather than one and past a certain width the reduction is more work than the dot product that fed it. Being a warp is what the reduction below then relies on.

The grid is `(ceil(T / (SPAN * MM_GROUPS)), rows)` and not the other way round, and that also matters. Blocks that share an output row have to be co-resident for the L2 to serve the weight row once, and the blocks that share a row are the ones that differ in the token index, so the token index is the fast axis.

The epilogue survives unchanged in meaning and changes in indexing. A bias reads `aux[r]`, the same element for every token in the block. A residual add reads and writes `o[t * rows + r]`. A gate reads `aux[t * rows + r]`. All three are still one thread's work at the end of a reduction.

## Three things the kernel cannot afford

The block reduces `SPAN` accumulators where the matvec reduced one, and everything that costs a register per token is now `SPAN` registers held for the length of the accumulation. Three of them were worth roughly a factor of five between them on CUDA, and all three are invisible in the shape of the code.

The tail lanes do not clamp. A group whose tokens run past the end of a short chunk reads past the last token rather than pinning onto it, because clamping needs a live token index per lane held in a register for the whole accumulation and reading slack is an affine offset the address unit folds in for free. What it computes is thrown away by the `t < live` at the end. The price is that every scratch vector a chunk uses is allocated rounded up to a whole block of tokens, which at the shipped chunk is exact and costs nothing.

The reduction is a warp butterfly and not a tree through shared memory. A group is a warp, so `lane_group_sum` reduces it in five shuffles and leaves the total in every lane, and lane `k` keeps the total for the token it is about to write. That removes the shared memory and all five barriers. It is worth 41 per cent on the 8B and nothing on the models that fit in cache, which is the shape of a change that buys occupancy.

The reduction loop has to be unrolled. This is the one that does not look like anything. A plain `for k in range(SPAN)` around the reduction indexes the accumulator array with a value the compiler will not treat as constant, and the whole array lands in local memory for the entire kernel, accumulation included. Writing it `comptime for` costs nothing and is worth 55 per cent on SmolLM2 and 64 per cent on Qwen. The loops inside the accumulation are the same shape for the same reason.

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
|---|---|---|---|---|
| SmolLM2 135M Q8_0 | 358 tok/s | 9018 tok/s | 1434 ms | 57 ms |
| Qwen2.5 0.5B Q4_K_M | 398 tok/s | 4673 tok/s | 1291 ms | 110 ms |
| Llama 3.1 8B Q4_K_M | 100 tok/s | 369 tok/s | 5143 ms | 1397 ms |

Twenty five times, twelve times and under four times, and the same on time to first token because time to first token is what prefill is. Decode is unchanged, which is the point of a separate path. Greedy output is identical to the token at a time build on all three models on both backends.

It is also what makes the 8B testable at all. At the old rate a 512 token prompt against it was five seconds before the first token, which is why every 8B row in bench.md used a short prompt.

## What is left

llama.cpp on the same 4090 and the same prompt does 27000 to 44000 tokens a second on the two small models and about 10000 on the 8B, so molla is three times behind on SmolLM2, six times behind on Qwen and twenty seven times behind on the 8B. Molla holds the memory side comfortably, at 282 MiB against 444 on SmolLM2 and 1000 MiB against 4900 on the 8B.

The 8B is where the remaining work is, and the number that says why is the arithmetic rate. At 369 tokens a second a chunk of 64 is 173 ms for about a teraflop of work and one pass over 4.6 GiB of weights, which is 6 TFLOP/s and 27 GB/s. Neither is close to a 4090, so the kernel is bound by neither the weights nor the multiplies, and what is left is the dequantization work and the latency the accumulation cannot hide. Closing that means staging weights through shared memory and feeding wider multiplies, which is a different kernel and its own issue.
