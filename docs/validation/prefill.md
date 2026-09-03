# Prefill

molla has no prefill. It has a decode loop that a prompt is also fed through, one token at a time, which is why molla's prefill rate and its decode rate are within a few per cent of each other on every row of every table in [bench.md](bench.md). A 514 token prompt is 514 forward passes and about 232000 kernel launches, and 1933 ms of time to first token against llama.cpp's 18 ms. Between 109 and 175 times off, and it is the largest single gap molla has against anything.

This is what the shape of the fix is, what it costs in scratch, and what each piece has to agree with.

## The shape

A prompt is a matrix. Every matvec on the prefill path is the same weight against many activation vectors, so it is a matmul with the prompt length as the free dimension, and the launch count for a whole prompt becomes the launch count for one token.

That last sentence is only true if the prompt is done in one pass, and it should not be, for a reason that has nothing to do with launches. Attention over a chunk of `T` tokens needs somewhere to put `T` rows of scores, and the score row for a token at position `p` is `p + 1` long. Sized for the whole context that is `T * heads * context` floats, which on a 4096 context 32 head model is 2 MiB a token. A 512 token prompt in one pass would be a gigabyte of scratch to save 7 ms of launch time.

So prefill is chunked. `PREFILL_CHUNK` tokens go through the stack together, the next chunk follows, and a 512 token prompt is four passes rather than 512. The chunk is 64, which is 128 times fewer launches than today and bounds the scores at 33 MiB on an 8B and 4.7 MiB on SmolLM2 135M. Going to 128 would halve the launches again and double that, and the launches are not what is left at four passes.

## What the matmul has to do that the matvec does not

The obvious generalization is a block per output row per token, which is the existing kernel with a second grid dimension and no thinking. It is also the wrong one, because it reads the whole weight matrix once per token and prefill would be bandwidth bound at `T` times the weight size.

The block covers `span` tokens instead. It reads and dequantizes each weight value once and multiplies it into `span` accumulators against `span` activations, so the dequantization work and the weight read are both amortized by `span`. That is the whole point: decode is bandwidth bound at one value per weight byte and prefill should be compute bound, and `span` is the knob that moves it.

`span` is 8. The block's reduction is then `span` tree reductions rather than one, done in one pass over shared memory laid out `[span][tile]` so that a step of the tree is still a coalesced read, at a cost of `span * tile` floats of shared memory. At the shipped tile of 256 that is 8 KiB a block.

The grid is `(ceil(T / span), rows)` and not the other way round, and that also matters. Blocks that share an output row have to be co-resident for the L2 to serve the weight row once, and the blocks that share a row are the ones that differ in the token index, so the token index is the fast axis. With a chunk of 64 and a span of 8 that is 8 blocks per row, launched together.

The epilogue survives unchanged in meaning and changes in indexing. A bias reads `aux[r]`, the same element for every token in the block. A residual add reads and writes `o[t * rows + r]`. A gate reads `aux[t * rows + r]`. All three are still one thread's work at the end of a reduction.

## What else has to be batched

The matmul is the interesting one and it is not the only one.

The embedding lookup becomes `T` rows in one launch. The norms become `T` independent reductions, one per block, which is what a block was already doing. Rope becomes a grid over `(T, heads)` with the position taken from the token index rather than passed in, and this is the one place where a batched kernel needs a per token scalar rather than a base and a stride, because the query and the key rotate against `pos0 + t`.

Attention is a grid over `(T, heads)`. The block for token `t` attends over `slot0 + t + 1` keys, which is where the causal mask comes from: there is no mask to apply because the count is the mask. Sliding window and sinks stay exactly as they are, since both are already expressed against `pos` and `pos` is now `pos0 + t`.

The keys and values are written straight into the cache by the projection, the way they already are in decode. A chunk of `T` tokens writes `T` contiguous slots, and because the cache is contiguous per slot the projection's output offset is `slot0 * kv_width` and its row stride is `kv_width`, which is what the matmul writes anyway.

Only the last token of the last chunk has logits worth anything, so the output head stays a matvec on one row. On a 49152 row head at a 32 token chunk that is thirty one thirty seconds of the largest single matmul in the pass, not done.

## What it has to agree with

The decode path and the prefill path have to produce the same logits for the same prompt, and that is a test rather than an argument, because the two do the same arithmetic in a different order and floating point addition is not associative. The tolerance is the one the logit corpus already uses.

Three checks:

- A prompt through `prefill` and the same prompt through `step` one token at a time leave the same KV cache and the same logits, inside `ELEM_TOL`.
- The whole logit corpus still passes on both backends, since the corpus feeds prompts and the prefill path is now what feeds them.
- A chunk boundary is not special: a prompt of `PREFILL_CHUNK + 1` tokens agrees with the same prompt run token by token.

## What it is worth

Prefill is compute bound rather than bandwidth bound, so it has its own roofline and the goal should be checked against it before anything is promised. Two times llama.cpp on SmolLM2 prefill is 58040 tokens per second, which at roughly 0.27 GFLOP a token is 15.7 TFLOP/s against something like 165 TFLOP/s of fp16 dense on a 4090. Ten per cent of peak, so there is room. On the 8B the same arithmetic says llama.cpp's 10609 tokens per second is already about 170 TFLOP/s, at or above the card's published dense number, so there this closes the gap rather than beating it.

It is also what makes the 8B testable at all. At the current rate a 512 token prompt against it is seven seconds before the first token, which is why every 8B row in bench.md uses a short prompt.
