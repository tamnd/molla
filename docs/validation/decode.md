# The cache, and the first real tokens

Issue #27. [blocks.md](blocks.md) is one layer and the table that configures it. This is the thing that holds state between tokens, and the loop that fills it.

## What the cache actually is

Not a new layout. `molla.nn.attention` already reads keys with position on the outside and `molla.nn.model.forward` already takes one flat list per layer plus a slot to write into, so `KvCache` owns those lists and answers the three questions the forward pass deliberately does not: how long each one is, which slot a position goes in, and what happens when the context fills.

One sequence, contiguous, no paging. `slot_for` returns the position it was given and it is a method rather than an expression at the call site, because paging in M3 changes that answer and nothing else. The day it stops being the identity, every caller that had inlined it becomes a bug, and there are no such callers because there is a method.

Everything is float32. A cache in float16 halves the memory and is the known cause of long context degradation, the kind that shows as a model that is fine at two thousand tokens and vague at eight. That is a defect that only appears in the cases people bought the long context for, so it is not a default. Quantizing the cache is a real option later and it will be a measured one.

## Overflow is a refusal

A cache that fills and starts overwriting slot zero gives a model that is still fluent and has forgotten its instructions. That is worse than an error, because nobody reads it as one. So `reserve` refuses a sequence that will not fit, before any of it is computed, and `generate` reserves the prompt and the whole continuation up front rather than discovering the problem eighty tokens in.

What to drop when the context is genuinely full is a policy and it belongs with paging. Until there is one, the honest answer is that the sequence does not fit.

## Prefill and decode are one loop

A prefill runs the forward pass once per prompt token and throws away every logit but the last. A decode runs it once per generated token and keeps each one. The only difference is whether anybody looks at the answer, so there is one `step` and two callers rather than two orderings of the same six operations.

That matters because of the failure mode from #26. A prefill that rotates by the wrong position produces fluent output rather than a crash, and if it were a second code path it would be the one with no oracle behind it.

So the check this work exists for is an equality, not a tolerance. Prefilling a prompt and then decoding has to leave exactly the same bytes in the cache as feeding those same tokens one at a time, on every layer, for keys and for values, and the last token's logits have to agree bit for bit. Numbers that were merely close would let an off by one position through, since a rotation by one position out of five hundred is a small change to every element.

A prefill here is still one position at a time. Attention takes a single query and a run of keys by construction, so a batched prefill that multiplies a whole prompt at once is a different kernel rather than a different loop.

## Binding a file to the network

`molla.engine.bind` is the twenty lines between a GGUF and the tensors `molla.nn` wants, and it is its own module because what it does wrong is silent. A weight bound to the wrong address produces output. A weight that is present and bound as absent is a layer quietly skipping a norm.

So it binds by name against `tensor_names` in the architecture table, a name the table asks for and the file does not have is an error rather than a `none`, and every shape goes through `LayerWeights.check` before a token is computed. A load that takes half a minute and then fails on a shape at token one has spent that half minute for nothing, and the check is a few hundred integer comparisons.

Everything binds to the mapping. The kernels are host kernels, so a tensor copied to a card is one they cannot read, and `molla generate` plans the load with a device budget of zero rather than leaving the placement to a heuristic that has no way to know what will read the result. Device placement becomes useful when there are device kernels, which is M3.

## What was run

`molla generate <model.gguf> <tokenizer.json> "<prompt>" [n] [ctx]`. It is the first thing in molla whose output is judged by reading it rather than asserted, and it stays that way until #30 puts the logits beside llama.cpp's.

| Model | Quantization | Prompt | Output |
| --- | --- | --- | --- |
| SmolLM2 135M Instruct | q8_0 | The capital of France is | Paris. Paris is the largest city in France and the capital of the French department of ... |
| Llama 3.1 8B Instruct | q4_K_M | The capital of France is | a city of grandeur and beauty, with a rich history |

The 8B is the interesting one, because it is the path that reads `rope_freqs.weight` off the file. Llama 3.1's position scaling arrives precomputed as one factor per rope pair rather than as a scheme to apply, and a model that ignores it is coherent at short context and falls apart at long, which is not something a twelve token sample would have shown either way. What the sample does show is that the factors were read and applied without breaking the short case.

Speed is not the point yet and it is worth writing down anyway. The 135M decodes at about 90 ms a token and the 8B at about 5.5 seconds a token, single threaded and scalar, on an M4. That is the number #120 exists to change.

## What is checked

Forty one checks. The cache's shape and the size it reports, the room arithmetic, and that a refused reservation leaves the cache exactly where it was rather than half advanced. That a step writes slot zero and leaves slot one alone, and that a second step leaves the first token's key untouched while writing its own.

The equality between prefill and decode, across two layers, keys and values, bit for bit.

That greedy decoding gives the same answer twice, which is the cheapest statement that nothing in the loop depends on uninitialised memory. A stop token ends the loop without being emitted. A prompt longer than the context is refused with the session still at position zero, so the error is recoverable rather than leaving a half filled cache behind.

## What this is not

There is no sampling. `pick` is the argmax, which is a sampler with the temperature at zero, and it is here rather than in a sampler module because a decode loop with nothing choosing the next token cannot be run at all. Temperature, top-k, top-p, min-p, penalties and seeds are #28.

There is no chat template applied and no server. `molla generate` takes raw text and prints raw text, so a prompt that should have been wrapped in an instruction template is not. That is #29.

There is no comparison against llama.cpp. Two models producing sensible English is evidence and it is not a measurement, and the difference matters because most of the ways to be slightly wrong here still produce sensible English. #30 is the measurement.

The vocabulary comes from a `tokenizer.json` and not from the GGUF. Reading the vocabulary and merges out of the file's metadata is real work with its own failure modes, and doing it as a side errand of the decode loop is how a tokenizer ends up with no oracle behind it.

One sequence at a time, no batching, no paging, and no attention to how many of them a server would want. That is M3.
