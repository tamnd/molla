# Two times the throughput at half the memory

Issue #172, the M2c tracking issue. The goal is that molla beats the fastest thing anybody would compare it against by a factor of two on tokens per second, while using half the memory. [bench.md](bench.md) says molla is currently 2.4 to 3.3 times slower than llama.cpp on decode, more than a hundred times slower on prefill, and uses 2.2 to 3.6 times the host memory. So the ask is a factor of five to seven on decode and a factor of four to seven on memory, and that is not a tuning exercise. This page is the research that says which parts of the goal are reachable, which part is not reachable on any architecture, and what has to change.

Everything measured here was measured on gpc, an RTX 4090 reached through WSL2, with a load average under 0.6. Every number in this page can be reproduced by the programs described beside it.

## The short version

One correction before the rest of this page, because it changes what the numbers below mean. This page reads the whole decode gap as launch cost. That is right on a small model and wrong on a large one: on the 8B, molla reads 2.03 times llama.cpp's bytes per token because the planar layout stores a byte per weight, and the decode gap is 2.11 times, so the layout is the whole of it and launches are 17 per cent of the token. [layout.md](layout.md) has the measurement and the fix. Read the launch argument below as applying to models where the repack is close to the file size, which is q8_0 and not q4_K.

That correction is itself now historical. The layout work landed and the 8B repack is 5151 MiB against a 4693 MiB file, so the bytes are within a tenth of llama.cpp's rather than double. [budget.md](budget.md) splits a current 8B token three ways by measuring each part on the kernel that runs it, and the answer is not the one this page or that correction expected: the projections are 52 per cent of the token and are within a fifth of the memory rate, and 38 per cent is attention over the key and value cache running at a twelfth of it on a grid of 32 blocks. Read what follows for the launch argument, which still holds, and read [budget.md](budget.md) for where a large model's token actually goes.

Decode is not slow because the kernels are slow. It is slow because there are 453 of them per token and each one costs 4.9 microseconds to launch whether it does anything or not. Prefill is slow because it does not exist as a separate path: a 514 token prompt is 514 decodes. Host memory is high for two reasons that are unrelated to each other, a flat 1.3 GiB that the first `map_to_host` call reserves, and a second full copy of the weight file that comes from mapping the GGUF and the repack cache at the same time.

Two times better than llama.cpp on single stream decode is reachable on small and mid sized models, where llama.cpp itself is using ten to twenty per cent of the card. It is not reachable on an 8B, because there llama.cpp is already at 79 per cent of the card's memory bandwidth and two times its rate is above what the hardware can do. That is stated plainly rather than promised and missed, and the honest route to two times the tokens per second out of a machine on a large model is concurrency rather than single stream speed.

Half the memory is reachable everywhere, and by a lot more than half, because the thing that dominates llama.cpp's host memory is a mapping of the weight file that molla does not have to keep.

## Why decode is slow

The forward pass launches one kernel per operation. `device_attention` is nine launches and `device_mlp` is six, so a layer is fifteen, and SmolLM2 has thirty layers, which with the embedding, the final norm and the output head is 453 launches for one token.

The cost of a launch was measured directly, with a kernel whose whole body is one conditional store, launched a hundred thousand times in a row on one stream.

| launches | issue cost each | wall cost each |
| --- | --- | --- |
| 1000 | 4.40 us | 5.57 us |
| 10000 | 5.86 us | 6.14 us |
| 100000 | 4.90 us | 4.93 us |

So 4.9 microseconds is what an empty kernel costs, and the issue side alone is most of it, meaning the host cannot even hand the card work faster than about 200000 launches per second. That is the number the whole architecture question turns on.

Put the two together. 453 launches at 4.9 us is 2.22 ms of pure launch cost per token, and a token currently takes 3.93 ms. So a bit over half of decode is the cost of asking, not the cost of doing. More usefully, it is a ceiling: with 453 launches per token, molla cannot exceed 450 tokens per second on this card even if every kernel became instantaneous. llama.cpp is at 782.5. The current architecture cannot reach the M7 gate, never mind the goal, and no amount of kernel work changes that.

One measurement agrees with this reading and one was misread. The one that agrees is the roofline. 138 MiB of weights at the 4090's 1008 GB/s is 0.144 ms per token, so the hardware ceiling for SmolLM2 decode is about 7000 tokens per second. molla is at 254.5 and llama.cpp is at 782.5, which is to say both are far from the card's limit and llama.cpp is only better at the same thing molla is bad at.

The one that was misread is the 8B. It has the smallest gap of the three at 2.4 times against 3.1 and 3.3, and this page took that as a fixed per launch cost being a smaller share of a launch that does more work. It is not that. The 8B repacks from 4693 MiB to 9573 MiB, so it reads twice the bytes llama.cpp reads, and 2.03 times the bytes against 2.11 times the time is the entire gap with nothing left over for launches. See [layout.md](layout.md).

## What the launch budget has to be

Two times llama.cpp on SmolLM2 is 1565 tokens per second, which is 0.639 ms per token. Bandwidth takes 0.144 ms of that. If launches are allowed a fifth of the remainder, the budget is 0.099 ms, which at 4.9 us is twenty launches per token.

Twenty, for a thirty layer model. That is the whole design constraint and it has one consequence: the number of launches per token has to stop being proportional to the number of layers. Fusing operations within a layer helps and is not enough. Fifteen launches per layer fused down to four is still 120 per token, which is 0.59 ms of launch cost and misses the target on its own.

So the per token launch count has to become a small constant. There are two ways to get there and they are not equally attractive.

Capture the whole token as one replayable unit, which is what a CUDA graph is for, and what Metal indirect command buffers are for. This is the standard answer and it would work, but it is two vendor specific mechanisms rather than one, and the Mojo `DeviceContext` in max-core 26.5 does not appear to expose either. That has to be verified rather than assumed before anything is built on it.

Or write the token as one kernel that keeps the whole model resident in the grid and walks the layers internally, synchronising across the grid between stages. One launch per token, portable by construction, and it fits D7's rule that there is one kernel source for four targets rather than a vendor path per vendor. It needs a grid wide barrier, which is a real primitive on both vendors but is not the block level `barrier` the kernels use today, so its availability is the other thing to verify first.

The recommendation is the second, with the first as the fallback if grid wide sync turns out to be unavailable, and with fusion inside the layer done first either way because it is useful under both and it is the part that is certain to work. The order matters: fusion is a two times improvement that can be measured next week, and the launch collapse is the part that reaches the goal.

## What fusion actually reached

The section above budgets fifteen launches a layer down to four and calls that not enough. Four was a guess. This is what the elementwise chain is actually worth once it is folded in, and it is three launches a layer rather than eleven.

The unit that can absorb elementwise work is the matvec, because it already owns one output row per thread block and it already has that row in a register when it finishes. Anything that reads only that row, and nothing else that the same launch is still writing, can be done by one thread of that block for free. That is a real constraint and it decides the list.

Three things fit it.

The projection bias is `o[r] = dot + bias[r]`, which reads one element of one plane at the row the block already owns. Three of them a layer on a model with qkv bias, and they were three whole launches doing one flop per element.

The residual add is `x[r] = x[r] + dot`, same shape, and it is better than the other two because it also deletes the scratch vector the projection was landing in and the round trip through device memory that went with it. Two a layer.

The gate half of a gated MLP is `o[r] = act(dot) * up[r]`, and it fits because `up` was written by the projection before it, so by the time the gate projection runs its epilogue the element it needs is final. One a layer.

Four things do not fit it, and it is worth saying why, because they are most of what is left.

A norm cannot go in a matvec epilogue in either direction. As a prologue it is a reduction over the whole input vector, so every block would recompute it, and there are three blocks per input on the qkv side. As an epilogue it is a reduction over the whole output vector, which the block that owns one row of it cannot see. That is two launches a layer that stay.

Rope cannot, because it rotates element `i` against element `i + head_dim / 2`, and those are two different blocks of the same launch. Two a layer that stay, and they could become one launch by taking the query and the key together, which is a launch merge rather than a fusion.

The post projection norm that Gemma has sits between the projection and the residual add, so on that architecture neither the add nor the scratch vector can go. Both paths are still there and it is the only reason `s.projected` still exists.

Attention itself is not elementwise.

So the count, before and after. These are counted rather than derived: nsys over five forward passes on gpc, every kernel instance summed and divided by five.

| model | layers | a layer before | a layer after | a token before | a token after |
| --- | --- | --- | --- | --- | --- |
| SmolLM2 135M q8_0 | 30 | 15 | 12 | 453 | 363 |
| Qwen 2.5 0.5B q4_K_M | 24 | 18 | 12 | 435 | 291 |

Twenty per cent off a model without bias and a third off one with it. The three launches over the layer total in both rows are the embedding, the final norm and the output head, and they do not move.

What that is worth in tokens per second, on the same card, decode only, eight repetitions of the three models interleaved so that a drift in the machine moves both sides of a row together.

| model | before | after | ratio |
| --- | --- | --- | --- |
| SmolLM2 135M q8_0 | 282.2 | 310.8 | 1.11 |
| Qwen 2.5 0.5B q4_K_M | 244.9 | 309.2 | 1.26 |
| Llama 3.1 8B q4_K_M | 99.8 | 101.4 | 1.02 |

The two small models are the ones the launch budget said were launch bound, and they are the two that moved. The 8B did not, which is the same answer from the other direction: at 5.6 GiB a token it is bandwidth bound, the launches are hidden behind the weight reads, and deleting a launch that was overlapped anyway buys nothing.

The two small rows do not reach what the budget above predicts. Ninety launches off SmolLM2 at 4.9 microseconds each is 0.44 ms off a 3.5 ms token, which would be 1.14, and the measurement is 1.11. On Qwen the same arithmetic gives 1.21 against a measured 1.26. Close enough on both to say the budget is roughly the right model of where the time goes, and not close enough to trust the fourth digit of it.

One thing was tried and rejected on the way, and it is worth recording because it looked obvious. The gate epilogue contains an `exp`, and register allocation is per thread and uniform across a kernel, so a runtime branch on the activation allocates that `exp`'s registers in every thread of every block of every matvec including the output head that never runs it. Making the activation a compile time parameter instead, three instantiations where there was one, measured as a wash: it was 2.4 per cent slower than the runtime branch on SmolLM2 and 2 per cent faster on Qwen, both inside the run to run spread on this machine. So the parameter is not there and the branch is.

It is exact and was checked as exact rather than argued to be. Every one of the three is the same arithmetic on the same values in the same order as the kernel it replaces, so the whole logit corpus is unchanged in every digit on both backends, and the layer by layer device against host comparison in `tests/test_gpu_block.mojo` holds at the tolerance it already had.

The epilogue is a runtime argument to the matvec and not a compile time parameter. That is deliberate. One thread of a block runs it, once, after everything else the block does, so making it a parameter would multiply five instantiations of the hottest kernel in the program by four to save one correctly predicted branch per block.

What this leaves is the two norms, the two ropes, attention and six matvecs. Of those, the qkv triple reads one input and the up and gate pair reads one input, so five of the six matvecs are two launches worth of work in a launch geometry that has not been written yet, and that is a launch merge rather than a fusion. It belongs with the launch collapse below, and it would take a layer from twelve to nine. The section above is still right that none of this reaches the target: nine launches a layer on a thirty layer model is 270 a token, and the budget is twenty.

## Why prefill is slow

molla has no prefill. It has a decode loop that the prompt is also fed through one token at a time, which is why molla's prefill rate and its decode rate are within a few per cent of each other on every row of every table in bench.md. A 514 token prompt is 514 forward passes, so it is 232000 kernel launches, and 1933 ms of time to first token against llama.cpp's 18 ms.

This is the largest gap on the page and it is also the most ordinary to close. A prompt is a matrix rather than a vector: the same weights, many rows. Every matvec in the forward pass becomes a matmul with the prompt length as one dimension, attention becomes a masked score matrix instead of a loop over positions, and the launch count for the whole prompt becomes the launch count for one token. That is where the hundred times comes from, and it is the same 453 launches doing 514 tokens of work instead of one.

Prefill is compute bound rather than bandwidth bound, so it has its own roofline and it is worth checking the goal against it before promising anything. Two times llama.cpp on SmolLM2 prefill is 58040 tokens per second, which at roughly 0.27 GFLOP per token is 15.7 TFLOP/s, against something like 165 TFLOP/s of fp16 dense on this card. Ten per cent of peak, so there is room. On the 8B the same arithmetic gives a different answer: llama.cpp's 10609 tokens per second is already about 170 TFLOP/s, which is at or slightly above the 4090's dense fp16 number. Either it is at the roofline or the card does better than its published figure on this shape, and either way there is no factor of two sitting there.

## Why memory is high

Two costs, measured separately, neither of them the one that was assumed.

The flat one is 1.3 GiB and it belongs to `map_to_host`. Two standalone programs, each creating a device context and then moving four kilobytes into three hundred small device buffers, one of them through `map_to_host` and the other through `enqueue_copy` from a host pointer.

| what the program does | through `map_to_host` | through `enqueue_copy` |
| --- | --- | --- |
| process start | 367 MiB | 363 MiB |
| after `DeviceContext()` | 454 MiB | 452 MiB |
| after the first buffer | 1767 MiB | 452 MiB |
| after 300 buffers | 1767 MiB | 452 MiB |

The first mapping costs 1313 MiB and the next 299 cost nothing. The copy path never pays it at all.

So it is a one time arena that the first mapping reserves, it does not scale with anything, and it is avoidable. Staged RSS probes put molla's own 1.3 GiB jump inside the layer construction loop of `DeviceModel.__init__`, which is where `_gain` uploads each norm weight through `upload_run`, which is `map_to_host`. The weight loader does not pay it, because `Pool.copy_in` already uses `enqueue_copy`. Two call sites are on the wrong side of this, `_gain` on the way in and the logit `download` on the way out, and both have a working example of the right way to do it in the same repository.

There is no arena on Metal. The same two programs on the M4 grow by 17.3 MiB through `map_to_host` and 31.9 MiB through `enqueue_copy`, so the mapping path is the cheaper of the two there and neither is within two orders of magnitude of 1313 MiB. That is the answer a unified memory machine should give, since a mapping is the memory rather than a window onto it and a copy has to find somewhere to copy to. The change that moved the CUDA numbers is therefore free on Metal rather than helpful, which is fine, but it means none of the 1.3 GiB was ever part of a Metal peak and any Metal memory that looks unexplained has to be explained by something else. It was: see [layout.md](layout.md), where the 8B's Metal footprint is the weights plus the KV cache plus about 130 MiB, adding up exactly.

The scaling one is about twice the model file size, from `1312 MiB + 2.14x file size` fitting the measurements across four models. Both the GGUF mapping and the planar repack cache mapping are open and touched during the upload, so the host holds the weights twice, in a form the card already has a third copy of. Neither is needed once the upload has finished.

Then there is what llama.cpp's number actually is, which changes what the target means. llama.cpp's 5034 MiB on the 8B is a host figure for a run whose weights are on the card, and it is that large because the weight file is mapped and the mapping is resident. Half of it is 2517 MiB. A molla that streams weights to the device in bounded chunks and never holds a whole file mapping would be at a couple of hundred megabytes on any model, large or small, because nothing on the host would scale with the model at all. That is not two times better, it is ten to twenty times better, and it is a genuine architectural difference rather than a saving.

## What this means for the goal

Stating it by piece, because one number for all of it would be wrong in both directions.

Decode, small and mid models: two times llama.cpp is reachable. Both engines are using a small fraction of the card, molla's ceiling is currently set by its own launch count, and removing that is the work.

Decode, large models: two times is not reachable and should not be claimed. llama.cpp at 161.2 tokens per second on a 4.58 GiB file is moving 793 GB/s against a card rated at 1008, so it is at 79 per cent of the bandwidth limit and two times its rate would need 1586 GB/s, which this card does not have. What is reachable is matching it and then beating it modestly, and then two times the aggregate throughput of the machine through continuous batching, where a second sequence costs almost nothing extra because the weights are read once for all of them. That is a real path to two times the tokens per second out of a box and it is the only honest one on a large model.

Prefill: more than two times is reachable on small and mid models and is bounded by compute on large ones. The first version of a batched prefill will be a hundred times better than what exists, so the interesting question is not whether it beats llama.cpp but by how much.

Memory: half is reachable, and considerably better than half, and it is the least uncertain part of the whole goal. 1.3 GiB is a wrong call site, one model's worth is a mapping held too long, and the remainder is a design where nothing on the host scales with the model.

## The order to do it in

Ranked by how much each one moves the numbers per unit of risk, not by how interesting it is.

Fix the memory first, because it is two call sites and a lifetime, the payoff is a factor of four to seven, and nothing else depends on it. `_gain` and the logit download move to `enqueue_copy`, which is issue #165, and the GGUF and repack cache mappings close once the upload is done, which is #166.

Batched prefill second, #167. It is the largest single gap on the page, it is a well understood change rather than a bet, and it makes the 8B testable at all: at the current rate a 512 token prompt against it is seven seconds before the first token, which is why bench.md could not carry that row until now.

Fusion within a layer third, #168. Fifteen launches per layer down to four or five is roughly a two times decode improvement, it is incremental and each step is measurable, and it is worth having whichever way the launch collapse goes.

The launch collapse fourth, #170, after #169 has established which primitive exists. This is the one that reaches the goal on decode and it is also the one with a real chance of not working the way it is planned, so it goes after the three things that pay off regardless.

Metal gets the same measurements taken against it in #171, because everything above is a CUDA number.

Continuous batching last, and it belongs to M3 rather than here. It is the only route to two times on a large model, and it needs the single sequence path to be settled before more than one of them can share it.

## What is still not known

All three of these were answered after this page was written, and the answers are in [max.md](max.md). Two of them changed the plan, so read that page before acting on the order above.

Whether max-core 26.5 exposes graph capture from Mojo, and whether it exposes a grid wide barrier. Both were looked for and not found, but the packages ship compiled and the absence of a string in a binary is not the absence of an API. These are the first two things to establish, because #170 is shaped differently depending on the answers.

Whether the launch cost is the same on Metal. The 4.9 microseconds is a CUDA through WSL2 number and Metal is a different driver with a different submission model, so the same measurement has to be taken on the M4 before any of this is assumed to apply there. The macbook is a loaded shared machine, which is a problem for measuring it and a reason to want a quiet Apple box in the fleet.

Whether the 1.3 GiB arena is resident because it is touched or resident because the driver pins it. It shows up in `wait4` either way, which is what makes it a real cost rather than an accounting artifact, but the two have different implications for whether any mapping at all is affordable.
