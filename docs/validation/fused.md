# One launch a token

molla launches a kernel for every step of every layer. A Llama shaped layer is twelve of them after #168 and #194 folded the biases, the residual adds and the gate into the projections that were already running, so a thirty layer model is 363 launches for one decoded token. [max.md](max.md) measures a launch at 4.82 microseconds on a 4090 through WSL2 and 20.10 on an M4, so those 363 launches are 1.75 ms and 7.30 ms of nothing but submission, which is a ceiling of 571 tokens a second on the 4090 and 137 on the M4 before a single multiply happens.

That ceiling is the binding constraint on the small models this milestone is measured on, and no arrangement of kernels per layer clears it. The budget table in max.md walks it: five kernels a layer, which is about as far as fusing within a layer goes, still spends 115 per cent of the CUDA budget and 203 per cent of the Metal one. The launch count has to stop being proportional to the layer count.

This is the design for that, what it rests on, and the order it gets built in.

## The shape

One kernel launch a token. The grid is chosen once, every block stays resident for the whole token, and the layers are walked inside the kernel with a grid wide barrier between the steps that depend on each other.

That is only possible because a barrier across the grid is possible, which was not known until #170's probe measured one. The findings that shape everything below are in the grid wide sync section of [max.md](max.md), and three of them are constraints rather than facts:

A barrier round is cheap on Metal, 2.5 microseconds against a 20.10 microsecond launch, and it is not cheap on CUDA at full residency, 4.59 microseconds at 1536 blocks against a 4.82 microsecond launch. It gets to 1.26 at 384 blocks and 0.75 at 96 and is flat below that, because the cost at the top is 1536 blocks contending on one counter word rather than the rendezvous itself. So the CUDA grid has to be a few hundred blocks and each block has to carry many output rows.

Ordinary global loads and stores are not coherent across a block on Metal. A value written by one block and read by another after a barrier is stale 634 times out of 640, and the barrier does not help, because the problem is that the value never left the core's own cache. Both sides have to be a relaxed device scope atomic. The probe read this as the store side alone being enough and that was wrong, for a reason the section below sets out.

There is no occupancy query on Metal, so the grid there is chosen by hand rather than asked for. `MULTIPROCESSOR_COUNT` works and gives a floor of one block a multiprocessor.

## What a step is

A layer is a sequence of steps and a step is the unit between two barriers. Everything inside a step is independent across blocks, everything across a step boundary is not.

A Llama shaped layer today, as twelve launches, with what depends on what:

| step | reads | writes |
| --- | --- | --- |
| attention norm | the residual stream | the norm scratch |
| the query projection | the norm scratch | the query scratch |
| the key projection | the norm scratch | the key cache at this slot |
| the value projection | the norm scratch | the value cache at this slot |
| rope on the query | the query scratch | the query scratch |
| rope on the key | the key cache at this slot | the key cache at this slot |
| attention | the query, the whole key and value cache | the heads output |
| the output projection with a residual add | the heads output | the residual stream |
| the mlp norm | the residual stream | the norm scratch |
| the up projection | the norm scratch | the up scratch |
| the gate projection with the gated activation | the norm scratch and the up scratch | the gate scratch |
| the down projection with a residual add | the gate scratch | the residual stream |

Six of those twelve boundaries are not real. The three attention projections read the same input and write three disjoint outputs. The two ropes touch disjoint memory. The up and gate projections read the same input, and the gate reading the up is the only ordering among the four.

They do not have to be merged into one step to lose the barrier. Every record in the plan carries a flag saying whether a barrier follows it, and a record with the flag clear runs and falls straight into the next one. So the three projections stay three records with a barrier after the third, the two ropes stay two records with a barrier after the second, and the up and the gate stay two records with a barrier after the gate. A record is still the unit of work and the barrier is a property of the boundary rather than of the record, which is both simpler than merging and more general, since a merged step would have needed a record shape that holds three weight matrices.

The gate reading the up survives that only because the two records walk the same rows in the same order, so the block that reads `up[r]` is the block that wrote it a moment earlier, in its own program order and its own cache. That is a real constraint on the partition and not a coincidence to lean on quietly: two records with no barrier between them may only share data through rows that the same block owns in both.

That leaves eight barriers a layer:

```text
norm, qkv, rope, attend, out, norm, up and gate, down
```

Plus the embedding lookup, the final norm and the output head, so a thirty layer token is 243 rounds. At 0.75 microseconds on a 4090 at 96 blocks that is 0.182 ms against a 0.639 ms budget, 29 per cent, and at 2.5 microseconds on an M4 it is 0.608 ms against 1.514 ms, 40 per cent. Both fit. Neither fits comfortably enough to spend barriers carelessly, which is why the six that collapse are collapsed.

## The plan, and why it is data rather than code

The kernel cannot call the host between layers, so everything the host currently passes as an argument has to be in device memory before the launch. That is one array of step records, built once when the model is loaded, walked by the kernel.

A record is an opcode, a barrier flag, a few pointers and the handful of integers the step needs, in a fixed width slot so that walking the table is an index rather than a parse. The pointers are what the host already computes: `w.device_address()` for a weight matrix, a scratch vector's base, a cache layer's base. The integers are the rows, the columns, the row stride in bytes, the epilogue flags, and the quant descriptor.

What is not in a record is anything that changes between tokens. The position, the cache slot and the token id are the same for every record in a pass, so they are kernel arguments and the table is built once when the model is loaded and never touched again. That is what keeps this from being a host side rebuild and an upload per token, which would have put back a smaller version of the cost this removes.

Two things make this an interpreter rather than a switch. The quant form, the group size and whether there is a minimum plane are comptime parameters on the matvec today, and five combinations are compiled. In the fused kernel they arrive as a runtime triple and the step branches over the same five, which is one uniform branch a step and five copies of the dot product loop in the binary. That is the price of one kernel source for every model, and it is the right price, because the alternative is a kernel instantiated per model shape and a compile on every load.

The other is that the step table is the same table on every backend and for every architecture. Gemma's post norms are two more steps in the list, Qwen's biases are an epilogue flag on a step that already exists, and neither is a branch in the kernel. The host code that today decides which kernels to enqueue becomes the host code that decides which records to write, and it is the same decisions in the same order.

## Sizing the grid

The grid is fixed for the whole token, so it has to be large enough to hold the widest step and small enough to synchronise cheaply, and the two pull in opposite directions.

On CUDA the occupancy query gives the resident bound, 1536 blocks of 128 threads on a 4090. Launching the bound is what guarantees no deadlock, and launching fewer than the bound is equally safe and much cheaper to synchronise. So the grid is the smaller of the occupancy bound and a tuning constant, and the sweep says the constant should be in the low hundreds.

On Metal there is no query, so the grid is the multiprocessor count times a constant found by experiment, floored at the multiprocessor count, which is always resident.

What this costs is arithmetic width. A 4096 row projection on 96 blocks is each block reducing 43 rows in a strided loop rather than one row a block. For a small model that is a gain, because the weights come out of the L2 for the blocks behind the first. For an 8B it is a risk, because a bandwidth bound matvec wants more threads in flight than a few hundred blocks provide, and that is the number this design is most likely to be wrong about. It is measured in stage one below, before anything is built on top of it.

## What crosses a block

On CUDA nothing special. On Metal a value that crosses a barrier has to be written with a relaxed device scope atomic and read with one, so a `Float32` store becomes a `bitcast` to `Int32` and an atomic store, and a load becomes an atomic load and a `bitcast` back.

The probe read its own table as saying the store side was enough on its own, and stage one does not reproduce that. With an atomic store and an ordinary load the fused layer disagrees with the unfused one from the first layer of the first token, deterministically, in every element of the residual stream, and a barrier after every single record rather than only the eight that need one does not change the answer by a bit. On one block it is exact. That combination is not a missing barrier, and the probe's own four way table has the reason in it: the two configurations that were correct on both devices were atomic store with atomic load and atomic store with plain load, and the second of those was measured on a probe where every block read an address it had never read before.

That is the case the real kernel is not. A block reads the norm scratch at the query projection, again at the key, again at the value, and again at the two projections in the mlp, and it reads the residual stream once a layer for thirty layers. The second read hits a line the block's own L1 already holds from the first, the barrier does not evict it, and what comes back is a round of the plan out of date. The probe never gave a block a chance to hold a stale line because it never gave a block the same address twice.

So the price is the one the design most wanted to avoid. A matvec block writes one value for each row it owns and reads the whole activation vector once for every one of those rows, so the store side is one extra instruction a row and the load side is one for every column of every row, in the inner loop, on the operand the whole kernel is bound by. What saves it is that the activation vector is small and hot, a few thousand floats against a weight matrix of millions, so the atomic is on the operand that was already going to be in cache rather than on the one the kernel is bandwidth bound by. That is an argument for it being affordable and not a measurement of it, and the measurement is the first number stage one owes.

It is not every load and not every store. A block that writes a row of the up scratch and reads it back in the next record is reading its own write, and that path is ordinary at both ends. What has to be atomic is what crosses a barrier, which is the norm scratch, the query, the heads output, the up and gate scratch, the residual stream and the cache. In practice that is nearly all of it and almost never worth the branch to work out which, and the one place it will be worth working out is the matvec inner loop if the measurement says the atomic costs anything there.

## The order it gets built in

Three stages, each of which is worth landing on its own and each of which is a test of the stage after it.

Stage one is one kernel a layer. Twelve launches become one, and a thirty layer token goes from 363 launches to 33, which is a ceiling of 6281 tokens a second on the 4090 and 1506 on the M4, both above this milestone's target. It uses the grid barrier, it uses the Metal atomic path, and it uses the strided row loop, so it answers all three of the open questions above at a fraction of the work of the full thing.

Stage two is one kernel a token, and it is the same kernel. The table is built for the whole model either way and the kernel walks a range of it, so stage one launches the range belonging to one layer and stage two launches the whole range. That is a change to two arguments and not a second kernel, which is the reason the table is built in stage one rather than deferred: writing the layer's steps out by hand in the kernel would have been throwaway work and a second thing to keep correct.

The launch count after stage two is one plus whatever the sampler needs, and the ceiling stops mattering at all.

Stage three is prefill on the same kernel. A chunk is the same steps with a token dimension, which is the matmul rather than the matvec, and the barrier count a chunk is the barrier count a token. This is worth much less than the first two, because prefill is compute bound and already batched by #167, and it is worth doing so that there is one kernel and not two.

## What it has to agree with

The same greedy text as the unfused path, on every backend and every fleet model. That is the test that catches a barrier in the wrong place, because a missing barrier reads a stale value and a stale value is a plausible number rather than an error.

The logit corpus green on host, Metal and CUDA at the existing tolerances. Nothing here is an approximation. Every step does the same arithmetic in the same order as the kernel it replaces, so the answers should be identical in every digit and not merely close, and anything that is merely close is a bug.

A test that asserts the launch count for one token and does not move when the layer count does, which is what this issue exists to produce.

`tests/test_gpu_block.mojo` covers the fused layer against the unfused one on a synthetic model small enough to run on every machine, the way it already covers a chunk against a run of decodes.

The one that is easy to forget: the fused kernel has to deadlock loudly rather than quietly. A grid that is not fully resident hangs the GPU, which on a laptop is the display, so the launch checks the grid against the residency bound and refuses rather than trusting it, and the spin inside the barrier is bounded with a flag the host reads afterwards.

## What stage one measured

The grid width first, because everything else depends on it. On a 4090, SmolLM2 135M decoding 128 tokens takes 451 ms unfused, and fused it takes 366 ms on 96 blocks, 239 on 256, 238 on 384 and 239 on 512. Everything from 256 up is flat and 96 gives away half of what the path is worth. Above that it does not get slower, it stops working: 640 blocks and 768 blocks never finish a token. The shipped constant is 384, clamped by the occupancy query at three blocks a multiprocessor so that a smaller card gets a smaller grid.

On Metal the same sweep on Qwen 2.5 0.5B, in ms a token, is 100 at 10 blocks, 63 at 20, 50 at 40, 49 at 50, 55 at 60, and a collapse at 70. The shipped constant is four blocks a multiprocessor, which is 40 on an M4.

The residency bound cannot be probed with a barrier and nothing else. `fused_selftest` and `scripts/fused_grid_probe.mojo` rendezvous happily at grids where the real kernel deadlocks, 160 blocks against 70 on the M4 and 1024 against 640 on the 4090, because a kernel that holds registers and a shared array is resident at a narrower grid than one that holds neither. The occupancy query is an upper bound and not the answer, for the same reason: an ordinary launch is not a cooperative one.

Then the attention step, which was the one place a block owned a whole head. A model has between nine and thirty two heads, so a grid of 384 blocks put at most thirty two of them to work on the one step whose cost grows with the context, and the rest waited at the next barrier. It did not show up at all at a short prompt and it took the whole advantage away at a long one: Qwen at a 512 token prompt was nine per cent slower fused than unfused. Splitting a head's keys across blocks in the flash attention way, with a fold after one extra barrier a layer, is what the numbers below are measured on.

Decode rate in tokens a second, 4090, 128 tokens generated, through `scripts/bench.py`.

| model | prompt | unfused | fused | ratio |
| --- | --- | --- | --- | --- |
| SmolLM2 135M q8_0 | 8 | 310.7 | 483.0 | 1.55 |
| SmolLM2 135M q8_0 | 512 | 254.0 | 460.4 | 1.81 |
| Qwen 2.5 0.5B q4_K_M | 8 | 316.8 | 372.1 | 1.17 |
| Qwen 2.5 0.5B q4_K_M | 512 | 268.9 | 367.8 | 1.37 |
| Llama 3.1 8B q4_K_M | 512 | 93.7 | 81.8 | 0.87 |

Two things in that table decide the default. The ratios grow with the context rather than shrinking, which says the split did what it was for, and the 8B loses an eighth, which says the path is not free. A token of the 8B reads 5151 MiB of weights and the fused grid is narrower than one block a row, so what it saves in launches it gives back in bandwidth. The default is therefore a question about the model and not about the machine, which is `fused_by_default` in `src/molla/engine/device.mojo`: on when a layer's seven matrices come to a gibibyte a token or less, off above it, and `MOLLA_FUSED` overrides either way.

On Metal it is off whatever the model. At a 721 token context an M4 decodes SmolLM2 at 19 ms a token unfused and 24 fused, and Qwen at 35 against 54. The split helps there too and not by enough, because 40 blocks is a fifth of what the 4090 holds and a barrier across them costs more. Stage two is what changes that, if anything does.

## What the path costs in host memory, and what it does not

A fused SmolLM2 session on gpc was 1468 MiB resident against 279 MiB unfused, for a model whose weights are 136 MiB and whose card figure is 648 either way. That was read as the price of compiling the fused kernel, because the kernel is large and the plan is not, and it was wrong. Three measurements killed that reading. Setting `MOLLA_FUSED_BLOCKS`, which returns before the occupancy query and so compiles the kernel once rather than twice, left the figure at 1468. Cutting the six inlined quant forms of the matvec accumulation down to one left it at 1468. Building the plan and then generating nothing at all cost the whole 1.2 GiB before a single fused launch, and the launch itself added 536 KB.

What was left in the plan build after that is `fused_selftest`, which allocates three integers and reads them back. Splitting it in half named the line: the probe kernel and its launch cost nothing, and the readback costs all of it. `enqueue_create_host_buffer` allocates pinned memory, and the first pinned allocation in a process costs 1.2 GiB of resident host memory whatever its size. An ordinary device to host copy into a plain list reads the same three words and costs nothing, and with that change a fused session is 279 MiB, which is the unfused figure to within a megabyte.

The reason it is worth writing down rather than just fixing is that the fused path had been carrying a memory number nobody could explain, and the explanation everybody reached for was the plausible one. The measurement that settled it was the cheapest of the four and it was the last one taken.
