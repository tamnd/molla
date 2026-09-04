# One launch a token

molla launches a kernel for every step of every layer. A Llama shaped layer is twelve of them after #168 and #194 folded the biases, the residual adds and the gate into the projections that were already running, so a thirty layer model is 363 launches for one decoded token. [max.md](max.md) measures a launch at 4.82 microseconds on a 4090 through WSL2 and 20.10 on an M4, so those 363 launches are 1.75 ms and 7.30 ms of nothing but submission, which is a ceiling of 571 tokens a second on the 4090 and 137 on the M4 before a single multiply happens.

That ceiling is the binding constraint on the small models this milestone is measured on, and no arrangement of kernels per layer clears it. The budget table in max.md walks it: five kernels a layer, which is about as far as fusing within a layer goes, still spends 115 per cent of the CUDA budget and 203 per cent of the Metal one. The launch count has to stop being proportional to the layer count.

This is the design for that, what it rests on, and the order it gets built in.

## The shape

One kernel launch a token. The grid is chosen once, every block stays resident for the whole token, and the layers are walked inside the kernel with a grid wide barrier between the steps that depend on each other.

That is only possible because a barrier across the grid is possible, which was not known until #170's probe measured one. The findings that shape everything below are in the grid wide sync section of [max.md](max.md), and three of them are constraints rather than facts:

A barrier round is cheap on Metal, 2.5 microseconds against a 20.10 microsecond launch, and it is not cheap on CUDA at full residency, 4.59 microseconds at 1536 blocks against a 4.82 microsecond launch. It gets to 1.26 at 384 blocks and 0.75 at 96 and is flat below that, because the cost at the top is 1536 blocks contending on one counter word rather than the rendezvous itself. So the CUDA grid has to be a few hundred blocks and each block has to carry many output rows.

Ordinary global loads and stores are not coherent across a block on Metal. A value written by one block and read by another after a barrier is stale 634 times out of 640, and the barrier does not help, because the problem is that the value never left the core's own cache. Through relaxed device scope atomics it is right every time. So every value that crosses a block on that backend is an atomic load or store with a bitcast at each end.

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

Six of those twelve boundaries are not real. The three attention projections read the same input and write three disjoint outputs, so they are one step over a concatenated row space. The two ropes touch disjoint memory, so they are one step. The up and gate projections read the same input and the gate reads the up, which is the only ordering among the four, and it is removed by having the gate step read the up plane it just wrote in the same block, so they are one step over a concatenated row space with the activation applied at the end.

That leaves eight barriers a layer:

```text
norm, qkv, rope, attend, out, norm, up and gate, down
```

Plus the embedding lookup, the final norm and the output head, so a thirty layer token is 243 rounds. At 0.75 microseconds on a 4090 at 96 blocks that is 0.182 ms against a 0.639 ms budget, 29 per cent, and at 2.5 microseconds on an M4 it is 0.608 ms against 1.514 ms, 40 per cent. Both fit. Neither fits comfortably enough to spend barriers carelessly, which is why the six that collapse are collapsed.

## The plan, and why it is data rather than code

The kernel cannot call the host between layers, so everything the host currently passes as an argument has to be in device memory before the launch. That is one array of step records, built once when the model is loaded, walked by the kernel.

A record is an opcode, up to three pointers, and the handful of integers the step needs. The pointers are what the host already computes: `w.device_address()` for a weight matrix, a scratch vector's base, a cache layer's base plus the slot offset. The integers are the rows, the columns, the row stride in bytes, the epilogue flags, and the quant descriptor.

Two things make this an interpreter rather than a switch. The quant form, the group size and whether there is a minimum plane are comptime parameters on the matvec today, and five combinations are compiled. In the fused kernel they arrive as a runtime triple and the step branches over the same five, which is one uniform branch a step and five copies of the dot product loop in the binary. That is the price of one kernel source for every model, and it is the right price, because the alternative is a kernel instantiated per model shape and a compile on every load.

The other is that the step table is the same table on every backend and for every architecture. Gemma's post norms are two more steps in the list, Qwen's biases are an epilogue flag on a step that already exists, and neither is a branch in the kernel. The host code that today decides which kernels to enqueue becomes the host code that decides which records to write, and it is the same decisions in the same order.

## Sizing the grid

The grid is fixed for the whole token, so it has to be large enough to hold the widest step and small enough to synchronise cheaply, and the two pull in opposite directions.

On CUDA the occupancy query gives the resident bound, 1536 blocks of 128 threads on a 4090. Launching the bound is what guarantees no deadlock, and launching fewer than the bound is equally safe and much cheaper to synchronise. So the grid is the smaller of the occupancy bound and a tuning constant, and the sweep says the constant should be in the low hundreds.

On Metal there is no query, so the grid is the multiprocessor count times a constant found by experiment, floored at the multiprocessor count, which is always resident.

What this costs is arithmetic width. A 4096 row projection on 96 blocks is each block reducing 43 rows in a strided loop rather than one row a block. For a small model that is a gain, because the weights come out of the L2 for the blocks behind the first. For an 8B it is a risk, because a bandwidth bound matvec wants more threads in flight than a few hundred blocks provide, and that is the number this design is most likely to be wrong about. It is measured in stage one below, before anything is built on top of it.

## What crosses a block

On CUDA nothing special. On Metal every scratch vector read by a block that did not write it has to move through a relaxed device scope atomic, so a `Float32` becomes a `bitcast` to `Int32`, an atomic store, and a `bitcast` back after an atomic load on the other side.

That is not every access. A block that writes a row of the query scratch and reads it back in the same step is reading its own writes and can use ordinary loads. It is only the value that crosses a barrier, which is the norm scratch, the query, the heads output, the up and gate scratch, the residual stream, and the cache. In practice that is nearly all of them, so the honest planning assumption is that every scratch access on Metal is an atomic.

What that costs is unmeasured. A relaxed atomic load on Apple should be a load that misses the core cache by construction, which is what correctness requires and what a barrier separated stage would have had to do anyway, so the expectation is that it is close to free and the expectation is not evidence. It is the first number stage one reports.

## The order it gets built in

Three stages, each of which is worth landing on its own and each of which is a test of the stage after it.

Stage one is one kernel a layer. Twelve launches become one, the layer is still launched per layer from the host, and there is no step table because the layer's own steps are written out in the kernel. A thirty layer token goes from 363 launches to 33, which is a ceiling of 6281 tokens a second on the 4090 and 1506 on the M4, and both are above this milestone's target. It uses the grid barrier, it uses the Metal atomic path, it uses the strided row loop, and it answers all three of the open questions above at a fraction of the work of the full thing.

Stage two is one kernel a token. The step table appears, the layer loop moves inside the kernel, and the launch count becomes one plus whatever the sampler needs. This is where the ceiling stops mattering at all.

Stage three is prefill on the same kernel. A chunk is the same steps with a token dimension, which is the matmul rather than the matvec, and the barrier count a chunk is the barrier count a token. This is worth much less than the first two, because prefill is compute bound and already batched by #167, and it is worth doing so that there is one kernel and not two.

## What it has to agree with

The same greedy text as the unfused path, on every backend and every fleet model. That is the test that catches a barrier in the wrong place, because a missing barrier reads a stale value and a stale value is a plausible number rather than an error.

The logit corpus green on host, Metal and CUDA at the existing tolerances. Nothing here is an approximation. Every step does the same arithmetic in the same order as the kernel it replaces, so the answers should be identical in every digit and not merely close, and anything that is merely close is a bug.

A test that asserts the launch count for one token and does not move when the layer count does, which is what this issue exists to produce.

`tests/test_gpu_block.mojo` covers the fused layer against the unfused one on a synthetic model small enough to run on every machine, the way it already covers a chunk against a run of decodes.

The one that is easy to forget: the fused kernel has to deadlock loudly rather than quietly. A grid that is not fully resident hangs the GPU, which on a laptop is the display, so the launch checks the grid against the residency bound and refuses rather than trusting it, and the spin inside the barrier is bounded with a flag the host reads afterwards.
