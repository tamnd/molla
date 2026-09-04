# Using what MAX already ships

[performance.md](performance.md) ended with three things it could not answer, and all three were about MAX. Whether graph capture is reachable from Mojo, whether a grid wide barrier is, and whether the 4.9 microsecond launch cost is the same on Metal. This page answers them and the answers move the plan, because the mechanism performance.md recommended as a fallback turns out to be unavailable on one backend and worth nothing on the other, while the mechanism it recommended as the main route turns out to be the only route.

It also answers a question nobody had asked: how much of what molla wrote by hand is already sitting in the packages molla links against. The answer is most of it, and it is importable today with no build change.

Every claim below is either compiled against max-core 26.5 as pinned in `pixi.toml` or measured on a machine in the fleet. The Modular source tree is checked out at `/Users/apple/github/modular-src` at Mojo 1.1.0.dev2026090205, which is ahead of the pinned runtime, so a signature read there is a lead and a signature that compiled is a fact. Both are labelled.

## The short version

Launch cost is now measured on both backends. It is 4.82 microseconds on the RTX 4090 through WSL2 and 20.10 microseconds on the M4. Metal is four times worse, which is a larger fact about molla's Metal numbers than anything to do with kernels.

Device graph capture is reachable from Mojo. It is also useless here. It raises on Metal by design, and on the fleet's only CUDA machine replaying a captured graph costs the same per node as launching the kernel eagerly. So the launch mechanism is not the lever.

There is no grid wide barrier in MAX and no cooperative launch, and the device wide semaphore that looked like one is a one writer lock and is NVIDIA only. A barrier built by hand out of device scope atomics works on both backends, and a round of it is ten times cheaper than a launch on Metal and no cheaper at all on CUDA unless the grid is kept to a few hundred blocks. On Metal it also does not carry ordinary global traffic across a block, so every value that crosses one has to move through an atomic.

MAX ships tuned kernels for nearly everything molla wrote by hand, in packages that are already on the import path. The exception matters: K quant matmul is CPU only in MAX, and molla's models are q4_K_M, so the one kernel molla most wanted to hand over is the one it has to keep.

Fusing a layer down to five kernels, which is about as far as fusion goes, still leaves launch cost at 115 per cent of the CUDA budget and 203 per cent of the Metal budget. Fusion is necessary and it is not sufficient on either backend. The persistent kernel is not the fallback any more, it is the plan.

Zero copy is worth doing for load time and for deleting the repack cache, and it is no longer worth doing for memory, because issue #166 already put peak memory where the goal wanted it.

## Launch cost on both backends

The measurement is an empty kernel launched in a long run on one stream, with issue time and completion time taken separately so it is visible whether the host is running ahead of the card. Two variants, one where the kernel is parametrised at each launch through `ctx.enqueue_function[kernel]` and one where it is compiled once with `ctx.compile_function` and only launched in the loop.

| device | eager, parametrised | compiled once | issue against completion |
| --- | --- | --- | --- |
| RTX 4090 through WSL2 | 4.82 us | 4.56 us | within 0.3 per cent |
| Apple M4 | 20.10 us | 20.08 us | within 0.6 per cent |

Two readings. Precompiling saves five per cent on CUDA and nothing at all on Metal, so hoisting compilation out of the loop is tidy but it is not a performance change and nothing should be restructured for it.

The one that matters is that issue time equals completion time on both. The host cannot get ahead of the card. Every launch is a round trip in effect, so the launch count is a hard divisor on tokens per second and not a cost that pipelining can hide behind the arithmetic.

The M4 number closes issue #171. molla launches 453 kernels for one token, which at 20.10 microseconds is 9.11 ms of launch cost, a ceiling of 110 tokens a second on Metal before a single multiply happens. Measured Metal decode is 24.7 tokens a second, so launches are not yet the binding constraint there, but they become it the moment the kernels are fixed.

The macbook is a shared machine whose load average is regularly above sixty, so the Metal figure is repeatable to about a microsecond rather than to three digits. It is good enough to carry the argument, which only needs to know that Metal is several times worse than CUDA rather than by exactly how much.

## Device graph capture

It exists and it is reachable: `from max.gpu.host.device_graph import DeviceGraph, DeviceGraphBuilder, DeviceGraphNode`, which compiles against the pinned 26.5. Build with `DeviceGraph.create(ctx, build)`, add nodes with `builder.add_function(compiled_fn, grid_dim=..., block_dim=..., dependencies=[])`, run with `graph.replay()`. There is a `DeviceGraphCache` and a `DeviceGraphMemoryPool` beside it.

The `DeviceGraph` docstring settles the portability question on its own. "Graph capture is currently implemented for CUDA and HIP devices only. Creating a graph on any other device, such as an Apple GPU or a CPU, raises." Running it on the M4 gives exactly that, `createGraphBuilder() not supported on this device context`. So half of molla's supported hardware cannot use this at all, and anything built on it would be a second code path rather than a shared one, which is the thing D7 exists to prevent.

The other half was measured. A graph of empty nodes against the same number of eager launches, on the 4090.

| nodes | eager total | eager each | graph replay total | replay each |
| --- | --- | --- | --- | --- |
| 1 | 8.78 us | 8.78 us | 4.77 us | 4.77 us |
| 30 | 147.22 us | 4.91 us | 242.96 us | 8.10 us |
| 120 | 623.72 us | 5.20 us | 695.77 us | 5.80 us |
| 453 | 2411.57 us | 5.32 us | 2281.55 us | 5.04 us |

Replay is linear in node count at about five microseconds a node, which is the eager launch cost. At thirty nodes the graph is slower than launching the kernels one at a time. At 453 nodes, which is a real token, it saves five per cent.

Two controls rule out the obvious explanations. Issue time and completion time are equal for replay just as they are for eager, 2757 against 2761 microseconds, so the cost is not the host waiting on a card that is behind. And a graph of 453 nodes with no dependencies between them costs the same as a chain of 453 nodes in a line, so it is not device side serialisation either. The cost is being paid per node, at replay, somewhere between the call and the hardware.

Building the graph is 8.9 ms for 453 nodes, paid once, which would be fine if replay were cheap.

This is a result about this machine and it should not be read as a result about CUDA. WSL2 reaches the GPU through a paravirtualised driver, so every submission crosses a VM boundary, and a replay that walks node by node on the host side of that boundary is exactly what these numbers look like. On native Linux, CUDA graph replay is normally well under a microsecond a node and this table would look completely different.

The fleet has no native Linux machine with an NVIDIA card. `server1`, `server2` and `server3` have no accelerator, `gpc` is Windows with WSL2, and the macbook is Metal. That gap is now the single most valuable thing to add to the fleet, because it is the difference between graph capture being a dead end and being a two times win on Linux CUDA. Until then the plan has to be one that works without it.

Worth recording for the day that machine exists: `DeviceGraphBuilder.recording_context()` is a zero rewrite adoption path. Its docstring says operations enqueued through the returned context "are recorded as graph nodes in enqueue order, simulating stream ordering, instead of executing eagerly", so that code written against `DeviceContext` can "record into a device graph without modification". molla's `device_forward` would need no changes at all, only a different context passed in. The caveat is that "host-visible waits (`synchronize()`, events, timers) raise", which the forward pass does not do.

## Grid wide sync

Issue #169's other half. `max.gpu.sync` has `barrier`, `named_barrier`, `syncwarp` and the whole `mbarrier_*` family, and every one of them is block level or cluster level. There is no `grid_barrier`. `enqueue_function` takes a `cluster_dim` and a list of `LaunchAttribute`, and there is no cooperative launch flag among them.

So the primitive performance.md hoped for is not there under that name, and the first answer to this question named the wrong replacement. `max.gpu.sync.semaphore.Semaphore` is not a rendezvous. It is a one writer lock, `fetch` and `wait` and `release` around a state a single block advances, which is what a split K matmul uses to hand a partial result on and not what a barrier needs. It also opens with `comptime assert is_nvidia_gpu()`, so it is unavailable on half of molla's hardware regardless. Neither fact was checked before it was written down.

The barrier has to be built out of device scope atomics, so `scripts/gridsync_probe.mojo` builds one and measures it. It is a sense reversing barrier over two `Atomic[DType.int32, scope="device"]` words, an arrival count and a generation. Every block adds one, the block that finds itself last resets the count and moves the generation on, and the rest spin on the generation. The spin is bounded rather than infinite, because a barrier that does not work would otherwise hang the GPU, and on a laptop that is the display.

### What each backend needs to make it order memory

The rendezvous itself works on both. No block reached the patience bound on either, at any grid size tried.

What the two backends do not agree on is what carries the ordering. On CUDA it rides the atomic: acquire release on the arrival, release on the generation store, acquire on the generation load, one instruction each.

An Apple GPU has none of those. `acquire`, `release` and `acquire_release` are all rejected on an atomic there, and a memory fence is worse than rejected: `fence` of every ordering, scoped and unscoped, crashes the Metal pipeline compiler with "Compiler encountered an internal error" rather than failing to compile. What works is `llvm_intrinsic["llvm.air.wg.barrier", NoneType](Int32(3), Int32(1))`, which is the same instruction `barrier` already emits with a different flag word, flag 3 being `mem_device | mem_threadgroup` where the ordinary block barrier passes 2. So on Metal the rendezvous is relaxed atomics with a device flagged threadgroup barrier either side of it.

That gets the blocks to meet. It does not get their data across, and this is the finding that changes the design. The probe has every block write its own slot, wait, and read its neighbour's, sixty four rounds over the grid.

| device | plain loads and stores | relaxed device atomics |
| --- | --- | --- |
| RTX 4090 through WSL2 | 98304 of 98304 correct | 98304 of 98304 correct |
| Apple M4 | 634 of 640 wrong | 640 of 640 correct |

On CUDA ordinary global traffic is coherent across the barrier and nothing special is needed. On Metal it is not, and the barrier makes no difference to that, because the problem is not ordering but that the value never leaves the core's own cache. The same addresses read and written through relaxed device scope atomics are correct every time. So a fused kernel on Metal has to move every value that crosses a block through an atomic load or store, which for float data means a bitcast to and from `int32` at both ends. That is a cost in the arithmetic rather than in the sync, and it is the first thing to measure when the fused kernel is built.

One more thing the probe found on the way, worth writing down because it costs an afternoon otherwise. A pointer rebuilt from an integer address, `Pointer[Int32, MutAnyOrigin](unsafe_from_address=Int(p) + i * 4)`, crashes the Metal pipeline compiler the moment an atomic touches it, while the same address reached as `Pointer[Int32, MutAnyOrigin](to=p[unsafe_offset=i])` compiles and runs. The address space appears to be lost on the way through the integer. Both forms are fine for ordinary loads and stores, which is why this only shows up once atomics are involved.

### Sizing the grid

`ctx.occupancy_max_active_blocks_per_multiprocessor` is the other half of the mechanism. Without a cooperative launch there is no guarantee from the driver that every block in the grid is resident at once, and a barrier across blocks that are not all resident deadlocks. Asking occupancy how many blocks fit and launching no more than that gives the same guarantee by construction.

It works on CUDA, where the probe gets 12 blocks a multiprocessor at 128 threads across 128 multiprocessors, so 1536 blocks. It is a `DeviceFunction` method and it is not supported on Metal, which raises `occupancyMaxActiveBlocksPerMultiprocessor is not supported on this device`. `ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)` does work there and reports 10 on the M4, so the floor of one block a multiprocessor is available, and anything above that has to be found by experiment rather than asked for.

### What a round costs

Best of five, a thousand rounds a run, nothing in the kernel but barriers, swept over the grid because a rendezvous costs more the more blocks are in it.

| blocks | RTX 4090 through WSL2 | blocks | Apple M4 |
| --- | --- | --- | --- |
| 1536 | 4.59 us | 10 | 2.50 us |
| 384 | 1.26 us | 2 | 2.30 us |
| 96 | 0.75 us | 1 | 1.76 us |
| 24 | 0.75 us | | |
| 6 | 0.72 us | | |
| 1 | 0.53 us | | |

Those repeat to three digits across runs. What to compare them against is the launch cost measured at the top of this page, 4.82 microseconds on the 4090 and 20.10 on the M4. The probe times an empty launch itself as a control and agrees with both, loosely: 5.2 to 9.5 microseconds on gpc and 21 to 25 on the M4 across runs, which is the same figure with the noise a 1536 block launch and a loaded laptop respectively add to it.

Metal is the easy read. A round is roughly ten times cheaper than a launch at every grid size, so the persistent kernel is a large win there and the only open question is what the atomic traffic costs.

CUDA is not. At full residency a barrier round costs 4.59 microseconds against a 4.82 microsecond launch, which is a saving of five per cent and not worth building anything for. The saving only appears when the grid is smaller: four times cheaper at 384 blocks, six times at 96, and flat from there down. All 1536 blocks are contending on one counter word, so the cost at the top of the sweep is contention on the arrival rather than the rendezvous itself, which a two level barrier that aggregates within a multiprocessor first would mostly remove.

Taken through the budget, at five stages a layer on a thirty layer model, so about 153 rounds a token.

| device and grid | sync a token | budget | share |
| --- | --- | --- | --- |
| 4090 at 1536 blocks | 0.702 ms | 0.639 ms | 110 per cent |
| 4090 at 384 blocks | 0.193 ms | 0.639 ms | 30 per cent |
| 4090 at 96 blocks | 0.116 ms | 0.639 ms | 18 per cent |
| M4 at 10 blocks | 0.383 ms | 1.514 ms | 25 per cent |

So the persistent kernel fits, and it fits with a condition that was not visible before: on CUDA the grid has to be a few hundred blocks rather than everything that is resident. That is a constraint on the fused kernel and not on the barrier, because a few hundred blocks of 128 threads is a small fraction of a 4090 and the arithmetic has to be arranged so that each block carries many output rows. It also means the residency question at the bottom of this page has an answer that cuts the other way from the one expected: the risk is not that the resident grid is too small to be worth it, it is that the resident grid is too large to synchronise cheaply.

Both pieces of the mechanism compile today and one of them is missing on Metal, which is a grid chosen by hand there rather than a blocker.

## What MAX already has that molla wrote by hand

These are all in `/Users/apple/github/modular-src/max/kernels/src`, and the packages they belong to are in `.pixi/envs/default/lib/mojo` as `nn.mojoc`, `kv_cache.mojoc`, `quantization.mojoc` and `linalg.mojoc`.

The load bearing fact is that they import from a plain build with no extra include path. This compiles and runs in molla's own environment:

```mojo
from nn.normalization import rms_norm_gpu
from nn.kv_cache_ragged import generic_flash_attention_kv_cache_ragged
from kv_cache.types import PagedKVCache
from quantization.qmatmul_gpu import matmul_gpu_qint4
```

So adopting any of this is a code change and not a packaging project.

| what molla has | what MAX has | where |
| --- | --- | --- |
| `device_rms_norm` | `rms_norm_gpu`, `rms_norm_gpu_warp_tiling`, `rms_norm_gpu_block` | `nn/normalization.mojo:839`, `:625`, `:803` |
| a norm and then an add | `rms_norm_fused_residual_add` | `nn/normalization.mojo:3703` |
| a norm and then a rope | `rms_norm_rope` | `nn/normalization.mojo:2951` |
| `device_rope` | `generic_fused_qk_rope_bshd_paged_ragged` | `nn/kv_cache_ragged.mojo:3462` |
| three matvecs into the cache | `generic_fused_qkv_matmul_kv_cache_paged_ragged` | `nn/kv_cache_ragged.mojo:98` |
| `device_attend` | `generic_flash_attention_kv_cache_ragged` | `nn/kv_cache_ragged.mojo:3606` |
| `DeviceCache`, a flat slab | `PagedKVCache`, `PagedKVCacheCollection` | `kv_cache/types.mojo:2221`, `:3502` |
| host K quant matmul | `matmul_Q4_K_pack_b`, `matmul_Q6_K_pack_b` | `quantization/qmatmul_k.mojo:571` |
| the repack cache file | `gpu_qint4_repack_Q4_0` | `quantization/qmatmul_gpu.mojo:2342` |

Two corrections to things assumed in earlier sessions. `.mojoc` packages are readable: they are MPKG with a zstd payload starting at byte 96, after a sixty four character digest, and Python's `compression.zstd` opens them. And `mega_ffn`, which that reading turned up and which sounded promising, is `mo.composite.mega_ffn_nvfp4`, a fused MoE feed forward for NVFP4 and MXFP8. molla's models are dense and q4_K, so it does not apply.

### The gap that matters

`unfused_qkv_matmul_ragged_paged_gguf_quantized` at `nn/kv_cache_ragged.mojo:3087` is the fused quantized qkv path, it takes GGUF encoding names as parameters, and its docstring says "This is only supported on CPU."

That is not an isolated limitation. MAX's K quant kernels are the CPU ones in `quantization/qmatmul_k.mojo`, exposed to the graph compiler as `vroom_q4_k_matmul` and `ggml_q4_k_dequantize`. The GPU quantized matmul is `matmul_gpu_qint4`, which is int4 with a group size, the shape Q4_0 and GPTQ have and K quants do not.

So MAX's answer for a q4_K_M model on a GPU is to dequantize and then run a dense matmul, which is what molla's repack cache already does in effect. The one kernel molla most wanted to hand over is the one MAX does not have on the GPU, and molla keeps its own K quant path.

This is less bad than it sounds. It says molla owns the matmul and adopts everything around it, and the matmul is the part where molla has already done the work and where the arithmetic is bandwidth bound anyway. What molla does not own after this is attention, the cache, rope and the norms, which is where the tuned versions are worth the most and where molla's hand written ones are naive.

### The fusion mechanism, which is the real prize

`matmul_gpu_qint4` and the `linalg` matmuls all take an `elementwise_lambda_fn` parameter of type `elementwise_epilogue_type`, defined at `linalg/utils.mojo:109` as a capturing function from an index and a SIMD value to nothing, with a compute variant that returns a value.

That is fusion without adopting anybody's kernel. An epilogue on a matvec means the bias add, the activation and the residual add stop being separate launches and become the tail of the launch that was already happening. molla's own `device_matvec_into` can take the same parameter. This is the highest leverage local change on the page because it costs one signature change and it removes launches everywhere at once.

## The launch budget, on both backends

llama.cpp on SmolLM2 is 782.5 tokens a second on gpc and 330.2 on the macbook, so two times is 1565 and 660.4, which is 0.639 ms and 1.514 ms a token.

Ceilings, counting nothing but launch cost, for a thirty layer model plus the embedding, the final norm and the head.

| kernels a layer | a token | CUDA ceiling | Metal ceiling |
| --- | --- | --- | --- |
| 15, which is today | 453 | 458 | 110 |
| 8 | 243 | 854 | 205 |
| 6 | 183 | 1134 | 272 |
| 5 | 153 | 1356 | 325 |
| 4 | 123 | 1687 | 404 |
| one for the whole token | 1 | 207469 | 49751 |

Five a layer is about as far as fusion goes on a Llama shaped block, and the mapping is not hand waving: the qkv projections become one matmul whose epilogue ropes and writes straight into the cache, attention is one, the output projection carries the residual add in its epilogue and the next norm folds into that, the gate and up projections are one matmul with a SwiGLU epilogue, and the down projection carries the second residual add. Five.

At five a layer, launch cost alone is 0.737 ms against a 0.639 ms budget on CUDA, which is 115 per cent, and 3.075 ms against 1.514 ms on Metal, which is 203 per cent. The target is missed on both before any arithmetic is done at all.

There is no arrangement of kernels per layer that works. Even four a layer, which needs a fusion nobody has written, only clears the CUDA target by eating 92 per cent of the budget and misses Metal by 39 per cent. To reach the goal with a launch per layer at all, launches would have to cost 0.54 microseconds, which is a tenth of what CUDA does through WSL2 and a fortieth of what Metal does.

A persistent kernel spends 0.8 per cent of the CUDA budget and 1.3 per cent of the Metal budget on launch. That is the only shape that fits, and it fits with room to spare on both.

## What this changes

performance.md recommended the persistent kernel with graph capture as the fallback. Both halves of that are now wrong in the same direction. Graph capture is not a fallback, because it does not exist on Metal and buys five per cent on the only CUDA machine available. And the persistent kernel is not one option among two, it is the only arrangement where the numbers close.

So the persistent kernel moves from fourth in the order to the thing everything else is arranged around. Its two prerequisites both exist, one of them has to be hand rolled rather than imported, and each backend puts a different condition on it: a small grid on CUDA, atomic traffic on Metal.

Fusion stays, and its justification changes. It was going to be a two times improvement on the way to the goal. It is now the step that proves out the fused arithmetic, the epilogue plumbing and the paged cache in a form that can be tested kernel by kernel against the logit corpus, before the same arithmetic is folded into one kernel where a wrong answer is much harder to localise. Fusion is the rehearsal, and it happens to be worth about three times on its own.

Adopting MAX's kernels stays too and it gets easier to justify, because `generic_flash_attention_kv_cache_ragged` and `PagedKVCache` are also what M3's continuous batching needs, and continuous batching is the only honest route to two times on an 8B. Writing molla's own paged cache and then replacing it would be two pieces of work for one result.

## Zero copy

Three copies of every weight exist during a load today. The GGUF is mapped, the repack cache is mapped and was written to disk by an earlier full pass over the file, and `Pool.copy_in` hands `enqueue_copy` an address in that second mapping.

The third one has a hidden fourth inside it. `enqueue_copy` from a pageable host address cannot DMA, so the driver stages it through its own pinned bounce buffer, which is one more full copy of the model on every load. `ctx.enqueue_create_host_buffer[dtype](size)` returns page locked memory whose docstring says devices "can use direct memory access (DMA) to transfer data without relying on the CPU". Reading each tensor into a bounded pinned window and copying from there removes the bounce, and the window is the same 64 MiB bound that #166 already established for dropping pages, so the two mechanisms are the same loop.

The second copy can be deleted outright. `gpu_qint4_repack_Q4_0` is MAX's precedent for the shape: upload the quantized blocks exactly as they sit in the file and rearrange them on the card. Doing that for K quants deletes the repack cache file, the second mapping, the disk space beside every model, and the slow first run that `generate_device` currently apologises for in its module docstring. It also removes an entire class of correctness question about whether the cache matches the file, which `model_key` exists to answer.

The third piece is only relevant if molla ever goes through the C API. `M_newWeightsRegistry` documents that "The data pointers are borrowed, not copied. You must keep the backing memory alive for the lifetime of the weights registry", so a registry over a mapping costs nothing. There is no graph authoring API in C and no `max` Python package installed, so building a MAX graph is not reachable from here today, and this is recorded rather than planned.

What zero copy is not is a memory win. Issue #166 already took peak resident memory to 2372 MiB on the 8B against llama.cpp's 5031, and 222 against 443 on SmolLM2, which is 1.6 to 2.1 times lighter and meets the memory half of the goal. Pinned staging and card side repacking are worth doing for load time, for deleting a file format, and for the disk, and claiming them as a second memory win would be counting the same thing twice.

## What is still not known

Whether 4.82 microseconds is CUDA or whether it is WSL2. This is now the largest single uncertainty on the page, because it decides whether graph capture is a dead end or a two times win on Linux CUDA, and it moves every CUDA ceiling in the budget table by roughly two. It needs a native Linux host with an NVIDIA card, which the fleet does not have.

What a persistent kernel's grid should be. The sweep above says the risk is the opposite of the one expected: a grid sized to full residency on a 4090 synchronises at the price of a launch, so the useful grid is a few hundred blocks, and whether the arithmetic of an 8B layer runs well on a few hundred blocks of 128 threads is now the question. A two level barrier would raise that ceiling and has not been written.

What the Metal atomic requirement costs. Every value crossing a block on that backend has to be bitcast through an `int32` atomic load or store, and how much that takes out of a fused layer is unmeasured. It is the first thing the fused kernel should report.

Whether MAX's flash attention and paged cache accept a K quant weight anywhere in their signatures, or whether adopting them forces the whole layer to dequantized activations first. The signatures were read and they take `LayoutTensor` of float32 for the hidden state, which suggests the latter, but reading a signature in a tree that is ahead of the pinned runtime is a lead and not a fact. It gets settled by compiling a call.

Whether the M4's 20.10 microseconds is Metal or is the macbook's load average. It should be taken again on a quiet Apple machine before anything is designed to the exact number, though the argument on this page only needs it to be several times CUDA and it is comfortably that.
