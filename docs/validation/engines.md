# Engines

molla has been tuned against its own previous numbers for three milestones. This document is what came out of stopping that and reading the two engines it is measured against instead, on the grounds that a kernel which is eight times slower than another kernel doing the same arithmetic is not a tuning problem.

The reading was done against `llama.cpp` at commit `f9f09f02cc44` and `ollama` at commit `b68365a`, both cloned fresh rather than remembered. Every claim below has a file and a line behind it. The long form notes are outside this repository; what is here is what molla is going to act on.

## The measurement this starts from

[bench.md](bench.md) was a full version stale when this began, and it was the source of the belief that prefill was more than a hundred times off. It is not. These are gpc, RTX 4090, molla 0.4.5, llama.cpp `b10760`, ollama 0.32.9, 514 prompt tokens and 128 generated, best of five.

SmolLM2 135M Q8_0:

| engine | prefill tok/s | decode tok/s | ttft ms |
| --- | --- | --- | --- |
| molla | 10489.8 | 270.0 | 49 |
| llama.cpp | 34834.6 | 870.4 | 15 |
| ollama | 13161.3 | 675.4 | 39 |

Qwen 2.5 0.5B Q4_K_M:

| engine | prefill tok/s | decode tok/s | ttft ms |
| --- | --- | --- | --- |
| molla | 5039.2 | 290.2 | 102 |
| llama.cpp | 42264.1 | 774.2 | 12 |
| ollama | 6527.7 | 520.5 | 79 |

Llama 3.1 8B Q4_K_M, 515 prompt tokens, best of three:

| engine | prefill tok/s | decode tok/s | ttft ms |
| --- | --- | --- | --- |
| molla | 380.1 | 86.0 | 1355 |
| llama.cpp | 10568.2 | 160.2 | 49 |
| ollama | 5803.9 | 149.1 | 89 |

Peak memory is left out of those tables on purpose. The gpc runs report host resident set, which does not include a CUDA allocation, so molla's column came out at 1464, 1480 and 1494 MiB for three models that differ by sixty times in size, which is the reporting bug from #180 wearing a different hat. The memory comparison that means something is the host path at a 2048 context, where molla holds 7029 MiB for the 8B and 262 MiB for SmolLM2, against llama.cpp's 5361 MiB and 421 MiB on unified memory with the KV cache inside those figures.

So the gaps are 3.3x, 8.4x and 27.8x on prefill and 3.2x, 2.7x and 1.9x on decode. Decode is a constant factor and prefill is not. A ratio that grows monotonically with model size is a scaling failure, and it is the only number in the set that behaves that way.

## Why prefill scales wrong

`planar_matmul_kernel` in [src/molla/nn/gpu.mojo](../../src/molla/nn/gpu.mojo) reads a weight value once and multiplies it into `SPAN * MM_GROUPS` accumulators, which is sixty four tokens at the shipped constants. That amortizes the weight read and the dequantization, and it is what took prefill from a decode per prompt token to where it is now.

It does nothing at all for the activation. The grid is `(ceil(tokens / (SPAN * MM_GROUPS)), rows)`, so there are `rows` blocks and each of them reads the whole `T` by `cols` activation tile from global memory:

```mojo
for k in range(SPAN):
    acc[k] += d * x[unsafe_offset=at0 + k * cols + i]
```

Activation traffic for one matmul is `rows * tokens * cols * 4` bytes against `rows * cols` bytes of weight, a ratio of about `4 * tokens` to one. The activation is the dominant term and it is multiplied by the output row count, which is exactly the thing that grows when the model grows. 3.3x to 8.4x to 27.8x is that term becoming visible.

There is no shared memory in the kernel and there are no matrix cores. Both operands come from global memory on every block. That is the defect, it is structural, and no constant in the file moves it.

## What llama.cpp does instead, on Metal

`kernel_mul_mm` is a 64 by 32 output tile with a K step of 32, run by 128 threads arranged as four simdgroups in a two by two grid, holding eight `simdgroup_float8x8` accumulators between them. It stages 4096 bytes of A and 2048 bytes of B in threadgroup memory per step, with A written k major so that `simdgroup_load` can read it, and both operands are then read from threadgroup memory rather than device memory. The activation tile is read once per output tile of sixty four rows, not once per row.

It is selected by `ggml_metal_op_mul_mat_use_mm`, which requires `ne00 >= 64 && ne11 > 8`, so a batch of nine tokens or more takes it. Between two and eight there is a separate `mul_mv_ext` family, and for K quants that family only covers four to eight. Below that it is the plain matvec, which is the shape molla's decode kernel already has.

## What llama.cpp does instead, on CUDA

`mmq.cu` line 312 is the whole design in one statement:

```cpp
if (turing_mma_available(cc)) {
    return true;
}
```

On Turing and later, a quantized weight is never dequantized into a dense buffer for cuBLAS at any batch size. Everything goes through MMQ, which is a 128 by J tile, J chosen at runtime from eight upward, 256 threads, a K step of 256 values held in shared memory, and stream k partitioning so that a matrix which does not divide evenly across the multiprocessors still fills them. The weight tile is a fixed 38912 bytes of shared memory and the activation tile is `J * 144` bytes, so the total runs from 40992 to 57856 bytes, which is why the kernel has to opt in to the 48 KiB carveout.

The inner instruction is `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`, or `__dp4a` on older cards. It is integer end to end.

## The three things that make the inner loop integer

These are the findings that transfer to molla whatever the tile geometry ends up being.

**Activations are quantized to int8 in a separate kernel launch before the matmul.** `quantize_row_q8_1_cuda` runs first and writes `block_q8_1`, two halves and thirty two signed bytes in thirty six, or `block_q8_1_mmq` at 144 bytes for the tiled path. After that no float is read in the accumulation loop at all. molla reads f32 activations and multiplies in float, one weight at a time, and that four to one on work per value is the same finding #186 arrived at from the other direction.

**The activation row sum is stored next to the scale.** The DS4 layout puts the sum of the row's int8 values in the second half of the block. A K quant correction term is `-dmin * m * sum(a)`, and with the sum already there it collapses from a second dot product into a single fma. molla currently pays for that correction with a full second multiply per value, which is the `m * a` and `m * (a0 + a1)` terms in the loop above.

**The weight unpack happens in the tile load, once, not in the math loop, every time.** `load_tiles_q4_K` expands nibbles to one four bit value per int8 lane and pre multiplies `{d * sc, -dmin * m}` into a single `half2` while filling the shared memory tile. The Metal q4_K matvec goes further and never shifts a nibble: it masks a `uint16` and corrects the resulting factors of sixteen and two hundred and fifty six inside the final scale expression. Both are free.

## What MAX gives us and what it does not

D7 says to use the dependency where it covers the target. It covers exactly half of this one, and the halves are worth stating precisely because the answer decides the shape of two kernels.

| symbol | present | usable here |
| --- | --- | --- |
| `layout.tensor_core.TensorCore` | yes | CUDA and ROCm only, `is_nvidia_gpu()` or AMD, no Metal path exists |
| `layout.tensor_core_async.TensorCoreAsync` | yes | Hopper wgmma, not the 4090 |
| `linalg.matmul.matmul` | yes | dense dtypes, would need the weights dequantized first |
| `quantization.qmatmul_gpu.matmul_gpu_qint4` | yes | NVIDIA only and wants a weight layout that is not molla's |
| `quantization.qmatmul_k.matmul_Q4_K` | yes | CPU only |

So on CUDA the matrix core path is reachable through `TensorCore`, used inside a kernel of ours that dequantizes the planar weights on the way into shared memory. That keeps the layout molla already has and buys the mma instruction, which is the arrangement llama.cpp arrived at as well.

On Metal there is nothing. `TensorCore` branches between NVIDIA and AMD and has no third case, and no Metal simdgroup matrix intrinsic is reachable from Mojo today. The Metal GEMM therefore stages both operands in threadgroup memory and multiplies with ordinary fused multiply adds, which still removes the `rows` factor from the activation traffic even though it does not get simdgroup matrices. Whether an `air.simdgroup_matrix` intrinsic can be reached at all is a separate spike and not a blocker for the tile.

## Launch count was the wrong suspect

M2c was opened on the reading that molla's decode ceiling is set by launch overhead. That reading came from molla's own profile and it does not survive contact with the reference.

llama.cpp on Metal issues about 580 `dispatchThreadgroups` calls per decoded token for a thirty two layer 8B, in two command buffers, with no graph capture anywhere in the backend. At 59.19 ms a token that is about 102 microseconds of wall time behind each dispatch, so encode cost is not a term in the result. CUDA does collapse five hundred to nine hundred launches into one `cudaGraphLaunch`, and nowhere in that backend is a speedup from doing so quantified.

That matches what molla measured on its own fused path: correct, and slower, 54 ms a token against 16 to 39 unfused on SmolLM2, because one kernel a layer gives a ten block grid where the unfused matvec launches a block per output row. Trading roughly a hundred and fifty times the parallelism for twelve times fewer launches is a bad trade, and the reference says the denominator was never worth much. #170 should stay parked and PR #200 should stay a draft.

The fusion that llama.cpp does keep is narrow and it is a different idea. It absorbs a cheap elementwise op into the expensive kernel next to it, only where that removes a round trip to memory. On a Llama decode graph on Metal exactly one pattern fires, `RMS_NORM` into `MUL`. On CUDA there are twenty patterns, all decode only, and MMQ has no fusion support at all.

## Memory, which is three separate problems

**A first run loads the model twice.** [src/molla/engine/device.mojo](../../src/molla/engine/device.mojo) around lines 486 to 497 runs a complete unbounded host load in order to write the repack sidecar, then reopens the file and loads it again in the same process, and the pages from the first pass are never released. Measured with a hardlinked copy so the cache misses, SmolLM2 under `generate --device=metal` goes from 217 MiB on a hit to 394 MiB on a miss. The 11066 MiB figure that has been quoted for the 8B is that, not a steady state.

**The repack grows the tensor.** molla's 8B repack cache is 6474 MiB against a 4693 MiB GGUF, 1.38 times, and it is resident for the life of the process. llama.cpp's CPU repack sets `get_alloc_size = nullptr` at `repack.cpp:4828`, so it is byte for byte size neutral, it is never written to disk, there is no `fwrite` or `ofstream` anywhere under `ggml/src/ggml-cpu/`, and the GPU path does not repack at all.

**Weights are not copied on unified memory.** `ggml_backend_dev_buffer_from_host_ptr` binds tensors to addresses inside the mmap, and the `CPU_Mapped` buffer vtable has `free_buffer = NULL` with a comment saying the pointer is not owned by the buffer. There is no point in the load where a host copy and a device copy of the whole model both exist. On Apple silicon the GPU's copy of the weights is the page cache.

And separately from all three, molla's KV cache is f32 at [src/molla/engine/runner.mojo](../../src/molla/engine/runner.mojo) lines 74 to 76 where llama.cpp defaults to f16 and offers q8_0. That is a factor of two on the KV plane for a dtype change, and 3.76 with q8_0.

## The structural thing molla does not have

Every llama.cpp Metal pipeline is compiled per shape. A name is built from the runtime shape, looked up in a cache, and on a miss a specialization is compiled with function constants baked in, so the GQA broadcast factors become compile time immediates, the modulo and the divide fold away, and the bounds checks vanish on shapes that are known to be aligned. The function constant bases are 600 for `mul_mv` and 700 for `mul_mm`.

Mojo expresses the same thing with comptime parameters, but it needs a cache keyed on shape and a compile on miss, which is a build and runtime change rather than a kernel change. It applies everywhere at once, which makes it the largest single lever left and also the one with the longest lead. It belongs in a milestone of its own.

## Ollama is not a second reference

On anything that is not an Apple machine, ollama no longer has an engine. `model/models/`, `kvcache/`, `runner/ollamarunner/`, `runner/llamarunner/`, `llm/memory.go` and the cgo ggml backend are all deleted, and a GGUF model is served by an upstream `llama-server` subprocess pinned at `b10760`. Its own engine is MLX, safetensors only, and has no batching whatever, with `B = 1` asserted in every cache. The ollama rows in the tables above are llama.cpp with a process boundary in front of it, which is why they sit between molla and llama.cpp rather than anywhere interesting.

Two ollama ideas are worth taking and neither is about speed. The lazy KV snapshot in `x/mlxrunner/prefix_cache.go`, where a snapshot indexes into the live buffer and owns no bytes until a write would clobber it, is a good prefix cache. And `Batch.Memo`, which resolves the attention mask once per forward rather than once per layer, is cheap and preserves the result exactly.

## What gets built

In order, largest ratio first.

1. A tiled GEMM with both operands staged in shared memory. On CUDA it dequantizes planar weights into the tile and multiplies with `TensorCore`. On Metal it stages the same tiles and multiplies with fused multiply adds. This is the item that removes the `rows` factor and it is the only one that can move 27.8x.
2. Int8 activation quantization as a separate pre pass, with the row sum stored beside the scale so the K quant minimum correction becomes one fma. Half of this exists on `perf-int8-activations` under #186.
3. The weight unpack folded into the tile load, with the scale and minimum pre multiplied once per tile, and the nibble shifts replaced by masks with the factor corrected in the scale.
4. An f16 KV cache, then q8_0 behind a flag.
5. Close the GGUF mapping between the two load passes, which removes the first run double residency.
6. Make the repack size neutral, or skip it on the device path the way llama.cpp does.
7. Shape specialized kernel compilation with a pipeline cache.

One through four are one milestone and five and six can run beside them because they touch load and not kernels. Seven is a milestone of its own. They are #201 through #206 and #208, tracked in M2d, #209.

## What this milestone has to reach

The standing goal is two times llama.cpp on tokens a second at half its memory. A 27.8x prefill gap does not become 0.5x in one milestone and saying otherwise would be a plan that cannot fail honestly, so the gate here is parity, measured on gpc against the tables at the top of this file, and the 2x gate stays where it is on #57.

| model | quantity | now | this milestone |
| --- | --- | --- | --- |
| Llama 3.1 8B Q4_K_M | prefill tok/s | 380.1 | 10568 |
| Llama 3.1 8B Q4_K_M | decode tok/s | 86.0 | 160 |
| Qwen 2.5 0.5B Q4_K_M | prefill tok/s | 5039.2 | 42264 |
| SmolLM2 135M Q8_0 | prefill tok/s | 10489.8 | 34835 |
| Llama 3.1 8B Q4_K_M | host peak MiB at ctx 2048 | 7029 | 5046 |

Every item lands with a run of `scripts/bench.py` on gpc against all three models and all three engines, and the result goes into [bench.md](bench.md) in the same pull request. bench.md is a version stale as of this writing, which is how the 109x number survived long enough to shape a milestone, and refreshing it is the first commit rather than the last.
