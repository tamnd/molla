# Engines

molla has been tuned against its own previous numbers for three milestones. This document is what came out of stopping that and reading the two engines it is measured against instead, on the grounds that a kernel which is eight times slower than another kernel doing the same arithmetic is not a tuning problem.

The reading was done against `llama.cpp` at commit `f9f09f02cc44` and `ollama` at commit `b68365a`, both cloned fresh rather than remembered. Every claim below has a file and a line behind it. The long form notes are outside this repository; what is here is what molla is going to act on.

## The measurement this starts from

[bench.md](bench.md) was a full version stale when this began, and it was the source of the belief that prefill was more than a hundred times off. It is not. These are gpc, RTX 4090, molla 0.4.5, llama.cpp `de8656b`, ollama 0.32.9, on 2026-09-04, 514 prompt tokens and 128 generated, best of five.

SmolLM2 135M Q8_0:

| engine | prefill tok/s | decode tok/s | ttft ms |
| --- | --- | --- | --- |
| molla | 10489.8 | 247.6 | 49 |
| llama.cpp | 32487.0 | 886.8 | 16 |
| ollama | 15667.9 | 684.4 | 33 |

Qwen 2.5 0.5B Q4_K_M:

| engine | prefill tok/s | decode tok/s | ttft ms |
| --- | --- | --- | --- |
| molla | 5039.2 | 288.3 | 102 |
| llama.cpp | 43338.7 | 772.6 | 12 |
| ollama | 7320.8 | 495.1 | 70 |

Llama 3.1 8B Q4_K_M, 515 prompt tokens, best of three:

| engine | prefill tok/s | decode tok/s | ttft ms |
| --- | --- | --- | --- |
| molla | 378.7 | 85.8 | 1360 |
| llama.cpp | 10548.6 | 161.6 | 49 |
| ollama | 5919.2 | 148.1 | 87 |

Peak memory is left out of those tables because the column the harness had was the host resident set, which on a card is close to meaningless and is not comparable between two engines that hold the model different ways. `scripts/bench.py` now samples the card as well, and on that column molla holds 656 MiB against llama.cpp's 694 on SmolLM2, 1110 against 1120 on Qwen, and 7404 against 5198 on the 8B. So memory is at parity or better on the two small models and 1.42 times off on the 8B, which is a much smaller problem than this repository has been treating it as, and the 8B gap has two named causes in the repack size and the f32 KV cache.

So the gaps are 3.1x, 8.6x and 27.9x on prefill and 3.6x, 2.7x and 1.9x on decode. Decode is a constant factor and prefill is not. A ratio that grows monotonically with model size is a scaling failure, and it is the only number in the set that behaves that way.

## Why prefill scales wrong

`planar_matmul_kernel` in [src/molla/nn/gpu.mojo](../../src/molla/nn/gpu.mojo) reads a weight value once and multiplies it into `SPAN * MM_GROUPS` accumulators, which is sixty four tokens at the shipped constants. That amortizes the weight read and the dequantization, and it is what took prefill from a decode per prompt token to where it is now.

It does nothing at all for the activation. The grid is `(ceil(tokens / (SPAN * MM_GROUPS)), rows)`, so there are `rows` blocks and each of them reads the whole `T` by `cols` activation tile from global memory:

```mojo
for k in range(SPAN):
    acc[k] += d * x[unsafe_offset=at0 + k * cols + i]
```

Activation traffic for one matmul is `rows * tokens * cols * 4` bytes against `rows * cols` bytes of weight, a ratio of about `4 * tokens` to one. The activation is the dominant term and it is multiplied by the output row count, which is exactly the thing that grows when the model grows. 3.1x to 8.6x to 27.9x is that term becoming visible.

There is no shared memory in the kernel and there are no matrix cores. Both operands come from global memory on every block. That is the defect, it is structural, and no constant in the file moves it.

## What staging alone is worth, which is nearly nothing

The paragraph above was written before the tile was built, and it reads as though staging the operands is the fix and the instruction the tile multiplies with is a detail. That is wrong, and it is worth keeping the correction next to the diagnosis rather than in a commit message.

`planar_gemm_kernel` was written and measured on gpc: a 64 row by 64 token output tile, a K step of 32, both operands staged in threadgroup memory, the weight dequantized on the way in with the minimum folded into the staged value, an eight by four accumulator block per thread, and fp32 fused multiply adds in the loop. It is the arrangement described above with no matrix cores in it.

| model | prefill before | prefill with the tile |
| --- | --- | --- |
| SmolLM2 135M Q8_0 | 10489.8 | 2274.3 |
| Qwen 2.5 0.5B Q4_K_M | 5039.2 | 1647.4 |
| Llama 3.1 8B Q4_K_M | 378.7 | 420.4 |

Eleven per cent on the model the change was aimed at, and a factor of three to five backwards on the two smaller ones.

The arithmetic says why, and it says it without profiling anything. Two flops a parameter puts llama.cpp's 8B prefill of 10548.6 tokens a second at 169 TFLOP/s. An RTX 4090 does 82.6 TFLOP/s of fp32 fused multiply add. llama.cpp is running at twice what the card can do in fp32, which is only reachable on the tensor cores, so no arrangement of fp32 multiplies gets there and the ones that get closest are still four times short. molla with the tile is at 6.7 TFLOP/s, which is eight per cent of the fp32 peak, so there is real headroom in the tile and the ceiling above it is still below the target.

The regression on the small models is a second and separate thing. A block covering 64 rows and the whole 64 token chunk means SmolLM2's projections run in nine blocks and its feed forward in twenty four, where the untiled kernel launches one block per output row and gets 576. Trading a factor of sixty four of parallelism for a factor of sixty four of activation traffic is only a win on a matrix with enough rows to keep the card full, and two of the three models do not have one. That is the same trade the fused layer work made and it fails the same way.

So the item is not a tile. It is int8 activations and a matrix core instruction, with a tile around them, and the tile on its own is not worth landing. The Metal half of this is worse than it looked for the same reason, because no matrix core instruction is reachable there at all, and the honest position is that the Metal prefill gap does not close until one is.

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

D7 says to use the dependency where it covers the target. What it covers is not what reading `layout/` first suggested, and the difference decides the shape of both kernels, so this table is what compiled and ran rather than what looked plausible.

| symbol | present | usable here |
| --- | --- | --- |
| `layout.tensor_core.TensorCore` | yes | CUDA and ROCm only, and float only: fp32, half, fp8, fp64, with no int8 case anywhere in it |
| `max.gpu.compute.arch.mma_apple` | yes | Metal simdgroup matrices, and it is in the 26.5.0 molla already pins |
| `layout.tensor_core_async.TensorCoreAsync` | yes | Hopper wgmma, not the 4090 |
| `linalg.matmul.matmul` | yes | dense dtypes, would need the weights dequantized first |
| `quantization.qmatmul_gpu.matmul_gpu_qint4` | yes | NVIDIA only and wants a weight layout that is not molla's |
| `quantization.qmatmul_k.matmul_Q4_K` | yes | CPU only |

Metal is the correction. An earlier draft of this file said no Metal matrix core instruction is reachable from Mojo and that the Metal GEMM would have to use ordinary multiplies. That is wrong. `max.gpu.compute.arch.mma_apple` exports `_mma_apple_8x8`, `_mma_apple_transposable` and fragment loaders and storers for both shapes, and a probe of 8x8x8 with half inputs and a float accumulator ran on this M4 and matched a host reference exactly, in both the f32 and the f16 input forms. That is `simdgroup_float8x8`, which is the instruction `kernel_mul_mm` is built out of. The building block llama.cpp uses on Metal is available today.

What does not run on an M4 is the 16x16x16 form, in either flavour. Both fail at pipeline creation with `Encountered unlowered function call to air.simdgroup_matrix_16x16x16_multiply_accumulate` for half inputs and the `_widening_` variant for int8 inputs. That is the M5 instruction, and MAX's own Apple matmul says as much by gating on `compute_capability == 5`. So on the reference machine the Metal tile is 8x8 fragments with half inputs and a float accumulator, which is exactly what llama.cpp does, and the 16x16 path is something a later Apple machine unlocks rather than something to design around now.

CUDA is the correction in the other direction. `TensorCore` is real and it works, and it has no integer case at all: `supported_fp32`, `supported_half`, `supported_fp8` and `supported_fp64` are the whole list, `get_shapes` asserts out on anything else, and there is no `dp4a` and no `s8.s8.s32` anywhere in the MAX tree. The int8 instruction llama.cpp's MMQ is built on is not reachable through the dependency. It is reachable through `inlined_assembly`, which is how MAX itself reaches the Apple intrinsics, but that is molla writing the mma by hand and it should be a decision taken on a number rather than by default.

The number is this. llama.cpp's 8B prefill of 10548.6 tokens a second is about 169 TFLOP/s at two flops a parameter. A 4090 does 165.2 TFLOP/s of fp16 with an fp32 accumulator and 660.6 TOPS of int8, so llama.cpp is running at roughly a quarter of the int8 peak, which is a normal number for a real kernel. A half precision tile that hits the same fraction of its own peak lands near 43 TFLOP/s, which is 2700 tokens a second on the 8B: six times what molla does now and still four times short of llama.cpp. Half precision through MAX is most of the distance and it is not all of it, and the last four times is the int8 instruction. So the plan is to build the half precision tile on both backends first, measure it, and let that measurement decide whether hand written int8 mma is worth carrying.

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

1. A matrix core GEMM, which was written as one item covering both backends and turned out to be two. On Metal it is `_mma_apple_8x8`, the same eight by eight instruction `kernel_mul_mm` is built on, and it is done: SmolLM2 prefill goes 472 to 2142 tokens a second and Qwen goes 169 to 601, which takes the gap from twenty times to four and a half and from nineteen to five and a half. On CUDA it is `mma` at 16x8x16, the same tile ported, and it loses to the ordinary kernel on two models out of three, so it is out of the milestone and into #212 with the measurements and the diagnosis.
2. The weight unpack folded into the tile load, with the scale and minimum pre multiplied once per tile, and the nibble shifts replaced by masks with the factor corrected in the scale. The tile makes this free and the fp32 version already showed it costs nothing to fold the minimum in.
3. An f16 KV cache, then q8_0 behind a flag.
4. Close the GGUF mapping between the two load passes, which removes the first run double residency.
5. Make the repack size neutral, or skip it on the device path the way llama.cpp does.
6. Answered, and it is no longer a spike. The Metal instruction is reachable, it is `_mma_apple_8x8`, and it is inside the first item. What replaces this line is the question the first item's number decides: whether an int8 mma written with `inlined_assembly` is worth carrying on CUDA to close the last four times. That question is now downstream of #212 rather than of item one, because there is no point hand writing an integer instruction into a tile that loses to the kernel it replaces at half precision.
7. Shape specialized kernel compilation with a pipeline cache.

One through three are one milestone and four and five can run beside them because they touch load and not kernels. Seven is a milestone of its own. They are #201 through #206 and #208, tracked in M2d, #209.

## What this milestone has to reach

The standing goal is two times llama.cpp on tokens a second at half its memory. A 27.9x prefill gap does not become 0.5x in one milestone and saying otherwise would be a plan that cannot fail honestly, so the gate here is parity, measured on gpc against the tables at the top of this file, and the 2x gate stays where it is on #57.

| model | quantity | now | this milestone |
| --- | --- | --- | --- |
| Llama 3.1 8B Q4_K_M | prefill tok/s | 378.7 | 10549 |
| Llama 3.1 8B Q4_K_M | decode tok/s | 85.8 | 162 |
| Qwen 2.5 0.5B Q4_K_M | prefill tok/s | 5039.2 | 43339 |
| SmolLM2 135M Q8_0 | prefill tok/s | 10489.8 | 32487 |
| Llama 3.1 8B Q4_K_M | card MiB | 7404 | 5198 |

Half precision matrix cores were not expected to reach that table on their own. The arithmetic two sections up put the first item near 2700 tokens a second on the 8B, a quarter of the way there and six times where molla is, and said that if the tile landed well below that then the tile was wrong rather than the instruction.

It landed well below. The CUDA tile measures 509 tokens a second on the 8B, about 8.2 TFLOP/s, which is five per cent of the tensor core peak, and it is slower than the ordinary kernel on the two smaller models. So the tile is wrong, that is worth knowing before any assembly gets written, and it is exactly what the prediction was there to find out. #212 has what the numbers say is missing, which is `ld_matrix` for the fragments, a swizzled shared layout and a K loop that stages ahead.

The Metal side is the other half of the same prediction and it went the other way. 472 to 2142 on SmolLM2 and 169 to 601 on Qwen, which is 4.5 and 3.6 times, and the reason the two backends disagree this much is not the kernel. An Apple simdgroup matrix is about ten times its own GPU's scalar rate, measured at 3.04 TFLOP/s against 0.30, where an NVIDIA half precision tensor core is 165.2 against 82.6 for ordinary fused multiply adds, which is two. The same tile buys ten times on one machine and two on the other before anything is spent on feeding it.

The gate stays at parity, because a gate set to what the plan already predicts measures nothing.

Every item lands with a run of `scripts/bench.py` on gpc against all three models and all three engines, and the result goes into [bench.md](bench.md) in the same pull request. bench.md is a version stale as of this writing, which is how the 109x number survived long enough to shape a milestone, and refreshing it is the first commit rather than the last.
