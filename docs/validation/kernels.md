# Quantized matmul from max/kernels on two GPUs

The M0 kernel spike. Issue #5 asks whether `max/kernels` is usable directly from our own Mojo code, on an Apple GPU and an NVIDIA GPU, without the MAX runtime, and it says the answer decides D6. The answer is no, for a reason that is not the one anyone expected, and the licensing argument in the README was already wrong before this spike started. All of that is below with the evidence, along with the numerics, because "you need a proprietary runtime" is a different claim from "the kernels do not work" and both matter to the decision in issue #7.

The code is in [spikes/qmatmul](../../spikes/qmatmul).

## The short version

`max/kernels` is Apache-2.0 source that does not build or run without `max-core`, which is `LicenseRef-Modular-Proprietary`. This is not a GPU thing. The CPU only K quant matmul needs it too.

Separately, and worse for the README: the `molla` binary we already ship links a library that comes out of the proprietary `mojo-compiler` package. That has been true since the first commit and has nothing to do with `max/kernels`.

The kernels themselves work. A Q4_K matmul from `max/kernels` matches a NumPy reference to 1.8e-7 of peak on both architectures we have. An int4 GEMM from `max/kernels` matches to two bfloat16 steps on the 4090. Neither of those is on an Apple GPU, because `max/kernels` has no quantized matmul that will launch on one below an M5.

## What the licence actually is

The starting point was that `from gpu.host import DeviceContext` does not compile. The `mojo` conda package ships exactly one thing in `lib/mojo`, which is `std.mojoc`. There is no `gpu` package in it.

The source does exist and it is Apache-2.0 with LLVM exceptions. In the open repository at commit `60c394f` it is `max/mojo/max/gpu/host/device_context.mojo`, and the first line of comment in it says "Implementation of the C++ backed DeviceContext in Mojo". It makes 106 `external_call` invocations into symbols named `AsyncRT_DeviceContext_*`. The Mojo is open. The C++ underneath it is not shipped anywhere as source, and the open `AsyncRT` directory in that repository is the CPU async runtime with no device context in it at all.

Building a program that constructs a `DeviceContext` and reading its undefined symbols gives `_AsyncRT_DeviceContext_create`, `_AsyncRT_DeviceContext_deviceApi`, `_AsyncRT_DeviceContext_deviceName`, `_AsyncRT_DeviceContext_release` and `_AsyncRT_DeviceContext_strfree`. They resolve out of `libAsyncRTMojoBindings.dylib`. That file ships in one package:

| Package | Version | Licence | What it supplies here |
| --- | --- | --- | --- |
| `max-core` | 26.5.0 | LicenseRef-Modular-Proprietary | `libAsyncRTMojoBindings`, `libMGPRT`, `libmax`, and the precompiled `quantization`, `linalg`, `layout` and `max` packages |
| `mojo-compiler` | 1.0.0 | LicenseRef-Modular-Proprietary | `libKGENCompilerRTShared`, `libMSupportGlobals`, `libAsyncRTRuntimeGlobals`, and `std.mojoc` |

So vendoring `max/kernels` at a pinned commit, which is what the issue asked for, does not help. The thing that is missing is not the Mojo.

### It is not only the GPU kernels

This was the part worth checking rather than assuming. 272 of the 519 Mojo source files under `max/kernels/src` import `gpu.host`, and the split is not GPU code on one side and CPU code on the other. `quantization/qmatmul_k.mojo` is the K quant matmul. Its own module docstring says "Provides CPU kernels for K-quant block-wise quantized matrix multiplication". Its imports include `max.gpu.host.DeviceContext`, `max.algorithm.sync_parallelize` and `max.runtime.asyncrt.parallelism_level`, and `asyncrt.mojo` calls `MLRT_TaskIdForDevice`. A CPU matmul in that tree reaches the closed runtime through the thread pool it parallelises over.

There is no configuration of this where you take the kernels and leave the runtime.

### The part that was already true

`otool -L build/molla` on the binary this repository builds today lists `@rpath/libKGENCompilerRTShared.dylib`. That file is in the `mojo-compiler` package, which is `LicenseRef-Modular-Proprietary`. molla has linked a proprietary library into every binary it has ever produced.

The README says, in "Why bother", that the gap molla fills is "a runner with no proprietary dependency anywhere in the stack", and lists "No required proprietary dependency" as the fourth of five things we commit to and test in CI. It also says "Mojo's compiler and standard library are Apache-2.0". The standard library is. The compiler is not, and neither is the runtime support library it links into the output.

None of that is news to the rest of the repository. The changelog for 0.0.1 and 0.0.2 both say the compiler is proprietary and that releases are source only because of it. The README was written before any of it was checked and was never brought back into line. So this is a correction rather than a discovery, and it is not caused by `max/kernels`. It is caused by choosing Mojo. Fixing the sentence is in this change. Deciding what to do about it is issue #7.

## What we ran

Weights come out of a real model rather than from a random number generator, because a random Q4_K block has scales and mins that never occur in practice and a wrong nibble order can still look plausible against one. `blk.11.ffn_down.weight` from `qwen2.5-0.5b-instruct-q4_k_m.gguf` is Q4_K, 896 by 4864, 2.4 MB of quantized bytes. It is the same file the GGUF spike used.

Two references, because the kernels do not all compute the same thing.

`ref_float` dequantizes the weights the way llama.cpp does and multiplies in float64 against float32 activations. That is the number a caller wants to be true.

`ref_exact` first quantizes the activations to int8 with one scale per 256 element super block per row, rounding half to even, which is what `matmul_Q4_K` does internally before it multiplies anything. It is the arithmetic that kernel actually performs.

Every result is reported against both. The one it is gated on depends on what it computes.

| Tolerance | Value | Why |
| --- | --- | --- |
| float32 output, gated against its own reference | 1e-5 of the peak magnitude in the reference | Float32 accumulation over 4864 terms costs a few times 1e-7, so this leaves two orders of magnitude of headroom and still catches a wrong scale, a swapped nibble or a mis-ordered group |
| bfloat16 output | 4 bfloat16 steps at the peak magnitude | Eight mantissa bits means one step is 1/256 of the exponent range, and a kernel that accumulates in float32 and rounds once at the end lands within a couple of steps of a float64 reference |

Both are stated against the peak magnitude of the output and not per element. A dot product over thousands of terms lands near zero often enough that a per element relative error is meaningless there. This is what D7 means by tolerances stated per dtype, and it is worth saying that D7 promised them without stating any, so these are the first two.

## Results

| Kernel | Where | Gate | Max abs error | As a fraction of peak |
| --- | --- | --- | --- | --- |
| `matmul_Q4_K`, max/kernels | M4, NEON | ref_exact | 8.34e-07 | 1.55e-07 |
| `matmul_Q4_K`, max/kernels | gpc, x86_64 AVX2 | ref_exact | 9.54e-07 | 1.77e-07 |
| `matmul_Q4_K`, max/kernels | server1, x86_64 AVX2 | ref_exact | 9.54e-07 | 1.77e-07 |
| `matmul_Q4_K`, max/kernels | server2, x86_64 AVX2 | ref_exact | 9.54e-07 | 1.77e-07 |
| `matmul_Q4_K`, max/kernels | server3, x86_64 AVX2 | ref_exact | 9.54e-07 | 1.77e-07 |
| ours, one source | M4, Metal | ref_float | 6.68e-06 | 1.24e-06 |
| ours, one source | 4090, sm_89 | ref_float | 6.68e-06 | 1.24e-06 |
| `matmul_gpu_qint4`, max/kernels | 4090, sm_89 | bfloat16 reference | 0.03125 | 2.00 bfloat16 steps |

The three x86 machines produce byte identical output to each other and differ from the M4 in the last place, which is what two different SIMD widths summing the same integers in a different order looks like.

The two GPU rows are the same 6.68e-06 because the Metal output and the CUDA output are byte for byte identical. Same source, two vendors, two completely different compilers behind it, `cmp` says nothing differs. That is a stronger result than the tolerance was written to accept and it is worth writing down, because D7 is a claim about exactly this and this is the first evidence for it.

`matmul_gpu_qint4` needed its own fixture. It takes Q4_0 rather than Q4_K, wants bfloat16 in and out, and needs the weights run through a second repack kernel first, so the same weight matrix was dequantized and requantized to Q4_0 and the activations rounded to bfloat16. It also needs K to be a multiple of 1024, which the 4864 wide tensor is not. The failure when K is wrong is a constraint error four frames down in `layout_tensor.mojo` about depth-1 layouts and nested shapes, and it never mentions K. That cost more time than it should have and it is the kind of thing that will cost it again.

## The Apple GPU has nothing to use

This is the finding that matters most for the roadmap.

`max/kernels` has no quantized matmul that runs on an Apple GPU below an M5. The Apple GPU matmuls live in `max/kernels/src/linalg/matmul/gpu/apple`, ten files of them, and the quantized ones are NVFP4 weight only and int8 W8A8, neither of which is a GGUF quantization. Both of them start with `_require_apple_m5`, which raises unless `compute_capability` is 5.

Asked directly on the M4, with `max-core` 26.5.0 rather than by reading the source:

```
api metal name Apple M4 cc 4
refused: Apple int8 W8A8 matmul requires Apple M5 (compute_capability == 5); got compute_capability=4
```

So for a GGUF model on an Apple GPU, `max/kernels` supplies nothing at all. Not a slow path, not an unoptimized path. Nothing.

That is why the middle two rows of the results table are our own kernel. It is about eighty lines, one thread per output element, no shared memory, no tiling, weights dequantized on the fly straight out of the GGUF block layout. It is not fast and it was not written to be. It exists because there was no other way to put a number in the Metal row, and having written it, the byte identical result across two vendors is the useful thing that came out of it.

## What this says about D7

D7 says every operation has "a single Mojo source compiled for CPU, NVIDIA, AMD, and Apple GPU", with divergence living "in tile parameters and small compile time branches inside one function, never in separate files".

`max/kernels` does not work that way for quantized matmul. There are three separate implementations in three separate places: `quantization/qmatmul_k.mojo` for CPU K quants, `quantization/qmatmul_gpu.mojo` for NVIDIA int4 with a repack step and a tensor core GEMM, and `linalg/matmul/gpu/apple/*` for Apple with different quantization formats again. Different entry points, different weight layouts, different dtypes.

So D7 is not a description of something we inherit. If molla wants it, molla writes it. The one encouraging piece of evidence is that the naive kernel here did exactly what D7 describes, with no compile time branches at all, and produced identical bits on Metal and CUDA. The claim is achievable. It is just work rather than a property of the dependency.

## The fleet

Per D8. The CPU kernel was run everywhere, the GPU kernels on the two machines that have one.

| Machine | What ran | Result |
| --- | --- | --- |
| M4, aarch64-apple-darwin | `matmul_Q4_K` on CPU, our kernel on Metal, the Apple M5 probe | both matmuls pass, probe refuses as expected |
| gpc, i9-13900K on WSL2, x86_64 | `matmul_Q4_K` on CPU, our kernel and `matmul_gpu_qint4` on the 4090 | all three pass |
| server1, EPYC, x86_64 | `matmul_Q4_K` on CPU | pass |
| server2, EPYC, x86_64 | `matmul_Q4_K` on CPU | pass |
| server3, EPYC, x86_64 | `matmul_Q4_K` on CPU | pass |

Two device classes with a GPU among them, which is what D8 asks for, and four in total.

No timings are quoted anywhere in this document. The CPU kernel is a tuned production kernel, our GPU kernel is deliberately the least optimized thing that could produce a correct answer, and the NVIDIA kernel ran on a shape chosen to make it compile rather than to make it fast. Putting those three in a table would invite a comparison that none of them supports.

## What is not covered

Only two quantization formats, Q4_K and Q4_0, and only matmul. Attention, the paged KV cache, MoE routing and collectives are all in `max/kernels` and none of them was touched. The licensing finding applies to all of them equally, since it is about the package rather than about any kernel, but the numerics finding does not generalize and nothing here says those kernels are correct.

No AMD. We have no AMD hardware, which D8 already records as a real gap.

No M5, so the Apple quantized matmuls in `max/kernels` are untested rather than known bad. What is known is that they refuse to run on the newest Apple GPU we have.

Nothing was measured for speed, for the reason above.

The source reading was done against the open repository at commit `60c394f`, which is slightly ahead of the `max-core` 26.5.0 binaries everything was run against. Every structural claim that could be checked at runtime was checked at runtime on 26.5.0, including the M5 refusal, which is why that section quotes the program output rather than the source.

## What this says about D6 and what happens next

D6 says "The MAX runtime is an optional backend and never required" and "Owning it is also what keeps molla clear of the Modular Community License". The first half is false for anything that calls `max/kernels`, on any target, including CPU. The second half is about a licence that is not the one in play: the packages are `LicenseRef-Modular-Proprietary`.

Three options exist and this document is not the place to pick one.

The first is to accept the dependency. molla already ships a proprietary runtime library from `mojo-compiler`, so `max-core` is a difference of degree rather than of kind, and the README stops claiming otherwise.

The second is to write our own kernels and depend only on `mojo-compiler`. The spike shows the GPU half of that is possible and gives a rough idea of the cost, which is a real kernel per operation per target class rather than one portable source, because tiling and tensor core use do not survive being written once.

The third is that D6 does not survive at all.

Choosing is issue #7. What this spike owed it was the facts, and the facts are that the split the README is built on does not exist.
