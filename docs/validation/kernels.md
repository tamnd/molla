# Quantized matmul on two GPUs

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

```text
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

## The matvec molla ships

Everything above is the M0 spike, which answered a licensing question and threw its kernel away. This section is the kernel that stayed, `molla.nn.gpu.planar_matvec_kernel`, added for issue #141. It is the same claim in a narrower form: the same weights give the same answer on a device as they do on a host, on both vendors, over every quantization molla reads.

It is a matvec and not a matmul, because decode is one token at a time and a matmul with a batch of one is a matvec with wasted machinery around it. Prefill wants the matmul and does not have it yet.

### What it computes and what it reads

One thread block per output row, `TILE` threads in the block, each walking the row with a stride of `TILE`, then a tree reduction through shared memory. `TILE` is 128.

It reads the planar layout from `molla.nn.repack` and refuses anything still in ggml blocks. That is the whole reason the planar layout exists. A ggml block is a header, a bit plane and a nibble order, and unpacking one is a dozen dependent integer operations before a single multiply, with each thread pulling bytes out of a header the thread beside it also needs. Planar is one signed byte per value with the scales in planes at the end of the row, so thread `t` reads byte `t` and the hardware coalesces the row into as few transactions as it has lanes.

The group size and whether the type carries a minimum plane are compile time parameters, so there are three instantiations of one function rather than three functions, and there are no target specific branches inside it at all. That is what D7 asks for, and the results below are the second piece of evidence that it is achievable.

### How it is checked

`scripts/kernel_oracle.mojo`, run by `pixi run conformance-kernels`, reads the same fixtures `pixi run conformance-quant` does and multiplies each of them by a fixed vector three ways.

| Path | What it is |
| --- | --- |
| ggml on the host | `molla.nn.kernel.matvec` reading the block bytes the fixture holds |
| planar on the host | the same, after `molla.nn.repack` has rewritten those bytes |
| planar on the device | `molla.nn.gpu.device_matvec` reading a copy of the planar bytes in a device pool |

Three rather than two so that a failure comes with a suspect. A disagreement between the first and the second is the repack, which is host arithmetic that never goes near a GPU. A disagreement between the second and the third is the kernel. Reporting one number for both would mean every device failure arrived with two possible causes and no way to separate them.

The activation vector is signed and never zero. That matters more than it looks like it should. A kernel that dropped the sign of the quants, or read the minimum plane with the wrong sign, still agrees with the host to several digits when every input is positive, because the errors cancel across a row.

The fixtures are viewed as two blocks per row, so a row stride bug has somewhere to show itself, and 64 or 512 columns against a tile of 128 means both the case where a thread walks the row more than once and the case where most threads contribute nothing are exercised rather than assumed.

### The tolerance

| Tolerance | Value | Why |
| --- | --- | --- |
| float32 output, ggml on the host against planar on the device | 1e-5 of the peak magnitude in the host reference | Float32 accumulation over a few hundred terms costs a few times 1e-7 whatever order it happens in, so this leaves two orders of magnitude of headroom for a reduction tree that sums the row in a different order from the host loop |

The same gate the spike used for a float32 output, for the same reason, stated against the peak magnitude of the output rather than per element because a dot product over hundreds of terms lands near zero often enough that a per element relative error is meaningless there.

It is loose in one direction and tight in the other. Two orders of magnitude of headroom sounds generous, and it is nowhere near enough to hide any of the failures this is looking for. A swapped nibble, a scale read from the wrong group and a row stride that is off by a row all move a result by a recognisable fraction of the whole. The one that actually happened during development moved four of the eight types by between 1.5 and 28.6 times the peak.

### Results

Both machines, every format, as a fraction of the peak magnitude in the host reference.

| Format | Shape | M4, Metal | 4090, sm_89 |
| --- | --- | --- | --- |
| q4_0 | 256 by 64 | 2.21e-07 | 2.21e-07 |
| q4_1 | 256 by 64 | 1.61e-07 | 1.61e-07 |
| q5_0 | 256 by 64 | 8.24e-08 | 8.24e-08 |
| q5_1 | 256 by 64 | 2.06e-07 | 2.06e-07 |
| q8_0 | 256 by 64 | 1.07e-07 | 1.07e-07 |
| q4_k | 256 by 512 | 2.40e-07 | 2.40e-07 |
| q4_k, from a real model | 256 by 512 | 3.16e-07 | 3.16e-07 |
| q5_k | 256 by 512 | 1.49e-07 | 1.49e-07 |
| q6_k | 256 by 512 | 2.21e-07 | 2.21e-07 |
| q6_k, from a real model | 256 by 512 | 1.69e-07 | 1.69e-07 |

Worst case 3.16e-07, which is a factor of thirty inside the gate.

The two columns are not merely close. Every figure the oracle prints on the two machines is identical, including the raw absolute errors and the repack and kernel columns that are not in this table. Same source, two vendors, two entirely different compilers and two entirely different memory systems, and the output does not differ anywhere it can be measured. The spike saw the same thing on a much simpler kernel; this one has shared memory, a reduction tree and three compile time instantiations in it and still does.

That is worth being slightly suspicious of, so it is worth saying what would break it. The reduction is a fixed tree over a fixed number of threads, so the association is the same on both machines, and float32 addition is deterministic once the association is fixed. There is no fast math, no fused multiply add that fires on one target and not the other in a way that changes the result, and no atomic. Identical output is what this kernel should produce, and if a change ever makes the two columns diverge, the divergence is the finding.

The M4 run reports `built for metal:4 running on metal Apple M4` and the 4090 run reports `built for nvidia:sm_89 running on cuda NVIDIA GeForce RTX 4090`, both against `max-core` 26.5.0.

### The fleet

Per D8.

| Machine | What ran | Result |
| --- | --- | --- |
| M4, aarch64-apple-darwin | the full suite, and the oracle on Metal | suite green, ten formats inside tolerance |
| gpc, i9-13900K on WSL2, x86_64 | the full suite, and the oracle on the 4090 | ten formats inside tolerance, suite green apart from issue #87, which fails the same way on main |
| server1, EPYC, x86_64 | the full suite | green, the matvec compiled out and the refusals ran |

Three of the five, which is every machine that can build this at all. server2 and server3 have no SDK on them.

The three machines with no accelerator are not idle here. `check_matvec` is deliberately separate from the launch so that everything a device matvec refuses is testable without a device, and `tests/test_gpu.mojo` runs those refusals everywhere. The one that matters is the residency check, and the reason it matters is below.

### A device kernel handed a host address reads zeros

Found while getting the first launch to work, and it is the reason `check_matvec` refuses a weight that is not in a device pool rather than trusting the caller.

On the M4, through `max-core` 26.5.0, a Metal kernel given a plain host allocation does not fault. It reads zeros. Every row of the output comes back as exactly zero and nothing anywhere reports an error. Wrapping the host memory in a `DeviceBuffer` with `owning=False` does not change it: the pointer value is the same and it still reads zeros.

The same probe found the other half of it. A device buffer's `unsafe_ptr()` is a device address, 1099512152064 in the run that was recorded, and reading it from the host segfaults. `map_to_host()` on the same buffer returns 4470079744, a real host address for the same bytes, with no copy. So unified memory on this machine is one physical pool with two different addresses into it, and not one address that both sides can use.

That matters beyond this kernel, because `WHERE_UNIFIED` in `molla.nn.tensor` was written on the assumption that a mapped file already is a device visible buffer and a device kernel can read it where it lies. It cannot. That is a real bug in a claim this repository has already published and it gets its own issue rather than being quietly fixed here.

For this kernel the consequence is narrower and the fix is a refusal. A model bound to host memory and multiplied on a device would run at full speed and answer with noise, which is the single worst failure mode available to an inference engine, so the check is in the one place every launch goes through and it names what would have happened.

### What is not covered

Only matvec. Attention, the softmax, the norms and the elementwise operations are all still host only, and putting the rest of a block on the device is issue #142, which is the section below.

No AMD, which D8 already records as a gap, and no arm64 Linux GPU, which nothing in the fleet has.

Nothing here is a speed measurement. Every call in the oracle uploads its activations and downloads its result, which is most of what a call costs at these shapes, and the weights are a few hundred kilobytes rather than a few gigabytes, so the memory system this kernel was designed around is not being exercised at all. Timings arrive when the residual stream stops making the trip, which is issue #143, and they go in their own document.

The corpus is the quantization corpus, so the shapes are small and rectangular and chosen to exercise the bit unpacking rather than the hardware. A 4096 by 4096 weight has not been through this kernel.

## The rest of a block

Issue #142, `molla.nn.gpu_ops`. The matvec above was one operation. This is everything else a transformer block does: the norm, rope, attention, the two gated feed forward shapes, the plain activations, the residual add, the scale and the argmax at the end of a decode step.

They went across together rather than one at a time, and that is the design decision worth recording. At decode shapes each of these operations is tiny. A norm over 4096 values is 4096 multiplies. What is not tiny is the trip. Moving the residual stream to the host and back once costs more than every small operation in the block put together, so a block with one host operation left in the middle of it is a block that pays the full round trip anyway and gets nothing for the kernels that did go across. The set is the unit, not the operation.

### What is in it

| Kernel | Shape of the launch |
| --- | --- |
| `rms_norm_kernel` | one block, a tree reduction over the row, then a scale |
| `softmax_kernel` | one block, a max tree then a sum tree over the same shared allocation |
| `swiglu_kernel` | elementwise over the gate and the up projection, silu or gelu by compile time parameter |
| `act_kernel` | elementwise, the same activation without a gate |
| `add_into_kernel`, `scale_into_kernel` | elementwise, capped at 256 blocks |
| `argmax_kernel` | one block, a tree carrying value and index together |
| `rope_kernel` | one thread per pair, neox or adjacent by compile time parameter |
| `attend_kernel` | one block per query head, three phases in one launch |

`attend_kernel` is the one that earns the module. It scores, softmaxes and takes the weighted sum without leaving the kernel, so a head never writes its scores anywhere the host can see them. Masked positions get the lowest finite float32 rather than an actual negative infinity, so that subtracting the row maximum can never produce a nan, and the visibility test is written term for term against `AttnSpec.visible` so that sliding windows and attention sinks mean the same thing on both sides.

There are no target specific branches in any of them, per D7. Where a target constant would have been the natural thing to reach for, it is a launch parameter instead: the reductions are trees over a tile rather than warp shuffles, and the tile is a parameter.

### How it is checked

Twice, in two places, for two different audiences.

`tests/test_gpu_ops.mojo` runs in the suite and asserts. It compares each device operation against the host function that already has tests of its own, which is the only comparison worth making here. A device softmax that is internally consistent and disagrees with the host one by a thousandth is a model that answers differently depending on which backend it was started with, and nobody would ever trace that back to a reduction order. The whole file is one compile time branch, so it is a skip rather than a failure on the three machines with no GPU.

`scripts/block_oracle.mojo`, run by `pixi run conformance-block`, prints the distance for every operation whether or not it passed. That is what a document recording per target numerics needs and it is what a test cannot give, because the only figure a test ever reports is one that already failed. It needs no fixtures at all, unlike every other conformance task in `pixi.toml`, so a machine with a GPU can run it straight after a clone.

Inputs are a fixed wave with both signs and a spread of magnitudes, not random. A ramp would pass a norm that had its scale wrong by a constant, since every element would be off the same way. Random would give a test that fails one run in fifty, which is a test people learn to rerun.

### The tolerances

Four gates rather than one, because the reasons differ and a single number across all of them would be too loose to catch a real fault in the tight ones or would fail the loose ones for behaving as designed.

| Gate | Value | Applies to | Why |
| --- | --- | --- | --- |
| tight | 1e-6 | activations, add, scale | every element touched once in an order that does not matter, so the two sides differ only where the host used a float64 intermediate |
| wide | 2e-6 | the norm, attention | both reduce over a few hundred terms and the host accumulates in float64, which Metal has not got |
| angled | 2e-6 | rope | the host takes its `cos` and `sin` in float64 and the device takes them in float32 |
| capped | 1e-5 | attention with a logit softcap | the cap multiplies its own `tanh` by 50, so a difference in the last digit of one arrives at the softmax fifty times larger |

Everything is measured as a fraction of the peak magnitude of the reference, except the softmax, which is measured absolutely. Relative to peak is the wrong question for a probability vector and it gets the answer backwards: a distribution over two thousand entries has a peak near a thousandth, so dividing by it inflates the difference most for the flattest distributions, which are the easiest case rather than the hardest. A probability is its own scale.

The softcap gate is the one number here set by a constant in a model file rather than by the arithmetic, and a model with a cap larger than Gemma 2's 50 would want a larger number.

### Results

Both machines, every case the oracle runs, as a fraction of the peak magnitude in the host reference. The softmax rows are absolute.

| Case | M4, Metal | 4090, sm_89 |
| --- | --- | --- |
| rms_norm 256 | 9.09e-08 | 9.09e-08 |
| rms_norm 512 | 9.04e-08 | 0.00 |
| rms_norm 4096 | 9.04e-08 | 9.04e-08 |
| softmax 64 | 1.19e-07 | 1.19e-07 |
| softmax 300 | 5.96e-08 | 1.49e-08 |
| softmax 2048 | 2.01e-07 | 1.94e-07 |
| swiglu | 5.09e-08 | 1.27e-08 |
| geglu | 4.78e-09 | 3.18e-09 |
| silu | 6.88e-08 | 3.44e-08 |
| gelu | 1.72e-08 | 1.29e-08 |
| add_into | 0.00 | 0.00 |
| scale_into | 0.00 | 0.00 |
| argmax over a vocabulary | index 99991, same as the host | index 99991, same as the host |
| rope llama 3 | 1.06e-07 | 2.45e-07 |
| rope llama 3 far out | 1.66e-07 | 2.63e-07 |
| rope adjacent pairs | 1.21e-07 | 1.81e-07 |
| rope yarn | 1.36e-07 | 2.73e-07 |
| rope gemma 3 local | 1.07e-07 | 1.88e-07 |
| rope gemma 3 global | 1.07e-07 | 2.14e-07 |
| attention grouped query | 1.21e-06 | 1.15e-06 |
| attention multi head | 6.59e-07 | 5.49e-07 |
| attention sliding window | 6.57e-07 | 5.56e-07 |
| attention window with sinks | 6.52e-07 | 5.59e-07 |
| attention softcapped | 1.80e-06 | 1.91e-06 |

Worst case is the softcapped attention at 1.91e-06 against its own 1e-5 gate, and everything under the tight and angled gates is at least four times inside.

The two columns are not identical here, which is the interesting difference from the matvec above. That kernel produces the same bits on both vendors and this one does not, and the reason is that this one calls transcendental functions. `exp`, `cos`, `sin` and the reciprocal square root are library functions in each vendor's own runtime, correct to within a bit or so each and not to the same bit. Everything in the table that is exactly equal on both machines, and everything that is exactly zero, is a kernel that does not call one. That is the expected shape of the result rather than a worry, and a divergence appearing in one of the exact rows would be the finding.

### What the 4090 found that the M4 could not

Three corrections came out of running this on the second GPU, and all three are in the kernel rather than in a tolerance. This is the section that justifies the fleet.

The first is rope on a long context. CUDA's float32 `cos` and `sin` reduce a large argument badly, and the angle at the fastest rotating pair is the position itself, so position 4096 is 652 whole turns and CUDA was 4e-4 of peak out where Metal was within 1e-7 of the float64 answer at the same point. The kernel now does its own Cody-Waite reduction before the call, splitting two pi across three float32 constants, which takes the 4090 to 2.6e-7 and leaves Metal where it was.

The second is the softcap. CUDA's float32 `tanh` is further from the float64 one than Metal's, and Gemma 2 multiplies it by 50, so it reached the softmax at 2.04e-5 of peak. Computing the same function through `exp` instead, arranged so the exponential always lands in `(0, 1]` rather than overflowing on a large score, took it to 1.9e-6.

The third was found by the oracle rather than by the second machine, and it is the one worth remembering. Rope failed on the M4 at 5.99e-6 against a 5e-6 gate, and the obvious fix was to widen the gate, which is what happened first. The oracle then showed the error growing with position: 4e-6 of peak at position 137 and 1.5e-4 at position 4096. The device was forming the rope frequency step from a float32 exponential where the host used a float64 one, about 1e-7 apart, and the angle multiplies that difference by the position. A tolerance wide enough for position 4096 would have covered a real fault at every position, and it would have gone on being not quite wide enough as contexts got longer. `molla.nn.rope.step_table` builds the steps on the host once per rope spec and uploads them, which is why the position 4096 row above now reports the same figure as the position 137 one.

The general point is that a tolerance that has to grow with an input is not a tolerance, it is an unfixed bug with a number in front of it.

### A CUDA process gets one device context

Found by running the suite on the 4090, and it is the reason the fleet earns its keep on something other than numerics.

The suite hung. Not failed, hung: the GPU at nought per cent, 141 seconds of CPU over 74 minutes, and all 35 threads asleep on a futex. It stopped at the first device norm, which reads exactly like a kernel that will not finish, and it is not one.

`tests/test_gpu.mojo` made a `DeviceContext` for the matvec and `tests/test_gpu_ops.mojo` made another for the block. On CUDA the second one is constructed without complaint and then hangs on the first buffer allocated against it, even though the first context went out of scope before it. Each module passes on its own, which is what made this slow to see. Metal does not mind at all, so the M4 ran two contexts in one process for as long as there were two modules to run them.

The fix is that `tests/main.mojo` makes the one context and lends it to both modules, and no test module makes its own. That is also the shape the engine wants, since a process that made a context per layer would hang on the second layer, and #143 binds against one context for the same reason.

What is worth carrying forward is the failure mode rather than the rule. A hang with an idle GPU looks like a kernel bug and is not one, and nothing in it names a context.

### The fleet

Per D8.

| Machine | What ran | Result |
| --- | --- | --- |
| M4, aarch64-apple-darwin | the full suite, and the oracle on Metal | suite green at 2424 checks, all twenty five cases inside tolerance |
| gpc, i9-13900K on WSL2, x86_64 | the full suite, and the oracle on the 4090 | all twenty five cases inside tolerance, suite green apart from issue #87, which fails the same way on main |
| server1, EPYC, x86_64 | the full suite | green, the device code compiled out and the refusals ran |

The refusals are the part that runs everywhere. Each host entry point checks its shapes and its residency before it reaches the compile time branch that holds the launch, so a machine with no GPU still tests that a mismatched length, a zero length and a weight outside a device pool are all rejected by name.

### What is not covered

No speed measurement, for the same reason as the matvec. Every case here uploads its input and downloads its result, which at these shapes is most of what the call costs, so the oracle measures agreement and nothing else. Timings arrive when the residual stream stops making the trip, which is issue #143.

The block is not yet wired into a forward pass. These are the pieces and they agree with the host one at a time. A whole layer running end to end on the device, with the KV cache in device memory, is #143 as well.

No prefill. Every shape here is a decode shape, one token at a time, and attention is one block per query head, which is the right split for a single query and the wrong one for a few hundred.

No AMD, and no arm64 Linux GPU, both of which D8 already records as gaps.
