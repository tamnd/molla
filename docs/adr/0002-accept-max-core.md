# 0002: max-core is a required dependency

Status: accepted, 2026-08-31. Decides the D6 gate, issue #78. D6 as written is wrong and this replaces it.

## The decision

molla uses `max/kernels` as designed and accepts `max-core` as a required dependency. Not optional, not a backend, required, at runtime, to generate a token.

D6's claim that "the MAX runtime is an optional backend and never required" is deleted rather than qualified. It was not a policy that got relaxed, it was a description of a seam that does not exist.

The README stops implying molla is free of proprietary dependencies and names the two it has. `mojo-compiler` supplies the toolchain and a support library linked into every binary. `max-core` supplies the runtime that `max/kernels` calls into.

## What was found

Evidence and measurements in [validation/kernels.md](../validation/kernels.md). Three findings, in the order they matter.

`max/kernels` is Apache-2.0 source that does not build or run without `max-core`, which is `LicenseRef-Modular-Proprietary`. This is not a GPU thing. `quantization/qmatmul_k.mojo` is the K quant matmul, its own module docstring calls it a CPU kernel, and it imports `max.gpu.host.DeviceContext`, `max.algorithm.sync_parallelize` and `max.runtime.asyncrt`. A CPU matmul in that tree reaches the closed runtime through the thread pool it parallelises over. There is no configuration where you take the kernels and leave the runtime, so vendoring at a pinned commit, which is what the spike was asked to try, does not help.

The molla binary already links a proprietary runtime support library. `otool -L` on what this repository builds today lists `libKGENCompilerRTShared`, which ships in `mojo-compiler` under the same licence. That has been true since the first commit and has nothing to do with `max/kernels`. It is why 0.0.1, 0.0.2 and 0.0.3 are all source only releases.

The kernels themselves are good. A Q4_K matmul from `max/kernels` matches a NumPy reference to 1.8e-7 of peak on both architectures available. An int4 GEMM matches to two bfloat16 steps on the 4090. Whatever else is true, the math is not the risk.

## Why this and not the alternative

The alternative was to write every kernel ourselves and depend only on `mojo-compiler`. It is not a fantasy. The spike compiled one source to Metal and sm_89 with byte identical output, so the approach works.

It was rejected on cost and on what the cost buys. Writing our own means a real kernel per operation per target class, because tiling and tensor core use do not survive being written once, and it means writing them before anything generates a token. That is the difference between M2 landing and M2 being a research project. molla with a proprietary runtime dependency serves models. molla with a pure licence story and no kernels serves nothing, and nobody is helped by the second one.

The dependency is also a difference of degree rather than of kind. A binary that already links a proprietary support library is not made pure by refusing a second proprietary package. The line that was supposed to be defended had already been crossed before this decision was on the table.

## What this costs, stated plainly

Anybody running molla installs a proprietary runtime. Not to build it, to run it. That is the cost and it is not small, and every part of the project that implied otherwise gets corrected rather than left to be discovered.

The fourth commitment in the README, "no required proprietary dependency", is gone. The four remaining commitments are about formats, portability of data, telemetry and uninstall, and all four survive this and are still tested in CI. What molla can no longer say is that its own stack is free.

Releases stay source only, for the same reason as before and now with a second reason on top.

## What it does not buy

`max/kernels` has no quantized matmul that will launch on an Apple GPU below an M5. On the M4 it raises at launch, and the M4 is the reference machine D1 names.

So this decision does not deliver an Apple GPU path. Every quantized matmul that runs on an M4 GPU is one molla writes, exactly as if the other option had been chosen, for that target. The spike wrote one to get a Metal number at all and it is in `spikes/qmatmul`.

What `max-core` covers is CPU and CUDA, which is enough for M2's acceptance criteria on the 4090 and on Linux CPU. The Apple GPU line in M2 is now a molla kernel and should be planned as one.

## What replaces D6

D6 in `docs/design.md` is rewritten. The engine is still ours: weight loading, layer composition, paged KV cache, continuous batching, sampling, speculative decoding and LoRA. That half of D6 was never in question and it is the half that decides how molla behaves. What changes is the sentence about the runtime being optional, and the sentence about licensing that was the reason for the split.

D7, one kernel source per operation across four targets, is unaffected in principle and now has an obligation attached: the Apple GPU kernels are ours, so D7's per target numerics tests are the only thing standing between "portable" and a claim that decays quietly.

## What would reverse this

`max-core` becoming redistributable, or Modular publishing the runtime under a permissive licence, would remove most of the cost of this decision and none of the benefit, so nothing would need to change except the README.

A licence change in the other direction, or terms that restrict what molla may do with `max/kernels`, forces the other option. The cost of that is known and written above, which is the point of recording the alternative rather than only the choice.

If the Apple GPU kernels we have to write anyway grow to cover most of what `max/kernels` supplies, the dependency stops paying for itself and this should be revisited on those grounds rather than on licensing ones.
