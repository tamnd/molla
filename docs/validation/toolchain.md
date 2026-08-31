# Toolchain

molla pins one exact Mojo version. Every machine in the fleet builds with that version and no other. Changing it is a deliberate commit with a green CI run behind it.

The pin lives in two places. `pixi.toml` is the one that matters, and `src/molla/build_info.mojo` carries a copy so `molla version` reports what the binary was built against without anyone having to read the manifest. The test suite asserts the two agree, so they cannot drift silently.

## Current pin

| Item | Value |
| --- | --- |
| Mojo | 1.0.0, build ed45d567 |
| Channel | https://conda.modular.com/max |
| Package license | LicenseRef-Modular-Proprietary |
| Pinned in | `pixi.toml` |

One thing worth being straight about: the Mojo language and standard library are open source under Apache-2.0, but the toolchain we install here is the prebuilt conda package from Modular's channel, and that package is distributed under Modular's own license. So molla's source is Apache-2.0 and molla's dependencies are Apache-2.0, but the compiler you use to build it today is not. That is a real gap between what the README claims and what a fresh `pixi install` actually pulls down, and it belongs in the openness charter rather than in a footnote. Building the toolchain from the open source tree is the fix, and it is not something we have done yet.

## The fleet

molla is developed against five machines. They are listed here with what they actually are, because the design keeps making claims about CPU and GPU behaviour and those claims need somewhere concrete to be checked.

| Name | Role | OS | CPU | RAM | GPU |
| --- | --- | --- | --- | --- | --- |
| macbook | dev, Apple GPU | macOS 15.8 | Apple M4, 10 cores | 24 GB | M4 integrated |
| server1 | small CPU box | Ubuntu 24.04 | AMD EPYC, 4 threads | 5 GB | none |
| server2 | small CPU box | Ubuntu 24.04 | AMD EPYC, 6 threads | 11 GB | none |
| server3 | main CPU box | Ubuntu 24.04 | AMD EPYC, 8 threads | 23 GB | none |
| gpc | CUDA box | Ubuntu 26.04 on WSL2 | i9-13900K, 32 threads | 31 GB | RTX 4090, 24 GB, driver 610.62 |

Two things about this fleet are worth knowing before reading anything else in the docs.

**The CUDA machine is Windows.** `gpc` is a Windows 11 gaming PC and we reach the GPU through WSL2 running Ubuntu 26.04. Mojo does not target Windows natively and molla does not either, so WSL2 is the supported path and native Windows is a non goal. This matters beyond convenience: CUDA under WSL2 goes through `/dev/dxg` rather than the native driver interface, host to device transfers behave differently, and pinned memory is not the same story it is on bare metal. Any performance number from `gpc` is a WSL2 number and gets labelled as one. We should not quietly present it as a Linux CUDA result.

**Three of the five machines have no GPU at all.** The servers are small AMD EPYC instances. That is a good thing for a project that claims CPU is a real target and not an afterthought, since it means the CPU path gets exercised constantly rather than only when someone remembers. It also means server1 with 5 GB of RAM is the machine that will tell us early when the memory estimates in `molla fit` are wrong.

There is no AMD GPU in the fleet. ROCm stays tier 2 and the design doc says so.

## Validated machines

Every row was produced by running `pixi install --locked`, `pixi run build`, `build/molla version`, and `pixi run test` on that machine. Do not add a row you have not run.

| Machine | Target reported | Cores reported | SIMD | Build | Test | Date |
| --- | --- | --- | --- | --- | --- | --- |
| macbook | macos arm64 (apple silicon) | 10 physical, 10 logical | 4 x f32, 8 x f16 | pass | pass | 2026-08-31 |
| server1 | linux x86_64 (avx2) | 4 physical, 4 logical | 8 x f32, 16 x f16 | pass | pass | 2026-08-31 |
| server2 | linux x86_64 (avx2) | 6 physical, 6 logical | 8 x f32, 16 x f16 | pass | pass | 2026-08-31 |
| server3 | linux x86_64 (avx2) | 8 physical, 8 logical | 8 x f32, 16 x f16 | pass | pass | 2026-08-31 |
| gpc | linux x86_64 (avx2) | 16 physical, 32 logical | 8 x f32, 16 x f16 | pass | pass | 2026-08-31 |

All five machines build the same commit from the same lock file and pass the same ten checks. That is the whole of what M0 asks of the toolchain.

Three results are worth recording because they look wrong and are not.

`gpc` reports `avx2` rather than `avx512` even though the i9-13900K is a recent part. Raptor Lake ships AVX-512 fused off so that the efficiency and performance cores present the same instruction set, so `avx2` is correct. Worth knowing before anyone gates a CPU kernel on AVX-512, because it would not run on the fastest CPU in the fleet.

The EPYC servers also report `avx2` rather than `avx512`. These are virtualised instances and the hypervisor is not exposing AVX-512 to the guest. Same conclusion.

Core counts on the servers show physical equal to logical. That is what the hypervisor presents, not a claim about the underlying silicon. Anything that sizes a thread pool from the physical count will size it correctly here by accident rather than by knowing the truth, which is fine for now but should not be relied on when the scheduler starts caring about the difference.

One thing that is not pinned: pixi itself. The fleet currently runs 0.77.1 and 0.78.0 and both resolve the same lock file to the same toolchain, which is the property we actually want. Pinning pixi as well would be pinning the thing that does the pinning, and it has not caused a problem yet.

## Reproducing

```
pixi install
pixi run build
build/molla version
pixi run test
```

Expected output on the M4:

```
molla 0.0.1
  mojo       1.0.0
  target     macos arm64 (apple silicon)
  cores      10 physical, 10 logical
  simd width 4 x f32, 8 x f16
```

## What Mojo 1.0 actually looks like

The spec was written against release notes. Building against the real toolchain turned up several differences that affect every file we write from here on, so they are recorded rather than rediscovered.

**`fn` is gone.** Mojo 1.0 removed the `fn` keyword entirely. Every function is a `def`. The compiler says so directly, which made this a two minute problem rather than an afternoon.

**`alias` is now `comptime`.** Still accepted, but deprecated and warned on. molla uses `comptime`.

**`@parameter if` is now `comptime if`.** Same story, deprecated with a warning.

**Standard library names are snake case.** `simdwidthof` became `simd_width_of`, `sizeof` became `size_of`, `alignof` became `align_of`.

**Platform predicates moved onto `CompilationTarget`.** There is no free standing `os_is_macos()`. It is `CompilationTarget.is_macos()`, along with `is_linux`, `is_apple_silicon`, `has_neon`, `has_avx2`, `has_avx512f`.

**Everything is under a `std` package.** Imports are `std.sys.info`, `std.ffi`, `std.gpu`, `std.testing`, not the bare module names. The whole library ships as one `std.mojoc` file.

**`len()` on a string is a compile error.** Mojo makes you say which length you mean: `byte_length()`, `len(s.codepoints())`, or `len(s.graphemes())`. This is a good decision and it is going to save us a category of tokenizer bug later.

**There is no `mojo test`.** The CLI has run, build, repl, debug, precompile, format, doc, and demangle. `std.testing` provides the assertions but nothing runs them. molla ships its own runner in `tests/harness.mojo`, which is a binary that exits nonzero on failure.

**There is no `mojo format --check`.** Only `-q`. The CI equivalent is to format in place and assert the tree is still clean.

**A plain function will not bind to a closure trait.** This is the one that cost real time. A top level `def` has a "thin" function type that is unique to that declaration, so you cannot collect functions into a table and call them generically, and `type_of` only gets you a type that matches exactly one function. That is why the test harness is assertion based instead of registering test functions. Worth remembering before designing anything else around a table of callbacks.
