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

## Validated machines

Every row here was produced by running `pixi run build` and then `build/molla version` on that machine. Do not add a row you have not run.

| Machine | OS | Arch | Cores | Mojo | Build | Test | Date |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MacBook Air M4 | macOS 15.8 | arm64 | 10 physical, 10 logical | 1.0.0 ed45d567 | pass | pass | 2026-08-31 |
| server1 | not yet run | | | | | | |
| server2 | not yet run | | | | | | |
| server3 | not yet run | | | | | | |
| gaming PC, RTX 4090 | not yet run | | | | | | |

The Linux and 4090 rows are open. Issue #1 is not closed until they are filled in, or until they are formally dropped from the tier 1 list.

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
