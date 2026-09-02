# Host kernels

The second piece of issue #26. `molla.nn.quant` turns packed bytes back into float32. This is the arithmetic that runs on those bytes without turning them back into anything, plus the handful of elementwise functions a transformer block is made of. Rope and attention are the next piece and are not here.

## What is in it

`molla.nn.tensor` has two types. `Tensor` is four integers, an address, a ggml type and two dimensions, and it owns nothing. `Buffer` owns float32 and is what activations live in. Weights are read hundreds of times per token and never written, so a view is right for them; activations are written every token and are small, so ownership is right for them.

Shape is ggml's. `dims[0]` is the fast axis, so a weight the directory prints as `[4096, 14336]` is 14336 rows of 4096 and a matvec against it takes 4096 in and gives 14336 out. Reading it the other way gives a shape error rather than wrong numbers, which is the one place in this layout where a mistake announces itself.

`molla.nn.kernel` has `matvec` and `row_dot`, `rms_norm`, `softmax`, `silu` and `gelu` and `swiglu`, `add_into` and `scale_into`, and `argmax`. Six of those are one loop each and are here so that the block code in the next piece reads like the paper rather than like a pile of indexing.

## The identity the fused paths use

The one that matters is the matvec. A dot product against a q4_k weight never builds a float32 copy of the row.

A q4_k value is `d * sc * q - dmin * m`, where `d` and `dmin` are the block's two float16 scales, `sc` and `m` are the group's six bit scale and minimum, and `q` is the four bit quant. Substituting that into a dot product over one group of 32 gives `d * sc * sum(q * x) - dmin * m * sum(x)`. Two sums over the group, and the four scale multiplies happen once per group of 32 rather than once per value. q5_k is the same with a fifth bit pulled from another plane. q6_k has no minimum term, so it is one sum, but its scale groups are sixteen wide and eight of them are in flight at once, which is why that one keeps eight accumulators. q4_0 is `d * (sum(q * x) - 8 * sum(x))`, since its values are centred on eight rather than carrying a minimum.

That rearrangement is not obviously the same computation as dequantizing and multiplying, and that is the whole risk. It is a different order of operations that has to give the same answer.

## How it is checked

`tests/test_kernel.mojo` runs every fused path against dequantizing the same bytes with `molla.nn.quant.dequant_run` and taking a plain dot product. The blocks are random bytes with the scale fields made into real float16 values, the same construction the quantization corpus uses and for the same reason, and the input vector is random in the range minus one to one.

That comparison has a tolerance and the two before it do not, which is worth explaining. The quant corpus compares against the `gguf` package exactly, because both sides do the same arithmetic in the same order and anything less than an exact match would be a bug hiding behind a threshold. Here the two sides genuinely add the same terms in different orders, so they differ in the last few bits of a float32 and always will.

The bound is relative to the sum of the term magnitudes and not to the answer. A dot product over a thousand random terms cancels down to something much smaller than the terms that went into it, so an error that is negligible against the terms can be a large fraction of the result. Measuring against the result would make the test fail on inputs that happen to cancel well, and loosening it until those passed would make it blind everywhere else.

The bound is `1e-4` of that magnitude. Float32 accumulation over a thousand terms drifts by roughly `1e-5` of it, and a real bug in any of these paths is a wrong nibble or a wrong scale group, which is off by tens of percent. There is a lot of room between those two numbers and nothing that lands in it.

The smaller checks are the other half. A softmax that sums to one, a norm with a divisor a person can work out, a gelu at a value that can be looked up, a matvec of a two by three matrix done by hand. An oracle tells you two things disagree. A hand worked value tells you which one is wrong.

## Two things that were wrong first

**An offset added to an address that already had it.** `Tensor.base()` returns a pointer to the tensor's first byte, and `dequant_run` takes a pointer and an offset into it. Passing `base()` and `tensor.address` together reads from twice the address. On a mapped file that is usually a segfault, but on a small test allocation it lands somewhere readable and comes back with plausible looking floats. It was caught because the norm produced `3.4e-36` instead of `0.365`, which is far enough out to notice, and it would not have been caught if the garbage had been in the right range.

**Nothing keeps the bytes under a tensor alive.** A `Tensor` holds an integer address, so the compiler cannot see that a list is still in use once a tensor has been built from it. Mojo frees a local at its last visible use, not at the end of the scope, so the list goes away and the tensor points at whatever the allocator did next. The fix is `keep()` from `molla.sys.mem` after the last real use, the same fix the loader needed in issue #25 and for the same reason. It is worth stating plainly: any type in this codebase that stores an address rather than a reference has this problem, and it is the price of a struct field not being able to carry an origin.

## What this is not

It is not fast. Every loop here is scalar, there is no SIMD and no blocking and no threading, and a matvec against an 8B weight will be slower than it has any right to be. That is deliberate for now. Issue #120 is the repacking work and is where the layout gets chosen for the arithmetic rather than for the file format, and doing that before there is a correct scalar version to check against would be optimizing something nobody has shown to be right.

What it is is the reference. When a fast path lands and the output degrades, there is a slow path to run the same weights through and a bisection to do, rather than a model that got worse for a reason nobody can name.
