# The weight repack and its cache

The last piece of issue #25, split out as #120 because it could not be built until there was a kernel to build it for. A repack has a destination and the destination is whatever the first real matmul reads, so writing one before #26 landed would have meant writing a cache keyed on a layout the first kernel then changed. This records the layout that was chosen, why a repack is worth doing at all, what the cache is keyed on, and the two things that were wrong before it worked.

## What the repack is for

A ggml block is a compression format and it is a good one. A q4_k block puts 256 weights into 144 bytes with two float16 scales and eight six bit scale and minimum pairs packed into twelve bytes with nothing wasted. That density is exactly what you want on disk and in the page cache, and it is exactly what you do not want in the inner loop of a matvec.

Every value read out of a q4_k block costs a nibble extract, a shift by the group index, a lookup into the twelve byte packed scale table, and two float multiplies. All of that happens once per token per value, and none of it depends on the token. The scales the fused kernel computes for a group of 32 are the same scales it computed for that group on the previous token and the same ones it will compute on the next. The nibble it pulls out of a byte was in that byte the last time and will be there the next time.

So the repack does that arithmetic once and writes the answer down. What comes out is not a decompressed tensor: a float32 copy of an 8B q4_K_M is 32 GB and the whole point of the file being 4.9 GB is that it is not. What comes out is a layout that keeps the compression ratio roughly and throws away all the packing.

## The layout

One signed byte per value, then float32 planes of scale.

```
row = [ cols bytes of int8 quants ][ groups float32 dscale ][ groups float32 mscale ]
```

A value is `dscale[g] * q[i] + mscale[g]` where `g = i // group_size`. The mscale plane is only there for the types that carry a minimum. The group is 32 for every type except q6_k, whose scales are sixteen wide, and a row is always a whole number of blocks so it is always a whole number of groups.

Two things fold away in the transform and neither leaves a trace in the layout.

**Centring folds into the byte.** q4_0 subtracts eight, q5_0 subtracts sixteen, q6_k subtracts thirty two, and q8_0 is already signed. That is an integer subtraction done once, and after it the quant is a signed byte and the type has no minimum term at all. So q4_0, q5_0, q6_k and q8_0 have no mscale plane and their dot product is one sum per group rather than two.

**The minimum folds into a sign.** q4_1 and q5_1 store a per block minimum `m` and their value is `d * q + m`, so `mscale = m`. q4_k and q5_k store `dmin` and a six bit `mm` per group and their value is `d * sc * q - dmin * mm`, so `mscale = -(dmin * mm)`. Both end up as a plus, and the kernel adds a plane without knowing which of the four kinds of arithmetic produced it.

Every quant fits in a signed byte after centring. The widest is q8_0, which is already an i8. q6_k is six bits centred on 32, so minus 32 to 31. q5_0 is five bits centred on sixteen. The four bit types are the easy ones. Nothing needed a wider element and nothing needed a saturating store.

## What it costs

| Type | ggml bytes per block | planar bytes | Growth |
| --- | --- | --- | --- |
| q4_0 | 18 | 36 | 2.00x |
| q4_1 | 20 | 40 | 2.00x |
| q5_0 | 22 | 36 | 1.64x |
| q5_1 | 24 | 40 | 1.67x |
| q8_0 | 34 | 36 | 1.06x |
| q4_k | 144 | 320 | 2.22x |
| q5_k | 176 | 320 | 1.82x |
| q6_k | 210 | 320 | 1.52x |

Whole files are less than the worst row, because a model is not one type and the f32 tensors in it are copied nowhere. A SmolLM2 135M at q8_0 is 138 MB of model and 144 MB of cache. A Qwen 2.5 0.5B whose weights are all q5_0 is 469 MB of model and 689 MB of cache.

That is real disk and it is the reason the cache is a separate file rather than something written into the model. A machine that does not have room for it deletes it and loses nothing but the time, and a machine that does keeps it and never pays that time again.

## What the cache is keyed on

A sha256 over the GGUF header, the metadata and the tensor directory, which is bytes zero to `data_start`, plus the file length as eight bytes on the end. Then, separately in the header and checked separately, the layout version and the target.

The digest deliberately does not cover the tensor data. Hashing 4 GB takes about 45 seconds on this laptop, which would make checking the cache slower than rebuilding it and would turn a feature that exists to save time into one that costs it. The directory is a strong key in practice: it names every tensor, its type, its shape and its offset, so two files that agree on all of that and disagree on their contents are a file somebody rewrote in place without changing its length. That is the one case this misses, it is not a case that happens by accident, and the cost of covering it is the whole benefit of having a cache.

The file length is in the digest because the directory alone is identical for a file that was truncated after it, and a truncated model is exactly the case where a cache claiming to match would be read past the end of.

The layout version is a separate field rather than part of the digest so that the miss can say which thing changed. A cache written by an older molla says so, a cache written for a different target says so, and a cache for a different model says so. All three are misses and the load carries on either way, but a person looking at a repack that reruns every time wants to know which of those it is.

## Eight ways to refuse

`open_cache` never raises. Every rejection sets a reason and returns an unusable cache, because a missing cache is what every first load has and turning the normal case into an exception would mean a try around a thing that is not exceptional.

| Reason | What it catches |
| --- | --- |
| no cache file beside the model | the first load, and a cache somebody deleted |
| the cache file is too short to have a header | a write that died in its first page |
| the cache file is not a molla repack | a name collision with something else |
| the cache was written by a different molla | a container format change |
| the cache is an older weight layout | the layout changed and the bytes mean something else now |
| the cache was repacked for a different target | a host cache on a machine that wants a device one |
| the cache is truncated | the header's length and the file's length disagree |
| the cache belongs to a different model | the digest does not match |

Two more sit below those and cover a file whose header is fine and whose directory is not: a directory that runs past its own data, and an entry that points outside the file. Those cannot be caught by the digest, because the digest is over the model and not over the cache, so they are bounds checks and they run on every entry before a single tensor address is handed out.

`tests/test_cache.mojo` writes a good cache and then edits one field of it eight times. That is the only way to reach the check being tested: a file full of noise fails at the magic and proves nothing about the seven checks after it.

## Where it runs

On the transfer pool the load already has, between the page touching loop and the ready queue push.

A worker that has just faulted a tensor's pages in is holding the warmest copy of those bytes that will ever exist. If a repack is wanted, that is the moment to do it, and the layout transform costs the arithmetic rather than the memory traffic. The alternative, a second pass after the load finishes, reads four gigabytes twice.

What a worker does with the result is one `pwrite` per few megabytes into a temporary file at an offset that was decided by `plan_repack` before any thread started. So the workers never talk to each other and never talk to the thread that is draining the ready queue. A repack that fails records its errno in one atomic slot, the load carries on and finishes normally, and the caller throws the half written file away.

The temporary file is renamed into place at the end. A process killed in the middle leaves a `.molla-repack.tmp` and no `.molla-repack`, so the next load misses and repacks rather than finding a file that is the right length and half garbage.

## How correctness is checked

Three layers, and the first one is the one that matters.

**A repacked row decodes to bit for bit the same float32 values as the blocks it came from.** No tolerance. `tests/test_repack.mojo` runs `repack_row` followed by `planar_run` against `dequant_run` on the same bytes for all eight types and asserts exact equality, naming the first index that differs. That is a stronger claim than the fused kernels get and it is available here because every fold is either integer arithmetic or the same float32 multiply reassociated identically. `d * sc` computed once in the repack and `d * sc` computed per token in the kernel are the same two floats through the same multiply, so they are the same bits.

**The planar dot product agrees with the fused ggml one within the kernel tolerance.** That one does have a tolerance, for the reason `kernel.md` gives at length: the two sides add the same terms in a different order and will differ in the last bits of a float32 forever. The bound is `1e-4` of the sum of term magnitudes, the same bound the fused paths are held to.

**A cached tensor read back through the mapping decodes the same as the model file.** `tests/test_cache.mojo` builds a small GGUF with a q8_0 tensor, a q4_0 tensor and an f32 tensor, loads it with the repack on, opens the file the load wrote the way a later process would, and compares. The f32 tensor is as much the point of that fixture as the other two: it has no planar form, so it has to be absent from the plan and absent from the cache and still bind, and a fixture where everything was repackable would not notice if it were quietly repacked into nonsense.

End to end, the check the issue asks for is that the second load of a model produces the same tokens as the first. It does, on SmolLM2 135M at q8_0 and on Qwen 2.5 0.5B at q5_0, greedy, with the cache deleted between the two runs of the pair.

## Two things that were wrong first

**The plan counted padding that was never written.** Tensors are laid out 64 byte aligned in the cache, and `plan.total` was computed by aligning after the last tensor as well as between them. Nothing is written into that trailing pad, so the header recorded a length the file did not have and every cache was rejected as truncated on the next load. It survived the first real model because the last repacked tensor in that file happened to end on a 64 byte boundary, which is the kind of thing a fixture with two small tensors finds in one run and a smoke test on a real model never finds at all.

**The directory entry fields overlapped.** The name length was written at offset 32 in an entry whose byte count field ended at offset 36, so a name length and the top four bytes of a size were the same four bytes. Both readers and both writers agreed with each other, so the cache round tripped perfectly and would have started producing wrong tensor sizes the first time a tensor was bigger than four gigabytes. It was found by reading the two functions side by side rather than by a test, which is the honest account: no test in this file would have caught it.

## What this is not

There is no device target yet. `TARGET_HOST` is zero and it is the only value written, so a cache built on a machine with a card is the same file as one built without. The field is in the header because adding it later means either a container version bump or a cache that is silently wrong on the first machine that has two targets, and a four byte field that always holds zero is cheaper than either.

There is no SIMD in `planar_row_dot` yet. The layout was chosen so that a vectorized version is a straight loop over int8 loads with two float32 planes indexed by a shift, and writing that is the next piece of work rather than this one. What this buys today is the unpacking, which the fused ggml paths were doing per token and now do not.

The cache is not shared between machines and is not meant to be. It is derived, it is keyed on a target that will eventually mean something, and copying one between hosts is a thing that would work today and stop working the moment the target field has a second value. Deleting it is always safe and always correct.
