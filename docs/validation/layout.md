# One byte per weight is the memory problem and half the throughput problem

Issue #172, the M2c tracking issue. [performance.md](performance.md) said decode is slow because there are 453 kernel launches per token and memory is high because of a driver arena and a mapping held too long. Both of those were true and both are now fixed. This page is about the thing underneath them, which is that molla's planar layout stores a signed byte for every weight no matter how many bits that weight had in the file, and on a four bit model that is a factor of two on disk, a factor of two in device memory, and a factor of two on the bytes a token has to read.

The layout was an explicit trade and [repack.md](repack.md) and the module docstring both say so. "A q4_k weight is 4.5 bits per value in the file and 10 bits here, so a 4 GB model is around 9 GB repacked." What was not known when that was written is how much of the remaining performance gap it accounts for, and the answer is all of it on a large model.

## The measurement

Meta-Llama-3.1-8B-Instruct-Q4_K_M, on gpc, an RTX 4090 through WSL2, 78 token prompt and 64 generated, three runs, current main.

| engine | decode tok/s | weight bytes read per token | achieved GB/s | share of the card's 1008 GB/s |
| --- | --- | --- | --- | --- |
| molla | 77.7 | 9.35 GB | 727 | 72 per cent |
| llama.cpp | 163.6 | 4.61 GB | 754 | 75 per cent |

The two engines are within four per cent of each other on bandwidth efficiency. molla is 2.11 times slower and reads 2.03 times as many bytes. There is nothing else in the difference.

That also settles what launch cost is worth on this model. 453 launches at 4.9 us is 2.22 ms, and a molla token here is 12.9 ms, so launches are 17 per cent of it. Fusing the whole layer down to nothing would leave the 8B still slower than llama.cpp. performance.md read the 8B's smaller gap as evidence that a fixed per launch cost matters less when the launch does more work, which gives the right ranking for the wrong reason.

The small model says the opposite and it is the same measurement. SmolLM2-135M-Instruct-Q8_0 repacks from 138 MiB to 144 MiB, so it reads essentially the same bytes as llama.cpp, and molla gets 273.5 tok/s against 870.1. There the gap is launches and only launches, and 453 of them at 4.9 us puts the ceiling at 450 tok/s, which is where molla is.

So the two problems are real, they are independent, and which one dominates depends on the model.

| model | quant | bytes ratio to llama.cpp | decode ratio | bound by |
| --- | --- | --- | --- | --- |
| SmolLM2 135M | q8_0 | 1.04 | 3.18 | launches |
| Llama 3.1 8B | q4_K_M | 2.03 | 2.11 | bytes |

## What the byte count actually is

Counting the weights in the 8B by type and pricing each one three ways. The file column is what ggml stores, the current column is what the repack cache holds, and the packed column is the layout proposed below.

| type | weights | file | current | packed |
| --- | --- | --- | --- | --- |
| q4_K | 6498025472 | 3486 MiB | 7746 MiB | 3873 MiB |
| q6_K | 1531969536 | 1198 MiB | 1826 MiB | 1278 MiB |
| f32 | 266304 | 1 MiB | 1 MiB | 1 MiB |
| total | | 4685 MiB | 9574 MiB | 5153 MiB |

The current column is 9574 MiB and the repack cache on disk is 9573 MiB, so the model is the arithmetic and there is nothing unaccounted for.

## Where the byte goes

`planar_row_bytes` in `src/molla/nn/repack.mojo:150` is `cols + planes * groups * 4`. One byte per weight, then one or two float32 planes per group of 32. For q4_K that is 8 bits of quant that only needed 4, and 2 bits of scale that only needed 1, so 10 bits where 5 would do.

The quant plane is the larger half and it is pure waste. Every value the repack writes into a signed byte for a four bit type is in 0 to 15, which the docstring says outright: "q4_k runs 0 to 15, and the widest is q8_0". Nothing reads the high nibble because nothing was ever written there.

The scale planes are float32 because the repack folds two numbers into one. A q4_K group's dscale is `d * sc` where `d` is a float16 out of the block and `sc` is a six bit integer, and its mscale is `-(dmin * mn)` on the same shape. Storing the product back in float16 costs at most one part in 2048 on a number that multiplies a value quantized to one part in 16. That is three orders of magnitude below the quantization the file already did.

## Why this was not visible until now

On CUDA it was not visible at all, because the weights are on the card and host resident memory does not count them. molla's host peak on the 8B is 1360 MiB against llama.cpp's 5042, which looks like a win and is one, but the 9573 MiB sitting in VRAM is not in any table in [bench.md](bench.md).

Metal is where it shows, because the pool is unified memory and the weights are host bytes by construction. The 8B on the macbook, six token prompt, four generated:

```text
repack:    226 tensors from cache, 9572 MiB
maximum resident set size   5347 MiB
peak memory footprint      10212 MiB
```

`maximum resident set size` is what `wait4` reports and what `scripts/bench.py` records, and on macOS it does not count a Metal buffer's pages. `peak memory footprint`, which is `phys_footprint`, does, and it is 10212 MiB against 9572 of weights plus 512 of KV cache plus about 130 of everything else. The two numbers add up exactly, so the arena question from performance.md is answered here too: on Metal there is no arena, there is a model.

This also means every Metal peak molla has published, in bench.md and in the issues, is understated by the size of the weights. That is a reporting bug in the harness as much as anything and it is listed below.

## The packed layout

Bump `LAYOUT_VERSION` and change two things about a row.

```text
row = [ quants at the type's native bit width ][ groups float16 dscale ][ groups float16 mscale ]
```

The mscale plane stays conditional on `has_min` exactly as now, the group is still 32 for everything except q6_K where it is 16, and the value is still `dscale[g] * q[i] + mscale[g]`. Everything a kernel does with the row is the same arithmetic. What changes is how many bytes it reads to get `q[i]`.

Native bit width by type, and what each one costs per weight including its scales:

| type | file | quant bits | scale bits | total | current |
| --- | --- | --- | --- | --- | --- |
| q4_0 | 4.5 | 4 | 0.5 | 4.5 | 9.0 |
| q4_1 | 5.0 | 4 | 1.0 | 5.0 | 10.0 |
| q4_K | 4.5 | 4 | 1.0 | 5.0 | 10.0 |
| q5_0 | 5.5 | 5 | 0.5 | 5.5 | 9.0 |
| q5_1 | 6.0 | 5 | 1.0 | 6.0 | 10.0 |
| q5_K | 5.5 | 5 | 1.0 | 6.0 | 10.0 |
| q6_K | 6.5625 | 6 | 1.0 | 7.0 | 10.0 |
| q8_0 | 8.5 | 8 | 0.5 | 8.5 | 9.0 |

Every type lands within a bit of what the file holds, and q4_0 and q8_0 land exactly on it, because a float16 scale per 32 values is the same header ggml has once the block is unpacked.

The four and eight bit cases are the whole win and they are trivial. Four bits is two values a byte, packed low nibble first in the order the values appear, which is not the order ggml packs them in but is the order the plane already stores them in, so the repack loop changes where it writes and not what it computes. Eight bits is what exists today.

Five and six bits are done as bit planes rather than as an unaligned stream, because an unaligned stream costs a shift chain per value on the read side and the whole point is to make the read cheap. Six bits is a four bit plane of `cols / 2` bytes followed by a two bit plane of `cols / 4` bytes. Five bits is a four bit plane followed by a one bit plane of `cols / 8` bytes. That is what ggml does inside a block and it is the right shape here for the same reason, with the difference that the planes are row wide rather than block wide so a thread reads one contiguous run of each.

## What it does to the kernel

`planar_matvec_kernel` at `src/molla/nn/gpu.mojo:245` has each thread walk `i = t, t + tile, ...` and read one byte per value. Packed, a thread reads one byte and gets two values, so it walks `i = 2t, 2t + 2 * tile, ...` and handles the pair. A block of 256 threads then covers 512 values with 256 contiguous byte loads, which is the same coalescing the current kernel gets and half the loads.

The pair is always inside one group, because the group is 32 and 32 is even, so the scale lookup happens once for both values instead of once each. Both values also multiply the same dscale, so the group accumulator can add `q_lo * x_lo + q_hi * x_hi` before the multiply. That is fewer float multiplies per weight than today, not more.

The scale planes moving to float16 needs a `bitcast[Float16]` view alongside the existing `Int8` and `Float32` ones and a widen on load. Every GPU in the fleet does that in the load instruction.

None of this is a new kernel. It is the same kernel reading a denser row.

## What it should be worth

Memory, which is arithmetic and not a guess. The 8B repack cache goes from 9573 MiB to 5153 MiB, device memory with it, and the Metal host footprint from 10212 MiB to about 5800 MiB. Against llama.cpp's 5042 MiB host peak on CUDA, molla's 1360 MiB is already 3.7 times better and does not move, because the host was never holding the weights on that path. The honest way to state the memory win is that molla stops needing twice the VRAM llama.cpp needs, which is the difference between an 8B fitting on an 8 GB card and not fitting.

Throughput, which is a prediction with a mechanism. If molla stays at 72 per cent of the card's bandwidth and reads 1.86 times fewer bytes, the 8B goes from 77.7 tok/s to about 145. llama.cpp is at 163.6. So this closes the 8B gap to about 12 per cent and does not by itself beat it, and it should not be sold as though it does. Two times llama.cpp on an 8B still needs either fewer bits per weight than llama.cpp reads or more than one sequence sharing the read, which is what performance.md already concluded and this does not change.

The memory half of that prediction was right to a megabyte and the throughput half was wrong. What actually happened is the next section, and it is the more useful of the two.

## What it was actually worth

The quant plane was packed and measured on gpc, same model, same prompt, same three runs, with the byte wide build and the packed build alternating on one machine.

| build | repack cache | decode tok/s | weight bytes read per token | achieved GB/s |
| --- | --- | --- | --- | --- |
| byte wide | 9573 MiB | 77.6 | 10.04 GB | 823 |
| packed | 6474 MiB | 74.2 | 6.79 GB | 590 |

Memory landed where the arithmetic said it would, 6474 MiB against a predicted 6475. Decode went the other way by four per cent.

nsys puts the whole of that in the q4_K matvec, which is the only kernel the change touches. Per forward pass it was 8.45 ms moving 8.12 GB, and it is now 8.97 ms moving 4.87 GB. Read as bandwidth that is 961 GB/s falling to 543. Read as values it is 769 billion a second falling to 724, which is the same number twice. The kernel was never being paid by the byte.

Four arrangements of the inner loop were written and measured, all within one per cent of each other: a value to a thread with two threads sharing a byte, a byte to a thread with both of its values, four bytes to a thread through a `UInt32`, and that last one again with the activations read four at a time into a SIMD accumulator. Larger thread blocks were tried, 256 made no difference and 512 was worse. Four rows to a block instead of one, which cuts the block count by four and reads the activation vector once for four rows, was ten per cent worse. Removing the activation read from the kernel entirely, which computes the wrong answer and exists only as a probe, changed nothing at all.

`scripts/mem_probe.mojo` reads a 2 GiB buffer with the same one block per row shape the matvec uses and says what that shape is worth on this card.

| row | one byte a thread | four bytes a thread |
| --- | --- | --- |
| 512 B | 611 GB/s | 423 GB/s |
| 1 KiB | 939 GB/s | 609 GB/s |
| 2 KiB | 938 GB/s | 704 GB/s |
| 4 KiB | 945 GB/s | 704 GB/s |
| 8 KiB | 947 GB/s | 702 GB/s |
| 16 KiB | 947 GB/s | 706 GB/s |

Two things fall out of that. Row length stops mattering above a kibibyte, so shortening the rows is not what cost the four per cent and the block reduction is not the problem the previous section guessed it might be. And the more arithmetic a kernel does per byte it reads, the fewer bytes a second it gets, which is what a kernel that is short of issue slots rather than short of bandwidth looks like.

So the reading in "The measurement" above needs correcting. Both engines' achieved bandwidth figures are right. The conclusion drawn from them, that the bytes are what separates them, is not, because molla's kernel does not go faster when the bytes go away. What separates them is work per value: llama.cpp's CUDA decode quantizes the activation vector to eight bits and multiplies four values at a time with an integer dot product, and molla's does one value at a time in float. That is a four to one difference in instructions on the hot loop and it is the size of the gap.

The packing still pays for itself. It is a third off device memory, a third off the cache on disk, it is the only reason #182 and #183 have anything to sit on, and it costs four per cent of a decode rate that the next change has to fix anyway.

On the small models it is worth nothing at all, because q8_0 barely moves. Those need #168 and #170 and this page does not help them.

The disk win is not nothing either. A repack cache that is 1.1 times the model instead of 2.04 times it is the difference between a cache people tolerate and one they turn off.

## Order

Pack the quant plane first and leave the scales at float32. That is 9574 MiB to 6475 MiB on the 8B, it is bit exact against the current layout because the integers written are the same integers, and it can be verified by running the corpus with the old cache and the new one and comparing logits with no tolerance at all.

Narrow the scale planes to float16 second, as its own change, because it is the only part of this that moves a number. It is worth 6475 MiB to 5518 MiB and it needs the corpus run with a tolerance and a statement of what the tolerance is.

Bit plane the five and six bit types third, which takes q6_K from eight bits of quant to six and gets the 8B to 5153 MiB. It is worth 365 MiB on this model and it is the fiddliest part, so it goes after the two that are worth more and are simpler.

Fix `scripts/bench.py` to report `phys_footprint` on macOS rather than `ru_maxrss`, before any of the above, so that the Metal numbers in bench.md mean what they say while this work is happening. There is no equivalent problem on Linux, where the CUDA weights are genuinely not host memory and `ru_maxrss` is right.

## What this does not answer

Whether a repack cache should exist at all once the layout is within ten per cent of the file. At 4.5 bits for q4_K the honest alternative is reading ggml blocks on the device directly, which is what llama.cpp's own CUDA and Metal kernels do, and it would remove the cache, the second copy on disk, the load stage that reads it, and the layout version problem in one move. The cost is that the eight per type unpacking loops come back, on the device this time. MAX's own GPU quantized matmul, `matmul_gpu_qint4` at `quantization/qmatmul_gpu.mojo:1773`, takes packed int4 with a group size, which is evidence that packed is the shape a GPU kernel wants and not evidence either way about blocks. This is worth a measurement before M3 and not a rewrite now.

Whether the 72 per cent of peak bandwidth holds once the rows are half as long. A shorter row is less work to amortise the block reduction over, and the reduction at `gpu.mojo:300` is a shared memory tree rather than a warp shuffle. If the packed kernel lands under 72 per cent then that reduction is the next thing to look at.
