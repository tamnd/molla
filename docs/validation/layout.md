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

## What the bit planes were worth

The five and six bit half of the layout landed after the four bit half and out of the order below, and the reason it moved up is that it stopped being a memory change. #203 measured the Metal matvec against a variant that does the loads and no arithmetic at all and found the shipped loop within ten per cent of that floor, which means on that backend the loop is paid by the byte and the only thing left that removes a byte is this.

Before it, `quant_form` widened every five and six bit type to a byte a value, so q5_0, q5_1, q5_K and q6_K were all stored at eight bits of quant where the file holds five or six. Now they are a nibble plane of `cols / 2` bytes followed by a high plane of `cols / 8` bytes at five bits and `cols / 4` at six, which is the shape the section above describes.

On an M4 laptop, greedy, the same prompt and 32 tokens out of it, three runs each of the shipped build and the same tree with the change reverted, both in one sitting.

| model | repack cache before | after | decode ms a token before | after |
| --- | --- | --- | --- | --- |
| Llama 3.1 8B q4_K_M | 6474 MiB | 6108 MiB | 240, 243, 252 | 177, 159, 182 |
| Qwen 2.5 0.5B | 663 MiB | 512 MiB | 38, 38, 38 | 29, 29, 29 |

The load average was between 13 and 25 across that sitting, which on this machine is normal and is usually enough to make a timing worthless. It is not here, and the Qwen rows say why: four runs of one build in a row gave 38 four times and four of the other gave 29 four times, at a load that moved by half over the same minutes. Decode on Metal at this size is waiting on the GPU and not competing for the cores, so the number holds while a prefill number on the same machine would not.

The 8B is the case worth reading. The cache is 5.7 per cent smaller and decode is a quarter faster, which is not the same number and is not supposed to be. The cache holds `token_embd` and the output head, which a decode token reads one row of and all of, and the tensors this change touches are the thirty three q6_K ones, which in a q4_K_M file are `ffn_down` and `attn_v`. `ffn_down` is the widest matrix in a layer, so the share of the bytes a token actually reads that moved from eight bits to six is much larger than the share of the file that did.

Qwen is the other case and it is there because the file is misnamed. It says q4_K_M and `molla gguf` reports 133 q5_0 tensors against 14 q4_K, so almost the whole model was on the widened path and almost the whole model moved.

Through `scripts/bench.py` at the 512 and 128 the gate uses, which is a longer context than the table above and reports prefill and resident bytes as well, Qwen goes 542.8 to 665.8 tokens a second of prefill, 21.1 to 28.0 of decode, and 1316 to 763 MiB resident, with the two builds a minute apart. That is the same result the table reports and it is the one to quote, because it is the harness the rest of the bench page uses. The 8B is not quoted at that length on this machine: the load went past fifty during the run and it produced 2.8 tokens a second of decode for a build that does better than that, which is a fact about the laptop and not about the layout.

On a 4090 it is a different shape of result, and the memory is the part that carries. Both builds, six runs, one sitting, `scripts/bench.py` at the same 512 and 128 the gate uses, the load between 1.4 and 2.0 the whole way through.

| model | prefill before | after | decode before | after | card before | after |
| --- | --- | --- | --- | --- | --- | --- |
| SmolLM2 135M | 10489.8 | 10280.0 | 275.3 | 263.9 | 656 MiB | 656 MiB |
| Qwen 2.5 0.5B | 5039.2 | 4942.3 | 297.7 | 297.7 | 1110 MiB | 960 MiB |
| Llama 3.1 8B q4_K_M | 377.3 | 366.3 | 85.8 | 89.1 | 7404 MiB | 7038 MiB |

SmolLM2 is the control and it is the row to read first. It is q8_0 throughout, this change cannot reach it, and it still moved 4 per cent on decode and 2 per cent on prefill between two builds that do the identical thing to it. That is the noise floor on the small models, and it is why the Qwen row is written down as no timing change rather than as a win: an earlier sitting had the same pair at 294 against 307 and this one has them equal, which means neither number is a measurement of anything. The Qwen memory is, and it is 150 MiB.

The 8B is outside that floor in both directions. Prefill costs 2.9 per cent, decode gains 3.8, and each of those is three runs a side clustered within a few tenths of a per cent with a gap between the clusters ten times wider. Prefill on CUDA is matmul bound and the wide path does two loads a pass where the nibble path does one, so the second plane is a second stream to carry. Decode is bound by the bytes and the bytes went down. The card holds 366 MiB less and the host holds 128 MiB less, so the trade is worth taking, but the prefill side of it is a real cost and not a rounding.

That gap is also the whole reason the loop steps the way it does. The first cut walked a whole high byte a pass, eight values at five bits, which is the natural shape on Metal and takes 4090 prefill on Qwen from 4895 to 2937 tokens a second. The shipped loop steps by the same `MATVEC_STEP` and 2 the nibble path beside it uses and computes the high index and shift a pass, which constant folds back to the hoisted form wherever the step covers a whole high byte, so Metal keeps its shape and CUDA keeps its occupancy.

This is the change that makes the layout worth what the table above says. It is also the last one that is bit exact: every value the planes assemble is the value the byte held, the offset that makes it signed comes off in the subtraction the magic number was doing anyway, and the corpus logits are the corpus logits.

## What the float16 scales were worth

The other half of the row. A group carried a float32 scale and a float32 minimum where the file carries float16 of each, which is 2 bits a value of scale against the 0.5 or 1 the file spends. #182 makes them float16 and that is the last piece of the layout the first section of this page proposed.

The 8B repack cache goes 6108 MiB to 5151 MiB, against the 5153 the table near the top of this page predicted before any of this was written. The model file is 4685 MiB, so the layout now costs ten per cent over the file where it cost twice the file when this started. Qwen 2.5 0.5B goes 512 to 468 MiB and SmolLM2 135M goes 144 to 136.

The 136 is the one to look at twice. SmolLM2 is q8_0 throughout and a q8_0 block is 32 bytes of quant and a float16 scale, which is exactly what a planar row of it now is, so the repack cache and the tensor data in the file are the same size to the byte. The same is true of q4_0. For those two types the layout is now free, and `tests/test_cache.mojo` checks it rather than leaving it as a claim.

On a 4090, three pairs of runs at the same 512 and 128, the load between 1.6 and 2.1 throughout.

| model | prefill before | after | decode before | after | card before | after |
| --- | --- | --- | --- | --- | --- | --- |
| SmolLM2 135M | 10078.4 | 10280.0 | 265.0 | 264.5 | 656 MiB | 648 MiB |
| Qwen 2.5 0.5B | 4942.3 | 4990.3 | 299.8 | 304.8 | 960 MiB | 916 MiB |
| Llama 3.1 8B q4_K_M | 367.1, 365.5, 364.2 | 360.9, 361.2, 359.6 | 88.6, 89.1, 88.5 | 95.5, 94.8, 95.4 | 7038 MiB | 6082 MiB |

The 8B holds 956 MiB less on the card, against 957 predicted, and decodes 7.3 per cent faster. Prefill costs 1.4 per cent, which is the same trade the bit planes made and for a smaller reason: the scale plane is half the loads it was, but a float16 load widens where a float32 load did not, and on a matmul bound pass that convert is not free. Decode is bound by the bytes and there are fewer of them.

Metal has the memory and not the timing. Qwen goes 512 to 468 MiB and decodes 30, 32, 34 against 32, 34, 33 ms a token, which is no change and is expected: a half billion parameter model on this backend is not waiting on the scale plane. The 8B could not be timed at all on the afternoon this landed. The laptop went from a load of 7 to a load of 21 across six runs and produced 232 to 330 ms a token on a build that had measured 159 to 182 the hour before, so the whole series says what the machine was doing and nothing about the layout. What is not in doubt on that backend is the 957 MiB, because a repack cache is a file and a file has a size.

Accuracy is in [logits.md](logits.md) with the numbers rather than an argument. The short version is that the corpus moves by at most 1.6e-5 on any of its four comparisons, the greedy token is the same token on all fourteen cases, and the movement is smaller than the difference between running the same case on an M4 and on a 4090.

## What was in the way, on Metal

The section above is a CUDA answer that was allowed to stand as a general one for a day. `scripts/matvec_probe.mojo` was written because nsys on an 8B is an hour a question: it runs the same kernel over a synthetic planar tensor of a real shape, and it runs five variants of it that each delete one piece of the work so that the piece can be priced.

The first variant deletes something nobody suspected. The kernel finds which group a value belongs to with `i // group`, and `group` is a compile time constant. On CUDA that is free, because NVVM turns it into a shift. On Metal it is not, and it is not close.

4096 by 4096, q4_K, on the M4, microseconds a launch:

| variant | Apple M4 | RTX 4090 |
| --- | --- | --- |
| shipped | 1916 | 29 |
| the same with `i >> 5` | 867 | 29 |
| and with 32 bit loop indices | 740 | 29 |
| a thread to a group | 639 | 64 |
| no scale plane read at all | 521 | 28 |
| the loads and no arithmetic | 199 | 10 |

More than half of the Metal matvec was a divide that the index can never need, because `i` counts up from a thread id and is never negative, and a signed divide has to handle a negative numerator whatever the divisor is. Metal keeps the correction, CUDA removes it.

End to end on the M4, same binary either side of the one line change, decode tokens per second:

| model | before | after |
| --- | --- | --- |
| Llama 3.1 8B q4_K_M | 0.8 | 2.1 |
| Qwen 2.5 0.5B q4_K_M | 9.9 | 19.0 |
| SmolLM2 135M q8_0 | 24.7 | 31.2 |

The 8B moves most and the 135M least, which is the right shape: the larger the model the more of a token is the matvec, and the small one is launch bound for the reason performance.md gives. The 8B was measured at 78 prompt tokens and 32 generated because a 305 token prompt through a prefill that is 305 decodes is a minute and a half either side, and the other two at 305 and 64. The macbook's load average was between 1.9 and 3.1 across all of it, which is quiet for that machine but is still why these are ratios on one machine rather than numbers to publish.

It is exact. `i // 32` and `i >> 5` are the same number for every index this kernel produces, and the whole logit corpus on Metal is identical in every digit either side of the change.

Two things this says beyond the fix. The first is that a compile time constant is not a compile time constant on every backend, and the only way to know which is to price it on each one. The second is that the CUDA reading in the section above stands but its scope does not: on CUDA the wall really is work per value, and on Metal the wall was one instruction that should never have been there. The remaining Metal variants say the rest of the ladder is there too, 740 with narrow indices and 199 for the loads alone, so Metal has another three times in it after this and it is the same three times CUDA has.

## What was in the way, on both

Deleting the divide left Metal at 899 microseconds on the 4096 by 4096 shape against a floor of 197, and CUDA where it already was at 29 against a floor of 11. Both are about three times off the loads, and the section above guessed that the remaining three times were the same on the two backends. They are, and the largest single piece of them is not the multiply. It is turning the quant into a float.

`Float32(n)` for a small integer compiles to a convert. On NVIDIA a convert issues on the same unit as transcendentals, at a quarter of the rate of a multiply, so the four converts that a byte of two nibbles costs are worth sixteen multiplies before any multiplying has happened. The kernel does four multiplies per byte. That is the ratio the probe kept running into from three directions and never named.

There is a way to do the conversion with no convert in it. A float32 whose exponent is 23 has a mantissa step of exactly one, so every representable number from `2^23` to `2^24` is an integer and its bit pattern is `0x4B000000 + n`. For `n` below `2^23` nothing carries out of the mantissa, so `0x4B000000 | n` is the same bits as `0x4B000000 + n` and the addition is an or. Subtract `2^23` back off and the result is `n` as a float. An integer or and a floating point subtract, both full rate on both vendors, in place of a convert.

A centred type gets its sign extension free. `n ^ 8` is `n + 8` below eight and `n - 8` above it, so biasing the subtraction by eight turns the same or into `(n ^ 8) - 8`, which is the four bit two's complement value the repack wrote. A byte wide type is the same one bit wider, `u ^ 0x80` and a bias of 128. So one shape covers all three quant forms and the type only picks a constant.

The probe, microseconds a launch, best of five batches of sixteen, against the shift kernel from the section above:

| shape | M4 before | M4 after | 4090 before | 4090 after | M4 floor | 4090 floor |
| --- | --- | --- | --- | --- | --- | --- |
| q4_K 14336 by 4096 | 3029 | 1924 | 93 | 42 | 620 | 26 |
| q4_K 4096 by 14336 | 2509 | 1377 | 93 | 35 | 578 | 21 |
| q4_K 4096 by 4096 | 899 | 553 | 28 | 14 | 197 | 11 |
| q8_0 1536 by 576 | 152 | 141 | 8 | 8 | | |
| q8_0 49152 by 576 | 4119 | 4125 | 49 | 49 | | |

The four bit shapes are 1.6 to 1.8 times on Metal and 2.0 to 2.7 times on CUDA. The byte wide shapes are neither, on either card, and that is the result worth having rather than the one worth hiding: a signed byte load already sign extends in the hardware, so the loop being replaced there is one convert where the nibble loop's is a mask, a shift, an xor, a subtract and a convert. Three instructions for one is a wash. Three instructions for five is not. The byte loop was changed anyway because it costs nothing and it keeps the two readers of a planar row the same shape, which is a thing `gpu_ops.mojo` already says it wants, and not because it buys anything.

It is exact. There is no rounding anywhere in it: every value the trick sees is under the mantissa step, so the or is exact, and the subtraction of a nearby power of two from a number of the same exponent is exact. The whole logit corpus was run on Metal and on CUDA and the worst case in it is unchanged in every digit that was already written down, sum 5.1e-4, value 3.1e-2, log probability 9.6e-2. No tolerance moved.

End to end, decode tokens per second, same binary either side:

| model | backend | before | after |
| --- | --- | --- | --- |
| Llama 3.1 8B q4_K_M | CUDA 4090 | 72.9 | 99.7 |
| SmolLM2 135M q8_0 | CUDA 4090 | 223.8 | 226.5 |
| Llama 3.1 8B q4_K_M | Metal M4 | 2.0 | 2.9 |

The 8B moves and the 135M does not, which is exactly what the probe said would happen, for the reason it said: one is four bit and the other is not. The Metal 8B was run twice alternating with the baseline and read 2.0, 2.9, 2.0, 2.9, which is the only reason a number off that machine is here at all, because its load average was above eleven throughout. SmolLM2 on Metal read within four per cent either way across four alternating runs on the same loaded machine, which resolves nothing the probe has not already resolved more cleanly at 152 against 141 and 4119 against 4125.

### The other way, which was measured and is not being taken yet

The probe also priced the thing #186 originally proposed, which is to quantize the activation vector as well and multiply weight by activation as integers, converting once per run of eight values instead of once per value. That is faster still on both cards, 1240 against 1924 on the Metal gate shape and 34 against 42 on the CUDA one, and it was implemented, and it is not what is landing.

Two reasons. The first is arithmetic. Over a group, `sum((d * w + m) * x)` is `d * dx * sum(w * q) + m * dx * sum(q)`, which is fine, but `q` has to be wide enough. At a signed byte per activation with a group of 32 the Metal corpus failed twelve of thirteen cases, sampled values 0.041 to 0.069 against a tolerance of 8e-2 and log probabilities 0.21 to 0.58 against 2e-1. Shrinking the group to 8 still read 0.113 to 0.202 on the head, so what is wrong is the step size and not the group, and the fix is a signed short, which is 258 times finer and puts every case back under the tolerances that were already there. That works. It also means the activation plane is two bytes a value, so the thing being saved is entirely the converts and the scale reads and not the activation being small.

The second reason is the launches, and it is the one that decides. Quantizing an activation is a kernel, and a layer reads three separate activation vectors with a matvec, so it is four more launches on a layer that has about fourteen. On the 8B that is invisible against the win, 74.4 to 97.6 on CUDA. On SmolLM2 135M it is the whole story, 269.5 to 212.3 on CUDA and 52.4 to 49.3 on Metal, because a 135M decode is launch bound and not bandwidth bound for the reason performance.md gives. SmolLM2 135M is what M2c's exit criteria are measured on.

So the integer path is worth roughly a further 1.3 times on the four bit shapes on top of what is landing here, and it costs a regression on the smallest model that has to be paid off first. Three of its four launches per layer disappear if the norm and the activation function write the quantized form as they go, which is #168. That is the order: this change now, because it is free everywhere and costs nothing anywhere, then #168, then the integer path on top of it with the launches already gone. The branch is kept rather than deleted so that the second half of it does not have to be rediscovered.

## The block width, which was never measured at all

The section above ends by asking whether the 72 per cent of peak holds once the rows are half as long, and says that if it does not then the block reduction is the next thing to look at. It does not, and it is.

`TILE` has been 128 since the matvec was written. It was not chosen for the matvec, it is the width every other kernel in `gpu_ops.mojo` launches at, and the matvec took it because it was there. The probe grew a block width parameter and the answer is that 128 is the wrong number on Metal at every shape a decode runs.

Microseconds a launch, best of five batches of sixteen, one run on the M4 at a load average of 3.7. The floor is a kernel that reads the same bytes and does no arithmetic with them.

| shape | t32 | t64 | t128 | floor |
| --- | --- | --- | --- | --- |
| q4_K 4096 by 4096 | 190 | 236 | 334 | 192 |
| q4_K 4096 by 14336 | 504 | 538 | 638 | 574 |
| q8_0 192 by 576 | 26 | | 30 | 25 |
| q8_0 576 by 576 | 29 | | 39 | 27 |
| q8_0 1536 by 576 | 40 | | 89 | 31 |
| q8_0 576 by 1536 | 33 | | 42 | 31 |
| q8_0 49152 by 576 | 818 | | 2769 | 315 |
| q8_0 4096 by 14336 | 651 | | 600 | 558 |

Two things cost at 128 and both get worse as the row gets narrower. The reduction is a tree over the whole block, so what it costs is the block width and not the work in it. And a 576 column row at eight values a thread is 72 threads of work, so 56 of the 128 threads arrive at that reduction with nothing in them. The last row is the control: a byte wide type at a 14336 column row does not care about the width and if anything prefers 128 by eight per cent, which is what says this is about the shape of the row and not about the arithmetic in the loop.

The largest single number on that table is the output head. 49152 by 576 is a fifth of what a SmolLM2 decode reads and it was running at a tenth of the floor.

End to end on the M4, the same binary either side of the one constant, decode of 128 tokens after a 1121 token prompt, best of two alternating runs at a load average between 3.3 and 7:

| model | t128 | t32 | tokens a second |
| --- | --- | --- | --- |
| SmolLM2 135M q8_0 | 3129 ms | 2296 ms | 40.9 to 55.7 |
| Qwen 2.5 0.5B q4_K_M | 5025 ms | 2761 ms | 25.5 to 46.4 |
| Llama 3.1 8B q4_K_M | 7115 ms | 4781 ms | 4.5 to 6.7, at 32 tokens |

16 and 32 tie on all three models and 64 is already worse, so the constant is 32, which is also the Metal simd width. That looked like it mattered for what comes next as well as for what landed, since at one simd group a row the reduction can become a shuffle with no shared memory and no barrier in it at all. The section after this one is that shuffle being written and measured, and it is worth nothing.

It is not exact and it is not meant to be. Nothing about the arithmetic changed, but a partial sum belongs to a different thread than it did and the reduction tree over it is two levels shallower, so the additions happen in a different order. The whole logit corpus was run on Metal on both widths. All thirteen cases pass on both, the same eleven of them report the same number of swaps, the one case with a greedy pick different from llama.cpp picks the same token at the same distance, and every figure moves in the fifth or sixth significant digit. The worst case in the corpus reads 0.029897989 at 32 against 0.029897009 at 128, against a tolerance of 8e-2.

CUDA keeps 128 until the same sweep runs on a 4090, which is #228. A block there is four warps of 32, so the arrangement that wins on Metal is one warp a row, and whether that starves a scheduler that holds many more blocks a multiprocessor than an M4 does is a real question rather than a formality.

## The shuffle that was worth nothing, and what that says

Once a matvec block is one simd group the reduction over it does not need shared memory. `planar_matmul_kernel` has reduced through `lane_group_sum` since `MM_TILE` was measured at 32, and the objection written against doing the same in the matvec, that a shuffle needs the warp width and that is a target constant in a kernel, is not true: `tile` is already a parameter and `WARP_SIZE` is what the target reports, so `tile <= WARP_SIZE` is the kernel asking whether its own block still spans more than one warp and it resolves at compile time at every launch site.

So it was written that way. Five rounds through shared memory and the six barriers around them become five shuffles, and the block stops holding a scratch array it touches only in its last few instructions. On the same protocol as the table above, three alternating runs of each binary at a load average between 2.8 and 5.9:

| model | tree | shuffle |
| --- | --- | --- |
| SmolLM2 135M q8_0 | 2260, 2262, 2282 ms | 2260, 2286, 2323 ms |
| Qwen 2.5 0.5B q4_K_M | 2733, 2816, 3268 ms | 2704, 2933, 3112 ms |

That is a wash on both, inside the spread of the runs either way, so the change was reverted rather than shipped. It is not free: it changes the order the partial sums are added in, and paying for that with nothing in return is a worse trade than leaving the tree alone.

The useful part is what the negative result rules out. Narrowing the block from 128 to 32 was worth 1.4 to 1.8 times, and removing the reduction underneath it entirely is worth zero, so what 128 was costing was never the reduction itself. It was the 56 of 128 threads that arrived at a 576 column row with no work in them. Once every thread in the block has something to do, the reduction over it is not on the critical path at all.

It also says the matvec is close to done as a place to look. The three models fit a straight line in bytes read per token: 136 MiB at 17.6 ms, and 4.6 GiB at about 149 ms, which is a slope of 29.4 ms a GiB and an intercept of 13.6 ms a token that does not depend on the model at all. The slope is 34 GB/s against an M4 that does about 120, and the intercept is most of what a small model spends. A decode layer here is a norm, three projections, two ropes, an attend, an output projection and an add, then a norm, a gate, an up, an activation, a down and an add, so a thirty layer token is around 450 launches, and `scripts/launch_probe.mojo` puts a launch on a quiet M4 at 19.6 us with 17.6 of that on the host thread before the device is told anything. 450 launches is 8.8 ms, which is the same order as the 13.6 ms the fit asks for and most of a 17.6 ms token.

Decode on the small models is launch bound. That is why the shuffle did nothing, and it is why the remaining distance to a rival is not in this kernel. It is in how many times a token crosses the driver, which is #170 stage two.

## Order

Pack the quant plane first and leave the scales at float32. That is 9574 MiB to 6475 MiB on the 8B, it is bit exact against the current layout because the integers written are the same integers, and it can be verified by running the corpus with the old cache and the new one and comparing logits with no tolerance at all.

Narrow the scale planes to float16 second, as its own change, because it is the only part of this that moves a number. It is worth 6475 MiB to 5518 MiB and it needs the corpus run with a tolerance and a statement of what the tolerance is.

Bit plane the five and six bit types third, which takes q6_K from eight bits of quant to six and gets the 8B to 5153 MiB. It is worth 365 MiB on this model and it is the fiddliest part, so it goes after the two that are worth more and are simpler.

That order was written when this was a memory change and it did not survive the measurement above. The bit planes went second and the float16 scales third, because #203 found the Metal matvec at the byte floor and a byte removed there is time and not only space. Swapping the two cost nothing, since they touch different planes of the same row. All three have landed and the 8B is at 5151 MiB against the 5153 this section predicted, so the arithmetic at the top of the page was right about every step of it and the only thing the measurement changed was which step went first.

Fix `scripts/bench.py` to report `phys_footprint` on macOS rather than `ru_maxrss`, before any of the above, so that the Metal numbers in bench.md mean what they say while this work is happening. There is no equivalent problem on Linux, where the CUDA weights are genuinely not host memory and `ru_maxrss` is right.

## What this does not answer

Whether a repack cache should exist at all once the layout is within ten per cent of the file. At 4.5 bits for q4_K the honest alternative is reading ggml blocks on the device directly, which is what llama.cpp's own CUDA and Metal kernels do, and it would remove the cache, the second copy on disk, the load stage that reads it, and the layout version problem in one move. The cost is that the eight per type unpacking loops come back, on the device this time. MAX's own GPU quantized matmul, `matmul_gpu_qint4` at `quantization/qmatmul_gpu.mojo:1773`, takes packed int4 with a group size, which is evidence that packed is the shape a GPU kernel wants and not evidence either way about blocks. This is worth a measurement before M3 and not a rewrite now.

Whether the 72 per cent of peak bandwidth holds once the rows are half as long. A shorter row is less work to amortise the block reduction over, and the reduction is a shared memory tree rather than a warp shuffle. If the packed kernel lands under 72 per cent then that reduction is the next thing to look at. Answered twice, in the two sections above. It does not hold, by a factor of ten at the narrowest shape, and the fix was the block width rather than the reduction: the shuffle was written, measured and reverted, because with a block that is one simd group wide the reduction is not what the kernel is paying for.

What the same sweep says on CUDA, which is #228 and needs a 4090 that is answering ssh.
