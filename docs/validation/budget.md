# Where an 8B token goes on a 4090

Issue #232 was opened on the claim that the 8B decode reads weights at 630 GB/s where the card gives 901, and that the missing third is occupancy or the order rows are visited in. That claim came from a two point fit and it is wrong. This page replaces it with three measurements that do not fit anything, and the answer they give is somewhere else entirely.

The model is Meta-Llama-3.1-8B-Instruct-Q4_K_M, the machine is gpc, and the command is `generate <model> <tokenizer> <prompt> 128 2048` with the prompt built the way [bench.md](bench.md) describes. Every number here is the median of three runs on an otherwise idle card.

The token was 12.97 ms when this page was written and is 9.16 ms now, because the second of the three terms below turned out to be a grid that was three quarters empty. The measurements are left as they were taken and the section that closed it is at the end.

## The three terms

A decode token is 12.97 ms at 1121 tokens of context and 8.31 ms at four. Nothing about the model changed between those two runs, so the 4.66 ms of difference is work that scales with how much context there is, and the 8.31 ms is what a token costs before any of that.

`scripts/proj_probe.mojo` measures the other end. It calls `device_matvec_into`, which is the kernel a token actually launches, at the ten shape and kind combinations the 8B has, with the cache defeated the way the cold sweep in [layout.md](layout.md) does it. The seven projections of a layer plus the output head come to 6.7 ms a token, twice in a row.

That accounts for the whole thing.

| term | at 1121 context | at 4 | how it was measured |
| --- | --- | --- | --- |
| the projections | 6.70 ms | 6.70 ms | `scripts/proj_probe.mojo`, directly |
| attention over the context | 4.94 ms | 0.28 ms | differencing two context lengths |
| everything else | 1.33 ms | 1.33 ms | what is left |
| a token | 12.97 ms | 8.31 ms | `generate`, measured |

The context term is 4.17 nanoseconds per position per token, taken from the difference of the two measured totals and then applied to both rows, which is why the columns close rather than being made to. Only the third row is a residual, and it is small.

So the projections are 52 per cent of a token at this context and the attention over the keys and values is 38. The first of those has had every optimization this project has attempted pointed at it. The second has had none.

One caveat on the first row. The probe's sixteen launches are independent, so the card can overlap the tail of one with the head of the next, and a decode's projections are a dependency chain that cannot. 6.7 ms is therefore a floor for that term and 1.33 ms is a ceiling for the residual. It does not move the reading, because the term that matters here is the middle one.

## The projections are close to done

`scripts/proj_probe.mojo` on the 4090, cold, best of five, second of two runs.

| tensor | kind | shape | bytes | a token | a launch | rate |
| --- | --- | --- | --- | --- | --- | --- |
| attn_q | q4_k | 4096 by 4096 | 10 MiB | 32 | 19 us | 538 GB/s |
| attn_k | q4_k | 4096 by 1024 | 2 MiB | 32 | 6 us | 404 GB/s |
| attn_v | q4_k | 4096 by 1024 | 2 MiB | 16 | 6 us | 387 GB/s |
| attn_v | q6_k | 4096 by 1024 | 3 MiB | 16 | 6 us | 559 GB/s |
| attn_output | q4_k | 4096 by 4096 | 10 MiB | 32 | 17 us | 616 GB/s |
| ffn_gate | q4_k | 4096 by 14336 | 35 MiB | 32 | 46 us | 794 GB/s |
| ffn_up | q4_k | 4096 by 14336 | 35 MiB | 32 | 46 us | 794 GB/s |
| ffn_down | q4_k | 14336 by 4096 | 35 MiB | 16 | 46 us | 783 GB/s |
| ffn_down | q6_k | 14336 by 4096 | 49 MiB | 16 | 61 us | 837 GB/s |
| output | q6_k | 4096 by 128256 | 438 MiB | 1 | 494 us | 929 GB/s |

4.72 GiB a token at 749 GB/s overall. The card is rated at 1008 and the cold sweep in [layout.md](layout.md) reads 1040 GB/s at these shapes with the arithmetic taken out, so the big projections are within a fifth of what the memory will give and the output head is within a tenth.

Half of Q4_K_M is q6_k, which no probe had ever timed. It is the wider form, seven bits a value in molla's planar layout against q4_k's five, and it is not slower. The q6_k down projection reads 40 per cent more bytes than the q4_k one and reads them faster.

## What is left in them is the grid, not the loop

Four of those rows have almost the same row width and differ only in how many rows there are, which is the grid.

| rows | rate |
| --- | --- |
| 1024 | 404 GB/s |
| 4096 | 538 GB/s |
| 14336 | 794 GB/s |
| 128256 | 929 GB/s |

Bandwidth is a function of grid size and nothing else across that range. A launch of 1024 blocks moves 2 MiB in 6 microseconds, and 2 MiB at the rate the same kernel reaches on a large grid would be 2. The missing 4 microseconds is the launch, which [performance.md](performance.md) priced at 4.9 for an empty kernel and `scripts/launch_probe.mojo` puts at 4.2 to 6.2.

Add it up over a token. If every projection ran at the 929 GB/s the output head reaches, the term would be 5.46 ms instead of 6.70, so 1.24 ms of a token is the price of small grids, and almost all of it is in the five shapes with 4096 rows or fewer. Turning the fused kernel on for this model, which `fused_by_default` currently refuses because the layer weights exceed a gibibyte, is worth 4.6 per cent measured: 12.37 ms a token against 12.97. That is the same effect from the other side, and it is available today by changing a constant.

## The attention is not close to done

`device_attend` launches `grid_dim=(spec.heads, tokens, 1)`, which during decode is 32 blocks of 128 threads. The table above says a grid of 1024 blocks already leaves half the bandwidth on the floor. This is 32.

The bytes are not the problem. The key and value cache is float32, 8 key heads of 128 at four bytes for each of keys and values, which is 8 KiB a layer a position and 256 KiB a position across 32 layers. At 1185 positions that is 296 MiB, and the term costs 4.94 ms, so the path is moving bytes at 63 GB/s while the projection kernel next to it moves them at 749.

Grouped query attention makes the issued figure four times the resident one, because there are 32 query heads over 8 key heads and the grid is one block per query head, so the same key head is read by four blocks. Even counting it that way the rate is 251 GB/s. The distinct keys and values of one layer are 9.7 MB and the card has 72 MB of L2, so those four reads are not four trips to memory, and 63 GB/s is the honest number for what leaves the memory.

This is where the next of the token is, and #204 does not reach it. Halving the cache to float16 halves 296 MiB to 148, and a path running at a twelfth of the memory rate does not get twice as fast when given half as much to read.

## Which it now is, because the grid was the whole of it

`scripts/attend_probe.mojo` measures the kernel rather than subtracting two decodes, and it agrees: 114 microseconds a layer at 1185 keys, which is 3.67 ms over 32 layers. Then it reads the same 1185 keys with a taller grid, by asking for more tokens than a decode has, so that the bytes stay where they are and only the block count moves.

| blocks | time | work |
| --- | --- | --- |
| 32 | 111 us | 1 token |
| 64 | 112 us | 2 tokens |
| 128 | 112 us | 4 tokens |
| 256 | 170 us | 8 tokens |
| 512 | 310 us | 16 tokens |

Four times the work for the same time. The card was three quarters idle and a decode had no way to ask it for more, because a decode has 32 query heads and that was the whole grid.

So `device_attend` now cuts the keys into slices, one block each, and joins them in a second launch. A slice subtracts its own maximum and writes its weighted value sum unnormalised alongside that maximum and its sum, and the join rescales each by the gap between its maximum and the row's. The arithmetic is exact rather than approximate, and 48 greedy tokens at 1122 tokens of context come back byte identical to what the single kernel produced.

| context | one kernel | split | speedup | 32 layers |
| --- | --- | --- | --- | --- |
| 64 | 13 us | 14 us | 0.9x | 0.45 ms |
| 256 | 26 us | 18 us | 1.4x | 0.58 ms |
| 512 | 49 us | 19 us | 2.5x | 0.61 ms |
| 1185 | 111 us | 32 us | 3.4x | 1.03 ms |
| 2048 | 184 us | 45 us | 4.0x | 1.44 ms |

The first row is a context too short to cut, which takes the single kernel unchanged, and the microsecond between the two is noise.

End to end on the 8B at 1121 tokens of context, three alternating runs each, the token goes from 13.03 ms to 9.16. That is 3.87 ms off, and the context term this page measured at 4.94 ms is now 1.03, so the two agree to within a tenth of a millisecond.

| term | at 1121 context | share |
| --- | --- | --- |
| the projections | 6.70 ms | 73 per cent |
| attention over the context | 1.03 ms | 11 per cent |
| everything else | 1.43 ms | 16 per cent |
| a token | 9.16 ms | |

The residual has gone from 1.33 ms to 1.43. A split token launches a second kernel per layer, which is 32 more launches at the 4.2 microseconds `scripts/launch_probe.mojo` prices them at, and that is 0.13 ms. The caveat at the top of the page applies to the difference either way.

A target of 256 blocks is what ships. 1024 was tried and is worse at the contexts that matter, 35 microseconds at 1185 against 32 and 47 at 2048 against 45, because past the point where the card is full the only thing more slices buy is a longer join.

The reading has moved back to the first row. The projections were 52 per cent of a token and are now 73, and the 1.24 ms of small grid overhead priced two sections above is now a seventh of a token rather than a tenth.

## What this page corrects

The 630 GB/s figure in #232 was a fitted parameter. The fit had two terms, a fixed cost a layer and a bandwidth, and it reproduced the measured token time and predicted Qwen to within 5 per cent, which is what made it look like a measurement. It is not one. Its fixed term came to 5.73 ms a token where the launch probe prices the same launches at about 1.9, and its bandwidth term came to 630 where the kernel measured directly at the same shapes gets 749. The two errors are in opposite directions and cancel, which is exactly what a two point fit with two free parameters will do to any pair of numbers you hand it.

The general lesson is the one [layout.md](layout.md) already records about the matvec probe reading out of L2. A number that was never measured on the thing it describes will agree with the total and disagree with the parts.
