# Continuous batching

What #32 is, why the shape it takes is the one the paged cache asked for, and what has to change in the kernels before a forward pass can carry more than one sequence.

The research this rests on is in `docs/validation/paging.md` for the cache and in the llama.cpp serving notes for the loop. This is the part that is molla's own: a mixed batch needs every token to say where it is and what it may read, and today a whole pass says that once.

## What molla holds today

One session owns one cache and one sequence. `DeviceSession.run` takes a list of tokens, and every token in that list belongs to the same sequence at consecutive positions, so the pass carries one `pos` for the first of them and every kernel adds its own token index to it. Attention takes one window over the pool, because there is one sequence to have one.

That is not a limitation of the cache any more. The cache holds cells, a cell knows which sequence owns it, and `CellTable` has sixty four sequences in it already. What is single is the pass.

Three things a token cannot say for itself right now, and all three are needed before two sequences share a forward:

Where it is. `device_rope` rotates token `i` of the chunk by `pos + i`, and attention masks against the same. A decode step for four sequences has four tokens at four unrelated positions.

What it may read. `DevicePaging.held` is one window, filled for one sequence. A token of sequence B must see B's cells and must not see A's, and the two are interleaved in the pool.

How much it may read. A sequence that is two hundred tokens in and one that is four thousand tokens in are in the same batch, and the second one scanning two hundred cells is wrong while the first one scanning four thousand is waste.

## The index is per sequence, not per pool

The window a step reads today is indexed by cell: entry `c` holds the position cell `c` holds for this sequence, or a negative if it does not hold one. That is one vector for the whole pool and it is right for one sequence, because one sequence's cells are the pool's cells.

With two sequences it stops being right, for a reason that is worth stating precisely. Sequence B's cells are scattered through the same pool as A's. If the vector stays pool indexed then B's window has to cover every cell A might be using, so each of the sixteen sequences in a batch scans the whole pool. Sixteen streams at four thousand positions is sixteen scans of sixty five thousand cells where sixty five thousand cells of work exist in total. The overscan is the batch size.

So the index turns around. Instead of a vector over cells saying which position each holds, a sequence gets a vector over its own positions saying which cell each is in. `index[base + i]` is the cell holding position `i` of that sequence. Three things follow, and each of them is a simplification rather than a cost.

Causality is the loop bound again. Position `i` is the entry's own subscript, so a query at position `p` reads entries zero through `p` and there is nothing to compare. The masking that the pool indexed form does per cell is a `for` loop's end.

The scan is the sequence's own length. A sequence two hundred tokens in reads two hundred entries whatever else is in the pool. A decode rounds that up, for a reason the measurement below explains, but the rounding is a constant and not the batch size.

The upload is a token, not a window. The list only ever grows at the end while a sequence decodes, so a step appends one entry per sequence rather than rewriting a window. The whole list is written once at prefill and touched once a token after that.

The window and sink arithmetic that `AttnSpec.sees` does survives unchanged, because it was always arithmetic on a position and the position is now the subscript. A trimmed cell leaves a hole, which is a negative entry, and a negative entry is skipped exactly the way a masked cell is skipped today.

## What a batch carries

Four small vectors, filled on the host once a step and uploaded ahead of the kernels.

A cell for each token of the step, which is what the store scatters through. This exists already as `DevicePaging.slots`.

A position for each token of the step. Today this is one integer for the pass.

For each token, where its sequence's index list starts. One integer a token and not the two this used to predict, because the length is not something a token has to be told. The entries a query reads are the positions before its own, so the count is the position plus one and the position is already in the descriptor. The second integer was there to carry a length that turned out to be arithmetic.

The index lists themselves, which live on the card between steps and are appended to rather than rewritten.

At a batch of sixty four tokens that is a few hundred bytes a step of descriptor, plus one appended entry a sequence. llama.cpp uploads an `n_kv` by tokens float matrix per micro batch for the same job, which at a chunk of 256 against 4096 cells is a megabyte.

## The lists live on the card

A sequence's index list is as long as the context it was admitted for, at four bytes a position. Sixteen sequences at four thousand positions is 256 KiB, which is nothing beside the 8 GiB of cells those same sequences hold. So the buffer is allocated once at the size of the pool, a sequence is given a region of it when it is admitted, and the region is released when the sequence is.

Admission is what keeps that honest. #32's scope says a request is admitted only when its worst case KV need fits, and the index region is part of that need. It is a thousandth of it.

## Chunked prefill is the batch cap

There is no separate mechanism for it, and llama.cpp does not have one either. One batch a step is filled from every sequence that has work: a decoding sequence contributes its one token, a prefilling sequence contributes as many prompt tokens as the cap has room for. A prompt longer than the cap contributes a prefix this step and the rest next step, and the sequences that are decoding decode in between. That is chunked prefill, and it costs nothing beyond the cap that a batch has anyway.

The cap is the knob that trades time to first token against inter token latency, so it is exposed rather than fixed. A large cap gets a prompt in fast and makes every decode step behind it late. A small cap keeps every stream smooth and makes the prompt take longer to start answering.

## Preempt and resume

The scope asks that preempting a sequence and resuming it produce the same output as an uninterrupted run. With the cell table that is `release_all` and then feeding the same tokens again, and the reason it is bit identical rather than merely close is that a recomputed prefill goes through the same kernels in the same order over the same weights. There is no path here that copies a cell to host memory, which `docs/validation/paging.md` argues for at length.

The gate is worth being exact about. Greedy sampling has to pick the same token every step, and the logits have to agree to the bound that two identical passes agree to, which is exactly. If a preempted and resumed run diverges at all, something is carrying state that the cells do not.

## The stages

Five, and the first two are single sequence changes with a bit identical gate, which is the same shape the paged cache took.

**Positions a token.** `device_rope` and the attention mask take a vector of positions rather than a base. With one sequence the vector is the base and its successors, so nothing about a pass changes and the check is the logit corpus.

**The index turns around.** The pool indexed window becomes a per sequence list of cells, held on the card between steps and appended to a token. With one sequence the list is the identity and the scan is the same length it was, so again nothing changes and the check is the same. What it buys is that the overscan stops being the batch size, since a sequence reads its own length rather than the pool's frontier. It does not buy the pad's removal, and the measurement that says so is below.

**A batch of sequences.** The per token descriptor arrives, `device_forward` stops taking one `pos` and one window, and two sequences share a pass. The check is that a batch of two run together gives each of them what it gets run alone.

**The loop.** Slots, admission, and one batch a step built from everything that has work. Chunked prefill falls out. The check is sixteen streams on the 4090.

**Fairness.** FIFO by default and a round robin by session, so one long generation cannot hold a shared server. The check is that a long stream and a short one interleave.

## The pad survives, for decode only

This spec said the turnaround would let `PAGE_PAD` go, since a sequence that reads its own length has nothing to round up to. The measurement says otherwise, and the measurement wins.

On the 4090 with Qwen 2.5 0.5B Q4_K_M, a 512 token prompt and 128 decoded, the fused decode off so the paged path is what runs, five runs a column and three triples:

| scan | decode tok/s | prefill tok/s | ttft |
| --- | --- | --- | --- |
| contiguous | 284.4 / 288.3 / 281.9 | 9518.5 | 54 ms |
| own length | 269.5 / 271.8 / 268.3 | 9345.5 | 55 ms |
| rounded on a decode | 293.6 / 291.6 / 292.2 | 9345.5 | 55 ms |
| rounded on every step | 275.3 / 280.7 | 7671.6 / 8158.7 / 8031.2 | 64 ms |

Scanning its own length costs a decode about five per cent against the contiguous path. Rounding the scan up to 256 gives that back and about three per cent more, so the paged path comes out ahead of the contiguous one. This is the same effect the paged cache measured when the pad went in, and the cause has not changed: a decode that grows its grid by one key a step pays for the shape change, and one that changes shape every 256 tokens does not.

Prefill is the other direction, and the size of the difference is why the rounding is not simply applied everywhere. A prefill chunk already launches a grid that covers its own triangle, so rounding the scan up squares the triangle off and adds work nothing reads. That is fifteen per cent of prefill and nine milliseconds of time to first token, which is a bad trade for a decode effect that only applies to a chunk of one. The fourth row's decode is quoted from two of its three runs because the first was cold.

So `SCAN_PAD` rounds the scan up when the chunk is a single token and leaves it alone otherwise. That splits cleanly along the decode and prefill line without asking the caller which it is doing, because a chunk of one token is what a decode is.

The pad costs an invariant. A scan that runs past the end of a sequence reads index entries the sequence does not own, so those entries have to be negative rather than whatever was there before, and a negative entry has to be skipped by the value loop rather than trusted to carry a zero weight. `DeviceKvCache.place` clears the entries a rewind leaves above the new end, `DevicePaging.forget` clears the whole list, and both the host list and the card's copy start negative, because from the turnaround on only the written span is uploaded and an entry nobody wrote is an entry nobody sent.

## What the batch descriptor came out as

Stage three is in, so this is what a pass carries rather than what it was going to.

Two vectors, both one entry a token of the chunk. `DevicePaging.seats` is where each token sits, which stage one added, and `DevicePaging.bases` is where each token's sequence's list starts in the index. `steady` fills them for a run of one sequence, which is a base and its successors and a start of zero, and `mixed` fills them for a batch. A pass that calls neither is a pass that has not said anything, so `device_forward` calls `steady` for any caller that has not already called `mixed`.

`pos` survives as an argument and stops meaning what it meant. Everything positional now reads the descriptor, so what the argument is left doing is sizing the scratch for the deepest token in the batch. That is a real job and it is not the same job, which is why it is worth saying rather than leaving a caller to infer.

Logits are a row a sequence. `device_forward` takes a list of token indices whose logits somebody wants, one per sequence, and writes them into the scratch in that order at a vocabulary apiece. An empty list is the run's last token, which is what a single sequence wants and what every caller wanted before there was a batch. `DeviceScratch` is told how many sequences it has to hold answers for, and a pass wanting more than that is refused rather than writing past the end.

The final norm needed one word of help. It reads one row out of the residual stream, and its check for a caller that wired the wrong gain in keys off the row being the whole vector, which the first row of a wide stream also looks like. So the caller says which it means. That is the whole of the change outside the descriptor.

What is not here is the scheduler. `DeviceKvCache.place` still allocates for one sequence and writes its list at the front of the index, because handing out regions of the index and cells to more than one sequence is stage four's job and there is no admission to do it under yet. The gate lays out two regions by hand and drives `CellTable` directly, which is the honest way to test a pass that nothing above it can yet build.

## What done means

Sixteen concurrent streams on the 4090 without latency collapse, which is M3's exit criterion as well as this issue's, and preempt and resume producing output identical to an uninterrupted greedy run.

Latency collapse is the part that needs a definition rather than a feeling. The measure is inter token latency at the ninety fifth percentile against the same number for one stream alone. Sixteen streams sharing one card cannot each decode as fast as one stream does, and they should not: the card is doing sixteen times the work. What must not happen is that the total throughput falls as streams are added, or that one stream starves while another runs. So the number to hold is aggregate tokens a second rising with the stream count until the card is saturated, and the spread between the fastest and slowest stream staying inside a factor that the fair mode sets.
