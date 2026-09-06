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

The scan is the sequence's own length. A sequence two hundred tokens in reads two hundred entries whatever else is in the pool.

The upload is a token, not a window. The list only ever grows at the end while a sequence decodes, so a step appends one entry per sequence rather than rewriting a window. The whole list is written once at prefill and touched once a token after that.

The window and sink arithmetic that `AttnSpec.sees` does survives unchanged, because it was always arithmetic on a position and the position is now the subscript. A trimmed cell leaves a hole, which is a negative entry, and a negative entry is skipped exactly the way a masked cell is skipped today.

## What a batch carries

Four small vectors, filled on the host once a step and uploaded ahead of the kernels.

A cell for each token of the step, which is what the store scatters through. This exists already as `DevicePaging.slots`.

A position for each token of the step. Today this is one integer for the pass.

For each token, where its sequence's index list starts and how long it is. Two integers a token, or one if the lists are laid out so that the start is a multiple of a fixed stride, which they are not going to be.

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

**The index turns around.** The pool indexed window becomes a per sequence list of cells, held on the card between steps and appended to a token. With one sequence the list is the identity and the scan is the same length it was, so again nothing changes and the check is the same. What it buys is that the overscan `PAGE_PAD` costs today is gone, since a sequence reads its own length rather than the pool's frontier.

**A batch of sequences.** The per token descriptor arrives, `device_forward` stops taking one `pos` and one window, and two sequences share a pass. The check is that a batch of two run together gives each of them what it gets run alone.

**The loop.** Slots, admission, and one batch a step built from everything that has work. Chunked prefill falls out. The check is sixteen streams on the 4090.

**Fairness.** FIFO by default and a round robin by session, so one long generation cannot hold a shared server. The check is that a long stream and a short one interleave.

## What done means

Sixteen concurrent streams on the 4090 without latency collapse, which is M3's exit criterion as well as this issue's, and preempt and resume producing output identical to an uninterrupted greedy run.

Latency collapse is the part that needs a definition rather than a feeling. The measure is inter token latency at the ninety fifth percentile against the same number for one stream alone. Sixteen streams sharing one card cannot each decode as fast as one stream does, and they should not: the card is doing sixteen times the work. What must not happen is that the total throughput falls as streams are added, or that one stream starves while another runs. So the number to hold is aggregate tokens a second rising with the stream count until the card is saturated, and the spread between the fastest and slowest stream staying inside a factor that the fair mode sets.
