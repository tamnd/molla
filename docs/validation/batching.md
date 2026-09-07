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

**The loop.** Slots, admission, and one batch a step built from everything that has work. Chunked prefill falls out. The check is sixteen streams on the 4090. Split in two, because admission is a thing the pass below can be tested against and the loop is a thing the runner above has to drive.

**Fairness.** FIFO by default and a round robin by session, so one long generation cannot hold a shared server. The check is that a long stream and a short one interleave.

What that came out as is below, and the short version is that the mechanism is right and the workload it was aimed at was not the one that hurts.

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

What is not here is the loop. A pass can carry a batch and nothing above it builds one, so the step that decides which sequences go into a batch is still ahead.

## What admission came out as

Stage four's first half, the part that hands out the index.

A lease is the reservation and it is a region of the index, not a count of cells. That is the one thing here worth stating plainly, because it makes two problems into one. A sequence admitted for `n` positions gets `n` index entries, and it can hold at most `n` cells because a cell it holds is a position it wrote and every position it writes has an entry. So admitting the region is admitting the cells, and there is nothing to keep in step. The index stays one buffer the size of the pool rather than a buffer per sequence, which is what the section above says it costs to hold on the card.

Regions are handed out first fit and coalesced when they come back. First fit over best fit because there are a few dozen regions at most, both policies fragment, and the one that is easier to reason about is the one whose failures can be described. Coalescing on release is what makes a pool that has been fully drained one region again whatever order the sequences left in, and that is checked rather than assumed.

A fresh cache gives its whole index to sequence zero. So a session that has never heard of admission gets the cache it always had, one region at the front as long as the context, and a scheduler's first move is to evict that and admit its own. The single sequence path did not change and does not know any of this happened.

`place_for` is `place` with two more arguments and it is where the descriptor gets filled. It takes the sequence, and it takes where that sequence's tokens sit in the step's chunk. Everything a token says about itself is written at that subscript by the call that allocated its cell, because that call is the one thing that knows both the token's position and where its region starts. Committing the descriptor is separate and happens once a step. So a batch of sixteen is sixteen `place_for` calls and one `mixed`, and nothing walks the batch a second time to work out what the first walk already knew.

## What the loop came out as

Stage four's second half, which is the thing admission was for.

`DeviceSession` is one sequence with a cache of its own and its loop is a token at a time down one stream. `DeviceBatch` is the other shape: several streams over one pool, one scratch and one pass, and its loop is a step at a time over all of them. Both exist, because a single request answered on an idle card does not want a batch's overheads and a server does not want a session per request.

A step is filled from every stream that has work, in slot order, until the cap is full. A stream reading its prompt gives as much of it as the cap has room for. A stream generating gives the one token it last produced, which is what makes a decode one token rather than two: the token sampled from this step's logits is the token that stream contributes to the next. So chunked prefill needed no mechanism of its own, exactly as the section above predicted. The cap the batch has anyway is what cuts a long prompt up, and the streams that are decoding decode in between the pieces.

A stream still owing prompt gets no answer, because the logits of a token in the middle of a prompt are a prediction nobody asked for. One that has just finished its prompt gets the answer the single sequence path samples its first token from. So the rows a step asks for are the streams that reached the end of what they owed, and that is usually all of them and occasionally none.

One thing did not fall out and had to be said. A step that turns out to carry a single token, which is the tail of any run and the whole of a run with one stream, needs a scratch sized for one token and a residual stream one row wide. The batch holds both sizes for the same reason a session does: the attention scores scale with the chunk and the context together, and the final norm reads a vector the width of what it was handed.

`molla batch` is what a person runs to see this. Several copies of a prompt at once, and it reports aggregate tokens a second, time to first token per stream, and the spread of inter token latency, which are the three numbers a single stream run has no version of. Every stream gets the same prompt and greedy settings, so every stream has to write the same tokens, and the command checks that rather than leaving it to whoever reads the text. Two sequences whose cells got mixed up would disagree there, and so would a batch that wrote one stream's logits into another's row.

## What the loop measures

On the 4090 with Qwen 2.5 0.5B Q4_K_M, a short prompt and 128 tokens a stream, one run a row:

| streams | aggregate tok/s | median | p95 | worst |
| --- | --- | --- | --- | --- |
| 1 | 335 | 3 ms | 4 ms | 5 ms |
| 2 | 378 | 5 ms | 6 ms | 6 ms |
| 4 | 619 | 6 ms | 8 ms | 12 ms |
| 8 | 994 | 8 ms | 9 ms | 9 ms |
| 16 | 1349 | 12 ms | 13 ms | 13 ms |
| 32 | 1790 | 18 ms | 19 ms | 28 ms |

Sixteen streams produce four times the tokens one stream does and each of them waits four times as long between tokens, which is what sharing a card is. What the criterion asks about is collapse, and the shape here is the opposite of collapse: the total rises at every count, and the ninety fifth percentile sits within a millisecond or two of the median all the way up, so no stream is being served late while the others are served on time. Every count agreed token for token across all its streams.

The one place the curve is not smooth is one stream to two, where the total goes up by a tenth rather than close to double. That is not the scheduler. A step carrying one token goes down the single token path and a step carrying two goes down the general one, and the second is about twice the work for the first token it carries. After that the marginal cost of a stream is small: a step is under five milliseconds at two streams and about sixteen at thirty two, so sixteen more streams cost less than the first two did. Which says the batching is doing what batching is for, and that the crossover between the two paths is the thing to look at if the low end matters.

## What the server does with it

Stage four's third part, which is the loop above answering requests instead of a benchmark.

`molla serve --slots=N` is the whole of the interface. The default is one, so a server nobody asked keeps the behaviour it had, and a second request in flight against it gets the 503 it always got.

The pool is shared rather than divided. A server started with four slots does not give each of them a quarter of the pool: every request is admitted against whatever is free at the time. Dividing would make the failure predictable, which is worth something, but the failure it makes predictable is refusing a long request on a server with nothing else running, and that is worse than the one it prevents.

A request that cannot be admitted gets a 503, and the message says which of the two ran out. Slots is a wait, so retrying is the right thing and the message says so. A pool that is too small for the request is not a wait, and a request longer than the whole context is still a 400 whatever the server is doing, because retrying that one is not going to help.

Whichever connection asks for its next token steps the batch. A step carries every stream that has work, so the connection that paid for it gets one token and the others find theirs already written when they come round. Nothing schedules that: the reactor holds every connection, and one of them being inside a step is the same thing as all of them making progress. It does mean the cost of a step lands on whoever asked first, which the numbers do not show and a profile would.

Two things came out of wiring it that the loop did not need on its own. A slot has to come back when a request finishes, since a server admits one for every one it answers and a batch whose slots only ever ran out would serve four requests and refuse forever. And ending a stream is two things rather than one: a caller that matched a stop string wants the stream to stop generating while it is still reading what the stream wrote, so `halt` stops the generating and `drop` hands the region back, and the connection does the second when it closes.

## What the server measures

The same 4090 and the same model, through HTTP this time. Streaming completions, a short prompt, 128 tokens a stream, a pool of 16384 positions and a client on the same machine:

| streams | aggregate tok/s | median | p95 | worst |
| --- | --- | --- | --- | --- |
| 1 | 154 | 6 ms | 8 ms | 8 ms |
| 2 | 242 | 8 ms | 9 ms | 9 ms |
| 4 | 433 | 9 ms | 10 ms | 28 ms |
| 8 | 671 | 11 ms | 12 ms | 68 ms |
| 16 | 959 | 15 ms | 16 ms | 46 ms |
| 32 | 1217 | 23 ms | 25 ms | 382 ms |
| 64 | 1555 | 34 ms | 39 ms | 1411 ms |

Sixteen concurrent streams through the server produce six times the tokens one stream does at two and a half times the inter token latency, and the ninety fifth percentile is a millisecond off the median at every count up to sixty four. Every count agreed token for token across all its streams, so no request read another's logits. That is the exit criterion, taken where the criterion asks for it rather than in the benchmark command.

The totals are below what `molla batch` reports for the same counts, by about a third at sixteen. Framing, JSON, the event stream and a Python client on the same box are all in this number and none of them are in the other one. The shape is what matters here and the shape is the same.

The worst column is the interesting one. At thirty two and sixty four streams there is a gap of hundreds of milliseconds in the middle of somebody's stream while the p95 stays where it was, so nothing is starving, since every stream finished and the percentile did not move, and something is still wrong. This was read at the time as one stream waiting behind a wave of prompts, on the argument that a step is filled in slot order and the last request admitted has its prefill land after everybody else's. Stage five was built to test that reading and the reading was wrong. What is actually in the gap is below, under what fairness came out as.

The pool being shared is visible in the failures rather than in the table. Thirty two streams against a pool of 4096 positions is 32 times 139 positions asked for and 4096 to hand out, and the three requests that did not fit got a 503 saying so. The same thirty two against 16384 all fit. That is the trade this made: a server can refuse a request it would have had room for under a fixed division, and it can also admit one that a fixed division would have refused, and the second happens far more often.

## What fairness came out as

Stage five, which is `--fair` on `molla serve` and on `molla batch`, and which changes the offer rather than the step.

The offer is which streams are handed the cap and in what order. In slot order, which is what a batch does unless it is told otherwise, the lowest slot is offered first every step. A prompt admitted into a low slot takes the whole cap and a stream in a higher slot that was already answering gets nothing that step, and since a slot comes back when a request finishes, which slot a new prompt lands in is whatever happened to be free. Fair mode offers the decodes first, all of them, because a decode is one token and every stream in flight decoding at once is `slots` tokens, which fits any cap worth having. What is left of the cap goes to the prefills, offered from a cursor that moves on by one a step, so a prompt that was cut off by the cap is at the head of the queue next time.

The batch level check is a stream in slot one that is answering and a prompt admitted into slot zero that is exactly the cap long. In slot order the answering stream writes nothing on that step. In fair mode it writes its token and the prompt takes what is left. Both modes write the same text as the same stream run on its own, which is the thing an offer order must not change.

### It does not move the server numbers

Three runs of each mode on the 4090 with the same model and the same client, streams arriving together, 128 tokens a stream, 64 slots and a pool of 16384. The median run of the three:

| streams | slot order tok/s | fair tok/s | slot order p95 | fair p95 |
| --- | --- | --- | --- | --- |
| 16 | 829 | 819 | 20 ms | 20 ms |
| 32 | 1127 | 1105 | 28 ms | 28 ms |
| 64 | 1305 | 1339 | 47 ms | 46 ms |

And the case the fill order was supposed to be worst in, which is churn: twelve clients each sending six requests one after another, 96 tokens a request, so slots are freed and taken again all the way through. Slot order gave 892, 788 and 868 aggregate tokens a second across three runs and fair gave 921, 910 and 764. The run to run spread is larger than the difference between the modes in both tables.

A single measurement of each had shown fair ahead by a tenth, which is why there are three of each here. There was no effect and the first pair was noise.

### Where the tail actually is

The worst column is what stage five was aimed at, and the arithmetic says the offer order was never going to reach it. At sixty four streams the slowest first token is about 1.9 seconds. Sixty four prompts of eleven tokens is 704 tokens of prefill, and 704 tokens at a cap of 256 is three steps. A step that size measures about 45 ms here, taken from a 1200 token prompt answering in 215 ms on an otherwise idle server. So the prefill those requests are waiting on is about a seventh of a second and the other 1.75 seconds is something else. No ordering of prefill chunks moves a cost that is not in the prefill chunks.

The next guess was the client, since a Python client running sixty four threads on the same box as the server is a plausible way to invent one and a half seconds. It is not that either. The same wave driven by sixty four separate `curl` processes, which share no interpreter lock, gives 35 ms median and 49 ms at the ninety fifth percentile against the Python client's 36 and 45, a worst gap of 1485 ms against 1655, and a spread of 1499 ms between the first stream's first token and the last one's. Two clients with nothing in common agreeing to within a few per cent is the server.

So there is a second or more of per request work between the socket and the first step, it is serialized, and it is not the batch. That is #292 rather than a guess here. What stage five is worth in the meantime is what the batch level check shows, which is that a stream already answering is not displaced by a prompt, and the knob is off by default until there is a workload where that is the thing in the way.

## What done means

Sixteen concurrent streams on the 4090 without latency collapse, which is M3's exit criterion as well as this issue's, and preempt and resume producing output identical to an uninterrupted greedy run.

Latency collapse is the part that needs a definition rather than a feeling. The measure is inter token latency at the ninety fifth percentile against the same number for one stream alone. Sixteen streams sharing one card cannot each decode as fast as one stream does, and they should not: the card is doing sixteen times the work. What must not happen is that the total throughput falls as streams are added, or that one stream starves while another runs. So the number to hold is aggregate tokens a second rising with the stream count until the card is saturated, and the spread between the fastest and slowest stream staying inside a factor that the fair mode sets.
