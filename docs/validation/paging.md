# Paging is a cell pool, not a block table

Issue #31, the first item of M3. It asks for a paged KV cache and a block allocator, in the vocabulary the vLLM paper uses: sixteen token blocks, per layer block tables, copy on write on a partial block. This page is the result of reading how llama.cpp actually does the same job, and the conclusion is that most of that vocabulary describes a solution to a problem molla does not have to create.

The reference reading is in the engine notes beside the CUDA and Metal ones. What follows is the part of it that decides molla's shape.

## What llama.cpp does

Its KV cache is a flat pool of cells, one cell to one token position, shared by every sequence in flight. A cell's metadata lives in host memory and is four parallel vectors: the position it holds, a bitset of the sequences that own it, a pending rope shift, and a spatial position for the models that need one. `LLAMA_MAX_SEQ` is 256, so the owner set is 32 bytes, and the whole metadata table for a pool of 32768 cells is about 1.4 MiB. None of it is uploaded.

The only paging structure that reaches the device is a vector of cell indices, one entry a token, computed on the host once per micro batch. The write is a scatter by that vector. The read is not indexed at all: attention takes a contiguous prefix of the pool, and a mask says which of those cells the token may see, built from the same owner bitsets.

That is paging at a block size of one, and at a block size of one three questions disappear. There is no internal fragmentation, because a cell is exactly a token. There is no partial block, so there is no copy on write. And there is no block table for a kernel to walk, because the indirection was resolved on the host before the launch.

What it pays instead is the mask, and the reading of cells the sequence does not own. The prefix it attends is `used_max_p1()` rounded up to 256, so a cache holding 700 live positions is read as 768. The rounding is deliberate and is documented in the source as being there so the graph shape stops changing between batches.

## What molla should take

The cell pool, and the mask, and the batch cap. Not the block table, and not copy on write with it.

The argument is not that llama.cpp is right by authority. It is that molla's attention kernels already take a slot per position and already read a contiguous run, and `molla.engine.cache` already funnels the question of which slot a position goes in through one function on the explicit grounds that the day it stops being the identity is the day every inlined copy of it becomes a bug. The cell pool is the smallest change that makes that function do real work. A block table would be a second indirection inside a kernel that does not have one today, bought to solve a fragmentation problem that only exists once blocks are sixteen wide.

The one thing to take that llama.cpp does not have is the radix tree of #33. Its server shares a prefix only when a request lands on the slot that ran the previous one, which covers a chat turn and does not cover two unrelated requests carrying the same system prompt. Over a cell pool a radix tree keys on runs of tokens rather than on block hashes, and a node can end anywhere, which is simpler than the vLLM form rather than harder.

## What molla holds today

One `DeviceHalf` for keys and one for values per layer, so a thirty two layer model is sixty four allocations, each `context * cache_row(form, kv_width)` halves. `slot_for(pos)` returns `pos`. Attention reads zero to `filled` and gets causality from the range rather than from a mask. One sequence per session, and a session owns its cache.

Two things follow from that shape and both of them are in the way.

A sequence reserves its whole context up front whether it uses it or not. Sixteen concurrent streams at a context of 4096 on an 8B is sixteen full caches, which is 8 GiB of a 24 GiB card before anything is shared, and M3's acceptance gate is sixteen concurrent streams.

The per layer allocation is also what blocks #170 stage two. One kernel a token has to be handed the cache as one pointer with a per layer offset, not as a list of sixty four, and the weights already work exactly that way: `molla.model.load` holds one device pool and hands out sub buffers by offset. So the pool is not a new mechanism here, it is the mechanism the weights already use, applied to the one buffer that still does not.

## The stages

Four, in this order, because each one is testable on its own and the first is worth landing whether or not the rest do.

**One pool.** The device cache becomes a single allocation with a per layer offset, `DeviceHalf` gains a constructor that aliases a slice of a pool the way `load.mojo` already does with `create_sub_buffer`, and every call site keeps its type. Nothing about behaviour changes, one sequence still reserves its whole context, and the test is that the whole logit corpus is bit identical. What it buys is the shape #170 needs and one allocation in place of sixty four. The host cache stays a list a layer, because it is the float reference the device one is checked against and a list of lists costs nothing there.

**Cells.** The pool is addressed by cell rather than by position. `slot_of` stops being the identity and becomes a search over a free list, the metadata table appears on the host with a position and an owner set a cell, and the per step index vector appears as a device buffer that the store scatters through. A sequence now occupies what it has written rather than what it reserved.

**The mask.** Attention stops deriving causality from the range and takes a mask built from the cell metadata, over a window rounded up so the launch shape stops changing every token. This is the stage that costs something, and what it costs is measured below rather than guessed at. The form the mask takes is below as well, because it is not the form llama.cpp uses and the difference is worth the paragraph.

**Sharing and eviction.** Sharing as a bit set on a range and releasing as a bit cleared with the cell freed when the set empties, both of which came with the table itself because an owner set is what they are. What this stage adds on top is the two policies: a bounded ring for window and sink models, and an eviction order over the cells no running sequence holds. Both are below.

Then the wiring, which is not a fifth stage so much as the point of the first four: a forward pass takes its cells from the table and reads them through the mask. It is below.

Continuous batching, #32, sits on top of stage two and does not need stage four. Chunked prefill is not separate work: it is what a cap on the batch does to a long prompt.

## The mask is a position a cell, not a float a pair

llama.cpp builds the mask as a matrix. It is `n_kv` by the number of tokens in the micro batch, zero where a token may look and minus infinity where it may not, uploaded every micro batch. At a prefill chunk of 256 tokens against 4096 cells that is a megabyte of floats a chunk, and it grows with the batch and with the context together.

molla builds it as a vector instead: one entry a cell, holding the position that cell holds for this sequence, or a negative for a cell the sequence may not read. Attention masks a pair by comparing that entry against the query's own position, which is one compare rather than a load of a precomputed answer.

Three things come out of the same entry. Causality is the entry against the query position. Ownership is the sign, because a cell that is free and a cell that belongs to somebody else are both written as negative and neither is a case attention has to know about. The sliding window and the sink count are arithmetic on the position, which they already were, and they stay counted in positions rather than in cells, which is the thing a block table gets wrong when a sequence's blocks are not in order.

The size is the point. A vector of positions does not grow with the batch, so a chunk of 256 tokens against 4096 cells is 16 KiB rather than a megabyte, and a batch of sixteen sequences is one vector a sequence rather than one entry a token a cell. It is also the host table nearly unchanged, which means there is one place a cell's position is written and one place it is read.

What molla gives up for that is generality. A mask matrix can express anything, including the cross attention and the custom masks llama.cpp supports and molla does not. Every model molla runs is causal with an optional window and optional sinks, and all three of those are functions of a position, so the general form would be paying a megabyte a chunk to express something nothing asks for.

## What the mask costs

Eight per cent of decode attention, flat over context. That is the fourth table of `scripts/attend_probe.mojo` on the 4090, at the 8B's decode geometry of 32 query heads over 8 key heads of 128, one token, keys at float16.

| context | a run | a pool | ratio | 32 layers |
| --- | --- | --- | --- | --- |
| 64 | 10 us | 11 us | 1.05x | 0.36 ms |
| 256 | 20 us | 22 us | 1.08x | 0.70 ms |
| 512 | 21 us | 22 us | 1.08x | 0.73 ms |
| 1185 | 25 us | 27 us | 1.09x | 0.88 ms |
| 2048 | 37 us | 40 us | 1.08x | 1.29 ms |

The pool in that table is exactly as large as the context and every cell in it belongs to the sequence, so both columns read the same bytes and run the same grid. What is left in the difference is the mask on its own: one int32 a cell loaded, and a comparison of two positions where the run got its causality from where the loop stopped.

Eight per cent, and it does not grow. That is the number the stage after this is worth measuring against, and it is small for a reason worth writing down: the kernel was already reading a key row of 128 halves for every entry it masks, so one more four byte load against 256 bytes is three per cent of the traffic and the rest is the compare. A mask matrix would have been reading a float a pair over the same rows, which is the same three per cent multiplied by the number of tokens in the batch.

What the table does not measure is the fragmentation. Every cell here is live, and a real pool asked for a window rounded up to a pad holds cells that belong to nobody, which are read and thrown away. That cost is set by `CellTable.window` and by how full the pool is, not by the kernel, and it is the thing stage four's eviction policy exists to keep small.

## A window model in constant memory

`CellTable.trim` is the bounded ring, and it is `AttnSpec.sees` negated term for term. Every cell it frees is a cell attention would have masked, so trimming changes how much memory a sequence holds and cannot change a logit. That is what makes it testable without a model: the check is that what the table kept is exactly what a query at the current position can see, asked of the same spec the kernel masks with.

The sinks are pinned by the expression that makes attention read them rather than by a separate rule, which is that a position below the sink count stays visible whatever the window says. So a sink is not a special kind of cell, it is a position the condition never rejects, and the ring gets it for free.

What it buys is that a conversation of any length holds `window + sinks` cells. A hundred tokens through a window of eight with two sinks never holds more than ten, which is the test, and the same arithmetic on a model with a 4096 window is 4096 cells however long the chat runs. Under the current shape the same conversation is refused the moment it passes the context it reserved.

## Retiring, not releasing

A turn that ends does not give its cells back. It stops being running, which leaves the cells where they are holding the positions they held, so the next request that begins with the same tokens can take them with `share` instead of computing them again. That is the difference between `retire` and `release_all`, and it is what #33 gets to build on: a prefix worth caching is a retired sequence's cells, and a hit is a bit set.

The cost of keeping them is that the pool fills with turns nobody came back for, so `evict` takes them back when the next request does not fit. Least recently used, over sequences rather than over cells. A turn's cells were all written at once and are all worth the same to the request after it, so the whole sequence goes and the order is the order the turns finished in. That is 64 stamps to order rather than one number a cell, and one scan of the pool a victim rather than one a cell.

Nothing running is evicted. Preempting a stream that is mid flight is a scheduler's decision and it belongs with #34, and when the scheduler makes it the way it says so is `release_all` and then recomputing. There is no path here that moves a cell to host memory, which is the preference #31 asks for. Swapping a 4096 cell sequence of an 8B out and back is 512 MiB in each direction, about 20 ms a direction over PCIe 4, and it holds the bus while it happens. Recomputing costs a prefill the engine already has a path for. Which is cheaper is worth measuring once there is a scheduler to measure it with, and until then the one that does not need a host buffer, a transfer queue and a policy for what to do when the swap itself does not fit is the one to have.

## The wiring

The four stages built the parts. The wiring is what makes a forward pass use them, and it is one function: `DeviceKvCache.place` takes a cell for each token of a step, writes the cell indices into the vector the store scatters through, writes the window into the vector attention masks with, and queues both on the stream the kernels are queued on and ahead of them. That is the one transfer paging costs a step, and it is two index vectors rather than the megabyte of floats a matrix mask would be.

Whether a step is paged is reported by `place` rather than decided by the caller. A pass over cells that are not the positions they hold has to be paged whatever anybody wanted, because the contiguous path addresses the cache by position, so `place` can turn paging on and never turns it off. While one sequence has the pool to itself the free list hands out cells in order, every cell is its own position, and the answer is no, which is what keeps this change bit identical on the path a token takes today. The fused decode asks the same question for the same reason: it reads a run and has no mask, so it runs when `place` says the cells allow it and falls back to the unfused path when they do not.

`MOLLA_PAGED=1` sends every step down the paged path anyway. Nothing in production sets it. It exists so the cost of the indirection can be measured now, on the same models and the same cards as everything else in `docs/validation/bench.md`, rather than discovered when #32 makes paging the only way. The cost has two parts. One is the scatter, which turns a contiguous store of a row into a store through an index and reads one more integer a token a layer. The other is `PAGE_PAD`, which rounds the window a step reads up to 256 cells so the launch shape stops changing every token, and which means a short context scans cells it did not have to. At a context under the pad the paged path does the pad's worth of work where the contiguous path did the context's worth, and above it the rounding is at most 255 cells of a window that is thousands.

Measured on the 4090, Qwen 2.5 0.5B at Q4_K_M, a 512 token prompt and 128 decoded, the unfused path on both sides so that the only difference is the paging. Three pairs, run alternately. Decode is 237.0, 266.7 and 271.8 tokens a second contiguous against 247.6, 275.9 and 280.7 paged, so the paged path is three to four per cent faster, consistently, in all three pairs. Time to first token is 51 to 53 ms contiguous against a steady 54 ms paged, so the scatter costs a millisecond or two on a 512 token prefill.

Faster is not what the pad predicted, and the reason it wins anyway is the thing the pad was for. At 640 positions the paged step scans 768 cells, which is twenty per cent more work than the contiguous step's 640 keys, and it still comes out ahead because its launch is the same shape every token. The contiguous path grows its grid by one key a step and pays for the change; the paged one changes shape once every 256 tokens. So the pad is not only what keeps the launch shape from churning, it is what pays for the mask.

The load on that machine was sixteen of thirty two cores during the runs, so the magnitudes are worth less than the direction, and the direction is the same in every pair. The 8B has not been taken yet because the machine did not have the memory free to map it.

The end to end check is a prompt run twice into two caches, once contiguously and once through a pool a decoy sequence has already taken the front of, so that every token lands in a cell its position never would have. The cache rows have to match byte for byte at the offset, because the store writes the same bytes and only the index changed, and the logits have to match to a tolerance a hundred times tighter than the host comparison carries, because the only thing left to disagree about is the order a sum is folded in.

## What done means

Issue #31 says a fuzz test over random conversation trees shows cached and uncached logits matching bitwise, and that is the right gate because every failure mode here is silent. A cell that is freed while a sequence still owns it produces fluent text about the wrong context. An off by one in the index vector rotates a key to the wrong position, which the existing prefill against decode check would catch only if the two routes disagreed, and after this change they take the same route.

So the check that survives the change is the one `molla.engine.cache` was written around: prefilling n tokens and then decoding has to leave the cache holding what feeding those n tokens one at a time leaves it holding. With cells that is no longer a byte comparison of two buffers, because the two routes may place the same positions in different cells. It becomes a comparison of what each position holds, read through the index, which is the same statement one level up.

## Sizing

The pool's size is what actually decides concurrency, so it is reported rather than inferred. Free device memory after the weights and the activation headroom, divided by `2 * layers * cache_row(form, kv_width) * 2` bytes a cell, is the token capacity, and that number belongs in `/molla/runners` beside the model.

For an 8B at Q4_K_M on a 24 GiB card with f16 cells, a cell is 32 layers times 1024 halves times two for keys and values, which is 128 KiB. The weights are 4781 MiB, so about 18 GiB is left and the pool holds roughly 147000 cells. Sixteen streams at 4096 positions is 65536 cells, which fits with room over, and the same sixteen streams under the current shape would reserve sixteen full contexts of the same size and fit only because they are the same number. The difference is that the pool holds what is written and the reservation holds what was asked for, and a conversation that ends at 300 tokens gives its cells back.
