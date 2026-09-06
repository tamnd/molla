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

**The mask.** Attention stops deriving causality from the range and takes a mask built from the cell metadata, over a window rounded up so the launch shape stops changing every token. This is the stage that costs something, and what it costs has to be measured before the stage after it is worth doing. The form the mask takes is below, because it is not the form llama.cpp uses and the difference is worth the paragraph.

**Sharing and eviction.** `seq_cp` as a bit set on a range, `seq_rm` as a bit cleared with the cell freed when the set empties, and an eviction policy over cells no sequence owns. Sliding window and sink models get a bounded ring with the sink cells pinned, which the cell form expresses directly.

Continuous batching, #32, sits on top of stage two and does not need stage four. Chunked prefill is not separate work: it is what a cap on the batch does to a long prompt.

## The mask is a position a cell, not a float a pair

llama.cpp builds the mask as a matrix. It is `n_kv` by the number of tokens in the micro batch, zero where a token may look and minus infinity where it may not, uploaded every micro batch. At a prefill chunk of 256 tokens against 4096 cells that is a megabyte of floats a chunk, and it grows with the batch and with the context together.

molla builds it as a vector instead: one entry a cell, holding the position that cell holds for this sequence, or a negative for a cell the sequence may not read. Attention masks a pair by comparing that entry against the query's own position, which is one compare rather than a load of a precomputed answer.

Three things come out of the same entry. Causality is the entry against the query position. Ownership is the sign, because a cell that is free and a cell that belongs to somebody else are both written as negative and neither is a case attention has to know about. The sliding window and the sink count are arithmetic on the position, which they already were, and they stay counted in positions rather than in cells, which is the thing a block table gets wrong when a sequence's blocks are not in order.

The size is the point. A vector of positions does not grow with the batch, so a chunk of 256 tokens against 4096 cells is 16 KiB rather than a megabyte, and a batch of sixteen sequences is one vector a sequence rather than one entry a token a cell. It is also the host table nearly unchanged, which means there is one place a cell's position is written and one place it is read.

What molla gives up for that is generality. A mask matrix can express anything, including the cross attention and the custom masks llama.cpp supports and molla does not. Every model molla runs is causal with an optional window and optional sinks, and all three of those are functions of a position, so the general form would be paying a megabyte a chunk to express something nothing asks for.

## What done means

Issue #31 says a fuzz test over random conversation trees shows cached and uncached logits matching bitwise, and that is the right gate because every failure mode here is silent. A cell that is freed while a sequence still owns it produces fluent text about the wrong context. An off by one in the index vector rotates a key to the wrong position, which the existing prefill against decode check would catch only if the two routes disagreed, and after this change they take the same route.

So the check that survives the change is the one `molla.engine.cache` was written around: prefilling n tokens and then decoding has to leave the cache holding what feeding those n tokens one at a time leaves it holding. With cells that is no longer a byte comparison of two buffers, because the two routes may place the same positions in different cells. It becomes a comparison of what each position holds, read through the index, which is the same statement one level up.

## Sizing

The pool's size is what actually decides concurrency, so it is reported rather than inferred. Free device memory after the weights and the activation headroom, divided by `2 * layers * cache_row(form, kv_width) * 2` bytes a cell, is the token capacity, and that number belongs in `/molla/runners` beside the model.

For an 8B at Q4_K_M on a 24 GiB card with f16 cells, a cell is 32 layers times 1024 halves times two for keys and values, which is 128 KiB. The weights are 4781 MiB, so about 18 GiB is left and the pool holds roughly 147000 cells. Sixteen streams at 4096 positions is 65536 cells, which fits with room over, and the same sixteen streams under the current shape would reserve sixteen full contexts of the same size and fit only because they are the same number. The difference is that the pool holds what is written and the reservation holds what was asked for, and a conversation that ends at 300 tokens gives its cells back.
