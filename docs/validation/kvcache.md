# The cache is written three times and it should be written once

Issue #204 asked for an f16 KV cache and then a q8_0 one behind a flag. The first half landed in #250 and was a dtype change and nothing else. The second half is not a dtype change, and this page is why, and what the shape it needs is.

## What a q8_0 cache is worth

An 8B at a context of 2048 holds 256 MiB of f16 cache. The same cache at q8_0 is a byte a value plus a float16 a block of thirty two, which is 1088 bytes a row against 2048, so it is 136 MiB. That is 114 MiB off the card at this context and it grows with the conversation rather than with the model: at 8192 it is 1024 MiB against 568.

For scale, the 8B currently holds 5540 MiB on a 4090 at a context of 2048, of which 4781 is weights and 256 is the cache. So this is 2 per cent of the card at a short context and a fifth of it at a long one, and the long one is the case that decides whether a conversation fits.

## Why it is not a dtype change

A q8_0 block is thirty two quants and one scale, and the scale is the maximum absolute value over the whole block. So a writer owns a whole block or it owns nothing, and it cannot know the scale until it has seen every element. That is a reduction inside the write rather than a narrowing of it.

The cache today has three writers and all three write it in place.

| Writer | What it touches |
| --- | --- |
| The k and v projections | One output row each, which is one element of a cache row |
| The per head key norm | One head of the row, read and written back |
| The rope rotation | Two elements a rotation, read and written back |

None of those owns a block. The projection assigns an output row to a block or a warp, so thirty two consecutive elements of a cache row come from thirty two different blocks. The rotation reads element `i` and element `i + dim / 2` and writes both, which at a head dimension of 128 is two elements sixty four apart and therefore in two different q8_0 blocks, and a rewrite of one element of a block needs the block's scale, which depends on the elements the other threads are changing at the same time.

So the cache cannot be quantized while it is written in place. It has to be written once, after everything that transforms it has finished.

## The shape

Project into the workspace, transform there, and store the finished row once.

```text
before:  wk -> KEYS ; norm(KEYS) -> KEYS ; rope(KEYS) -> KEYS ; attend reads KEYS
after:   wk -> WORK ; norm(WORK) -> WORK ; rope(WORK) -> WORK ; store(WORK) -> KEYS
```

The value side is the same with no norm and no rotation, because a value is projected and stored and nothing happens to it in between. It still has to go through the workspace, for the same reason the key does: the projection does not own a block either.

That adds one record and one grid wide barrier a layer, and `chunk * kv_width` floats of workspace for each of the key and the value, which for an 8B decode is 4096 floats a layer and is nothing against the scores.

This landed in 0.4.18 with the cache still f16, as `OP_STORE` in the fused plan and `device_store_kv` on the unfused path. A layer's plan went from twelve records to fourteen.

## What the shape was worth on its own, before any quantization

Three things, and they are why it was worth landing as its own change.

The pairing went away from three writers. Metal has no coherent sixteen bit store, so every writer of the cache inside the fused launch has to own an aligned thirty two bit word, which is `PAIRED` and which each of the three solved differently: the projections took rows two at a time, the key norm owned a word and touched only its own two elements, and the rotation owned two adjacent rotation pairs. With the transforms in a float32 workspace none of them writes the cache at all, so all three went back to one element a thread and the store is the only writer that has to think about it.

The constraint at `_rope_record` went with it. A rotation over a head whose `rope.dim` was not a multiple of four was refused, because the rotation owned two rotations at once so that what it read was what it wrote. No model molla has met has one, and the day one does it is now a rotation rather than an error.

And the projections stopped needing an `EPI_HALF` epilogue, which is the compile time store width parameter #249 is about. That parameter had reached four kernels, so it was eight instantiations of the matvec, the matmul and the mma against four, and it is four now. The same deletion took the float16 rope kernel, the float16 norm kernel and the two half projection entry points with it, because the one thing that wanted them was a cache written in place.

The cost is the extra barrier and the extra pass. The pass is `kv_width` elements a layer against the projections that produced them, which are `kv_width` rows of a `cols` wide matrix, so it is one part in `cols`. The barrier is the term to watch and it is what the measurement below is for.

## What the barrier cost

A 4090, the 8B at Q4_K_M, 128 decode tokens, five alternating pairs at a short prompt and three at a long one, best of each. Alternating the two builds rather than running one after the other is what stops a machine that was never fully idle from deciding the answer, and the one minute load sat between 2.5 and 6.3 throughout with both halves of every pair a few seconds apart.

| measurement | before | after |
| --- | --- | --- |
| decode, 6 token prompt | 1062 ms | 1085 ms |
| prefill, 1801 tokens | 3164 ms | 3171 ms |
| decode, at 1801 of context | 1209 ms | 1227 ms |

The prefill difference is 7 ms in 3164, which is a fifth of a per cent and smaller than the spread within either build. That is what should happen. Prefill runs a chunk of tokens through a layer at a time, so the barrier is paid once for a chunk rather than once for a token, and the pass is `chunk * kv_width` elements against a matmul of that same chunk over a `cols` wide matrix.

Decode pays it once a token and it costs about 20 ms in 128 tokens at both ends of the context sweep, so 0.16 ms a token over 32 layers, which is 5 microseconds a layer for the two stores and the two barriers together. A decode token is 8.3 ms at the short context, so a layer is 260 microseconds and this is 2 per cent of it.

Not measured on Metal. This laptop has not been under a load of 25 at any point while this was being written, so a timing taken here would report the load rather than the change. Metal has the correctness side, which is the suite and the logit corpus, and it will get the timing when there is a quiet window.

## Then the quantization

With one writer the encode is local. A thread owns a block of thirty two elements of the finished row, reduces the maximum absolute value over them, and writes thirty two bytes and one float16.

The row is the same planar shape the weights use, a plane of quant bytes and then a plane of scales, because there is no reason for the cache to have a second one:

```text
row = [ kv_width quant bytes ][ kv_width / 32 float16 scales ][ pad ]
```

It lives in the same float16 buffer an f16 cache lives in, and that is the decision the whole implementation turns on. The alternative is a second cache type carried down every signature that touches keys and values, and there is nothing to gain from it: what changes at q8_0 is how a row is read, not what holds it. So there is still one allocation a layer, one pointer to hand a kernel and one space in the fused plan, and a row is measured in halves at both forms. `cache_row` in `molla.nn.repack` is the arithmetic and `CACHE_F16` and `CACHE_Q8` are the two answers. It sits in `repack.mojo` beside `SCALE_BYTES` and `LAYOUT_VERSION` because `molla.nn` cannot import `molla.engine`, and the engine's cache needs the number too.

The padding rounds a row up to eight halves, so that the next row starts somewhere a thirty two bit store can own. Without it a row of an odd number of scales puts the next row's first byte in the middle of a word, and two positions written at once would then be two threads reading and writing the same word.

On Metal a thread owns two adjacent blocks rather than one, so the two scales it writes are an aligned word. The quant bytes need no pairing of their own at either form, because a block is thirty two bytes and therefore eight whole words. That is the same answer `PAIRED` gives everywhere else and it is asked in the same place.

The read side is `cache_load`, which takes the form as a compile time parameter and is the one function all three readers call: `attend_kernel`, `attend_split_kernel` and the fused `OP_ATTEND`. The branch on the form is outside the loop over the head dimension in every one of them, so a form costs a uniform branch a key rather than a branch an element.

A note on what this saves, because it is easy to overstate. A value costs 1.0625 bytes here against 2, so a q8_0 cache is 53 per cent of an f16 one and not 25. The saving is a byte a value and the scales are the rest.

## What it cost

A 4090, the 8B at Q4_K_M, a context of 4096, four alternating pairs, best of each. The prompt is 3152 tokens, so the decode reads a cache that is most of the context rather than a few dozen rows, which is the case this is for.

| measurement | f16 | q8_0 |
| --- | --- | --- |
| cache at a context of 4096 | 512 MiB | 272 MiB |
| prefill, 3152 tokens | 5910 ms | 6369 ms |
| decode, 128 tokens at 3152 of context | 1355 ms | 1674 ms |

So it is 53 per cent of the memory and it costs 24 per cent of a decode at this context. The 96 greedy tokens off a short prompt are byte identical between the two, and at a short context the decode difference is under one per cent, because there the cache is a few dozen rows and the pass is not what a token spends its time on.

The direction is the surprise and the reason is not bandwidth. A q8 row is 6.9 MB a layer at this context against 12.9, so it is less traffic and it is still slower, which is what a read that has stopped being bandwidth bound looks like. Per element the f16 path is one half load and a convert, and the q8 path is a byte load, a factor load, two converts and a multiply. The factor load is a broadcast across the warp and costs almost nothing, so what is left is the arithmetic, and there is twice as much of it.

That is fixable, it was #258, and the next section is the fix.

## Four elements a lane, and where the twenty four per cent went

A lane owned one element of a block at a time, because `key_dot` gives a warp to a key and the lanes stride the head dimension by the warp width. So every element paid a byte load, a factor load, two converts and a multiply, and a head of 128 was four rounds of that.

Four adjacent elements a lane instead. They are one aligned thirty two bit word of quants, which is the load `coherent_load_i8` was already doing and throwing three quarters of away. They are inside one block, so they share a factor, and the factor multiplies their partial sum once rather than multiplying each of them. A head of 128 goes from four rounds of about six operations to one round of about eleven, with the same lanes doing the same total work.

It costs the order the products of a key are added in, which is allowed here for the reason `ALANES` gives: the order only has to agree between the fused path and the unfused one at the same form, and both of them are this one function.

Four at a time needs the head to start on a multiple of four and to be a multiple of four long. Every model molla has met is both. One that is not goes down the one at a time path rather than being refused, and the condition is a model constant, so the branch is uniform across the warp.

A 4090, the 8B at Q4_K_M, a context of 4096, a 3003 token prompt, four alternating rounds of the three, best of each. The load average sat between 10.9 and 13.2 throughout, which is higher than a timing should be taken at, and the spread within a column was under one per cent because all three columns were interleaved rather than run one after another.

| measurement | f16 | q8_0 before | q8_0 after |
| --- | --- | --- | --- |
| prefill, 3003 tokens | 5570 ms | 5986 ms | 5700 ms |
| decode, 128 tokens at 3003 of context | 1352 ms | 1673 ms | 1406 ms |
| decode, against f16 | | 23.7 per cent slower | 4.0 per cent slower |

So 83 per cent of what q8_0 cost a decode is gone, and prefill went from 7.5 per cent over f16 to 2.3. The soak says the accuracy did not move: over the same 8192 positions the two forms now agree on 0.992 to 0.995 of the tokens in every eighth against 0.990 to 0.997 before, the divergence is 1.4e-4 early and 1.9e-4 late where it was 1.3e-4 and 1.9e-4, and the f16 top token is still the q8_0 top token at every checkpoint. The whole stepped 8192 position pass is 95950 ms against f16's 92688, where before the change it was 119122.

The cache bytes did not change, because the store side did not. The suite compares the two allocations bit for bit over the live halves of every row and that is what says so.

The four per cent that is left is the value fold, and it does not take the same treatment. There the inner loop walks keys with the element fixed, so every iteration is a different row and a different factor and there is nothing to hoist. Four elements a thread would cut the loads, and it would also cut the threads doing the fold by four, from 128 of a 256 thread block down to 32, which is the shape #234 and #239 were fixed to get away from. Making that pay needs four times the key splits to go with it, which changes the reduction order and the split tuning together, and it wants a quiet machine to measure. #258 stays open for it.

## The flag

`--cache-type f16|q8_0`, default f16, with the accuracy note in the help text rather than in a document nobody reads. llama.cpp spells it `--cache-type-k` and `--cache-type-v` and lets the two differ, which molla should not copy until something asks for it: llama.cpp needs the split because a quantized value cache there requires flash attention and a quantized key cache does not, and molla's attention has no such split.

## What has to be true before this is believed

The logit corpus on both backends, which is the bar every layout change here has been held to, and it will not be enough on its own. A q8_0 cache rounds every key and value it stores, and a five token prompt reads back what it wrote five positions ago. What that cannot see is a rounding that accumulates over a long conversation, which is the failure mode this change actually has.

So it needs a long soak: fill the context, read back what was written thousands of positions earlier, and watch the answer stay where it was. `httpsoak` is the soak molla has and it exercises the systems layer and touches no cache at all. It runs clean and it is not evidence about this. `scripts/cache_soak.mojo` is the one that is, and the next section is what it saw.

## The soak

The comparison is teacher forced. The same fixed sequence goes through an f16 session and a q8_0 session, one token at a time, and every token is the one the text says rather than the one the model picked.

Free running generation is the more natural soak and it is the wrong measurement. The two forms disagree about one token somewhere early, the texts diverge, and every difference after that is the divergence rather than the cache. Teacher forcing holds the prefix identical at every position, so the only thing left that differs between the two runs is what the cache did to the keys and the values on the way in.

A token at a time and not in chunks, which costs two minutes a pass where a chunked prefill of the same tokens would cost fifteen seconds. A chunk reads the cache once for the whole chunk. Stepping means every position reads every position before it, which is the traffic the question is about.

Three numbers come out, at sixteen checkpoints and over every position:

| what | why it is there |
| --- | --- |
| Top one agreement, in eighths of the run | The answer to the actual question. A rounding that accumulates shows up as a later eighth agreeing less than an earlier one |
| The divergence of the whole row, in nats | Agreement counts the winner and this counts everything |
| The worst move in log probability in the head | A form that keeps the ranking and stretches the spacing moves this and moves neither of the others |

And a control, which is the part worth insisting on. It runs f16 twice and compares those two, and the answer has to be exactly zero. A run whose control is not exact is measuring the machine.

## What the soak saw

A 4090, the 8B at Q4_K_M, 8192 positions of molla's own documentation, checkpoints every 512.

The control is exact. Every one of the 8192 positions picked the same token both times and all sixteen checkpoints diverged by zero, so what follows is the cache and nothing else.

| measurement | value |
| --- | --- |
| top one agreement, worst eighth | 0.9902 |
| top one agreement, best eighth | 0.9971 |
| divergence, worst checkpoint | 3.1e-4 nats |
| divergence, mean over the first quarter | 1.30e-4 nats |
| divergence, mean over the last quarter | 1.86e-4 nats |
| worst move in log probability in the head | 0.093 |
| rank of the f16 top token under q8_0 | 1, at every checkpoint |

The eighths are 0.990, 0.997, 0.996, 0.990, 0.996, 0.990, 0.992, 0.993 in order, which is flat. The last quarter of the checkpoints diverges 1.44 times as much as the first quarter, and the checkpoints inside a quarter differ from each other by four orders of magnitude, so 1.44 is noise and not a trend. Two of the sixteen came in near 1e-8 because the text at that point was a word the model was already certain about, and one came in at 3.1e-4 for the opposite reason.

That is the result the change needed. The error a q8_0 cache introduces is bounded rather than accumulating: a key rounded at position 40 is no worse when it is read at position 8000 than it was when it was read at position 41, which is what should happen, because the rounding happens once at the store and nothing reads it back and rounds it again.

One in a hundred tokens differs. That is not nothing and it is the price on the label, and it is the same price llama.cpp's `--cache-type-k q8_0` charges for the same reason.

## What this does not answer

Whether q4_0 is worth offering after it. llama.cpp offers it and ollama documents it, and it is half the bytes again, but the accuracy argument that carries f16 does not carry down there and the soak would have to be the thing that decides rather than a default anyone copies.

Whether the store record should write the cache at all once #31 lands. A paged cache holds a sequence's positions in blocks of sixteen or thirty two tokens with a block table, which is a quantization of the position axis where this is a quantization of the width axis, so the two compose rather than collide. What changes is where the store writes, not what it writes.
