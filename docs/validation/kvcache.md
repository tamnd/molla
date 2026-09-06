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

## The flag

`--cache-type f16|q8_0`, default f16, with the accuracy note in the help text rather than in a document nobody reads. llama.cpp spells it `--cache-type-k` and `--cache-type-v` and lets the two differ, which molla should not copy until something asks for it: llama.cpp needs the split because a quantized value cache there requires flash attention and a quantized key cache does not, and molla's attention has no such split.

## What has to be true before this is believed

The logit corpus on both backends, which is the bar every layout change here has been held to, and it will not be enough on its own. A q8_0 cache rounds every key and value it stores, and a five token prompt reads back what it wrote five positions ago. What that cannot see is a rounding that accumulates over a long conversation, which is the failure mode this change actually has.

So it needs a long generation soak: fill the context repeatedly, read back what was written thousands of positions earlier, and watch the output stay coherent and the perplexity stay flat. `httpsoak` is the soak molla has and it exercises the systems layer and touches no cache at all. It runs clean and it is not evidence about this.

## What this does not answer

Whether q4_0 is worth offering after it. llama.cpp offers it and ollama documents it, and it is half the bytes again, but the accuracy argument that carries f16 does not carry down there and the soak would have to be the thing that decides rather than a default anyone copies.

Whether the store record should write the cache at all once #31 lands. A paged cache holds a sequence's positions in blocks of sixteen or thirty two tokens with a block table, which is a quantization of the position axis where this is a quantization of the width axis, so the two compose rather than collide. What changes is where the store writes, not what it writes.
