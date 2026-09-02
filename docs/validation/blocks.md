# Typed blocks and the architecture table

The fourth and last piece of issue #26. [kernel.md](kernel.md) is the arithmetic and [rope.md](rope.md) is where position enters. This is the part that calls them in the right order, and the table that says what order a given model wants.

## The bug this is built to catch

A layer that runs the mlp before attention produces numbers in the right range. So does one that normalises a sublayer's output instead of its input, or rotates a key after caching it instead of before, or feeds the value projection through the query norm. None of them crash, none of them produce NaN, and all of them give a model that emits plausible words.

That is a different failure mode from the one the kernels have, and it needs a different check. An oracle for arithmetic cannot see it, because every individual operation is correct.

So the reference in `tests/test_block.mojo` is a second implementation rather than a second source of numbers. It walks plain `List[Float32]` weights with naive loops, calls nothing in `molla.nn`, and uses `exp`, `sqrt`, `cos` and `sin` out of the standard library and nothing else. If the layer calls things in the wrong order, the two disagree.

Two implementations that agree can still both be wrong, so there is also a case with hand computed values. Identity projections, gains of one and a single key at position zero collapse a layer to `x + normed(x)`, and rope at position zero is a rotation by nothing, so the whole thing becomes arithmetic a person can check on paper.

## What is parameterised

Attention takes head counts, head width, a rope spec, a sliding window, sink tokens and a logit softcap. Per head query and key norms sit beside it. The mlp is gated or not and uses silu or gelu, which are two questions rather than one. Sublayer outputs can be normed on the way to the residual add as well as inputs being normed on the way in.

A weight the model does not have is `Tensor.none()`, which is address zero, and `present()` is the only thing that says whether it is there. The alternative is a boolean beside every weight, and two things that have to agree are one thing that can disagree.

`LayerWeights.check` runs once when a model loads and refuses a missing projection or a shape that does not match the spec. That is not defensive coding for its own sake. A missing projection is otherwise a read from address zero on the first token, and a projection with the wrong number of rows is worse than a crash, because it produces output.

## Nothing allocates

Every intermediate is in a `Scratch` the caller owns and sizes once. A decode step runs a layer once per layer per token, an 8B has thirty two of them, and an allocation per intermediate would be a thousand allocations per token for numbers that are dead within the microsecond.

`Scratch` is passed field by field into the layer functions rather than as a whole, because two fields of one struct cannot both be borrowed and mutated in the same call. That is why the signatures are long.

## Keys and values go straight into the cache

`attention_layer` writes its key and value into the caller's storage at a given slot rather than into scratch and then copying. The cache is where they are needed and this is the only place they are written, so a temporary would be a copy of a copy.

`pos` and `slot` are two arguments. They are the same number today and they are separate because the day a cache starts evicting is the day one argument becomes a bug in two places.

There is still no cache. `forward` takes one list per layer and a slot, and does not decide how long those lists are, when a slot gets reused, or what happens when the context fills. That is issue #27.

## The table

`molla.nn.arch` is a row per architecture and `docs/adding-an-architecture.md` is how to add one. What is in the table is what the architecture always does, like whether the query is normed per head, because no metadata key says so. What is in the file is what this model does, and head counts, widths, the rope base and the epsilon are read off `Geometry` and never guessed.

Three things in it are worth calling out because they are the ones that give a model that talks rather than one that crashes.

The Llama pairing. Llama pairs adjacent rope elements because the conversion script permuted the query and key weights to suit it, and everything else pairs across the half point. The first version of `rope_spec` here got this wrong on exactly one path: the scaled constructors take a pairing argument and the unscaled one does not, so an unscaled Llama came out neox. It was caught by a test asserting the pairing on a plain Llama spec, which is the check that looked most redundant when it was written.

Gemma's window alternation. Gemma 2 alternates every other layer and Gemma 3 runs five local layers then one global. The rule is llama.cpp's, `layer % pattern < pattern - 1`, so the full attention layer is the last of each run. Getting the phase off by one gives a model that is subtly worse rather than one that is broken.

Gemma's embedding scale. A looked up embedding is multiplied by the square root of the model width, which is about sixty on a 3072 wide model. Leaving it out does not degrade the output, it replaces it.

## What is checked

Ninety nine checks across the layer, the table and the model level. The layer against the independent reference with weights that are not identity, at a position that is not zero, with two keys already in the cache. The hand computed collapse case. The per head norms, checked by asserting each head of the cached key comes out with unit root mean square and that the value did not, which says the norm went to two of the three projections and not the third. The post norms, checked by scaling the gain and watching the sublayer's contribution move while the stream it joins does not.

Two tokens in a row, which is the smallest thing that is a sequence, checking that the second leaves the first one's key alone, writes its own into the next slot, and produces a different answer from the same token run with an empty cache.

The table's alternation at the boundaries, the two Gemma 3 rope bases, and every way a `Geometry` can be refused: a mixture of experts, keys and values of different widths, no epsilon, no rope base, a scaling scheme molla has not implemented, and a scaling factor with no type beside it. That last one is the interesting one, because it is what a converter writes when it half understood the config, and ignoring it gives a model that works at short context and falls apart at long.

## What this is not

No model has been run. This produces logits from weights, and every weight it has been given was built by a test. Reading a real GGUF into a `ModelWeights` and a `List[LayerWeights]` needs the cache and the loop around it, which is #27, and comparing the logits against llama.cpp is #30. Until #30 passes, `supported` in the table means the row was written carefully, not that the numbers were checked.

It is also still scalar and single threaded, like everything else in `molla.nn`. #120 is the repacking work.
