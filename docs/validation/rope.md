# Rope and attention

The third piece of issue #26. The kernels in [kernel.md](kernel.md) are position blind: a matvec does not know or care where in a sequence its input came from. This is where position enters, and where a query gets compared against the keys that came before it.

## Rope is one idea and four workarounds

Rope encodes position by rotating pairs of numbers inside each head. Pair `i` turns at `base ** (-2i/d)` radians per token, so the first pair goes round every few tokens and the last takes longer than any context anybody has trained on. An attention score between two positions then depends on the angle between them, which is their distance and nothing else. That part is one line of trigonometry and it has not changed since the paper.

Everything since is people running models past the context they were trained on.

| Scheme | What changes | Where it lives |
| --- | --- | --- |
| None | nothing | `scale` is 1, `ext_factor` is 0 |
| Linear | the position is divided by the factor | `scale` is `1 / factor` |
| NTK aware | the base goes up instead | `RopeSpec.ntk` computes a base, then it is an ordinary rope |
| YaRN | interpolate only the pairs that need it | `ext_factor`, a ramp, and a warmed logit scale |
| Llama 3.1 | a factor per pair | already in the file as `rope_freqs.weight` |

Two of those are not implementations. NTK aware scaling is `base * factor ** (d / (d - 2))` computed once when the spec is built, and after that it is an ordinary rope, so it is a constructor and not a branch in the loop. Llama 3.1's scheme is per pair frequency factors, and llama.cpp computes them at conversion time and writes them into the GGUF, so the file arrives with the answer in it and `rotate_scaled` divides by it. Both of those are worth knowing rather than rediscovering, which is most of why this page exists.

YaRN is the one with real machinery. A pair whose wavelength is shorter than the trained context has already seen every angle and is left alone. A pair whose wavelength is longer has never completed a turn and gets interpolated fully. The ones in between are blended across a ramp between `beta_fast` and `beta_slow` turns, 32 and 1 in the paper. It also multiplies the sine and cosine by `1 + 0.1 * ln(factor)`, which is not a position correction at all: interpolation takes entropy out of the attention softmax and that puts some back.

## The pairing is not a detail

Llama pairs adjacent elements, `(0,1)`, `(2,3)`, and so on. Qwen and most others pair element `i` with element `i + d/2`. It is the same rotation in a different memory order, and it is a property of the file rather than of the model, because the conversion script permutes the query and key weights of a Llama so that the adjacent layout comes out right.

Getting it backwards does not crash and does not produce noise. Every value is still in range, the vector still has the right length, and the model still emits words. It reads as a model that is fluent for a few tokens and then wanders, which is the hardest kind of wrong to attribute.

## Where the expected values came from

`scripts/rope_ref.py` is ggml's rope loop and its `rope_yarn` written out in Python with nothing rearranged. The numbers in `tests/test_rope.mojo` are its output.

That is a transcription of the reference and not an independent derivation, and it is deliberate. The question worth answering is whether molla rotates a vector the same way llama.cpp does, because that is what a model's weights were trained against. A second derivation from the YaRN paper would answer a different question, and it would go green in exactly the case that matters, which is a place where ggml and the paper disagree.

The transcription is checked at every pair for plain rope in both pairings, linear scaling, YaRN including its ramp ends and its logit factor, and per pair frequency factors. All of it matches to float32 precision on the first run.

There are also two checks that hold whatever the reference says. A rotation preserves the length of the vector, and linear scaling by four at position eight has to give the same answer as no scaling at position two. Those catch a transcription that was pasted wrong, which the reference values by themselves cannot.

## Attention holds nothing

`molla.nn.attention` is handed a query, a run of keys and a run of values, and writes one vector per head. Where those keys came from and when they get evicted is a cache, and a cache is issue #27. Splitting it that way means the arithmetic gets checked on three keys built by hand rather than only through a model.

Keys and values are laid out with position on the outside, so token `t`, head `h`, element `d` is at `t * kv_heads * head_dim + h * head_dim + d`. That is one contiguous write per token on the way in and a forward walk per head on the way out, which is what a decode step does. ggml stores K transposed, which suits a batched prefill and not this, and changing it later is a repacking question rather than a correctness one.

Grouped query attention is not a separate path. Head `h` reads key head `h / (heads / kv_heads)`, multi head is that with the two counts equal, and multi query is that with one key head. One loop, no modes.

Four things get bolted onto the softmax by one architecture or another and all four live here rather than in the block code:

A sliding window, which Mistral, Gemma and long context Qwen alternate with full layers. Sink tokens, because a window that cuts off the first position or two costs far more than it should, since a transformer parks attention it does not need there. Logit softcapping, which is Gemma 2 squashing scores through `cap * tanh(score / cap)` so one enormous score cannot take the whole row. And a scale that is not one over the root of the head dimension, which a few models were trained with and which is a field rather than a computation for that reason.

Masked keys are skipped rather than set to negative infinity. A row of `-inf` has to be exponentiated like any other and then contributes nothing, so with a 4096 token window on a 128k context that is most of the work done for nothing. The scores that survive are written packed and the softmax runs over those.

A head that can see no key at all is an error and not an empty sum. A softmax over nothing is undefined rather than zero, and the alternative to raising is writing a NaN into the residual stream, which poisons every later layer and shows up as a model that produces one token and then garbage.

## What is checked

The attention numbers are worked out on two element vectors with at most four keys, so an expected output is a line of arithmetic. Alongside those are the properties that hold for any input: the weights sum to one, a single key comes through untouched, and the output stays inside the box its values span no matter how absurd the query. Those catch a setup that is wrong in a way the expected numbers would agree with.

The window is checked three ways on one setup, open, windowed, and windowed with a sink, so the numbers say which keys were actually looked at rather than only that something came out.

## What is not here

There is no flash attention and no online softmax, so a whole row of scores is materialised. On a long context that is a real amount of memory and it is the obvious thing to fix later. There is no batching: this attends for one query position at a time, which is a decode step, and a prefill will want to do many at once. Both belong with the cache in #27, where there is something to measure them against.
