# Adding an architecture

Most of what people call a new architecture is a row in a table. This page is what to fill in, where, and what to check before believing it.

Read it alongside `src/molla/nn/arch.mojo`, which is the table, and `src/molla/nn/block.mojo`, which is the layer every row configures.

## The shape everything shares

A dense decoder layer is two sublayers and two residual adds:

```console
x = x + attention(norm(x))
x = x + mlp(norm(x))
```

Attention projects the normed stream to a query, a key and a value, rotates the query and the key by the position, scores the query against every key it can see, mixes the values by those scores, and projects the result back. The mlp projects up, applies a nonlinearity, and projects down. Above the stack there is an embedding lookup at the bottom and a norm plus an output head at the top.

Every model molla runs is that, with different answers to the questions below. If a model you want is not that, this page does not cover it and the work is larger than a row.

## The questions

| Question | Field on `Arch` | Who answers differently |
| --- | --- | --- |
| Does the mlp gate | `gated` | Everything modern gates. Older models do not. |
| Which activation | `act` | silu for Llama, Qwen and Gemma 3. gelu for Gemma 2. |
| How is rope paired | `neox` | False for Llama, true for almost everything else. |
| Are q and k normed per head | `qk_norm` | Qwen 3 and Gemma 3. |
| Do the attention projections carry a bias | `qkv_bias` | Qwen 2. Qwen 3 dropped it and Llama never had one. |
| Are sublayer outputs normed | `post_norms` | Gemma 2 and Gemma 3. |
| Is there a sliding window | `window`, `window_pattern` | Gemma alternates, Mistral does not. |
| Do windowed layers use a different rope base | `local_rope_base` | Gemma 3. |
| Are attention logits capped | `softcap` | Gemma 2, at fifty. |
| Are output logits capped | `final_softcap` | Gemma 2, at thirty. |
| Is the embedding scaled on lookup | `scale_embedding` | All the Gemmas. |

Everything not on that list comes out of the file. Head counts, widths, the rope base, the scaling factor and the norm epsilon are all metadata keys, and `block_spec` reads them off `Geometry` and never guesses. If you find yourself wanting to put a head count in the table, the file is missing a key and the fix belongs in `molla.model.spec`.

## The steps

Add the id in `src/molla/model/spec.mojo` if it is not already there, and map the `general.architecture` string onto it in `architecture_id`. Add it to `is_causal` if it generates text.

Add a row to `arch_of` in `src/molla/nn/arch.mojo`. Start from `Arch(id, name)`, which is a gated silu decoder with neox pairing and nothing else, and set only what differs. Leave `supported` false.

Add the tensor names to `tensor_names` if the model has weights the existing families do not. The order there matches the fields of `LayerWeights`, and a name a layer does not have is the empty string rather than a shorter list.

A field that adds a tensor should be refused in both directions. `LayerWeights.check` errors when the table asks for something the file does not have and when the file has something the table does not ask for, because either way the table and the file disagree about what the model is and picking a side silently gives a model that runs at full speed and writes noise.

Add checks to `tests/test_arch.mojo` for the two or three facts that would give a model that talks rather than a model that crashes. The pairing and the per head norms are always worth writing down twice. So is any alternation, because the phase is off by one in the obvious implementation.

Run a real file and compare logits against llama.cpp. Only then set `supported` to true.

## Why the table and not a branch

The alternative is `if arch == ARCH_GEMMA2` inside the forward pass, and the problem with that is not the branching, it is that the differences stop being enumerable. Six architectures with three conditionals each is eighteen places to look and no place that lists them. Written as a table, adding a model is a row, and reading the table tells you what the differences between two families actually are, which is less than most people expect.

The cost is real and worth stating. A model whose difference is not one of the fields above cannot be added as a row, and the honest response to that is a new field rather than a special case in the layer. Mixture of experts is the obvious one coming: it changes what an mlp is rather than how one is configured, and `block_spec` refuses a file with experts in it today rather than routing to the first one and producing something.

## What `supported` means

An architecture can be in the table without being supported. Being in the table is a claim about what the architecture does, which can be made from reading llama.cpp. Being supported is a claim that molla gets it right, which can only be made from running a file.

Today Llama, Qwen 2 and Qwen 3 are marked supported and the Gemmas and Phi 3 are not. That is a statement about what has been checked and not about how hard the remaining work is: the Gemma rows are written and the layer code handles post norms, softcaps, alternating windows and the two rope bases, but nobody has put a Gemma file through it and compared the numbers.

## What a layer still does not do

There is no mixture of experts, so `block_spec` refuses a file that has experts rather than routing to one of them.

Keys and values have to be the same width. DeepSeek's latent attention compresses them to different sizes and is not a row in this table.

There is no cross attention and no encoder, so this covers decoders only. Bert and the embedding models are in `molla.model.spec` because a file can be read and reported on, and they are not here because nothing runs them yet.

Attention is not batched. One query position at a time, which is a decode step. A prefill wants many at once and that lands with the cache in issue #27.
