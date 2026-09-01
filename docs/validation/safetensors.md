# From a safetensors repository to a ModelSpec

Issue #20. GGUF puts a model in one file. Hugging Face spreads the same model across a directory, and M2 has to load from both, so `molla spec` now reads either and answers the same questions in the same order. The done criterion was a sharded fp16 repository producing the same spec shape as the GGUF path, and the check that turned out to be worth more is the narrower one: the same four models, read both ways, agreeing number for number.

## What it is

`molla spec <path>` on a directory reads `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` and the safetensors header, and prints the architecture, the geometry, the tokenizer, what the tensor directory adds up to, and what this build can do with it. `molla safetensors <path>` prints the container on its own, which is the tensor by tensor view.

Nothing reads a weight. The header is at the front of the file and the rest is never touched, so pointing this at a 7.3 GB two shard repository costs twenty milliseconds and twelve megabytes of resident set.

| Fixture | Size | Best of five | Peak RSS |
| --- | --- | --- | --- |
| bge-small-en-v1.5 | 129 MB | 6ms | 10 MB |
| SmolLM2-135M-Instruct | 260 MB | 8ms | 11 MB |
| Qwen2.5-0.5B-Instruct | 954 MB | 17ms | 16 MB |
| gemma-3-270m-it | 545 MB | 50ms | 50 MB |
| Phi-3-mini-4k-instruct, 2 shards | 7.3 GB | 20ms | 12 MB |

The gemma row is the odd one and it is the tokenizer, not the weights. Its `tokenizer.json` is 33 MB holding 262144 vocabulary entries and 514906 merges, and every byte of it is read. Phi-3 is fifteen times the model and a fifteenth of the time, because its tokenizer is 1.9 MB. The four small rows ran on the laptop and the Phi-3 row on server1.

## The three decisions in it

The container is checked rather than trusted. A safetensors header gives an absolute byte range per tensor, and those ranges come out of a downloaded file, so an unchecked one is a read at whatever offset the file asked for. Every range is checked against the length of the mapping, and then against the dtype and the shape: a BF16 tensor of [10, 10] is two hundred bytes and nothing else, so a header claiming a different number is describing a file nothing wrote. That second check is the safetensors equivalent of the block geometry check on the GGUF side, and it does the same job of making a directory checkable rather than merely readable.

The tokenizer is streamed rather than held. `tokenizer.json` is the only large file in a repository and all that is wanted from it is four counts and five ids. Parsing gemma's into a tree would cost about fifty megabytes of nodes to answer nine questions, so it is walked as events instead, comparing keys against the token names as they go past. That is the difference between the 50 MB row above and a 100 MB one.

Half precision is not quantization. The eight bit float types are the only thing in safetensors doing what a GGUF quantization does. F16 and BF16 are the width the weights were trained and saved at, and counting them as compressed would report every fp16 repository on the Hub as quantized. So all four fixtures report nothing quantized, while the three of them that were also converted to a quantized GGUF file report almost every byte compressed.

## Against the GGUF path

The four local fixtures are the same four models as `docs/validation/spec.md`, in their original repository form. Reading a model twice through two readers and one printer is a check that neither reader can pass alone.

| Field | bge-small | SmolLM2-135M | gemma-3-270m | Qwen2.5-0.5B |
| --- | --- | --- | --- | --- |
| architecture | bert, bert | llama, llama | gemma3, gemma3 | qwen2, qwen2 |
| layers | 12, 12 | 30, 30 | 18, 18 | 24, 24 |
| context | 512, 512 | 8192, 8192 | 32768, 32768 | 32768, 32768 |
| embedding | 384, 384 | 576, 576 | 640, 640 | 896, 896 |
| feed forward | 1536, 1536 | 1536, 1536 | 2048, 2048 | 4864, 4864 |
| heads | 12, 12 | 9, 9 | 4, 4 | 14, 14 |
| key value heads | 12, 12 | 3, 3 | 1, 1 | 2, 2 |
| head dimension | 32, 32 | 64, 64 | 256, 256 | 64, 64 |
| rope base | none, none | 100000, 100000 | 1000000, 1000000 | 1000000, 1000000 |
| epsilon | 1e-12, 1e-12 | 1e-5, 1e-5 | 1e-6, 1e-6 | 1e-6, 1e-6 |
| tokenizer | wpm, wpm | bpe, bpe | spm, bpe | bpe, bpe |
| bos, eos | 101 102, 101 102 | 1 2, 1 2 | 2 106, 2 106 | 151643 151645, 151643 151645 |
| tokens | 30522, 30522 | 49152, 49152 | 262144, 262145 | 151936, 151665 |
| merges | 0, 0 | 48900, 48900 | 0, 514906 | 151387, 151387 |
| tensors | 197, 200 | 272, 272 | 236, 236 | 291, 290 |

Each cell is the GGUF answer and then the repository answer. The epsilons differ in the last digits and are not shown that way here: GGUF stores it as an f32, so llama.cpp writes 9.999999960041972e-13 where `config.json` says 1e-12. Everything else in the top half of the table is identical, which is the result this issue wanted.

The bottom half is where the two formats say different things about the same model, and all four differences are real.

The gemma token count differs by one because `<image_soft_token>` has id 262144 in a matrix of 262144 rows. It is the image placeholder from the multimodal checkpoints, kept in the text only one, and there is no row behind it. The report says so rather than picking the larger number.

The qwen token count differs by 271 in the other direction. `tokenizer.json` holds 151665 ids and the embedding matrix has 151936 rows, because a vocabulary is padded up to a round number so the matrix divides evenly across devices. llama.cpp writes the padded array, so the GGUF file genuinely contains 151936 tokens, 271 of which are `<|extra_N|>` placeholders. Both counts are right about a different question, which is why `TokenizerSpec` now carries both.

The gemma merge count differs because llama.cpp converts gemma as a SentencePiece tokenizer with scores while the repository ships it as BPE with merges. Both tokenize the same text the same way. Only one of the two files states how.

The tensor counts differ for two different reasons. bge-small has three tensors in the repository that llama.cpp drops: `embeddings.position_ids`, which is a buffer rather than a weight, and the two `pooler.dense` tensors, which the embedding path does not use. Qwen goes the other way, with one more tensor in the GGUF file than in the repository: the repository ties the output projection to the embedding and ships one tensor, and the quantizer stored the embedding at Q5_0 and the output at Q8_0, which unties them.

One more difference matters for M2 and does not show up in the spec at all. GGUF writes the shape fastest varying dimension first and safetensors writes it row major, so the same tensor is `[896, 151936]` in one file and `[151936, 896]` in the other. Nothing here transposes anything. Issue #25 loads weights and that is where the two orders have to meet.

## The sharded repository

Phi-3-mini-4k-instruct is 7.3 GB in two shards with a `model.safetensors.index.json` naming which shard holds each of its 195 tensors. It lives on server1, since the laptop does not have the disk for it.

```
weights
  files         2 shards, written by pt, 1 dtype
  model-00001-of-00002.safetensors  128 tensors, header 14952 bytes
  model-00002-of-00002.safetensors  67 tensors, header 7808 bytes
  index         195 names, 0 not in the shard the index named, 0 in a shard and not in the index
  total size    7642159104 bytes, which is what the index says
```

Every number there was checked against a Python script that reads the two headers with `struct` and `json`: 14952 and 7808 header bytes, 128 and 67 tensors, and 7642159104 bytes of weights, which is what `metadata.total_size` in the index declares and 22776 less than the two files put together, that being the two headers and their eight byte length prefixes.

Agreement between the index and the shards is counted in both directions, because they fail differently. A name in the map with no tensor behind it is a load that stops partway through, and a tensor in a shard that the map never names is a weight nothing will look for. Both are counted rather than assumed, and both are zero here.

Shards are opened in name order rather than in the order the index mentions them. Phi-3's weight map starts at `lm_head.weight`, which is in the second shard, so the file order is 2 then 1 and a report following it reads like a bug.

## What the tests cover and what they cannot

`tests/test_safetensors.mojo` builds safetensors files byte by byte with the header written as text, for the same reason the GGUF tests build GGUF files: the interesting cases are the ones no real repository is. A header longer than the file it sits in. A length prefix with the top bit set. A tensor ending 32 bytes past the end of the data. Eight F32 elements in a 16 byte range. Five dimensions, a negative dimension, offsets the wrong way round, an entry with no dtype. An index promising a tensor no shard has, and a shard holding a tensor the index never names. A directory with two safetensors files and no index, which is a repository nothing can order.

The repository half is tested the same way, from small directories written per test: a Qwen shaped `config.json`, a `tokenizer.json` whose special tokens have to be resolved out of both the vocabulary and the added token list, a multimodal config whose geometry is one level down in `text_config`, and a bert config with `cls_token` and `sep_token` and no bos or eos.

What the built files cannot check is that the mapping is the right mapping. A `config.json` written by this test suite and read by this reader agrees with itself. That is what the four real repositories are for, and specifically what reading them through both readers is for: the GGUF numbers in the table above came from llama.cpp, so the repository column is being compared against llama.cpp at one remove.

Nothing here validates a weight. The bytes behind the offsets are never read, so a repository whose tensors are the right shape and the wrong numbers passes everything in this document. That is issue #25.

## Reproducing it

```sh
molla spec ~/models/st/Qwen2.5-0.5B-Instruct
molla spec ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
molla safetensors ~/models/st/Qwen2.5-0.5B-Instruct
```

The repositories are not in the repository. They are the four families from `docs/validation/gguf.md` in their original form, from Hugging Face, plus microsoft/Phi-3-mini-4k-instruct for the sharded case. gemma-3-270m-it is the ungated `unsloth/gemma-3-270m-it` mirror, which carries the same weights as the Google repository.
