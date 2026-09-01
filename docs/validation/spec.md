# From a GGUF file to a ModelSpec

Issue #19. The reader from M0 turns a file into a key value block and a tensor directory, which is the file's vocabulary. This is the layer that turns that into the handful of facts every layer above it wants, and it is done when three model families produce a correct spec and the geometry matches what llama.cpp reports. Four families were used, the same four the M0 spike used, and the comparison is field by field against llama.cpp loading the same files.

## What it is

`molla spec <path>` prints what a model is: the architecture, the geometry, the tokenizer, what the tensor directory adds up to, and what this build can actually do with it.

Nothing in it reads a weight. The spec comes out of the metadata and the directory, so a machine can say what a model is without having the memory to run it, and pointing it at a 468 MB file costs seven milliseconds and fourteen megabytes of resident set, most of which is the process starting.

| Fixture | File size | Best of seven | Peak RSS |
| --- | --- | --- | --- |
| bge-small-en-v1.5-f16 | 64 MB | 6.3ms | 9 MB |
| SmolLM2-135M-Instruct-Q8_0 | 138 MB | 6.4ms | 10 MB |
| gemma-3-270m-it-Q8_0 | 278 MB | 6.9ms | 13 MB |
| qwen2.5-0.5b-instruct-q4_k_m | 468 MB | 7.1ms | 14 MB |

## The three decisions in it

A default is recorded as a default. A key that is absent and a key holding the value the default would have given are different situations: the second is the file agreeing with us and the first is us guessing. `Geometry` carries a flag beside each of the values that are commonly missing, and the report says which numbers came from the file. Three of the four fixtures do not state a head dimension and three do not state a rope dimension count, so on real models this is the common case rather than the corner.

An encoding carries its block geometry rather than only a name. A ggml type is a block size and a byte size per block, and those two numbers turn a tensor directory into an exact byte count. That is what makes the directory checkable, which is the second decision.

Capabilities are intersected with the engine. What a file declares and what this binary can do with it are different questions, and a model that announces vision on a build with no projector kernels should say so in a report rather than fail when the first request arrives. Today the engine answers no to everything, because nothing in this binary has produced a token yet, so every fixture below reports its declaration withheld. That answer lives in one function, so issues #26 and #27 have one place to change it.

## The directory audit

Tensors are laid out in directory order, each one starting at the next alignment boundary after the previous one ended. `audit` recomputes that from the block geometry and compares it against the offsets in the file, which checks two things at once.

The first is the size table. Those numbers are the sizes of ggml's own block structs and a wrong row does not fail, it hands back a byte count that overlaps the next tensor. Recomputing every offset in a real file and requiring each one to match is a direct check of the table against what llama.cpp wrote, and all 996 tensors across the four fixtures agree, covering F32, F16, Q8_0, Q4_K, Q5_0 and Q6_K.

The second is the file. An offset trusted from a downloaded model is a read wherever the file says to read, so the last tensor has to end inside the mapping before anything maps a weight. A truncated download and a file with a tensor moved are both caught here rather than at the point where a pointer is formed.

## Against llama.cpp

`llama-cli -v` prints what it loaded as `print_info` lines. Every field molla derives has one there to compare against.

| Field | bge-small (bert) | SmolLM2 (llama) | gemma-3-270m (gemma3) | qwen2.5-0.5b (qwen2) |
| --- | --- | --- | --- | --- |
| layers | 12 | 30 | 18 | 24 |
| trained context | 512 | 8192 | 32768 | 32768 |
| embedding | 384 | 576 | 640 | 896 |
| feed forward | 1536 | 1536 | 2048 | 4864 |
| attention heads | 12 | 9 | 4 | 14 |
| key value heads | 12, assumed | 3 | 1 | 2 |
| head dimension | 32, derived | 64, derived | 256, stated | 64, derived |
| rope dimensions | none in the file | 64, stated | 256, derived | 64, derived |
| rope base | none in the file | 100000 | 1000000 | 1000000 |
| epsilon | 1e-12 | 1e-5 | 1e-6 | 1e-6 |
| vocabulary | 30522 | 49152 | 262144 | 151936 |

Every number in that table is the number llama.cpp printed for the same file. The derived ones are the interesting half: gemma-3-270m has an embedding of 640 over 4 heads, which is 160, and a stated key length of 256, so a spec layer that computed the head dimension instead of reading it would be wrong by ninety six on the one model in the fleet where it matters. The other three do not state it and 32, 64 and 64 are what llama.cpp computes the same way.

Two fields differ in presentation and neither is a disagreement about the model.

llama.cpp prints `n_rot = 32` and `freq_base_train = 10000.0` for the bert model, and there are no rope keys in that file at all. Those are llama.cpp's unconditional defaults, printed whether or not the architecture uses rope, and bge-small-en-v1.5 has a learned `position_embd.weight` rather than a rotation. molla reports that the file carries no rope keys, which is the fact, and leaves what to do about it to the layer that will build the model.

llama.cpp prints `rope scaling = linear` for all four, and none of the four states a scaling type. Same shape of difference: that is its default for an unstated key. molla reports it as unstated.

That distinction is the whole reason this layer keeps a flag beside each default. Both tools are right about the model and only one of them can tell you afterwards which numbers the file actually contained.

## What the tests cover and what they cannot

`tests/test_spec.mojo` builds GGUF files byte by byte, the way the reader's own tests do, because the interesting cases are the ones a real model never is. A file with a tensor eight bytes past where the running total puts it. A file whose directory adds up and whose data section runs past the end. A file with a Q8_0 tensor of twenty weights, which is not a whole number of blocks and so has no honest size. A file that states a key value head count and one that leaves every optional key out, which is how the defaults get checked separately from the values.

What the hand built files cannot do is prove the block geometry is right, because the same table would be used to build the file and to read it back. That is what the four real models are for, and it is why the audit runs on them rather than only in the suite.

Capabilities are declared from the file and nothing checks the declaration against a run, because there is no run yet. A model that says it is causal and is not would pass this layer and fail at the first token. That is not a gap this issue can close.

## Reproducing it

```sh
molla spec ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
llama-cli -m ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf -n 1 -p hi -no-cnv --no-warmup -v 2>&1 | grep print_info
```

The models are not in the repository. They are the four from `docs/validation/gguf.md`, all from Hugging Face, chosen because their metadata differs in ways that matter rather than because they are small.
