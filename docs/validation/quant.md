# ggml block formats

The first piece of issue #26. Before a kernel can multiply a weight it has to know what the weight is, and in a q4_K_M file a weight is four and a half bits inside a block of 256 that shares two scales and eight six bit multipliers. This records the layouts, how the decoder is checked, and the two places where the reading that looks right is not.

## What the formats are

Every quantized ggml tensor is a run of fixed size blocks. A block holds a set number of values and one or two scales they share, and the whole tensor is nothing but those blocks end to end, so a tensor's byte length is its element count divided by the block size times the block byte size, with no header and no padding.

| Format | Bytes | Values | Layout |
| --- | --- | --- | --- |
| f32 | 4 | 1 | the value |
| f16 | 2 | 1 | the value |
| bf16 | 2 | 1 | the top half of a float32 |
| q4_0 | 18 | 32 | `d:f16`, `qs[16]` |
| q8_0 | 34 | 32 | `d:f16`, `qs[32]:i8` |
| q4_k | 144 | 256 | `d:f16`, `dmin:f16`, `scales[12]`, `qs[128]` |
| q5_k | 176 | 256 | `d:f16`, `dmin:f16`, `scales[12]`, `qh[32]`, `qs[128]` |
| q6_k | 210 | 256 | `ql[128]`, `qh[64]`, `scales[16]:i8`, `d:f16` |

Those eight are what `molla.nn.quant` reads. A Llama 3.1 8B q4_K_M contains three of them, 193 tensors of q4_k, 33 of q6_k and 66 of f32, and a Qwen 3 dense at the same quantization is the same three. The rest are there because they cost a screenful each and a model that uses one is otherwise a model molla refuses.

The IQ formats and the two and three bit K quants are not here. Each of those needs its own codebook table, none appears in a model molla runs today, and adding one that nothing exercises is a decoder that rots. `molla.model.spec` still knows their block geometry, so `molla gguf` and `molla spec` still read the directory of a file full of them and say what it is. Knowing what a file claims to be and being able to run it are separate questions and the code keeps them separate.

## Two readings that look right and are not

**The two nibbles of a byte are not neighbours.** In q4_0 the low nibble of byte `l` is element `l` and the high nibble is element `l + 16`, so the two halves of a block are sixteen apart rather than interleaved. In q4_k and q5_k the same thing happens with a stride of 32 inside each group of 64. Read them as adjacent pairs and every value is still in range, the tensor still has the right shape, and the model still emits words. It is a permutation of correct output, which is the kind of wrong that survives a smoke test.

**q6_k puts its scale last.** Every other format here starts with a float16 scale at offset zero. q6_k ends with one at offset 208, and offset zero is the low bit plane. A decoder that assumes the scale is at the front reads two bytes of packed quants as a float16, gets a number that is usually small and occasionally enormous, and produces a tensor that is wrong by a random factor per block. There is no reason for the difference beyond the order the formats were added to ggml.

A third one is smaller but the same shape: the q6_k scales are signed and the q4_k and q5_k ones are not. An unsigned reading of `0xFF` is 255 and a signed one is minus one, so a group of sixteen values comes out with the wrong sign and roughly 255 times too large.

## The scale packing

`get_scale_min_k4` is the densest thing in any of these formats and it is worth writing out. A q4_k block has eight groups of 32 values, and each group has a six bit scale and a six bit minimum of its own, on top of the block's two float16 scales. That is 96 bits, and it is stored in exactly twelve bytes with nothing wasted.

The first four groups are easy: group `j` takes the low six bits of byte `j` for its scale and the low six bits of byte `j + 4` for its minimum. The last four are where the two spare bits per byte go. Group `j` for `j` in 4 to 7 takes the low nibble of byte `j + 4` for the bottom four bits of its scale and the top two bits of byte `j - 4` for the top two, and the high nibble of byte `j + 4` plus the top two bits of byte `j` for its minimum.

A value is then `d * sc * q - dmin * m`. That is affine and not a plain scale, and dropping the minimum term gives a tensor that is shifted rather than scaled, which shows up as a model that is fluent and wrong rather than one that produces noise.

## How it is checked

Two ways, because they catch different things.

`scripts/quant_oracle.mojo` decodes fixture files and compares every value against the `gguf` Python package, which is the reference implementation's own reader. `scripts/gen-quant.py` writes the fixtures. The comparison is exact rather than within a tolerance: both sides are decoding the same bytes with the same arithmetic in the same order, so a bit for bit match is what correct looks like, and a tolerance would let a wrong nibble order through every time the two nibbles happened to be close.

The block bytes are random rather than taken from a model, which is the part worth explaining. Weights out of a real file are well behaved. Their nibbles cluster, their scales are all the same sign, and a decoder that gets a rare path wrong still matches on them and fails later inside a matmul, which is a much worse place to find out. Random bytes hit every branch on the first block. The one thing that cannot be random is a scale, because a random pair of bytes read as a float16 is a NaN or an infinity about one time in a hundred and twenty and comparing NaN against NaN says nothing, so the scale fields are written as ordinary float16 values in a sane range and everything else is whatever the generator produced.

Point `gen-quant.py` at a GGUF file and it writes a second set of fixtures cut out of that model. Random bytes prove the decoder reads the format. Real bytes prove it is pointed at the right offset in a file somebody else wrote, which is a different mistake and one the synthetic fixtures cannot catch.

```console
$ uv run --with gguf --with numpy scripts/gen-quant.py corpus/quant 512 ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
f16  512 blocks  1024 bytes  512 values
bf16  512 blocks  1024 bytes  512 values
q4_0  512 blocks  9216 bytes  16384 values
q8_0  512 blocks  17408 bytes  16384 values
q4_k  512 blocks  73728 bytes  131072 values
q5_k  512 blocks  90112 bytes  131072 values
q6_k  512 blocks  107520 bytes  131072 values
q4_k-real  512 blocks from token_embd.weight  131072 values
q6_k-real  512 blocks from blk.0.ffn_down.weight  131072 values

$ pixi run conformance-quant
f16  512 blocks, 512 values  exact
bf16  512 blocks, 512 values  exact
q4_0  512 blocks, 16384 values  exact
q8_0  512 blocks, 16384 values  exact
q4_k  512 blocks, 131072 values  exact
q4_k-real  512 blocks, 131072 values  exact
q5_k  512 blocks, 131072 values  exact
q6_k  512 blocks, 131072 values  exact
q6_k-real  512 blocks, 131072 values  exact
every format matches the oracle exactly
```

CI runs the synthetic half at 4096 blocks per format, which is about half a million values. The fixtures are deterministic from a fixed seed so there is nothing to download and nothing to cache.

The second way is `tests/test_quant.mojo`, 47 checks that run in the ordinary suite with no Python anywhere. Those are the geometry table, the error paths, and a handful of blocks built by hand so the expected value is something a person worked out rather than something another program said. An oracle tells you that a decoder disagrees with the reference. It does not tell you which of the two is wrong, or why, and a hand built block that pins where a nibble lands does.

One of those checks is that the geometry in `molla.nn.quant` agrees with the table `molla.model.spec` uses to add up a tensor directory. Those are two separate tables on purpose, since a file can contain a type molla can size but not decode, and two tables that disagree means a tensor whose length is computed one way and read the other.

## What this is for, and what it is not for

`dequant_run` is the slow, obvious version. Nothing in a forward pass should call it on a whole tensor, because turning a 4 GB model into float32 is 15 GB of host memory spent to do arithmetic that could have been done against the packed bytes.

What it is for is being right. The fused kernels that read a block and accumulate in the same pass are the only thing fast enough to run a model and they are also the easiest place in the project to be wrong in a way that produces plausible output. So this stays, it is checked against an outside reference, and the fast path is checked against it. When the two disagree there is a bisection to run rather than a model that got slightly worse for a reason nobody can name.
