# Logit agreement with llama.cpp

Issue #30. Every kernel in `molla.nn` has a test that says it computes the right thing, and none of them says the kernels are wired together in the right order. A correct attention with the wrong rope base, a correct norm reading the wrong weight, a correct head tied when the file says untied: each of those is a stack of pieces that all pass their own tests and produce fluent text from the wrong distribution. This is the thing that catches those.

The shape of it is one program that writes down what llama.cpp computed and one that checks molla against what was written down. `scripts/gen-logits.py` needs llama.cpp on PATH and runs on the laptop, `scripts/logit_oracle.mojo` needs neither and runs anywhere, and the references in `scripts/logits` are committed so that the second one does not depend on the first.

## What is in the corpus

Fourteen cases over three models, nine quantizations and three prompts.

| Case | Model | Type | Prompt |
| --- | --- | --- | --- |
| smollm2-q8_0-capital | SmolLM2 135M Instruct | q8_0 | The capital of France is |
| smollm2-q8_0-count | SmolLM2 135M Instruct | q8_0 | One, two, three, four |
| smollm2-q8_0-code | SmolLM2 135M Instruct | q8_0 | def add(a, b): return |
| smollm2-f16-capital | SmolLM2 135M Instruct | F16 | The capital of France is |
| smollm2-q4_0-capital | SmolLM2 135M Instruct | q4_0 | The capital of France is |
| smollm2-q4_1-capital | SmolLM2 135M Instruct | q4_1 | The capital of France is |
| smollm2-q5_0-capital | SmolLM2 135M Instruct | q5_0 | The capital of France is |
| smollm2-q5_1-capital | SmolLM2 135M Instruct | q5_1 | The capital of France is |
| smollm2-q4_k-capital | SmolLM2 135M Instruct | q4_K_M | The capital of France is |
| smollm2-q5_k-capital | SmolLM2 135M Instruct | q5_K_M | The capital of France is |
| smollm2-q6_k-capital | SmolLM2 135M Instruct | q6_K | The capital of France is |
| qwen25-q5_0-capital | Qwen 2.5 0.5B Instruct | q5_0 | The capital of France is |
| qwen25-q5_0-count | Qwen 2.5 0.5B Instruct | q5_0 | One, two, three, four |
| llama31-q4_k-capital | Llama 3.1 8B Instruct | q4_K_M | The capital of France is |

Nine of those files do not exist anywhere to download. `llama-quantize --allow-requantize --pure` makes them out of the q8_0 SmolLM2 that does, which is why one small model covers every block format molla reads. Requantizing loses a little more than quantizing from the original weights would, and it does not matter here, because nothing in this compares molla against the model somebody trained. It compares molla against llama.cpp reading the same bytes, and both sides read whatever came out of the requantize.

The `q4_K_M` and `q5_K_M` SmolLM2 files are less pure than their names suggest. K-quants need a row width divisible by 256 and SmolLM2 is 576 wide, so llama.cpp falls back to q5_0 and q5_1 for all but the thirty `ffn_down` tensors, which are 1536 wide and come out as real q4_K and q5_K. Qwen 2.5 at 896 wide has the same problem. Llama 3.1 at 4096 wide is the only case in the corpus where K-quants are on the attention and the head as well, which is most of why it is worth the five gigabytes it costs.

The models are not committed and the case runs without them. A case whose model is missing from `corpus/logits` is skipped and said so, and the run fails only if every case was skipped.

## How a reference gets written

Two llama.cpp programs and neither one alone is enough.

`llama-server` gives the final distribution. Asking `/completion` for `n_probs` equal to the vocabulary size returns a log probability for every token in it, all 128256 of them for Llama 3.1. The prompt goes in as a token id array rather than as text, so llama.cpp adds no beginning of sequence marker of its own and the ids in the reference are exactly the ids both sides evaluate. The generator asserts that `tokens_evaluated` came back equal to the number of ids it sent, because a silently prepended token is the failure mode that would make every number in the file wrong in a way that still looks plausible.

`llama-eval-callback` gives one line per intermediate tensor with its sum and six of its values, printed to four decimal places. The ones worth keeping are `embd`, every `l_out-N`, and `result_norm`, which are the residual stream on the way in, after each layer, and after the final norm. That is what turns a disagreement into a layer number.

Log probabilities on both sides rather than logits. A row of logits carries an arbitrary additive constant and log softmax is that same row with the constant removed, so comparing log probabilities compares the only part of the answer that means anything. molla's logits go through the same log softmax before anything is compared.

The tokenizer is out of it entirely. The reference records the ids and the oracle prefills those ids, so a tokenizer that disagrees with llama.cpp cannot make this pass or fail. It has [its own oracle](tokenizer.md) and that is where it belongs.

## Why the default backend and not the CPU one

This is the opposite of what you would guess. The first reference set was taken with `-ngl 0` to keep llama.cpp on the CPU, on the reasoning that a CPU reference is the fair comparison for a program that has only host kernels. It produced disagreements of half a per cent on the sampled values and five per cent on the log probabilities, fifty times worse than what came out of the default backend, which here is Metal.

The reason is that llama.cpp's CPU backend quantizes the activation vector to eight bits before every matmul. That is a good decision for a program that wants throughput on a CPU and it puts that backend further from exact arithmetic than molla is, so a reference taken off it is not a tighter bound on molla, it is a looser one with extra noise in it. The Metal backend keeps activations in float and lands within a thousandth of molla on the small model, which is fifty times tighter and therefore fifty times more likely to notice a real mistake.

`gen-logits.py --cpu` still takes the reference off the CPU backend, and every file records which one it was on a `backend` line, so a corpus generated the other way announces itself rather than quietly loosening what the tolerances mean.

## What molla records

`Scratch` carries a `tracing` flag and a `trace` list, both off and empty by default, and `forward` appends the residual stream to the trace when the flag is on. It lives on the scratch space rather than in an argument of its own because the scratch is the only mutable thing already threaded through every layer, a `mut` parameter in Mojo cannot carry a default value, and a thirteenth argument to `forward` would have to be passed at six call sites, five of which are tests that do not want one. Off, it costs one bool test per layer.

A model with n layers leaves n plus two snapshots per token. Snapshot zero is the embedding, snapshot k for k up to n is what layer k minus one produced, and the last is the final norm. That numbering is what makes a failure a location. A disagreement in snapshot zero is the embedding lookup or the file offsets and not any layer at all. A disagreement that starts in snapshot seventeen is layer sixteen. A case whose snapshots all agree and whose logits do not is the output head and nothing else, which is worth its own snapshot because without it a wrong head, a wrong norm and thirty layers of drift that the sums happened not to catch all look the same.

## The three comparisons

**The sum over each snapshot**, against a bound relative to the total absolute mass of that snapshot rather than to the sum itself. A sum over thousands of signed activations cancels down to far less than the numbers that went into it, so an error that is nothing against the activations can be most of the total, and measuring against the total would fail on the snapshots that happen to cancel well. The same argument [kernel.md](kernel.md) makes about dot products.

**Six values out of each snapshot**, the first three and last three of the last position, against a bound relative to the larger of a value's own magnitude and the root mean square of the vector it sits in. The floor is the part worth explaining. The error a dot product makes in any one of its outputs is set by the length of the whole input vector, not by the size of that output, so a channel that happens to sit at 0.04 in a stream whose typical channel is two cannot be resolved to a per cent of 0.04 by any arrangement of float32. Measuring it against its own magnitude alone asks for a precision that was never available and fails on whichever channel happens to be smallest.

**The distribution**, as the sixty four highest log probabilities in order plus an evenly spaced sample of two hundred and fifty six across the rest of the vocabulary. The head is what a wrong kernel moves first. The tail is what a wrong kernel that happens to keep the ranking still gets wrong.

Rank swaps inside the top sixty four are counted and printed and do not fail a case. Two tokens a thousandth apart trading places is float arithmetic. The greedy token is asserted, but only when llama.cpp itself had a clear winner: on `The capital of France is` at four bits, Llama 3.1 puts the tokens for "Paris" and "a", both with the leading space they carry, 0.032 apart in log probability, which is inside the tolerance either of them is checked against, so which one comes out first is a coin the arithmetic flips rather than an answer either program got wrong. The oracle prints that it happened and moves on.

## What the corpus measures

Worst disagreement per case on the host, from the run that set the tolerances. `sum` and `value` are the snapshot comparisons, `head` and `tail` are the two halves of the distribution. The next section has the same numbers per backend.

| Case | sum | value | head | tail | swaps |
| --- | --- | --- | --- | --- | --- |
| smollm2-q8_0-capital | 1.01e-4 | 1.41e-3 | 9.81e-3 | 9.60e-3 | 4 |
| smollm2-q8_0-count | 3.89e-5 | 1.16e-3 | 3.75e-3 | 4.47e-3 | 0 |
| smollm2-q8_0-code | 2.90e-5 | 1.07e-3 | 4.98e-3 | 7.19e-3 | 1 |
| smollm2-f16-capital | 2.69e-5 | 8.50e-4 | 3.42e-3 | 6.57e-3 | 2 |
| smollm2-q4_0-capital | 3.69e-5 | 7.61e-4 | 2.20e-3 | 4.14e-3 | 1 |
| smollm2-q4_1-capital | 5.48e-5 | 7.88e-4 | 3.53e-3 | 4.52e-3 | 3 |
| smollm2-q5_0-capital | 1.98e-5 | 4.91e-4 | 4.33e-3 | 5.57e-3 | 0 |
| smollm2-q5_1-capital | 3.65e-5 | 9.13e-4 | 3.30e-3 | 7.65e-3 | 0 |
| smollm2-q4_k-capital | 4.92e-5 | 1.31e-3 | 7.76e-3 | 1.04e-2 | 2 |
| smollm2-q5_k-capital | 3.10e-5 | 8.52e-4 | 3.42e-3 | 5.70e-3 | 0 |
| smollm2-q6_k-capital | 3.14e-5 | 8.36e-4 | 3.65e-3 | 4.55e-3 | 2 |
| qwen25-q5_0-capital | 5.15e-4 | 1.98e-2 | 7.47e-2 | 6.28e-2 | 11 |
| qwen25-q5_0-count | 4.73e-4 | 3.02e-2 | 9.60e-2 | 8.40e-2 | 11 |
| llama31-q4_k-capital | 1.64e-4 | 3.06e-2 | 6.68e-2 | 6.12e-2 | 12 |

The tolerances are 2e-3 on the sum, 8e-2 on a value and 2e-1 on a log probability, each written down in `scripts/logit_oracle.mojo` next to the number it was set from. They have between two and four times the headroom over the worst case in the corpus, and a composition bug is not a factor of three, it is a factor of ten or a hundred. A wrong rope base moves the sum comparison by tens of per cent.

## The same corpus on three backends

One set of tolerances covers all three, which was not a given and is the main thing this section reports. Worst case over the whole corpus per target, against the same reference files.

| Comparison | tolerance | host | metal | cuda |
| --- | --- | --- | --- | --- |
| sum | 2e-3 | 5.15e-4 | 5.15e-4 | 5.15e-4 |
| value | 8e-2 | 3.06e-2 | 3.06e-2 | 3.06e-2 |
| head | 2e-1 | 9.60e-2 | 9.60e-2 | 9.60e-2 |
| tail | 2e-1 | 8.40e-2 | 8.40e-2 | 8.40e-2 |

The host column is the M4 and the numbers on server1 are the same to the digits printed. Metal is the M4 and cuda is a 4090.

Those columns are not merely close, they agree to every digit shown, and the reason to say so is that they came from three different pieces of arithmetic. The host path is scalar float32 in one thread order, the Metal path is a threadgroup reduction, and the CUDA path is a warp reduction over a different block size. The largest difference between any two of the three, on any case and any of the four comparisons, is 1.75e-5, which is four orders of magnitude below the loosest tolerance and a hundred times below the smallest number in the table above.

So the three backends disagree with each other by far less than any of them disagrees with llama.cpp, and the gap in the next section is a property of the reference rather than of a backend. It also means a per target tolerance would be three copies of one number, which is why there is not one.

## What a device run skips

`smollm2-f16-capital` does not run on a device and is reported as a skip with the reason, not as a pass. The device matvecs read the planar form of a quantized weight and there is no planar form of an f16 one, so an unquantized model has nothing on the card for them to read. That is checked off the tensor directory before anything is loaded, by `device_refusal`, which is the same check that makes `--device=metal` on an f16 model an error and `--device=auto` on one a host run with the reason printed under it.

Thirteen of the fourteen cases run on a device and the fourteenth is the only unquantized file in the corpus. It is worth keeping, because the host path does run it and an f16 model is the case where a dequantization bug cannot hide.

## The gap that is not explained

The eleven SmolLM2 cases agree with llama.cpp about twenty times more closely than the three cases on the two larger models do, and that is not understood.

It is not the quantization. SmolLM2 at four bits lands at 7.6e-4 on the sampled values, which is ten times closer than Qwen 2.5 or Llama 3.1 manage at five and four bits, and SmolLM2 at F16 is no better than SmolLM2 at q4_0. It is not one layer going wrong either. The divergence on the two larger models starts around layer three and grows smoothly from there, which is drift accumulating rather than a piece that is wired wrongly, and the sums stay within 5.2e-4 the whole way, which a wrong kernel would not do.

The likely explanation is precision inside llama.cpp rather than inside molla. Its Metal backend takes a different matrix multiply path once a tensor is large enough, and that path holds its tiles in float16, which loses a thousandth of relative precision per multiply on a model where SmolLM2 stays on the path that does not. That is a guess with no measurement behind it and it is written here as a guess.

#145 was expected to settle it and did not, which is worth recording because the result is informative in the other direction. Running the corpus through molla's own Metal and CUDA kernels moved the gap by nothing at all: three implementations that share no reduction order land within 1.75e-5 of each other and all three sit the same twenty times away from llama.cpp on the two larger models. So whatever this is, it is not one backend of molla's being imprecise, and it is not a threadgroup or warp reduction losing what a scalar loop keeps. It is either llama.cpp's side or something all three of molla's paths do the same way, which for a matvec is accumulating the whole row in float32.

Two things make it liveable in the meantime. The greedy token matches on every case in the corpus, once the near tie on Llama 3.1 is set aside. And the tolerances are still tight enough to be worth having: three hundredths of a stream is far below what any composition error produces.

## What this does not cover

Issue #30 asks for four things and this is three of them.

**Safetensors against transformers is not here.** `bind` takes a `Gguf` and the engine has no path that loads a model from safetensors, so there is nothing to compare. `molla.model.safetensors` reads the files and stops at the directory. That half needs the engine to accept a second file format first and is its own piece of work.

**CPU against Metal against CUDA is here as of #145**, in the section above, and it did not cost the corpus a single reference file. The oracle grew a `--device` flag and a second run path that loads onto the card and traces the device forward pass, and everything downstream of that, the snapshot numbering, the three comparisons and the tolerances, is the same code reading the same files.

**Only one of the three runs on every machine.** The host column can be produced anywhere. The metal and cuda columns need a machine with that card and the models on its disk, which is two of the five boxes here, so those two columns are pasted from a run rather than checked by anything automatic.

## Running it

```sh
pixi run conformance-logits
pixi run conformance-logits-device
```

The second one asks for `auto`, so it runs on whatever card the machine has. Every line of the output names the backend that produced it, which is what stops a run on a machine with no accelerator from being read as a device result. `--device=cuda:1` and the rest of the spellings work here as they do everywhere else, so a box with two cards can be asked about each of them.

Writing the references again, which needs llama.cpp on PATH and the source models in `~/models`:

```sh
scripts/gen-logits.py scripts/logit_cases.txt scripts/logits corpus/logits
```

That writes the requantized models into `corpus/logits`, which is gitignored, and the reference files into `scripts/logits`, which is not. The first run takes a few minutes and most of it is `llama-server` starting up once per case.

There is no CI job for it, unlike the quantization and tokenizer corpora, and that is not an oversight. Those two check against files a runner can fetch in seconds. This one needs five gigabytes of model that is not downloadable, because most of the corpus was requantized locally and none of it is published, so a job on a hosted runner would skip all fourteen cases and report a pass for having done nothing. It runs on a machine that has the models, which today means the laptop.
