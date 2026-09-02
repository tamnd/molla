# Changelog

Notable changes per release. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.8] - 2026-09-02

A model file goes in one end and English comes out the other.

This is the first release where molla runs a real model. `molla generate` reads a GGUF, prefills a prompt, decodes it and prints tokens as they arrive, with the sampling settings anyone would expect on the command line. There is no server yet and no comparison against llama.cpp, both of which are the rest of M2.

### Added

- `molla.engine.cache`, the keys and values one sequence has accumulated. Contiguous, one list per layer, float32, allocated once when the session is made. A sequence that would run past the end of the context is refused rather than wrapping onto slot zero, because a cache that wraps gives a model that is still fluent and has forgotten its instructions, and nobody reads that as an error.
- `molla.engine.session`, prefill and decode as one loop rather than two. The check that matters is that prefilling a prompt and then decoding leaves the cache byte for byte identical to feeding the same tokens one at a time, which is the statement that both routes agree about position. See [docs/validation/decode.md](docs/validation/decode.md).
- `molla.engine.bind`, which points the network at a file's bytes. By name, against the architecture table, with every shape checked before a token is computed rather than on the first forward pass.
- Qwen 2's bias on each of the three attention projections, which Qwen 3 dropped and Llama never had. Without it a qwen2 file loaded, passed every shape check and ran at full speed writing noise, because a missing bias is a constant vector absent from every head of every layer rather than a crash. A file and an architecture table that disagree about whether the biases are there is now refused in both directions. Qwen 2.5 0.5B generates English.
- q4_1, q5_0 and q5_1, the three block formats next to the ones molla already read. A Qwen 2.5 0.5B download named q4_k_m is q5_0 for every weight in it, and the type numbers in the directory are what molla believes rather than the name the file was published under. Decoded and fused, both checked against the `gguf` package, exactly. See [docs/validation/quant.md](docs/validation/quant.md).
- `molla.engine.sample`, the sampling pipeline: a grammar hook that M4 fills in, logit bias, repetition and frequency and presence penalties over a per sequence window, temperature, top-k, top-p, min-p and typical, in llama.cpp's order because that is the order every preset in circulation was tuned against. The randomness is counter based rather than a stream, so the same seed and the same prompt give the same tokens whatever else the server is doing, and greedy is the exact argmax rather than a temperature approaching zero. `molla generate` takes `--temp`, `--top-k`, `--top-p`, `--min-p`, `--typical`, `--repeat-penalty`, `--frequency-penalty`, `--presence-penalty`, `--repeat-last-n` and `--seed`, and no flags still means greedy. See [docs/validation/sampling.md](docs/validation/sampling.md).
- `molla generate <model.gguf> <tokenizer.json> "<prompt>" [n] [ctx]`, which is molla generating text for the first time. SmolLM2 135M at q8_0 and Llama 3.1 8B Instruct at q4_K_M both produce coherent English, the second of those through the precomputed rope frequency factors that Llama 3.1 ships in the file. It is scalar and single threaded, about 90 ms a token on the 135M and 5.5 seconds a token on the 8B, which is what issue #120 exists to change.

## [0.2.7] - 2026-09-02

The arithmetic a transformer is made of, checked against something outside molla.

Nothing in here runs a model. It decodes the weights, multiplies them, puts position into a query, scores it against the keys, and stacks that into layers, and every piece of it is checked against llama.cpp's numbers or against a second implementation. Loading a real file and getting tokens out is the next milestone step.

### Added

- `molla.nn.quant`, which decodes ggml blocks back to float32. f32, f16, bf16, q4_0, q8_0, q4_k, q5_k and q6_k, which covers everything a Llama 3 or Qwen 3 q4_K_M file contains. This is the slow obvious version and it stays that way: the fused kernels get checked against it and it gets checked against the reference implementation, so a fast path that drifts fails against something rather than making a model quietly worse. See [docs/validation/quant.md](docs/validation/quant.md).
- A quantization conformance corpus. `scripts/gen-quant.py` writes fixtures of random block bytes and the values the `gguf` package decodes them to, `scripts/quant_oracle.mojo` compares every one, and CI runs about half a million values per commit. The match is exact rather than within a tolerance, because both sides decode the same bytes in the same order and a tolerance would let a wrong nibble order through whenever the two nibbles happened to be close.
- `molla.nn.tensor`, a weight view that holds an address and a shape and owns nothing, and a float32 buffer for activations. Shape is ggml's, so `dims[0]` is the fast axis and a weight printed as `[4096, 14336]` is 14336 rows of 4096.
- `molla.nn.kernel`, the host arithmetic a transformer block is made of: a matvec that reads packed weights without dequantizing them first, rms_norm, softmax, silu and gelu and swiglu, the residual add, and argmax. Every fused path is checked against dequantizing the same bytes and taking a plain dot product. It is scalar and it is not fast yet, which is what issue #120 is for. See [docs/validation/kernel.md](docs/validation/kernel.md).
- `molla.nn.rope`, rotary position embedding with the four scaling schemes that turn up in the wild: none, linear, NTK aware, and YaRN, plus the per pair frequency factors that Llama 3.1 ships precomputed in the file. Both pairings are supported, adjacent for converted Llama and split half for Qwen, which is a property of the file rather than of the model and is the kind of mistake that produces fluent output that wanders. The expected values come from `scripts/rope_ref.py`, which is ggml's rope loop transcribed to Python. See [docs/validation/rope.md](docs/validation/rope.md).
- `molla.nn.attention`, which scores one query against a run of keys and mixes the values. Grouped query, multi head and multi query are one loop rather than three paths. Sliding windows, sink tokens, logit softcapping and a scale that is not one over the root of the head dimension are all fields on the spec, because each of them is one architecture's idea and none of them deserves its own code path. It holds no cache, which is issue #27.
- `molla.nn.block`, one transformer layer with the knobs every architecture turns: gated or plain mlps, silu or gelu, per head query and key norms, norms on a sublayer's output as well as its input, and the attention spec from above. Nothing in it allocates, because a decode step runs it once per layer per token. It is checked against a second implementation written with naive loops over plain lists, since a layer that calls the right kernels in the wrong order produces numbers in the right range every time. See [docs/validation/blocks.md](docs/validation/blocks.md).
- `molla.nn.arch`, a row per architecture saying what that family always does, and `block_spec`, which turns a row plus a file's `Geometry` into a layer. Head counts, widths, the rope base and the epsilon come off the file and are never guessed. Llama, Qwen 2 and Qwen 3 are marked supported; Gemma and Phi 3 are described without being claimed, because nobody has run one yet. See [docs/adding-an-architecture.md](docs/adding-an-architecture.md).
- `molla.nn.model`, the part above a layer: an embedding lookup, the stack, a final norm and an output head that is tied to the embedding or not according to the file. Gemma's embedding scale and output logit cap live here. It produces logits and stops, because turning logits into a token is issue #28 and holding a cache is #27.

## [0.2.6] - 2026-09-02

molla can now say what a machine has and put a model's weights on it.

### Added

- `max-core`, pinned to 26.5.0, as a required dependency. The `mojo` package ships the standard library and nothing else, and every device call lives in `max-core`, so there is no route to a GPU without it. This was accepted in [docs/adr/0002-accept-max-core.md](docs/adr/0002-accept-max-core.md) but had never been added to the manifest. It is a runtime dependency and not only a build time one, and it is proprietary. See [docs/validation/toolchain.md](docs/validation/toolchain.md).
- `molla.sys.device`, which reports what a machine can put a tensor on: the api, the name, the index, and free and total memory, with a CPU entry that is always first and is a real placement rather than a fallback. The unified flag says whether a mapped file is already visible to the device or has to cross a bus, and it is read off the reported api rather than guessed from the build target.
- `build_targets_gpu` and `build_target_arch`, because `max-core` resolves the device architecture at compile time. A build made on a machine with no GPU has no device code in it and reports no accelerators even on a machine that has one.
- `molla.model.load`, which gets weights from a mapped file to wherever the kernels will read them. A pool of transfer threads walks the mapping while the thread that owns the device drains a queue of finished tensors and enqueues their copies, so the card starts moving the first tensor while the pool is still reading the second. Every tensor is placed once, before any byte moves, as host, unified or device. Device tensors get an aligned slot in one pool buffer rather than an allocation each, and when the model does not fit the planner gives the card everything read once per token first and leaves the embedding behind. An 8B q4_K_M loads in 3746 ms on the M4 and 499 ms on the 4090, with a read and a copy that overlap almost completely. See [docs/validation/load.md](docs/validation/load.md).
- `molla load <path> [workers]`, which runs a load and reports each stage as it happens, because a load that takes half a minute and says nothing reads as a hang.
- `molla devices`, which lists what this machine can put a tensor on.
- `page_size` and `will_need` in `molla.sys.mmap`, for readahead on a range of the mapping. A page is 16384 bytes on Apple silicon and 4096 on x86, and `madvise` rejects an address that is not aligned to it, so the number is asked for rather than assumed.

## [0.2.5] - 2026-09-02

The tokenizer corpus has nothing left in it that molla refuses.

60 of the 338 files carried a `Precompiled` normalizer, which is a SentencePiece charsmap, and molla refused every one of them rather than loading it with the normalizer missing and producing ids that are quietly wrong. That is 18 per cent of the popular tokenizers on the hub and it is all of T5, mT5, XLM-R, ALBERT, DeBERTa-v3 and NLLB. They load now and the full tier reports no refusals and no mismatches.

### Added

- The `Precompiled` normalizer, which is the SentencePiece charsmap. It was 60 of the 338 files in the tokenizer conformance corpus, 18 per cent of them, and all of the T5, mT5, XLM-R, ALBERT, DeBERTa-v3 and NLLB families. Those 60 load now and the full corpus reports no refusals and no mismatches. See [docs/validation/tokenizer.md](docs/validation/tokenizer.md).
- `molla.text.graphemes`, UAX #29 extended grapheme clusters, with all of the rules including the emoji rule GB11 and the Indic conjunct rule GB9c. The charsmap needs it because SentencePiece looks text up one cluster at a time. The grapheme break property is a fifth generated table in `src/molla/text/tables.mojo`.
- `strip_combining` in `molla.text.normalize`, which drops spacing and enclosing marks as well as non spacing ones.

### Fixed

- A `StripAccents` normalizer written on its own dropped only non spacing marks. Hugging Face drops all three mark categories there and drops only the non spacing ones for the `strip_accents` flag inside a `BertNormalizer`, and molla was using one function for both. It lost the enclosing keycap off a digit and the vowel sign off a Devanagari syllable, and it was two files in the corpus.

## [0.2.4] - 2026-09-01

The template engine now answers to an oracle on every commit.

494 real chat templates from the hub, 20 conversation shapes each, compared for exact string equality against the function `transformers.apply_chat_template` calls to build its string. 9500 renders compared and 9500 identical. It found that `tojson` was ignoring every argument except `indent`, which was putting a space after every comma in the tool definitions of four models that had asked for the compact spelling.

Not one template in 494 uses a construct the engine refuses to compile, so the refusal list this was meant to produce is empty. That is a fact about what model authors write rather than a claim about the engine, and the machinery that records refusals runs on every commit so the first one gets noticed.

### Added

- The chat template conformance corpus. 494 real chat templates from the hub, pinned by commit hash, rendered against 20 conversation shapes and compared for exact string equality against `transformers.utils.chat_template_utils.render_jinja_template`, which is the function `apply_chat_template` calls. 9500 renders compared and 9500 identical, of which 750 are cases both sides refuse. It runs on every commit as the `Template conformance` job and a mismatch blocks the merge. See [docs/validation/jinja.md](docs/validation/jinja.md).
- `scripts/templates.tsv`, `scripts/fetch-templates.py`, `scripts/check-template.py` and `scripts/template_oracle.mojo`, laid out the same way the tokenizer corpus is. The manifest carries the sha256 of the reference answer per repository, so the everyday check needs no Python at all, and the Python half runs when the reference version moves.
- `strftime_now` reads a clock that can be pinned, so a template that stamps today's date into the system prompt has one answer rather than one a day. `Template.render` and `Template.render_object` take the second to read, and zero is the real clock.
- `{% generation %}` and `{% endgeneration %}` render their body. They are a `transformers` extension that marks the span a model was supposed to have produced, so a training script can build a loss mask, and they contribute nothing to the string. Seven templates in the corpus use them.

### Fixed

- `tojson` ignored every argument except `indent`. Four templates in the corpus write `tojson(separators=(',', ':'))` to get the compact spelling and were getting the spaced one, which put a space after every comma in their tool definitions. The filter now takes the signature transformers gives it, which is `ensure_ascii`, `indent`, `separators` and `sort_keys` in that order, so the first positional argument is `ensure_ascii` and not `indent`.
- `{'a': 1}.items()` printed its pairs as lists rather than as tuples, so a template writing a pair straight out got square brackets where Python writes round ones.
- `2 ** 3 ** 2` was 64 rather than 512. Exponentiation is the one operator in the language that associates to the right.

## [0.2.3] - 2026-09-01

A chat template renders to the same bytes Python produces, and anything it cannot render refuses to load.

This is the Jinja2 subset. It is bounded on purpose: seven constructs are excluded and each of them raises when the template is compiled, which is when a model loads, rather than at render time on somebody's request. A template that would misrender does not get to serve traffic. Four execution limits are on by default, because a chat template is code out of a repository anybody can publish.

The checking is the part that matters. 38 real chat templates from the hub, nine conversation shapes each, compared for exact string equality against Python `jinja2` in the environment `transformers.apply_chat_template` builds. 342 renders, 342 identical. It found four defects in code that had already passed its unit tests, three of which produced plausible looking output rather than an error, and the worst of them made every binary operator past a certain depth in a template silently evaluate to its left operand.

A 20 turn conversation renders in 59.9 microseconds on the Llama 3.1 template, against the 200 the milestone asks for, with the cost of parsing the request JSON inside that number rather than beside it.

### Added

- `molla.jinja`, a bounded Jinja2 subset for chat templates. Ten modules: a lexer with `trim_blocks` and `lstrip_blocks` and the explicit whitespace markers, a parser onto a flat node list, and an evaluator with 70 filters, 30 tests and the five globals `range`, `dict`, `namespace`, `strftime_now` and `raise_exception`. Statements are `if`, `for` with the loop object and `break` and `continue`, `set` in both forms, `macro`, `call` and `filter` blocks. Checked against Python `jinja2` in the environment `transformers.apply_chat_template` builds, over 38 real chat templates and nine conversation shapes each: 342 renders, 342 identical. See [docs/validation/jinja.md](docs/validation/jinja.md).
- Seven constructs are excluded on purpose and each is a named error at compile time, which is model load time rather than request time: `include`, `extends`, `import`, `from`, `do`, `autoescape` and the async forms. There is no template loader, because a chat template is one self contained string. A refusal carries the construct, the line, the column and a snippet with a caret under it.
- Four execution limits, because a template is untrusted input from a repository anybody can publish: a step budget, an output cap, a recursion depth and a wall clock deadline. The clock is read every 1024 steps rather than every step, which kept it out of the profile.
- `Template` and `Cache` in `molla.jinja.template`. Compiling and rendering are separate because a template is compiled when a model loads and rendered on every request, and the cache keys compiled trees by the SHA-256 of the source rather than by the model, since most forks of a model ship the same template bytes. Values go in as JSON, which is the shape a request body and a tokenizer config already have.
- `molla template <template> <vars> [rounds]`, which renders a template and times it. A 20 turn conversation is 59.9 us on the Llama 3.1 template against the 200 us the milestone asks for, with Qwen3 at 79.8, Mistral Small at 85.4 and Granite at 43.4, all on the M4 and all including the cost of parsing the variables JSON on every round.

## [0.2.2] - 2026-09-01

Text goes in and ids come out, and there is an oracle saying they are the right ids.

This is the tokenizer, and underneath it the whole character layer Mojo does not ship: UTF-8 that rejects what it should reject, Unicode categories and combining classes and decompositions and case mappings generated from the current database, the four normal forms, and a backtracking regular expression engine. On top of that the five stage pipeline a `tokenizer.json` describes, with BPE, WordPiece, Unigram and word level models, twelve normalizers, ten pre-tokenizers, eight decoders and four post processors.

None of that is worth anything without an independent implementation to check it against, because a tokenizer that is wrong produces output that still looks sensible. So most of the work here is the checking. 4560 differential cases across four real models, eight million bytes through each of them, and then the conformance corpus: 338 real `tokenizer.json` files from the hub and 355 pieces of text, every id and every decode round trip compared against Hugging Face `tokenizers` 0.23.1. It runs on every commit and a mismatch blocks the merge.

The corpus earned its keep immediately. It found six defects in code that had already passed 1430 unit checks and four models, and the beginning of text property found a seventh. Two of them were GPT-2 refusing to load at all, one was a Chinese word being cut in half at the ideographic zero, and one was Whisper getting two beginning of text tokens where it should get one. All seven are fixed and all seven have a fixture in the suite now.

It is also fast, between five and eight times the reference on six of the eight throughput rows, and seven of the eight clear the 20 MB/s the milestone asked for. The eighth is gemma on unwrapped documentation, which is a property of that file rather than of the code, and the reference is slower on it too.

What is still refused is a `Precompiled` normalizer, the SentencePiece charsmap, which is 60 of the 338 corpus files and covers the T5, XLM-R and NLLB families. The corpus asserts a clean refusal rather than a quiet wrong answer, and the reference digests are already recorded for the day it lands.

### Added

- `molla.text`, everything a tokenizer needs to know about characters before it can start: UTF-8 encoding and decoding that rejects overlong forms and surrogates and reports an incomplete sequence as one, Unicode categories, combining classes, decompositions and full lowercase mappings generated from the database by `scripts/gen-unicode.py`, the four normal forms, and a backtracking regular expression engine with Unicode categories, lookahead and possessive quantifiers. Checked against Python over 294552 normalization cases and 6732 regular expression cases, all identical. See [docs/validation/text.md](docs/validation/text.md). The regex engine now works out which characters a match starting at each instruction could begin with, which is what makes the seven way alternation at the front of every GPT-2 style pattern cheap, and normalization skips runs of characters below the floor where every form is the identity.
- `scripts/check-text.py` and `scripts/text_oracle.mojo`, the differential run behind that, which also reports the pre-tokenizer split throughput against a file it generates. The GPT-2 pattern over 4 MB of mixed text is 174ms on the M4, which is 23 MB/s.
- `molla.tokenizer`, a `tokenizer.json` reader and the five stage pipeline behind it: added tokens, normalizer, pre-tokenizer, model and post processor, with BPE, WordPiece and Unigram, and the decoders read back the other way. Checked against Hugging Face `tokenizers` 0.23.1 over 4560 differential cases on four real models, one known mismatch caused by the reference reading a Unicode 9 category table, and eight million bytes through each of the four producing exactly the same token counts. See [docs/validation/tokenizer.md](docs/validation/tokenizer.md).
- `Tokenizer.encode_rendered`, for text a chat template produced. A template writes the beginning of text token into the text and the post processor then writes a second one, which is not what the model was trained on and does not fail loudly. `encode` still reproduces that, because its job is to match the reference. This is the entry point that does not.
- `DecodeStream`, which decodes one id at a time and never hands back half a character, so a token that carries one third of a character produces nothing until the rest of it arrives.
- The tokenizer conformance corpus: 338 real `tokenizer.json` files from the hub and 355 pieces of text, every id and every decode round trip checked against Hugging Face `tokenizers` 0.23.1. `scripts/tokenizers.tsv` is the manifest, `scripts/fetch-tokenizers.py` downloads the files and verifies their digests, `scripts/check-tokenizer.py` is the reference half and `scripts/tokenizer_oracle.mojo` is ours. A quick tier of 59 files runs in CI as the `Tokenizer conformance` job, which the required check waits for, so a mismatch blocks the merge. 272 files identical, 6 identical apart from a case where the reference reads a Unicode 9 table, 60 refused as the manifest says they should be, zero mismatched. See [docs/validation/tokenizer.md](docs/validation/tokenizer.md).
- The beginning of text reconciliation rule is now asserted as a property over the whole corpus, since the reference has no such rule to diff against. 210 of the 278 loadable files have an opening special and all 210 hold.

### Fixed

- The model loader read `vocab` at the point it reached it, which needs the type, because the type says whether a vocabulary is an object or an array. 13 corpus files write `unk_token` or `dropout` ahead of the type and one of them is GPT-2. The spans of `vocab` and `merges` are noted and skipped on the way past now, and read once the object has closed.
- A model object with no `type` member at all was refused. GPT-2 is one: its model object has seven members and the type is not among them. The kind is read off the other members now, an array vocabulary being Unigram, a `merges` list being BPE and a word length limit being WordPiece, which leaves a plain word level vocabulary as the fourth model kind. That kind is new and is now implemented.
- Two wrong character lists in the Nmt normalizer. It used U+2000 through U+200F where SentencePiece uses U+200B through U+200F, so four space characters were turned into ordinary spaces that should have been left alone, and it never mapped tab, newline, form feed or carriage return to a space at all.
- `\w` in the regular expression engine meant the letter categories, and the crate the model files were tested against reads it as Alphabetic. The two differ by the letter numbers, where the ideographic zero and the Roman numerals live, and by the circled and squared Latin letters, which the database files as symbols. A pre-tokenizer that calls the ideographic zero a symbol cuts a Chinese word in half.
- A byte level pre-tokenizer with `add_prefix_space` behind a Bert pre-tokenizer, given a string of nothing but spaces, invented a token containing a space, because the prefix step made a piece to put the space in front of when the earlier step had correctly thrown everything away.
- The same step, given several pieces, prefixed only the first. The reference runs the prepend over every one of them, which is how `a  b` keeps the space in front of `b` that the splitting threw away.
- The beginning of text reconciliation in `encode_rendered` only fired when the post processor template was one run of specials followed by the sequence. Whisper writes three separate specials in front of the text, so the id that doubles up is the third and the rule never looked at it. It finds the sequence and takes the special before it now, wherever that is.

## [0.2.1] - 2026-09-01

The model plane starts. Two readers, one model spec, and nothing that touches a weight yet.

molla now reads a GGUF file and a Hugging Face directory and answers the same questions about either: what architecture it is, what shape it is, which tokenizer it wants, what the tensors add up to, and what this build could actually do with it. `molla spec <path>` prints that for both, and dispatches on what is at the path rather than on the extension.

The point of doing both formats before doing anything with the weights is that everything above this has to be written once. The tokenizer in #21, the weight loading in #25 and the architecture blocks in #26 all read a `ModelSpec` and none of them will care which file it came out of. The one place the formats genuinely disagree is shape order, GGUF writing the fastest varying dimension first and safetensors writing row major, and that is written down where the code that has to reconcile it will find it.

Both readers check rather than trust, since both formats hand you byte offsets out of a downloaded file. The GGUF tensor directory is recomputed from the block geometry of every type and required to match every offset in the file. Every safetensors range is checked against the mapping and then against the dtype and the shape. Neither reader allocates per tensor and neither reads past the header, so a 7.3 GB two shard repository costs twenty milliseconds and twelve megabytes of resident set.

The check that mattered most was reading the same four models both ways. bge-small, SmolLM2-135M, gemma-3-270m and Qwen2.5-0.5B agree on the architecture, every geometry field, the tokenizer algorithm and every special token id, and the four places the numbers differ are each a real difference between the two files rather than a bug in one of the readers.

### Added

- `molla.model.spec`, the mapping from a GGUF file to a `ModelSpec`: architecture id, geometry, tokenizer shape, the block geometry of every ggml tensor type, and the capabilities the file declares intersected with what this build can do with them. Nothing in it reads a weight. Checked field by field against what llama.cpp loads from the same four models, and the tensor directory is recomputed from the block sizes and required to match every offset in the file. See [docs/validation/spec.md](docs/validation/spec.md).
- `molla spec <path>`, which prints all of that for a model file. Seven milliseconds and fourteen megabytes of resident set against a 468 MB model.
- `Gguf.flt`, `float_or`, `bool_or`, `has`, `array_count`, `tensor_index` and `tensor_prefixed`, so the layer above can ask about floats, flags and tensor names without decoding anything it did not ask for.
- `molla.model.safetensors`, the safetensors container: the header, the tensor directory, and sharded repositories resolved through `model.safetensors.index.json`. Every byte range is checked against the mapping and against the dtype and the shape before it is kept, and index against shard disagreement is counted in both directions.
- `molla.model.repo`, the Hugging Face directory around it, producing the same `ModelSpec` the GGUF path produces. `config.json` for the geometry, the transformers class name for the architecture, and a streaming pass over `tokenizer.json` for the counts and the special token ids, so a 33 MB tokenizer costs no heap. Checked against the same four models read both ways, and against a 7.3 GB two shard repository. See [docs/validation/safetensors.md](docs/validation/safetensors.md).
- `molla safetensors <path>`, which prints the header and tensor directory of a file or a directory. `molla spec <path>` now takes either format and dispatches on what is actually there rather than on the extension.
- `ModelSpec.source`, which says which reader produced the spec, and `TokenizerSpec.embedding_rows`, which says how many rows the embedding matrix has when the model states it. Qwen2.5 has 151665 token ids in a matrix of 151936 rows, and gemma 3 has one id past the last row, so the two numbers are not the same question.
- `TokenizerSpec.model_source`, the tokenizer name as the file spelled it. `model` is now normalised across the two formats, so GGUF's gpt2 and `tokenizer.json`'s BPE both report as bpe.

## [0.2.0] - 2026-09-01

M1 is done. Everything Mojo 1.0 does not ship and a server cannot do without: syscall wrappers, buffers and arenas, a reactor, HTTP/1.1, SSE and NDJSON, a JSON scanner, client TLS, threads and queues and a shutdown that finishes what it started, config and logging and metrics, and two commands that assert the properties the rest of it claims.

The milestone was to write eight layers a normal server project gets for free, and the point of ending it here is that there is now something to build on that has been measured rather than asserted. The server answers HTTP on Linux, macOS and Windows under WSL2, it holds a thousand connections through an hour without leaking, and it allocates nothing on the request path. Those are three claims and each of them has a command that fails if it stops being true.

This release adds the last of those, the hour long soak, and fixes the leak it found. The timing wheel cancelled lazily on a written assumption that nothing would ever create enough dead timers to matter, and every connection that closes cancels its idle timer, so the assumption was never true for any server anybody would run. It took an hour of real churn to become visible: on the laptop the run went from 127 MB to 2.3 GB. The sign off round was four machines, an hour each, a thousand connections, seven hundred million answers, no 5xx and no wrong statuses, and on every one of them the busiest timing wheel ended holding exactly as many entries as its reactor had connection slots.

There is still no model, no routing beyond the handful of built in routes, and no server side TLS. Those are M2 and later. What is here is the floor.

### Added

- `molla httpsoak`, an hour long soak on the systems layer. Five kinds of client at once against a real server: keep alive, streaming, slow readers that fill the write ring and hold it full, abrupt disconnects that never read the answer, and oversized bodies that get a 413 and a close. It watches resident memory, descriptors, the log ring and the connection table, the size of the busiest timing wheel, and latency drift across ten segments of the run. Runs nightly on Linux and macOS through `.github/workflows/soak.yml`, and a short version runs in the test suite. See [docs/validation/soak.md](docs/validation/soak.md).
- `molla.net.latency`, the segmented latency histogram both soaks now share, so the drift gate means the same thing in each.

### Fixed

- The timing wheel cancelled lazily, marking a timer dead and leaving it in its slot to be freed when that slot was next walked. Every connection that closes cancels its idle timer, and a slot is not walked until time gets close to the deadline it holds, so a busy server accumulated a dead slab entry per connection for the length of its idle timeout and never gave the memory back. The hour long soak grew from 127 MB to 2.3 GB on macOS and from 60 MB to 1.4 GB on Windows before this. Slot lists are doubly linked now and cancel unlinks and releases in constant time.
- `HttpProtocol._error` answered a 413, a 414 or a 431 without going through the handler path, which is where the status accounting lived, so those responses were never counted. A server that refused a hundred thousand oversized bodies reported `molla_http_responses_4xx_total 0`. Found by the soak on its first complete run.

## [0.1.7] - 2026-09-01

A per request allocation on a server whose whole pitch is not having one, found by the assertion added to stop exactly this.

The design has always said the request path allocates nothing in steady state. Nothing checked it, and it had already stopped being true: the reactor rebuilt a `Connection` when it reused a slot, so every accept freed and re-allocated the read buffer and the write ring, and the read buffer then grew again on the first request that did not fit. Three allocations per connection, which against a client that opens a connection per request is a per request allocation.

`molla allocs` is the check that will not let that happen again. It runs a mixed load until one pass of it costs nothing and then requires the next pass to cost nothing too, with no tolerance, because a tolerance is a budget and a budget gets spent. It runs in CI on every commit on all three platforms and a smaller version runs in the test suite.

### Added

- `molla allocs`, which runs a mixed load against a real server until a pass of it costs nothing and fails if the pass after that allocates anything. The load is a pipelined batch of a GET, a HEAD, a 404, a Content-Length body, a chunked body, eight pipelined GETs and both streaming routes, so an allocation on the fifth kind of request is not invisible. Runs in CI on every commit on all three platforms, and a smaller version runs in the test suite. See [docs/validation/allocations.md](docs/validation/allocations.md).

### Fixed

- The reactor rebuilt a `Connection` when it reused a slot, which freed and re-allocated the read buffer and the write ring on every accept and then grew the read buffer again on the first request. That is three allocations per connection, and against a client that opens a connection per request it is a per request allocation. `Connection.reuse` now takes over the new socket and keeps the buffers at whatever size the traffic already paid for.

## [0.1.6] - 2026-09-01

The operations surface. Config, structured logging, Prometheus metrics, and three `/molla` routes, none of which makes molla faster and all of which is what makes molla debuggable by somebody who is not holding the source open.

Config carries where each value came from rather than only the value. The precedence is flag over environment over file over default, and `molla config get` prints both halves, because the value on its own is not the question anybody has at three in the morning.

Logging is a byte ring per worker, written by that worker and read by the housekeeping thread, which makes it single producer single consumer and means nothing on the request path waits for anything. Records are built straight in the ring and become visible only when the length header is written last. The criterion was no allocations at a disabled level, and the cost turned out to be one atomic load and a comparison.

Metrics are one set of counters per worker on its own cache line, summed only at scrape time, so a request never touches a line another core owns. Statuses are bucketed by class rather than per code, and durations are integer nanoseconds beside a count, with the HELP text admitting there is no histogram rather than exporting a number that looks like a quantile.

The three admin routes share the main port. A second listener is the safer answer and is also a second thing to configure, expose in a container, and forget, and everything they return is already printed on startup.

### Added

- `molla.ops.config`, settings with a precedence of flag over environment over file over default, and every setting carrying where its value came from.
- `molla config get [key]`, which prints the effective value and the source, because the value on its own is not the question anybody has.
- `molla.ops.log`, structured logfmt logging on a byte ring per worker, written by the worker and flushed by the housekeeping thread. A disabled level costs one atomic load and no allocations at all.
- `molla.ops.metrics`, Prometheus counters with one set per worker on its own cache line, summed only when somebody scrapes them. Durations are exported as integer nanoseconds.
- `GET /molla/version`, `/molla/health` and `/molla/metrics`, served on the main port and off unless a caller turns them on.
- `sys.clock.monotonic_ns` and `sys.clock.realtime_ns`, so durations and timestamps stop sharing a clock.
- `tests/test_ops.mojo`, including the check that a thousand log calls at a disabled level allocate exactly nothing.
- `docs/validation/ops.md`.

### Changed

- `molla drain` now runs with logging, metrics and the admin routes on, asks for all three routes before the load starts, and prints the counters after the drain.

## [0.1.5] - 2026-09-01

The concurrency layer, and a shutdown that finishes what it started. Also the TLS policy work, which decides per host whether a certificate has to check out.

One rule runs through all of it: anything two threads touch lives at an address rather than in a value. A Mojo value moves, so two threads reaching a counter through two copies of it are incrementing two counters, and that looks entirely correct in a single threaded test. Every shared thing here is allocated once, kept at a fixed address, and handed to a thread as the one integer a thread entry point gets.

Three pieces are not built the way the design said they would be, and each of them is a Mojo 1.0 fact rather than a preference. `Once` cannot be `pthread_once`, whose callback takes no argument, because an initialiser would have nowhere to leave its result except a global and there are no globals. Signals cannot arrive on signalfd, which needs the signal blocked in every thread of the process, because the runtime starts threads before `main` and offers no way to reach their masks. And the signal has to be armed before the server starts, which the first version got wrong in a way that only shows up when something signals faster than a person can press Ctrl-C.

Draining means closing the listeners and then closing each connection at the first moment it owes the client nothing, cutting and counting what is left at the deadline. Writing the test for it found a real bug: the poller is edge triggered, so a request that arrived between two drain passes has already spent its edge, and a connection about to ask for something looked exactly like one that was asleep. The difference is a request the client sent and never got an answer to.

The acceptance test is a command rather than a unit test, because every time in a hundred runs is not something a unit test reaches. A hundred runs of thirty two connections are clean on all four machines in the fleet, and the only failing check anywhere is the WSL2 backpressure one that was already failing.

There is still no routing and no config file. Both are M1 and neither is here.

### Added

- `molla.sys.atomic`, with `AtomicRef` for one atomic integer reached by address and `AtomicBlock` for several of them each alone on a cache line. Everything two threads share in molla lives at an address rather than in a value, because a Mojo value moves and two threads reaching a counter through two copies are incrementing two counters, which looks correct in a single threaded test.
- `molla.sys.queue`, a bounded ticket based MPSC queue with padded cells and an SPSC ring with no compare and exchange in it at all. The cell padding matters more here than in the usual description of the algorithm, since molla's queues run close to empty and a queue holding one item has the producer's cell and the consumer's cell next to each other.
- `Once` in `molla.sys.thread`, which is not `pthread_once`. The callback `pthread_once` takes has no argument, so an initialiser has nowhere to leave what it made except a global, and Mojo 1.0 has no globals. This one takes the same function and integer a spawned thread takes, and tells the caller that ran the body apart from the callers that waited.
- `molla.net.context`, a `ServerContext` holding every setting a server has, made by the caller and passed down. It is what makes a server testable in process, and it is the reason there is no configuration global to remove later.
- `molla.net.supervisor`, signals delivered as a readable descriptor through a self pipe, and `serve_until_signal`. Not signalfd, which needs the signal blocked in every thread and Mojo starts threads before `main` with no way to reach their masks, and not EVFILT_SIGNAL, which works but is macOS only and would leave the shutdown path a different mechanism on each platform. SIGTERM and SIGINT drain, SIGQUIT dumps every worker's state, thread, connection count and queue depth first.
- `Server.drain` and the `DrainReport` it returns. A draining reactor closes its listeners and then closes each connection at the first moment that connection owes the client nothing, with a deadline after which what is left is cut and counted. A shutdown that reports it dropped four connections after nine seconds is one you can act on, and a process that just exits is not.
- `molla drain [connections] [deadline_ms]` and `scripts/drain-loop.sh`, which is issue #15's acceptance test. Each run loads every connection with a pipelined batch, sends the process SIGTERM, and fails unless every client received every answer whole. A hundred runs of 32 connections is clean in 3s on the M4.
- `ServerContext.send_buffer_bytes`, which sets a small kernel send buffer on accepted sockets. Zero in production, where Linux sizes this per connection and does it better than a number written down once. The drain test sets it low so most of a connection's answers are still molla's problem when the signal lands, and it goes on the accepted socket rather than the listener because a listener does pass the option down and macOS then autotunes the inherited buffer back up.
- `tests/test_concurrency.mojo`, 80 checks that all use real threads. Four threads and twenty thousand increments each against exactly eighty thousand, eight threads racing one `Once` for exactly one winner, three producers against a queue too small to hold one producer's share so every one of them meets a full queue, and a check that each of the 1500 values comes out exactly once.
- `docs/validation/threading.md`, with the sharing rule the layer is built on, the three places the obvious version does not work in Mojo 1.0, and the measurements behind the drain test.
- `molla.tls.policy`, which decides whether a certificate has to check out, by host name. There is no global insecure flag and there will not be one: a pull is not one connection, since ghcr.io answers a blob request with a redirect to a signed URL on a host it names, so a process wide switch would turn verification off for a host chosen by the response. `molla pull --insecure` and `molla tls --insecure` cover the host on the command line and nothing else, and every connection that skips verification says so on stdout.
- `probe` in `molla.tls.client` and a `tls` line in `molla version`, saying which backend loaded and how high it can negotiate, or why there is none. TLS is dlopened, so a machine without it runs everything except HTTPS, and that is now a line of output rather than a promise in a document. It also puts the macOS TLS 1.2 cap on the screen of the machine it applies to.
- `MOLLA_SECURITY` and `MOLLA_COREFOUNDATION` overrides for the two macOS framework paths, matching `MOLLA_LIBSSL` on the other platform. Nobody moves Security.framework, so these exist to point the loader at something that does not load, which is the only way to test what molla does on a host with no TLS library.
- `tests/test_tls.mojo`, 17 checks over the policy and the probe, including the missing library case on every machine that runs the suite. The one that matters is negative: naming a registry insecure must leave the CDN it redirects to verified.

### Fixed

- A drain no longer treats a connection with an unread request on it as idle. The poller is edge triggered, so a request that arrived between the last pass and this one has already spent its edge and there is no second one coming, and a connection about to ask for something looked exactly like one that was asleep. The difference is a request the client sent and never got an answer to. The drain now reads every live connection once more before deciding it is finished, which costs one recv per idle connection per drain pass and is only paid during a shutdown.
- `molla version` prints the version it was built at. It said 0.1.2 for the 0.1.3 and 0.1.4 releases, because the release process bumps `pixi.toml` and the changelog and nothing was checking that `VERSION` in `build_info.mojo` came with them. `scripts/check-version.sh` now fails CI when the three disagree, which is cheaper than noticing it in a bug report six months from now.

## [0.1.4] - 2026-09-01

JSON, in both directions, at a bit over 2 GB/s on the M4. Two modes over one SIMD scanner: a pull loop for request bodies, which is nearly all of the traffic, and a small tree for config and manifests. Object keys keep the order they arrived in, which matters because a tool call's arguments came from a model and the order is part of what it said.

Numbers are the half of a JSON library that is either right or nearly right, and nearly right shows up months later as one customer whose floats come back different. There is no `strtod` here, so no locale and no copy to get a NUL terminated string, and the conversion is exact for every input including the ones written specifically to break converters.

Running the suite on four machines was worth more than the parser was. It passed on the M4 and failed six checks on both EPYC boxes, because Mojo destroys a local at its last use and the reader holds its document as an address, so the buffer was freed while the parse was still reading it. The M4 passed because the freed block still held the bytes. That is a bug nothing about x86 caused and one machine would have shipped.

There is still no routing. That is M2.

### Added

- `molla.json`, a JSON parser and serializer with two modes over one SIMD scanner. `scan.mojo` classifies bytes a vector at a time and finds the quote, the backslash and any raw control byte with one mask, so validating a string costs nothing extra. `reader.mojo` is streaming mode, `dom.mojo` is DOM mode, `serialize.mojo` writes into a buffer with no intermediate allocation and keeps object keys in the order they arrived.
- Exact integers and correctly rounded doubles with no `strtod` and no locale, in `number.mojo` and `decimal.mojo`. Three paths: an integer that fits in 64 bits stays an integer, a double with 53 bits of digits and a small exponent goes through Clinger's fast path, and anything else goes through an exact decimal expansion. Printing goes back through the same struct and gives the shortest form that reads back as the same double, with the two cutoffs JavaScript uses.
- `molla jsonbench [kb] [n]`, the acceptance test for #13 as a command. On the M4 a 100 kB chat body parses at 2283 MB/s with zero allocations, against a gate of 1 GB/s. DOM mode is 1920 MB/s and five allocations for a 1234 node document.
- `tests/test_json.mojo`, 154 checks over the scanner, both number directions, the reader, the DOM and the writer, including the inputs that break converters and a round trip over four thousand doubles built from random bit patterns.
- `keep` in `molla.sys.mem`, which counts as a use of a value and does nothing else. Mojo destroys a local at its last use, so handing a buffer's address to a reader is the last thing the compiler sees using it and the buffer is freed while the parse is still reading it. It passed on the M4, where the freed block still held the bytes, and failed six checks on x86_64 Linux, where the allocator hands the block straight to someone else.
- `docs/validation/json.md`, with the numbers, the three places the design departs from what issue #13 describes, and the lifetime bug the fleet run found.

## [0.1.3] - 2026-09-01

The request path. A parser, bodies, multipart, a serializer, and the two streaming writers a completion needs, all on the reactor from 0.1.2.

The parser stops at the blank line and the body is read separately after it, which sounds like a detail and is the reason a body larger than the read buffer is no longer a contradiction. Content-Length, chunked and multipart all go through one call that says how much it took and whether it is done, and over a megabyte the body spills to a file, so an upload costs bounded memory rather than its own size.

The streaming writers are the half of the request path an inference server actually spends its time in. An event is flushed on its own as soon as it exists and events are only combined into one chunk when the socket cannot take them. Writing them found a real hole in 0.1.2: the reactor only called `on_writable` when the poller reported a socket writable, which is correct for a request and a response and leaves a stream stopped after one ring's worth of bytes, because a client that sent one request and is waiting produces no read edge and a drained ring produces no write edge. Nothing before this produced without being asked, so nothing had caught it.

There is still no routing and no JSON. Both are M1 work and neither is here.

### Added

- `molla.http` gets the parser the request path will actually run on. `scan.mojo` finds delimiters sixteen bytes at a time, `request.mojo` parses a request line and a header block into zero copy spans and stops at the blank line, `body.mojo` reads the body after that, `serialize.mojo` writes responses without allocating, `multipart.mojo` streams multipart/form-data, and `protocol.mojo` puts all of it on the reactor from #10.
- Bodies are read separately from headers, so a body larger than the read buffer is no longer a contradiction. Content-Length, chunked and multipart all go through one call that says how much it took and whether it is done. Over a megabyte the body spills to a file opened `O_EXCL` with mode 600, so peak memory for an upload is bounded by the threshold rather than by the upload.
- A hostile input corpus in `tests/test_http.mojo`, one case per refusal, asserting the status and not just the rejection. Bare LF, bare CR, Content-Length with Transfer-Encoding, duplicate framing headers, a Transfer-Encoding that is not chunked, whitespace before a colon, obsolete line folding, zero or two Host headers, control characters in a field, an Expect we cannot answer, and the 8 KiB, 64 KiB and 128 header bounds.
- An assertion that two thousand responses allocate nothing, measured after a warmup so the number does not hide the first few responses growing the buffer.
- `MODE_600` in `molla.sys.file`.
- `molla.http.stream`, the SSE and NDJSON writers, over chunked transfer encoding. An event is flushed on its own as soon as it exists and events are only combined into one chunk when the socket cannot take them, which falls out of holding framed payload and chunk framed bytes in separate buffers. Past the staging limit a producer gets `STREAM_FULL` rather than a bigger buffer. Event names, ids and NDJSON records are refused if they contain a line break, since either one silently desynchronises the client rather than failing.
- The SSE heartbeat, as `heartbeat_due` and `heartbeat` taking the current time rather than reading a clock, so it fits a protocol trait with no tick in it and so the test is deterministic. NDJSON deliberately has none, because it has no comment syntax and a blank line is not portably ignored.
- `Connection.produce`, one bit saying the protocol has more to write on its own initiative, and `/stream/sse` and `/stream/ndjson` on the M1 protocol.
- `tests/test_stream.mojo`, 59 checks over framing, validation, backpressure, the heartbeat, a reader taking one byte at a time against pinned 8 kB socket buffers and a 2 kB output ring, and a client that hangs up mid stream.

### Changed

- The reactor calls `on_writable` for a connection whose protocol says it is producing, not only for one the poller reported writable, and counts bytes leaving the output ring as progress for the service loop. Without both, a streaming response stops after one ring's worth: the client is not going to send anything else, so there is no read edge, and once the ring drains there is no write edge either. Nothing before #12 produced without being asked, so nothing had exercised it.
- `write_decimal` is a free function in `molla.http.serialize` rather than a method, since the streaming writers need the same non allocating decimal into their own buffers.
- `molla.http.server`, the M0 spike, answers 501 to a request with a body instead of half reading one. The parser it calls no longer consumes bodies, and the spike is kept as the evidence behind the M0 throughput numbers rather than as something to build on. Benchmark anything other than a bare GET against `molla.http.protocol`.

## [0.1.2] - 2026-09-01

The event loop the request path will run on. One reactor per worker thread, each owning its poller, its connection table and its timers, with nothing shared on the request path.

There is still no request. This is the layer HTTP gets written against, and the reason it is tagged on its own is that it is the last piece that can be validated purely against a kernel, before anything above it can hide a bug in it.

Three things came out of building it. Servicing a connection cannot be a single pass over a readiness event, which is the shape every event loop tutorial has, because a connection that reads more than its output ring holds ends up with no read edge and no write edge coming and a response half written. It never happens under light load and happens every time under the load that makes it worst. The backpressure test that catches it was only honest on one platform until the socket buffers were pinned at both ends, since Linux starts them larger and grows them, so the amount of data needed to provoke a short write is different on every machine. And a test that waits for another thread has to wait in milliseconds rather than in loop iterations, because the two convert at a rate that depends on how loaded the machine is, and the budget runs out soonest on the machine that needed it to last longest.

A thousand connections held for an hour on the M4 and on server1, no mismatched payloads across 657195369 and 59256042 round trips, no descriptor growth, and a p99 flat to the bucket across all ten segments of both runs.

### Added

- `molla.net` gets a real reactor. One event loop per worker thread, each owning its poller, its connection table and its timers, with no shared state on the request path. `reactor.mojo` is the loop and the four call protocol trait everything above it will be written against. `conn.mojo` is one connection and the four states a non blocking socket can be in. `listener.mojo` decides how connections are spread, which is SO_REUSEPORT sharding on Linux and a round robin handoff on macOS because macOS gives the last binder every connection instead of balancing. `server.mojo` is N reactors on N threads behind one address, TCP or unix.
- `molla.net.wheel`, a four level timing wheel at 100 ms resolution. An idle connection costs one entry in a slab and no timer descriptor, which is what makes a thousand idle keep alive connections free rather than a thousand syscalls per second.
- `molla netsoak`, the acceptance test for issue #10. A thousand connections with mixed idle and active traffic, latency compared across ten segments of the run so drift shows up as a number, and descriptors and peak memory checked at the end.
- `tests/test_reactor.mojo`, 61 checks on macOS and 62 on Linux over the wheel against an explicit clock, the reactor stepped by hand, backpressure, idle timeouts, unix sockets and the threaded server.
- `set_keepalive`, `listen_tcp_shared` and `set_buffer_size` in `molla.sys.socket`, and `monotonic_ms` in `molla.sys.clock`. `set_buffer_size` exists because an accepted socket inherits its send and receive buffers from the listener, which is the only way to make a backpressure test hit the same wall on both platforms.

### Fixed

- The threaded server tests waited for a worker thread by counting empty non blocking reads rather than by watching a clock, so on a loaded runner the whole budget could burn through before the thread that owed the answer was scheduled at all. Both waits now run against a monotonic deadline and sleep between attempts instead of spinning on the core the worker needs.

## [0.1.1] - 2026-08-31

Two layers of the standard library Mojo 1.0 does not have. The OS boundary, and the memory the request path will live in. Nothing above them exists yet, so nothing here changes what molla can do, and both are the kind of thing that is much cheaper to get right before there is a server on top of it.

The sys layer is every OS call molla makes, in one module, returning one result type that carries errno from the call site. Files, threads, mutexes, condition variables and signals, all tested against a real kernel on three machines and both architectures.

The io layer is buffers, rings and arenas, with the growth policy of each written down next to the code rather than left to be inferred, and an allocation counter underneath so the zero allocation claim in M1 can be a number instead of a promise.

### Added

- `molla.sys` grows the rest of the OS boundary. `result.mojo` holds the one type every wrapper returns, carrying a value and the errno captured at the call site. `file.mojo` covers open, read, write, seek, truncate, sync, stat, unlink, rename and directory listing. `thread.mojo` covers pthreads, mutexes and condition variables, which is what Mojo 1.0 has no threading module for. `signal.mojo` covers dispositions, masks and a self pipe that turns a signal into a readable descriptor.
- `tests/test_sys.mojo`, which runs every wrapper against the real OS. FFI mistakes show up as memory corruption somewhere else entirely, so they are caught at the boundary or not at all.
- `access` behind `exists`, `writev` behind `write_vectored`, `socketpair`, unix domain sockets and `shutdown` in `molla.sys.socket`.
- `docs/validation/sys.md`, which records what the boundary covers, what ran green on which machine, and the four platform traps that cost a session each.
- `molla.io`, the memory layer the request path is built on. `buffer.mojo` is an owned growable buffer with a written down growth policy, doubling to 64 kB and then a fixed step. `ring.mojo` is the per connection output ring, so a short write costs two integer updates instead of a memmove, and it hands `writev` its one or two pieces directly. `arena.mojo` is a bump allocator with a per request lifetime, freed in constant time. `bytes.mojo` compares, searches, trims and parses spans without allocating.
- `molla.sys.mem`, which is where every allocation in molla goes, and the allocation counter that makes "this request allocated nothing" a number rather than a claim. Issue #17 is what the counter is for.
- `tests/test_io.mojo`, 116 checks over the growth policy, the ring wrap, the arena and the byte helpers.

### Changed

- `molla.sys.mmap` opens and closes through `molla.sys.file` instead of declaring `openat` a second time. Two declarations of one C symbol with different argument counts in the same build fail to lower, and the error points at the standard library rather than at either file that caused it.

## [0.1.0] - 2026-08-31

M0 is done. The question it existed to answer was whether Mojo 1.0 can hold a socket, parse HTTP fast enough, map a model file, call a kernel on a GPU and reach a TLS library, and the answer is yes on all five, with numbers rather than opinions behind each one. Two decisions were taken at the gate and both are recorded with the measurements next to them.

D1 holds. The network edge stays in Mojo, and the Rust fallback stays documented and untaken. On the M4 a trivial handler runs between 43705 and 249896 requests per second depending on how loaded the machine was, against a gate of 5000. A thousand concurrent connections held for sixty seconds on kqueue and on epoll with flat memory. The TLS binding pulls the same blob from ghcr.io through three different libraries on four machines. One condition, the multi threaded one, cannot be tested because Mojo 1.0 has no threading module, so it moves to the M1 gate rather than being rounded up.

D6 does not. `max/kernels` needs proprietary `max-core` at runtime, its CPU kernels included, so the promise of an optional MAX runtime was describing a seam that does not exist. molla accepts the dependency, which means running molla means installing a proprietary runtime, and the README says so instead of claiming otherwise.

This release is still a foundation. It does not serve a model. M2 is the first one that does.

### Added

- `docs/adr/`, for decisions taken at a gate against measurements, with `0001-network-edge-stays-in-mojo.md` and `0002-accept-max-core.md`, the two M0 gate records

### Changed

- D1 in `docs/design.md` records the M0 gate outcome. The network edge stays in Mojo. The multi threaded half of the third reversal condition moves to the M1 gate, because Mojo 1.0 has no threading module to test it with.
- D6 is rewritten. `max-core` is a required dependency at runtime rather than an optional backend, because `max/kernels` does not run without it and its CPU kernels do not either. Running molla now means installing a proprietary runtime, and the README names both proprietary packages and what each is needed for.
- D7 is marked load bearing. `max/kernels` has no quantized matmul that will launch on an Apple GPU below an M5, so every Apple GPU kernel is one molla writes, and D7's per target numerics tests are what keep the portability claim honest.

## [0.0.3] - 2026-08-31

The M0 kernel spike ran, and it changed what the README is allowed to claim. The TLS spike ran after it, and molla can now pull a blob from ghcr.io over HTTPS on macOS and Linux.

### Added

- `molla.tls`, client TLS over OpenSSL 3.x and 1.1.1 on Linux and Secure Transport on macOS, both loaded with dlopen so a machine without a TLS library still runs molla and only loses HTTPS
- `molla.http.client`, a GET only HTTPS client with redirect following and chunked bodies, and `molla.registry.ghcr`, enough of the OCI distribution protocol to fetch a blob and check its digest
- `molla tls <host>` prints the backend, protocol, cipher and certificate chain, and `molla pull <ref>` pulls a blob from ghcr.io and verifies it
- `molla.sys.dns` for `getaddrinfo`, `molla.sys.sha256`, `molla.sys.cstr`, and `dial` in `molla.sys.socket` for a blocking socket with timeouts
- `MOLLA_LIBSSL` and `MOLLA_LIBCRYPTO` to point at a specific OpenSSL, which is also how the 1.1 fallback gets tested on a machine that has 3.x
- `docs/validation/tls.md` with the results from four machines and three TLS libraries
- `spikes/qmatmul/`, the kernel spike for issue #5, with its own pixi manifest so its proprietary dependency stays out of the root build
- `docs/validation/kernels.md` with the licence audit, the numbers from six machines, and the three options for what molla does next
- Numerics tolerances for Q4_K matmul on CPU and on GPU, which D7 asked for and never gave

### Changed

- The README no longer claims there is no proprietary dependency in the stack, because there is, and it names it
- D6 in `docs/design.md` is marked under review, since `max/kernels` does not build or run without proprietary `max-core`, its CPU kernels included
- D7 in `docs/design.md` is marked achievable but not inherited, since one source did compile to Metal and sm_89 with byte identical output, but `max/kernels` is not organised that way

### Known issues

- Carried over from 0.0.2 unchanged: the compiler is proprietary, so releases are source only. Build with `pixi run build`.
- `max/kernels` has no quantized matmul that will launch on an Apple GPU below an M5. On an M4 it raises at launch. The spike wrote its own kernel to get a Metal number at all.
- What molla actually does about the licence finding is not decided here. That is issue #7.
- TLS on macOS caps at 1.2. Secure Transport has no TLS 1.3, and the framework that does is built on Objective-C blocks, which Mojo cannot emit. Linux gets 1.3 through OpenSSL.
- The HTTPS client is IPv4 only, opens a connection per request, and reads bodies into memory whole. None of that is suitable for pulling a model and M3 replaces it.

## [0.0.2] - 2026-08-31

Four of the seven M0 spikes are done. molla can now map a model file and read what is in it, though it still cannot read a tensor.

### Added

- `molla.sys.mmap`, a read only whole file memory map
- `molla.model.gguf`, a GGUF v2 and v3 metadata reader that walks the header, the key value block and the tensor directory in place without copying the file, and `molla gguf <path>` to dump one
- `docs/validation/gguf.md` with the comparison against `gguf-dump` on four models covering bert, llama, gemma3 and qwen2, and what the zero copy read is actually worth

### Known issues

- Carried over from 0.0.1 unchanged: the compiler is proprietary, so releases are source only. Build with `pixi run build`.
- Nothing reads a tensor. The GGUF reader records where each one is and what type it is, and stops there.
- Metadata arrays are measured and skipped rather than decoded, so there is no way to read a tokenizer vocabulary yet.

## [0.0.1] - 2026-08-31

First tagged release. Three of the seven M0 spikes are done: the toolchain is pinned across the fleet, sockets and the event loop work on epoll and kqueue, and HTTP/1.1 parse and respond clears the throughput gate. Nothing serves a model yet.

### Added

- Pixi workspace with the Mojo toolchain pinned to 1.0.0, locked for macOS arm64, Linux x86_64, and Linux arm64
- A `molla` binary with `version` and `help`, reporting the toolchain and detected host
- A test runner, since Mojo 1.0 has no `mojo test`
- CI builds and tests on all three platforms for real, and smoke tests the binary
- `docs/validation/toolchain.md` recording the pin, the machines validated so far, and what Mojo 1.0 actually looks like against the release notes
- `molla.sys`, the libc boundary: errno, descriptors, IPv4 TCP sockets, and one `Poller` over kqueue and epoll with read and write interest
- `molla.net.echo`, a non blocking edge triggered TCP echo server, and `molla echo` to run it
- `molla soak`, which holds a thousand connections for sixty seconds and checks for descriptor and memory leaks
- `docs/validation/sockets.md` with the soak results on all three platforms and what the spike says about D1
- `molla.http`, a zero copy HTTP/1.1 request parser and a prebuilt response with an in place `Date` field, and `molla http` to run the throughput spike
- `docs/validation/http.md` with the M0 throughput measurements, the fleet results, and the two allocation and socket problems that cost more than the parser did
- Design document, roadmap, and milestone plan
- CI with docs linting, workflow linting, CodeQL on workflow definitions, OpenSSF Scorecard, and dependency review
- Release pipeline with SBOM, build provenance attestation, and keyless signing
- `scripts/check-action-pins.sh`, run in CI, which fails if an action is pinned to an annotated tag object rather than a commit

### Changed

- The toolchain version lives only in `pixi.toml` now, rather than also in a CI environment variable

### Fixed

- Six actions were pinned to annotated tag object SHAs instead of commit SHAs, which made the OpenSSF Scorecard workflow fail on publish with an imposter commit error even though the scan itself succeeded

### Known issues

- The Mojo compiler we build with comes from Modular's conda channel under a proprietary license, so the build is not yet Apache-2.0 end to end even though the source is. See `docs/design.md`.
- Releases are source only for the same reason. A Mojo 1.0 binary links `libKGENCompilerRTShared` and two other runtime libraries that ship only as shared objects under `LicenseRef-Modular-Proprietary`, and the linker bakes RUNPATH to the pixi directory that built it, so a bare binary does not start anywhere else. Publishing a working tarball would mean redistributing Modular's runtime inside molla's own artifacts. Build with `pixi run build` until that changes.

molla answers HTTP requests as of the M0 spike, but every path returns the same fixed body. The first milestone that serves a model is M2.
