# Unicode and regular expressions under the tokenizer

The first half of issue #21. A `tokenizer.json` is a text processing pipeline written down as JSON, and before any of it can run there has to be something underneath that knows what a character is. A normalizer named NFC has to be NFC. A pre-tokenizer that splits on a pattern has to split the way the pattern says. Neither of these fails loudly when it is wrong, so both are checked here against an outside answer rather than against our own expectations.

This records what `molla.text` is, the two places where it deliberately does not do what Python does, how the whole thing was checked, and the one performance mistake that cost more than everything else put together.

## What was built

`scripts/gen-unicode.py` reads the tables out of the Python that is running it and writes `src/molla/text/tables.mojo`. The current file is Unicode 16.0.0 and holds 3368 category ranges, 393 combining class ranges, 5913 decompositions and 1460 full lowercase mappings, in 56 KB of source. Everything is a base 64 varint, five payload bits to a digit, and range tables store a delta from the end of the range before rather than an absolute start, which is what keeps a table of 3368 ranges down to a few kilobytes. Hangul is not in the decomposition table at all because 11172 syllables decompose by arithmetic.

There is no way to avoid generating this. The alternative is a dependency on ICU, which is 30 MB of C++ for the four functions a tokenizer needs, and the four functions are a category lookup, a combining class lookup, a decomposition and a lowercase mapping.

`src/molla/text/utf8.mojo` is the encoder and decoder. It rejects overlong forms, surrogates and anything past U+10FFFF, and a truncated sequence at the end of a buffer comes back as incomplete with the number of bytes still wanted rather than as an error, which is what an incremental decoder needs when a token arrives split across two chunks.

`src/molla/text/props.mojo` decodes the tables once into an object the caller owns, because Mojo 1.0 has no module level mutable state. The composition map is not generated. It is built by running the canonical decompositions backwards and dropping the pairs the exclusion bit says do not compose, which is 5913 entries in and 1360 out.

`src/molla/text/normalize.mojo` is the four normal forms as two operations and two flags. Full decomposition, then a stable sort by combining class, then optionally the composition pass.

`src/molla/text/regex.mojo` is a backtracking engine, about 1200 lines. It has classes, alternation, the three quantifiers with lazy and possessive forms, counted repeats, anchors, word boundaries, Unicode categories and lookahead. It does not have captures, back references or lookbehind, because no tokenizer pattern has ever used one and a capture buffer costs something on every match. Matching is leftmost first, which is what Perl does and what the `fancy-regex` crate behind Hugging Face does once a pattern has a lookahead in it.

## Two rules that are choices rather than bugs

An empty match advances the cursor by one character. Python retries at the same position looking for a non empty match instead, so `a*` over `xx` gives three empty matches here and two empty matches and nothing else there. The Rust `regex` crate does what this does, and the tokenizer files were written against the Rust crate, so this is the answer that produces the same tokens as the model was trained with.

`^` is the start of the text and `$` is the end of it, not the start and end of a line, and `$` does not match just before a trailing newline. Python reads both the other way round. `(?m)` asks for the line reading. Again this is the Rust rule.

Both are pinned by name in `tests/test_text.mojo` so that a future change to either has to be deliberate.

## How it was checked

`scripts/check-text.py` is the differential run. It builds `scripts/text_oracle.mojo`, writes the cases into a temporary directory, runs both sides and compares them line for line. It takes about two minutes, almost all of it Python computing the expected answers, which is why it is a script somebody runs and not part of `pixi run test`.

| What | Cases | Result |
| --- | --- | --- |
| All four normal forms, every assigned code point on its own, plus marks piled onto bases, plus Hangul from both ends | 294552 | identical to `unicodedata` |
| 51 patterns over 132 texts, as the offsets of every match | 6732 | identical to the `regex` module |

The 51 patterns are the three real pre-tokenizer patterns out of a GPT-2, a Qwen and a Llama 3 `tokenizer.json`, plus 48 written one per feature, because a differential run only tells you about the constructs you thought to write down. The 132 texts are 52 written by hand and 80 generated from a seeded alphabet that mixes two scripts, a combining mark, both kinds of space and the quote that starts a contraction.

The normalization cases are the ones that separate a correct implementation from a plausible one. Single code points catch a wrong table. Two marks of different classes on one base catch a reordering that is not stable. A pair that decomposes and then refuses to compose catches a composer that was built by running the decomposition table backwards without reading the exclusion list, which turns U+0915 U+093C into U+0958 and silently changes what the model reads.

`tests/test_text.mojo` is the part that runs on every commit. It is not a smaller version of the differential run, it is the cases a differential run would find too late: overlong encodings, surrogates, the incomplete sequence, the composition exclusion, equal combining classes keeping their order, and every regular expression construct with the answer written next to it.

## Throughput

Issue #21 wants encode at 20 MB/s single threaded on the M4. The pre-tokenizer split is the part of encode that this layer decides, so that is what is measured, with the file already decoded to code points.

| GPT-2 pre-tokenizer pattern, 4 MB of mixed text, M4 | Time | Rate |
| --- | --- | --- |
| Class lookup through a `List[CharClass]` | 605ms | 6.6 MB/s |
| The same, plus an ASCII table inside `CharClass` and a reusable scratch space | 942ms | 4.2 MB/s |
| Class ranges and the ASCII table flattened onto the compiled pattern | 174ms | 23 MB/s |

The middle row is the interesting one. Two optimizations that should both have helped made it 55% slower, and the reason is the first row rather than either of them. `self.classes[index].contains(cp)` copies the class out of the list, and a class is three `List` fields, so every character of input was three heap allocations and three frees. Adding a 128 byte ASCII table to the struct made the copy bigger, which is why the improvement made it worse. Flattening the ranges into three lists on the compiled pattern, indexed by class number, took the inner loop from three allocations per character to one load.

`scripts/check-text.py` prints this measurement at the end of its run against a file it generates itself, so the number can be reproduced from a clean checkout.

## What was wrong before it was right

The class copy above, which is worth stating plainly: the engine was more than three times slower than it needed to be, and no profiler was involved in finding it, only the observation that an optimization had made it slower.

`^` and `$` meant the start and end of a line. This is what Perl means by them and what a person writing an engine from memory writes down, and every one of the tokenizer patterns still split correctly with it, because none of them uses either anchor. The unit tests found it, the differential run did not, because the corpus at that point had no pattern that anchored across a newline.

A quantifier with nothing to repeat, and a quantifier stacked on a quantifier, were both accepted. `*a` matched a literal asterisk and `a**` compiled to a repeat of a repeat, which is one of the standard ways to write an exponential pattern by accident. Both are errors now, at compile time, because a pattern arrives in a file somebody downloaded.

The `(a+)+b` case is the one that stays. There is a step limit and it raises, rather than an engine that cannot backtrack exponentially, because the second thing is a different engine and this one has to run a lookahead. Thirty characters of `a` against that pattern raises after four million steps, which is a few milliseconds, and a real pattern never comes close.
