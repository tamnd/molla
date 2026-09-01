# The tokenizer, and the same ids as the reference

Issue #21. The done criterion was 20 MB/s single threaded on the M4 with the conformance corpus passing. Both are met on three of the four real fixtures and the fourth is explained below. What is worth more than either number is the thing underneath them: 4560 differential cases across four real `tokenizer.json` files, and eight million bytes of text through each one, producing the same ids as Hugging Face `tokenizers` 0.23.1 down to the last token.

## What it is

`molla.tokenizer` reads a `tokenizer.json` and runs the same five stages the reference runs, in the same order. Added tokens are found first in the text as it arrived, because they have to survive everything after them. What is left between them is normalized, searched for added tokens again in case one of them is only spelled correctly after normalization, cut into pieces by the pre-tokenizer, turned into ids by the model, and wrapped by the post processor. Decoding is that list read upside down, using the decoder the file names rather than the pre-tokenizer, because those two are allowed to disagree and a tokenizer whose decoder disagrees with its pre-tokenizer will encode correctly and print nonsense.

All three model kinds are here. BPE with a ranked merge heap, byte fallback, fused unknowns and the `ignore_merges` flag. WordPiece with the longest match walk and the continuing subword prefix. Unigram with a Viterbi pass over the lattice.

Around them: NFC, NFD, NFKC, NFKD, Lowercase, StripAccents, Nmt, Strip, Prepend, Replace with either a string or a regular expression, Precompiled, BertNormalizer and any Sequence of those. ByteLevel, Whitespace, WhitespaceSplit, BertPreTokenizer, Punctuation, Digits, CharDelimiterSplit, Metaspace and Split with all five split behaviours. ByteLevel, ByteFallback, Fuse, Metaspace, WordPiece, BPEDecoder, Replace and Strip decoders. TemplateProcessing, BertProcessing, RobertaProcessing and ByteLevel post processors, all four compiled into one template so there is one thing to get right rather than four.

A file naming something not on those lists is refused at load rather than loaded with the part it did not understand quietly missing. Skipping a stage changes the ids, and a tokenizer that is silently wrong is worse than one that will not open.

## The SentencePiece charsmap

`Precompiled` is the odd one on that list and it has its own section because it was the last thing missing and because two of its rules are surprising. It is issue #109, and it was 60 of the 338 files in the conformance corpus, which is 18 per cent of the popular tokenizers on the hub: T5, mT5, XLM-R, ALBERT, DeBERTa-v3 and NLLB, most of the multilingual encoders anybody runs.

What the file carries is a quarter of a megabyte of base64 that SentencePiece produced when the model was trained. Inside it is a little endian length, then a double array trie whose keys are UTF-8 byte sequences, then every replacement string laid end to end and separated by NUL bytes. A rule is a key in the trie whose value is an offset into that table. The rules are a compiled NFKC variant plus whatever else the training script asked for, so the result is close to NFKC and not equal to it, and approximating it with NFKC gives ids that are wrong in a way nothing reports.

The first surprise is that lookup happens per grapheme cluster rather than per character or over the whole string, and only when the cluster is under six bytes. That is what turns a letter and a combining acute into one precomposed letter, and it is why a rule keyed on a flag or a family emoji can never fire. Six is the reference implementation's number and there is no principle behind it. Getting there needed a UAX #29 grapheme cluster walker, which is now `molla.text.graphemes` and which implements all of the rules including the emoji rule GB11 and the Indic conjunct rule GB9c, even though the six byte cutoff means the charsmap can never reach either.

The second is that the trie search returns the shortest key that is a prefix of what it was given rather than the longest. SentencePiece itself takes the longest. The Rust port that Hugging Face `tokenizers` uses takes the first result the search produced, which is the shortest, and since the corpus is checked against that port, so does molla. It is visible from outside: a subscript i followed by a combining breve comes back as a plain i, because the rule for the subscript matched first and the breve went with it.

Both were settled before any Mojo was written, by reimplementing the algorithm in Python and running it against `tokenizers.normalizers.Precompiled` directly, over each of the three distinct charsmaps the corpus contains and 42288 inputs each. Zero mismatches on all three.

Implementing this also found a defect that had nothing to do with charsmaps. Hugging Face has two ways to spell stripping accents and they are not the same function: a `StripAccents` normalizer written on its own drops Mn, Mc and Me, while the `strip_accents` flag inside a `BertNormalizer` drops only Mn. molla used one function for both, which lost the enclosing keycap off a digit and the vowel sign off a Devanagari syllable. There are two functions now, `strip_marks` and `strip_combining`, and the difference is checked in `tests/test_text.mojo`.

## Against the reference

Two corpora, both generated by running `tokenizers` 0.23.1 over the four models in `~/models/st` and writing down what it said. Each case checks three things: the ids with special tokens, the ids without, and the text that comes back from decoding.

| Corpus | Cases | Mismatched |
| --- | --- | --- |
| Hand written, 116 cases across four models | 464 | 0 |
| Generated, 1024 cases across four models | 4096 | 1 |

The hand written set is the awkward inputs: empty strings, whitespace only, tabs, code, URLs, Windows paths, JSON, HTML, Greek, Cyrillic, Arabic, Hebrew, Chinese, Japanese, Korean, Thai, Devanagari, emoji, and a zero width joiner family sequence. The generated set is 500 byte windows of this repository's own markdown and Mojo source, random strings drawn from thirteen Unicode ranges, and random soup built from a word list chosen to hit the seams: `##`, `Ġ`, `▁`, apostrophes, single and double spaces, bare newlines.

The one mismatch is bge on a case containing U+061D, U+2E4F and U+1E95E. We treat them as punctuation and the reference does not, so the BertNormalizer splits in a different place. The reference reads its character properties from `unicode_categories`, a crate whose tables are frozen at roughly Unicode 9, and all three characters were assigned after that. Our tables are generated from the current database by `scripts/gen-unicode.py`. Following the reference here would mean shipping a deliberately stale copy of Unicode, so the divergence is written down rather than reproduced. Nothing any of the four models was trained on contains these characters.

## Eight million bytes

Two corpora again, both 8 MB, both encoded without special tokens, best of three runs after a warm up, on an Apple M4 running macOS 15.8. The first is this repository's markdown and Mojo source repeated to size, which is what a code assistant actually sees and is almost pure ASCII. The second is prose in ten scripts, thirty per cent of the bytes non ASCII, which is what everything else sees.

Every cell of the token count columns is identical between the two implementations, on both corpora, all four models. That is the result. The rates are the rest of it.

| Model | Tokens, code corpus | molla MB/s | reference MB/s | Tokens, prose corpus | molla MB/s | reference MB/s |
| --- | --- | --- | --- | --- | --- | --- |
| bge-small-en-v1.5 | 2048704 | 29.0 | 3.59 | 2701637 | 25.3 | 3.24 |
| Qwen2.5-0.5B-Instruct | 2060724 | 22.8 | 3.88 | 2275002 | 20.2 | 4.12 |
| SmolLM2-135M-Instruct | 2274920 | 20.5 | 3.92 | 3981287 | 23.5 | 3.89 |
| gemma-3-270m-it | 2287721 | 4.9 | 4.27 | 1601494 | 30.3 | 7.71 |

Seven of the eight rows clear 20 MB/s and every one of the eight is faster than the reference, between five and eight times on the six ordinary rows. The gemma code corpus row is the exception and it is the file, not the code.

gemma's normalizer replaces every space with U+2581 and its pre-tokenizer then splits on a space, which no longer exists anywhere in the text. So the pre-tokenizer never fires and the only thing that ever cuts the input is the added token matcher, because gemma ships every run of newlines from one to thirty one as an added token. The unit BPE is asked to merge is therefore a whole line. This repository writes documentation without wrapping paragraphs, so its longest line is 34606 characters, and the merge heap for a word that size does real work for every byte. The prose corpus has a newline every seventy characters and gemma is the fastest of the four on it. The reference behaves the same way for the same reason and is slower on both.

## What made it fast

The first version was correct and slow, and the four changes that fixed it were each found by profiling rather than guessed. Each row is the measurement that motivated the next one.

| Change | Effect |
| --- | --- |
| First sets per instruction, and skipping a split branch that cannot match | regex on the GPT-2 pattern, 14.9 to 22.6 MB/s |
| Hoisting the instruction arrays out of the interpreter loop as raw pointers | the same, 22.6 to 61.7 MB/s |
| Walking the added token trie over code points instead of a byte copy | added token split, 146 to 514 MB/s |
| A floor below which normalization is the identity, and run segmentation | Qwen NFC, 90 to 1024 MB/s |

The first two are in `molla.text.regex`. An alternation compiles to nested splits, so the seven apostrophe alternatives that open every GPT-2 style pattern used to be tried one at a time, each one pushing and popping the backtracking stack, before reaching the branch that could match an ordinary letter. Knowing which characters a match starting at each instruction could begin with turns all seven into seven array reads. The second change is the reminder that `Span` carries a bounds check: the same interpreter reading through `Span` runs at 29.9 MB/s and through `UnsafePointer` at 61.7.

The third is in `molla.tokenizer.added`. The trie is keyed on bytes and the input is counted in code points, and building a byte copy of the input plus the two maps between the two ways of counting cost more than every other part of that stage together. Encoding each code point as the trie walks it, and rejecting almost every position on its first byte, removes all of it.

The fourth is in `molla.text.normalize` and `molla.text.props`. Below U+00C0 no character decomposes, none has a nonzero combining class, none is a Hangul syllable and none appears as the second half of a canonical composition, so a run of characters under that is already in NFC and NFD both and can be copied across. The floor is computed from the tables rather than written down, so it stays right when the tables are regenerated, and the compatibility forms get their own floor of U+00A0. A run of real work starts one character early, because the first character of it may be a mark that composes with the letter in front of it.

The word cache in front of the model is what carries the rest. Real text repeats its pieces constantly, the table never evicts and never grows, and words longer than 256 bytes are not cached at all, which is the other half of why gemma on long lines is slow: nothing it produces is ever short enough to cache.

## Three decisions worth writing down

**Unassigned characters are not control characters.** BertNormalizer drops controls, and an unassigned code point has category Cn which sits with the other C categories. The reference reads its property from a table that only lists assigned characters, so a code point it has never heard of comes back as ordinary text and is kept. Treating Cn as a control instead drops those characters, and a word that lost one of them is a different word and takes a different id. This one difference accounted for most of the bge mismatches until it was found.

**Added tokens are matched twice.** Once in the text as it arrived and once after normalization, because a normalizer that lowercases will only produce the spelling of a lowercase added token on the second pass, and a matcher that ran only on the first pass would miss it.

**Double BOS is opt in to avoid, not on by default.** A chat template writes the beginning of text token into the text it renders, the added token matcher turns it back into an id, and then the post processor writes another one in front of it. Two of them is not what the model was trained on and it does not fail, it just makes the answers worse in a way nobody traces back to here. The reference does exactly this and can be watched doing it: on the byte fallback fixture in `tests/test_tokenizer.mojo`, `encode("<s>hello", add_special_tokens=True)` returns `[1, 1, 5, 114, 111, 118, 118, 7]`. `Tokenizer.encode` reproduces that, because its job is to be the reference. `Tokenizer.encode_rendered` is the one to call for text a template produced, and it returns `[1, 5, 114, 111, 118, 118, 7]`. Only the caller knows where the text came from, so it is a second entry point rather than a guess made inside the post processor.

## The conformance corpus

Issue #22. Four models are four models, and four is not enough to find the things that are wrong in one file out of fifty. So the corpus is 338 real `tokenizer.json` files and 355 pieces of text, and it runs on every commit.

`scripts/tokenizers.tsv` is the manifest. One row per file: the repository, the commit it was read at, the tier, the size, the sha256 of the file, whether molla is expected to load it, the sha256 of the answer, and the cases excluded for that file. The files themselves are not in the repository, because they come to 1284 MB. `scripts/fetch-tokenizers.py` downloads them and checks every digest, and `.gitignore` keeps `corpus/` out.

The 338 were picked from 2740 popular repositories on the hub. 1734 of them carry a `tokenizer.json` and 722 of those files are distinct, and the corpus is a greedy set cover over all 42 pipeline component types, one representative per coarse pipeline signature, and sixteen flagship models forced in by name. It is split into two tiers because 1284 MB is too much to fetch on every commit: a quick tier of 59 files and 172 MB that CI runs, and the full 338 for the ones that need running by hand.

`scripts/tokenizer_cases.txt` is the text, one case per line written as hex. Hex because the corpus is full of things an editor would quietly fix: lone control characters, a byte order mark, trailing spaces, and the empty string, which is case zero and therefore an empty line. The 355 cases are whitespace on its own in every combination, plain English, every C0 control, the invisible marks, combining sequences composed and decomposed, Hangul in both spellings, CJK in four scripts, ten right to left cases with the bidirectional overrides, emoji with skin tones and joiner families and flags, twenty other scripts, strings that look like special tokens but are not, normalization targets, realistic prompts, every Latin-1 byte as a character, and one eight character sample from each of 48 Unicode blocks.

The answer for one file is one line per case: the case number, the ids with special tokens, the ids without them, and the bytes the ids decode back to, in hex. The sha256 of the whole run of them is what the manifest records. A digest rather than the answers because the answers are forty megabytes, and because a digest that matches is the only thing anybody reads.

Both halves are in the repository. `scripts/check-tokenizer.py` is the reference: it runs Hugging Face `tokenizers` 0.23.1 and writes the digests. `scripts/tokenizer_oracle.mojo` is molla: it produces the same answer in the same spelling and compares. Python is a test time oracle and nothing else, and the everyday check does not run it at all, which is the point of storing digests rather than a script that regenerates them.

Where it stands:

| Outcome | Files |
| --- | --- |
| Identical ids and identical decode round trip | 332 |
| Identical apart from an excluded case | 6 |
| Refused at load, as the manifest says they should be | 0 |
| Mismatched | 0 |

The six with an excluded case are all the same thing in three places. Case 279 is U+32FF, the square era name Reiwa, whose NFKC decomposition was added in Unicode 12.1. Cases 319 and 320 contain U+0C04 and U+0D04, Telugu and Malayalam signs assigned in Unicode 11. The reference reads its character properties from a crate whose tables are frozen at roughly Unicode 9 and molla's are generated from the current database, so on those three characters the reference is reading an older Unicode than we are. Following it would mean shipping a deliberately stale copy of the database, so the case is excluded for those files with the reason written here, and it still runs against the other 332.

There are no refusals left. There were 60 of them until issue #109, all the same thing, all a `Precompiled` normalizer, and the corpus asserted that molla refused each one cleanly rather than loading it and producing ids that are quietly wrong. The answer digests were already in the manifest the whole time, recorded from the reference, so implementing the charsmap turned out to be one word per row in `scripts/tokenizers.tsv` and no regeneration of anything.

One thing in the corpus has no reference to compare against. `encode_rendered` drops the beginning of text token the post processor would otherwise write in front of one the chat template already wrote, and the reference has no such rule, so there is nothing to diff. It is checked as a property instead: the opening special is read off the file by encoding nothing and encoding one letter and taking the ids they agree on, then the same text is encoded both ways, and the rendered call has to come back with exactly one of the doubled ids gone and nothing else changed. When the id does not double there is nothing to drop and the two calls have to agree exactly, which is the half that catches dropping too eagerly. 243 of the 338 files have an opening special and all 243 hold.

## What the corpus found

Six defects, all of them in code that 1430 unit checks and 4560 differential cases across four models had passed.

**A model object whose members are in the wrong order.** The loader read `vocab` when it reached it, which needs the type, because the type says whether a vocabulary is an object or an array. 13 files write `unk_token` or `dropout` ahead of the type and one of them is GPT-2. The spans of `vocab` and `merges` are noted and skipped on the way past now, and read once the object has closed.

**A model object with no type at all.** GPT-2 again: its model object has seven members and `type` is not one of them. The reference answers this by trying each model in turn until one accepts the other members. What the members say is unambiguous, so it is read off them directly. An array vocabulary is Unigram, a `merges` list is BPE, a word length limit is WordPiece, and what is left is a plain word level vocabulary, which is a fourth model kind that molla did not have and now does.

**Two wrong character lists in the Nmt normalizer.** It used U+2000 through U+200F where SentencePiece uses U+200B through U+200F, so four space characters were being turned into ordinary spaces that should have been left alone, and it never mapped tab, newline, form feed or carriage return to a space at all.

**A gap in what `\w` means.** The regex crate the model files were tested against reads `\w` as Alphabetic, and Alphabetic is not the letter categories. It is wider by the letter numbers, which is where the ideographic zero and the Roman numerals live, and by the circled and squared Latin letters, which the database files as symbols. A pre-tokenizer that calls the ideographic zero a symbol cuts a Chinese word in half.

**A space invented out of nothing.** A byte level pre-tokenizer with `add_prefix_space` behind a Bert pre-tokenizer, given a string of nothing but spaces, produced one token containing a space, because the prefix step made a piece to put the space in front of when the earlier step had correctly thrown everything away.

**A prefix space on the first piece only.** The same step, given several pieces, prefixed the first. The reference runs the prepend over every one of them, which is how `a  b` keeps the space in front of `b` that the splitting threw away.

The seventh was found by the beginning of text property rather than by the reference. The rule only fired when the template was one run of specials followed by the sequence, and Whisper writes three separate specials in front of the text, so the id that would double up is the third and the rule never looked at it. It looks for the sequence now and takes the special before it, wherever that is.

## What is in CI

`tests/test_tokenizer.mojo` builds fourteen small `tokenizer.json` files in a temporary directory and checks ninety five things against them. Every expected id and every expected string in that file came out of `tokenizers` 0.23.1 running on the same bytes, because a tokenizer that is wrong produces output that still looks sensible and the only question worth asking is whether it matches.

Five of them cover the stages: a byte level BPE shaped like GPT-2, a WordPiece shaped like BERT with the template processor and both sequence types, a Unigram with a metaspace pre-tokenizer and scores chosen so the greedy answer and the Viterbi answer differ, a BPE with byte fallback and a fused decoder shaped like Llama and Gemma, and one file whose added tokens carry the `lstrip`, `rstrip` and `single_word` flags one each. Between them they exercise every stage the loader can build. The streaming decoder gets its own check: six ids that spell two three byte characters, fed one at a time, must say nothing four times and then produce the two characters.

The fourteenth is a `Precompiled` normalizer, on a charsmap with six rules in it built by hand, because the real ones are a quarter of a megabyte and there is no way to read one. Two of the six can never fire and that is the point of them: one is shadowed by a shorter rule and one is a seven byte key, so between them they pin the shortest match rule and the six byte cutoff, and a rewrite that fixed either would break sixty files in the corpus.

The other eight are the corpus findings brought back as small files, so that the next person to break one of them sees it in a second rather than in a corpus run. Four files with no `type` at all, one per model kind, the BPE one also writing its vocabulary ahead of everything else the way GPT-2 does. A word level vocabulary with its type, to check that it lands in the same place the untyped one does. An Nmt normalizer read one character at a time. A byte level prefix space behind a whitespace split. And a Whisper shaped template with three specials in front of the text. Every id in all of them came out of the reference reading the same bytes.

The conformance corpus runs in CI too, as the `Tokenizer conformance` job, on the quick tier. It fetches the 59 files, checks their digests, and runs `pixi run conformance-tokenizer`, and it is one of the jobs the required `CI OK` check waits for, so a mismatch blocks the merge. The corpus directory is cached on the manifest hash, so an ordinary commit restores it rather than downloading it.

The full tier is `pixi run conformance-tokenizer-full` after `python3 scripts/fetch-tokenizers.py --tier full --into corpus/tokenizers`. The fetch is 1284 MB and the run is about twenty seconds. Run it when the loader or any stage changes.

The four real models in `~/models/st` are not in CI because their tokenizer files run to 33 MB and they are what the throughput numbers above were measured on. That part is still by hand and this document is the record of it.

## Running it again

The corpus is in the repository and runs with two commands:

```console
$ python3 scripts/fetch-tokenizers.py --tier quick --into corpus/tokenizers
tier quick rows 59 have 0 fetched 59 failed 0 bytes 172.4 MB
$ pixi run conformance-tokenizer
checked 59 refused 0 failed 0 cases 355 bos 35
```

A mismatch prints the repository and the two digests. To find out which case, print both sides and diff them:

```console
$ pixi run mojo run -I src scripts/tokenizer_oracle.mojo scripts/tokenizer_cases.txt \
    scripts/tokenizers.tsv corpus/tokenizers full --print openai-community/gpt2 > mine.txt
$ python3 scripts/check-tokenizer.py --dir corpus/tokenizers --explain openai-community/gpt2 > theirs.txt
$ diff theirs.txt mine.txt
```

The first field of a differing line is the case number, and `python3 -c "print(bytes.fromhex(open('scripts/tokenizer_cases.txt').read().split(chr(10))[N]))"` says what the text was.

Regenerating the manifest digests needs a Python environment with `tokenizers` in it and the full tier on disk. `python3 scripts/check-tokenizer.py --dir corpus/tokenizers --refresh` rewrites the answer column and leaves the `status` and `skip` columns alone, because those are statements about molla rather than about the reference and nothing on the Python side can work them out.

The throughput numbers are a separate thing and their benchmark programs are not in the repository, since they need `~/models/st`.
