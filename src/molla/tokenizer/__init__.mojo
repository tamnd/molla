"""The tokenizer: text to token ids and back.

A `tokenizer.json` is a pipeline written down as JSON, and this package is that
pipeline. Text goes through the added token split, a normalizer, a
pre-tokenizer, a model and a post-processor on the way in, and ids come back
through a decoder on the way out. Every stage is a tagged struct with a switch
rather than a trait object, which is what the rest of molla does with things
that are read from a file and dispatched on.

The parts:

- `vocab`, the token arena and the merge table
- `config`, one streaming pass over `tokenizer.json`
- `normalizer`, `pretok`, `model`, `post`, `decoder`, the five stages
- `added`, the trie of tokens that bypass the model
- `tokenizer`, the whole thing, and incremental decode
"""
