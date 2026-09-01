"""Tests for the tokenizer.

The fixtures here are small files built in code, and every id and every string
they are checked against came out of the Hugging Face `tokenizers` crate
running on the same file. That is the only way this can be tested. A tokenizer
that is wrong does not crash, it hands the model a sequence it was not trained
on and the answers get worse, so the question is never whether the output looks
sensible, it is whether it is the same as the reference to the id.

The four model kinds each get a file: byte level BPE the way GPT-2 is shaped,
WordPiece the way BERT is shaped, Unigram with a metaspace pre-tokenizer, and
BPE with byte fallback and a fused decoder the way Llama and Gemma are shaped.
Between them they cover every normalizer, pre-tokenizer, post-processor and
decoder that the loader can build. A fifth file exercises the three flags an
added token can carry.

Running the same thing over the real files, the four models in `~/models/st`
and eight million bytes of text, is in `docs/validation/tokenizer.md`, because
those files are gigabytes and CI cannot download them on every push.
"""

from std.ffi import external_call

from harness import Suite

from molla.sys.file import MODE_755, mkdir, rmdir, unlink
from molla.sys.mem import AllocCounter
from molla.text.utf8 import encode
from molla.tokenizer.bytelevel import byte_to_point
from molla.tokenizer.tokenizer import DecodeStream, Session, Tokenizer


def _pid() -> Int:
    return Int(external_call["getpid", Int32]())


def _temp_dir() raises -> String:
    var path = String("/tmp/molla_tok_") + String(_pid())
    _ = mkdir(path, MODE_755)
    return path^


def _write(path: String, text: String) raises:
    with open(path, "w") as f:
        f.write(text)


def _bool(value: Bool) -> String:
    return String("true") if value else String("false")


def _added(
    id: Int,
    content: StringSpan,
    single_word: Bool,
    lstrip: Bool,
    rstrip: Bool,
) -> String:
    """One entry of the `added_tokens` array.

    All five flags are written out because the reference implementation refuses
    a file that leaves any of them out, and a fixture it refuses is a fixture
    that cannot be checked against it.
    """
    return (
        '{"id":'
        + String(id)
        + ',"content":"'
        + String(content)
        + '","single_word":'
        + _bool(single_word)
        + ',"lstrip":'
        + _bool(lstrip)
        + ',"rstrip":'
        + _bool(rstrip)
        + ',"normalized":false,"special":true}'
    )


def _vocab(tokens: List[String]) -> String:
    """A word vocabulary, each token taking the next id."""
    var out = String("{")
    for i in range(len(tokens)):
        if i > 0:
            out += ","
        out += '"' + tokens[i] + '":' + String(i)
    return out + "}"


def _byte_level_vocab() -> String:
    """The 256 byte level characters, each mapped to the byte it stands for.

    Every byte level vocabulary opens with these, and writing them out by hand
    would be four hundred lines that say nothing. Two of them are the quote and
    the backslash, which are escaped so the file is still JSON afterwards.
    """
    var points = byte_to_point()
    var out = String("")
    var one = List[UInt8]()
    for b in range(256):
        if b > 0:
            out += ","
        one.clear()
        encode(points[b], one)
        out += '"'
        if len(one) == 1 and (one[0] == 34 or one[0] == 92):
            out += "\\"
        out += String(StringSpan(unsafe_from_utf8=Span(one)))
        out += '":' + String(b)
    return out^


def _bpe_json() -> String:
    """Byte level BPE, shaped like GPT-2.

    The merges are written as pairs of strings, which is one of the two
    spellings in the wild. Twelve of them is enough to show that ranking
    decides the answer: `Ġthe` is in the vocabulary but the word comes out as
    `Ġt` and `he`, because the merge that builds `he` has a lower rank than the
    one that starts building `Ġthe` and BPE always takes the lowest rank first.
    """
    var merged = (
        '"he":256,"hel":257,"hell":258,"hello":259,'
        + '"Ġw":260,"Ġwo":261,"Ġwor":262,"Ġworl":263,"Ġworld":264,'
        + '"Ġt":265,"Ġth":266,"Ġthe":267'
    )
    var merges = (
        '[["h","e"],["he","l"],["hel","l"],["hell","o"],'
        + '["Ġ","w"],["Ġw","o"],["Ġwo","r"],["Ġwor","l"],["Ġworl","d"],'
        + '["Ġ","t"],["Ġt","h"],["Ġth","e"]]'
    )
    return (
        '{"version":"1.0","truncation":null,"padding":null,'
        + '"added_tokens":['
        + _added(268, "<|endoftext|>", False, False, False)
        + "],"
        + '"normalizer":null,'
        + '"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,'
        + '"trim_offsets":true,"use_regex":true},'
        + '"post_processor":{"type":"ByteLevel","add_prefix_space":true,'
        + '"trim_offsets":true,"use_regex":true},'
        + '"decoder":{"type":"ByteLevel","add_prefix_space":true,'
        + '"trim_offsets":true,"use_regex":true},'
        + '"model":{"type":"BPE","dropout":null,"unk_token":null,'
        + '"continuing_subword_prefix":null,"end_of_word_suffix":null,'
        + '"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,'
        + '"vocab":{'
        + _byte_level_vocab()
        + ","
        + merged
        + "},"
        + '"merges":'
        + merges
        + "}}"
    )


def _wordpiece_json() -> String:
    """WordPiece, shaped like BERT, with the template processor and the two
    sequence types that go with it."""
    var tokens = List[String]()
    tokens.append(String("[PAD]"))
    tokens.append(String("[UNK]"))
    tokens.append(String("[CLS]"))
    tokens.append(String("[SEP]"))
    tokens.append(String("[MASK]"))
    tokens.append(String("hello"))
    tokens.append(String("world"))
    tokens.append(String("un"))
    tokens.append(String("##aff"))
    tokens.append(String("##able"))
    tokens.append(String("the"))
    tokens.append(String("cafe"))
    tokens.append(String("!"))
    tokens.append(String(","))
    tokens.append(String("run"))
    tokens.append(String("##ning"))

    var added = (
        _added(0, "[PAD]", False, False, False)
        + ","
        + _added(1, "[UNK]", False, False, False)
        + ","
        + _added(2, "[CLS]", False, False, False)
        + ","
        + _added(3, "[SEP]", False, False, False)
        + ","
        + _added(4, "[MASK]", False, False, False)
    )
    var single = (
        '[{"SpecialToken":{"id":"[CLS]","type_id":0}},'
        + '{"Sequence":{"id":"A","type_id":0}},'
        + '{"SpecialToken":{"id":"[SEP]","type_id":0}}]'
    )
    var pair = (
        '[{"SpecialToken":{"id":"[CLS]","type_id":0}},'
        + '{"Sequence":{"id":"A","type_id":0}},'
        + '{"SpecialToken":{"id":"[SEP]","type_id":0}},'
        + '{"Sequence":{"id":"B","type_id":1}},'
        + '{"SpecialToken":{"id":"[SEP]","type_id":1}}]'
    )
    var specials = (
        '{"[CLS]":{"id":"[CLS]","ids":[2],"tokens":["[CLS]"]},'
        + '"[SEP]":{"id":"[SEP]","ids":[3],"tokens":["[SEP]"]}}'
    )
    return (
        '{"version":"1.0","truncation":null,"padding":null,'
        + '"added_tokens":['
        + added
        + "],"
        + '"normalizer":{"type":"BertNormalizer","clean_text":true,'
        + '"handle_chinese_chars":true,"strip_accents":null,"lowercase":true},'
        + '"pre_tokenizer":{"type":"BertPreTokenizer"},'
        + '"post_processor":{"type":"TemplateProcessing","single":'
        + single
        + ',"pair":'
        + pair
        + ',"special_tokens":'
        + specials
        + "},"
        + '"decoder":{"type":"WordPiece","prefix":"##","cleanup":true},'
        + '"model":{"type":"WordPiece","unk_token":"[UNK]",'
        + '"continuing_subword_prefix":"##","max_input_chars_per_word":100,'
        + '"vocab":'
        + _vocab(tokens)
        + "}}"
    )


def _unigram_json() -> String:
    """Unigram with a metaspace pre-tokenizer and a sequence normalizer.

    The scores are what the Viterbi pass reads, and they are chosen so that the
    answer is not the obvious one: `hello` on its own is spelled `hell` and `o`
    by the pieces, but `▁hello` after a space beats both.
    """
    var pieces = (
        '[["<unk>",0.0],["▁",-2.0],["▁hello",-1.0],["▁world",-1.5],'
        + '["hell",-3.0],["o",-4.0],["▁the",-1.2],["h",-5.0],["e",-5.0],'
        + '["l",-5.0],["w",-5.0],["r",-5.0],["d",-5.0],["▁w",-3.5]]'
    )
    var metaspace = (
        '{"type":"Metaspace","replacement":"▁",'
        + '"prepend_scheme":"always","split":true}'
    )
    return (
        '{"version":"1.0","truncation":null,"padding":null,'
        + '"added_tokens":['
        + _added(0, "<unk>", False, False, False)
        + "],"
        + '"normalizer":{"type":"Sequence","normalizers":['
        + '{"type":"Nmt"},{"type":"NFKC"},'
        + '{"type":"Replace","pattern":{"Regex":" {2,}"},"content":" "}]},'
        + '"pre_tokenizer":'
        + metaspace
        + ","
        + '"post_processor":null,'
        + '"decoder":'
        + metaspace
        + ","
        + '"model":{"type":"Unigram","unk_id":0,"vocab":'
        + pieces
        + ',"byte_fallback":false}}'
    )


def _fallback_json() -> String:
    """BPE with byte fallback, shaped like Llama and Gemma.

    Everything the vocabulary does not have becomes a run of `<0xNN>` tokens
    rather than one unknown, and the decoder turns those back into bytes and
    fuses them, which is the only way a model with a few thousand pieces can
    reproduce a character it has never seen. The merges are written in the
    other spelling, one string with a space in it, because both are in the wild
    and the loader has to take either.
    """
    var words = List[String]()
    words.append(String("<unk>"))
    words.append(String("<s>"))
    words.append(String("</s>"))
    words.append(String("▁hello"))
    words.append(String("▁world"))
    words.append(String("▁"))
    words.append(String("hell"))
    words.append(String("o"))
    words.append(String("w"))
    words.append(String("hello"))

    for b in range(256):
        words.append(String("<0x") + _hex(b) + ">")

    var added = (
        _added(0, "<unk>", False, False, False)
        + ","
        + _added(1, "<s>", False, False, False)
        + ","
        + _added(2, "</s>", False, False, False)
    )
    var single = (
        '[{"SpecialToken":{"id":"<s>","type_id":0}},'
        + '{"Sequence":{"id":"A","type_id":0}}]'
    )
    var pair = (
        '[{"SpecialToken":{"id":"<s>","type_id":0}},'
        + '{"Sequence":{"id":"A","type_id":0}},'
        + '{"Sequence":{"id":"B","type_id":1}}]'
    )
    return (
        '{"version":"1.0","truncation":null,"padding":null,'
        + '"added_tokens":['
        + added
        + "],"
        + '"normalizer":{"type":"Sequence","normalizers":['
        + '{"type":"Prepend","prepend":"▁"},'
        + '{"type":"Replace","pattern":{"String":" "},"content":"▁"}]},'
        + '"pre_tokenizer":null,'
        + '"post_processor":{"type":"TemplateProcessing","single":'
        + single
        + ',"pair":'
        + pair
        + ',"special_tokens":'
        + '{"<s>":{"id":"<s>","ids":[1],"tokens":["<s>"]}}},'
        + '"decoder":{"type":"Sequence","decoders":['
        + '{"type":"Replace","pattern":{"String":"▁"},"content":" "},'
        + '{"type":"ByteFallback"},{"type":"Fuse"},'
        + '{"type":"Strip","content":" ","start":1,"stop":0}]},'
        + '"model":{"type":"BPE","dropout":null,"unk_token":"<unk>",'
        + '"continuing_subword_prefix":null,"end_of_word_suffix":null,'
        + '"fuse_unk":true,"byte_fallback":true,"ignore_merges":false,'
        + '"vocab":'
        + _vocab(words)
        + ',"merges":["hell o"]}}'
    )


def _flags_json() -> String:
    """Three added tokens, one per flag, over a vocabulary of single letters."""
    var words = List[String]()
    words.append(String("[UNK]"))
    words.append(String("a"))
    words.append(String("b"))
    words.append(String("ab"))
    words.append(String("x"))
    words.append(String("y"))
    words.append(String("xy"))
    var added = (
        _added(7, "<L>", False, True, False)
        + ","
        + _added(8, "<R>", False, False, True)
        + ","
        + _added(9, "<W>", True, False, False)
    )
    return (
        '{"version":"1.0","truncation":null,"padding":null,'
        + '"added_tokens":['
        + added
        + "],"
        + '"normalizer":null,'
        + '"pre_tokenizer":{"type":"Whitespace"},'
        + '"post_processor":null,"decoder":null,'
        + '"model":{"type":"WordPiece","unk_token":"[UNK]",'
        + '"continuing_subword_prefix":"##","max_input_chars_per_word":100,'
        + '"vocab":'
        + _vocab(words)
        + "}}"
    )


def _hex(value: Int) -> String:
    """Two upper case digits, the way a byte fallback token spells a byte."""
    var digits = StaticString("0123456789ABCDEF")
    var out = List[UInt8]()
    out.append(digits.unsafe_ptr().unsafe_load(value >> 4))
    out.append(digits.unsafe_ptr().unsafe_load(value & 15))
    return String(StringSpan(unsafe_from_utf8=Span(out)))


def _ids(
    tokenizer: Tokenizer,
    mut session: Session,
    text: StringSpan,
    add_special: Bool,
) raises -> List[Int]:
    var out = List[Int]()
    tokenizer.encode(text, add_special, session, out)
    return out^


def _same(got: List[Int], want: List[Int]) -> Bool:
    if len(got) != len(want):
        return False
    for i in range(len(got)):
        if got[i] != want[i]:
            return False
    return True


def _encodes(
    mut suite: Suite,
    tokenizer: Tokenizer,
    mut session: Session,
    text: StringSpan,
    add_special: Bool,
    want: List[Int],
    name: String,
) raises:
    var got = _ids(tokenizer, session, text, add_special)
    suite.check(_same(got, want), name)


def _bpe(mut suite: Suite, dir: String) raises:
    suite.group("byte level bpe")
    var path = dir + "/bpe.json"
    _write(path, _bpe_json())
    var counter = AllocCounter()
    var tokenizer = Tokenizer(path, counter.raw())
    var session = Session()

    _encodes(
        suite,
        tokenizer,
        session,
        "hello world",
        False,
        [259, 264],
        "a word that merges all the way is one id",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "hello the world",
        False,
        [259, 265, 256, 264],
        "the lowest rank merges first even when a longer token exists",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        " hello",
        False,
        [32, 259],
        "a leading space becomes its own piece",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "",
        False,
        List[Int](),
        "empty text is no ids at all",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "hello<|endoftext|>world",
        False,
        [259, 268, 119, 111, 114, 108, 100],
        "an added token in the middle of a word is still one id",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "Café RUNNING un中文!",
        False,
        [
            67,
            97,
            102,
            195,
            169,
            32,
            82,
            85,
            78,
            78,
            73,
            78,
            71,
            32,
            117,
            110,
            228,
            184,
            173,
            230,
            150,
            135,
            33,
        ],
        "everything outside the merges falls back to single bytes",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "  spaced  out  ",
        False,
        [32, 32, 115, 112, 97, 99, 101, 100, 32, 32, 111, 117, 116, 32, 32],
        "runs of spaces survive the split exactly",
    )

    var pair = List[Int]()
    var types = List[Int]()
    tokenizer.encode_pair(
        "hello world", "the world", True, True, False, session, pair, types
    )
    suite.check(
        _same(pair, [259, 264, 116, 256, 264]), "a pair with no processor joins"
    )
    suite.check(
        _same(types, [0, 0, 1, 1, 1]), "and the second sequence types 1"
    )

    var keep: List[Int] = [259, 268, 119, 111, 114, 108, 100]
    suite.check(
        tokenizer.decode(keep, False) == "hello<|endoftext|>world",
        "decoding keeps the special token when asked to",
    )
    suite.check(
        tokenizer.decode(keep, True) == "helloworld",
        "and drops it when asked to skip it",
    )
    suite.check(
        tokenizer.decode([228, 184, 173, 230, 150, 135], True) == "中文",
        "single byte ids join back into the characters they came from",
    )

    _stream(suite, tokenizer)
    _ = unlink(path)


def _stream(mut suite: Suite, tokenizer: Tokenizer) raises:
    """A three byte character arriving one byte at a time.

    The whole point of the streaming decoder. Each of the first two ids carries
    part of a character and nothing else, and a decoder that printed each token
    as it arrived would print two replacement characters before the character
    itself.
    """
    var ids: List[Int] = [228, 184, 173, 230, 150, 135]
    var stream = DecodeStream(True)
    var joined = String("")
    var empties = 0
    for i in range(len(ids)):
        var piece = stream.step(tokenizer, ids[i])
        if piece == "":
            empties += 1
        joined += piece
    suite.check(joined == "中文", "the stream ends up with the whole text")
    suite.check(empties == 4, "and says nothing until a character is complete")


def _wordpiece(mut suite: Suite, dir: String) raises:
    suite.group("wordpiece")
    var path = dir + "/wp.json"
    _write(path, _wordpiece_json())
    var counter = AllocCounter()
    var tokenizer = Tokenizer(path, counter.raw())
    var session = Session()

    _encodes(
        suite,
        tokenizer,
        session,
        "hello world",
        False,
        [5, 6],
        "two words are two ids",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "hello world",
        True,
        [2, 5, 6, 3],
        "and the template wraps them",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "",
        True,
        [2, 3],
        "an empty string still gets its markers",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "Café RUNNING un中文!",
        True,
        [2, 11, 14, 15, 7, 1, 1, 12, 3],
        "case, accents and cjk spacing all happen before the vocabulary",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "hello<|endoftext|>world",
        False,
        [5, 1, 1, 1, 1, 1, 6],
        "punctuation splits and what is left over is unknown",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "  spaced  out  ",
        False,
        [1, 1],
        "a word with no pieces behind it is one unknown, not several",
    )

    var pair = List[Int]()
    var types = List[Int]()
    tokenizer.encode_pair(
        "hello world", "the world", True, True, False, session, pair, types
    )
    suite.check(
        _same(pair, [2, 5, 6, 3, 10, 6, 3]), "the pair template puts two in one"
    )
    suite.check(
        _same(types, [0, 0, 0, 0, 1, 1, 1]),
        "with the type ids the file asks for",
    )

    suite.check(
        tokenizer.decode([7, 8, 9], True) == "unaffable",
        "the decoder joins the continuation pieces back into a word",
    )
    suite.check(
        tokenizer.decode([2, 5, 6, 3], False) == "[CLS] hello world [SEP]",
        "and keeps the markers when they are not skipped",
    )
    suite.check(
        tokenizer.decode([11, 14, 15, 7, 1, 1, 12], True) == "cafe running un!",
        "cleanup pulls the punctuation back against the word",
    )
    _ = unlink(path)


def _unigram(mut suite: Suite, dir: String) raises:
    suite.group("unigram")
    var path = dir + "/uni.json"
    _write(path, _unigram_json())
    var counter = AllocCounter()
    var tokenizer = Tokenizer(path, counter.raw())
    var session = Session()

    _encodes(
        suite,
        tokenizer,
        session,
        "hello world",
        False,
        [2, 3],
        "the highest scoring path is two whole words",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "hello the world",
        False,
        [2, 6, 3],
        "and three when there are three",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        " hello",
        False,
        [2],
        "a leading space is the one metaspace already put there",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "  spaced  out  ",
        False,
        [1, 0, 8, 12, 1, 5, 0, 1],
        "what the pieces cannot spell becomes the unknown id",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "Café RUNNING un中文!",
        False,
        [1, 0, 1, 0, 1, 0],
        "and a whole word of it is one unknown per word",
    )

    suite.check(
        tokenizer.decode([2, 3], True) == "hello world",
        "metaspace decoding turns the marks back into spaces",
    )
    suite.check(
        tokenizer.decode([6, 4, 5], True) == "thehello",
        "and drops only the one at the very front",
    )
    _ = unlink(path)


def _fallback(mut suite: Suite, dir: String) raises:
    suite.group("byte fallback bpe")
    var path = dir + "/fb.json"
    _write(path, _fallback_json())
    var counter = AllocCounter()
    var tokenizer = Tokenizer(path, counter.raw())
    var session = Session()

    _encodes(
        suite,
        tokenizer,
        session,
        "hello world",
        False,
        [5, 114, 111, 118, 118, 7, 5, 8, 7, 124, 118, 110],
        "letters with no merge behind them become byte tokens",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "hello world",
        True,
        [1, 5, 114, 111, 118, 118, 7, 5, 8, 7, 124, 118, 110],
        "and the template puts the beginning of text token in front",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "é",
        False,
        [5, 205, 179],
        "a two byte character becomes two byte tokens",
    )

    suite.check(
        tokenizer.decode([5, 114, 111, 118, 118, 7], True) == "hello",
        "byte tokens fuse back into the word",
    )
    suite.check(
        tokenizer.decode([5, 205, 179], True) == "é",
        "and back into a character that spans two of them",
    )
    suite.check(
        tokenizer.decode([1, 3, 4], False) == "<s> hello world",
        "the marker is kept when it is not skipped",
    )
    suite.check(
        tokenizer.decode([1, 3, 4], True) == "hello world",
        "and the leading space goes with it when it is",
    )

    # BOS reconciliation. The reference gives [1, 1, ...] here, which is the
    # bug this is meant to avoid rather than a behaviour to copy.
    var double = _ids(tokenizer, session, "<s>hello", True)
    suite.check(
        _same(double, [1, 1, 5, 114, 111, 118, 118, 7]),
        "plain encoding writes the beginning of text token twice",
    )
    var once = List[Int]()
    tokenizer.encode_rendered("<s>hello", session, once)
    suite.check(
        _same(once, [1, 5, 114, 111, 118, 118, 7]),
        "and encoding rendered text writes it once",
    )
    _ = unlink(path)


def _flags(mut suite: Suite, dir: String) raises:
    suite.group("added token flags")
    var path = dir + "/fl.json"
    _write(path, _flags_json())
    var counter = AllocCounter()
    var tokenizer = Tokenizer(path, counter.raw())
    var session = Session()

    _encodes(
        suite,
        tokenizer,
        session,
        "a <L> b",
        False,
        [1, 7, 2],
        "lstrip eats the space in front of the token",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "a<L>b",
        False,
        [1, 7, 2],
        "and does nothing when there is no space to eat",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "a <R> b",
        False,
        [1, 8, 2],
        "rstrip eats the space behind it",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "a<R> b",
        False,
        [1, 8, 2],
        "from either side of it",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "a <W> b",
        False,
        [1, 9, 2],
        "single word matches when the token stands on its own",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "ax<W>yb",
        False,
        [0, 0, 0, 0, 0],
        "and does not match when it is glued to a word",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "ab<W>",
        False,
        [3, 0, 0, 0],
        "a word boundary on the left is not enough on its own",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "<W>ab",
        False,
        [0, 0, 0, 3],
        "and neither is one on the right",
    )
    _ = unlink(path)


def _refuses(mut suite: Suite, dir: String) raises:
    suite.group("tokenizer files that are refused")
    var path = dir + "/bad.json"

    _write(path, String("not json at all"))
    suite.check(not _opens(path), "a file that is not json is refused")

    _write(path, String('{"model":{"type":"Nonsense","vocab":{}}}'))
    suite.check(not _opens(path), "a model kind nobody has heard of is refused")

    _write(
        path,
        String(
            '{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},'
            + '"normalizer":{"type":"Precompiled","precompiled_charsmap":"x"}}'
        ),
    )
    suite.check(
        not _opens(path),
        "a normalizer that would silently do nothing is refused",
    )
    _ = unlink(path)


def _opens(path: String) -> Bool:
    try:
        var counter = AllocCounter()
        var tokenizer = Tokenizer(path, counter.raw())
        _ = tokenizer^
        return True
    except:
        return False


def run(mut suite: Suite) raises:
    var dir = _temp_dir()
    try:
        _bpe(suite, dir)
        _wordpiece(suite, dir)
        _unigram(suite, dir)
        _fallback(suite, dir)
        _flags(suite, dir)
        _refuses(suite, dir)
    except e:
        _ = rmdir(dir)
        raise e
    _ = rmdir(dir)
