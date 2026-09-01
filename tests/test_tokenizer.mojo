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
from molla.text.props import Unicode
from molla.text.utf8 import encode
from molla.tokenizer.bytelevel import byte_to_point
from molla.tokenizer.precompiled import Precompiled
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


def _untyped_bpe_json() -> String:
    """A BPE model with no `type` and the vocabulary written first.

    Shaped after `openai-community/gpt2`, which really is written this way. The
    member order matters because a vocabulary cannot be read before the type
    says whether it is an object or an array, and the missing type matters
    because the reference implementation works it out from the other members
    rather than refusing the file.
    """
    return String(
        '{"version":"1.0","truncation":null,"padding":null,'
        + '"added_tokens":[],"normalizer":null,'
        + '"pre_tokenizer":{"type":"Whitespace"},'
        + '"post_processor":null,"decoder":null,"model":{"vocab":'
        + '{"h":0,"e":1,"l":2,"o":3,"w":4,"r":5,"d":6,"he":7,"ll":8,'
        + '"hell":9,"hello":10,"wo":11,"wor":12,"worl":13,"world":14},'
        + '"merges":["h e","l l","he ll","hell o","w o","wo r","wor l",'
        + '"worl d"],"dropout":null,"unk_token":null,'
        + '"continuing_subword_prefix":"","end_of_word_suffix":"",'
        + '"fuse_unk":false}}'
    )


def _untyped_unigram_json() -> String:
    """No type either, and the vocabulary is an array rather than an object,
    which is the one character that says Unigram."""
    return String(
        '{"version":"1.0","truncation":null,"padding":null,'
        + '"added_tokens":[],"normalizer":null,'
        + '"pre_tokenizer":{"type":"Whitespace"},'
        + '"post_processor":null,"decoder":null,"model":{"unk_id":0,'
        + '"vocab":[["<unk>",0.0],["a",-1.0],["b",-1.5],["ab",-1.2],'
        + '["c",-3.0]],"byte_fallback":false}}'
    )


def _untyped_wordpiece_json() -> String:
    """No type, no merges, and a word length limit, which is what tells
    WordPiece apart from a plain word level vocabulary."""
    return String(
        '{"version":"1.0","truncation":null,"padding":null,'
        + '"added_tokens":[],"normalizer":null,'
        + '"pre_tokenizer":{"type":"Whitespace"},'
        + '"post_processor":null,"decoder":null,"model":{"vocab":'
        + '{"[UNK]":0,"play":1,"##ing":2,"##ed":3},"unk_token":"[UNK]",'
        + '"continuing_subword_prefix":"##","max_input_chars_per_word":100}}'
    )


def _word_level_json(typed: Bool) -> String:
    """A word level vocabulary, once with its type and once without.

    The two files have to give the same ids, because the untyped one is what
    is left when a vocabulary has no merges and no word length limit.
    """
    var kind = String('"type":"WordLevel",') if typed else String("")
    return (
        '{"version":"1.0","truncation":null,"padding":null,'
        + '"added_tokens":[],"normalizer":null,'
        + '"pre_tokenizer":{"type":"Whitespace"},'
        + '"post_processor":null,"decoder":null,"model":{'
        + kind
        + '"vocab":{"[UNK]":0,"hello":1,"world":2,"cat":3},'
        + '"unk_token":"[UNK]"}}'
    )


def _nmt_json() -> String:
    """Three whole strings in a vocabulary, so the cleanup shows up as an id.

    There is no pre-tokenizer, so the whole string is one word and the id says
    exactly what the normalizer left behind: `a b` when the character in the
    middle became a space, `ab` when it was deleted, and the unknown token when
    it was left alone.
    """
    return String(
        '{"version":"1.0","truncation":null,"padding":null,'
        + '"added_tokens":[],"normalizer":{"type":"Nmt"},'
        + '"pre_tokenizer":null,"post_processor":null,"decoder":null,'
        + '"model":{"type":"WordLevel","vocab":{"[UNK]":0,"a b":1,"ab":2},'
        + '"unk_token":"[UNK]"}}'
    )


def _charsmap() -> String:
    """A sentencepiece charsmap with six rules in it, built by hand.

    The real ones are a quarter of a megabyte and there is no way to read one,
    so this is the same format with a trie small enough to reason about. The
    rules are `A` to `a`, a no break space to nothing, `a` and a combining
    acute to a precomposed one, `e` to a capital `E`, `e` and a combining
    acute to a precomposed one, and `o` with three combining marks after it to
    a `Z`.

    The last two rules are there because they can never fire. The `e` rule is
    a prefix of the `e` acute rule and the search takes the shorter one, and
    the `o` rule is a seven byte key which is past the point where a cluster
    stops being looked up whole. Both are the reference implementation's
    behaviour rather than anything intended, and a rewrite that fixed either
    of them would break real files.
    """
    return String(
        "QAMAAAAEAAAAAAAAAAAAgAAAAAAAAAAAAwAAgAYAAIAAAAAACAAAgAAAAAAA"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAsAAIAAAAAAAgAAgAAAAAAAAAAAAAAAAAAA"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABBCQEAAAAAAAAA"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGGMAQAAAAAAAAAAAAAAAABl"
        + "iQEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAb5wBAAAA"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACBAQIA"
        + "gTkCAAAAAAAAAAAAAAAAAAAAAACBAAIAAAAAAIMJAgCCDAIAAAAAAAAAAAAA"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAoIUCAAAAAAAAAAAA"
        + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        + "AAAAAAAAAAAAAADMPAMAAAAAAMJMAwAAAAAAzDwDAAAAAADMLAMAAAAAAAAA"
        + "AADMNAMAAAAAAAAAAAAAAAAAAAAAAMwsAwBhAADDoQBFAMOpAFoA"
    )


def _precompiled_json() -> String:
    """The charsmap in front of a word level vocabulary, the way T5 is shaped.

    No pre-tokenizer, so the whole string is one word and the id is a straight
    reading of what the normalizer left behind.
    """
    return String(
        '{"version":"1.0","truncation":null,"padding":null,'
        + '"added_tokens":[],"normalizer":{"type":"Precompiled",'
        + '"precompiled_charsmap":"'
        + _charsmap()
        + '"},"pre_tokenizer":null,"post_processor":null,"decoder":null,'
        + '"model":{"type":"WordLevel","vocab":{"[UNK]":0,"a":1,'
        + '"\\u00e1":2,"E":3,"ab":4},"unk_token":"[UNK]"}}'
    )


def _prefix_json() -> String:
    """A byte level prefix space behind a whitespace split.

    This is the shape a few dozen of the real files have, and it is the one
    that says whether the prefix space goes on the first piece or on all of
    them. The vocabulary is the byte alphabet and nothing else, so every id is
    the byte it stands for and the answer reads as the string it came from.
    """
    return String(
        '{"version":"1.0","truncation":null,"padding":null,'
        + '"added_tokens":[],"normalizer":null,'
        + '"pre_tokenizer":{"type":"Sequence","pretokenizers":['
        + '{"type":"Whitespace"},{"type":"ByteLevel",'
        + '"add_prefix_space":true,"trim_offsets":true,"use_regex":false}]},'
        + '"post_processor":null,"decoder":{"type":"ByteLevel",'
        + '"add_prefix_space":true,"trim_offsets":true,"use_regex":false},'
        + '"model":{"type":"BPE","vocab":{'
        + _byte_level_vocab()
        + '},"merges":[],"dropout":null,"unk_token":null,'
        + '"continuing_subword_prefix":"","end_of_word_suffix":"",'
        + '"fuse_unk":false}}'
    )


def _template_json() -> String:
    """Three separate specials in front of the text, the way Whisper writes it.

    Whisper opens with a start marker, then a language, then a timestamp
    setting, and each one is its own item rather than one run of three. The id
    the reconciliation rule is about is the third, because that is the one
    sitting next to the text, and a rule that only looked at the first item
    would never fire here.
    """
    return String(
        '{"version":"1.0","truncation":null,"padding":null,"added_tokens":['
        + _added(10, "<|s|>", False, False, False)
        + ","
        + _added(11, "<|n|>", False, False, False)
        + ","
        + _added(12, "<|e|>", False, False, False)
        + '],"normalizer":null,"pre_tokenizer":null,"decoder":null,'
        + '"post_processor":{"type":"TemplateProcessing","single":['
        + '{"SpecialToken":{"id":"<|s|>","type_id":0}},'
        + '{"SpecialToken":{"id":"<|n|>","type_id":0}},'
        + '{"Sequence":{"id":"A","type_id":0}},'
        + '{"SpecialToken":{"id":"<|e|>","type_id":0}}],'
        + '"pair":[{"Sequence":{"id":"A","type_id":0}},'
        + '{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{'
        + '"<|s|>":{"id":"<|s|>","ids":[10],"tokens":["<|s|>"]},'
        + '"<|n|>":{"id":"<|n|>","ids":[11],"tokens":["<|n|>"]},'
        + '"<|e|>":{"id":"<|e|>","ids":[12],"tokens":["<|e|>"]}}},'
        + '"model":{"type":"WordLevel","vocab":{"hello":4,"[UNK]":5,'
        + '"<|s|>":10,"<|n|>":11,"<|e|>":12},"unk_token":"[UNK]"}}'
    )


def _shapes(mut suite: Suite, dir: String) raises:
    """Files that say what model they are in a roundabout way.

    Every id in here came out of the reference implementation reading the same
    bytes. A file with no `type` is not a broken file, it is what the oldest
    and most downloaded tokenizers on the hub look like, and refusing one means
    refusing GPT-2.
    """
    suite.group("model shapes")
    var path = dir + "/sh.json"
    var counter = AllocCounter()

    _write(path, _untyped_bpe_json())
    var bpe = Tokenizer(path, counter.raw())
    var session = Session()
    _encodes(
        suite,
        bpe,
        session,
        "hello world",
        False,
        [10, 14],
        "a bpe model with no type and its vocabulary written first",
    )
    _encodes(suite, bpe, session, "hell", False, [9], "and it merges")
    _encodes(
        suite,
        bpe,
        session,
        "ohh",
        False,
        [3, 0, 0],
        "and leaves alone what it cannot merge",
    )
    _ = bpe^

    _write(path, _untyped_unigram_json())
    var unigram = Tokenizer(path, counter.raw())
    var us = Session()
    _encodes(
        suite,
        unigram,
        us,
        "ab",
        False,
        [3],
        "an array vocabulary with no type is unigram",
    )
    _encodes(suite, unigram, us, "abc", False, [3, 4], "and scores its path")
    _encodes(suite, unigram, us, "zz", False, [0], "and falls back to unknown")
    _ = unigram^

    _write(path, _untyped_wordpiece_json())
    var wordpiece = Tokenizer(path, counter.raw())
    var ws = Session()
    _encodes(
        suite,
        wordpiece,
        ws,
        "playing",
        False,
        [1, 2],
        "a word length limit with no type is wordpiece",
    )
    _encodes(suite, wordpiece, ws, "played", False, [1, 3], "and it continues")
    _encodes(suite, wordpiece, ws, "zzz", False, [0], "and gives up as a whole")
    _ = wordpiece^

    # The same file twice, once saying what it is and once not, because the
    # untyped one is only right if it lands on the same model.
    for typed in [True, False]:
        _write(path, _word_level_json(typed))
        var level = Tokenizer(path, counter.raw())
        var ls = Session()
        var said = String(" with a type") if typed else String(" without one")
        _encodes(
            suite,
            level,
            ls,
            "hello world",
            False,
            [1, 2],
            "a word level vocabulary" + said,
        )
        _encodes(
            suite,
            level,
            ls,
            "hello dog",
            False,
            [1, 0],
            "and an unseen word costs one id" + said,
        )
        _encodes(suite, level, ls, "cat", False, [3], "and a seen one" + said)
        _ = level^
    _ = unlink(path)


def _nmt(mut suite: Suite, dir: String) raises:
    """The sentencepiece cleanup, one character at a time.

    Two lists that are nearly the same and are not: some characters become a
    space and some are deleted, and neither list is what anything else calls
    whitespace. Every answer here came out of the reference implementation.
    """
    suite.group("nmt normalizer")
    var path = dir + "/nmt.json"
    _write(path, _nmt_json())
    var counter = AllocCounter()
    var tokenizer = Tokenizer(path, counter.raw())
    var session = Session()

    _encodes(suite, tokenizer, session, "a\tb", False, [1], "a tab is a space")
    _encodes(
        suite, tokenizer, session, "a\nb", False, [1], "and so is a newline"
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "a\x0Cb",
        False,
        [1],
        "and so is a form feed",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "a\rb",
        False,
        [1],
        "and so is a carriage return",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "a\x0Bb",
        False,
        [2],
        "a vertical tab is deleted rather than spaced",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "a​b",
        False,
        [1],
        "a zero width space is a space",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "a b",
        False,
        [0],
        "and an en quad just above it is left alone",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "a b",
        False,
        [0],
        "and so is a no break space",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "a▁b",
        False,
        [1],
        "the metaspace character becomes a space",
    )
    _ = unlink(path)


def _points_hex(points: List[Int]) -> String:
    """Code points as space separated hex, the way a mismatch reads."""
    var digits = String("0123456789ABCDEF")
    var out = String("")
    for i in range(len(points)):
        if i > 0:
            out += " "
        var cp = points[i]
        var one = String("")
        while cp > 0:
            var nibble = cp & 0xF
            one = digits[byte = nibble : nibble + 1] + one
            cp >>= 4
        if one.byte_length() == 0:
            one = "0"
        out += one
    return out^


def _precompiled(mut suite: Suite, dir: String) raises:
    """The sentencepiece charsmap, on a map small enough to read.

    Two of the checks here are about behaviour that looks like a bug and is
    not. The reference takes the shortest rule that matches rather than the
    longest, and it only looks a cluster up whole when the cluster is under
    six bytes. Both are pinned, because sixty files in the corpus tokenize
    differently if either changes.
    """
    suite.group("precompiled normalizer")
    var path = dir + "/pc.json"
    _write(path, _precompiled_json())
    var counter = AllocCounter()
    var tokenizer = Tokenizer(path, counter.raw())
    var session = Session()

    # Everything with a combining mark in it is built out of chr, because a
    # mark in a source file is invisible and a precomposed letter that
    # looked the same would take a different path through the map.
    _encodes(suite, tokenizer, session, "A", False, [1], "a rule rewrites")
    _encodes(
        suite,
        tokenizer,
        session,
        String("a") + chr(0x301),
        False,
        [2],
        "a letter and a combining mark are looked up as one cluster",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        String("a") + chr(0xA0) + "b",
        False,
        [4],
        "a rule with an empty replacement deletes what it matched",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        String("e") + chr(0x301),
        False,
        [3],
        "the shortest rule wins and takes the whole cluster with it",
    )
    _ = tokenizer^
    _ = unlink(path)

    # The six byte limit, which needs a cluster nothing shorter matches and so
    # cannot be said in a vocabulary. The rule for o with three marks is in
    # the map and the cluster is seven bytes, so the lookup never happens and
    # all four code points come through untouched.
    var tables = Unicode()
    var encoded = _charsmap()
    var map = Precompiled(encoded.as_bytes())
    var stacked: List[Int] = [0x6F, 0x301, 0x302, 0x303]
    suite.check(
        _points_hex(map.apply(tables, stacked)) == "6F 301 302 303",
        "a cluster of six bytes or more is not looked up whole",
    )
    var plain: List[Int] = [0x62, 0x63]
    suite.check(
        _points_hex(map.apply(tables, plain)) == "62 63",
        "and a character no rule mentions is copied across",
    )
    var empty = List[Int]()
    suite.check(
        _points_hex(map.apply(tables, empty)) == "",
        "and an empty string stays empty",
    )


def _prefix(mut suite: Suite, dir: String) raises:
    """The byte level prefix space, once there is more than one piece."""
    suite.group("byte level prefix space")
    var path = dir + "/px.json"
    _write(path, _prefix_json())
    var counter = AllocCounter()
    var tokenizer = Tokenizer(path, counter.raw())
    var session = Session()

    _encodes(
        suite,
        tokenizer,
        session,
        "a b",
        False,
        [32, 97, 32, 98],
        "every piece gets the space and not just the first",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "a  b",
        False,
        [32, 97, 32, 98],
        "which is what makes two spaces read as one",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        " a",
        False,
        [32, 97],
        "a piece that has the space already does not get another",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "   ",
        False,
        [],
        "nothing but spaces produces no tokens at all",
    )
    _encodes(
        suite, tokenizer, session, "", False, [], "and neither does nothing"
    )
    _ = unlink(path)


def _template(mut suite: Suite, dir: String) raises:
    """A template with three specials in front of the text, and the rule about
    the last of them."""
    suite.group("template with several leading specials")
    var path = dir + "/tp.json"
    _write(path, _template_json())
    var counter = AllocCounter()
    var tokenizer = Tokenizer(path, counter.raw())
    var session = Session()

    _encodes(
        suite,
        tokenizer,
        session,
        "hello",
        True,
        [10, 11, 4, 12],
        "the template writes all three specials",
    )
    _encodes(
        suite,
        tokenizer,
        session,
        "<|n|>hello",
        True,
        [10, 11, 11, 4, 12],
        "and writes the third one twice when the text has it already",
    )
    var once = List[Int]()
    tokenizer.encode_rendered("<|n|>hello", session, once)
    suite.check(
        _same(once, [10, 11, 4, 12]),
        "encoding rendered text drops the template's copy of it",
    )

    # The first special doubles up here rather than the third. Nothing is
    # dropped, because the rule is about the id next to the text and this one
    # is two places away from it.
    _encodes(
        suite,
        tokenizer,
        session,
        "<|s|>hello",
        True,
        [10, 11, 10, 4, 12],
        "a different special in the text doubles up on its own",
    )
    var kept = List[Int]()
    tokenizer.encode_rendered("<|s|>hello", session, kept)
    suite.check(
        _same(kept, [10, 11, 10, 4, 12]),
        "and rendered text leaves that one alone",
    )
    _ = unlink(path)


def _refuses(mut suite: Suite, dir: String) raises:
    suite.group("tokenizer files that are refused")
    var path = dir + "/bad.json"

    _write(path, String("not json at all"))
    suite.check(not _opens(path), "a file that is not json is refused")

    _write(path, String('{"model":{"type":"Nonsense","vocab":{}}}'))
    suite.check(not _opens(path), "a model kind nobody has heard of is refused")

    # A charsmap is a quarter of a megabyte of base64 in the middle of a file
    # and there is nothing in the format that would catch a truncated one, so
    # the loader checks the three things it can check and refuses rather than
    # walking a trie off the end of itself.
    _write(path, _bad_charsmap("x"))
    suite.check(
        not _opens(path), "a charsmap too short to hold a trie is refused"
    )
    _write(path, _bad_charsmap("!!!!"))
    suite.check(not _opens(path), "a charsmap that is not base64 is refused")
    _write(path, _bad_charsmap("6AMAAEFBQUE="))
    suite.check(
        not _opens(path),
        "and one whose trie runs past the end of the blob is refused",
    )
    _ = unlink(path)


def _bad_charsmap(encoded: StringSpan) -> String:
    """A tokenizer that is fine apart from the charsmap it carries."""
    return String(
        '{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},'
        + '"normalizer":{"type":"Precompiled","precompiled_charsmap":"'
        + String(encoded)
        + '"}}'
    )


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
        _shapes(suite, dir)
        _nmt(suite, dir)
        _precompiled(suite, dir)
        _prefix(suite, dir)
        _template(suite, dir)
        _refuses(suite, dir)
    except e:
        _ = rmdir(dir)
        raise e
    _ = rmdir(dir)
