"""A Hugging Face model directory, read into the same spec a GGUF file gives.

GGUF puts everything in one file. A Hugging Face repository spreads the same
facts across five, and the mapping is the whole job here:

| Fact | GGUF | Repository |
| --- | --- | --- |
| architecture | `general.architecture` | `config.json` `architectures[0]` |
| geometry | `<arch>.` prefixed keys | `config.json` |
| tokenizer | `tokenizer.ggml.` keys | `tokenizer.json` |
| special tokens | `tokenizer.ggml.*_token_id` | `tokenizer_config.json` |
| chat template | `tokenizer.chat_template` | `tokenizer_config.json` |
| weights | the tensor directory | `model.safetensors` and its index |

The result is a `ModelSpec`, the same struct, printed by the same function. That
is the point of doing it this way: the layers above should never have to ask
which format a model arrived in, and the two paths agreeing on the numbers is
something a person can check by running the command twice.

The three places the formats genuinely disagree are worth naming, because they
show up as different numbers rather than as errors.

A GGUF token list is padded up to the row count of the embedding matrix and a
`tokenizer.json` vocabulary is not, so the same model reports 151936 tokens as
GGUF and 151665 as a repository. Both are right about a different question, and
the report carries both once it knows them.

A repository states nothing about how the weights are laid out beyond the byte
ranges in the header, while GGUF states an alignment and pads to it. So the
directory audit checks contiguity in both cases and means slightly different
things by it.

The tokenizer names are spelled differently for the same three algorithms. GGUF
calls them after the model each first shipped on, gpt2 and llama and bert, while
`tokenizer.json` names the algorithm, BPE and Unigram and WordPiece. Both are
normalised to one word by `tokenizer_algorithm`, and the report quotes what the
file actually said beside it.

Nothing here reads a weight either. The tokenizer file is the only large thing
that gets touched, and it is streamed rather than held: a 33 MB `tokenizer.json`
with 262144 entries is walked for counts and the handful of ids we need, and
none of it is kept.
"""

from molla.io.buffer import Buffer
from molla.json.dom import (
    JS_ARRAY,
    JS_OBJECT,
    JS_STRING,
    NO_NODE,
    Document,
    parse,
)
from molla.json.reader import (
    EV_ARRAY_BEGIN,
    EV_ARRAY_END,
    EV_END,
    EV_ERROR,
    EV_KEY,
    EV_NUMBER,
    EV_OBJECT_BEGIN,
    EV_OBJECT_END,
    EV_STRING,
    Reader,
)
from molla.model.safetensors import (
    SafeTensors,
    dtype_bytes,
    dtype_narrow,
)
from molla.model.spec import (
    ARCH_BERT,
    ARCH_NOMIC_BERT,
    ARCH_UNKNOWN,
    CAP_EMBEDDING,
    CAP_TEXT,
    CAP_VISION,
    Geometry,
    Layout,
    ModelSpec,
    TokenizerSpec,
    architecture_id,
    engine_capabilities,
    is_causal,
    report,
    run_spec,
    tokenizer_algorithm,
)
from molla.sys.file import FileInfo, exists, stat_path
from molla.sys.mem import AllocCounter
from molla.sys.mmap import Mapping


def _text(bytes: Span[UInt8, MutAnyOrigin]) -> String:
    var out = List[UInt8]()
    out.reserve(len(bytes))
    for i in range(len(bytes)):
        out.append(bytes[i])
    return String(StringSpan(unsafe_from_utf8=out))


def class_architecture(name: StringSpan) -> String:
    """Map a transformers class name onto the architecture GGUF would name.

    `architectures[0]` is the class transformers will instantiate, which is one
    string that answers both what family the model is and what head is on top of
    it. Only the family is wanted here, and the classes that share a family are
    listed rather than pattern matched, for the same reason `architecture_id`
    has a short list: a name nobody has checked comes back as itself and reports
    as unrecognised, instead of being filed under llama because it ends in
    ForCausalLM.
    """
    if name == "LlamaForCausalLM" or name == "LlamaModel":
        return String("llama")
    if name == "Qwen2ForCausalLM" or name == "Qwen2Model":
        return String("qwen2")
    if name == "Qwen3ForCausalLM" or name == "Qwen3Model":
        return String("qwen3")
    if name == "GemmaForCausalLM":
        return String("gemma")
    if name == "Gemma2ForCausalLM":
        return String("gemma2")
    if (
        name == "Gemma3ForCausalLM"
        or name == "Gemma3ForConditionalGeneration"
        or name == "Gemma3TextModel"
    ):
        return String("gemma3")
    if name == "Phi3ForCausalLM":
        return String("phi3")
    if (
        name == "BertModel"
        or name == "BertForMaskedLM"
        or name == "BertForSequenceClassification"
    ):
        return String("bert")
    if name == "NomicBertModel":
        return String("nomic-bert")
    if name == "CLIPModel" or name == "CLIPVisionModel":
        return String("clip")
    return String(name)


struct JsonFile(Movable):
    """A small JSON file, mapped and parsed, or absent.

    Absent is not an error. `generation_config.json` is missing from every
    embedding repository and `tokenizer_config.json` from a few, and a caller
    asking a missing file for a key should get the same answer as one asking a
    file that does not have that key. So an absent file parses to no root and
    every lookup falls through to its default.

    The mapping, the document and the reader live together because they have to:
    a parsed document holds spans into the bytes it was parsed from, so freeing
    the mapping first leaves the strings pointing at unmapped pages.
    """

    var path: String
    var present: Bool
    var mapping: Mapping
    var doc: Document
    var reader: Reader

    def __init__(out self, path: String, counter: Int) raises:
        self.path = path
        self.doc = Document(counter)
        self.reader = Reader(counter, 4096)
        self.present = exists(path)
        self.mapping = Mapping()
        if not self.present:
            return
        self.mapping = Mapping(path)
        var body = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=self.mapping.base(), length=self.mapping.length
        )
        if not parse(self.doc, self.reader, body):
            var at = self.doc.error_at
            self.mapping.close()
            raise Error(path + " is not valid json, at byte " + String(at))

    def root(self) -> Int:
        return self.doc.root

    def close(mut self):
        self.mapping.close()


def _first_int(doc: Document, node: Int, fallback: Int) -> Int:
    """An integer, or the first element when the value is a list.

    `eos_token_id` is an integer on most models and a list on the ones that can
    stop on more than one token. The first is the one the model was trained to
    emit, and the rest belong to the sampler rather than to the spec.
    """
    if node == NO_NODE:
        return fallback
    if doc.kind(node) == JS_ARRAY:
        return doc.as_int(doc.at(node, 0), fallback)
    return doc.as_int(node, fallback)


def read_repo_geometry(doc: Document, node: Int) -> Geometry:
    """`config.json` in the terms the engine asks for.

    The defaults are the same defaults the GGUF path takes and they are recorded
    the same way, because they are the same guesses: a missing
    `num_key_value_heads` means multi head attention and a missing `head_dim`
    means the embedding divides evenly across the heads.
    """
    var heads = doc.get_int(node, "num_attention_heads", 0)
    var embedding = doc.get_int(node, "hidden_size", 0)

    var kv_stated = doc.get(node, "num_key_value_heads") != NO_NODE
    var heads_kv = doc.get_int(node, "num_key_value_heads", heads)

    var head_dim_stated = doc.get(node, "head_dim") != NO_NODE
    var derived = embedding // heads if heads > 0 else 0
    var key_length = doc.get_int(node, "head_dim", derived)

    # A rotary factor below one rotates part of the head and leaves the rest,
    # which is what the phi models do. Everything else rotates all of it.
    var partial = doc.get_double(node, "partial_rotary_factor", 0.0)
    var rope_dims_stated = doc.get(node, "partial_rotary_factor") != NO_NODE
    var rope_dims = key_length
    if rope_dims_stated:
        rope_dims = Int(partial * Float64(key_length))

    var scaling = doc.get(node, "rope_scaling")
    var rope_scaling = String("unstated")
    var rope_factor = 0.0
    if scaling != NO_NODE and doc.kind(scaling) == JS_OBJECT:
        rope_factor = doc.get_double(scaling, "factor", 0.0)
        var kind = doc.get(scaling, "rope_type")
        if kind == NO_NODE:
            kind = doc.get(scaling, "type")
        if kind != NO_NODE and doc.kind(kind) == JS_STRING:
            rope_scaling = _text(doc.text(kind))

    var epsilon = doc.get_double(node, "rms_norm_eps", 0.0)
    if epsilon == 0.0:
        epsilon = doc.get_double(node, "layer_norm_eps", 0.0)
    if epsilon == 0.0:
        epsilon = doc.get_double(node, "layer_norm_epsilon", 0.0)

    var experts = doc.get_int(node, "num_local_experts", 0)
    if experts == 0:
        experts = doc.get_int(node, "num_experts", 0)

    return Geometry(
        block_count=doc.get_int(node, "num_hidden_layers", 0),
        context_length=doc.get_int(node, "max_position_embeddings", 0),
        embedding_length=embedding,
        feed_forward_length=doc.get_int(node, "intermediate_size", 0),
        head_count=heads,
        head_count_kv=heads_kv,
        key_length=key_length,
        value_length=key_length,
        expert_count=experts,
        expert_used_count=doc.get_int(node, "num_experts_per_tok", 0),
        rope_dimension_count=rope_dims,
        rope_freq_base=doc.get_double(node, "rope_theta", 0.0),
        rope_scale_factor=rope_factor,
        rope_scaling=rope_scaling,
        epsilon=epsilon,
        head_dim_stated=head_dim_stated,
        kv_stated=kv_stated,
        rope_dims_stated=rope_dims_stated,
    )


@fieldwise_init
struct TokenizerScan(Copyable, ImplicitlyCopyable, Movable):
    """What one pass over `tokenizer.json` found."""

    var present: Bool
    var model: String
    var model_source: String
    var pre: String
    var vocab: Int
    var merges: Int
    var added: Int
    var highest_id: Int
    """The largest token id seen. Added tokens are commonly numbered past the
    end of the vocabulary, and a caller that sized an array by the vocabulary
    count would write past it on the first special token."""

    def token_count(self) -> Int:
        if self.highest_id + 1 > self.vocab:
            return self.highest_id + 1
        return self.vocab


def _reader_text_is(reader: Reader, name: StringSpan) -> Bool:
    """Whether the last string event was this text. `key_is` for a value."""
    var text = reader.text()
    if len(text) != name.byte_length():
        return False
    var p = name.unsafe_ptr()
    for i in range(len(text)):
        if text[i] != p.unsafe_load(i):
            return False
    return True


def _count_array(mut reader: Reader) raises -> Int:
    """Elements in the array the last event opened, without decoding any."""
    var count = 0
    while True:
        var e = reader.next()
        if e == EV_ERROR or e == EV_END:
            raise Error("tokenizer.json ends inside an array")
        if e == EV_ARRAY_END:
            return count
        _ = reader.skip_value()
        count += 1


def _object_type(mut reader: Reader) raises -> String:
    """The `type` of the object already opened, unrolled if it is a sequence.

    A pre-tokenizer of type Sequence is a list of pre-tokenizers, and reporting
    the word Sequence says nothing at all. What matters is what is in it, so a
    sequence reports as its members joined, and Qwen prints as Split+ByteLevel
    rather than as Sequence.
    """
    var kind = String("none")
    var inner = String("")
    while True:
        var member = reader.next()
        if member == EV_OBJECT_END:
            break
        if member == EV_ERROR or member == EV_END:
            raise Error("tokenizer.json ends inside an object")
        if member != EV_KEY:
            raise Error("tokenizer.json has a member with no key")
        var is_type = reader.key_is("type")
        var is_list = reader.key_is("pretokenizers")
        var value = reader.next()
        if value == EV_ERROR or value == EV_END:
            raise Error("tokenizer.json ends inside an object")
        if is_type and value == EV_STRING:
            kind = _text(reader.text())
        elif is_list and value == EV_ARRAY_BEGIN:
            while True:
                var element = reader.next()
                if element == EV_ARRAY_END:
                    break
                if element == EV_ERROR or element == EV_END:
                    raise Error("tokenizer.json ends inside a sequence")
                if element != EV_OBJECT_BEGIN:
                    _ = reader.skip_value()
                    continue
                if inner != "":
                    inner += "+"
                inner += _object_type(reader)
        else:
            _ = reader.skip_value()
    if inner != "":
        return inner
    return kind


def _scan_type(mut reader: Reader) raises -> String:
    """The `type` of the object the next value is, or none for a null."""
    var e = reader.next()
    if e != EV_OBJECT_BEGIN:
        _ = reader.skip_value()
        return String("none")
    return _object_type(reader)


def _scan_vocab(
    mut reader: Reader,
    mut scan: TokenizerScan,
    wanted: List[String],
    mut ids: List[Int],
) raises:
    """Count the vocabulary and pick up the ids of the tokens we asked about.

    A vocabulary is an object of token to id on a BPE or WordPiece tokenizer and
    an array of pairs on a Unigram one. Both are counted, and only the first can
    answer what id a token has, which is the reason the added token list is
    scanned as well.
    """
    var opened = reader.next()
    if opened == EV_ARRAY_BEGIN:
        scan.vocab = _count_array(reader)
        return
    if opened != EV_OBJECT_BEGIN:
        _ = reader.skip_value()
        return
    while True:
        var member = reader.next()
        if member == EV_OBJECT_END:
            return
        if member == EV_ERROR or member == EV_END:
            raise Error("tokenizer.json ends inside the vocabulary")
        if member != EV_KEY:
            raise Error("tokenizer.json vocabulary has a member with no key")
        var hit = -1
        for w in range(len(wanted)):
            if ids[w] < 0 and reader.key_is(wanted[w]):
                hit = w
                break
        var value = reader.next()
        if value == EV_ERROR or value == EV_END:
            raise Error("tokenizer.json ends inside the vocabulary")
        if value == EV_NUMBER:
            var id = reader.number.as_int()
            if hit >= 0:
                ids[hit] = id
            if id > scan.highest_id:
                scan.highest_id = id
        else:
            _ = reader.skip_value()
        scan.vocab += 1


def _scan_added(
    mut reader: Reader,
    mut scan: TokenizerScan,
    wanted: List[String],
    mut ids: List[Int],
) raises:
    """Walk `added_tokens`, which is where the special tokens live.

    Every entry is an id and its text, and the two arrive in whichever order the
    writer used, so both are held until the entry closes.
    """
    var opened = reader.next()
    if opened != EV_ARRAY_BEGIN:
        _ = reader.skip_value()
        return
    while True:
        var element = reader.next()
        if element == EV_ARRAY_END:
            return
        if element == EV_ERROR or element == EV_END:
            raise Error("tokenizer.json ends inside added_tokens")
        if element != EV_OBJECT_BEGIN:
            _ = reader.skip_value()
            continue
        scan.added += 1
        var id = -1
        var hit = -1
        while True:
            var member = reader.next()
            if member == EV_OBJECT_END:
                break
            if member == EV_ERROR or member == EV_END:
                raise Error("tokenizer.json ends inside an added token")
            if member != EV_KEY:
                raise Error(
                    "tokenizer.json added token has a member with no key"
                )
            var is_id = reader.key_is("id")
            var is_content = reader.key_is("content")
            var value = reader.next()
            if value == EV_ERROR or value == EV_END:
                raise Error("tokenizer.json ends inside an added token")
            if is_id and value == EV_NUMBER:
                id = reader.number.as_int()
            elif is_content and value == EV_STRING:
                for w in range(len(wanted)):
                    if _reader_text_is(reader, wanted[w]):
                        hit = w
                        break
            else:
                _ = reader.skip_value()
        if id > scan.highest_id:
            scan.highest_id = id
        if hit >= 0 and id >= 0 and ids[hit] < 0:
            ids[hit] = id


def scan_tokenizer(
    path: String, wanted: List[String], mut ids: List[Int], counter: Int
) raises -> TokenizerScan:
    """One streaming pass over `tokenizer.json`.

    Streaming rather than a tree because of what is in the file. A gemma
    vocabulary is 262144 entries and half a million merges in 33 MB of JSON, and
    holding that as a document costs fifty megabytes of nodes to answer four
    questions about counts and five about ids. The counts are a walk and the ids
    are a comparison against the keys as they go past, so the file is read once
    and nothing is kept.
    """
    var scan = TokenizerScan(
        present=False,
        model=String("none"),
        model_source=String("none"),
        pre=String("none"),
        vocab=0,
        merges=0,
        added=0,
        highest_id=-1,
    )
    if not exists(path):
        return scan
    scan.present = True

    var mapping = Mapping(path)
    var reader = Reader(counter, 65536)
    var body = Span[UInt8, MutAnyOrigin](
        unsafe_ptr=mapping.base(), length=mapping.length
    )
    reader.begin(body)
    try:
        if reader.next() != EV_OBJECT_BEGIN:
            raise Error("tokenizer.json is not an object")
        while True:
            var member = reader.next()
            if member == EV_OBJECT_END or member == EV_END:
                break
            if member == EV_ERROR:
                raise Error("tokenizer.json is not valid json")
            if member != EV_KEY:
                raise Error("tokenizer.json has a member with no key")
            if reader.key_is("pre_tokenizer"):
                scan.pre = _scan_type(reader)
            elif reader.key_is("added_tokens"):
                _scan_added(reader, scan, wanted, ids)
            elif reader.key_is("model"):
                var opened = reader.next()
                if opened != EV_OBJECT_BEGIN:
                    _ = reader.skip_value()
                    continue
                while True:
                    var field = reader.next()
                    if field == EV_OBJECT_END:
                        break
                    if field == EV_ERROR or field == EV_END:
                        raise Error("tokenizer.json ends inside the model")
                    if field != EV_KEY:
                        raise Error(
                            "tokenizer.json model has a member with no key"
                        )
                    if reader.key_is("type"):
                        var kind = reader.next()
                        if kind == EV_STRING:
                            scan.model_source = _text(reader.text())
                            scan.model = tokenizer_algorithm(scan.model_source)
                        else:
                            _ = reader.skip_value()
                    elif reader.key_is("vocab"):
                        _scan_vocab(reader, scan, wanted, ids)
                    elif reader.key_is("merges"):
                        var merges = reader.next()
                        if merges == EV_ARRAY_BEGIN:
                            scan.merges = _count_array(reader)
                        else:
                            _ = reader.skip_value()
                    else:
                        _ = reader.skip_next_value()
            else:
                _ = reader.skip_next_value()
    except e:
        mapping.close()
        raise e
    mapping.close()
    return scan


def _token_text(doc: Document, node: Int) -> String:
    """A special token, which is a string or an object holding one.

    `"eos_token": "<|im_end|>"` and `"eos_token": {"content": "<|im_end|>", ...}`
    are both in the wild and mean the same thing.
    """
    if node == NO_NODE:
        return String("")
    if doc.kind(node) == JS_STRING:
        return _text(doc.text(node))
    if doc.kind(node) == JS_OBJECT:
        var content = doc.get(node, "content")
        if content != NO_NODE and doc.kind(content) == JS_STRING:
            return _text(doc.text(content))
    return String("")


def read_repo_tokenizer(
    dir: String, config: JsonFile, node: Int, counter: Int
) raises -> TokenizerSpec:
    """The tokenizer, out of `tokenizer_config.json` and `tokenizer.json`.

    The special tokens are named as text in the config and numbered in the
    tokenizer, and the number is what everything downstream wants. So the config
    is read first, the five names it gives are carried into the pass over the
    tokenizer, and an id that came back unresolved falls back to the one
    `config.json` states. The fallback matters: a repository is allowed to name
    a token that is not in its own vocabulary, and that is a broken repository
    rather than a reason to report no end of sequence at all.
    """
    var tc = JsonFile(dir + "/tokenizer_config.json", counter)
    var gen = JsonFile(dir + "/generation_config.json", counter)

    var names = List[String]()
    names.append(_token_text(tc.doc, tc.doc.get(tc.root(), "bos_token")))
    names.append(_token_text(tc.doc, tc.doc.get(tc.root(), "eos_token")))
    names.append(_token_text(tc.doc, tc.doc.get(tc.root(), "pad_token")))
    names.append(_token_text(tc.doc, tc.doc.get(tc.root(), "unk_token")))
    names.append(_token_text(tc.doc, tc.doc.get(tc.root(), "sep_token")))
    names.append(_token_text(tc.doc, tc.doc.get(tc.root(), "cls_token")))

    var ids = List[Int]()
    for i in range(len(names)):
        # An empty name cannot match a token, and asking for it would match the
        # empty key some vocabularies really do contain.
        ids.append(-1 if names[i] != "" else -2)

    var scan = scan_tokenizer(dir + "/tokenizer.json", names, ids, counter)

    var bos = ids[0]
    var eos = ids[1]
    var pad = ids[2]
    var unk = ids[3]
    var sep = ids[4]
    var cls = ids[5]
    # An encoder names its first and last token cls and sep, and llama.cpp
    # writes those two ids into the bos and eos keys when it converts one. The
    # same substitution here is what makes a bert repository and the GGUF file
    # built from it report the same four numbers.
    if bos < 0:
        bos = cls
    if eos < 0:
        eos = sep
    if bos < 0:
        bos = _first_int(config.doc, config.doc.get(node, "bos_token_id"), -1)
    if bos < 0:
        bos = _first_int(gen.doc, gen.doc.get(gen.root(), "bos_token_id"), -1)
    if eos < 0:
        eos = _first_int(config.doc, config.doc.get(node, "eos_token_id"), -1)
    if eos < 0:
        eos = _first_int(gen.doc, gen.doc.get(gen.root(), "eos_token_id"), -1)
    if pad < 0:
        pad = _first_int(config.doc, config.doc.get(node, "pad_token_id"), -1)
    if unk < 0:
        unk = -1
    if sep < 0:
        sep = -1

    var template = tc.doc.get(tc.root(), "chat_template") != NO_NODE
    # Newer repositories keep the template in its own file, because a Jinja
    # template inside a JSON string is unreadable and unreviewable.
    if not template:
        template = exists(dir + "/chat_template.jinja")

    var spec = TokenizerSpec(
        model=scan.model,
        model_source=scan.model_source,
        pre=scan.pre,
        vocab_size=scan.token_count() if scan.present else 0,
        merge_count=scan.merges,
        bos=bos,
        eos=eos,
        eot=-1,
        pad=pad,
        unk=unk,
        sep=sep,
        add_bos=tc.doc.get_bool(tc.root(), "add_bos_token", False),
        add_eos=tc.doc.get_bool(tc.root(), "add_eos_token", False),
        has_chat_template=template,
        embedding_rows=config.doc.get_int(node, "vocab_size", 0),
    )
    tc.close()
    gen.close()
    return spec


def audit_repo(st: SafeTensors) -> Layout:
    """Add up the tensors and check they follow each other inside each shard.

    Shorter than the GGUF audit because the header already had to hold up: a
    byte range that ran past the end of the file, or that disagreed with the
    dtype and the shape, was refused when the file was opened. What is left to
    check is whether the tensors are laid out one after another with nothing
    between them, which is what every writer does and what a reader that wants
    to map a whole layer at once depends on.
    """
    var total = 0
    var narrow = 0
    var unknown = 0
    var packed = True
    var first_bad = -1
    var expected = 0
    var shard = -1
    for i in range(len(st.tensors)):
        var t = st.tensors[i]
        if t.shard != shard:
            shard = t.shard
            expected = 0
        if dtype_bytes(t.dtype) == 0:
            unknown += 1
            packed = False
            if first_bad < 0:
                first_bad = i
            break
        if t.begin != expected and packed:
            packed = False
            first_bad = i
        expected = t.end
        total += t.bytes()
        if dtype_narrow(t.dtype):
            narrow += t.bytes()
    return Layout(
        len(st.tensors), total, narrow, unknown, packed, True, first_bad
    )


def repo_capabilities(
    doc: Document, node: Int, root: Int, arch: Int, st: SafeTensors, dir: String
) -> Int:
    """What the repository says it can do.

    Same three questions the GGUF path asks, answered from different evidence. A
    sentence-transformers repository carries a `1_Pooling` directory holding the
    pooling configuration, which is the same statement `<arch>.pooling_type` is
    in a GGUF file: this model is meant to produce one vector rather than a
    token.
    """
    var caps = 0
    if is_causal(arch):
        caps |= CAP_TEXT
    if (
        arch == ARCH_BERT
        or arch == ARCH_NOMIC_BERT
        or exists(dir + "/1_Pooling")
        or exists(dir + "/modules.json")
    ):
        caps |= CAP_EMBEDDING
    if (
        doc.get(root, "vision_config") != NO_NODE
        or doc.get(node, "vision_config") != NO_NODE
        or st.prefixed("vision_tower.")
        or st.prefixed("vision_model.")
        or st.prefixed("visual.")
    ):
        caps |= CAP_VISION
    return caps


def _basename(path: String) -> String:
    var cut = path.rfind("/")
    if cut < 0:
        return path
    return String(path[byte = cut + 1 :])


def from_repo(dir: String, st: SafeTensors, counter: Int) raises -> ModelSpec:
    """Build the spec from an open set of shards and the JSON beside them."""
    var config = JsonFile(dir + "/config.json", counter)
    if not config.present:
        config.close()
        raise Error("no config.json in " + dir)

    # A multimodal repository keeps the language model geometry one level down
    # and the vision tower beside it. A text only one is flat. Both name the
    # same keys, so the only question is which object to read them from.
    var node = config.root()
    var text_config = config.doc.get(config.root(), "text_config")
    if text_config != NO_NODE and config.doc.kind(text_config) == JS_OBJECT:
        node = text_config

    var architecture = String("unknown")
    var classes = config.doc.get(config.root(), "architectures")
    if classes != NO_NODE and config.doc.size(classes) > 0:
        architecture = class_architecture(
            _text(config.doc.text(config.doc.at(classes, 0)))
        )
    else:
        var model_type = config.doc.get(config.root(), "model_type")
        if model_type != NO_NODE:
            architecture = _text(config.doc.text(model_type))
    var id = architecture_id(architecture)

    var declared = repo_capabilities(
        config.doc, node, config.root(), id, st, dir
    )
    var layout = audit_repo(st)
    var tokenizer = read_repo_tokenizer(dir, config, node, counter)
    var geometry = read_repo_geometry(config.doc, node)
    var file_type = st.dominant_dtype()

    var spec = ModelSpec(
        name=_basename(dir),
        source=String("safetensors"),
        architecture=architecture,
        arch_id=id,
        file_type=file_type,
        quantized=layout.quantized_bytes > 0,
        geometry=geometry,
        tokenizer=tokenizer,
        layout=layout,
        declared=declared,
        supported=declared & engine_capabilities(id),
    )
    config.close()
    return spec^


def run_repo(path: StringSpan) raises:
    """`molla spec` on a directory, or on one safetensors file inside one."""
    var counter = AllocCounter()
    var st = SafeTensors(path, counter.raw())
    var dir = st.dir
    try:
        var spec = from_repo(dir, st, counter.raw())
        report(spec, path)
    except e:
        st.close()
        counter.close()
        raise e

    print("")
    print("weights")
    print(
        "  files         "
        + String(len(st.shards))
        + (" shard" if len(st.shards) == 1 else " shards")
        + ", written by "
        + st.format
        + ", "
        + String(st.dtype_count())
        + (" dtype" if st.dtype_count() == 1 else " dtypes")
    )
    for i in range(len(st.shards)):
        print(
            "  "
            + st.shards[i].name
            + "  "
            + String(st.shards[i].tensors)
            + " tensors, header "
            + String(st.shards[i].header_bytes)
            + " bytes"
        )
    if st.sharded:
        var counted = st.bytes()
        print(
            "  index         "
            + String(st.mapped_names)
            + " names, "
            + String(st.missing)
            + " not in the shard the index named, "
            + String(st.unmapped)
            + " in a shard and not in the index"
        )
        if st.total_size == counted:
            print(
                "  total size    "
                + String(counted)
                + " bytes, which is what the index says"
            )
        else:
            print(
                "  total size    "
                + String(counted)
                + " bytes, and the index says "
                + String(st.total_size)
            )
    st.close()
    counter.close()


def _is_gguf(path: StringSpan) -> Bool:
    """Whether this file starts with the GGUF magic.

    The extension is a hint and this is the answer. A model downloaded without
    one is common, and reading four bytes is cheaper than being wrong about
    which reader to use.
    """
    try:
        var mapping = Mapping(path)
        var magic = False
        if mapping.length >= 4:
            magic = (
                mapping.base().unsafe_load(0) == 71
                and mapping.base().unsafe_load(1) == 71
                and mapping.base().unsafe_load(2) == 85
                and mapping.base().unsafe_load(3) == 70
            )
        mapping.close()
        return magic
    except:
        return False


def run_model_spec(path: StringSpan) raises:
    """Entry point for `molla spec`, whichever of the two a path holds."""
    var info = FileInfo()
    var found = stat_path(path, info)
    if found.is_err():
        raise Error("nothing at " + String(path))
    if info.is_dir():
        run_repo(path)
        return
    if _is_gguf(path):
        run_spec(path)
        return
    run_repo(path)
