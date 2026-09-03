"""Tests for the safetensors reader and the Hugging Face repository reader.

Same approach as the GGUF tests and for the same reason. Real repositories are
the other half and they live in `docs/validation/safetensors.md`, because a
two shard fp16 model is 7 GB and CI cannot download one on every push. What a
real repository cannot test is the failure paths: nothing on the Hub has a
header that runs past the end of the file or a shape that disagrees with its
byte count, and those are exactly the cases where a reader that trusts the file
reads at an offset the file chose.

The files here are built byte by byte with the header written as text, so a
mistake in the reader cannot cancel itself out against a mistake in a writer we
also wrote.
"""

from std.ffi import external_call

from harness import Suite

from molla.model.repo import class_architecture, from_repo
from molla.model.safetensors import (
    SafeTensors,
    dtype_bytes,
    dtype_narrow,
)
from molla.model.spec import (
    ARCH_BERT,
    ARCH_GEMMA3,
    ARCH_QWEN2,
    CAP_EMBEDDING,
    CAP_TEXT,
    CAP_VISION,
    tokenizer_algorithm,
)
from molla.sys.file import MODE_755, mkdir, rmdir, unlink
from molla.sys.mem import AllocCounter


def _write(path: String, bytes: List[UInt8]) raises:
    with open(path, "w") as f:
        f.write_bytes(Span(bytes))


def _write_text(path: String, text: StringSpan) raises:
    var bytes = List[UInt8]()
    for i in range(text.byte_length()):
        bytes.append(text.unsafe_ptr().unsafe_load(i))
    _write(path, bytes)


def _remove(path: String):
    _ = unlink(path)


def _pid() -> Int:
    return Int(external_call["getpid", Int32]())


def _temp_dir(name: StringSpan) raises -> String:
    """A directory nothing else will collide with.

    CI runs the suite on three platforms at once and a developer can run it
    twice at once, so the pid goes in the name for the same reason every
    listener in test_net binds to port 0.
    """
    var path = String("/tmp/molla_st_") + String(name) + "_" + String(_pid())
    _ = mkdir(path, MODE_755)
    return path^


def _st_bytes(header: StringSpan, data: Int) -> List[UInt8]:
    """A safetensors file: eight bytes of length, the header, then the data."""
    var out = List[UInt8]()
    var n = header.byte_length()
    for i in range(8):
        out.append(UInt8((n >> (8 * i)) & 0xFF))
    for i in range(n):
        out.append(header.unsafe_ptr().unsafe_load(i))
    for i in range(data):
        out.append(UInt8(i & 0xFF))
    return out^


def _write_st(path: String, header: StringSpan, data: Int) raises:
    _write(path, _st_bytes(header, data))


comptime PAIR = '{"__metadata__":{"format":"pt"},"a.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b.bias":{"dtype":"BF16","shape":[4],"data_offsets":[24,32]}}'
"""Two tensors, thirty two bytes of data, one of each of the two widths. F32
holds more of the bytes than BF16 does, which is what makes it the dominant
dtype even though there are as many tensors of each."""


def _opens(path: String) -> Bool:
    """True when the file or directory reads as a model. The bad ones want
    False, and the reason a bad one is bad is in the message, which the report
    prints and this does not check."""
    var counter = AllocCounter()
    var ok = True
    try:
        var st = SafeTensors(path, counter.raw())
        st.close()
    except:
        ok = False
    counter.close()
    return ok


def _bad(
    mut suite: Suite, dir: String, name: String, header: StringSpan
) raises:
    """Write a file with one thing wrong with it and check it is refused."""
    var path = dir + "/bad.safetensors"
    _write_st(path, header, 32)
    suite.check(not _opens(path), name)
    _remove(path)


def run(mut suite: Suite) raises:
    suite.group("safetensors dtypes")

    suite.check(dtype_bytes("F32") == 4, "F32 is four bytes")
    suite.check(dtype_bytes("BF16") == 2, "BF16 is two bytes")
    suite.check(dtype_bytes("F8_E4M3") == 1, "an eight bit float is one byte")
    suite.check(dtype_bytes("BOOL") == 1, "BOOL is one byte")
    suite.check(dtype_bytes("F4") == 0, "a dtype we do not know is zero")
    suite.check(dtype_narrow("F8_E5M2"), "an eight bit float is compressed")
    # The line that decides whether every fp16 repository on the Hub reports as
    # quantized. It is the width the weights were saved at, not a compression.
    suite.check(not dtype_narrow("F16"), "half precision is not compressed")
    suite.check(not dtype_narrow("F32"), "full precision is not compressed")

    suite.group("safetensors reader")

    var dir = _temp_dir("single")
    var single = dir + "/model.safetensors"
    _write_st(single, PAIR, 32)

    var counter = AllocCounter()
    var st = SafeTensors(single, counter.raw())
    suite.check(len(st.shards) == 1, "one file is one shard")
    suite.check(len(st.tensors) == 2, "both tensors are read")
    suite.check(st.format == "pt", "the format comes out of __metadata__")
    suite.check(not st.sharded, "a single file is not sharded")
    suite.check(st.total_size == -1, "no index means no declared total")
    suite.check(st.bytes() == 32, "the weight bytes add up")
    suite.check(st.shards[0].tensors == 2, "the shard counts its own tensors")
    suite.check(
        st.shards[0].header_bytes == PAIR.byte_length(),
        "the header length is the length of the header",
    )
    suite.check(
        st.shards[0].data_start == 8 + PAIR.byte_length(),
        "the data starts after the header",
    )
    suite.check(st.find("b.bias") == 1, "a tensor is found by name")
    suite.check(st.find("c.bias") < 0, "a name that is not there is not found")
    suite.check(st.tensors[0].dtype == "F32", "the dtype is read")
    suite.check(
        st.tensors[0].n_dims == 2, "the rank is the length of the shape"
    )
    suite.check(st.tensors[0].shape_text() == "2, 3", "the shape is read")
    suite.check(st.tensors[0].elements() == 6, "the elements multiply out")
    suite.check(st.tensors[1].bytes() == 8, "the byte range is the size")
    suite.check(st.tensors[1].shard == 0, "one shard is shard zero")
    suite.check(st.prefixed("a."), "a prefix that is there")
    suite.check(not st.prefixed("vision_tower."), "a prefix that is not")
    suite.check(st.dtype_count() == 2, "two dtypes in the file")
    suite.check(
        st.dominant_dtype() == "F32",
        "the dominant dtype is by bytes, not count",
    )
    st.close()
    counter.close()

    # The same file, named by its directory rather than by its own name.
    suite.check(_opens(dir), "a directory with one model.safetensors opens")

    suite.group("safetensors refusals")

    # Every one of these is a file that no writer produced. The point of the
    # group is that each is refused before anything reads at the offset it
    # gives, because the alternative to a message here is a segfault later.
    _bad(
        suite,
        dir,
        "a header that is not json",
        "not json at all",
    )
    _bad(suite, dir, "a header that is not an object", "[1,2,3]")
    _bad(
        suite,
        dir,
        "an entry that is not a tensor",
        '{"a":4}',
    )
    _bad(
        suite,
        dir,
        "an entry with no dtype",
        '{"a":{"shape":[2],"data_offsets":[0,8]}}',
    )
    _bad(
        suite,
        dir,
        "an entry with no shape",
        '{"a":{"dtype":"F32","data_offsets":[0,8]}}',
    )
    _bad(
        suite,
        dir,
        "an entry with no offsets",
        '{"a":{"dtype":"F32","shape":[2]}}',
    )
    _bad(
        suite,
        dir,
        "one offset rather than two",
        '{"a":{"dtype":"F32","shape":[2],"data_offsets":[0]}}',
    )
    _bad(
        suite,
        dir,
        "five dimensions",
        '{"a":{"dtype":"F32","shape":[1,1,1,1,1],"data_offsets":[0,4]}}',
    )
    _bad(
        suite,
        dir,
        "a negative dimension",
        '{"a":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}}',
    )
    _bad(
        suite,
        dir,
        "offsets the wrong way round",
        '{"a":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}',
    )
    # 32 bytes of data in the file and a tensor that claims to end at 64. This
    # is the one that matters most: the range is what a loader would map.
    _bad(
        suite,
        dir,
        "a tensor that runs past the end of the file",
        '{"a":{"dtype":"F32","shape":[16],"data_offsets":[0,64]}}',
    )
    # The size check. Eight F32 elements are thirty two bytes and the header
    # says sixteen, so one of the two is a lie and the file is refused.
    _bad(
        suite,
        dir,
        "a shape that disagrees with the byte count",
        '{"a":{"dtype":"F32","shape":[8],"data_offsets":[0,16]}}',
    )

    # A length prefix with the top bit set, which is the shape of a hostile
    # file: every arithmetic step after it overflows.
    var huge = _st_bytes(PAIR, 32)
    huge[7] = 0x80
    var huge_path = dir + "/bad.safetensors"
    _write(huge_path, huge)
    suite.check(not _opens(huge_path), "an absurd header length is rejected")
    _remove(huge_path)

    # Four bytes is not even a length prefix.
    var stub = List[UInt8]()
    for i in range(4):
        stub.append(UInt8(i))
    _write(huge_path, stub)
    suite.check(not _opens(huge_path), "a file shorter than a header prefix")
    _remove(huge_path)

    # A header longer than the file it is in. The prefix is the only thing that
    # says how long the header is, so it has to be checked against the mapping.
    var short = _st_bytes(PAIR, 0)
    for _ in range(20):
        _ = short.pop()
    _write(huge_path, short)
    suite.check(not _opens(huge_path), "a header that runs past the end")
    _remove(huge_path)

    suite.check(
        not _opens(dir + "/nothing.safetensors"), "a file that is not there"
    )

    _remove(single)
    suite.check(not _opens(dir), "a directory with no safetensors file")

    # Two files and no index is a repository nothing can order, which is a
    # different problem from a repository with nothing in it.
    _write_st(dir + "/one.safetensors", PAIR, 32)
    suite.check(_opens(dir), "one oddly named file is still a model")
    _write_st(dir + "/two.safetensors", PAIR, 32)
    suite.check(not _opens(dir), "two files and no index is refused")
    _remove(dir + "/one.safetensors")
    _remove(dir + "/two.safetensors")
    _ = rmdir(dir)

    suite.group("safetensors shards")

    var shard_dir = _temp_dir("shards")
    var first = (
        '{"a.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}'
    )
    var second = (
        '{"b.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}'
    )
    _write_st(shard_dir + "/model-00001-of-00002.safetensors", first, 16)
    _write_st(shard_dir + "/model-00002-of-00002.safetensors", second, 16)
    # The map names the second shard first, which is what a real index does
    # when it starts at lm_head. The shards are still opened in name order.
    _write_text(
        shard_dir + "/model.safetensors.index.json",
        (
            '{"metadata":{"total_size":32},"weight_map":'
            '{"b.weight":"model-00002-of-00002.safetensors",'
            '"a.weight":"model-00001-of-00002.safetensors"}}'
        ),
    )

    var shard_counter = AllocCounter()
    var sharded = SafeTensors(shard_dir, shard_counter.raw())
    suite.check(sharded.sharded, "an index means sharded")
    suite.check(len(sharded.shards) == 2, "both shards are opened")
    suite.check(len(sharded.tensors) == 2, "the tensor list covers the model")
    suite.check(sharded.mapped_names == 2, "the weight map is counted")
    suite.check(sharded.missing == 0, "nothing the index promised is missing")
    suite.check(sharded.unmapped == 0, "nothing in a shard is unnamed")
    suite.check(sharded.total_size == 32, "the declared total is read")
    suite.check(
        sharded.bytes() == sharded.total_size,
        "the tensors add up to what the index says",
    )
    suite.check(
        sharded.tensors[1].shard == 1,
        "the second tensor is in the second shard",
    )
    suite.check(
        sharded.shards[0].name == "model-00001-of-00002.safetensors"
        and sharded.shards[1].name == "model-00002-of-00002.safetensors",
        (
            "shards are opened in name order, not in the order the index names"
            " them"
        ),
    )
    # Offsets are per shard, not global, which is the one thing about a sharded
    # repository that is easy to get wrong.
    suite.check(sharded.tensors[1].begin == 0, "offsets restart in each shard")
    sharded.close()
    shard_counter.close()

    # An index that promises a tensor no shard has, and a shard holding a
    # tensor the index never mentions. Both directions are counted because they
    # fail differently: the first is a load that stops at a missing weight and
    # the second is a weight nothing will look for.
    _write_text(
        shard_dir + "/model.safetensors.index.json",
        (
            '{"weight_map":'
            '{"a.weight":"model-00001-of-00002.safetensors",'
            '"c.weight":"model-00002-of-00002.safetensors"}}'
        ),
    )
    var odd_counter = AllocCounter()
    var odd = SafeTensors(shard_dir, odd_counter.raw())
    suite.check(odd.missing == 1, "a name with no tensor behind it is counted")
    suite.check(
        odd.unmapped == 1, "a tensor with no name in the index is counted"
    )
    suite.check(
        odd.total_size == -1, "an index with no metadata declares nothing"
    )
    odd.close()
    odd_counter.close()

    _write_text(shard_dir + "/model.safetensors.index.json", '{"metadata":{}}')
    suite.check(not _opens(shard_dir), "an index with no weight_map is refused")

    _remove(shard_dir + "/model.safetensors.index.json")
    _remove(shard_dir + "/model-00001-of-00002.safetensors")
    _remove(shard_dir + "/model-00002-of-00002.safetensors")
    _ = rmdir(shard_dir)

    suite.group("repository names")

    suite.check(
        class_architecture("Qwen2ForCausalLM") == "qwen2", "a class name maps"
    )
    suite.check(
        class_architecture("Gemma3ForConditionalGeneration") == "gemma3",
        "a multimodal class maps to its family",
    )
    suite.check(
        class_architecture("BertForMaskedLM") == "bert",
        "the heads on one family share a name",
    )
    # A class nobody has checked comes back as itself, so it reports as
    # unrecognised rather than being filed under llama for ending in the same
    # four words.
    suite.check(
        class_architecture("MambaForCausalLM") == "MambaForCausalLM",
        "an unknown class is not guessed at",
    )

    suite.check(tokenizer_algorithm("gpt2") == "bpe", "gguf spells BPE gpt2")
    suite.check(
        tokenizer_algorithm("BPE") == "bpe", "tokenizer.json spells it BPE"
    )
    suite.check(
        tokenizer_algorithm("llama") == "spm", "gguf spells unigram llama"
    )
    suite.check(
        tokenizer_algorithm("Unigram") == "spm",
        "tokenizer.json spells it Unigram",
    )
    suite.check(
        tokenizer_algorithm("bert") == "wpm", "gguf spells wordpiece bert"
    )
    suite.check(
        tokenizer_algorithm("WordPiece") == "wpm",
        "tokenizer.json spells it WordPiece",
    )
    suite.check(
        tokenizer_algorithm("rwkv") == "rwkv", "anything else is left alone"
    )

    suite.group("repository reader")

    var repo = _temp_dir("repo")
    _write_st(repo + "/model.safetensors", PAIR, 32)
    _write_text(
        repo + "/config.json",
        (
            '{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2",'
            '"hidden_size":8,"num_hidden_layers":2,"num_attention_heads":4,'
            '"num_key_value_heads":2,"intermediate_size":16,'
            '"max_position_embeddings":128,"rope_theta":10000.0,'
            '"rms_norm_eps":1e-06,"vocab_size":10,"bos_token_id":1,'
            '"eos_token_id":[2,3]}'
        ),
    )
    _write_text(
        repo + "/tokenizer.json",
        (
            '{"added_tokens":[{"id":8,"content":"<pad>","special":true}],'
            '"pre_tokenizer":{"type":"Sequence","pretokenizers":'
            '[{"type":"Split"},{"type":"ByteLevel"}]},'
            '"model":{"type":"BPE","vocab":{"a":0,"b":1,"<s>":2,"</s>":3},'
            '"merges":["a b","b a"]}}'
        ),
    )
    _write_text(
        repo + "/tokenizer_config.json",
        (
            '{"bos_token":"<s>","eos_token":{"content":"</s>"},'
            '"pad_token":"<pad>","add_bos_token":true,"chat_template":"x"}'
        ),
    )

    var repo_counter = AllocCounter()
    var weights = SafeTensors(repo, repo_counter.raw())
    var spec = from_repo(repo, weights, repo_counter.raw())

    suite.check(
        spec.source == "safetensors", "the spec says which reader read it"
    )
    suite.check(
        spec.name == "molla_st_repo_" + String(_pid()),
        "the name is the directory the model is in",
    )
    suite.check(
        spec.architecture == "qwen2", "the architecture comes from the class"
    )
    suite.check(spec.arch_id == ARCH_QWEN2, "and it is recognised")
    suite.check(spec.file_type == "F32", "the file type is the dominant dtype")
    suite.check(not spec.quantized, "half precision weights are not quantized")

    var geo = spec.geometry
    suite.check(geo.block_count == 2, "layers")
    suite.check(geo.context_length == 128, "context")
    suite.check(geo.embedding_length == 8, "embedding")
    suite.check(geo.feed_forward_length == 16, "feed forward")
    suite.check(geo.head_count == 4, "heads")
    suite.check(geo.head_count_kv == 2, "key value heads")
    suite.check(geo.kv_stated, "the file states its key value heads")
    suite.check(not geo.head_dim_stated, "and it does not state a head dim")
    suite.check(
        geo.key_length == 2, "so the head dim is the embedding over the heads"
    )
    suite.check(geo.rope_freq_base == 10000.0, "the rope base")
    suite.check(geo.rope_dimension_count == 2, "the whole head rotates")
    suite.check(geo.rope_scaling == "unstated", "no rope scaling object")
    suite.check(geo.epsilon > 0.0, "the epsilon is read")

    var tok = spec.tokenizer
    suite.check(tok.model == "bpe", "the tokenizer algorithm is normalised")
    suite.check(
        tok.model_source == "BPE", "and what the file called it is kept"
    )
    # A pre-tokenizer of type Sequence is a list of pre-tokenizers, and the word
    # Sequence on its own says nothing about how text gets split.
    suite.check(tok.pre == "Split+ByteLevel", "a sequence reports its members")
    suite.check(tok.merge_count == 2, "the merges are counted")
    # Four entries in the vocabulary and an added token numbered eight, so the
    # ids run to nine even though the vocabulary has four things in it.
    suite.check(tok.vocab_size == 9, "the token count covers the added tokens")
    suite.check(
        tok.embedding_rows == 10, "the embedding rows come from config.json"
    )
    suite.check(tok.bos == 2, "bos is resolved out of the vocabulary")
    suite.check(tok.eos == 3, "eos is resolved through the object form")
    suite.check(tok.pad == 8, "pad is resolved out of the added tokens")
    suite.check(tok.unk == -1, "a token the file does not name is -1")
    suite.check(tok.add_bos, "add_bos_token is read")
    suite.check(not tok.add_eos, "and a missing one is false")
    suite.check(tok.has_chat_template, "the chat template is found")

    suite.check(spec.layout.tensors == 2, "the audit counts the tensors")
    suite.check(spec.layout.bytes == 32, "and adds up the bytes")
    suite.check(spec.layout.packed, "the tensors follow each other")
    suite.check(spec.layout.fits, "and they are inside the file")
    suite.check(spec.layout.unknown_types == 0, "every dtype is known")
    suite.check(spec.declared == CAP_TEXT, "a causal model declares text")
    suite.check(spec.supported == 0, "and this build supports none of it")
    weights.close()
    repo_counter.close()

    # The template moved into its own file, which is what the newer repositories
    # do because a Jinja template inside a JSON string is unreviewable.
    _write_text(repo + "/tokenizer_config.json", "{}")
    _write_text(repo + "/chat_template.jinja", "hello")
    var plain_counter = AllocCounter()
    var plain_weights = SafeTensors(repo, plain_counter.raw())
    var plain = from_repo(repo, plain_weights, plain_counter.raw())
    suite.check(
        plain.tokenizer.has_chat_template, "a chat_template.jinja counts as one"
    )
    # Nothing names a special token now, so the ids fall back to config.json,
    # where eos is a list and the first entry is the one the model emits.
    suite.check(plain.tokenizer.bos == 1, "bos falls back to config.json")
    suite.check(
        plain.tokenizer.eos == 2, "and a list of eos ids takes the first"
    )
    plain_weights.close()
    plain_counter.close()
    _remove(repo + "/chat_template.jinja")

    _remove(repo + "/config.json")
    var no_config = AllocCounter()
    var orphan = SafeTensors(repo, no_config.raw())
    var refused = False
    try:
        var never = from_repo(repo, orphan, no_config.raw())
        _ = never.name
    except:
        refused = True
    suite.check(refused, "a directory with no config.json is not a model")
    orphan.close()
    no_config.close()

    _remove(repo + "/model.safetensors")
    _remove(repo + "/tokenizer.json")
    _remove(repo + "/tokenizer_config.json")
    _ = rmdir(repo)

    suite.group("repository shapes")

    # A multimodal repository keeps the language model geometry one level down
    # and the vision tower beside it. Reading the top level object would give
    # the geometry of neither.
    var multi = _temp_dir("multi")
    _write_st(multi + "/model.safetensors", PAIR, 32)
    _write_text(
        multi + "/config.json",
        (
            '{"architectures":["Gemma3ForConditionalGeneration"],'
            '"vision_config":{"hidden_size":1152},'
            '"text_config":{"hidden_size":640,"num_hidden_layers":18,'
            '"num_attention_heads":4,"head_dim":256,"vocab_size":262144,'
            '"rope_scaling":{"rope_type":"linear","factor":8.0}}}'
        ),
    )
    var multi_counter = AllocCounter()
    var multi_weights = SafeTensors(multi, multi_counter.raw())
    var multi_spec = from_repo(multi, multi_weights, multi_counter.raw())
    suite.check(multi_spec.arch_id == ARCH_GEMMA3, "the class names the family")
    suite.check(
        multi_spec.geometry.embedding_length == 640,
        "the geometry comes from text_config",
    )
    suite.check(
        multi_spec.geometry.head_dim_stated, "a stated head dim is stated"
    )
    suite.check(multi_spec.geometry.key_length == 256, "and it is used")
    suite.check(
        multi_spec.geometry.rope_scaling == "linear", "rope scaling is read"
    )
    suite.check(
        multi_spec.geometry.rope_scale_factor == 8.0, "and so is the factor"
    )
    suite.check(
        multi_spec.declared == CAP_TEXT | CAP_VISION,
        "a vision config means the model takes images",
    )
    suite.check(
        multi_spec.tokenizer.model == "none",
        "a repository with no tokenizer.json says so",
    )
    suite.check(multi_spec.tokenizer.vocab_size == 0, "and counts no tokens")
    multi_weights.close()
    multi_counter.close()
    _remove(multi + "/config.json")
    _remove(multi + "/model.safetensors")
    _ = rmdir(multi)

    # An encoder names its first and last token cls and sep. llama.cpp writes
    # those two ids into the bos and eos keys when it converts one, so doing the
    # same here is what makes the two readers agree on the same model.
    var bert = _temp_dir("bert")
    _write_st(bert + "/model.safetensors", PAIR, 32)
    _write_text(
        bert + "/config.json",
        (
            '{"architectures":["BertModel"],"hidden_size":384,'
            '"num_hidden_layers":12,"num_attention_heads":12,'
            '"layer_norm_eps":1e-12,"vocab_size":30522}'
        ),
    )
    _write_text(
        bert + "/tokenizer.json",
        (
            '{"added_tokens":[],"pre_tokenizer":{"type":"BertPreTokenizer"},'
            '"model":{"type":"WordPiece","vocab":{"[PAD]":0,"[UNK]":1,'
            '"[CLS]":2,"[SEP]":3}}}'
        ),
    )
    _write_text(
        bert + "/tokenizer_config.json",
        (
            '{"cls_token":"[CLS]","sep_token":"[SEP]","unk_token":"[UNK]",'
            '"pad_token":"[PAD]"}'
        ),
    )
    var bert_counter = AllocCounter()
    var bert_weights = SafeTensors(bert, bert_counter.raw())
    var bert_spec = from_repo(bert, bert_weights, bert_counter.raw())
    suite.check(bert_spec.arch_id == ARCH_BERT, "an encoder is recognised")
    suite.check(
        bert_spec.declared == CAP_EMBEDDING, "and declares embedding only"
    )
    suite.check(bert_spec.tokenizer.model == "wpm", "wordpiece is normalised")
    suite.check(bert_spec.tokenizer.bos == 2, "cls stands in for bos")
    suite.check(bert_spec.tokenizer.eos == 3, "and sep for eos")
    suite.check(bert_spec.tokenizer.unk == 1, "unk is resolved")
    suite.check(bert_spec.tokenizer.pad == 0, "and so is pad")
    suite.check(bert_spec.tokenizer.merge_count == 0, "wordpiece has no merges")
    suite.check(
        bert_spec.geometry.epsilon > 0.0,
        "an encoder states layer_norm_eps instead",
    )
    bert_weights.close()
    bert_counter.close()

    # sentence-transformers ships a pooling configuration in its own directory,
    # which is the same statement a pooling type is in a GGUF file.
    _write_text(bert + "/config.json", '{"architectures":["MambaForCausalLM"]}')
    _ = mkdir(bert + "/1_Pooling", MODE_755)
    var pool_counter = AllocCounter()
    var pool_weights = SafeTensors(bert, pool_counter.raw())
    var pooled = from_repo(bert, pool_weights, pool_counter.raw())
    suite.check(
        pooled.declared == CAP_EMBEDDING, "a pooling directory means embedding"
    )
    suite.check(
        pooled.architecture == "MambaForCausalLM", "an unknown class is kept"
    )
    pool_weights.close()
    pool_counter.close()
    _ = rmdir(bert + "/1_Pooling")

    _remove(bert + "/config.json")
    _remove(bert + "/tokenizer.json")
    _remove(bert + "/tokenizer_config.json")
    _remove(bert + "/model.safetensors")
    _ = rmdir(bert)
