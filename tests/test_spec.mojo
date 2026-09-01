"""Tests for the mapping from a GGUF file to a ModelSpec.

The reader has its own tests and this is the layer above it, so what is checked
here is the deciding rather than the parsing: which default is used when a key
is absent, how many bytes a ggml type takes, and whether a tensor directory adds
up.

Two of those are worth more than the rest. The block geometry table is a set of
numbers copied from ggml's own struct definitions, and a wrong row in it does
not fail, it hands back a byte count that overlaps the next tensor. And the
defaults are where a spec layer is most likely to be quietly wrong, because a
model that states every key it has an opinion about looks identical to one that
states none of them until something reads the flag.

Real models are the other half of this and they live in
`docs/validation/spec.md`, where the geometry is compared field by field against
what llama.cpp loads from the same four files. A 468 MB model is not something
to put in a git repository.
"""

from std.ffi import c_int, external_call

from harness import Suite

from molla.model.gguf import Gguf
from molla.model.spec import (
    ARCH_BERT,
    ARCH_LLAMA,
    ARCH_UNKNOWN,
    CAP_EMBEDDING,
    CAP_TEXT,
    CAP_VISION,
    architecture_id,
    audit,
    capability_names,
    encoding_of,
    engine_capabilities,
    from_gguf,
    is_causal,
    tensor_bytes,
)

from test_gguf import Builder

comptime T_U32 = 4
comptime T_F32 = 6
comptime T_BOOL = 7
comptime T_STRING = 8
comptime T_ARRAY = 9

comptime GGML_F32 = 0
comptime GGML_F16 = 1
comptime GGML_Q8_0 = 8


def _temp_path(name: StringSpan) -> String:
    var pid = Int(external_call["getpid", Int32]())
    return (
        String("/tmp/molla_spec_") + String(name) + "_" + String(pid) + ".gguf"
    )


def _write(path: StringSpan, bytes: List[UInt8]) raises:
    with open(String(path), "w") as f:
        f.write_bytes(Span(bytes))


def _remove(path: StringSpan):
    var buf = List[UInt8]()
    for i in range(path.byte_length()):
        buf.append(path.unsafe_ptr().unsafe_load(i))
    buf.append(0)
    _ = external_call["unlink", c_int](buf.unsafe_ptr())


def _llama_file(
    second_offset: Int, data_bytes: Int, vision: Bool
) -> List[UInt8]:
    """A small file shaped like a grouped query llama, with two tensors.

    The first is Q8_0 over 128 elements, which is four blocks of 34 bytes and
    so 136 bytes, and 136 rounds up to 160 at the default alignment of 32. So
    the second tensor belongs at 160, and `second_offset` is how a caller puts
    it somewhere it does not belong.
    """
    var b = Builder()
    b.raw("GGUF")
    b.u32(3)
    b.u64(UInt64(3) if vision else UInt64(2))
    b.u64(17)

    b.kv_header("general.architecture", T_STRING)
    b.gstring("llama")
    b.kv_header("general.name", T_STRING)
    b.gstring("tiny llama")
    b.kv_header("general.file_type", T_U32)
    b.u32(7)

    b.kv_header("llama.block_count", T_U32)
    b.u32(30)
    b.kv_header("llama.context_length", T_U32)
    b.u32(8192)
    b.kv_header("llama.embedding_length", T_U32)
    b.u32(576)
    b.kv_header("llama.feed_forward_length", T_U32)
    b.u32(1536)
    b.kv_header("llama.attention.head_count", T_U32)
    b.u32(9)
    b.kv_header("llama.attention.head_count_kv", T_U32)
    b.u32(3)
    b.kv_header("llama.attention.layer_norm_rms_epsilon", T_F32)
    b.f32(0.00001)
    b.kv_header("llama.rope.freq_base", T_F32)
    b.f32(100000.0)

    b.kv_header("tokenizer.ggml.model", T_STRING)
    b.gstring("gpt2")
    b.kv_header("tokenizer.ggml.tokens", T_ARRAY)
    b.u32(UInt64(T_STRING))
    b.u64(3)
    b.gstring("a")
    b.gstring("b")
    b.gstring("c")
    b.kv_header("tokenizer.ggml.bos_token_id", T_U32)
    b.u32(1)
    b.kv_header("tokenizer.ggml.eos_token_id", T_U32)
    b.u32(2)
    b.kv_header("tokenizer.ggml.add_bos_token", T_BOOL)
    b.u8(1)
    b.kv_header("tokenizer.chat_template", T_STRING)
    b.gstring("{{ messages }}")

    b.gstring("token_embd.weight")
    b.u32(2)
    b.u64(64)
    b.u64(2)
    b.u32(UInt64(GGML_Q8_0))
    b.u64(0)

    b.gstring("output_norm.weight")
    b.u32(1)
    b.u64(4)
    b.u32(UInt64(GGML_F32))
    b.u64(UInt64(second_offset))

    if vision:
        b.gstring("mm.0.weight")
        b.u32(1)
        b.u64(4)
        b.u32(UInt64(GGML_F32))
        b.u64(UInt64(second_offset + 32))

    b.pad_to(32)
    for i in range(data_bytes):
        b.u8(UInt8(i & 0xFF))
    return b^.finish()


def _ragged_file() -> List[UInt8]:
    """A file whose quantized tensor is not a whole number of blocks.

    Twenty weights is not something Q8_0 can hold, since a block is thirty two
    of them, so this is not a file ggml could have written. It is a file
    somebody could hand molla, and the size of that tensor has no answer.
    """
    var b = Builder()
    b.raw("GGUF")
    b.u32(3)
    b.u64(1)
    b.u64(1)

    b.kv_header("general.architecture", T_STRING)
    b.gstring("llama")

    b.gstring("token_embd.weight")
    b.u32(1)
    b.u64(20)
    b.u32(UInt64(GGML_Q8_0))
    b.u64(0)

    b.pad_to(32)
    for i in range(64):
        b.u8(UInt8(i & 0xFF))
    return b^.finish()


def _bert_file() -> List[UInt8]:
    """A file that states a pooling type and nothing about grouped query.

    Everything the llama file spells out about head counts and head dimensions
    is left out here, which is what makes it the test of the defaults.
    """
    var b = Builder()
    b.raw("GGUF")
    b.u32(3)
    b.u64(1)
    b.u64(10)

    b.kv_header("general.architecture", T_STRING)
    b.gstring("bert")
    b.kv_header("bert.block_count", T_U32)
    b.u32(12)
    b.kv_header("bert.context_length", T_U32)
    b.u32(512)
    b.kv_header("bert.embedding_length", T_U32)
    b.u32(384)
    b.kv_header("bert.feed_forward_length", T_U32)
    b.u32(1536)
    b.kv_header("bert.attention.head_count", T_U32)
    b.u32(12)
    b.kv_header("bert.attention.layer_norm_epsilon", T_F32)
    b.f32(0.000000000001)
    b.kv_header("bert.pooling_type", T_U32)
    b.u32(1)
    b.kv_header("tokenizer.ggml.model", T_STRING)
    b.gstring("bert")
    b.kv_header("tokenizer.ggml.tokens", T_ARRAY)
    b.u32(UInt64(T_STRING))
    b.u64(2)
    b.gstring("[CLS]")
    b.gstring("[SEP]")

    b.gstring("token_embd.weight")
    b.u32(2)
    b.u64(64)
    b.u64(2)
    b.u32(UInt64(GGML_F16))
    b.u64(0)

    b.pad_to(32)
    for i in range(256):
        b.u8(UInt8(i & 0xFF))
    return b^.finish()


def _check_encodings(mut suite: Suite) raises:
    suite.group("model.spec encodings")

    var f32 = encoding_of(GGML_F32)
    suite.check(
        f32.block == 1 and f32.block_bytes == 4 and not f32.quantized,
        "an F32 element is its own block of four bytes",
    )

    var q8 = encoding_of(GGML_Q8_0)
    suite.check(
        q8.block == 32 and q8.block_bytes == 34 and q8.quantized,
        "Q8_0 is thirty two weights in thirty four bytes",
    )
    suite.check(
        q8.bits_per_weight() == 8.5,
        "which is eight and a half bits a weight, the scale being the half",
    )

    var q4k = encoding_of(12)
    suite.check(
        q4k.block == 256 and q4k.block_bytes == 144,
        "Q4_K is a superblock of 256 in 144 bytes",
    )

    var unknown = encoding_of(200)
    suite.check(
        not unknown.known and unknown.block == 0,
        (
            "a type number this build has no geometry for says so rather than"
            " guessing a size"
        ),
    )


def _check_sizes(mut suite: Suite) raises:
    suite.group("model.spec tensor sizes")

    var path = _temp_path("llama")
    _write(path, _llama_file(160, 192, False))
    var g = Gguf(path)

    suite.check(
        tensor_bytes(g.tensors[0]) == 136,
        "128 Q8_0 weights are four blocks and 136 bytes",
    )
    suite.check(
        tensor_bytes(g.tensors[1]) == 16, "four F32 weights are sixteen bytes"
    )

    var lay = audit(g)
    suite.check(lay.packed, "both tensors sit where the running total says")
    suite.check(lay.fits, "and the data section is inside the file")
    suite.check(lay.bytes == 176, "which is 176 bytes of weights in total")
    suite.check(
        lay.quantized_bytes == 136,
        "of which the quantized tensor is 136 and the norm is not counted",
    )
    suite.check(lay.unknown_types == 0, "and every type had a known size")
    g.close()
    _remove(path)

    var ragged = _temp_path("ragged")
    _write(ragged, _ragged_file())
    var r = Gguf(ragged)
    var refused = False
    try:
        _ = tensor_bytes(r.tensors[0])
    except:
        refused = True
    suite.check(
        refused,
        (
            "twenty Q8_0 weights is not a whole number of blocks, and a size"
            " for it would run into whatever came next"
        ),
    )
    r.close()
    _remove(ragged)


def _check_bad_directory(mut suite: Suite) raises:
    suite.group("model.spec directory audit")

    # The second tensor belongs at 160. Eight bytes further on is a gap, which
    # is what a file that has had something taken out of it looks like.
    var moved = _temp_path("moved")
    _write(moved, _llama_file(168, 192, False))
    var g = Gguf(moved)
    var lay = audit(g)
    suite.check(
        not lay.packed,
        "a tensor eight bytes past where it belongs fails the audit",
    )
    suite.check(lay.first_bad == 1, "and the audit says which one")
    g.close()
    _remove(moved)

    # The directory is fine and the file is too short to hold what it promises,
    # which is the shape of a truncated download.
    var short = _temp_path("short")
    _write(short, _llama_file(160, 32, False))
    var h = Gguf(short)
    var short_lay = audit(h)
    suite.check(
        short_lay.packed and not short_lay.fits,
        (
            "a directory that adds up and runs past the end of the file is"
            " caught before anything maps an offset out of it"
        ),
    )
    h.close()
    _remove(short)


def _check_geometry(mut suite: Suite) raises:
    suite.group("model.spec geometry")

    var path = _temp_path("geom")
    _write(path, _llama_file(160, 192, False))
    var g = Gguf(path)
    var spec = from_gguf(g)
    var geo = spec.geometry

    suite.check(spec.arch_id == ARCH_LLAMA, "the architecture is recognised")
    suite.check(
        geo.block_count == 30 and geo.context_length == 8192,
        "layer count and trained context come straight off the file",
    )
    suite.check(
        geo.head_count == 9 and geo.head_count_kv == 3 and geo.kv_stated,
        "a stated key value head count is read and marked as stated",
    )
    suite.check(geo.grouped_query(), "and three of nine is grouped query")
    suite.check(
        geo.key_length == 64 and not geo.head_dim_stated,
        (
            "the head dimension is the embedding over the head count when the"
            " file does not say, and it is marked as derived"
        ),
    )
    suite.check(
        geo.rope_dimension_count == 64 and not geo.rope_dims_stated,
        (
            "rope rotates the whole head when the file does not say, which is"
            " what llama.cpp does with the same file"
        ),
    )
    suite.check(
        geo.rope_freq_base == 100000.0, "the rope base is read as a float"
    )
    suite.check(
        geo.epsilon > 0.0 and geo.epsilon < 0.001,
        "and the rms epsilon is found under the rms spelling",
    )
    suite.check(not geo.mixture_of_experts(), "with no experts in this one")
    g.close()
    _remove(path)


def _check_defaults(mut suite: Suite) raises:
    suite.group("model.spec defaults")

    var path = _temp_path("bert")
    _write(path, _bert_file())
    var g = Gguf(path)
    var spec = from_gguf(g)
    var geo = spec.geometry

    suite.check(spec.arch_id == ARCH_BERT, "bert is recognised")
    suite.check(
        geo.head_count_kv == 12 and not geo.kv_stated,
        (
            "a file with no key value head count is multi head, and the flag"
            " says the number came from us rather than from it"
        ),
    )
    suite.check(not geo.grouped_query(), "so it is not grouped query")
    suite.check(geo.key_length == 32, "384 over 12 heads is a head dimension")
    suite.check(
        geo.rope_freq_base == 0.0 and not geo.rope_dims_stated,
        "and there are no rope keys at all, which is reported and not invented",
    )
    suite.check(
        geo.epsilon > 0.0,
        "the plain layer norm epsilon is found under its own spelling",
    )
    suite.check(
        spec.file_type == "unstated",
        "a file with no general.file_type says so rather than guessing one",
    )
    g.close()
    _remove(path)


def _check_tokenizer(mut suite: Suite) raises:
    suite.group("model.spec tokenizer")

    var path = _temp_path("tok")
    _write(path, _llama_file(160, 192, False))
    var g = Gguf(path)
    var spec = from_gguf(g)
    var tok = spec.tokenizer

    # The file says gpt2, which is the model byte level BPE first shipped on.
    # The spec says bpe, because that is the question the layers above ask, and
    # it keeps what the file said so a report can quote it.
    suite.check(tok.model == "bpe", "the tokenizer family is read")
    suite.check(
        tok.model_source == "gpt2", "and what the file called it is kept"
    )
    suite.check(
        tok.vocab_size == 3,
        "the vocabulary is counted without any of it being decoded",
    )
    suite.check(tok.merge_count == 0, "and a file with no merges reports none")
    suite.check(tok.bos == 1 and tok.eos == 2, "the special ids are read")
    suite.check(
        tok.eot == -1 and tok.unk == -1,
        "and an id the file does not carry is minus one rather than zero",
    )
    suite.check(tok.add_bos and not tok.add_eos, "the add flags are read")
    suite.check(tok.has_chat_template, "and the chat template is found")
    g.close()
    _remove(path)


def _check_capabilities(mut suite: Suite) raises:
    suite.group("model.spec capabilities")

    suite.check(
        architecture_id("llama") == ARCH_LLAMA
        and architecture_id("qwen2") != ARCH_LLAMA,
        "known architectures map to their own ids",
    )
    suite.check(
        architecture_id("some-model-nobody-has-heard-of") == ARCH_UNKNOWN,
        (
            "and a name this build does not know comes back unknown rather"
            " than being filed under llama"
        ),
    )
    suite.check(
        is_causal(ARCH_LLAMA) and not is_causal(ARCH_BERT),
        "a decoder predicts the next token and an encoder does not",
    )

    var text_path = _temp_path("caps_text")
    _write(text_path, _llama_file(160, 192, False))
    var g = Gguf(text_path)
    var spec = from_gguf(g)
    suite.check(
        spec.declared == CAP_TEXT,
        "a causal architecture declares text and only text",
    )
    suite.check(
        spec.supported == 0 and not spec.runnable(),
        (
            "and nothing is supported, because no kernel in this build has"
            " produced a token yet"
        ),
    )
    suite.check(
        spec.withheld() == CAP_TEXT,
        "so the whole declaration is withheld, which is what a report shows",
    )
    g.close()
    _remove(text_path)

    var embed_path = _temp_path("caps_embed")
    _write(embed_path, _bert_file())
    var e = Gguf(embed_path)
    var embed = from_gguf(e)
    suite.check(
        embed.declared == CAP_EMBEDDING,
        "a pooling type declares embedding and does not declare text",
    )
    e.close()
    _remove(embed_path)

    var vision_path = _temp_path("caps_vision")
    _write(vision_path, _llama_file(160, 256, True))
    var v = Gguf(vision_path)
    var vision = from_gguf(v)
    suite.check(
        vision.declared == CAP_TEXT | CAP_VISION,
        (
            "a projector tensor declares vision, whatever the metadata does or"
            " does not say about one"
        ),
    )
    v.close()
    _remove(vision_path)

    suite.check(
        engine_capabilities(ARCH_LLAMA) == 0,
        "the engine answers no to everything, in one place",
    )
    suite.check(
        capability_names(0) == "none"
        and capability_names(CAP_TEXT | CAP_VISION) == "text, vision",
        "and the names read as a list rather than as a number",
    )


def run(mut suite: Suite) raises:
    _check_encodings(suite)
    _check_sizes(suite)
    _check_bad_directory(suite)
    _check_geometry(suite)
    _check_defaults(suite)
    _check_tokenizer(suite)
    _check_capabilities(suite)
