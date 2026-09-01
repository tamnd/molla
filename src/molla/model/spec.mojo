"""What a GGUF file says the model is, in the terms the rest of molla uses.

The reader in `molla.model.gguf` gives back a key value block and a tensor
directory, which is the file's vocabulary and not ours. Every layer above this
one wants the same handful of facts, and if each of them looks up
`<arch>.attention.head_count_kv` for itself then each of them gets to decide
separately what to do when it is missing. `ModelSpec` is that decision made
once.

Nothing here reads a tensor. The whole spec comes out of the metadata and the
directory, so it costs the pages the header touches and no more, and a machine
can answer what a model is without having the memory to run it.

Three things are worth saying about how this maps rather than what it maps.

Defaults are recorded as defaults. A key that is absent and a key that holds the
value the default would have given are different situations, and the second one
is the file agreeing with us while the first is us guessing. `Geometry` carries
a `stated` flag beside the values that are commonly missing, so a report can
show which numbers came from the file.

Encodings carry their block geometry, not just a name. A ggml type is a block
size and a byte size per block, and knowing those turns a tensor directory into
an exact byte count, which is the only way to check that a directory is
internally consistent before anything trusts an offset in it. `audit` walks the
directory doing exactly that, and it is a real check rather than a formality:
every tensor has to sit at the offset the running total says it should, and the
whole thing has to fit inside the file.

Capabilities are intersected with the engine. What a file declares and what this
binary can do with it on this host are two different questions, and a model that
announces vision on a build with no projector kernels should say so here rather
than fail when the first request arrives. Today the engine answers no to
everything, because no kernel in this binary has produced a token yet, and that
is written in one function so the issues that change it have one place to go.
"""

from molla.model.gguf import (
    Gguf,
    TensorInfo,
    file_type_name,
    ggml_type_name,
)

comptime ARCH_UNKNOWN = 0
comptime ARCH_LLAMA = 1
comptime ARCH_QWEN2 = 2
comptime ARCH_QWEN3 = 3
comptime ARCH_GEMMA = 4
comptime ARCH_GEMMA2 = 5
comptime ARCH_GEMMA3 = 6
comptime ARCH_PHI3 = 7
comptime ARCH_BERT = 8
comptime ARCH_NOMIC_BERT = 9
comptime ARCH_CLIP = 10

comptime CAP_TEXT = 1
"""Generates text one token at a time. The thing most people mean by a model."""

comptime CAP_EMBEDDING = 2
"""Turns a sequence into one vector. Implied by a pooling type in the file."""

comptime CAP_VISION = 4
"""Accepts an image. Implied by projector tensors or a clip vision encoder."""


def architecture_id(name: String) -> Int:
    """Map `general.architecture` onto an id, or ARCH_UNKNOWN.

    The list is short on purpose. An architecture is on it when molla has a
    reason to treat it differently, and a name molla has never heard of comes
    back unknown rather than being quietly filed under llama because the tensor
    names look familiar. Guessing here is how a model runs and produces fluent
    nonsense.
    """
    if name == "llama":
        return ARCH_LLAMA
    if name == "qwen2":
        return ARCH_QWEN2
    if name == "qwen3":
        return ARCH_QWEN3
    if name == "gemma":
        return ARCH_GEMMA
    if name == "gemma2":
        return ARCH_GEMMA2
    if name == "gemma3":
        return ARCH_GEMMA3
    if name == "phi3":
        return ARCH_PHI3
    if name == "bert":
        return ARCH_BERT
    if name == "nomic-bert":
        return ARCH_NOMIC_BERT
    if name == "clip":
        return ARCH_CLIP
    return ARCH_UNKNOWN


def is_causal(arch: Int) -> Bool:
    """Whether this architecture is a decoder that predicts the next token."""
    return (
        arch == ARCH_LLAMA
        or arch == ARCH_QWEN2
        or arch == ARCH_QWEN3
        or arch == ARCH_GEMMA
        or arch == ARCH_GEMMA2
        or arch == ARCH_GEMMA3
        or arch == ARCH_PHI3
    )


@fieldwise_init
struct Encoding(Copyable, ImplicitlyCopyable, Movable):
    """One ggml tensor type, with the block geometry that gives it a size."""

    var kind: Int
    var name: StaticString
    var block: Int
    """Elements per block. One for the types that are not quantized."""

    var block_bytes: Int
    """Bytes one block occupies on disk."""

    var quantized: Bool
    var known: Bool
    """False for a type number this build has no geometry for. A file using one
    is readable as a directory and not as weights, and saying so is the
    difference between an error and a wrong byte count."""

    def bits_per_weight(self) -> Float64:
        if self.block == 0:
            return 0.0
        return Float64(self.block_bytes * 8) / Float64(self.block)


def encoding_of(kind: Int) -> Encoding:
    """Block geometry for a ggml tensor type.

    The numbers are the sizes of ggml's own block structs, which is what makes
    a byte count checkable rather than approximate. `audit` compares the sizes
    this returns against the offsets llama.cpp wrote, so a wrong row here fails
    on any file that uses that type instead of quietly mis-sizing a tensor.
    """
    var name = ggml_type_name(kind)
    # Not quantized: one element per block, and the block is the element.
    if kind == 0:
        return Encoding(kind, name, 1, 4, False, True)
    if kind == 1:
        return Encoding(kind, name, 1, 2, False, True)
    if kind == 24:
        return Encoding(kind, name, 1, 1, False, True)
    if kind == 25:
        return Encoding(kind, name, 1, 2, False, True)
    if kind == 26:
        return Encoding(kind, name, 1, 4, False, True)
    if kind == 27:
        return Encoding(kind, name, 1, 8, False, True)
    if kind == 28:
        return Encoding(kind, name, 1, 8, False, True)
    if kind == 30:
        return Encoding(kind, name, 1, 2, False, True)
    # Blocks of 32.
    if kind == 2:
        return Encoding(kind, name, 32, 18, True, True)
    if kind == 3:
        return Encoding(kind, name, 32, 20, True, True)
    if kind == 6:
        return Encoding(kind, name, 32, 22, True, True)
    if kind == 7:
        return Encoding(kind, name, 32, 24, True, True)
    if kind == 8:
        return Encoding(kind, name, 32, 34, True, True)
    if kind == 9:
        return Encoding(kind, name, 32, 36, True, True)
    if kind == 20:
        return Encoding(kind, name, 32, 18, True, True)
    if kind == 39:
        return Encoding(kind, name, 32, 17, True, True)
    # Blocks of 256, which is what ggml calls QK_K.
    if kind == 10:
        return Encoding(kind, name, 256, 84, True, True)
    if kind == 11:
        return Encoding(kind, name, 256, 110, True, True)
    if kind == 12:
        return Encoding(kind, name, 256, 144, True, True)
    if kind == 13:
        return Encoding(kind, name, 256, 176, True, True)
    if kind == 14:
        return Encoding(kind, name, 256, 210, True, True)
    if kind == 15:
        return Encoding(kind, name, 256, 292, True, True)
    if kind == 16:
        return Encoding(kind, name, 256, 66, True, True)
    if kind == 17:
        return Encoding(kind, name, 256, 74, True, True)
    if kind == 18:
        return Encoding(kind, name, 256, 98, True, True)
    if kind == 19:
        return Encoding(kind, name, 256, 50, True, True)
    if kind == 21:
        return Encoding(kind, name, 256, 110, True, True)
    if kind == 22:
        return Encoding(kind, name, 256, 82, True, True)
    if kind == 23:
        return Encoding(kind, name, 256, 136, True, True)
    if kind == 29:
        return Encoding(kind, name, 256, 56, True, True)
    if kind == 34:
        return Encoding(kind, name, 256, 54, True, True)
    if kind == 35:
        return Encoding(kind, name, 256, 66, True, True)
    return Encoding(kind, name, 0, 0, False, False)


def tensor_bytes(t: TensorInfo) raises -> Int:
    """How many bytes this tensor occupies on disk.

    Raises rather than rounding when the element count is not a whole number of
    blocks. A quantized tensor whose first dimension is not a multiple of the
    block size is not a tensor ggml could have written, and treating it as one
    would hand back a length that runs into the next tensor.
    """
    var enc = encoding_of(t.kind)
    if not enc.known:
        raise Error(
            "gguf tensor type " + String(t.kind) + " has no known block size"
        )
    var n = Int(t.elements())
    if n % enc.block != 0:
        raise Error(
            "gguf tensor has "
            + String(n)
            + " elements, which is not a whole number of "
            + String(enc.block)
            + " element blocks"
        )
    return (n // enc.block) * enc.block_bytes


@fieldwise_init
struct Layout(Copyable, ImplicitlyCopyable, Movable):
    """The result of walking the tensor directory and adding it up."""

    var tensors: Int
    var bytes: Int
    var quantized_bytes: Int
    var unknown_types: Int
    var packed: Bool
    """Every tensor sits at the offset the running total says it should. This is
    what llama.cpp writes, so a file that is not packed is either from another
    writer or has had something inserted into it."""

    var fits: Bool
    """The last tensor ends at or before the end of the file."""

    var first_bad: Int
    """Index of the first tensor that is not where it should be, or -1."""


def audit(g: Gguf) raises -> Layout:
    """Add up the tensor directory and check it against the file.

    Tensors are laid out in directory order, each one starting at the next
    alignment boundary after the previous one ended. Recomputing that from the
    block geometry and comparing it against the offsets in the file checks two
    things at once: that the size table is right, and that the directory has
    not been tampered with. Both matter before anything maps a tensor, because
    an offset trusted from a downloaded file is a read wherever the file says.
    """
    var total = 0
    var quantized = 0
    var unknown = 0
    var packed = True
    var first_bad = -1
    var expected = 0
    for i in range(len(g.tensors)):
        var t = g.tensors[i]
        var enc = encoding_of(t.kind)
        if not enc.known:
            unknown += 1
            packed = False
            if first_bad < 0:
                first_bad = i
            break
        var size = tensor_bytes(t)
        if t.offset != expected and packed:
            packed = False
            first_bad = i
        total = t.offset + size
        if enc.quantized:
            quantized += size
        var pad = (g.alignment - (total % g.alignment)) % g.alignment
        expected = total + pad
    var fits = g.data_start + total <= g.mapping.length
    return Layout(
        len(g.tensors), total, quantized, unknown, packed, fits, first_bad
    )


@fieldwise_init
struct Geometry(Copyable, ImplicitlyCopyable, Movable):
    """The shape of the network, in the terms the engine will ask for."""

    var block_count: Int
    var context_length: Int
    var embedding_length: Int
    var feed_forward_length: Int
    var head_count: Int
    var head_count_kv: Int
    var key_length: Int
    var value_length: Int
    var expert_count: Int
    var expert_used_count: Int
    var rope_dimension_count: Int
    var rope_freq_base: Float64
    var rope_scale_factor: Float64
    var rope_scaling: String
    var epsilon: Float64
    var head_dim_stated: Bool
    """Whether `attention.key_length` was in the file. When it was not, the head
    dimension is embedding length over head count, which is true of every model
    here and not true of every model."""

    var kv_stated: Bool
    """Whether `attention.head_count_kv` was in the file. When it was not the
    model is multi head rather than grouped query, which is the same as saying
    it equals the head count."""

    var rope_dims_stated: Bool
    """Whether `rope.dimension_count` was in the file. Most files leave it out,
    including both of the ones here that use rope, and llama.cpp then rotates
    the whole head dimension. Defaulting to anything else would rotate part of
    a vector and leave the rest, which produces output rather than an error."""

    def grouped_query(self) -> Bool:
        return self.head_count_kv > 0 and self.head_count_kv < self.head_count

    def mixture_of_experts(self) -> Bool:
        return self.expert_count > 0


def read_geometry(g: Gguf) -> Geometry:
    """Pull the geometry keys, which are all prefixed with the architecture."""
    var arch = g.architecture()
    var block_count = g.uint_or(arch + ".block_count", 0)
    var context_length = g.uint_or(arch + ".context_length", 0)
    var embedding_length = g.uint_or(arch + ".embedding_length", 0)
    var feed_forward = g.uint_or(arch + ".feed_forward_length", 0)
    var heads = g.uint_or(arch + ".attention.head_count", 0)

    var kv_stated = g.has(arch + ".attention.head_count_kv")
    var heads_kv = g.uint_or(arch + ".attention.head_count_kv", heads)

    var head_dim_stated = g.has(arch + ".attention.key_length")
    var rope_dims_stated = g.has(arch + ".rope.dimension_count")
    var derived = embedding_length // heads if heads > 0 else 0
    var key_length = g.uint_or(arch + ".attention.key_length", derived)
    var value_length = g.uint_or(arch + ".attention.value_length", key_length)

    # Both spellings are in the wild. A model has one or the other, never both,
    # so reading the rms key first and falling back covers each without needing
    # to know which family uses which.
    var epsilon = g.float_or(arch + ".attention.layer_norm_rms_epsilon", 0.0)
    if epsilon == 0.0:
        epsilon = g.float_or(arch + ".attention.layer_norm_epsilon", 0.0)

    return Geometry(
        block_count=block_count,
        context_length=context_length,
        embedding_length=embedding_length,
        feed_forward_length=feed_forward,
        head_count=heads,
        head_count_kv=heads_kv,
        key_length=key_length,
        value_length=value_length,
        expert_count=g.uint_or(arch + ".expert_count", 0),
        expert_used_count=g.uint_or(arch + ".expert_used_count", 0),
        rope_dimension_count=g.uint_or(
            arch + ".rope.dimension_count", key_length
        ),
        rope_freq_base=g.float_or(arch + ".rope.freq_base", 0.0),
        rope_scale_factor=g.float_or(arch + ".rope.scaling.factor", 0.0),
        rope_scaling=g.string_or(arch + ".rope.scaling.type", "unstated"),
        epsilon=epsilon,
        head_dim_stated=head_dim_stated,
        kv_stated=kv_stated,
        rope_dims_stated=rope_dims_stated,
    )


@fieldwise_init
struct TokenizerSpec(Copyable, ImplicitlyCopyable, Movable):
    """What the file says about turning text into tokens.

    The vocabulary itself is not here. It is a thirty thousand entry array in
    the mapping and it stays there until something actually tokenizes, which is
    issue #21. What this holds is the shape of it and the special token ids,
    which is what a report and a template need.
    """

    var model: String
    """The algorithm: spm, bpe, wpm, or none."""

    var model_source: String
    """What the file called it. GGUF names the tokenizer after the model it
    first shipped on, so byte level BPE is `gpt2` and SentencePiece is `llama`,
    while `tokenizer.json` names the algorithm as `BPE` and `Unigram`. Both are
    kept because `model` is the one to branch on and this is the one to quote
    when a file turns out to be lying about which it is."""

    var pre: String
    """`tokenizer.ggml.pre`, the pre-tokenizer regex family. Absent on older
    files, which is a real difference in behaviour rather than a missing
    label."""

    var vocab_size: Int
    var merge_count: Int
    var bos: Int
    var eos: Int
    var eot: Int
    var pad: Int
    var unk: Int
    var sep: Int
    var add_bos: Bool
    var add_eos: Bool
    var has_chat_template: Bool

    var embedding_rows: Int
    """How many rows the embedding matrix has, when the model says. It is not
    always the number of tokens: a vocabulary is commonly padded up to a round
    number so the matrix divides evenly across devices, and the rows past the
    last real token are never selected. Zero when nothing states it."""


def tokenizer_algorithm(name: String) -> String:
    """The algorithm behind a tokenizer name, in one spelling.

    GGUF and `tokenizer.json` describe the same three algorithms in different
    words, and a caller that has to know which one it is should not have to know
    both vocabularies. `gpt2` and `BPE` are byte level BPE. `llama`, `t5` and
    `Unigram` are SentencePiece. `bert` and `WordPiece` are WordPiece. Anything
    else comes back unchanged, which reports as itself rather than as a guess.
    """
    if name == "gpt2" or name == "BPE":
        return String("bpe")
    if name == "llama" or name == "t5" or name == "Unigram":
        return String("spm")
    if name == "bert" or name == "WordPiece":
        return String("wpm")
    return name


def read_tokenizer(g: Gguf) -> TokenizerSpec:
    var model = g.string_or("tokenizer.ggml.model", "none")
    return TokenizerSpec(
        model=tokenizer_algorithm(model),
        model_source=model,
        pre=g.string_or("tokenizer.ggml.pre", "none"),
        vocab_size=g.array_count("tokenizer.ggml.tokens"),
        merge_count=g.array_count("tokenizer.ggml.merges"),
        bos=g.uint_or("tokenizer.ggml.bos_token_id", -1),
        eos=g.uint_or("tokenizer.ggml.eos_token_id", -1),
        eot=g.uint_or("tokenizer.ggml.eot_token_id", -1),
        pad=g.uint_or("tokenizer.ggml.padding_token_id", -1),
        unk=g.uint_or("tokenizer.ggml.unknown_token_id", -1),
        sep=g.uint_or("tokenizer.ggml.seperator_token_id", -1),
        add_bos=g.bool_or("tokenizer.ggml.add_bos_token", False),
        add_eos=g.bool_or("tokenizer.ggml.add_eos_token", False),
        has_chat_template=g.has("tokenizer.chat_template"),
        embedding_rows=g.uint_or(g.architecture() + ".vocab_size", 0),
    )


def declared_capabilities(g: Gguf) -> Int:
    """What the file says it can do.

    Vision is inferred from tensors rather than from a key, because a projector
    is a set of tensors and the key that announces one is only present on files
    written by tools that bother. A model carrying `mm.` or `v.` tensors has a
    vision tower in it whatever the metadata says.
    """
    var arch = g.architecture()
    var id = architecture_id(arch)
    var caps = 0
    if is_causal(id):
        caps |= CAP_TEXT
    if (
        g.has(arch + ".pooling_type")
        or id == ARCH_BERT
        or (id == ARCH_NOMIC_BERT)
    ):
        caps |= CAP_EMBEDDING
    if (
        id == ARCH_CLIP
        or g.bool_or("clip.has_vision_encoder", False)
        or g.tensor_prefixed("mm.")
        or g.tensor_prefixed("v.blk.")
    ):
        caps |= CAP_VISION
    return caps


def engine_capabilities(arch: Int) -> Int:
    """What this build can actually do with this architecture on this host.

    Zero, for every architecture, because nothing in this binary has produced a
    token yet. The prefill and decode path is issue #27 and the dense blocks are
    issue #26, and when they land this is where they say so.

    It is a function rather than a constant because the answer will stop being
    uniform. An architecture molla implements on CPU and not on Metal is the
    normal case in a project with four targets, and the intersection has to
    happen somewhere that knows both halves. Reporting a capability that the
    engine does not have moves the failure from here, where it is a line in a
    report, to the first request, where it is a five hundred.
    """
    _ = arch
    return 0


def capability_names(caps: Int) -> String:
    if caps == 0:
        return String("none")
    var out = String("")
    if caps & CAP_TEXT != 0:
        out += "text"
    if caps & CAP_EMBEDDING != 0:
        if out != "":
            out += ", "
        out += "embedding"
    if caps & CAP_VISION != 0:
        if out != "":
            out += ", "
        out += "vision"
    return out


@fieldwise_init
struct ModelSpec(Movable):
    """Everything molla needs to know about a model before it loads one."""

    var name: String
    var source: String
    """Which reader produced this, gguf or safetensors. A report says so because
    the two formats state different things about the same model, and knowing
    which one was read is the first question about any number below."""

    var architecture: String
    var arch_id: Int
    var file_type: String
    var quantized: Bool
    var geometry: Geometry
    var tokenizer: TokenizerSpec
    var layout: Layout
    var declared: Int
    var supported: Int
    """Declared, intersected with what the engine can do here."""

    def runnable(self) -> Bool:
        return self.supported != 0

    def withheld(self) -> Int:
        """Capabilities the file declares that this build cannot serve."""
        return self.declared & ~self.supported


def from_gguf(g: Gguf) raises -> ModelSpec:
    """Build the spec. Reads metadata and the directory, never a weight."""
    var arch = g.architecture()
    var id = architecture_id(arch)
    var declared = declared_capabilities(g)
    var lay = audit(g)

    var file_type = String("unstated")
    var i = g.find("general.file_type")
    if i >= 0:
        file_type = String(file_type_name(Int(g.uint(i))))

    return ModelSpec(
        name=g.string_or("general.name", "unnamed"),
        source=String("gguf"),
        architecture=arch,
        arch_id=id,
        file_type=file_type,
        quantized=lay.quantized_bytes > 0,
        geometry=read_geometry(g),
        tokenizer=read_tokenizer(g),
        layout=lay,
        declared=declared,
        supported=declared & engine_capabilities(id),
    )


def _mib(bytes: Int) -> String:
    """Bytes as whole mebibytes, which is the unit a model is discussed in."""
    return String(bytes // (1024 * 1024)) + " MiB"


def report(spec: ModelSpec, path: StringSpan):
    """Print a spec.

    One printer for both readers. `molla spec` on a GGUF file and on a Hugging
    Face directory answer the same questions in the same order, and where the
    two formats state different things the difference shows up as a different
    number under the same heading rather than as a differently shaped report.
    """
    var geo = spec.geometry
    var tok = spec.tokenizer

    print("path:           " + String(path))
    print("read as:        " + spec.source)
    print("name:           " + spec.name)
    print(
        "architecture:   "
        + spec.architecture
        + (" (not recognised)" if spec.arch_id == ARCH_UNKNOWN else "")
    )
    print("file type:      " + spec.file_type)
    print("")
    print("geometry")
    print("  layers        " + String(geo.block_count))
    print("  context       " + String(geo.context_length))
    print("  embedding     " + String(geo.embedding_length))
    print("  feed forward  " + String(geo.feed_forward_length))
    print(
        "  heads         "
        + String(geo.head_count)
        + " attention, "
        + String(geo.head_count_kv)
        + (" key value" if geo.kv_stated else " key value (assumed)")
        + (", grouped query" if geo.grouped_query() else "")
    )
    print(
        "  head dim      "
        + String(geo.key_length)
        + ("" if geo.head_dim_stated else " (embedding over heads)")
    )
    if geo.mixture_of_experts():
        print(
            "  experts       "
            + String(geo.expert_count)
            + ", "
            + String(geo.expert_used_count)
            + " used per token"
        )
    if geo.rope_freq_base == 0.0 and not geo.rope_dims_stated:
        print("  rope          the file carries no rope keys")
    else:
        print(
            "  rope          base "
            + String(geo.rope_freq_base)
            + ", "
            + String(geo.rope_dimension_count)
            + (
                " dimensions" if geo.rope_dims_stated else " dimensions (the whole head)"
            )
            + ", scaling "
            + geo.rope_scaling
        )
    print("  epsilon       " + String(geo.epsilon))
    print("")
    print("tokenizer")
    var spelling = String("")
    if tok.model_source != tok.model:
        spelling = ", spelled " + tok.model_source
    print("  model         " + tok.model + spelling + ", pre " + tok.pre)
    print(
        "  vocabulary    "
        + String(tok.vocab_size)
        + " tokens, "
        + String(tok.merge_count)
        + " merges"
    )
    if tok.embedding_rows > tok.vocab_size and tok.vocab_size > 0:
        print(
            "  embedding     "
            + String(tok.embedding_rows)
            + " rows, "
            + String(tok.embedding_rows - tok.vocab_size)
            + " of them past the last token"
        )
    elif tok.embedding_rows > 0 and tok.embedding_rows < tok.vocab_size:
        # Not padding, the other way round: a token that can be produced and has
        # no row to look up. Gemma 3 does this in its text only checkpoints,
        # which keep the image token id from the multimodal ones.
        print(
            "  embedding     "
            + String(tok.embedding_rows)
            + " rows, token ids run "
            + String(tok.vocab_size - tok.embedding_rows)
            + " past the last row"
        )
    print(
        "  specials      bos "
        + String(tok.bos)
        + ", eos "
        + String(tok.eos)
        + ", eot "
        + String(tok.eot)
        + ", pad "
        + String(tok.pad)
        + ", unk "
        + String(tok.unk)
    )
    print(
        "  adds          bos "
        + ("yes" if tok.add_bos else "no")
        + ", eos "
        + ("yes" if tok.add_eos else "no")
    )
    print(
        "  chat template " + ("present" if tok.has_chat_template else "absent")
    )
    print("")
    print("tensors")
    print(
        "  count         "
        + String(spec.layout.tensors)
        + " tensors, "
        + String(spec.layout.bytes)
        + " bytes of weights ("
        + _mib(spec.layout.bytes)
        + ")"
    )
    if spec.layout.quantized_bytes > 0:
        print(
            "  quantized     "
            + String(spec.layout.quantized_bytes)
            + " bytes of that, "
            + String(spec.layout.bytes - spec.layout.quantized_bytes)
            + " stored at full width"
        )
    else:
        print("  quantized     none, every tensor is stored at its own width")
    if spec.layout.unknown_types > 0:
        print("  directory     stops at a tensor type this build cannot size")
    elif not spec.layout.packed:
        print(
            "  directory     tensor "
            + String(spec.layout.first_bad)
            + " is not where the running total says it should be"
        )
    elif not spec.layout.fits:
        print("  directory     runs past the end of the file")
    else:
        print(
            "  directory     packed and inside the file, every offset checked"
            " against the block geometry"
        )
    print("")
    print("capabilities")
    print("  declared      " + capability_names(spec.declared))
    print("  supported     " + capability_names(spec.supported))
    if spec.withheld() != 0:
        print(
            "  withheld      "
            + capability_names(spec.withheld())
            + ", because no kernel in this build produces a token yet"
        )


def run_spec(path: StringSpan) raises:
    """Entry point for `molla spec` on a GGUF file."""
    var g = Gguf(path)
    var spec = from_gguf(g)
    report(spec, path)
    g.close()
