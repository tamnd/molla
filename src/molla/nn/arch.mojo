"""The table that says what each architecture does differently.

`molla.nn.block` can run any layer that fits its spec. This is where a file
turns into that spec: an architecture id and a `Geometry` go in, and a
`BlockSpec` for a particular layer comes out, along with the ggml names of the
tensors that layer needs.

The table is a table on purpose. The alternative is a branch on the
architecture id inside the forward pass, and the trouble with that is not the
branching, it is that the differences stop being enumerable. Written out like
this, adding an architecture is a row and a paragraph in
`docs/adding-an-architecture.md`, and reading it tells you what the differences
between two models actually are, which is less than most people expect.

What is in the table is what the architecture always does. What is in the file
is what this particular model does. Head counts, widths, rope base and the norm
epsilon come from `Geometry` and are never guessed here. The table only holds
the things that are not in a GGUF at all, like whether the query is normalised
per head, because nothing writes a key saying so.

A sliding window is per layer rather than per model, because Gemma alternates
local and global attention and the alternation is part of the architecture.
That is why `block_spec` takes a layer index.
"""

from molla.model.spec import (
    ARCH_GEMMA,
    ARCH_GEMMA2,
    ARCH_GEMMA3,
    ARCH_LLAMA,
    ARCH_PHI3,
    ARCH_QWEN2,
    ARCH_QWEN3,
    ARCH_UNKNOWN,
    Geometry,
)
from molla.nn.attention import AttnSpec
from molla.nn.block import ACT_GELU, ACT_SILU, BlockSpec
from molla.nn.rope import RopeSpec


struct Arch(Copyable, ImplicitlyCopyable, Movable):
    """One architecture's answers to the questions a layer can ask."""

    var id: Int
    var name: String

    var gated: Bool
    """Whether the mlp computes a gate as well as an up projection."""

    var act: Int
    """`ACT_SILU` or `ACT_GELU`."""

    var neox: Bool
    """The rope pairing. False pairs adjacent elements, which is what a
    converted Llama wants because the conversion script permuted the query and
    key weights to suit it. True pairs `i` with `i + dim/2`, which is everything
    else. This is a property of the file rather than of the model."""

    var qk_norm: Bool
    """Whether each head of the query and key is rms normed before rope."""

    var qkv_bias: Bool
    """Whether the three attention projections carry a bias. Qwen 2 does, Qwen
    3 dropped it, and Llama never had one. A bias molla does not add is a
    constant vector missing from every head of every layer, which is a model
    that runs at full speed and writes noise."""

    var post_norms: Bool
    """Whether each sublayer's output is normed on the way to the residual
    add, as well as its input being normed on the way in."""

    var window: Int
    """The sliding window, on the layers that have one. Zero is none."""

    var window_pattern: Int
    """One layer in every `window_pattern` is full attention and the rest are
    windowed. Zero means every layer is the same. Two is Gemma 2 alternating,
    six is Gemma 3's five local layers to one global."""

    var local_rope_base: Float32
    """The rope base on windowed layers, when it differs from the global one.
    Gemma 3 uses 10000 locally and a million globally, because a local layer
    only ever looks back a thousand positions and does not need frequencies
    that resolve a hundred thousand. Zero means use the file's base
    everywhere."""

    var softcap: Float32
    """`cap * tanh(score / cap)` on attention logits. Zero is off. Gemma 2 caps
    at fifty, and Gemma 3 dropped it again."""

    var final_softcap: Float32
    """The same treatment applied to the output logits. Gemma 2 caps those at
    thirty, which is a different number from its attention cap and has to be a
    different field for that reason."""

    var scale_embedding: Bool
    """Whether a looked up embedding is multiplied by the square root of the
    model width before the first layer. Gemma does this and it is not small:
    the factor is about sixty on a 3072 wide model, so leaving it out does not
    degrade the output, it replaces it."""

    var supported: Bool
    """Whether molla will try to run this. An architecture can be in the table,
    which is a claim about what it does, without being supported, which is a
    claim that molla gets it right."""

    def __init__(out self, id: Int, name: String):
        """A plain gated silu decoder with neox pairing, which is the modern
        default and what every entry below starts from."""
        self.id = id
        self.name = name
        self.gated = True
        self.act = ACT_SILU
        self.neox = True
        self.qk_norm = False
        self.qkv_bias = False
        self.post_norms = False
        self.window = 0
        self.window_pattern = 0
        self.local_rope_base = 0.0
        self.softcap = 0.0
        self.final_softcap = 0.0
        self.scale_embedding = False
        self.supported = False

    def windowed(self, layer: Int) -> Bool:
        """Whether layer `layer` slides its window.

        With no pattern every layer is the same and the answer is whether there
        is a window at all. With a pattern of `n`, the last layer of each run of
        `n` is the full attention one, so layers `n-1`, `2n-1` and so on see
        everything. That is llama.cpp's rule and it is worth matching exactly,
        because getting the phase off by one gives a model that is subtly worse
        rather than one that is broken.
        """
        if self.window <= 0:
            return False
        if self.window_pattern <= 0:
            return True
        return layer % self.window_pattern < self.window_pattern - 1


def arch_of(id: Int) raises -> Arch:
    """The table."""
    if id == ARCH_LLAMA:
        # Llama 1 through 3. The conversion script permutes the query and key
        # weights so that adjacent pairing comes out right, which is the one
        # thing about this family that surprises people.
        var a = Arch(ARCH_LLAMA, "llama")
        a.neox = False
        a.supported = True
        return a

    if id == ARCH_QWEN2:
        # A bias on each of the three attention projections, which Qwen 3
        # dropped and which nothing in the metadata announces. A file has the
        # tensors or it does not.
        var a = Arch(ARCH_QWEN2, "qwen2")
        a.qkv_bias = True
        a.supported = True
        return a

    if id == ARCH_QWEN3:
        # The one difference from Qwen 2 that matters, and it is not visible in
        # any metadata key: each head of q and k is rms normed before rope.
        # Skipping it gives a model that produces words.
        var a = Arch(ARCH_QWEN3, "qwen3")
        a.qk_norm = True
        a.supported = True
        return a

    if id == ARCH_PHI3:
        var a = Arch(ARCH_PHI3, "phi3")
        return a

    if id == ARCH_GEMMA:
        var a = Arch(ARCH_GEMMA, "gemma")
        a.act = ACT_GELU
        a.scale_embedding = True
        return a

    if id == ARCH_GEMMA2:
        var a = Arch(ARCH_GEMMA2, "gemma2")
        a.act = ACT_GELU
        a.post_norms = True
        a.window = 4096
        a.window_pattern = 2
        a.softcap = 50.0
        a.final_softcap = 30.0
        a.scale_embedding = True
        return a

    if id == ARCH_GEMMA3:
        var a = Arch(ARCH_GEMMA3, "gemma3")
        a.act = ACT_GELU
        a.post_norms = True
        a.qk_norm = True
        a.window = 1024
        a.window_pattern = 6
        a.local_rope_base = 10000.0
        a.scale_embedding = True
        return a

    raise Error(
        "architecture "
        + String(id)
        + " is not in the table, so molla does not know what its layers do"
    )


def in_table(id: Int) -> Bool:
    if id == ARCH_UNKNOWN:
        return False
    try:
        _ = arch_of(id)
        return True
    except:
        return False


def rope_spec(a: Arch, g: Geometry, layer: Int) raises -> RopeSpec:
    """The rope for one layer, from the table and the file.

    The scaling type is a string in the metadata rather than an enum, and an
    unstated one is not the same as `none`: a file that says nothing is a file
    that was never scaled, while a file that says `none` was written by a
    converter that had an opinion. Both mean no scaling here, and they are read
    the same way on purpose, but only after checking that a factor is not also
    sitting there contradicting them.
    """
    var dim = g.rope_dimension_count
    if dim <= 0:
        raise Error("a rope needs a rotary dimension, and the file states none")

    var base = Float32(g.rope_freq_base)
    if a.local_rope_base > 0 and a.windowed(layer):
        base = a.local_rope_base
    if base <= 0:
        raise Error("a rope needs a frequency base, and the file states none")

    var kind = g.rope_scaling
    var factor = Float32(g.rope_scale_factor)

    if kind == "unstated" or kind == "none":
        if factor > 1.0:
            raise Error(
                "the file gives a rope scaling factor of "
                + String(factor)
                + " and a scaling type of "
                + kind
                + ", which do not agree"
            )
        # The pairing is set here and not in the constructor, which defaults to
        # neox. An unscaled Llama goes through this branch and pairs adjacent
        # elements, and forgetting that here is a scheme that is right on every
        # scaled model and wrong on the plain one.
        var plain = RopeSpec(dim, base)
        plain.neox = a.neox
        return plain

    if factor <= 0:
        raise Error(
            "rope scaling of type "
            + kind
            + " needs a factor and the file states none"
        )

    if kind == "linear":
        return RopeSpec.linear(dim, base, factor, a.neox)
    if kind == "yarn":
        var orig = g.context_length
        if orig <= 0:
            raise Error("yarn needs the context the model was trained on")
        return RopeSpec.yarn(dim, base, factor, orig, a.neox)

    raise Error(
        "rope scaling of type "
        + kind
        + " is not something molla knows how to do"
    )


def block_spec(a: Arch, g: Geometry, layer: Int) raises -> BlockSpec:
    """The spec for one layer of this model."""
    if layer < 0 or (g.block_count > 0 and layer >= g.block_count):
        raise Error(
            "layer "
            + String(layer)
            + " is out of range for a model with "
            + String(g.block_count)
            + " of them"
        )
    if g.mixture_of_experts():
        raise Error(
            "this model has "
            + String(g.expert_count)
            + " experts per layer, and molla routes to none of them yet"
        )
    if g.key_length != g.value_length:
        raise Error(
            "keys are "
            + String(g.key_length)
            + " wide and values are "
            + String(g.value_length)
            + ", which molla does not handle"
        )
    if g.epsilon <= 0:
        raise Error("the file states no norm epsilon")

    var attn = AttnSpec(g.head_count, g.head_count_kv, g.key_length)
    attn.softcap = a.softcap
    if a.windowed(layer):
        attn.window = a.window

    var spec = BlockSpec(
        attn,
        rope_spec(a, g, layer),
        g.embedding_length,
        g.feed_forward_length,
        Float32(g.epsilon),
    )
    spec.gated = a.gated
    spec.act = a.act
    spec.qkv_bias = a.qkv_bias
    return spec


def layer_prefix(layer: Int) -> String:
    return "blk." + String(layer) + "."


def tensor_names(a: Arch, layer: Int) -> List[String]:
    """The ggml names of every tensor this layer needs, in a fixed order.

    The order matches the fields of `LayerWeights`, so a loader can walk the
    two together. Names a layer does not have are the empty string rather than
    being left out, because a list whose length depends on the architecture is
    a list that has to be indexed by searching.
    """
    var p = layer_prefix(layer)
    var out = List[String]()
    out.append(p + "attn_norm.weight")
    out.append(p + "post_attention_norm.weight" if a.post_norms else "")
    out.append(p + "attn_q.weight")
    out.append(p + "attn_k.weight")
    out.append(p + "attn_v.weight")
    out.append(p + "attn_output.weight")
    out.append(p + "attn_q.bias" if a.qkv_bias else "")
    out.append(p + "attn_k.bias" if a.qkv_bias else "")
    out.append(p + "attn_v.bias" if a.qkv_bias else "")
    out.append(p + "attn_q_norm.weight" if a.qk_norm else "")
    out.append(p + "attn_k_norm.weight" if a.qk_norm else "")
    out.append(p + "ffn_norm.weight")
    out.append(p + "post_ffw_norm.weight" if a.post_norms else "")
    out.append(p + "ffn_gate.weight" if a.gated else "")
    out.append(p + "ffn_up.weight")
    out.append(p + "ffn_down.weight")
    return out^


comptime TOKEN_EMBD = "token_embd.weight"
"""The embedding, which is also the output head when they are tied."""

comptime OUTPUT_NORM = "output_norm.weight"
"""The norm before the output head."""

comptime OUTPUT_HEAD = "output.weight"
"""Absent when the head is tied to the embedding, which is most small models
and is a real saving: a 1B with a 128k vocabulary spends a seventh of its
weights on the embedding and would spend two sevenths untied."""

comptime ROPE_FREQS = "rope_freqs.weight"
"""One frequency factor per rope pair. This is how a Llama 3.1 file carries its
context extension, computed by the converter and written into the file, so
there is nothing to implement for it beyond dividing by it."""


def describe(a: Arch) -> String:
    """A one line summary, for `molla spec` and for error messages."""
    var out = a.name
    out += ": " + ("gated " if a.gated else "")
    out += "silu" if a.act == ACT_SILU else "gelu"
    out += " mlp, " + ("neox" if a.neox else "adjacent") + " rope pairing"
    if a.qk_norm:
        out += ", per head query and key norms"
    if a.post_norms:
        out += ", norms on both sides of each sublayer"
    if a.window > 0:
        out += ", a " + String(a.window) + " token window"
        if a.window_pattern > 0:
            out += " on " + String(a.window_pattern - 1)
            out += " layers in every " + String(a.window_pattern)
    if a.softcap > 0:
        out += ", attention logits capped at " + String(Int(a.softcap))
    if not a.supported:
        out += ". Described but not supported."
    return out
