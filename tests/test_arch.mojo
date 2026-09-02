"""The architecture table.

A table's tests are mostly a second copy of the table, which would be worth
very little on its own. What is worth checking is the part that is not a
lookup: the window alternation, which is arithmetic and is off by one in the
obvious implementation, and the way a `Geometry` out of a real file turns into
a `BlockSpec`, which is where a missing key or a contradictory pair of keys has
to become an error rather than a default.

The rows themselves are checked for the two or three facts per architecture
that are worth writing down twice, chosen because getting them wrong gives a
model that talks rather than a model that crashes.
"""

from harness import Suite

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
from molla.nn.arch import (
    Arch,
    arch_of,
    block_spec,
    describe,
    in_table,
    rope_spec,
    tensor_names,
)
from molla.nn.block import ACT_GELU, ACT_SILU


def _close(a: Float32, b: Float32, within: Float32) -> Bool:
    var d = a - b
    if d < 0:
        d = -d
    return d <= within


def _geometry() -> Geometry:
    """A small Llama shaped model, with every key a file would actually have."""
    return Geometry(
        block_count=8,
        context_length=8192,
        embedding_length=64,
        feed_forward_length=128,
        head_count=4,
        head_count_kv=2,
        key_length=16,
        value_length=16,
        expert_count=0,
        expert_used_count=0,
        rope_dimension_count=16,
        rope_freq_base=500000.0,
        rope_scale_factor=0.0,
        rope_scaling="unstated",
        epsilon=1e-5,
        head_dim_stated=True,
        kv_stated=True,
        rope_dims_stated=True,
    )


def run(mut suite: Suite) raises:
    test_rows(suite)
    test_windows(suite)
    test_names(suite)
    test_specs(suite)
    test_rope(suite)
    test_errors(suite)


def test_rows(mut suite: Suite) raises:
    suite.group("architecture table")

    var llama = arch_of(ARCH_LLAMA)
    suite.check(
        not llama.neox,
        (
            "llama pairs adjacent elements, because the converter permuted the"
            " weights to suit it"
        ),
    )
    suite.check(
        llama.gated and llama.act == ACT_SILU and not llama.qk_norm,
        "and it is a plain gated silu mlp with no query norms",
    )

    var qwen2 = arch_of(ARCH_QWEN2)
    var qwen3 = arch_of(ARCH_QWEN3)
    suite.check(
        qwen2.neox and qwen3.neox, "both qwens pair across the half point"
    )
    suite.check(
        qwen3.qk_norm and not qwen2.qk_norm,
        "and qwen3 norms each head of the query and key where qwen2 does not",
    )
    suite.check(
        qwen2.qkv_bias and not qwen3.qkv_bias,
        "while qwen2 biases the three attention projections and qwen3 does not",
    )
    suite.check(not llama.qkv_bias, "and no llama has ever had one")

    var gemma2 = arch_of(ARCH_GEMMA2)
    suite.check(
        gemma2.act == ACT_GELU and gemma2.post_norms,
        "gemma2 gates with gelu and norms both sides of a sublayer",
    )
    suite.check(
        _close(gemma2.softcap, 50.0, 1e-6)
        and _close(gemma2.final_softcap, 30.0, 1e-6),
        "and caps attention logits at fifty and output logits at thirty",
    )
    suite.check(
        gemma2.scale_embedding,
        "and multiplies the embedding by the root of the model width",
    )

    var gemma3 = arch_of(ARCH_GEMMA3)
    suite.check(
        gemma3.softcap == 0.0,
        "gemma3 dropped the attention softcap that gemma2 had",
    )
    suite.check(
        _close(gemma3.local_rope_base, 10000.0, 1e-6),
        "and gives its windowed layers a lower rope base than its global ones",
    )

    suite.check(
        arch_of(ARCH_LLAMA).supported
        and arch_of(ARCH_QWEN2).supported
        and arch_of(ARCH_QWEN3).supported,
        "llama and both qwens are supported",
    )
    suite.check(
        not arch_of(ARCH_GEMMA3).supported and not arch_of(ARCH_PHI3).supported,
        "and the rest are described without being claimed",
    )

    suite.check(
        not in_table(ARCH_UNKNOWN),
        "an architecture molla never heard of is not in the table",
    )
    suite.check(
        in_table(ARCH_GEMMA) and in_table(ARCH_LLAMA),
        "and the ones that are, are",
    )


def test_windows(mut suite: Suite) raises:
    """The alternation, which is the only arithmetic in the table."""
    suite.group("architecture sliding windows")

    var llama = arch_of(ARCH_LLAMA)
    suite.check(
        not llama.windowed(0) and not llama.windowed(31),
        "a model with no window has none on any layer",
    )

    # Gemma 2 alternates, and the full attention layer is the second of each
    # pair rather than the first.
    var g2 = arch_of(ARCH_GEMMA2)
    suite.check(
        g2.windowed(0) and not g2.windowed(1),
        "gemma2 slides on the first layer of each pair",
    )
    suite.check(
        g2.windowed(2) and not g2.windowed(3) and g2.windowed(40),
        "and keeps alternating all the way up",
    )

    # Gemma 3 is five local layers then one global, so layer five sees
    # everything and layer six starts the next run.
    var g3 = arch_of(ARCH_GEMMA3)
    var run_ok = True
    for i in range(5):
        if not g3.windowed(i):
            run_ok = False
    suite.check(run_ok, "gemma3 slides on five layers in a row")
    suite.check(
        not g3.windowed(5) and g3.windowed(6),
        "and the sixth is the global one",
    )
    suite.check(
        not g3.windowed(11) and not g3.windowed(17),
        "and every sixth after that",
    )


def test_names(mut suite: Suite) raises:
    suite.group("architecture tensor names")

    var names = tensor_names(arch_of(ARCH_LLAMA), 7)
    suite.check(
        names[0] == "blk.7.attn_norm.weight",
        "tensor names carry the layer number",
    )
    suite.check(
        names[1] == "" and names[9] == "" and names[12] == "",
        "and a llama has no post norms and no query norms",
    )
    suite.check(
        names[6] == "" and names[7] == "" and names[8] == "",
        "and no attention biases either",
    )
    suite.check(
        names[13] == "blk.7.ffn_gate.weight",
        "and it does have a gate projection",
    )

    var qwen = tensor_names(arch_of(ARCH_QWEN3), 0)
    suite.check(
        qwen[9] == "blk.0.attn_q_norm.weight"
        and qwen[10] == "blk.0.attn_k_norm.weight",
        "a qwen3 has the two per head norms",
    )
    suite.check(qwen[6] == "", "and dropped the biases its predecessor carried")

    var qwen2 = tensor_names(arch_of(ARCH_QWEN2), 0)
    suite.check(
        qwen2[6] == "blk.0.attn_q.bias"
        and qwen2[7] == "blk.0.attn_k.bias"
        and qwen2[8] == "blk.0.attn_v.bias",
        "a qwen2 asks for a bias on each of the three projections",
    )
    suite.check(
        qwen2[9] == "",
        "and has no per head norms, which is the other way round",
    )

    var gemma = tensor_names(arch_of(ARCH_GEMMA3), 2)
    suite.check(
        gemma[1] == "blk.2.post_attention_norm.weight"
        and gemma[12] == "blk.2.post_ffw_norm.weight",
        "and a gemma3 has the two post norms",
    )
    suite.check(
        len(names) == len(qwen) and len(qwen) == len(gemma),
        "and every architecture returns the same number of slots",
    )


def test_specs(mut suite: Suite) raises:
    suite.group("architecture to block spec")

    var g = _geometry()
    var spec = block_spec(arch_of(ARCH_LLAMA), g, 0)
    suite.check(
        spec.width == 64 and spec.hidden == 128,
        "the widths come from the file and not from the table",
    )
    suite.check(
        spec.attn.heads == 4
        and spec.attn.kv_heads == 2
        and spec.attn.head_dim == 16,
        "and so do the head counts",
    )
    suite.check(
        spec.attn.group() == 2, "which makes this two query heads per key head"
    )
    suite.check(
        spec.gated and spec.act == ACT_SILU and not spec.rope.neox,
        "and the pairing and the mlp shape come from the table",
    )
    suite.check(
        spec.attn.window == 0 and spec.attn.softcap == 0.0,
        "a llama layer has no window and no cap",
    )

    var g2 = block_spec(arch_of(ARCH_GEMMA2), g, 0)
    var g2_full = block_spec(arch_of(ARCH_GEMMA2), g, 1)
    suite.check(
        g2.attn.window == 4096 and g2_full.attn.window == 0,
        (
            "and a gemma2 gets a window on one layer of each pair and not the"
            " other"
        ),
    )
    suite.check(
        _close(g2.attn.softcap, 50.0, 1e-6)
        and _close(g2_full.attn.softcap, 50.0, 1e-6),
        "while the softcap is on every layer",
    )

    # Gemma 3's two rope bases, which is the only place the layer index changes
    # something other than the window.
    var local = block_spec(arch_of(ARCH_GEMMA3), g, 0)
    var glob = block_spec(arch_of(ARCH_GEMMA3), g, 5)
    suite.check(
        _close(local.rope.base, 10000.0, 1e-3)
        and _close(glob.rope.base, 500000.0, 1e-3),
        (
            "a gemma3 windowed layer uses the table's base and a global one"
            " uses the file's"
        ),
    )


def test_rope(mut suite: Suite) raises:
    suite.group("architecture rope scaling")

    var g = _geometry()
    var llama = arch_of(ARCH_LLAMA)

    var plain = rope_spec(llama, g, 0)
    suite.check(
        plain.scale == 1.0 and plain.ext_factor == 0.0,
        "an unstated scaling type is no scaling",
    )

    var linear = g
    linear.rope_scaling = "linear"
    linear.rope_scale_factor = 4.0
    var lin = rope_spec(llama, linear, 0)
    suite.check(
        _close(lin.scale, 0.25, 1e-6) and lin.ext_factor == 0.0,
        "linear scaling divides the position by the factor",
    )

    var yarned = g
    yarned.rope_scaling = "yarn"
    yarned.rope_scale_factor = 4.0
    var y = rope_spec(llama, yarned, 0)
    suite.check(
        _close(y.scale, 0.25, 1e-6)
        and y.ext_factor == 1.0
        and y.orig_context == 8192,
        "yarn takes the trained context off the file as well as the factor",
    )
    suite.check(
        _close(y.attn_factor, 1.1386294, 1e-5),
        "and warms the logits by a tenth of the log of the factor",
    )

    var none = g
    none.rope_scaling = "none"
    suite.check(
        rope_spec(llama, none, 0).scale == 1.0,
        "a stated type of none is the same as an unstated one",
    )


def test_errors(mut suite: Suite) raises:
    suite.group("architecture errors")

    var g = _geometry()
    var llama = arch_of(ARCH_LLAMA)

    var raised = False
    try:
        _ = arch_of(ARCH_UNKNOWN)
    except:
        raised = True
    suite.check(raised, "an architecture that is not in the table is an error")

    raised = False
    try:
        _ = block_spec(llama, g, 8)
    except:
        raised = True
    suite.check(raised, "and a layer past the end of the model")

    raised = False
    var moe = g
    moe.expert_count = 8
    moe.expert_used_count = 2
    try:
        _ = block_spec(llama, moe, 0)
    except:
        raised = True
    suite.check(
        raised, "and a mixture of experts, which molla does not route yet"
    )

    raised = False
    var lopsided = g
    lopsided.value_length = 32
    try:
        _ = block_spec(llama, lopsided, 0)
    except:
        raised = True
    suite.check(raised, "and keys and values of different widths")

    raised = False
    var no_eps = g
    no_eps.epsilon = 0.0
    try:
        _ = block_spec(llama, no_eps, 0)
    except:
        raised = True
    suite.check(raised, "and a file that states no norm epsilon")

    raised = False
    var no_base = g
    no_base.rope_freq_base = 0.0
    try:
        _ = rope_spec(llama, no_base, 0)
    except:
        raised = True
    suite.check(raised, "and a file that states no rope base")

    # A factor with no type is the interesting one. It is what a converter
    # writes when it half understood the config, and silently ignoring it gives
    # a model that works at short context and falls apart at long.
    raised = False
    var contradiction = g
    contradiction.rope_scale_factor = 8.0
    try:
        _ = rope_spec(llama, contradiction, 0)
    except:
        raised = True
    suite.check(
        raised, "and a scaling factor with no scaling type to go with it"
    )

    raised = False
    var unknown = g
    unknown.rope_scaling = "longrope"
    unknown.rope_scale_factor = 4.0
    try:
        _ = rope_spec(llama, unknown, 0)
    except:
        raised = True
    suite.check(raised, "and a scaling scheme molla has not implemented")

    suite.check(
        describe(arch_of(ARCH_LLAMA)).find("adjacent") >= 0,
        "and an architecture can say what it is in one line",
    )
