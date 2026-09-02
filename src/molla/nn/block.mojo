"""One transformer layer, with the knobs every architecture turns.

A dense decoder layer is two sublayers and two residual adds, and every model
molla is going to run is that shape with different answers to about a dozen
questions. Rather than a function per architecture there is one function here
and a spec that records which answers this model gave. The questions are few
enough to write down.

Is there a norm on a sublayer's output as well as on its input. Gemma 2 and
Gemma 3 have both and almost nothing else does.

Are the query and key normalised per head before rope. Qwen 3 does, and it is a
real rms norm over `head_dim` with its own weight rather than a rescale.

Does the mlp gate. Llama, Qwen and Gemma compute two projections and multiply
one through an activation into the other. The older shape is one projection and
an activation. Which activation is a separate question from whether it gates,
so it is a separate field.

Everything else, head counts, head width, rope base and scaling, sliding
windows, sink tokens and softcaps, is already a field on `AttnSpec` or
`RopeSpec` and is not repeated here.

Nothing in this file allocates. A decode step runs it once per layer per token,
an 8B has thirty two layers, and an allocation per intermediate would be a
thousand allocations per token for numbers that are dead within the
microsecond. The signatures are long because the caller passes every
intermediate in, and that is the trade rather than an oversight.
"""

from molla.nn.attention import AttnSpec, attend
from molla.nn.kernel import (
    add_into,
    gelu,
    matvec,
    matvec_into,
    rms_norm,
    rms_norm_at,
    silu,
    swiglu,
)
from molla.nn.rope import RopeSpec, rotate_heads, rotate_run
from molla.nn.tensor import Buffer, Tensor

comptime ACT_SILU = 0
"""What Llama, Qwen and Gemma 3 gate with."""

comptime ACT_GELU = 1
"""The tanh approximation, which Gemma 2 uses."""


def activation_name(act: Int) -> String:
    if act == ACT_SILU:
        return "silu"
    if act == ACT_GELU:
        return "gelu"
    return "unknown"


struct BlockSpec(Copyable, ImplicitlyCopyable, Movable):
    """What one layer does, as opposed to what it was trained to say."""

    var attn: AttnSpec
    var rope: RopeSpec

    var width: Int
    """The residual stream, which both sublayers read and write."""

    var hidden: Int
    """The mlp's inner width."""

    var eps: Float32
    """The epsilon inside every norm in the layer. One number rather than one
    per norm, because no model in the wild uses two."""

    var gated: Bool
    """True computes `act(gate(x)) * up(x)`, false computes `act(up(x))`."""

    var act: Int
    """`ACT_SILU` or `ACT_GELU`."""

    def __init__(
        out self,
        attn: AttnSpec,
        rope: RopeSpec,
        width: Int,
        hidden: Int,
        eps: Float32,
    ) raises:
        """A gated silu layer, which is what almost everything is."""
        if width <= 0 or hidden <= 0:
            raise Error("a layer needs a positive width and hidden size")
        if eps <= 0:
            raise Error("a norm epsilon has to be positive")
        self.attn = attn
        self.rope = rope
        self.width = width
        self.hidden = hidden
        self.eps = eps
        self.gated = True
        self.act = ACT_SILU

    def q_width(self) -> Int:
        return self.attn.heads * self.attn.head_dim

    def kv_width(self) -> Int:
        return self.attn.kv_heads * self.attn.head_dim


struct LayerWeights(Copyable, ImplicitlyCopyable, Movable):
    """Every tensor a layer can have. Absent ones are `Tensor.none()`."""

    var attn_norm: Tensor
    var attn_post_norm: Tensor
    var wq: Tensor
    var wk: Tensor
    var wv: Tensor
    var wo: Tensor
    var q_norm: Tensor
    var k_norm: Tensor
    var ffn_norm: Tensor
    var ffn_post_norm: Tensor
    var gate: Tensor
    var up: Tensor
    var down: Tensor

    def __init__(out self):
        """Everything absent. A loader fills in what the file actually has."""
        var nothing = Tensor.none()
        self.attn_norm = nothing
        self.attn_post_norm = nothing
        self.wq = nothing
        self.wk = nothing
        self.wv = nothing
        self.wo = nothing
        self.q_norm = nothing
        self.k_norm = nothing
        self.ffn_norm = nothing
        self.ffn_post_norm = nothing
        self.gate = nothing
        self.up = nothing
        self.down = nothing

    def check(self, spec: BlockSpec) raises:
        """Everything that has to be there, and every shape that has to agree.

        Run once when a model is loaded rather than once per token. A missing
        projection should be a file molla refuses, not a read from address zero
        on the first token. A projection with the wrong number of rows is worse
        than either, because it produces output instead of an error.
        """
        if not self.attn_norm.present():
            raise Error("a layer needs a norm before attention")
        if not self.wq.present() or not self.wk.present():
            raise Error("a layer needs query and key projections")
        if not self.wv.present() or not self.wo.present():
            raise Error("a layer needs value and output projections")
        if not self.ffn_norm.present():
            raise Error("a layer needs a norm before the mlp")
        if not self.up.present() or not self.down.present():
            raise Error("a layer needs an mlp")
        if spec.gated and not self.gate.present():
            raise Error("a gated mlp needs a gate projection")
        if not spec.gated and self.gate.present():
            raise Error(
                "a non gated mlp has a gate projection, which means the"
                " architecture table and the file disagree"
            )
        if self.q_norm.present() != self.k_norm.present():
            raise Error(
                "a layer has one of the query and key norms and not the other"
            )

        _shape(self.attn_norm, spec.width, 1, "attn_norm")
        _shape(self.wq, spec.width, spec.q_width(), "wq")
        _shape(self.wk, spec.width, spec.kv_width(), "wk")
        _shape(self.wv, spec.width, spec.kv_width(), "wv")
        _shape(self.wo, spec.q_width(), spec.width, "wo")
        _shape(self.ffn_norm, spec.width, 1, "ffn_norm")
        _shape(self.up, spec.width, spec.hidden, "up")
        _shape(self.down, spec.hidden, spec.width, "down")
        if spec.gated:
            _shape(self.gate, spec.width, spec.hidden, "gate")
        if self.attn_post_norm.present():
            _shape(self.attn_post_norm, spec.width, 1, "attn_post_norm")
        if self.ffn_post_norm.present():
            _shape(self.ffn_post_norm, spec.width, 1, "ffn_post_norm")
        if self.q_norm.present():
            _shape(self.q_norm, spec.attn.head_dim, 1, "q_norm")
            _shape(self.k_norm, spec.attn.head_dim, 1, "k_norm")


def _shape(t: Tensor, cols: Int, rows: Int, name: String) raises:
    if t.cols != cols or t.rows != rows:
        raise Error(
            name
            + " is "
            + String(t.cols)
            + " by "
            + String(t.rows)
            + " where the layer wants "
            + String(cols)
            + " by "
            + String(rows)
        )


struct Scratch(Movable):
    """The intermediates one layer needs, sized once and reused every token.

    Held together because a caller that has to declare nine buffers in the
    right sizes will eventually declare one of them wrong, and a buffer that is
    the wrong size here is a shape error rather than a wrong answer only
    because every kernel checks. Passed field by field into the layer
    functions, since two fields of one struct cannot both be borrowed and
    mutated in the same call.
    """

    var norm: Buffer
    var q: Buffer
    var heads_out: Buffer
    var projected: Buffer
    var gate: Buffer
    var up: Buffer
    var scores: List[Float32]

    def __init__(out self, spec: BlockSpec, context: Int) raises:
        if context <= 0:
            raise Error("a layer needs room for at least one position")
        self.norm = Buffer(spec.width)
        self.q = Buffer(spec.q_width())
        self.heads_out = Buffer(spec.q_width())
        self.projected = Buffer(spec.width)
        self.gate = Buffer(spec.hidden)
        self.up = Buffer(spec.hidden)
        self.scores = List[Float32]()
        for _ in range(context):
            self.scores.append(0.0)


def attention_layer(
    spec: BlockSpec,
    w: LayerWeights,
    mut x: Buffer,
    mut s: Scratch,
    mut keys: List[Float32],
    mut values: List[Float32],
    slot: Int,
    pos: Int,
    factors: List[Float32],
    use_factors: Bool,
) raises:
    """The attention sublayer, in place on the residual stream.

    `keys` and `values` are the layer's cache. This writes its own key and
    value into slot `slot` and then attends over slots zero through `slot`,
    which is what a decode step wants: the cache owns the storage and knows how
    long it is, and the layer knows the layout inside it. Where that storage
    comes from and when it gets reused is #27.
    """
    if x.elements() != spec.width:
        raise Error(
            "the residual stream is "
            + String(x.elements())
            + " wide where the layer wants "
            + String(spec.width)
        )
    if slot < 0:
        raise Error("a cache slot cannot be negative")

    var kv_width = spec.kv_width()
    var at = slot * kv_width
    if len(keys) < at + kv_width or len(values) < at + kv_width:
        raise Error(
            "the cache has no room for slot "
            + String(slot)
            + ", which needs "
            + String(at + kv_width)
            + " values"
        )

    rms_norm(s.norm, x, w.attn_norm, spec.eps)
    matvec(w.wq, s.norm, s.q)

    # k and v go straight into the cache rather than into scratch and then a
    # copy. The cache is where they are needed and this is the only place they
    # are written, so a temporary would be a copy of a copy.
    matvec_into(w.wk, s.norm.data, 0, keys, at)
    matvec_into(w.wv, s.norm.data, 0, values, at)

    # Qwen 3 normalises each head of q and k before rotating it. This is before
    # rope rather than after, because rope is a rotation and a rotation of a
    # normalised vector is still normalised while the other order is not.
    if w.q_norm.present():
        for h in range(spec.attn.heads):
            rms_norm_at(
                s.q.data,
                h * spec.attn.head_dim,
                spec.attn.head_dim,
                w.q_norm,
                spec.eps,
            )
        for h in range(spec.attn.kv_heads):
            rms_norm_at(
                keys,
                at + h * spec.attn.head_dim,
                spec.attn.head_dim,
                w.k_norm,
                spec.eps,
            )

    rotate_heads(
        spec.rope,
        s.q,
        spec.attn.heads,
        spec.attn.head_dim,
        pos,
        factors,
        use_factors,
    )
    rotate_run(
        spec.rope,
        keys,
        at,
        spec.attn.kv_heads,
        spec.attn.head_dim,
        pos,
        factors,
        use_factors,
    )

    attend(spec.attn, s.q, keys, values, slot + 1, pos, s.heads_out, s.scores)
    matvec(w.wo, s.heads_out, s.projected)

    # Gemma normalises a sublayer's output before it joins the stream, which is
    # a different thing from normalising the input and both can be present.
    if w.attn_post_norm.present():
        rms_norm_at(s.projected.data, 0, spec.width, w.attn_post_norm, spec.eps)

    add_into(x, s.projected)


def mlp_layer(
    spec: BlockSpec, w: LayerWeights, mut x: Buffer, mut s: Scratch
) raises:
    """The mlp sublayer, in place on the residual stream."""
    if x.elements() != spec.width:
        raise Error(
            "the residual stream is "
            + String(x.elements())
            + " wide where the layer wants "
            + String(spec.width)
        )

    rms_norm(s.norm, x, w.ffn_norm, spec.eps)
    matvec(w.up, s.norm, s.up)

    if spec.gated:
        matvec(w.gate, s.norm, s.gate)
        if spec.act == ACT_SILU:
            swiglu(s.gate, s.up)
        else:
            for i in range(spec.hidden):
                s.gate.data[i] = gelu(s.gate.data[i]) * s.up.data[i]
        matvec(w.down, s.gate, s.projected)
    else:
        for i in range(spec.hidden):
            if spec.act == ACT_SILU:
                s.up.data[i] = silu(s.up.data[i])
            else:
                s.up.data[i] = gelu(s.up.data[i])
        matvec(w.down, s.up, s.projected)

    if w.ffn_post_norm.present():
        rms_norm_at(s.projected.data, 0, spec.width, w.ffn_post_norm, spec.eps)

    add_into(x, s.projected)


def layer(
    spec: BlockSpec,
    w: LayerWeights,
    mut x: Buffer,
    mut s: Scratch,
    mut keys: List[Float32],
    mut values: List[Float32],
    slot: Int,
    pos: Int,
    factors: List[Float32],
    use_factors: Bool,
) raises:
    """Both sublayers, which is one decoder layer."""
    attention_layer(
        spec, w, x, s, keys, values, slot, pos, factors, use_factors
    )
    mlp_layer(spec, w, x, s)
