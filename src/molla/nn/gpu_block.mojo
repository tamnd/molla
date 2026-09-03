"""A whole forward pass on the device, from a token to a row of logits.

The device twin of `molla.nn.block` and `molla.nn.model` together. Everything
below it already existed: `molla.nn.gpu` has the matvec and `molla.nn.gpu_ops`
has the twenty other operations, each one checked against a host reference. What
was missing was the thing that runs them in order without leaving, and until it
existed every device kernel was called with host activations either side, which
at decode shapes costs more than the arithmetic it surrounds.

So the rule this file exists to hold is one sentence. A token goes in, a row of
logits comes out, and nothing in between crosses back to the host. The residual
stream is a device vector from the embedding lookup to the final norm, the keys
and values are written where they will be read and never copied, and the only
transfer a token pays for is the logits, which somebody has to look at to pick
the next one.

## What it asks of the weights

Every weight a kernel reads has to be in a device pool and in the planar layout.
Both of those are refusals rather than fallbacks, for the reason `molla.nn.gpu`
gives at length: a device kernel handed a host address reads zeros without
faulting, so a model bound the wrong way round runs at full speed and answers
with noise. `DeviceModel.check` asks the question once when a model binds, so a
plan that left half the weights in the mapping is an error naming the tensor
rather than a token of nonsense.

The small weights are the exception and they are copied rather than pointed at.
A norm gain is a few thousand floats read every token of every layer, a bias is
one per projection, and the rope tables are a few dozen numbers. Those are
dequantized once from the host copy and uploaded into device vectors of their
own, which is why the constructors here take two bindings: one against the
mapping, which is where the values are readable, and one against the pool, which
is where the matrices are.

## Two bindings, one file

`host` and `dev` are the same GGUF bound twice. That is not a copy of the
weights, it is two lists of addresses, and the reason there are two is that a
weight in a pool has no host address and a norm gain has to be read on the host
to be uploaded. The load leaves both available: the pool is a copy of bytes that
are still in the mapping, and the mapping stays open for as long as the model is
loaded anyway.
"""

from std.math import sqrt

from max.gpu.host import DeviceContext

from molla.nn.arch import Arch
from molla.nn.block import ACT_SILU, BlockSpec, LayerWeights
from molla.nn.gpu import (
    DeviceQuantVec,
    DeviceVec,
    device_matvec_q8_into,
    device_quantize,
)
from molla.nn.gpu_ops import (
    RopeTables,
    device_add_into,
    device_add_run,
    device_attend,
    device_geglu,
    device_gelu,
    device_rms_norm,
    device_rms_norm_inplace,
    device_rms_norm_run,
    device_rope,
    device_scale_into,
    device_silu,
    device_softcap,
    device_swiglu,
    device_unpack_row,
)
from molla.nn.model import ModelWeights
from molla.nn.repack import LAYOUT_PLANAR, unpack_run
from molla.nn.tensor import WHERE_DEVICE, Buffer, Tensor


def _values(t: Tensor, n: Int) raises -> List[Float32]:
    """A small host weight as float32, whatever it was stored as.

    Norm gains and biases are f32 in every file anybody ships and this does not
    assume it, because `unpack_run` already handles the other cases and an
    assumption that holds for every file until it does not is how a model loads
    and answers strangely.
    """
    if t.elements() < n:
        raise Error(
            "wanted "
            + String(n)
            + " values from a weight that has "
            + String(t.elements())
        )
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    unpack_run(t.kind, t.layout, t.base(), 0, n, out, 0)
    return out^


def _gain(ctx: DeviceContext, t: Tensor, n: Int) raises -> DeviceVec:
    """One small host weight, copied up once and kept for the model's life.

    `copy_in` rather than `upload_run`, because the mapping route costs 1.3 GiB
    on the first call in a process and a thirty layer model calls this sixty
    times. That was most of the resident set of a device run and it is the whole
    of docs/validation/performance.md's memory section.
    """
    var values = _values(t, n)
    var out = DeviceVec(ctx, n)
    out.copy_in(values)
    return out^


def _absent(ctx: DeviceContext) raises -> DeviceVec:
    """A vector for a weight the model does not have.

    One element rather than none, because a device buffer of length zero is not
    a thing on either vendor, and nothing ever reads it: every use is behind the
    flag that says whether the weight was there.
    """
    return DeviceVec(ctx, 1)


def _resident(t: Tensor, name: String) raises:
    """The two things every matrix a device kernel reads has to be.

    Asked when the model binds rather than when the kernel launches. Both would
    be caught either way, since the matvec checks them too, but a model that has
    finished loading and then fails on the first token has spent a minute of
    somebody's time to report something that was decidable from the plan.
    """
    if not t.present():
        raise Error("the device forward pass needs " + name)
    if t.place != WHERE_DEVICE:
        raise Error(
            name
            + " is not in the device pool, so a device kernel would read it as"
            " zeros rather than as an error. Either the budget did not cover"
            " this model or the load was asked for host placement"
        )
    if t.layout != LAYOUT_PLANAR:
        raise Error(
            name
            + " is still in ggml blocks and the device kernels read the planar"
            " layout, so this model needs a repack cache beside it"
        )


struct DeviceLayer(Movable):
    """One layer's weights, split by which of them a kernel points at.

    The seven matrices stay where the load put them and are read out of the pool
    by the matvec. The nine small vectors were dequantized on the host and
    uploaded, because the norm kernel takes a gain as a device vector and a gain
    is the same few thousand floats every token of every layer.
    """

    var attn_norm: DeviceVec
    var attn_post_norm: DeviceVec
    var has_attn_post: Bool
    var ffn_norm: DeviceVec
    var ffn_post_norm: DeviceVec
    var has_ffn_post: Bool

    var q_norm: DeviceVec
    var k_norm: DeviceVec
    var has_qk_norm: Bool
    """Qwen 3 normalises each head of a query and a key before rotating it."""

    var q_bias: DeviceVec
    var k_bias: DeviceVec
    var v_bias: DeviceVec
    var has_bias: Bool
    """Qwen 2 biases all three attention projections and almost nothing since
    does."""

    var wq: Tensor
    var wk: Tensor
    var wv: Tensor
    var wo: Tensor
    var gate: Tensor
    var up: Tensor
    var down: Tensor
    var has_gate: Bool

    var rope: RopeTables
    """Built per layer rather than per model, because Gemma 3 alternates a local
    and a global rope spec and the tables are a few hundred bytes each."""

    def __init__(
        out self,
        ctx: DeviceContext,
        spec: BlockSpec,
        host: LayerWeights,
        dev: LayerWeights,
        factors: List[Float32],
        use_factors: Bool,
    ) raises:
        _resident(dev.wq, "the query projection")
        _resident(dev.wk, "the key projection")
        _resident(dev.wv, "the value projection")
        _resident(dev.wo, "the attention output projection")
        _resident(dev.up, "the mlp up projection")
        _resident(dev.down, "the mlp down projection")
        if spec.gated:
            _resident(dev.gate, "the mlp gate projection")

        self.attn_norm = _gain(ctx, host.attn_norm, spec.width)
        self.ffn_norm = _gain(ctx, host.ffn_norm, spec.width)
        self.has_attn_post = host.attn_post_norm.present()
        self.attn_post_norm = _gain(
            ctx, host.attn_post_norm, spec.width
        ) if self.has_attn_post else _absent(ctx)
        self.has_ffn_post = host.ffn_post_norm.present()
        self.ffn_post_norm = _gain(
            ctx, host.ffn_post_norm, spec.width
        ) if self.has_ffn_post else _absent(ctx)

        self.has_qk_norm = host.q_norm.present()
        var head_dim = spec.attn.head_dim
        self.q_norm = _gain(
            ctx, host.q_norm, head_dim
        ) if self.has_qk_norm else _absent(ctx)
        self.k_norm = _gain(
            ctx, host.k_norm, head_dim
        ) if self.has_qk_norm else _absent(ctx)

        self.has_bias = spec.qkv_bias
        self.q_bias = _gain(
            ctx, host.q_bias, spec.q_width()
        ) if self.has_bias else _absent(ctx)
        self.k_bias = _gain(
            ctx, host.k_bias, spec.kv_width()
        ) if self.has_bias else _absent(ctx)
        self.v_bias = _gain(
            ctx, host.v_bias, spec.kv_width()
        ) if self.has_bias else _absent(ctx)

        self.wq = dev.wq
        self.wk = dev.wk
        self.wv = dev.wv
        self.wo = dev.wo
        self.gate = dev.gate
        self.up = dev.up
        self.down = dev.down
        self.has_gate = spec.gated
        self.rope = RopeTables(ctx, spec.rope, factors, use_factors)


struct DeviceModel(Movable):
    """A whole model on the device, ready to be handed a token.

    Held by the session for as long as the model is loaded, and built once. The
    two bindings it is built from are addresses rather than bytes, so this owns
    the small vectors it uploaded and nothing else.
    """

    var arch: Arch
    var specs: List[BlockSpec]
    var layers: List[DeviceLayer]

    var output_norm: DeviceVec
    var embedding: Tensor
    var head: Tensor
    """`output.weight`, or the embedding again when the head is tied. Resolved
    here rather than per token, because the tie is a property of the file."""

    def __init__(
        out self,
        ctx: DeviceContext,
        arch: Arch,
        specs: List[BlockSpec],
        host_model: ModelWeights,
        dev_model: ModelWeights,
        host_layers: List[LayerWeights],
        dev_layers: List[LayerWeights],
        factors: List[Float32],
    ) raises:
        var count = len(specs)
        if count == 0:
            raise Error("a model with no layers is not a model")
        if len(host_layers) != count or len(dev_layers) != count:
            raise Error(
                "the model has "
                + String(count)
                + " layer specs and "
                + String(len(dev_layers))
                + " sets of weights"
            )
        _resident(dev_model.embedding, "the token embedding")
        _resident(dev_model.head(), "the output head")

        var use_factors = len(factors) > 0
        self.layers = List[DeviceLayer]()
        for i in range(count):
            self.layers.append(
                DeviceLayer(
                    ctx,
                    specs[i],
                    host_layers[i],
                    dev_layers[i],
                    factors,
                    use_factors,
                )
            )
        self.output_norm = _gain(ctx, host_model.output_norm, specs[0].width)
        self.embedding = dev_model.embedding
        self.head = dev_model.head()
        self.arch = arch
        self.specs = specs.copy()

    def block_count(self) -> Int:
        return len(self.specs)

    def width(self) -> Int:
        return self.specs[0].width

    def vocab(self) -> Int:
        return self.head.rows


def snapshot(ctx: DeviceContext, x: DeviceVec) raises -> Buffer:
    """One device vector brought back to the host, for a trace.

    The synchronize is not optional and it is not free. Everything queued for
    this token is still on the stream when this is called, so reading the vector
    without waiting would read whichever kernels happened to have finished,
    which is a trace that is right most of the time. That is the cost of tracing
    and it is why tracing is off unless something asked for it.
    """
    ctx.synchronize()
    var out = Buffer(x.elements())
    x.download(out)
    return out^


struct DeviceScratch(Movable):
    """The intermediates one layer needs, in device memory, sized once.

    The counterpart of `molla.nn.block.Scratch` field for field, plus the logits,
    which live here rather than with the session for the same reason everything
    else does: a decode step touches no allocator, and a buffer allocated per
    token is a device allocation per token.
    """

    var norm: DeviceVec
    var q: DeviceVec
    var heads_out: DeviceVec
    var projected: DeviceVec
    var gate: DeviceVec
    var up: DeviceVec
    var scores: DeviceVec
    var logits: DeviceVec

    var qnorm: DeviceQuantVec
    var qheads: DeviceQuantVec
    var qhidden: DeviceQuantVec
    """The three activation vectors a matvec reads, quantized to a byte a value.

    Every matvec on a token's path reads one of these. The attention norm feeds
    the query, key and value projections and the final norm feeds the head, the
    attention output feeds the output projection, and the gate or the up
    projection feeds the down projection. Sized once here for the same reason
    everything else in this struct is, which is that a decode step touches no
    allocator.
    """

    var tracing: Bool
    """Whether `device_forward` records the residual stream as it goes. Off.

    On, it costs a synchronize and a download per layer, which is exactly the
    thing this file exists to avoid. That is the right trade: the trace is what
    the logit corpus compares layer by layer, and a trace that could only be
    taken by running a different code path would be checking a different program.
    """

    var trace: List[Float32]

    def __init__(
        out self, ctx: DeviceContext, spec: BlockSpec, context: Int, vocab: Int
    ) raises:
        if context <= 0:
            raise Error("a layer needs room for at least one position")
        if vocab <= 0:
            raise Error("a model needs a vocabulary to write logits into")
        self.norm = DeviceVec(ctx, spec.width)
        self.q = DeviceVec(ctx, spec.q_width())
        self.heads_out = DeviceVec(ctx, spec.q_width())
        self.projected = DeviceVec(ctx, spec.width)
        self.gate = DeviceVec(ctx, spec.hidden)
        self.up = DeviceVec(ctx, spec.hidden)
        self.scores = DeviceVec(ctx, spec.attn.heads * context)
        self.logits = DeviceVec(ctx, vocab)
        self.qnorm = DeviceQuantVec(ctx, spec.width)
        self.qheads = DeviceQuantVec(ctx, spec.q_width())
        self.qhidden = DeviceQuantVec(ctx, spec.hidden)
        self.tracing = False
        self.trace = List[Float32]()

    def record(mut self, ctx: DeviceContext, x: DeviceVec) raises:
        """Append one snapshot of the residual stream, if anybody asked."""
        if not self.tracing:
            return
        self.take(snapshot(ctx, x))

    def take(mut self, snap: Buffer):
        """Append a snapshot that has already been brought back.

        Separate from `record` for the one caller that wants to record something
        that lives in this same struct. Reading `s.norm` and writing `s` in one
        call is an aliasing error the compiler refuses, so the final norm is
        brought back first and handed here, which is what the host version does
        for the same reason.
        """
        for i in range(snap.elements()):
            self.trace.append(snap.data[i])

    def forget(mut self):
        self.trace.clear()

    def snapshots(self, width: Int) raises -> Int:
        if width <= 0:
            raise Error("a snapshot cannot be zero wide")
        if len(self.trace) % width != 0:
            raise Error("the trace is not a whole number of snapshots")
        return len(self.trace) // width


def device_attention(
    ctx: DeviceContext,
    spec: BlockSpec,
    w: DeviceLayer,
    mut x: DeviceVec,
    mut s: DeviceScratch,
    mut keys: DeviceVec,
    mut values: DeviceVec,
    slot: Int,
    pos: Int,
) raises:
    """The attention sublayer, in place on the residual stream.

    The same eleven steps in the same order as `molla.nn.block.attention_layer`,
    and the order is the part that matters rather than the kernels: the bias
    before the per head norm because the bias is part of the projection, and the
    per head norm before rope because a rotation of a normalised vector is still
    normalised and the other way round is not.
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
    if keys.elements() < at + kv_width or values.elements() < at + kv_width:
        raise Error(
            "the cache has no room for slot "
            + String(slot)
            + ", which needs "
            + String(at + kv_width)
            + " values"
        )

    device_rms_norm(ctx, x, w.attn_norm, s.norm, spec.eps)
    device_quantize(ctx, s.norm, s.qnorm)
    device_matvec_q8_into(ctx, w.wq, s.qnorm, s.q)

    # Straight into the cache rather than into scratch and then a copy, which is
    # what the offset on the matvec is for. The cache is where they are needed
    # and this is the only place they are written.
    device_matvec_q8_into(ctx, w.wk, s.qnorm, keys, at)
    device_matvec_q8_into(ctx, w.wv, s.qnorm, values, at)

    if w.has_bias:
        device_add_run(ctx, s.q, 0, w.q_bias, spec.q_width())
        device_add_run(ctx, keys, at, w.k_bias, kv_width)
        device_add_run(ctx, values, at, w.v_bias, kv_width)

    if w.has_qk_norm:
        var head_dim = spec.attn.head_dim
        for h in range(spec.attn.heads):
            device_rms_norm_run(
                ctx, s.q, h * head_dim, head_dim, w.q_norm, spec.eps
            )
        for h in range(spec.attn.kv_heads):
            device_rms_norm_run(
                ctx, keys, at + h * head_dim, head_dim, w.k_norm, spec.eps
            )

    device_rope(
        ctx,
        spec.rope,
        s.q,
        0,
        spec.attn.heads,
        spec.attn.head_dim,
        pos,
        w.rope,
    )
    device_rope(
        ctx,
        spec.rope,
        keys,
        at,
        spec.attn.kv_heads,
        spec.attn.head_dim,
        pos,
        w.rope,
    )

    device_attend(
        ctx,
        spec.attn,
        s.q,
        keys,
        values,
        slot + 1,
        pos,
        s.heads_out,
        s.scores,
    )
    device_quantize(ctx, s.heads_out, s.qheads)
    device_matvec_q8_into(ctx, w.wo, s.qheads, s.projected)

    if w.has_attn_post:
        device_rms_norm_inplace(ctx, s.projected, w.attn_post_norm, spec.eps)
    device_add_into(ctx, x, s.projected)


def device_mlp(
    ctx: DeviceContext,
    spec: BlockSpec,
    w: DeviceLayer,
    mut x: DeviceVec,
    mut s: DeviceScratch,
) raises:
    """The mlp sublayer, in place on the residual stream."""
    if x.elements() != spec.width:
        raise Error(
            "the residual stream is "
            + String(x.elements())
            + " wide where the layer wants "
            + String(spec.width)
        )

    device_rms_norm(ctx, x, w.ffn_norm, s.norm, spec.eps)
    device_quantize(ctx, s.norm, s.qnorm)
    device_matvec_q8_into(ctx, w.up, s.qnorm, s.up)

    if w.has_gate:
        device_matvec_q8_into(ctx, w.gate, s.qnorm, s.gate)
        if spec.act == ACT_SILU:
            device_swiglu(ctx, s.gate, s.up)
        else:
            device_geglu(ctx, s.gate, s.up)
        device_quantize(ctx, s.gate, s.qhidden)
    else:
        if spec.act == ACT_SILU:
            device_silu(ctx, s.up)
        else:
            device_gelu(ctx, s.up)
        device_quantize(ctx, s.up, s.qhidden)
    device_matvec_q8_into(ctx, w.down, s.qhidden, s.projected)

    if w.has_ffn_post:
        device_rms_norm_inplace(ctx, s.projected, w.ffn_post_norm, spec.eps)
    device_add_into(ctx, x, s.projected)


def device_layer(
    ctx: DeviceContext,
    spec: BlockSpec,
    w: DeviceLayer,
    mut x: DeviceVec,
    mut s: DeviceScratch,
    mut keys: DeviceVec,
    mut values: DeviceVec,
    slot: Int,
    pos: Int,
) raises:
    """Both sublayers, which is one decoder layer."""
    device_attention(ctx, spec, w, x, s, keys, values, slot, pos)
    device_mlp(ctx, spec, w, x, s)


def device_forward(
    ctx: DeviceContext,
    m: DeviceModel,
    mut s: DeviceScratch,
    mut x: DeviceVec,
    token: Int,
    pos: Int,
    slot: Int,
    mut keys: List[DeviceVec],
    mut values: List[DeviceVec],
) raises:
    """One token through the whole stack, leaving its logits on the device.

    Queued and not synchronized. Everything here is one stream of launches with
    no wait between them, which is the thing that makes a token one command
    buffer rather than several hundred, and the caller waits once when it wants
    to read the logits.

    `pos` and `slot` are two arguments for the reason the host version gives:
    they are the same number until something evicts, and the day they stop being
    the same is the day a single argument becomes a bug in two places.
    """
    var count = m.block_count()
    if len(keys) != count or len(values) != count:
        raise Error(
            "the cache has room for "
            + String(len(keys))
            + " layers and the model has "
            + String(count)
        )
    if pos < 0:
        raise Error("a position cannot be negative")
    if x.elements() != m.width():
        raise Error(
            "the residual stream is "
            + String(x.elements())
            + " wide where the model is "
            + String(m.width())
        )

    device_unpack_row(ctx, m.embedding, token, x)
    if m.arch.scale_embedding:
        device_scale_into(ctx, x, Float32(sqrt(Float64(m.width()))))

    # The same numbering the host trace uses, so snapshot k is the residual
    # stream layer k was handed and a comparison can say which layer a
    # divergence started in.
    s.record(ctx, x)
    for i in range(count):
        device_layer(
            ctx, m.specs[i], m.layers[i], x, s, keys[i], values[i], slot, pos
        )
        s.record(ctx, x)

    device_rms_norm(ctx, x, m.output_norm, s.norm, m.specs[0].eps)
    device_quantize(ctx, s.norm, s.qnorm)
    device_matvec_q8_into(ctx, m.head, s.qnorm, s.logits)
    if m.arch.final_softcap > 0:
        device_softcap(ctx, s.logits, m.arch.final_softcap, m.vocab())
    if s.tracing:
        # One more after the final norm, so the output head sits between the
        # last snapshot and the logits with nothing else in it. Brought back
        # first and appended second, because `s.norm` cannot be read and `s`
        # written in one call.
        var normed = snapshot(ctx, s.norm)
        s.take(normed)
