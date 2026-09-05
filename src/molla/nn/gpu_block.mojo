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
from molla.nn.block import ACT_GELU, ACT_SILU, BlockSpec, LayerWeights
from molla.nn.gpu import (
    ACT_BIT,
    EPI_ADD,
    EPI_BIAS,
    EPI_GLU,
    EPI_HALF,
    EPI_NONE,
    MM_GROUPS,
    PREFILL_CHUNK,
    SPAN,
    DeviceHalf,
    DeviceVec,
    device_matmul_into,
    device_matmul_into_half,
    device_matvec_into,
    device_matvec_into_half,
)
from molla.nn.gpu_fused import (
    OP_ACT,
    OP_ADD,
    OP_ATTEND,
    OP_MATVEC,
    OP_NORM,
    OP_ROPE,
    R_ATTN,
    R_COLS,
    R_DIM,
    R_EPI,
    R_EPS,
    R_EXT,
    R_G,
    R_GROUP,
    R_H,
    R_HEAD_DIM,
    R_HEADS,
    R_HIGH,
    R_KIND,
    R_KV,
    R_LOW,
    R_N,
    R_ROW,
    R_ROWS,
    R_RUNS,
    R_SCALE,
    R_SINKS,
    R_SOFTCAP,
    R_SPLIT,
    R_STRIDE,
    R_W,
    R_WINDOW,
    SPACE_ARENA,
    SPACE_KEYS,
    SPACE_RESID,
    SPACE_VALS,
    SPACE_WORK,
    FusedPlan,
    StepPlan,
    WorkPlan,
    fused_selftest,
    launch_fused,
    quant_kind,
    rope_kind,
)
from molla.nn.gpu_ops import (
    RopeTables,
    device_add_into,
    device_add_run,
    attend_partials,
    device_attend,
    device_gelu,
    device_rms_norm,
    device_rms_norm_half,
    device_rms_norm_inplace,
    device_rms_norm_run,
    device_rope,
    device_rope_half,
    device_scale_into,
    device_silu,
    device_softcap,
    device_unpack_rows,
)
from molla.nn.model import ModelWeights
from molla.nn.repack import LAYOUT_PLANAR, unpack_run
from molla.nn.rope import RopeSpec, corr_range, step_table
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


struct LayerArena(Movable):
    """One layer's small weights, concatenated into a single device vector.

    The fused kernel names an operand as a base and an offset rather than as a
    pointer, because a pointer rebuilt from an integer address loses its address
    space on Metal, so the ten or eleven vectors a layer reads have to be one
    vector for a record to be able to point at any of them. See
    `molla.nn.gpu_fused`.

    It is built beside the individual gains rather than instead of them, which
    duplicates about thirty kilobytes a layer. The unfused kernels take a gain
    as a `DeviceVec` and read its length, so removing the duplication means
    changing those signatures to a pointer and a width, and that is stage two's
    work rather than a change to make while the two paths still have to agree
    with each other.
    """

    var vec: DeviceVec
    var attn_norm: Int
    var ffn_norm: Int
    var attn_post: Int
    var ffn_post: Int
    var q_norm: Int
    var k_norm: Int
    var q_bias: Int
    var k_bias: Int
    var v_bias: Int
    var steps: Int
    var factors: Int

    def __init__(
        out self,
        ctx: DeviceContext,
        spec: BlockSpec,
        host: LayerWeights,
        rope: RopeSpec,
        factors: List[Float32],
        use_factors: Bool,
    ) raises:
        var all = List[Float32]()
        var head_dim = spec.attn.head_dim
        self.attn_norm = _append(all, _values(host.attn_norm, spec.width))
        self.ffn_norm = _append(all, _values(host.ffn_norm, spec.width))
        self.attn_post = len(all)
        if host.attn_post_norm.present():
            self.attn_post = _append(
                all, _values(host.attn_post_norm, spec.width)
            )
        self.ffn_post = len(all)
        if host.ffn_post_norm.present():
            self.ffn_post = _append(
                all, _values(host.ffn_post_norm, spec.width)
            )
        self.q_norm = len(all)
        self.k_norm = len(all)
        if host.q_norm.present():
            self.q_norm = _append(all, _values(host.q_norm, head_dim))
            self.k_norm = _append(all, _values(host.k_norm, head_dim))
        self.q_bias = len(all)
        self.k_bias = len(all)
        self.v_bias = len(all)
        if spec.qkv_bias:
            self.q_bias = _append(all, _values(host.q_bias, spec.q_width()))
            self.k_bias = _append(all, _values(host.k_bias, spec.kv_width()))
            self.v_bias = _append(all, _values(host.v_bias, spec.kv_width()))
        self.steps = _append(all, step_table(rope))
        self.factors = len(all)
        if use_factors:
            var pairs = rope.dim // 2
            var take = List[Float32]()
            for i in range(pairs):
                take.append(factors[i])
            self.factors = _append(all, take)
        # Never empty and never zero length, because a device buffer of length
        # zero is not a thing on either vendor and a model with no biases and no
        # post norms would otherwise reach one.
        if len(all) == 0:
            all.append(Float32(0))
        self.vec = DeviceVec(ctx, len(all))
        self.vec.copy_in(all)


def _append(mut all: List[Float32], values: List[Float32]) -> Int:
    """Put one small weight at the end of the arena and say where it went."""
    var at = len(all)
    for i in range(len(values)):
        all.append(values[i])
    return at


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

    var arena: LayerArena
    """The same small weights again, in one vector, for the fused kernel."""

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
        self.arena = LayerArena(
            ctx, spec, host, spec.rope, factors, use_factors
        )


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
    var partials: DeviceVec
    """Room for attention to cut the context up and join it back.

    Sized by `attend_partials` at the largest context this scratch is for, which
    is what decides how many pieces it can be cut into. See `ATTEND_BLOCKS`.
    """

    var logits: DeviceVec
    var ids: DeviceVec
    """The token indices this pass is looking up, one float each.

    A device vector because the embedding lookup reads its rows from the device
    now that a prefill chunk looks up many at once, and float32 because every
    vocabulary is exact in one. See `device_unpack_rows`.
    """

    var chunk: Int
    """How many tokens this scratch is sized for. One, for a decode."""

    var tracing: Bool
    """Whether `device_forward` records the residual stream as it goes. Off.

    On, it costs a synchronize and a download per layer, which is exactly the
    thing this file exists to avoid. That is the right trade: the trace is what
    the logit corpus compares layer by layer, and a trace that could only be
    taken by running a different code path would be checking a different program.
    """

    var trace: List[Float32]

    def __init__(
        out self,
        ctx: DeviceContext,
        spec: BlockSpec,
        context: Int,
        vocab: Int,
        chunk: Int = 1,
    ) raises:
        """Sized for `chunk` tokens at once, which is one unless this is prefill.

        Every field except the logits scales with the chunk, and the scores
        scale with the chunk and the context together, which is what decides
        how large a chunk is worth having. See
        [docs/validation/prefill.md](../../../docs/validation/prefill.md).
        """
        if context <= 0:
            raise Error("a layer needs room for at least one position")
        if vocab <= 0:
            raise Error("a model needs a vocabulary to write logits into")
        if chunk <= 0:
            raise Error("a pass has to carry at least one token")
        # Everything a matmul reads is rounded up to a whole block of tokens,
        # because the dead lanes of a short chunk read past its last token
        # rather than clamping onto it. See `planar_matmul_kernel`. At the
        # shipped chunk this is exact and costs nothing, and it is only a
        # caller asking for a smaller chunk that pays for the rounding.
        var wide = chunk
        if chunk > 1:
            var block = SPAN * MM_GROUPS
            wide = (chunk + block - 1) // block * block
        self.norm = DeviceVec(ctx, wide * spec.width)
        self.q = DeviceVec(ctx, wide * spec.q_width())
        self.heads_out = DeviceVec(ctx, wide * spec.q_width())
        self.projected = DeviceVec(ctx, wide * spec.width)
        self.gate = DeviceVec(ctx, wide * spec.hidden)
        self.up = DeviceVec(ctx, wide * spec.hidden)
        self.scores = DeviceVec(ctx, chunk * spec.attn.heads * context)
        self.partials = DeviceVec(
            ctx, attend_partials(spec.attn, chunk, context)
        )
        self.logits = DeviceVec(ctx, vocab)
        self.ids = DeviceVec(ctx, chunk)
        self.chunk = chunk
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


def _project(
    ctx: DeviceContext,
    w: Tensor,
    x: DeviceVec,
    mut out: DeviceVec,
    tokens: Int,
    at: Int = 0,
    epi: Int = EPI_NONE,
    aux: Optional[Pointer[Float32, MutAnyOrigin]] = None,
) raises:
    """One token through the matvec, more than one through the matmul.

    The two kernels compute the same thing and neither is a good substitute for
    the other. The matmul carries `SPAN` tokens in a block whether they exist or
    not, so at one token it would do eight dot products to keep one and hold
    eight times the shared memory doing it. The matvec launches once a token, so
    at a hundred tokens it would read the weights a hundred times. The dividing
    line is not delicate and it is at one.
    """
    if tokens == 1:
        device_matvec_into(ctx, w, x, out, at, epi, aux)
    else:
        device_matmul_into(ctx, w, x, out, tokens, at, epi, aux)


def _project_half(
    ctx: DeviceContext,
    w: Tensor,
    x: DeviceVec,
    mut out: DeviceHalf,
    tokens: Int,
    at: Int = 0,
    epi: Int = EPI_NONE,
    aux: Optional[Pointer[Float32, MutAnyOrigin]] = None,
) raises:
    """`_project` with the key or value cache as its output.

    The same two kernels and the same dividing line between them. What the cache
    changes is the last store of a row and nothing above it, so the two
    projections that write one come through here and the other five do not.
    """
    if tokens == 1:
        device_matvec_into_half(ctx, w, x, out, at, epi, aux)
    else:
        device_matmul_into_half(ctx, w, x, out, tokens, at, epi, aux)


def device_attention(
    ctx: DeviceContext,
    spec: BlockSpec,
    w: DeviceLayer,
    mut x: DeviceVec,
    mut s: DeviceScratch,
    mut keys: DeviceHalf,
    mut values: DeviceHalf,
    slot: Int,
    pos: Int,
    tokens: Int = 1,
) raises:
    """The attention sublayer, in place on the residual stream.

    `tokens` of them at once, occupying cache slots `slot` through
    `slot + tokens - 1` at positions `pos` through `pos + tokens - 1`. Every
    kernel below takes the count and none of them branch on it, because a chunk
    of one is the geometry a decode already had.

    The same eleven steps in the same order as `molla.nn.block.attention_layer`,
    and the order is the part that matters rather than the kernels: the bias
    before the per head norm because the bias is part of the projection, and the
    per head norm before rope because a rotation of a normalised vector is still
    normalised and the other way round is not.
    """
    if tokens < 1:
        raise Error("a layer has to be given at least one token")
    if x.elements() < tokens * spec.width:
        raise Error(
            "the residual stream is "
            + String(x.elements())
            + " wide where the layer wants "
            + String(tokens * spec.width)
        )
    if slot < 0:
        raise Error("a cache slot cannot be negative")

    var kv_width = spec.kv_width()
    var at = slot * kv_width
    var span = tokens * kv_width
    if keys.elements() < at + span or values.elements() < at + span:
        raise Error(
            "the cache has no room for slot "
            + String(slot)
            + ", which needs "
            + String(at + span)
            + " values"
        )

    device_rms_norm(ctx, x, w.attn_norm, s.norm, spec.eps, tokens)

    # The bias goes on in the projection's own epilogue rather than in three
    # kernels behind it. One thread of each block adds one float to the row it
    # just reduced, which is the same addition to the same number, and the three
    # launches and their three round trips through device memory are gone.
    #
    # Straight into the cache rather than into scratch and then a copy, which is
    # what the offset on the matvec is for. The cache is where they are needed
    # and this is the only place they are written.
    var bias_epi = EPI_BIAS if w.has_bias else EPI_NONE
    _project(ctx, w.wq, s.norm, s.q, tokens, 0, bias_epi, w.q_bias.ptr())
    _project_half(ctx, w.wk, s.norm, keys, tokens, at, bias_epi, w.k_bias.ptr())
    _project_half(
        ctx, w.wv, s.norm, values, tokens, at, bias_epi, w.v_bias.ptr()
    )

    # Every head of every token in the chunk is one launch, because the heads of
    # a token lie end to end and so do the tokens.
    if w.has_qk_norm:
        var head_dim = spec.attn.head_dim
        device_rms_norm_run(
            ctx, s.q, 0, head_dim, w.q_norm, spec.eps, tokens * spec.attn.heads
        )
        device_rms_norm_half(
            ctx,
            keys,
            at,
            head_dim,
            w.k_norm,
            spec.eps,
            tokens * spec.attn.kv_heads,
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
        tokens,
        spec.q_width(),
    )
    device_rope_half(
        ctx,
        spec.rope,
        keys,
        at,
        spec.attn.kv_heads,
        spec.attn.head_dim,
        pos,
        w.rope,
        tokens,
        kv_width,
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
        s.partials,
        tokens,
    )
    # The residual add rides the output projection's epilogue, so `s.projected`
    # is only used by the models that put a norm between the two. Those still
    # pay for the scratch vector and the two launches, because a norm over the
    # whole projection cannot be computed by a block that owns one row of it.
    if w.has_attn_post:
        _project(ctx, w.wo, s.heads_out, s.projected, tokens)
        device_rms_norm_inplace(
            ctx, s.projected, w.attn_post_norm, spec.eps, tokens
        )
        device_add_run(ctx, x, 0, s.projected, tokens * spec.width)
    else:
        _project(ctx, w.wo, s.heads_out, x, tokens, 0, EPI_ADD)


def device_mlp(
    ctx: DeviceContext,
    spec: BlockSpec,
    w: DeviceLayer,
    mut x: DeviceVec,
    mut s: DeviceScratch,
    tokens: Int = 1,
) raises:
    """The mlp sublayer, in place on the residual stream, `tokens` at once."""
    if tokens < 1:
        raise Error("a layer has to be given at least one token")
    if x.elements() < tokens * spec.width:
        raise Error(
            "the residual stream is "
            + String(x.elements())
            + " wide where the layer wants "
            + String(tokens * spec.width)
        )

    device_rms_norm(ctx, x, w.ffn_norm, s.norm, spec.eps, tokens)
    _project(ctx, w.up, s.norm, s.up, tokens)

    # The gate half of a gated MLP is `act(gate) * up` and the up projection has
    # already been written by the line above, so the gate projection's own
    # epilogue can finish the element rather than a kernel behind it reading
    # both halves back. The ungated form has nothing to pair with, so its
    # activation is still its own launch over `s.up`.
    if w.has_gate:
        var glu = EPI_GLU | (0 if spec.act == ACT_SILU else ACT_BIT)
        _project(ctx, w.gate, s.norm, s.gate, tokens, 0, glu, s.up.ptr())
        if w.has_ffn_post:
            _project(ctx, w.down, s.gate, s.projected, tokens)
            device_rms_norm_inplace(
                ctx, s.projected, w.ffn_post_norm, spec.eps, tokens
            )
            device_add_run(ctx, x, 0, s.projected, tokens * spec.width)
        else:
            _project(ctx, w.down, s.gate, x, tokens, 0, EPI_ADD)
    else:
        if spec.act == ACT_SILU:
            device_silu(ctx, s.up, tokens * spec.hidden)
        else:
            device_gelu(ctx, s.up, tokens * spec.hidden)
        if w.has_ffn_post:
            _project(ctx, w.down, s.up, s.projected, tokens)
            device_rms_norm_inplace(
                ctx, s.projected, w.ffn_post_norm, spec.eps, tokens
            )
            device_add_run(ctx, x, 0, s.projected, tokens * spec.width)
        else:
            _project(ctx, w.down, s.up, x, tokens, 0, EPI_ADD)


def device_layer(
    ctx: DeviceContext,
    spec: BlockSpec,
    w: DeviceLayer,
    mut x: DeviceVec,
    mut s: DeviceScratch,
    mut keys: DeviceHalf,
    mut values: DeviceHalf,
    slot: Int,
    pos: Int,
    tokens: Int = 1,
) raises:
    """Both sublayers, which is one decoder layer."""
    device_attention(ctx, spec, w, x, s, keys, values, slot, pos, tokens)
    device_mlp(ctx, spec, w, x, s, tokens)


def _pool_base(m: DeviceModel) raises -> Int:
    """The address the step table measures every weight offset from.

    The load puts every matrix a kernel reads into one device buffer, so the
    lowest address any of them has is inside that buffer and every other one is
    above it. Taking the minimum rather than asking the pool keeps this a
    property of the model that was actually bound, and it is the same number
    every time because the addresses are fixed when the model loads.
    """
    var base = -1
    for i in range(m.block_count()):
        ref w = m.layers[i]
        _lower(base, w.wq)
        _lower(base, w.wk)
        _lower(base, w.wv)
        _lower(base, w.wo)
        _lower(base, w.up)
        _lower(base, w.down)
        if w.has_gate:
            _lower(base, w.gate)
    if base < 0:
        raise Error("a model with no matrices in it cannot be planned")
    return base


def _lower(mut base: Int, t: Tensor) raises:
    """Keep the lowest address seen so far, starting from a negative one."""
    var at = t.device_address()
    if base < 0 or at < base:
        base = at


def _matvec_record(
    mut plan: StepPlan,
    base: Int,
    w: Tensor,
    xs: Int,
    xo: Int,
    os: Int,
    oo: Int,
    ok: Int,
    epi: Int,
) raises -> Int:
    """One projection, as a record.

    The helper operand defaults to the output, which is what the unfused matvec
    does with an epilogue that reads nothing, so there is no null for the kernel
    to test for. A caller whose epilogue does read something overwrites it.
    """
    var at = w.device_address() - base
    if at < 0 or at % 4 != 0:
        raise Error(
            "a weight sits "
            + String(at)
            + " bytes from the pool base, and the scale planes of a planar row"
            " are read through a float32 view, so the offset has to be positive"
            " and a multiple of four"
        )
    var rec = plan.open(OP_MATVEC)
    plan.input(rec, xs, xo)
    plan.output(rec, os, oo, ok)
    plan.helper(rec, os, oo, ok)
    plan.set(rec, R_W, at)
    plan.set(rec, R_COLS, w.cols)
    plan.set(rec, R_ROWS, w.rows)
    plan.set(rec, R_STRIDE, w.row_bytes())
    plan.set(rec, R_EPI, epi)
    plan.set(rec, R_KIND, quant_kind(w))
    return rec


def _norm_record(
    mut plan: StepPlan,
    gain: Int,
    eps: Float32,
    n: Int,
    runs: Int,
    xs: Int,
    xo: Int,
    xk: Int,
    os: Int,
    oo: Int,
    ok: Int,
) -> Int:
    """One norm, as a record, over `runs` vectors of `n` laid end to end.

    A whole model width is one run, so one block does it and the rest of the
    grid waits. Splitting it would make the sum of squares a grid wide reduction
    and cost two more barriers, which is more than the few microseconds a block
    spends on four thousand elements. The per head norms are `heads` runs and
    spread across the grid the way everything else does.
    """
    var rec = plan.open(OP_NORM)
    plan.input(rec, xs, xo, xk)
    plan.output(rec, os, oo, ok)
    plan.set(rec, R_N, n)
    plan.set(rec, R_RUNS, runs)
    plan.set(rec, R_G, gain)
    plan.setf(rec, R_EPS, eps)
    return rec


def _rope_record(
    mut plan: StepPlan,
    rope: RopeSpec,
    a: LayerArena,
    kind: Int,
    low: Float32,
    high: Float32,
    heads: Int,
    head_dim: Int,
    space: Int,
    off: Int,
    slot_mul: Int,
) raises -> Int:
    """One rotation in place, as a record.

    A rotation on the cache is the one that runs on halves, and the fused kernel
    gives a thread two rotations there so that what it writes is whole words.
    That needs an even number of rotations in a head, which every model molla
    has met has, and this is where the day one does not turns into a message
    rather than into two threads writing over each other.
    """
    if space == SPACE_KEYS and rope.dim % 4 != 0:
        raise Error(
            "a rotation over "
            + String(rope.dim)
            + " dimensions of a head does not divide into pairs of rotations,"
            " and the key cache is written a pair at a time"
        )
    var rec = plan.open(OP_ROPE)
    plan.input(rec, space, off, slot_mul)
    plan.output(rec, space, off, slot_mul)
    plan.set(rec, R_HEADS, heads)
    plan.set(rec, R_HEAD_DIM, head_dim)
    plan.set(rec, R_DIM, rope.dim)
    plan.set(rec, R_KIND, kind)
    plan.set(rec, R_G, a.steps)
    plan.set(rec, R_H, a.factors)
    plan.setf(rec, R_SCALE, rope.scale)
    plan.setf(rec, R_EXT, rope.ext_factor)
    plan.setf(rec, R_ATTN, rope.attn_factor)
    plan.setf(rec, R_LOW, low)
    plan.setf(rec, R_HIGH, high)
    return rec


def _layer_records(
    mut plan: StepPlan,
    base: Int,
    spec: BlockSpec,
    w: DeviceLayer,
    shape: WorkPlan,
    use_factors: Bool,
) raises:
    """One layer, as the records the fused kernel walks.

    The same steps in the same order as `device_attention` and `device_mlp`,
    with the barrier flag set on eight of them. The six boundaries that do not
    get one are the three attention projections, which read the same input and
    write disjoint outputs, the two per head norms and the two rotations, which
    touch disjoint memory, and the up projection, which the gate reads back off
    rows the same block owns in both records. See
    [docs/validation/fused.md](../../../docs/validation/fused.md).
    """
    ref a = w.arena
    var kv_width = spec.kv_width()
    var head_dim = spec.attn.head_dim
    var bias_epi = EPI_BIAS if w.has_bias else EPI_NONE

    var rec = _norm_record(
        plan,
        a.attn_norm,
        spec.eps,
        spec.width,
        1,
        SPACE_RESID,
        0,
        0,
        SPACE_WORK,
        shape.norm,
        0,
    )
    plan.sync(rec)

    # Straight into the cache rather than into scratch and then a copy, which is
    # what the slot multiplier on the operand is for. The slot is the one thing
    # here that changes between tokens, so it is a kernel argument and the
    # record carries what to multiply it by.
    rec = _matvec_record(
        plan,
        base,
        w.wq,
        SPACE_WORK,
        shape.norm,
        SPACE_WORK,
        shape.q,
        0,
        bias_epi,
    )
    if w.has_bias:
        plan.helper(rec, SPACE_ARENA, a.q_bias)
    rec = _matvec_record(
        plan,
        base,
        w.wk,
        SPACE_WORK,
        shape.norm,
        SPACE_KEYS,
        0,
        kv_width,
        bias_epi | EPI_HALF,
    )
    if w.has_bias:
        plan.helper(rec, SPACE_ARENA, a.k_bias)
    rec = _matvec_record(
        plan,
        base,
        w.wv,
        SPACE_WORK,
        shape.norm,
        SPACE_VALS,
        0,
        kv_width,
        bias_epi | EPI_HALF,
    )
    if w.has_bias:
        plan.helper(rec, SPACE_ARENA, a.v_bias)
    plan.sync(rec)

    if w.has_qk_norm:
        _ = _norm_record(
            plan,
            a.q_norm,
            spec.eps,
            head_dim,
            spec.attn.heads,
            SPACE_WORK,
            shape.q,
            0,
            SPACE_WORK,
            shape.q,
            0,
        )
        rec = _norm_record(
            plan,
            a.k_norm,
            spec.eps,
            head_dim,
            spec.attn.kv_heads,
            SPACE_KEYS,
            0,
            kv_width,
            SPACE_KEYS,
            0,
            kv_width,
        )
        plan.sync(rec)

    var low = Float32(0)
    var high = Float32(0)
    if spec.rope.ext_factor != 0:
        var ends = corr_range(spec.rope)
        low = ends[0]
        high = ends[1]
    var kind = rope_kind(spec.rope, use_factors)
    _ = _rope_record(
        plan,
        spec.rope,
        a,
        kind,
        low,
        high,
        spec.attn.heads,
        head_dim,
        SPACE_WORK,
        shape.q,
        0,
    )
    rec = _rope_record(
        plan,
        spec.rope,
        a,
        kind,
        low,
        high,
        spec.attn.kv_heads,
        head_dim,
        SPACE_KEYS,
        0,
        kv_width,
    )
    plan.sync(rec)

    rec = plan.open(OP_ATTEND)
    plan.input(rec, SPACE_WORK, shape.q)
    plan.output(rec, SPACE_WORK, shape.heads_out)
    plan.set(rec, R_HEADS, spec.attn.heads)
    plan.set(rec, R_HEAD_DIM, head_dim)
    plan.set(rec, R_KV, kv_width)
    plan.set(rec, R_GROUP, spec.attn.group())
    plan.set(rec, R_WINDOW, spec.attn.window)
    plan.set(rec, R_SINKS, spec.attn.sinks)
    plan.set(rec, R_ROW, shape.scores)
    plan.set(rec, R_SPLIT, shape.splits)
    plan.setf(rec, R_SCALE, spec.attn.scale)
    plan.setf(rec, R_SOFTCAP, spec.attn.softcap)
    plan.sync(rec)

    # The residual add rides the output projection's epilogue, so the projected
    # scratch is only used by the models that put a norm between the two, and
    # those pay for two more records and two more barriers the way they pay for
    # two more launches today.
    if w.has_attn_post:
        rec = _matvec_record(
            plan,
            base,
            w.wo,
            SPACE_WORK,
            shape.heads_out,
            SPACE_WORK,
            shape.projected,
            0,
            EPI_NONE,
        )
        plan.sync(rec)
        rec = _norm_record(
            plan,
            a.attn_post,
            spec.eps,
            spec.width,
            1,
            SPACE_WORK,
            shape.projected,
            0,
            SPACE_WORK,
            shape.projected,
            0,
        )
        plan.sync(rec)
        rec = plan.open(OP_ADD)
        plan.input(rec, SPACE_WORK, shape.projected)
        plan.output(rec, SPACE_RESID, 0)
        plan.set(rec, R_N, spec.width)
        plan.sync(rec)
    else:
        rec = _matvec_record(
            plan,
            base,
            w.wo,
            SPACE_WORK,
            shape.heads_out,
            SPACE_RESID,
            0,
            0,
            EPI_ADD,
        )
        plan.sync(rec)

    rec = _norm_record(
        plan,
        a.ffn_norm,
        spec.eps,
        spec.width,
        1,
        SPACE_RESID,
        0,
        0,
        SPACE_WORK,
        shape.norm,
        0,
    )
    plan.sync(rec)

    var into = shape.up
    var up = _matvec_record(
        plan,
        base,
        w.up,
        SPACE_WORK,
        shape.norm,
        SPACE_WORK,
        shape.up,
        0,
        EPI_NONE,
    )
    if w.has_gate:
        var glu = EPI_GLU | (0 if spec.act == ACT_SILU else ACT_BIT)
        rec = _matvec_record(
            plan,
            base,
            w.gate,
            SPACE_WORK,
            shape.norm,
            SPACE_WORK,
            shape.gate,
            0,
            glu,
        )
        plan.helper(rec, SPACE_WORK, shape.up)
        plan.sync(rec)
        into = shape.gate
    else:
        # The gated form needs no barrier between the two projections because
        # the gate reads the up off rows the same block wrote. The ungated form
        # has an activation over the whole vector instead, and that one is
        # partitioned by element rather than by row, so it does.
        plan.sync(up)
        rec = plan.open(OP_ACT)
        plan.input(rec, SPACE_WORK, shape.up)
        plan.output(rec, SPACE_WORK, shape.up)
        plan.set(rec, R_N, spec.hidden)
        plan.set(rec, R_KIND, spec.act)
        plan.sync(rec)

    if w.has_ffn_post:
        rec = _matvec_record(
            plan,
            base,
            w.down,
            SPACE_WORK,
            into,
            SPACE_WORK,
            shape.projected,
            0,
            EPI_NONE,
        )
        plan.sync(rec)
        rec = _norm_record(
            plan,
            a.ffn_post,
            spec.eps,
            spec.width,
            1,
            SPACE_WORK,
            shape.projected,
            0,
            SPACE_WORK,
            shape.projected,
            0,
        )
        plan.sync(rec)
        rec = plan.open(OP_ADD)
        plan.input(rec, SPACE_WORK, shape.projected)
        plan.output(rec, SPACE_RESID, 0)
        plan.set(rec, R_N, spec.width)
        plan.sync(rec)
    else:
        rec = _matvec_record(
            plan,
            base,
            w.down,
            SPACE_WORK,
            into,
            SPACE_RESID,
            0,
            0,
            EPI_ADD,
        )
        plan.sync(rec)


def build_fused_plan(
    ctx: DeviceContext,
    m: DeviceModel,
    context: Int,
    use_factors: Bool,
    wanted: Bool = True,
) raises -> FusedPlan:
    """The whole model's step table, built once when a session opens.

    The self test runs before this returns, because a grid that is not fully
    resident deadlocks rather than failing, and a deadlock inside a forward pass
    is a hung display rather than an error message.

    `wanted` is False for a session that has decided against the fused path, and
    then nothing is built. That is a memory decision and not a speed one:
    sizing the grid compiles the fused kernel to ask it about occupancy, and
    compiling it costs about 1.2 GiB of host memory, which a session that is
    never going to launch it should not be holding. It is measured on gpc, where
    SmolLM2 135M goes from 274 MiB resident to 1467 with the plan built.
    """
    if not wanted:
        return FusedPlan(ctx)
    var shape = WorkPlan(m.specs, context)
    var plan = StepPlan()
    var starts = List[Int]()
    var base = _pool_base(m)
    for i in range(m.block_count()):
        starts.append(plan.records)
        _layer_records(plan, base, m.specs[i], m.layers[i], shape, use_factors)
    starts.append(plan.records)
    var built = FusedPlan(ctx, plan, starts, shape, base)
    fused_selftest(ctx, built.blocks)
    return built^


def _embed(
    ctx: DeviceContext,
    m: DeviceModel,
    mut s: DeviceScratch,
    mut x: DeviceVec,
    tokens: List[Int],
    pos: Int,
) raises -> Int:
    """Everything a pass does before the first layer, and the checks it needs.

    Shared by the two forward passes rather than written twice, because the
    fused path and the unfused one differ in the layer loop and in nothing else,
    and two copies of the validation is two places for them to stop agreeing.
    Returns the number of rows the residual stream carries, which is the run
    rounded up to a whole block of tokens for a chunk and the run itself for a
    decode.
    """
    var run = len(tokens)
    if pos < 0:
        raise Error("a position cannot be negative")
    if run < 1:
        raise Error("a forward pass needs at least one token")
    if run > s.chunk:
        raise Error(
            "a run of "
            + String(run)
            + " tokens was given scratch sized for "
            + String(s.chunk)
        )
    # A chunk is read a whole block of tokens at a time, so the residual stream
    # has to hold the rounded up count and not the run. See the scratch.
    var rows = run
    if run > 1:
        var block = SPAN * MM_GROUPS
        rows = (run + block - 1) // block * block
    if x.elements() < rows * m.width():
        raise Error(
            "the residual stream is "
            + String(x.elements())
            + " wide where the model wants "
            + String(rows * m.width())
        )
    if s.tracing and run != 1:
        raise Error(
            "a trace records one residual stream per layer and this pass"
            " carries "
            + String(run)
            + " of them, so tracing runs a token at a time"
        )

    var row = List[Float32]()
    for i in range(run):
        if tokens[i] < 0 or tokens[i] >= m.embedding.rows:
            raise Error(
                "token "
                + String(tokens[i])
                + " is outside a vocabulary of "
                + String(m.embedding.rows)
            )
        row.append(Float32(tokens[i]))
    # Padded to the scratch, because a copy fills the vector it is given and a
    # short chunk is the ordinary case at the end of a prompt. The lookup only
    # launches `run` rows deep, so the padding is never read.
    while len(row) < s.ids.elements():
        row.append(Float32(0))
    s.ids.copy_in(row)

    device_unpack_rows(ctx, m.embedding, s.ids, run, x)
    if m.arch.scale_embedding:
        device_scale_into(
            ctx, x, Float32(sqrt(Float64(m.width()))), run * m.width()
        )
    # The same numbering the host trace uses, so snapshot k is the residual
    # stream layer k was handed and a comparison can say which layer a
    # divergence started in.
    s.record(ctx, x)
    return run


def _finish(
    ctx: DeviceContext,
    m: DeviceModel,
    mut s: DeviceScratch,
    mut x: DeviceVec,
    run: Int,
) raises:
    """The final norm, the output head and the cap, for whichever path ran."""
    device_rms_norm(
        ctx,
        x,
        m.output_norm,
        s.norm,
        m.specs[0].eps,
        1,
        (run - 1) * m.width(),
    )
    device_matvec_into(ctx, m.head, s.norm, s.logits)
    if m.arch.final_softcap > 0:
        device_softcap(ctx, s.logits, m.arch.final_softcap, m.vocab())
    if s.tracing:
        # One more after the final norm, so the output head sits between the
        # last snapshot and the logits with nothing else in it. Brought back
        # first and appended second, because `s.norm` cannot be read and `s`
        # written in one call.
        var normed = snapshot(ctx, s.norm)
        s.take(normed)


def device_forward_fused(
    ctx: DeviceContext,
    m: DeviceModel,
    p: FusedPlan,
    mut s: DeviceScratch,
    mut x: DeviceVec,
    token: Int,
    pos: Int,
    slot: Int,
    mut keys: List[DeviceHalf],
    mut values: List[DeviceHalf],
) raises:
    """One token through the whole stack, one launch a layer.

    The same arithmetic in the same order as `device_forward` on a run of one,
    and it has to agree with it in every digit rather than closely, which is
    what `tests/test_gpu_fused.mojo` asserts. What is different is the number of
    commands: a thirty layer token is thirty three launches here where the
    unfused path is 363, and at 20 microseconds a launch on an M4 that is the
    difference between 7.3 milliseconds of submission and 0.66.

    One token, because stage one is the decode path. A prefill chunk still goes
    through `device_forward` and the matmul until stage three.
    """
    var count = m.block_count()
    if len(keys) != count or len(values) != count:
        raise Error(
            "the cache has room for "
            + String(len(keys))
            + " layers and the model has "
            + String(count)
        )
    if len(p.starts) != count + 1:
        raise Error(
            "the step table covers "
            + String(len(p.starts) - 1)
            + " layers and the model has "
            + String(count)
        )
    if slot < 0:
        raise Error("a cache slot cannot be negative")

    var one = List[Int]()
    one.append(token)
    _ = _embed(ctx, m, s, x, one, pos)
    for i in range(count):
        launch_fused(
            ctx,
            p,
            m.layers[i].arena.vec,
            x,
            keys[i],
            values[i],
            p.starts[i],
            p.starts[i + 1],
            pos,
            slot,
            slot + 1,
        )
        s.record(ctx, x)
    _finish(ctx, m, s, x, 1)


def device_forward(
    ctx: DeviceContext,
    m: DeviceModel,
    mut s: DeviceScratch,
    mut x: DeviceVec,
    tokens: List[Int],
    pos: Int,
    slot: Int,
    mut keys: List[DeviceHalf],
    mut values: List[DeviceHalf],
) raises:
    """A run of tokens through the whole stack, logits left on the device.

    Queued and not synchronized. Everything here is one stream of launches with
    no wait between them, which is the thing that makes a token one command
    buffer rather than several hundred, and the caller waits once when it wants
    to read the logits.

    `pos` and `slot` are two arguments for the reason the host version gives:
    they are the same number until something evicts, and the day they stop being
    the same is the day a single argument becomes a bug in two places. A run
    occupies `len(tokens)` of each, starting at these.

    Only the last token of the run gets logits, because it is the only one a
    caller can do anything with and the output head is the largest single
    projection in the pass. See
    [docs/validation/prefill.md](../../../docs/validation/prefill.md).
    """
    var count = m.block_count()
    if len(keys) != count or len(values) != count:
        raise Error(
            "the cache has room for "
            + String(len(keys))
            + " layers and the model has "
            + String(count)
        )

    var run = _embed(ctx, m, s, x, tokens, pos)
    for i in range(count):
        device_layer(
            ctx,
            m.specs[i],
            m.layers[i],
            x,
            s,
            keys[i],
            values[i],
            slot,
            pos,
            run,
        )
        s.record(ctx, x)
    _finish(ctx, m, s, x, run)
