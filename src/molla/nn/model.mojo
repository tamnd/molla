"""A token in, a row of logits out.

Everything above a layer. An embedding lookup, the stack of layers, a final
norm, and the output head. It is about a hundred lines because the layers are
in `molla.nn.block` and the differences between architectures are in
`molla.nn.arch`, and once those are elsewhere there is not much of a model
left, which is the point of putting them there.

There is no cache in here. `forward` takes the key and value storage as one
list per layer and a slot to write into, and does not decide how long those
lists are, when a slot gets reused, or what happens when the context is full.
That is #27. Splitting it here means the arithmetic can be checked against
weights small enough to work out by hand, before there is a cache to be wrong
about at the same time.

There is no sampling either. This produces logits and stops. Turning logits
into a token is #28.
"""

from std.math import sqrt, tanh

from molla.nn.arch import Arch
from molla.nn.block import BlockSpec, LayerWeights, Scratch, layer
from molla.nn.kernel import matvec, rms_norm
from molla.nn.repack import unpack_run
from molla.nn.tensor import Buffer, Tensor


struct ModelWeights(Copyable, ImplicitlyCopyable, Movable):
    """The tensors that are not inside a layer."""

    var embedding: Tensor
    """`token_embd.weight`. One row per token, `width` wide."""

    var output_norm: Tensor
    """The norm before the head."""

    var output: Tensor
    """`output.weight`, or `Tensor.none()` when the head is tied to the
    embedding. Tying is not an optimisation a loader may apply, it is what the
    model was trained with, so an untied model loaded as tied produces fluent
    output from the wrong distribution."""

    var rope_freqs: Tensor
    """`rope_freqs.weight`, or none. One factor per rope pair."""

    def __init__(out self):
        var nothing = Tensor.none()
        self.embedding = nothing
        self.output_norm = nothing
        self.output = nothing
        self.rope_freqs = nothing

    def tied(self) -> Bool:
        return not self.output.present()

    def head(self) -> Tensor:
        """The tensor the logits come out of, whichever it is."""
        return self.embedding if self.tied() else self.output

    def check(self, spec: BlockSpec) raises:
        if not self.embedding.present():
            raise Error("a model needs an embedding")
        if not self.output_norm.present():
            raise Error("a model needs a norm before the output head")
        if self.embedding.cols != spec.width:
            raise Error(
                "the embedding is "
                + String(self.embedding.cols)
                + " wide where the layers want "
                + String(spec.width)
            )
        if self.output_norm.elements() != spec.width:
            raise Error(
                "the output norm is "
                + String(self.output_norm.elements())
                + " wide where the layers want "
                + String(spec.width)
            )
        if self.output.present() and self.output.cols != spec.width:
            raise Error(
                "the output head takes "
                + String(self.output.cols)
                + " where the layers give "
                + String(spec.width)
            )
        if self.rope_freqs.present():
            var pairs = spec.rope.dim // 2
            if self.rope_freqs.elements() != pairs:
                raise Error(
                    "the file has "
                    + String(self.rope_freqs.elements())
                    + " rope frequency factors where the rotary dimension"
                    + " needs "
                    + String(pairs)
                )

    def vocab(self) -> Int:
        return self.head().rows


def embed(w: ModelWeights, a: Arch, token: Int, mut out: Buffer) raises:
    """Look one token's row up and put it on the residual stream.

    A lookup and not a matvec. The one hot matrix multiply that this stands in
    for is a real multiply in a training framework and is a row copy here,
    which is the only place in a forward pass where the obvious implementation
    is thousands of times slower than the right one.
    """
    if token < 0 or token >= w.embedding.rows:
        raise Error(
            "token "
            + String(token)
            + " is out of range for an embedding with "
            + String(w.embedding.rows)
            + " rows"
        )
    if out.elements() != w.embedding.cols:
        raise Error("the residual stream is not the width of the embedding")
    var at = token * w.embedding.row_bytes()
    unpack_run(
        w.embedding.kind,
        w.embedding.layout,
        w.embedding.base(),
        at,
        out.elements(),
        out.data,
        0,
    )
    if a.scale_embedding:
        var by = Float32(sqrt(Float64(out.elements())))
        for i in range(out.elements()):
            out.data[i] = out.data[i] * by


def head(
    w: ModelWeights,
    a: Arch,
    spec: BlockSpec,
    x: Buffer,
    mut normed: Buffer,
    mut logits: Buffer,
) raises:
    """The final norm and the output head."""
    rms_norm(normed, x, w.output_norm, spec.eps)
    matvec(w.head(), normed, logits)
    if a.final_softcap > 0:
        var cap = a.final_softcap
        for i in range(logits.elements()):
            logits.data[i] = cap * tanh(logits.data[i] / cap)


def frequency_factors(w: ModelWeights) raises -> List[Float32]:
    """`rope_freqs.weight` as float32, or an empty list when absent.

    Read once when a model is loaded. It is a few dozen numbers and they do not
    change, so reading them per token per layer would be the same dequant done
    a thousand times a second for the same answer.
    """
    var out = List[Float32]()
    if not w.rope_freqs.present():
        return out^
    var n = w.rope_freqs.elements()
    for _ in range(n):
        out.append(0.0)
    unpack_run(
        w.rope_freqs.kind,
        w.rope_freqs.layout,
        w.rope_freqs.base(),
        0,
        n,
        out,
        0,
    )
    return out^


def forward(
    a: Arch,
    w: ModelWeights,
    specs: List[BlockSpec],
    layers: List[LayerWeights],
    mut s: Scratch,
    mut x: Buffer,
    token: Int,
    pos: Int,
    slot: Int,
    mut keys: List[List[Float32]],
    mut values: List[List[Float32]],
    factors: List[Float32],
    mut logits: Buffer,
) raises:
    """One token through the whole stack.

    `keys` and `values` are one list per layer and the caller owns them. `pos`
    is where this token sits in the sequence, which is what rope and the
    sliding window ask about, and `slot` is where in the cache it goes. They
    are the same number until something evicts, and they are two arguments
    because the day they stop being the same is the day a cache gets clever and
    a single argument becomes a bug in two places.
    """
    var count = len(specs)
    if count == 0:
        raise Error("a model with no layers is not a model")
    if len(layers) != count:
        raise Error(
            "the model has "
            + String(count)
            + " layer specs and "
            + String(len(layers))
            + " sets of weights"
        )
    if len(keys) != count or len(values) != count:
        raise Error(
            "the cache has room for "
            + String(len(keys))
            + " layers and the model has "
            + String(count)
        )
    if pos < 0:
        raise Error("a position cannot be negative")

    w.check(specs[0])
    embed(w, a, token, x)
    var use_factors = len(factors) > 0
    for i in range(count):
        layer(
            specs[i],
            layers[i],
            x,
            s,
            keys[i],
            values[i],
            slot,
            pos,
            factors,
            use_factors,
        )
    head(w, a, specs[0], x, s.norm, logits)
