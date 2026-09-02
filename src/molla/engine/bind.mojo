"""Pointing the network at a file's bytes.

`molla.nn` takes tensors that are an address and a shape and does not care where
they came from. `molla.model` reads a GGUF and can say where every tensor in it
starts. This is the twenty lines in between, and it is its own module because
the thing it does wrong is silent.

A weight bound to the wrong address produces output. A weight bound with its
rows and columns swapped produces a shape error, which is the good case. A
weight that is absent and bound as present is a read from wherever address zero
maps, and a weight that is present and bound as absent is a layer quietly
skipping a norm. So the binding is done by name against a table that says which
names this architecture has, `LayerWeights.check` runs over the result before a
token is computed, and a name that is in the table and not in the file is an
error rather than a `none`.

Everything binds to a mapping. Weights that were copied to a device sit at an
address the host kernels cannot read, and every kernel in `molla.nn` is a host
kernel today, so `bind` refuses a plan that put anything on a card rather than
handing back a pointer that faults on the first token. Device placement becomes
useful when there are device kernels, which is M3.

Which mapping is the one thing that varies. A tensor that the repack cache has a
copy of binds to the cache and carries the planar layout with it, and a tensor
that it does not binds to the model file in the layout the file has. That choice
is made once per weight, here, and nothing downstream branches on it beyond the
one hoisted test in `matvec`. A cache that is absent is not a special case: every
lookup misses and every weight comes from the file, which is exactly what
happened before there was a cache at all.
"""

from molla.model.gguf import Gguf, TensorInfo
from molla.model.repack import RepackCache
from molla.model.spec import Geometry, architecture_id, read_geometry
from molla.nn.arch import (
    Arch,
    OUTPUT_HEAD,
    OUTPUT_NORM,
    ROPE_FREQS,
    TOKEN_EMBD,
    arch_of,
    block_spec,
    tensor_names,
)
from molla.nn.block import BlockSpec, LayerWeights
from molla.nn.model import ModelWeights
from molla.nn.tensor import Tensor


struct Bound(Movable):
    """A whole model, pointed at a file that is still open.

    Nothing in here owns bytes. Every tensor is an address inside the mapping,
    so a `Bound` outliving the `Gguf` it came from is a use after free, and the
    caller holds both for the same length of time on purpose.
    """

    var arch: Arch
    var geometry: Geometry
    var model: ModelWeights
    var layers: List[LayerWeights]
    var specs: List[BlockSpec]

    def __init__(
        out self,
        arch: Arch,
        geometry: Geometry,
        model: ModelWeights,
        var layers: List[LayerWeights],
        var specs: List[BlockSpec],
    ):
        self.arch = arch
        self.geometry = geometry
        self.model = model
        self.layers = layers^
        self.specs = specs^

    def block_count(self) -> Int:
        return len(self.specs)

    def kv_width(self) -> Int:
        return self.specs[0].kv_width()

    def width(self) -> Int:
        return self.specs[0].width

    def vocab(self) -> Int:
        return self.model.vocab()


def tensor_address(g: Gguf, info: TensorInfo) -> Int:
    """Where one tensor's first byte is in the mapping.

    The directory holds an offset from the start of the data section rather
    than from the start of the file, and the data section starts after the
    metadata at an alignment the file states.
    """
    return g.mapping.address + g.data_start + info.offset


def _shape_of(info: TensorInfo) -> Tensor:
    """A `TensorInfo` as a two dimensional view.

    `dims[0]` is the fast axis in ggml, so it is the column count, and a one
    dimensional tensor is one row of it rather than a column of length one.
    Getting this backwards is a shape error at load rather than wrong numbers
    at token one, which is the one place the layout is forgiving.
    """
    var cols = Int(info.d0)
    var rows = Int(info.d1) if info.n_dims > 1 else 1
    return Tensor(0, info.kind, cols, rows)


def find(g: Gguf, cache: RepackCache, name: String) raises -> Tensor:
    """The named tensor, or `Tensor.none()` when the file does not have it.

    The cache is consulted only for a tensor the file has, so a cache holding a
    name the model does not is ignored rather than conjuring a weight. The shape
    and type are checked against the file's even though the cache key already
    covers the directory they came from, because the cost is three integer
    comparisons and the failure it catches is a weight bound to the right name
    and the wrong bytes.
    """
    if name.byte_length() == 0:
        return Tensor.none()
    var at = g.tensor_index(name)
    if at < 0:
        return Tensor.none()
    var out = _shape_of(g.tensors[at])
    var hit = cache.find(name)
    if hit >= 0:
        var cached = cache.tensor(hit)
        if (
            cached.kind == out.kind
            and cached.cols == out.cols
            and cached.rows == out.rows
        ):
            return cached
    out.address = tensor_address(g, g.tensors[at])
    return out


def require(g: Gguf, cache: RepackCache, name: String) raises -> Tensor:
    """The named tensor, or an error naming what is missing."""
    var out = find(g, cache, name)
    if not out.present():
        raise Error("the file has no tensor called " + name)
    return out


def bind_model(g: Gguf, cache: RepackCache) raises -> ModelWeights:
    """The tensors that are not inside a layer."""
    var out = ModelWeights()
    out.embedding = require(g, cache, TOKEN_EMBD)
    out.output_norm = require(g, cache, OUTPUT_NORM)
    out.output = find(g, cache, OUTPUT_HEAD)
    out.rope_freqs = find(g, cache, ROPE_FREQS)
    return out


def asked_for(g: Gguf, cache: RepackCache, name: String) raises -> Tensor:
    """A tensor the architecture may or may not have asked for.

    An empty name is one this family does not have and stays a `none`. A name
    that is there is one the table asked for, so a file without it is a
    mismatch between what molla believes the architecture is and what this file
    actually contains, and binding it as absent would run the layer without a
    norm or without a bias and produce text.
    """
    if name.byte_length() == 0:
        return Tensor.none()
    var out = find(g, cache, name)
    if not out.present():
        raise Error("the architecture wants " + name + " and it is missing")
    return out


def bind_layer(
    g: Gguf, cache: RepackCache, a: Arch, layer: Int
) raises -> LayerWeights:
    """One layer's weights, by name, in the order `LayerWeights` declares them.

    The two lists are walked together rather than the names being looked up
    field by field, which is what keeps the table in `arch.mojo` and the struct
    in `block.mojo` from drifting apart.
    """
    var names = tensor_names(a, layer)
    if len(names) != 16:
        raise Error(
            "the architecture table gave "
            + String(len(names))
            + " tensor names for a layer that has 16 fields"
        )
    var w = LayerWeights()
    w.attn_norm = require(g, cache, names[0])
    w.attn_post_norm = asked_for(g, cache, names[1])
    w.wq = require(g, cache, names[2])
    w.wk = require(g, cache, names[3])
    w.wv = require(g, cache, names[4])
    w.wo = require(g, cache, names[5])
    w.q_bias = asked_for(g, cache, names[6])
    w.k_bias = asked_for(g, cache, names[7])
    w.v_bias = asked_for(g, cache, names[8])
    w.q_norm = asked_for(g, cache, names[9])
    w.k_norm = asked_for(g, cache, names[10])
    w.ffn_norm = require(g, cache, names[11])
    w.ffn_post_norm = asked_for(g, cache, names[12])
    w.gate = asked_for(g, cache, names[13])
    w.up = require(g, cache, names[14])
    w.down = require(g, cache, names[15])
    return w


def bind(g: Gguf, cache: RepackCache) raises -> Bound:
    """A file to a network, checked before a token is computed.

    Every shape is verified here rather than on the first forward pass. A load
    that takes half a minute and then fails on a shape at token one has spent
    that half minute for nothing, and the check is a few hundred integer
    comparisons against work that is measured in gigabytes.
    """
    var arch = arch_of(architecture_id(g.architecture()))
    if not arch.supported:
        raise Error(
            arch.name
            + " is in the architecture table and has not been run against"
            " llama.cpp yet, so molla will not claim to generate with it"
        )
    var geometry = read_geometry(g)
    if geometry.block_count <= 0:
        raise Error("the file does not say how many layers it has")

    var model = bind_model(g, cache)
    var layers = List[LayerWeights]()
    var specs = List[BlockSpec]()
    for i in range(geometry.block_count):
        var spec = block_spec(arch, geometry, i)
        var w = bind_layer(g, cache, arch, i)
        w.check(spec)
        specs.append(spec)
        layers.append(w)
    model.check(specs[0])
    return Bound(arch, geometry, model, layers^, specs^)
