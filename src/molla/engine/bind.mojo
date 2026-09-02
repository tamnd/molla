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

Everything binds to the mapping. Weights that were copied to a device sit at an
address the host kernels cannot read, and every kernel in `molla.nn` is a host
kernel today, so `bind` refuses a plan that put anything on a card rather than
handing back a pointer that faults on the first token. Device placement becomes
useful when there are device kernels, which is M3.
"""

from molla.model.gguf import Gguf, TensorInfo
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


def find(g: Gguf, name: String) raises -> Tensor:
    """The named tensor, or `Tensor.none()` when the file does not have it."""
    if name.byte_length() == 0:
        return Tensor.none()
    var at = g.tensor_index(name)
    if at < 0:
        return Tensor.none()
    var out = _shape_of(g.tensors[at])
    out.address = tensor_address(g, g.tensors[at])
    return out


def require(g: Gguf, name: String) raises -> Tensor:
    """The named tensor, or an error naming what is missing."""
    var out = find(g, name)
    if not out.present():
        raise Error("the file has no tensor called " + name)
    return out


def bind_model(g: Gguf) raises -> ModelWeights:
    """The tensors that are not inside a layer."""
    var out = ModelWeights()
    out.embedding = require(g, TOKEN_EMBD)
    out.output_norm = require(g, OUTPUT_NORM)
    out.output = find(g, OUTPUT_HEAD)
    out.rope_freqs = find(g, ROPE_FREQS)
    return out


def bind_layer(g: Gguf, a: Arch, layer: Int) raises -> LayerWeights:
    """One layer's weights, by name, in the order `LayerWeights` declares them.

    The two lists are walked together rather than the names being looked up
    field by field, which is what keeps the table in `arch.mojo` and the struct
    in `block.mojo` from drifting apart. A name the architecture does not have
    is the empty string and stays a `none` here.
    """
    var names = tensor_names(a, layer)
    if len(names) != 13:
        raise Error(
            "the architecture table gave "
            + String(len(names))
            + " tensor names for a layer that has 13 fields"
        )
    var w = LayerWeights()
    w.attn_norm = require(g, names[0])
    w.attn_post_norm = find(g, names[1])
    w.wq = require(g, names[2])
    w.wk = require(g, names[3])
    w.wv = require(g, names[4])
    w.wo = require(g, names[5])
    w.q_norm = find(g, names[6])
    w.k_norm = find(g, names[7])
    w.ffn_norm = require(g, names[8])
    w.ffn_post_norm = find(g, names[9])
    w.gate = find(g, names[10])
    w.up = require(g, names[11])
    w.down = require(g, names[12])

    # A name the table asked for and the file does not have is a mismatch
    # between what molla believes the architecture is and what this file
    # actually contains. Loading it as absent would run the layer without the
    # norm and produce text.
    if names[1].byte_length() > 0 and not w.attn_post_norm.present():
        raise Error("the architecture wants " + names[1] + " and it is missing")
    if names[6].byte_length() > 0 and not w.q_norm.present():
        raise Error("the architecture wants " + names[6] + " and it is missing")
    if names[7].byte_length() > 0 and not w.k_norm.present():
        raise Error("the architecture wants " + names[7] + " and it is missing")
    if names[9].byte_length() > 0 and not w.ffn_post_norm.present():
        raise Error("the architecture wants " + names[9] + " and it is missing")
    if names[10].byte_length() > 0 and not w.gate.present():
        raise Error(
            "the architecture wants " + names[10] + " and it is missing"
        )
    return w


def bind(g: Gguf) raises -> Bound:
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

    var model = bind_model(g)
    var layers = List[LayerWeights]()
    var specs = List[BlockSpec]()
    for i in range(geometry.block_count):
        var spec = block_spec(arch, geometry, i)
        var w = bind_layer(g, arch, i)
        w.check(spec)
        specs.append(spec)
        layers.append(w)
    model.check(specs[0])
    return Bound(arch, geometry, model, layers^, specs^)
