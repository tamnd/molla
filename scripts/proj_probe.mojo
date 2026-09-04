"""What the seven projections of one layer cost, measured on the shipped kernel.

`scripts/matvec_probe.mojo` answers a different question. It has its own copy of
the read loop so it can take one piece out at a time, which is what makes it
useful for deciding between two ways of writing the inner loop and what makes it
useless for saying how long a decode takes. Its copy is not the kernel that
ships, it only covers the two quant forms somebody needed for a comparison, and
its numbers are per variant rather than per token.

This one calls `device_matvec_into`, so what it times is the kernel a token
actually launches, at the shapes and quant kinds a model actually has, with the
cache taken away the same way the cold sweep does it. The output is a budget:
microseconds a launch, and what those add up to over a whole token.

The shapes below are Meta-Llama-3.1-8B-Instruct-Q4_K_M, read off its tensor
table. Two of the seven are q6_k and the rest are q4_k, which matters because
q6_k stores sixteen scales per 256 values where q4_k stores eight, so a q6_k row
is seven bits a value in molla's planar layout against q4_k's five. Two fifths of
the bytes a token reads are in the wider form and no probe had ever timed it.

The weights are not real weights. Timing a matvec does not depend on what the
numbers are, with one exception worth naming: a quant plane of all zeros with a
scale plane of all zeros produces zeros and nothing else, which is a normal float
and not a denormal, so there is no stall hiding in an unwritten buffer. The
buffers are left as allocated for that reason and the answers are thrown away.

Run it on a machine that is not doing anything else. On this laptop the load
average makes every number here meaningless.
"""

from std.sys.info import has_accelerator
from std.time import monotonic

from max.gpu.host import DeviceContext

from molla.nn.gpu import DeviceVec, device_matvec_into
from molla.nn.quant import Q_Q4_K, Q_Q6_K
from molla.nn.repack import LAYOUT_PLANAR, planar_row_bytes
from molla.nn.tensor import WHERE_DEVICE, Tensor

comptime LAUNCHES = 16
"""Launches in one timed run, averaged. Sixteen because that is enough that the
first one's cost is a sixteenth of the answer and few enough that the pool it
needs still fits on a card."""

comptime COLD_BYTES = 512 * 1024 * 1024
"""How much distinct weight a run reads before it repeats itself.

A 4090 has 72 MB of L2 and an M4 has far less, so half a gibibyte of distinct
bytes is several times either. Below this a shape is partly timing a cache and
the number comes out higher than anything a decode will ever see, which is the
mistake documented in docs/validation/layout.md.
"""


def _slices(span: Int) -> Int:
    """How many copies of a tensor a run needs to outrun the cache.

    Two at the least, because one copy read sixteen times is the warm case this
    exists to avoid, and no more than `LAUNCHES`, because a seventeenth copy
    would never be read.
    """
    var want = COLD_BYTES // span + 1
    if want < 2:
        want = 2
    return want if want < LAUNCHES else LAUNCHES


def _time(
    ctx: DeviceContext,
    kind: Int,
    cols: Int,
    rows: Int,
    reps: Int,
) raises -> Float64:
    """Nanoseconds one launch of this projection takes, best of `reps`.

    Best rather than mean, for the reason every timing loop in this repository
    takes the best: a run that came out slow was sharing the card with something
    and a run that came out fast was not, and only one of those is a property of
    the kernel.
    """
    var stride = planar_row_bytes(kind, cols)
    var span = stride * rows
    var slices = _slices(span)
    var pool = ctx.enqueue_create_buffer[DType.uint8](span * slices)
    var x = DeviceVec(ctx, cols)
    var out = DeviceVec(ctx, rows)
    ctx.synchronize()
    var base = Int(pool.unsafe_ptr())

    var best = Float64(0)
    for rep in range(reps):
        var began = monotonic()
        for k in range(LAUNCHES):
            var w = Tensor(
                base + (k % slices) * span,
                kind,
                cols,
                rows,
                LAYOUT_PLANAR,
                WHERE_DEVICE,
            )
            device_matvec_into(ctx, w, x, out)
        ctx.synchronize()
        var took = Float64(monotonic() - began) / Float64(LAUNCHES)
        if rep == 0 or took < best:
            best = took
    return best


struct Shape(Copyable, Movable):
    var name: String
    var kind: Int
    var cols: Int
    var rows: Int
    var each: Int
    """How many launches of this exact shape and kind a token makes.

    Counted over the whole model rather than per layer, because Q4_K_M does not
    give every layer the same types. Half the value projections and half the
    down projections are q6_k and the other half are q4_k, which is what the M
    in the name means, so a per layer count would have to carry a fraction.
    """

    def __init__(
        out self, name: String, kind: Int, cols: Int, rows: Int, each: Int
    ):
        self.name = name
        self.kind = kind
        self.cols = cols
        self.rows = rows
        self.each = each


def _pad(s: String, to: Int) -> String:
    var out = s
    while out.byte_length() < to:
        out += " "
    return out^


def main() raises:
    comptime if not has_accelerator():
        print("proj_probe: no accelerator on this machine")
        return
    else:
        var ctx = DeviceContext()
        print("device:", ctx.name())
        print("")

        # Read off the tensor table of the 8B. `cols` is the fast dimension,
        # which is the length of the row a matvec dots, and `rows` is how many
        # of them there are, which is the grid.
        var shapes = List[Shape]()
        shapes.append(Shape("attn_q  ", Q_Q4_K, 4096, 4096, 32))
        shapes.append(Shape("attn_k  ", Q_Q4_K, 4096, 1024, 32))
        shapes.append(Shape("attn_v  ", Q_Q4_K, 4096, 1024, 16))
        shapes.append(Shape("attn_v 6", Q_Q6_K, 4096, 1024, 16))
        shapes.append(Shape("attn_out", Q_Q4_K, 4096, 4096, 32))
        shapes.append(Shape("ffn_gate", Q_Q4_K, 4096, 14336, 32))
        shapes.append(Shape("ffn_up  ", Q_Q4_K, 4096, 14336, 32))
        shapes.append(Shape("ffn_down", Q_Q4_K, 14336, 4096, 16))
        shapes.append(Shape("ffn_dn 6", Q_Q6_K, 14336, 4096, 16))
        shapes.append(Shape("output  ", Q_Q6_K, 4096, 128256, 1))

        var token_ns = Float64(0)
        var token_bytes = Float64(0)
        print("one launch of each, cold, best of five")
        for i in range(len(shapes)):
            var s = shapes[i].copy()
            var span = planar_row_bytes(s.kind, s.cols) * s.rows
            var ns = _time(ctx, s.kind, s.cols, s.rows, 5)
            token_ns += ns * Float64(s.each)
            token_bytes += Float64(span) * Float64(s.each)
            print(
                "  "
                + _pad(s.name, 10)
                + _pad("x" + String(s.each), 5)
                + _pad(String(Int(Float64(span) / 1048576.0)) + " MiB", 9)
                + _pad(String(Int(ns / 1000.0)) + " us", 8)
                + String(Int(Float64(span) / ns))
                + " GB/s"
            )

        print("")
        print(
            "a token   "
            + String(Int(token_ns / 1000000.0))
            + "."
            + String(Int(token_ns / 100000.0) % 10)
            + " ms over "
            + String(Int(token_bytes / 1073741824.0))
            + "."
            + String(Int(token_bytes / 10737418.24) % 100)
            + " GiB, at "
            + String(Int(token_bytes / token_ns))
            + " GB/s"
        )
        print("")
        print(
            "That is the projections alone. Everything a decode does that is"
            " not one of these"
        )
        print(
            "launches is the difference between it and what `generate` reports"
            " a token takes."
        )
