"""What the decode matvec spends its time on, one shape, no model.

Not part of the library and not part of the test suite. Issue #186 says the
matvec is limited by values a second rather than by bytes a second, and the
evidence for it came from an 8B and nsys, which is an hour a question. This
runs the same kernel over a synthetic planar tensor of the same shape and
answers a question in a second.

Every variant computes a different answer than the one before it and none of
them is checked, which is the point: each one deletes a piece of the work so
that the piece can be priced. The baseline is a copy of the shipped kernel
rather than a call to it, so that all of them read the same way and differ only
where the comment says.

Usage:

    mojo run -I src scripts/matvec_probe.mojo
"""

from std.gpu import block_idx, thread_idx

from max.gpu import barrier
from std.memory import AddressSpace, stack_allocation
from std.time import monotonic

from max.gpu.host import DeviceContext

from molla.nn.repack import (
    QUANT_I8,
    QUANT_U4,
    planar_row_bytes,
)
from molla.nn.quant import Q_Q4_K

comptime TILE = 128

# What each variant takes away, in the order they are printed.
comptime V_SHIPPED = 0
"""The kernel as it is in `molla.nn.gpu`, for a number to compare against."""
comptime V_SHIFT = 1
"""Same, with the group index by a shift rather than a signed divide."""
comptime V_NARROW = 2
"""Same again, with the loop counter and the group index 32 bits wide."""
comptime V_PER_GROUP = 3
"""A thread walks whole groups, so a scale is loaded once per 32 values."""
comptime V_NO_SCALE = 4
"""The nibbles and the activations, with no scale plane read at all."""
comptime V_NO_MATH = 5
"""The loads and nothing else, which is the floor this shape can reach."""


def probe_kernel[
    tile: Int, group: Int, with_min: Bool, form: Int, variant: Int
](
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    cols_dev: Int32,
    stride_dev: Int32,
):
    var cols = Int(cols_dev)
    var stride = Int(stride_dev)
    var r = Int(block_idx.x)
    var t = Int(thread_idx.x)

    var row = r * stride
    var groups = cols // group
    var packed = w
    var scales = w.unsafe_bitcast[Float32]()
    var quant_bytes = cols if form == QUANT_I8 else cols // 2
    var d_base = (row + quant_bytes) // 4
    var m_base = d_base + groups

    # Both group sizes in the type table are powers of two, so the divide the
    # shipped kernel writes is a shift the compiler has to prove is one.
    comptime shift = 5 if group == 32 else 4

    var acc = Float32(0)

    comptime if variant == V_SHIPPED:
        var i = t * 2
        while i < cols:
            var gi = i // group
            var b = Int(packed[unsafe_offset=row + (i >> 1)])
            var lo = b & 0xF
            var hi = (b >> 4) & 0xF
            var a0 = x[unsafe_offset=i]
            var a1 = x[unsafe_offset=i + 1]
            var d = scales[unsafe_offset=d_base + gi]
            acc += d * Float32(lo) * a0
            acc += d * Float32(hi) * a1
            comptime if with_min:
                var m = scales[unsafe_offset=m_base + gi]
                acc += m * a0
                acc += m * a1
            i += tile * 2

    elif variant == V_SHIFT:
        var i = t * 2
        while i < cols:
            var gi = i >> shift
            var b = Int(packed[unsafe_offset=row + (i >> 1)])
            var lo = b & 0xF
            var hi = (b >> 4) & 0xF
            var a0 = x[unsafe_offset=i]
            var a1 = x[unsafe_offset=i + 1]
            var d = scales[unsafe_offset=d_base + gi]
            acc += d * Float32(lo) * a0
            acc += d * Float32(hi) * a1
            comptime if with_min:
                var m = scales[unsafe_offset=m_base + gi]
                acc += m * a0
                acc += m * a1
            i += tile * 2

    elif variant == V_NARROW:
        # Everything that changes every iteration is 32 bits wide. `Int` is 64
        # bits and a 64 bit add is two instructions, so a loop that carries four
        # of them is paying for a range it never goes near.
        var n = UInt32(cols)
        var i = UInt32(t * 2)
        var stepping = UInt32(tile * 2)
        while i < n:
            var gi = Int(i >> UInt32(shift))
            var b = Int(packed[unsafe_offset=row + Int(i >> UInt32(1))])
            var lo = b & 0xF
            var hi = (b >> 4) & 0xF
            var a0 = x[unsafe_offset=Int(i)]
            var a1 = x[unsafe_offset=Int(i) + 1]
            var d = scales[unsafe_offset=d_base + gi]
            acc += d * Float32(lo) * a0
            acc += d * Float32(hi) * a1
            comptime if with_min:
                var m = scales[unsafe_offset=m_base + gi]
                acc += m * a0
                acc += m * a1
            i += stepping

    elif variant == V_PER_GROUP:
        # A thread takes a whole group rather than a byte, so the scale, the
        # minimum and the group index are read once for thirty two values
        # instead of once for two. A warp reads 32 * 16 contiguous bytes an
        # iteration, which is the wide access `mem_probe.mojo` says is worth
        # less than a byte a thread, so this trades one known cost for another.
        var g = t
        while g < groups:
            var half = group // 2
            var at = row + g * half
            var d = scales[unsafe_offset=d_base + g]
            var m = scales[unsafe_offset=m_base + g] if with_min else Float32(0)
            var qsum = Float32(0)
            var asum = Float32(0)
            var k = 0
            while k < half:
                var b = Int(packed[unsafe_offset=at + k])
                var a0 = x[unsafe_offset=g * group + k * 2]
                var a1 = x[unsafe_offset=g * group + k * 2 + 1]
                qsum += Float32(b & 0xF) * a0
                qsum += Float32((b >> 4) & 0xF) * a1
                asum += a0 + a1
                k += 1
            acc += d * qsum
            comptime if with_min:
                acc += m * asum
            g += tile

    elif variant == V_NO_SCALE:
        var i = t * 2
        while i < cols:
            var b = Int(packed[unsafe_offset=row + (i >> 1)])
            var a0 = x[unsafe_offset=i]
            var a1 = x[unsafe_offset=i + 1]
            acc += Float32(b & 0xF) * a0
            acc += Float32((b >> 4) & 0xF) * a1
            i += tile * 2

    else:
        var i = t * 2
        while i < cols:
            acc += Float32(Int(packed[unsafe_offset=row + (i >> 1)]))
            acc += x[unsafe_offset=i]
            i += tile * 2

    var part = stack_allocation[
        tile, Float32, address_space=AddressSpace.SHARED
    ]()
    part[unsafe_offset=t] = acc
    barrier()
    var step = tile // 2
    while step > 0:
        if t < step:
            part[unsafe_offset=t] = (
                part[unsafe_offset=t] + part[unsafe_offset=t + step]
            )
        barrier()
        step //= 2
    if t == 0:
        o[unsafe_offset=r] = part[unsafe_offset=0]


def _time[
    group: Int, with_min: Bool, form: Int, variant: Int
](
    ctx: DeviceContext,
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    rows: Int,
    cols: Int,
    stride: Int,
    reps: Int,
) raises -> Float64:
    """Nanoseconds a launch, the best of `reps` batches of sixteen."""
    var best = Float64(0)
    for rep in range(reps):
        var at = monotonic()
        for _ in range(16):
            ctx.enqueue_function[
                probe_kernel[TILE, group, with_min, form, variant]
            ](
                w,
                x,
                o,
                Int32(cols),
                Int32(stride),
                grid_dim=(rows, 1, 1),
                block_dim=(TILE, 1, 1),
            )
        ctx.synchronize()
        var took = Float64(monotonic() - at) / 16.0
        if rep == 0 or took < best:
            best = took
    return best


def _report(name: String, ns: Float64, values: Float64, bytes: Float64):
    print(
        name
        + "  "
        + String(Int(ns / 1000.0))
        + " us, "
        + String(Int(values / ns))
        + " G values/s, "
        + String(Int(bytes / ns))
        + " GB/s"
    )


def _sweep[
    kind: Int, group: Int, with_min: Bool, form: Int
](ctx: DeviceContext, label: String, rows: Int, cols: Int) raises:
    var stride = planar_row_bytes(kind, cols)
    var bytes = rows * stride
    var wbuf = ctx.enqueue_create_buffer[DType.uint8](bytes)
    var xbuf = ctx.enqueue_create_buffer[DType.float32](cols)
    var obuf = ctx.enqueue_create_buffer[DType.float32](rows)
    ctx.synchronize()
    var w = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(wbuf.unsafe_ptr())
    )
    var x = Pointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(xbuf.unsafe_ptr())
    )
    var o = Pointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(obuf.unsafe_ptr())
    )

    var values = Float64(rows * cols)
    var read = Float64(bytes)
    print(
        label
        + "  "
        + String(rows)
        + " by "
        + String(cols)
        + ", row "
        + String(stride)
        + " bytes, "
        + String(Int(read / 1048576.0))
        + " MiB"
    )
    _report(
        "  shipped   ",
        _time[group, with_min, form, V_SHIPPED](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )
    _report(
        "  shift     ",
        _time[group, with_min, form, V_SHIFT](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )
    _report(
        "  narrow    ",
        _time[group, with_min, form, V_NARROW](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )
    _report(
        "  per group ",
        _time[group, with_min, form, V_PER_GROUP](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )
    _report(
        "  no scale  ",
        _time[group, with_min, form, V_NO_SCALE](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )
    _report(
        "  loads only",
        _time[group, with_min, form, V_NO_MATH](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )


def main() raises:
    var ctx = DeviceContext()
    print("device  " + String(ctx.name()))

    # The three shapes a Llama 3.1 8B decode spends its matvec time in. The two
    # feed forward ones are 3.5 times the attention one between them and they
    # are where the answer is, but the square one is here because it is the
    # shape every other model has too.
    _sweep[Q_Q4_K, 32, True, QUANT_U4](ctx, "q4_k gate", 14336, 4096)
    _sweep[Q_Q4_K, 32, True, QUANT_U4](ctx, "q4_k down", 4096, 14336)
    _sweep[Q_Q4_K, 32, True, QUANT_U4](ctx, "q4_k attn", 4096, 4096)
