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
from std.memory import AddressSpace, bitcast, stack_allocation
from std.time import monotonic

from max.gpu.host import DeviceContext

from molla.nn.repack import (
    QUANT_I8,
    QUANT_U4,
    planar_row_bytes,
)
from molla.nn.quant import Q_Q4_K, Q_Q8_0

comptime TILE = 128

# What each variant takes away, in the order they are printed.
comptime V_DIVIDE = 0
"""The kernel as it was before #190, with the group index by a signed divide."""
comptime V_SHIFT = 1
"""Same with the group index by a shift, which is what #190 made it.

This is the baseline the rest are against. `V_MAGIC` below is what #186 made it
after that, so the two of them together are the before and after of this file.
"""
comptime V_NARROW = 2
"""Same again, with the loop counter and the group index 32 bits wide."""
comptime V_PER_GROUP = 3
"""A thread walks whole groups, so a scale is loaded once per 32 values."""
comptime V_NO_SCALE = 4
"""The nibbles and the activations, with no scale plane read at all."""
comptime V_NO_MATH = 5
"""The loads and nothing else, which is the floor this shape can reach."""
comptime V_DOT4 = 6
"""Four values a thread against an int8 activation, multiplied as integers."""
comptime V_DOT8 = 7
"""Eight, so the converts and the scale reads are halved again."""
comptime V_DOT16 = 8
"""Sixteen, which is half a group and as far as this can be taken."""
comptime V_MAGIC = 10
"""The shipped loop with the nibble to float conversion done by bit pattern.

`Float32(n)` for a small integer is an `I2F`, which on NVIDIA issues on the same
unit as transcendentals at a quarter of the rate of a multiply. Or the nibble
into the mantissa of `2^23` instead and subtract `2^23` back off, and the whole
thing is an integer or and a floating point subtract, both full rate. It is
exact, it changes no layout and it adds no launch, which is everything the
integer path is not."""
comptime V_MAGIC8 = 15
"""`V_MAGIC4` again with eight, to find where the step width stops paying."""
comptime V_MAGIC4 = 14
"""`V_MAGIC` with four values an iteration instead of two, and nothing else.

The control for `V_MASK16`. That variant changes two things at once, the mask
and the width of a step, and this one changes only the width, so the difference
between the two of them is what the mask is actually worth."""
comptime V_MASK16 = 13
"""The shipped loop with the nibble shifts replaced by masks on a `uint16`.

What #203 asks for, from the Metal q4_K matvec in llama.cpp. Read two bytes at
once, pull the four nibbles out with `0x000F`, `0x00F0`, `0x0F00` and `0xF000`,
and let each one come out sixteen, two hundred and fifty six or four thousand
and ninety six times too large. Nothing corrects it in the loop. Each nibble
position gets its own accumulator and the correction is one multiply by a power
of two per position at the end of the row, which is exact.

Against `V_MAGIC` the mask costs the same as the shift, so the only thing this
can win is the second byte load going away, one load per four values."""
comptime V_BYTE = 11
"""The shipped loop for a byte wide type, which is a different loop entirely."""
comptime V_BYTE_MAGIC = 12
"""Same, with the byte to float conversion done by bit pattern.

Worth asking separately from the nibble case rather than assumed from it. A
signed byte load already sign extends in the hardware, so the loop it replaces
is one convert where the nibble loop's is a mask, a shift, an xor, a subtract
and a convert. The bit pattern version costs the same three instructions in
both. So the nibble case has more to win and the byte case might have nothing,
and q8_0 is the type the smallest model in the bench is in."""
comptime V_DOT8_I16 = 9
"""Eight, against an activation quantized to a signed short rather than a byte.

The win the integer variants show is the converts and the scale reads going
away, not the activation getting smaller, and an activation vector is a few
kilobytes that every block in the launch reads. So this asks what the second
byte costs, and the answer decides whether the accuracy this work spends has to
be spent at all."""


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

    comptime if variant == V_DIVIDE:
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

    elif variant == V_MAGIC:
        var i = t * 2
        while i < cols:
            var gi = i >> shift
            var b = UInt32(packed[unsafe_offset=row + (i >> 1)])
            var lo = bitcast[DType.float32, 1](UInt32(0x4B000000) | (b & 0xF))
            var hi = bitcast[DType.float32, 1](UInt32(0x4B000000) | (b >> 4))
            var a0 = x[unsafe_offset=i]
            var a1 = x[unsafe_offset=i + 1]
            var d = scales[unsafe_offset=d_base + gi]
            acc += d * (lo - Float32(8388608.0)) * a0
            acc += d * (hi - Float32(8388608.0)) * a1
            comptime if with_min:
                var m = scales[unsafe_offset=m_base + gi]
                acc += m * a0
                acc += m * a1
            i += tile * 2

    elif variant == V_MAGIC4 or variant == V_MAGIC8:
        comptime step = 4 if variant == V_MAGIC4 else 8
        var i = t * step
        while i < cols:
            var gi = i >> shift
            var at = row + (i >> 1)
            var d = scales[unsafe_offset=d_base + gi]
            var m = scales[unsafe_offset=m_base + gi] if with_min else Float32(
                0
            )
            comptime for k in range(step // 2):
                var b = UInt32(packed[unsafe_offset=at + k])
                var lo = bitcast[DType.float32, 1](
                    UInt32(0x4B000000) | (b & 0xF)
                )
                var hi = bitcast[DType.float32, 1](
                    UInt32(0x4B000000) | (b >> 4)
                )
                var a0 = x[unsafe_offset=i + k * 2]
                var a1 = x[unsafe_offset=i + k * 2 + 1]
                acc += d * (lo - Float32(8388608.0)) * a0
                acc += d * (hi - Float32(8388608.0)) * a1
                comptime if with_min:
                    acc += m * (a0 + a1)
            i += tile * step

    elif variant == V_MASK16:
        var pairs = w.unsafe_bitcast[UInt16]()
        var acc1 = Float32(0)
        var acc2 = Float32(0)
        var acc3 = Float32(0)
        var i = t * 4
        while i < cols:
            var gi = i >> shift
            var b = UInt32(pairs[unsafe_offset=(row + (i >> 1)) >> 1])
            var n0 = bitcast[DType.float32, 1](UInt32(0x4B000000) | (b & 0xF))
            var n1 = bitcast[DType.float32, 1](UInt32(0x4B000000) | (b & 0xF0))
            var n2 = bitcast[DType.float32, 1](UInt32(0x4B000000) | (b & 0xF00))
            var n3 = bitcast[DType.float32, 1](
                UInt32(0x4B000000) | (b & 0xF000)
            )
            var a0 = x[unsafe_offset=i]
            var a1 = x[unsafe_offset=i + 1]
            var a2 = x[unsafe_offset=i + 2]
            var a3 = x[unsafe_offset=i + 3]
            var d = scales[unsafe_offset=d_base + gi]
            acc += d * (n0 - Float32(8388608.0)) * a0
            acc1 += d * (n1 - Float32(8388608.0)) * a1
            acc2 += d * (n2 - Float32(8388608.0)) * a2
            acc3 += d * (n3 - Float32(8388608.0)) * a3
            comptime if with_min:
                var m = scales[unsafe_offset=m_base + gi]
                acc += m * (a0 + a1)
                acc += m * (a2 + a3)
            i += tile * 4
        acc += acc1 * Float32(0.0625)
        acc += acc2 * Float32(0.00390625)
        acc += acc3 * Float32(0.000244140625)

    elif variant == V_BYTE:
        var quants = w.unsafe_bitcast[Int8]()
        var i = t
        while i < cols:
            var gi = i >> shift
            var q = Float32(Int(quants[unsafe_offset=row + i]))
            var a = x[unsafe_offset=i]
            acc += scales[unsafe_offset=d_base + gi] * q * a
            comptime if with_min:
                acc += scales[unsafe_offset=m_base + gi] * a
            i += tile

    elif variant == V_BYTE_MAGIC:
        var i = t
        while i < cols:
            var gi = i >> shift
            var u = UInt32(packed[unsafe_offset=row + i]) ^ 0x80
            var q = bitcast[DType.float32, 1](UInt32(0x4B000000) | u)
            var a = x[unsafe_offset=i]
            acc += (
                scales[unsafe_offset=d_base + gi] * (q - Float32(8388736.0)) * a
            )
            comptime if with_min:
                acc += scales[unsafe_offset=m_base + gi] * a
            i += tile

    elif variant == V_NO_SCALE:
        var i = t * 2
        while i < cols:
            var b = Int(packed[unsafe_offset=row + (i >> 1)])
            var a0 = x[unsafe_offset=i]
            var a1 = x[unsafe_offset=i + 1]
            acc += Float32(b & 0xF) * a0
            acc += Float32((b >> 4) & 0xF) * a1
            i += tile * 2

    elif variant == V_NO_MATH:
        var i = t * 2
        while i < cols:
            acc += Float32(Int(packed[unsafe_offset=row + (i >> 1)]))
            acc += x[unsafe_offset=i]
            i += tile * 2

    else:
        # What #186 proposes. The activation vector arrives already quantized to
        # a signed byte per value with a scale a group, so a weight and an
        # activation are both small integers and their product is an integer.
        # Two things go away. The four `Int32` to `Float32` converts a byte pair
        # costs now become one per run, and on NVIDIA a convert is a quarter
        # rate instruction where a multiply is full rate, so four of them are
        # worth sixteen multiplies. And the group's scale is read once for the
        # run rather than once for a pair.
        #
        # A run has to sit inside one group for that to work, which it does:
        # every group here is 32 values and 4 and 8 both divide it.
        #
        # The minimum is the second integer sum. Over a group, sum of
        # `(d * w + m) * x` is `d * dx * sum(w * q) + m * dx * sum(q)`, so a
        # `with_min` type costs one more integer accumulator and one more
        # convert per run rather than a multiply and an add per value.
        comptime run = 4 if variant == V_DOT4 else (
            16 if variant == V_DOT16 else 8
        )
        comptime wide = variant == V_DOT8_I16
        var qx = x.unsafe_bitcast[Int8]()
        var qw = x.unsafe_bitcast[Int16]()
        var xd = x.unsafe_bitcast[Float32]()
        var xd_base = cols  # the activation scales, a float a group
        var i = t * run
        while i < cols:
            var gi = i >> shift
            var at = row + (i >> 1)
            var dot = Int32(0)
            var qsum = Int32(0)
            var k = 0
            while k < run // 2:
                var b = Int32(packed[unsafe_offset=at + k])
                var lo = b & 0xF
                var hi = (b >> 4) & 0xF
                var q0: Int32
                var q1: Int32
                comptime if wide:
                    q0 = Int32(qw[unsafe_offset=i + k * 2])
                    q1 = Int32(qw[unsafe_offset=i + k * 2 + 1])
                else:
                    q0 = Int32(qx[unsafe_offset=i + k * 2])
                    q1 = Int32(qx[unsafe_offset=i + k * 2 + 1])
                dot += lo * q0 + hi * q1
                comptime if with_min:
                    qsum += q0 + q1
                k += 1
            var dx = xd[unsafe_offset=xd_base + gi]
            acc += scales[unsafe_offset=d_base + gi] * dx * Float32(dot)
            comptime if with_min:
                acc += scales[unsafe_offset=m_base + gi] * dx * Float32(qsum)
            i += tile * run

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
    # The float variants want `cols` floats. The integer ones read the same
    # bytes as `cols` signed bytes and then want a float a group behind them, at
    # float index `cols`, so the buffer carries both layouts at once and no
    # variant reads past the end of it.
    var xbuf = ctx.enqueue_create_buffer[DType.float32](
        cols + cols // group + 4
    )
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
        "  divide    ",
        _time[group, with_min, form, V_DIVIDE](
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
    _report(
        "  magic cvt ",
        _time[group, with_min, form, V_MAGIC](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )
    _report(
        "  magic x4  ",
        _time[group, with_min, form, V_MAGIC4](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )
    _report(
        "  magic x8  ",
        _time[group, with_min, form, V_MAGIC8](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )
    _report(
        "  mask16    ",
        _time[group, with_min, form, V_MASK16](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )
    _report(
        "  int dot 4 ",
        _time[group, with_min, form, V_DOT4](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )
    _report(
        "  int dot 8 ",
        _time[group, with_min, form, V_DOT8](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )
    _report(
        "  int16 dot8",
        _time[group, with_min, form, V_DOT8_I16](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )
    _report(
        "  int dot 16",
        _time[group, with_min, form, V_DOT16](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )


def _sweep_byte[
    kind: Int, group: Int, with_min: Bool
](ctx: DeviceContext, label: String, rows: Int, cols: Int) raises:
    """The two byte wide variants, on their own, because they read differently.
    """
    comptime form = QUANT_I8
    var stride = planar_row_bytes(kind, cols)
    var bytes = rows * stride
    var wbuf = ctx.enqueue_create_buffer[DType.uint8](bytes)
    var xbuf = ctx.enqueue_create_buffer[DType.float32](
        cols + cols // group + 4
    )
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
        "  byte cvt  ",
        _time[group, with_min, form, V_BYTE](
            ctx, w, x, o, rows, cols, stride, 5
        ),
        values,
        read,
    )
    _report(
        "  byte magic",
        _time[group, with_min, form, V_BYTE_MAGIC](
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

    # And the byte wide type, at the shape SmolLM2 135M spends its decode in,
    # because that is the model the milestone is measured on and it is q8_0.
    _sweep_byte[Q_Q8_0, 32, False](ctx, "q8_0 gate", 1536, 576)
    _sweep_byte[Q_Q8_0, 32, False](ctx, "q8_0 head", 49152, 576)
