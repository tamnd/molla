"""How fast a block per row kernel can read a buffer, and what changes that.

Not part of the library and not part of the test suite. It exists because the
packed quant plane halved the bytes the 8B decode reads and did not make it any
faster, and every explanation for that is about the memory system rather than
about the model. This reads a buffer with the same shape of access the matvec
uses, one block per row, `tile` threads walking the row, and reports what it
got, so the question of how many bytes a second this arrangement is worth can
be answered in a second rather than in a model load.

Three things vary and each one is a column in the output. How wide a load each
thread issues, one byte or four. How long a row is, which is the thing that
changed under the matvec when the quant plane was packed. And how many rows
there are, which is the block count.

It runs on whichever accelerator is the default one, since a machine with two
of them is not a machine anyone here measures on.

Usage:

    mojo run -I src scripts/mem_probe.mojo
"""

from std.gpu import block_idx, thread_idx
from std.time import monotonic

from max.gpu import barrier
from max.gpu.host import DeviceContext
from std.memory import AddressSpace, stack_allocation

comptime TILE = 128


def sum_kernel[
    tile: Int, wide: Int
](
    w: Pointer[UInt8, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    row_bytes_dev: Int32,
):
    """Sum a row of bytes, one block to a row, and write the total.

    The sum is real work so the loads cannot be dropped, and it is the only
    work, so what this measures is the reading.
    """
    var row_bytes = Int(row_bytes_dev)
    var r = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var base = r * row_bytes

    var acc = Float32(0)
    comptime if wide == 1:
        var i = t
        while i < row_bytes:
            acc += Float32(Int(w[unsafe_offset=base + i]))
            i += tile
    else:
        var words = w.unsafe_bitcast[UInt32]()
        var word_base = base // 4
        var count = row_bytes // 4
        var i = t
        while i < count:
            var word = Int(words[unsafe_offset=word_base + i])
            comptime for k in range(4):
                acc += Float32((word >> (8 * k)) & 0xFF)
            i += tile

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


def _run[
    wide: Int
](ctx: DeviceContext, rows: Int, row_bytes: Int, reps: Int) raises -> Float64:
    """Nanoseconds a repetition, the best of `reps`, so a slow start is not the
    answer."""
    var bytes = rows * row_bytes
    var buf = ctx.enqueue_create_buffer[DType.uint8](bytes)
    var out = ctx.enqueue_create_buffer[DType.float32](rows)
    ctx.synchronize()
    var w = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(buf.unsafe_ptr())
    )
    var o = Pointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(out.unsafe_ptr())
    )

    var best = Float64(0)
    for rep in range(reps):
        var at = monotonic()
        ctx.enqueue_function[sum_kernel[TILE, wide]](
            w,
            o,
            Int32(row_bytes),
            grid_dim=(rows, 1, 1),
            block_dim=(TILE, 1, 1),
        )
        ctx.synchronize()
        var took = Float64(monotonic() - at)
        if rep == 0 or took < best:
            best = took
    return best


def _line(ctx: DeviceContext, label: String, rows: Int, row_bytes: Int) raises:
    var bytes = Float64(rows * row_bytes)
    var one = _run[1](ctx, rows, row_bytes, 5)
    var four = _run[4](ctx, rows, row_bytes, 5)
    print(
        label
        + "  rows "
        + String(rows)
        + ", row "
        + String(row_bytes)
        + " bytes, "
        + String(Int(bytes / 1048576.0))
        + " MiB: one byte a thread "
        + String(Int(bytes / one))
        + " GB/s, four "
        + String(Int(bytes / four))
        + " GB/s"
    )


def main() raises:
    var ctx = DeviceContext()
    print("device  " + String(ctx.name()))

    # Two gibibytes every time, so the only thing that changes down the list is
    # how long a row is and therefore how many blocks the launch has in it. Two
    # gibibytes because a 4090 has 72 MiB of L2 and a run that fits in it
    # measures the cache rather than the memory.
    comptime total = 2 * 1024 * 1024 * 1024
    _line(ctx, "512    ", total // 512, 512)
    _line(ctx, "1024   ", total // 1024, 1024)
    _line(ctx, "2048   ", total // 2048, 2048)
    _line(ctx, "4096   ", total // 4096, 4096)
    _line(ctx, "8192   ", total // 8192, 8192)
    _line(ctx, "16384  ", total // 16384, 16384)
