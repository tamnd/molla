"""What one kernel launch costs when the kernel itself costs nothing.

Not part of the library and not part of the test suite. The decode matvec loop
was made twice as fast in `scripts/matvec_probe.mojo` and decode on the model
did not move at all, so the time a decode token spends is somewhere other than
the arithmetic. This asks the smallest question that can distinguish the two:
queue a kernel that reads one float and writes it back, queue it many times,
synchronize once, and divide.

A decode token in molla is on the order of two hundred launches, so a launch
that costs a hundred microseconds is twenty milliseconds a token before any
work happens at all.

Usage:

    mojo run -I src scripts/launch_probe.mojo
"""

from std.gpu import block_idx, thread_idx
from std.time import monotonic

from max.gpu.host import DeviceContext


def touch_kernel(
    o: Pointer[Float32, MutAnyOrigin],
):
    """As little as a kernel can do and still not be deleted."""
    var t = Int(thread_idx.x) + Int(block_idx.x)
    if t == 0:
        o[unsafe_offset=0] = o[unsafe_offset=0] + Float32(1)


def _time(
    ctx: DeviceContext,
    o: Pointer[Float32, MutAnyOrigin],
    count: Int,
    blocks: Int,
    threads: Int,
) raises -> Float64:
    """Nanoseconds a launch, the best of five batches of `count`."""
    var best = Float64(0)
    for rep in range(5):
        var at = monotonic()
        for _ in range(count):
            ctx.enqueue_function[touch_kernel](
                o,
                grid_dim=(blocks, 1, 1),
                block_dim=(threads, 1, 1),
            )
        ctx.synchronize()
        var took = Float64(monotonic() - at) / Float64(count)
        if rep == 0 or took < best:
            best = took
    return best


def _report(name: String, ns: Float64, per_token: Int):
    print(
        name
        + "  "
        + String(Int(ns / 1000.0))
        + "."
        + String(Int(ns / 100.0) % 10)
        + " us a launch, "
        + String(Int(ns * Float64(per_token) / 1000000.0))
        + " ms for "
        + String(per_token)
        + " of them"
    )


def main() raises:
    var ctx = DeviceContext()
    print("device  " + String(ctx.name()))

    var obuf = ctx.enqueue_create_buffer[DType.float32](4)
    ctx.synchronize()
    var o = Pointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(obuf.unsafe_ptr())
    )

    # Three block counts, because a launch on Metal is a command encoder and it
    # is fair to ask whether the cost depends on how much it is encoding.
    _report("  1 block   ", _time(ctx, o, 512, 1, 128), 200)
    _report("  256 blocks", _time(ctx, o, 512, 256, 128), 200)
    _report("  4096 blks ", _time(ctx, o, 512, 4096, 128), 200)

    # And once with a synchronize after every launch, which is what the design
    # is trying not to do, to price what the batching is already saving.
    var at = monotonic()
    for _ in range(256):
        ctx.enqueue_function[touch_kernel](
            o, grid_dim=(1, 1, 1), block_dim=(128, 1, 1)
        )
        ctx.synchronize()
    _report("  sync each ", Float64(monotonic() - at) / 256.0, 200)
