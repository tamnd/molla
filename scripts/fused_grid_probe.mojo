"""How wide a grid the fused kernel can hold, and what a barrier costs there.

`METAL_PER_SM` is one because one block a multiprocessor is the only count every
Metal device is known to hold, and the first measurement of the fused path says
that is far too narrow: SmolLM2 decodes slower fused than unfused on an M4, and
the obvious reason is that the unfused matvec launches a block a row, 576 or
1536 of them, where the fused one has ten in total.

Raising it is a measurement rather than a guess, for the reason in
`METAL_PER_SM`'s docstring: a grid that is not resident deadlocks, and a
deadlock on a laptop is the display. So this sweeps the block count with nothing
but the rendezvous in the kernel, which is what `fused_selftest` does at one
grid, and reports for each count whether every block met and what a round cost.
A count that reports blocks giving up is a count that is not resident and is the
top of the usable range.

    pixi run -- mojo run -I src scripts/fused_grid_probe.mojo
"""

from std.sys.info import has_accelerator
from std.time import monotonic

from max.gpu.host import DeviceAttribute, DeviceContext

from molla.nn.gpu_fused import FTILE, barrier_probe_kernel

comptime ROUNDS = 200
"""Barrier rounds a measured launch, which is close to one token's worth.

A thirty layer token is 243 rounds, so this times the thing at the scale it is
going to run at rather than at a scale where the launch dominates.
"""


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator in this build, nothing to probe")
        return
    else:
        var ctx = DeviceContext()
        var sms = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        print("multiprocessors  " + String(sms))
        print("threads a block  " + String(FTILE))
        print("")
        print("| blocks | a multiprocessor | a round | gave up |")
        print("| --- | --- | --- | --- |")

        var sync = ctx.enqueue_create_buffer[DType.int32](3)
        var host = ctx.enqueue_create_host_buffer[DType.int32](3)
        var sp = Pointer[Int32, MutAnyOrigin](
            unsafe_from_address=Int(sync.unsafe_ptr())
        )

        var per = 1
        while per <= 64:
            var blocks = sms * per

            # One warm launch that is not timed, and then the best of three, so
            # that what comes out is the rendezvous rather than a cold buffer
            # or whatever else this shared machine was doing at the time.
            var best = Float64(0)
            for rep in range(4):
                ctx.enqueue_memset(sync, Int32(0))
                ctx.synchronize()
                var at = monotonic()
                ctx.enqueue_function[barrier_probe_kernel](
                    sp,
                    Int32(blocks),
                    Int32(ROUNDS),
                    grid_dim=(blocks, 1, 1),
                    block_dim=(FTILE, 1, 1),
                )
                ctx.synchronize()
                var took = Float64(monotonic() - at)
                if rep == 1 or took < best:
                    best = took
            ctx.enqueue_copy(host, sync)
            ctx.synchronize()

            var gave_up = Int(host[2])
            print(
                "| "
                + String(blocks)
                + " | "
                + String(per)
                + " | "
                + String(best / Float64(ROUNDS) / 1000.0)
                + " us | "
                + String(gave_up)
                + " |"
            )
            if gave_up != 0:
                print("")
                print("stopping, that grid is not resident")
                return
            per *= 2
