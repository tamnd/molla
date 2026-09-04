"""Whether a grid wide barrier works here, and what one round of it costs.

Not part of the library and not part of the test suite. It exists because #170
wants a token to be one kernel launch that walks the layers internally, and that
needs every block in the grid to be able to wait for every other block between
stages. `max.gpu` has no such barrier. It has block level `barrier`, and
`max.gpu.sync.Semaphore`, which is a one writer lock rather than a rendezvous
and is NVIDIA only besides. So the barrier has to be built out of device scope
atomics, and before anything is built on top of one the question is whether it
works on both backends and whether it is cheaper than the launch it replaces.

Two things are measured. Whether a barrier round actually orders plain global
memory, which is checked by having every block write a value, wait, and read its
neighbour's, and whether the launch can be sized so that every block is resident,
which is what stops a barrier deadlocking on blocks that have not started. The
occupancy query is the second half of that and it is allowed to be missing on a
backend, which is itself the answer for that backend.

The spin is bounded. A barrier that does not work would otherwise hang the GPU
rather than report anything, and on a laptop that is the display.

Usage:

    mojo run -I src scripts/gridsync_probe.mojo
"""

from std.atomic import Atomic, Ordering
from std.gpu import block_idx, thread_idx
from std.sys import llvm_intrinsic
from std.sys.info import CompilationTarget
from std.time import monotonic

from max.gpu import barrier
from max.gpu.host import DeviceAttribute, DeviceContext

comptime TILE = 128
"""Threads a block, which the occupancy query is asked about."""

comptime PATIENCE = 200000000
"""How long a block waits at the barrier before deciding it is broken.

A number rather than forever, because forever is a hung GPU and a probe that
reports nothing. It is large enough that no working barrier reaches it.
"""

comptime dev = Atomic[DType.int32, scope="device"]

# An Apple GPU has no acquire, no release and no acquire release on an atomic,
# and no memory fence of any ordering at all: every one of them crashes the
# Metal pipeline compiler rather than failing to compile. So on that target the
# rendezvous is relaxed atomics and the ordering is carried by a threadgroup
# barrier asked for with the device memory flag, which is the same instruction
# `barrier` already emits with a different flag word. On CUDA the ordering
# rides the atomic, which is one instruction rather than three.
comptime RMW = (
    Ordering.RELAXED if CompilationTarget.is_macos() else Ordering.ACQUIRE_RELEASE
)
comptime GET = (
    Ordering.RELAXED if CompilationTarget.is_macos() else Ordering.ACQUIRE
)
comptime PUT = (
    Ordering.RELAXED if CompilationTarget.is_macos() else Ordering.RELEASE
)


def flush():
    """A block level barrier that also orders this block's device memory.

    On Apple this is `threadgroup_barrier(mem_flags::mem_device |
    mem_flags::mem_threadgroup)`, which is the only way to order device memory
    on that target from here. Everywhere else the ordering is on the atomics
    and this is the ordinary block barrier.
    """
    comptime if CompilationTarget.is_macos():
        llvm_intrinsic["llvm.air.wg.barrier", NoneType](Int32(3), Int32(1))
    else:
        barrier()


def grid_barrier(
    count: Pointer[Int32, MutAnyOrigin],
    gen: Pointer[Int32, MutAnyOrigin],
    blocks: Int,
    mut seen: Int32,
) -> Bool:
    """Wait until every block in the grid has arrived. False if it gave up.

    The sense reversing barrier. Every block adds one to a counter, the block
    that finds itself last resets the counter and moves the generation on, and
    everyone else waits for the generation to move. The counter is reset before
    the generation moves so that the next round starts from zero, which is safe
    because nobody is past the generation yet.
    """
    flush()
    var ok = True
    if thread_idx.x == 0:
        var was = dev.fetch_add[ordering=RMW](count, 1)
        if was == Int32(blocks - 1):
            dev.store[ordering=Ordering.RELAXED](count, Int32(0))
            dev.store[ordering=PUT](gen, seen + 1)
        else:
            var spins = 0
            while dev.load[ordering=GET](gen) == seen:
                spins += 1
                if spins > PATIENCE:
                    ok = False
                    break
    flush()
    seen += 1
    return ok


def ring_kernel[
    atomic_data: Bool
](
    count: Pointer[Int32, MutAnyOrigin],
    gen: Pointer[Int32, MutAnyOrigin],
    data: Pointer[Int32, MutAnyOrigin],
    errs: Pointer[Int32, MutAnyOrigin],
    blocks_dev: Int32,
    rounds_dev: Int32,
):
    """Every block writes its own slot and reads its neighbour's, `rounds` times.

    Without a working barrier the read either sees the previous round's value or
    a value the neighbour has not written yet, so the count of wrong reads is
    the check.

    The parameter is what the write and the read are made of. False is ordinary
    global traffic, which is what a fused layer would be doing and so the case
    that decides whether this shape is usable at all. True is a relaxed device
    scope atomic on the same address, which on a target with no fence is the
    other thing left to try. Running both says whether a backend that fails the
    first passes the second, and the difference between the two is the price of
    making it work there.
    """
    var blocks = Int(blocks_dev)
    var b = Int(block_idx.x)
    var seen = Int32(0)
    var bad = 0
    var stuck = 0

    for r in range(Int(rounds_dev)):
        if thread_idx.x == 0:
            comptime if atomic_data:
                # `Pointer(to=...)` and not a pointer rebuilt from an integer
                # address. The second form crashes the Metal pipeline compiler
                # the moment an atomic touches it, presumably because the
                # address space is lost on the way through the integer.
                dev.store[ordering=Ordering.RELAXED](
                    Pointer[Int32, MutAnyOrigin](to=data[unsafe_offset=b]),
                    Int32(r * 1000 + b),
                )
            else:
                data[unsafe_offset=b] = Int32(r * 1000 + b)
        if not grid_barrier(count, gen, blocks, seen):
            stuck += 1
            break
        var nb = (b + 1) % blocks
        if thread_idx.x == 0:
            var saw = Int32(0)
            comptime if atomic_data:
                saw = dev.load[ordering=Ordering.RELAXED](
                    Pointer[Int32, MutAnyOrigin](to=data[unsafe_offset=nb])
                )
            else:
                saw = data[unsafe_offset=nb]
            if saw != Int32(r * 1000 + nb):
                bad += 1
        # A second round, because without it a fast block would be writing the
        # next round's value into its slot while a slow one is still reading it.
        if not grid_barrier(count, gen, blocks, seen):
            stuck += 1
            break

    if thread_idx.x == 0:
        if bad != 0:
            _ = dev.fetch_add(errs, Int32(bad))
        if stuck != 0:
            _ = dev.fetch_add(errs, Int32(1000000))


def verdict(label: String, wrong: Int, reads: Int):
    """What the ring kernel's error count means, in words."""
    if wrong >= 1000000:
        print(label + "  a block gave up waiting, there is no rendezvous here")
    elif wrong != 0:
        print(
            label
            + "  "
            + String(wrong)
            + " wrong reads of "
            + String(reads)
            + ", the barrier does not order this traffic"
        )
    else:
        print(
            label
            + "  "
            + String(reads)
            + " reads, every one saw this round's write"
        )


def spin_kernel(
    count: Pointer[Int32, MutAnyOrigin],
    gen: Pointer[Int32, MutAnyOrigin],
    errs: Pointer[Int32, MutAnyOrigin],
    blocks_dev: Int32,
    rounds_dev: Int32,
):
    """Nothing but barriers, so what it times is the barrier."""
    var blocks = Int(blocks_dev)
    var seen = Int32(0)
    for _ in range(Int(rounds_dev)):
        if not grid_barrier(count, gen, blocks, seen):
            if thread_idx.x == 0:
                _ = dev.fetch_add(errs, Int32(1000000))
            return


def main() raises:
    var ctx = DeviceContext()
    print("device  " + String(ctx.name()))

    var sms = 0
    try:
        sms = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    except e:
        print("multiprocessor count  unavailable, " + String(e))
    print("multiprocessors  " + String(sms))

    var plain = ctx.compile_function[ring_kernel[False]]()
    var atomic = ctx.compile_function[ring_kernel[True]]()
    var spin = ctx.compile_function[spin_kernel]()

    var per_sm = 0
    try:
        per_sm = plain.occupancy_max_active_blocks_per_multiprocessor(TILE, 0)
    except e:
        print("occupancy  unavailable, " + String(e))
    print(
        "blocks a multiprocessor at "
        + String(TILE)
        + " threads  "
        + String(per_sm)
    )

    # Without the occupancy query there is no size that is known to be resident,
    # so the probe falls back to one block a multiprocessor, which every backend
    # that runs at all can hold.
    var blocks = per_sm * sms if per_sm > 0 and sms > 0 else 0
    if blocks == 0:
        blocks = sms if sms > 0 else 8
        print("falling back to one block a multiprocessor")
    print("blocks in the grid  " + String(blocks))

    var count = ctx.enqueue_create_buffer[DType.int32](1)
    var gen = ctx.enqueue_create_buffer[DType.int32](1)
    var data = ctx.enqueue_create_buffer[DType.int32](blocks)
    var errs = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(count, Int32(0))
    ctx.enqueue_memset(gen, Int32(0))
    ctx.enqueue_memset(data, Int32(-1))
    ctx.enqueue_memset(errs, Int32(0))
    ctx.synchronize()

    var cp = Pointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(count.unsafe_ptr())
    )
    var gp = Pointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(gen.unsafe_ptr())
    )
    var dp = Pointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(data.unsafe_ptr())
    )
    var ep = Pointer[Int32, MutAnyOrigin](
        unsafe_from_address=Int(errs.unsafe_ptr())
    )

    comptime rounds = 64
    var host = ctx.enqueue_create_host_buffer[DType.int32](1)

    ctx.enqueue_function(
        plain,
        cp,
        gp,
        dp,
        ep,
        Int32(blocks),
        Int32(rounds),
        grid_dim=(blocks, 1, 1),
        block_dim=(TILE, 1, 1),
    )
    ctx.enqueue_copy(host, errs)
    ctx.synchronize()
    verdict("plain loads and stores ", Int(host[0]), rounds * blocks)

    # The counters have to go back to zero, because the first kernel left the
    # generation wherever it finished and every block starts the next one at
    # zero again.
    ctx.enqueue_memset(count, Int32(0))
    ctx.enqueue_memset(gen, Int32(0))
    ctx.enqueue_memset(data, Int32(-1))
    ctx.enqueue_memset(errs, Int32(0))
    ctx.enqueue_function(
        atomic,
        cp,
        gp,
        dp,
        ep,
        Int32(blocks),
        Int32(rounds),
        grid_dim=(blocks, 1, 1),
        block_dim=(TILE, 1, 1),
    )
    ctx.enqueue_copy(host, errs)
    ctx.synchronize()
    verdict("relaxed device atomics", Int(host[0]), rounds * blocks)

    # The cost of a round, with nothing else in the kernel. Best of five, so a
    # cold start is not the answer, and against the launch it would replace.
    #
    # Swept over the grid, because a rendezvous costs more the more blocks are
    # in it and the fused kernel would size its grid to the work rather than to
    # residency. The full grid is the ceiling and the useful number is whichever
    # row matches the shape a layer wants.
    comptime spins = 1000
    var wide = blocks
    while wide >= 1:
        var best = Float64(0)
        for rep in range(5):
            ctx.enqueue_memset(count, Int32(0))
            ctx.enqueue_memset(gen, Int32(0))
            ctx.synchronize()
            var at = monotonic()
            ctx.enqueue_function(
                spin,
                cp,
                gp,
                ep,
                Int32(wide),
                Int32(spins),
                grid_dim=(wide, 1, 1),
                block_dim=(TILE, 1, 1),
            )
            ctx.synchronize()
            var took = Float64(monotonic() - at)
            if rep == 0 or took < best:
                best = took
        print(
            "a barrier round over "
            + String(wide)
            + " blocks  "
            + String(best / Float64(spins) / 1000.0)
            + " microseconds"
        )
        if wide == 1:
            break
        wide //= 4
        if wide < 1:
            wide = 1

    # What it is being compared against: an empty launch, which is what a stage
    # of a layer costs today.
    var empty = Float64(0)
    comptime launches = 200
    for rep in range(5):
        var at = monotonic()
        for _ in range(launches):
            ctx.enqueue_function(
                spin,
                cp,
                gp,
                ep,
                Int32(blocks),
                Int32(0),
                grid_dim=(blocks, 1, 1),
                block_dim=(TILE, 1, 1),
            )
        ctx.synchronize()
        var took = Float64(monotonic() - at)
        if rep == 0 or took < empty:
            empty = took
    print(
        "an empty launch  "
        + String(empty / Float64(launches) / 1000.0)
        + " microseconds, over "
        + String(launches)
        + " launches"
    )
