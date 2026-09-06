"""The device matvec, and everything it refuses before it launches.

Split in two on purpose. The refusals are integer and string comparisons and
they run on all five machines, because they are what stops a mis-wired engine
from handing a device kernel a host address, and that is a mistake that has to
be caught on the machines that cannot reproduce it as much as on the ones that
can.

The matvec itself only runs where there is a GPU, which is two of the five. It
is a small tensor with numbers chosen by hand rather than a corpus, because the
corpus lives in `scripts/kernel_oracle.mojo` where it can print the error per
format instead of asserting a bound. What this one is for is that the suite
itself launches a kernel on the machines that have one, so a change that breaks
the launch shows up in `pixi run check` and not only in a script somebody has to
remember to run.
"""

from std.memory import bitcast
from std.sys.info import (
    CompilationTarget,
    has_accelerator,
    has_nvidia_gpu_accelerator,
)

from max.gpu.host import DeviceContext

from harness import Suite

from molla.model.load import DevicePool
from molla.nn.gpu import (
    DeviceVec,
    check_matvec,
    device_matmul_into,
    device_matvec,
    device_ready,
)
from molla.nn.quant import Q_F32, Q_Q8_0
from molla.nn.repack import (
    LAYOUT_PLANAR,
    SCALE_BYTES,
    planar_row_bytes,
    planar_row_dot,
)
from molla.nn.tensor import (
    WHERE_DEVICE,
    WHERE_HOST,
    WHERE_UNIFIED,
    Buffer,
    Tensor,
)
from molla.sys.device import default_device
from molla.sys.mem import keep
from molla.sys.mmap import RawPtr


def run(mut suite: Suite) raises:
    """Everything that needs no device, which is the refusals and the skip.

    The launches are in `run_on_device`, which `main` calls with the one
    context the process owns. A CUDA process gets one `DeviceContext` and hangs
    on the first allocation against a second, so no test module may make its
    own.
    """
    test_refusals(suite)
    comptime if not has_accelerator():
        suite.group("device matvec")
        suite.check(True, "skipped, this build has no device code in it")


def run_on_device(mut suite: Suite, ctx: DeviceContext) raises:
    test_matvec(suite, ctx)
    test_tile(suite, ctx)
    test_pool(suite, ctx)


def _planar(cols: Int, rows: Int) -> Tensor:
    """A q8_0 shaped planar weight in a pool, at a made up address."""
    return Tensor(0x1000, Q_Q8_0, cols, rows, LAYOUT_PLANAR, WHERE_DEVICE)


def test_refusals(mut suite: Suite) raises:
    suite.group("device matvec refusals")

    var good = _planar(64, 4)
    var x = Buffer(64)
    check_matvec(good, x, 4)
    suite.check(True, "a planar weight in a pool with matching shapes is fine")

    var raised = False
    var message = String("")
    try:
        check_matvec(Tensor(0x1000, Q_Q8_0, 64, 4), x, 4)
    except e:
        raised = True
        message = String(e)
    suite.check(raised, "a weight still in ggml blocks is refused")
    suite.check(
        "repacked" in message,
        "and the message says repacking is what it wants",
    )

    raised = False
    try:
        check_matvec(
            Tensor(0x1000, Q_F32, 64, 4, LAYOUT_PLANAR, WHERE_DEVICE), x, 4
        )
    except:
        raised = True
    suite.check(raised, "a type with no planar form is refused")

    # The one that matters. A device kernel handed a host address does not
    # fault, it reads zeros, so a model bound that way runs at full speed and
    # answers with noise.
    raised = False
    message = String("")
    try:
        check_matvec(
            Tensor(0x1000, Q_Q8_0, 64, 4, LAYOUT_PLANAR, WHERE_HOST), x, 4
        )
    except e:
        raised = True
        message = String(e)
    suite.check(raised, "a weight in host memory is refused")
    suite.check(
        "zeros" in message,
        "and the message says what reading it anyway would have done",
    )

    raised = False
    try:
        check_matvec(
            Tensor(0x1000, Q_Q8_0, 64, 4, LAYOUT_PLANAR, WHERE_UNIFIED), x, 4
        )
    except:
        raised = True
    suite.check(
        raised, "and so is a unified one, which has no device address to give"
    )

    raised = False
    try:
        check_matvec(good, Buffer(32), 4)
    except:
        raised = True
    suite.check(raised, "an input of the wrong width is refused")

    raised = False
    try:
        check_matvec(good, x, 7)
    except:
        raised = True
    suite.check(raised, "and so is an output of the wrong height")

    suite.check(
        device_ready() == has_accelerator(),
        "and whether there is a device at all is a property of the build",
    )


def test_matvec(mut suite: Suite, ctx: DeviceContext) raises:
    """One matvec on whatever GPU this machine has.

    q8_0 shaped because its planar form has one scale plane and no minimum,
    which is the instantiation the three other types that centre their quants
    also use, and because signed quants are what the kernel gets wrong when a
    byte is widened rather than reinterpreted.
    """
    suite.group("device matvec")

    # A machine with no device never reaches this, `run` having reported the
    # skip, and the branch is what keeps the body out of that build.
    comptime if not has_accelerator():
        return
    else:
        var cols = 64
        var rows = 5
        var stride = planar_row_bytes(Q_Q8_0, cols)
        var total = stride * rows

        var x = Buffer(cols)
        for i in range(cols):
            x.data[i] = (Float32((i * 11) % 29) - 14.0) / 7.0

        var pool = ctx.enqueue_create_buffer[DType.uint8](total)
        var want = Buffer(rows)
        # The weight is written straight into the pool's host mapping and the
        # reference is read back out of the same mapping, so there is one copy
        # of these bytes and no chance of the host and the device being handed
        # two different rows that were meant to be the same.
        with pool.map_to_host() as mapped:
            var p = RawPtr(unsafe_from_address=Int(mapped.unsafe_ptr()))
            # Quants that run either side of zero, and a scale that differs per
            # group and per row, so a row stride that is off by a row and a
            # group index that is off by one both change the answer.
            for r in range(rows):
                var row = r * stride
                for i in range(cols):
                    var q = ((i * 7 + r * 13) % 251) - 125
                    p.unsafe_store(row + i, UInt8(q & 0xFF))
                for g in range(cols // 32):
                    var scale = Float32(0.125) * Float32(g + 1) / Float32(r + 1)
                    var bits = bitcast[DType.uint16, 1](
                        scale.cast[DType.float16]()
                    )
                    for b in range(SCALE_BYTES):
                        p.unsafe_store(
                            row + cols + g * SCALE_BYTES + b,
                            UInt8((bits >> UInt16(b * 8)) & 0xFF),
                        )
            for r in range(rows):
                want.data[r] = planar_row_dot(
                    Q_Q8_0, p, r * stride, x.data, 0, cols
                )

        var resident = Tensor(
            Int(pool.unsafe_ptr()),
            Q_Q8_0,
            cols,
            rows,
            LAYOUT_PLANAR,
            WHERE_DEVICE,
        )
        var got = Buffer(rows)
        device_matvec(ctx, resident, x, got)
        # The tensor holds the pool's address and not the pool, so without this
        # the buffer is freed at the line that read its pointer, before the
        # kernel that reads it has run, and every row comes back exactly zero.
        keep(pool)

        var peak = Float32(0)
        var worst = Float32(0)
        var nonzero = 0
        for r in range(rows):
            var m = want.data[r] if want.data[r] > 0 else -want.data[r]
            if m > peak:
                peak = m
            var gap = got.data[r] - want.data[r]
            if gap < 0:
                gap = -gap
            if gap > worst:
                worst = gap
            if got.data[r] != 0.0:
                nonzero += 1

        suite.check(peak > 0, "the reference is not all zeros")
        suite.check(
            worst <= peak * Float32(1e-5),
            "and the device agrees with the host inside a tenth of a per mille",
        )
        suite.check(
            nonzero == rows,
            "every row came back, which a freed pool would not have given",
        )


comptime TILE_COLS = 128
comptime TILE_ROWS = 4096
comptime TILE_TOKENS = 256
"""A matmul big enough to reach the matrix core kernel on both backends.

Nothing else in the suite does. The batched path is otherwise checked against
the decode path on a synthetic model of sixty four columns and ninety six rows,
which is three orders of magnitude below the line `NV_MIN_WORK` draws, so on
CUDA every batched matmul in every other test runs the ordinary kernel and the
tensor core tile has no coverage at all. Two hundred and fifty six tokens by
four thousand and ninety six rows is that line exactly.

A hundred and twenty eight columns rather than sixty four so the reduction loop
runs twice, because a kernel that stages one step correctly and advances the
weight pointer wrongly passes at one step.
"""


def test_tile(mut suite: Suite, ctx: DeviceContext) raises:
    """The batched matmul at a shape that reaches the matrix core kernel.

    The tolerance is per backend and that is the point of the group. Apple
    simdgroup matrices multiply in float here, so the gate is the one every
    other kernel in molla is held to. NVIDIA tensor cores take half precision
    operands and there is no float shape to fall back on, since tf32 has the
    same ten bit mantissa at half the rate, so a dot product of them agrees with
    a float one to a few times 1e-4 and no better whatever the kernel does.
    Holding that path to 1e-5 would be holding it to something the hardware
    cannot do, and holding it to nothing would be no test, so it is 2e-3 and the
    number it actually reaches is printed when it fails.
    """
    suite.group("device matmul tile")

    comptime if not has_accelerator():
        return
    else:
        var stride = planar_row_bytes(Q_Q8_0, TILE_COLS)
        var total = stride * TILE_ROWS

        var x = Buffer(TILE_TOKENS * TILE_COLS)
        for t in range(TILE_TOKENS):
            for i in range(TILE_COLS):
                x.data[t * TILE_COLS + i] = (
                    Float32((i * 11 + t * 5) % 29) - 14.0
                ) / 7.0

        var pool = ctx.enqueue_create_buffer[DType.uint8](total)
        var want = Buffer(TILE_TOKENS * TILE_ROWS)
        with pool.map_to_host() as mapped:
            var p = RawPtr(unsafe_from_address=Int(mapped.unsafe_ptr()))
            for r in range(TILE_ROWS):
                var row = r * stride
                for i in range(TILE_COLS):
                    var q = ((i * 7 + r * 13) % 251) - 125
                    p.unsafe_store(row + i, UInt8(q & 0xFF))
                for g in range(TILE_COLS // 32):
                    var scale = Float32(0.02) + Float32((g + r) % 5) * 0.003
                    var bits = bitcast[DType.uint16, 1](
                        scale.cast[DType.float16]()
                    )
                    for b in range(SCALE_BYTES):
                        p.unsafe_store(
                            row + TILE_COLS + g * SCALE_BYTES + b,
                            UInt8((bits >> UInt16(b * 8)) & 0xFF),
                        )
            for t in range(TILE_TOKENS):
                for r in range(TILE_ROWS):
                    want.data[t * TILE_ROWS + r] = planar_row_dot(
                        Q_Q8_0, p, r * stride, x.data, t * TILE_COLS, TILE_COLS
                    )

        var resident = Tensor(
            Int(pool.unsafe_ptr()),
            Q_Q8_0,
            TILE_COLS,
            TILE_ROWS,
            LAYOUT_PLANAR,
            WHERE_DEVICE,
        )
        var d_x = DeviceVec(ctx, TILE_TOKENS * TILE_COLS)
        var d_o = DeviceVec(ctx, TILE_TOKENS * TILE_ROWS)
        d_x.upload(x)
        device_matmul_into(ctx, resident, d_x, d_o, TILE_TOKENS)
        ctx.synchronize()
        keep(pool)

        var got = Buffer(TILE_TOKENS * TILE_ROWS)
        d_o.download(got)

        var peak = Float32(0)
        var worst = Float32(0)
        for i in range(TILE_TOKENS * TILE_ROWS):
            var m = want.data[i] if want.data[i] > 0 else -want.data[i]
            if m > peak:
                peak = m
            var gap = got.data[i] - want.data[i]
            if gap < 0:
                gap = -gap
            if gap > worst:
                worst = gap

        comptime gate = Float32(1e-5) if _float_tile() else Float32(2e-3)
        suite.check(peak > 0, "the reference is not all zeros")
        suite.check(
            worst <= peak * gate,
            "and the tile agrees with the host at this backend's precision",
        )
        if worst > peak * gate:
            suite.fail("tile", "worst " + String(worst / peak))


def _float_tile() -> Bool:
    """Whether this build's matrix core kernel multiplies in float.

    Apple's does and NVIDIA's cannot, which is one line here rather than a
    condition spelled out at the gate, because the day a third backend gets a
    tile the question it has to answer is this one.
    """
    return CompilationTarget.is_macos() or not has_nvidia_gpu_accelerator()


def test_pool(mut suite: Suite, ctx: DeviceContext) raises:
    """The loader's own pool, filled by the loader's own copy, read by a kernel.

    `test_matvec` above writes its weight through `map_to_host`, which is not
    how a load puts bytes on a device. A load allocates a `DevicePool` and calls
    `copy_in` per tensor, and until #152 that path never ran on a unified
    machine at all, because the plan there placed nothing on the device. It runs
    now, so this is the check that the two ends meet: the same slot arithmetic
    the plan does, the same asynchronous copy the load issues, and a kernel
    reading the answer back out.

    Two tensors and not one, at different slots, because a pool with a single
    tensor in it cannot tell a correct base address from a correct offset.
    """
    suite.group("device pool")

    comptime if not has_accelerator():
        return
    else:
        var cols = 128
        var rows = 3
        var stride = planar_row_bytes(Q_Q8_0, cols)
        var each = stride * rows
        var slot = _align(each, 256)

        var x = Buffer(cols)
        for i in range(cols):
            x.data[i] = (Float32((i * 5) % 23) - 11.0) / 5.0

        # Two weights in host memory, differing in every byte, so a copy that
        # takes the wrong source or lands at the wrong slot gives the other
        # tensor's answer rather than a plausible one.
        var host = List[UInt8]()
        for _ in range(slot + each):
            host.append(0)
        var hp = RawPtr(unsafe_from_address=Int(host.unsafe_ptr()))
        for t in range(2):
            var at = t * slot
            for r in range(rows):
                var row = at + r * stride
                for i in range(cols):
                    var q = ((i * 3 + r * 29 + t * 97) % 251) - 125
                    hp.unsafe_store(row + i, UInt8(q & 0xFF))
                for g in range(cols // 32):
                    var scale = (
                        Float32(0.25) * Float32(g + 1 + t) / Float32(r + 1)
                    )
                    var bits = bitcast[DType.uint16, 1](
                        scale.cast[DType.float16]()
                    )
                    for b in range(SCALE_BYTES):
                        hp.unsafe_store(
                            row + cols + g * SCALE_BYTES + b,
                            UInt8((bits >> UInt16(b * 8)) & 0xFF),
                        )

        var dev = default_device()
        var pool = DevicePool(dev, slot + each, ctx)
        for t in range(2):
            pool.copy_in(t * slot, Int(host.unsafe_ptr()) + t * slot, each)
        pool.wait()

        suite.check(
            pool.slot_address(slot) == pool.base() + slot,
            "a slot address is the pool base plus the plan's offset",
        )

        var worst = Float32(0)
        var peak = Float32(0)
        for t in range(2):
            var weight = Tensor(
                pool.slot_address(t * slot),
                Q_Q8_0,
                cols,
                rows,
                LAYOUT_PLANAR,
                WHERE_DEVICE,
            )
            var got = Buffer(rows)
            device_matvec(ctx, weight, x, got)
            for r in range(rows):
                var want = planar_row_dot(
                    Q_Q8_0, hp, t * slot + r * stride, x.data, 0, cols
                )
                var m = want if want > 0 else -want
                if m > peak:
                    peak = m
                var gap = got.data[r] - want
                if gap < 0:
                    gap = -gap
                if gap > worst:
                    worst = gap
        keep(pool)
        keep(host)

        suite.check(peak > 0, "the reference is not all zeros")
        suite.check(
            worst <= peak * Float32(1e-5),
            "and both tensors read back out of the pool as they went in",
        )


def _align(n: Int, to: Int) -> Int:
    return ((n + to - 1) // to) * to
