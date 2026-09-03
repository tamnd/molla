"""The matvec, on a GPU, from one source that compiles for Metal and for CUDA.

Decode is memory bound and almost every byte it touches is a weight, so this is
the kernel that decides whether a device is being used or merely owned. It is
also the kernel `max/kernels` does not have: its K quant matmul is CPU only, its
four bit GPU matmul is NVIDIA only and wants a weight layout of its own, and its
Apple quantized matmuls refuse to launch below an M5. The M4 is the reference
machine. So this one is ours, which is what D7 says happens when the dependency
does not cover a target.

It reads the planar layout from `molla.nn.repack` rather than ggml blocks. That
is the whole reason the planar layout exists. A ggml block is a header, a bit
plane and a nibble order, and unpacking one is a dozen dependent integer
operations before a single multiply; reading a row of it across the threads of a
block would have each thread pulling bytes from a different part of a header
that the thread beside it also needs. Planar is a value at a fixed width with
the scales in planes at the end of the row, so thread `t` reads the byte at
value `t`, the thread beside it reads the byte at value `t + 1`, and the
hardware coalesces the row into as few transactions as it has lanes. At four
bits a thread takes a whole byte and both of the values in it, so a warp still
reads one contiguous run and gets twice as many values out of it.

One block per output row, `TILE` threads in it, each walking the row with a
stride of `TILE`, then a tree reduction in shared memory. That is the arrangement
a matvec wants rather than the one a matmul wants: there is one input vector and
every thread in the block reads the same elements of it, so it stays in cache and
the only traffic that scales with the model is the weight row, read once, in
order, by consecutive lanes.

`TILE`, the group size, whether the type carries a minimum and how wide a quant
is are all compile time parameters, so there are five instantiations of one
function rather than five functions, and the divergence between targets is a
launch geometry rather than a second source file. That is what D7 asks for and
it is cheap here because the arithmetic is genuinely identical on both vendors.

There is no host fallback. A build with no device code in it raises when asked
for a device matvec rather than quietly running the host kernel and reporting a
number that says nothing about the device. D7 says a target that cannot pass
numerics is unsupported, and a silent fallback is how a target stays listed as
supported for a year after it stopped working.

The activations still come in and go out as host buffers, so every call pays for
an upload and a download that a real forward pass would not. That is deliberate
and temporary: keeping the whole residual stream on the device is #143, and
until there is something to keep it there for, a kernel that could only be
called with device buffers could not be checked against the host at all.
"""

from std.gpu import block_idx, thread_idx
from std.memory import AddressSpace, stack_allocation
from std.sys.info import has_accelerator

from max.gpu import barrier
from max.gpu.host import DeviceBuffer, DeviceContext

from molla.nn.repack import (
    LAYOUT_PLANAR,
    QUANT_I8,
    QUANT_S4,
    QUANT_U4,
    group_shift,
    group_size,
    has_min,
    quant_form,
)
from molla.nn.tensor import WHERE_DEVICE, Buffer, Tensor

comptime TILE = 128
"""Threads per output row.

A power of two because the reduction halves the live set every step, and 128
because it is at or under the maximum threadgroup width every target in the
fleet reports while still being wide enough that a 4096 wide row is 32 elements
of work per thread. It is a parameter of the kernel rather than a constant
inside it so that changing it is a launch site edit and not a rewrite, which is
the form D7 asks divergence between targets to take.
"""

comptime ACT_GROUP = 32
"""Activations sharing one scale, once the activation vector is quantized.

32 and not the weight's own group size, because a q4_K_M model has q4_K weights
grouped by 32 and q6_K weights grouped by 16 and both read the same activation
vector. The matvec takes two shifts rather than one, and pinning the activation
group to the smaller of the two would cost accuracy on the type that does not
need it.
"""

comptime ACT_RUN = 8
"""Values a thread multiplies before it converts and scales.

The whole point of quantizing the activation is that a run of products is
integer arithmetic that leaves the integer domain once. Eight measured best on
both backends: `scripts/matvec_probe.mojo` on a 4096 by 4096 q4_K matvec reads
498 microseconds at four, 415 at eight and 376 at sixteen on an M4, and 15, 12
and 14 on a 4090. Sixteen is nine per cent better on Metal and seventeen per
cent worse on CUDA, because a warp then reads 256 contiguous bytes an iteration
and `scripts/mem_probe.mojo` says the wide access is worth less than the narrow
one. Eight is the width where neither card gives anything up.

It has to divide every group size in the type table, which is 32 and 16, so
that a run sits inside one weight group and inside one activation group and
both scales are loaded once for it.
"""

comptime ACT_MAX = 32767
"""The largest magnitude an activation quant takes, so a signed short.

A byte was the obvious choice and it is what llama.cpp quantizes activations to,
and it is wrong here. The speed in this change comes from the converts and the
scale reads going away and not from the activation being small: an activation
vector is a few kilobytes that every block in the launch reads out of cache,
where the weights are gigabytes read once. `scripts/matvec_probe.mojo` prices
the second byte at nothing on Metal, where a 4096 by 4096 q4_K matvec is 415
microseconds with a byte and 356 with a short, and at one microsecond in twelve
on a 4090.

What the second byte buys is the whole accuracy argument. A byte over a group of
32 put the logit corpus at 0.05 on the sampled value tolerance and 0.3 to 0.58
on the log probability one, against a corpus that had been running at 3.1e-2 and
9.6e-2, and it failed twelve of thirteen cases. A short is 258 times finer and
lands back under the tolerances that were already there. Shrinking the group
does not substitute for it: at a group of 8 a byte still read 0.17 to 0.20 on
the head, which says the error is the step size and not the group.
"""


def act_plane_bytes(n: Int) -> Int:
    """Bytes an activation vector of `n` values takes once quantized.

    The planar weight layout with one plane fewer: `n` signed shorts, padded to
    a multiple of four so the scales behind them are four byte aligned, then a
    `Float32` scale for every `ACT_GROUP` of them.
    """
    var quants = (n * 2 + 3) & ~3
    return quants + ((n + ACT_GROUP - 1) // ACT_GROUP) * 4


def act_scale_offset(n: Int) -> Int:
    """Where the scales start, counted in floats rather than in bytes."""
    return ((n * 2 + 3) & ~3) // 4


struct DeviceQuantVec(Movable):
    """An activation vector in device memory, quantized to an integer a value.

    What #186 is. The matvec used to convert every weight to `Float32` and
    multiply it by a `Float32` activation, and on NVIDIA an integer to float
    convert issues at a quarter of the rate of a multiply, so the four converts
    a byte pair costs were worth sixteen multiplies before any multiplying
    happened. With the activation already an integer a weight and an activation
    multiply as integers and a run of eight leaves the integer domain once. That
    is 29 microseconds to 13 on a 4090 and 868 to 356 on an M4 for a 4096 by
    4096 q4_K row, which is the first arrangement of this kernel that has moved
    a 4090 at all.

    It is not exact, and it is the first thing on the device path that is not.
    `ACT_MAX` is where the size of the error is argued and
    [docs/validation/layout.md](../../../docs/validation/layout.md) is where it
    is measured. The rule that goes with it is that the greedy text does not
    change and the corpus tolerances do not move.

    Not `ImplicitlyCopyable`, for the reason `DeviceVec` is not.
    """

    var buf: DeviceBuffer[DType.uint8]
    var n: Int

    def __init__(out self, ctx: DeviceContext, n: Int) raises:
        if n <= 0:
            raise Error("a quantized device vector needs a positive length")
        if n % ACT_GROUP != 0:
            raise Error(
                "a quantized device vector is grouped by "
                + String(ACT_GROUP)
                + " and "
                + String(n)
                + " is not a multiple of it"
            )
        self.buf = ctx.enqueue_create_buffer[DType.uint8](act_plane_bytes(n))
        self.n = n

    def elements(self) -> Int:
        return self.n

    def ptr(self) -> Pointer[UInt8, MutAnyOrigin]:
        return Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(self.buf.unsafe_ptr())
        )


struct DeviceVec(Movable):
    """Float32 activations in device memory, owned.

    The counterpart of `molla.nn.tensor.Buffer` and the thing that makes a
    block run without leaving the device. Every operation in `molla.nn.gpu_ops`
    takes these and returns nothing, so a norm feeding a matvec feeding a
    residual add is three launches on one stream and no transfer between them.
    A round trip to the host between two kernels costs more than either of them
    at decode shapes, which is the whole argument for putting the small
    operations on the device at all.

    `copy_in` and `copy_out` are how bytes cross the boundary. `upload`,
    `upload_run`, `download` and `at` do the same job through a host mapping and
    are for tests and for a trace, because the first `map_to_host` call in a
    process reserves 1.3 GiB that it never gives back and never reuses. On a 138
    MiB model that was most of the resident set. The copy path never pays it.
    See [docs/validation/performance.md](../../../docs/validation/performance.md).

    Either way, a transfer is for the ends of a block. It is not on the path a
    token takes, and every use of one inside a sequence of kernels is a bug
    rather than a slow spot.

    Not `ImplicitlyCopyable`, deliberately. Two vectors that name the same
    device buffer is exactly the aliasing bug that produces a norm reading its
    own half written output, and it is invisible in the numbers.
    """

    var buf: DeviceBuffer[DType.float32]
    var n: Int

    def __init__(out self, ctx: DeviceContext, n: Int) raises:
        if n <= 0:
            raise Error("a device vector needs a positive length")
        self.buf = ctx.enqueue_create_buffer[DType.float32](n)
        self.n = n

    def elements(self) -> Int:
        return self.n

    def ptr(self) -> Pointer[Float32, MutAnyOrigin]:
        """The device address, for a kernel argument.

        A host load through this faults. `map_to_host` is the other end and it
        is a different address for the same bytes, which is the finding in
        `docs/validation/load.md`.
        """
        return Pointer[Float32, MutAnyOrigin](
            unsafe_from_address=Int(self.buf.unsafe_ptr())
        )

    def ptr_at(self, at: Int) raises -> Pointer[Float32, MutAnyOrigin]:
        """The device address of element `at`, for a kernel argument.

        Every kernel here writes from element zero of whatever pointer it was
        handed, so writing into the middle of something is done by moving the
        pointer rather than by giving each kernel an offset it would have to
        apply and a bound it would have to be told. A key projection lands at
        `slot * kv_width` in a layer's cache, a bias covers one run of a query,
        and both are this call and the kernel unchanged.

        Four bytes an element, which is what a float32 device vector is by
        construction, so the arithmetic is the same on both vendors.
        """
        if at < 0 or at > self.n:
            raise Error(
                "offset "
                + String(at)
                + " is outside a device vector of "
                + String(self.n)
            )
        return Pointer[Float32, MutAnyOrigin](
            unsafe_from_address=Int(self.buf.unsafe_ptr()) + at * 4
        )

    def copy_in(mut self, x: List[Float32]) raises:
        """Fill this vector from the front of a host list, without a mapping.

        Synchronous on purpose. The card reads `x` directly and `x` is usually a
        local that dies when the caller returns, so handing the copy to the
        stream and going home would leave the card reading a freed list.
        Everything that calls this calls it once per weight when a model binds,
        so the wait costs nothing anybody can measure.
        """
        if len(x) < self.n:
            raise Error(
                "copying "
                + String(len(x))
                + " values into a device vector of "
                + String(self.n)
            )
        var ctx = self.buf.context()
        ctx.enqueue_copy(
            self.buf,
            Pointer[Float32, MutAnyOrigin](
                unsafe_from_address=Int(x.unsafe_ptr())
            ),
        )
        ctx.synchronize()

    def copy_out(self, mut out: Buffer) raises:
        """This vector back into a host buffer, without a mapping.

        Synchronous for the ordinary reason rather than the lifetime one:
        whatever queued the kernels that wrote this vector has not necessarily
        finished, and reading before the stream drains reads whichever of them
        happened to land.
        """
        if out.elements() != self.n:
            raise Error(
                "copying a device vector of "
                + String(self.n)
                + " into "
                + String(out.elements())
                + " values"
            )
        var ctx = self.buf.context()
        ctx.enqueue_copy(
            Pointer[Float32, MutAnyOrigin](
                unsafe_from_address=Int(out.data.unsafe_ptr())
            ),
            self.buf,
        )
        ctx.synchronize()

    def upload(mut self, x: Buffer) raises:
        if x.elements() != self.n:
            raise Error(
                "uploading "
                + String(x.elements())
                + " values into a device vector of "
                + String(self.n)
            )
        with self.buf.map_to_host() as h:
            for i in range(self.n):
                h[i] = x.data[i]

    def upload_run(mut self, x: List[Float32], at: Int, n: Int) raises:
        """Part of a host list into the front of this vector.

        A key cache is one long list per layer with a token's heads laid end to
        end, so the thing an attention test wants to move is a run out of the
        middle of one rather than a whole `Buffer`.
        """
        if at < 0 or n <= 0 or len(x) < at + n or n > self.n:
            raise Error("a run to upload has to fit in both ends")
        with self.buf.map_to_host() as h:
            for i in range(n):
                h[i] = x[at + i]

    def download(self, mut out: Buffer) raises:
        if out.elements() != self.n:
            raise Error(
                "downloading a device vector of "
                + String(self.n)
                + " into "
                + String(out.elements())
                + " values"
            )
        with self.buf.map_to_host() as h:
            for i in range(self.n):
                out.data[i] = h[i]

    def at(self, index: Int) raises -> Float32:
        """One value, for a test that wants a scalar out of a reduction.

        A map and an unmap for one float, which is why nothing on a hot path
        calls it.
        """
        if index < 0 or index >= self.n:
            raise Error("index " + String(index) + " is outside this vector")
        var got: Float32
        with self.buf.map_to_host() as h:
            got = h[index]
        return got


def planar_matvec_kernel[
    tile: Int, group: Int, with_min: Bool, form: Int
](
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    cols_dev: Int32,
    stride_dev: Int32,
):
    """`o[r] = dot(planar row r of w, x)`, one thread block per row.

    The scales are read through a float32 view of the same bytes rather than
    assembled a byte at a time. A planar row is a multiple of four bytes long by
    construction and so is its quant plane at either width, so the scale planes
    are four byte aligned in every row of every tensor, and a device buffer
    starts at an alignment far larger than that.

    Shapes arrive as `Int32` because a kernel argument has to be device
    passable and `Int` is not. Nothing here is near two billion.
    """
    var cols = Int(cols_dev)
    var stride = Int(stride_dev)
    var r = Int(block_idx.x)
    var t = Int(thread_idx.x)

    var row = r * stride
    var groups = cols // group
    # Three views of the same bytes rather than one view and a conversion. The
    # byte wide quants are signed and are read through a signed pointer for
    # that reason: converting a `UInt8` to an `Int8` after loading it is a value
    # conversion and not a reinterpretation, and it turns every quant of 128 and
    # up into something that is not the negative number the repack wrote. Four
    # of the eight types centre their quants, so half the corpus is wrong and
    # the other half is exact, which is a failure that looks like a type table
    # bug rather than like a cast. The nibble wide ones are read unsigned and
    # the sign extension is done in the open below, because there is no four
    # bit type to reinterpret through.
    var quants = w.unsafe_bitcast[Int8]()
    var packed = w
    var scales = w.unsafe_bitcast[Float32]()
    var quant_bytes = cols if form == QUANT_I8 else cols // 2
    var d_base = (row + quant_bytes) // 4
    var m_base = d_base + groups

    # A shift and not `i // group`, which is a signed divide and which Metal
    # keeps. See `group_shift` for what it costs there.
    comptime shift = group_shift(group)

    var acc = Float32(0)
    comptime if form == QUANT_I8:
        var i = t
        while i < cols:
            var gi = i >> shift
            var q = Float32(Int(quants[unsafe_offset=row + i]))
            var a = x[unsafe_offset=i]
            acc += scales[unsafe_offset=d_base + gi] * q * a
            comptime if with_min:
                acc += scales[unsafe_offset=m_base + gi] * a
            i += tile
    else:
        # A byte to a thread and both of its values, so a warp reads thirty two
        # contiguous bytes and gets sixty four values out of them. Both values
        # of a byte sit in one group, because every group size here is even, so
        # their scale and minimum are loaded once for the pair. `(n ^ 8) - 8` is
        # the four bit two's complement sign extension, done with arithmetic
        # because a branch on which nibble is which is one that every warp would
        # take both sides of.
        #
        # Three other arrangements of this loop were written and measured on an
        # RTX 4090 against an 8B: a value to a thread with two threads sharing a
        # byte, four bytes to a thread through a `UInt32`, and that one again
        # with the activations read four at a time into a SIMD accumulator. All
        # four are within one per cent of each other, and all four leave the
        # kernel reading 543 GB/s where `scripts/mem_probe.mojo` says this shape
        # of access is worth 945. What holds it there on that card is the number
        # of instructions a value costs rather than anything about how the bytes
        # are arranged, which is #186, so the simplest of the four is the one
        # that is here. That was read as a statement about both backends for a
        # day and it is not one: on Metal the same loop was spending more than
        # half its time on the group index, which is why that is a shift now.
        # See [docs/validation/layout.md](../../../docs/validation/layout.md).
        var i = t * 2
        while i < cols:
            var gi = i >> shift
            var b = Int(packed[unsafe_offset=row + (i >> 1)])
            var lo = b & 0xF
            var hi = (b >> 4) & 0xF
            comptime if form == QUANT_S4:
                lo = (lo ^ 8) - 8
                hi = (hi ^ 8) - 8
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

    # The reduction is over the block rather than over a warp because `tile` is
    # a parameter and the warp width is not the same number on the two vendors.
    # A shuffle based version would be faster and would put a target specific
    # constant in the one place D7 says not to put one.
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


def act_quant_kernel[
    group: Int
](
    x: Pointer[Float32, MutAnyOrigin],
    q: Pointer[UInt8, MutAnyOrigin],
    scale_base_dev: Int32,
):
    """One group of activations to a signed byte each and one scale.

    A block to a group, `group` threads in it, so the absolute maximum is a
    reduction the width of the group and every thread already holds the value it
    is about to write. The vector is a few thousand values, so this is a hundred
    odd blocks of 32 and the launch costs more than the work does. That is the
    right shape anyway: the alternative is a grid stride loop with an atomic or
    a second pass, and both are more machinery for a kernel that is not on
    anybody's critical path.

    Symmetric, `absmax / ACT_MAX`, and not affine. The weight side already
    carries a minimum where the type has one, and a second offset here would put
    a cross term in every dot product. A group that is all zeros gets a scale of
    zero and quants of zero rather than a division by nothing.
    """
    var g = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var i = g * group + t

    var v = x[unsafe_offset=i]
    var a = v if v >= 0 else -v

    var part = stack_allocation[
        group, Float32, address_space=AddressSpace.SHARED
    ]()
    part[unsafe_offset=t] = a
    barrier()
    var step = group // 2
    while step > 0:
        if t < step:
            var l = part[unsafe_offset=t]
            var r = part[unsafe_offset=t + step]
            part[unsafe_offset=t] = l if l >= r else r
        barrier()
        step //= 2
    var amax = part[unsafe_offset=0]

    if t == 0:
        var scales = q.unsafe_bitcast[Float32]()
        scales[unsafe_offset=Int(scale_base_dev) + g] = amax / Float32(ACT_MAX)

    # Half away from zero, written out rather than called, because the rounding
    # a `round` maps to is a target decision and this one has to be the same
    # number on both.
    var scaled = (v * Float32(ACT_MAX) / amax) if amax > 0 else Float32(0)
    var rounded = scaled + (Float32(0.5) if scaled >= 0 else Float32(-0.5))
    q.unsafe_bitcast[Int16]()[unsafe_offset=i] = Int16(Int32(rounded))


def planar_matvec_q8_kernel[
    tile: Int, group: Int, with_min: Bool, form: Int
](
    w: Pointer[UInt8, MutAnyOrigin],
    xq: Pointer[UInt8, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    cols_dev: Int32,
    stride_dev: Int32,
    scale_base_dev: Int32,
):
    """The same dot product, with both sides integers until the run ends.

    Line for line the kernel above as far as the weight planes go. What differs
    is that the activation arrives as a signed byte with a scale a group, so a
    run of `ACT_RUN` values is `ACT_RUN` integer multiplies into an `Int32` and
    then one convert and one scale, where the float version is a convert and two
    floating point operations per value.

    The minimum is the second accumulator and it is where the algebra is. Over a
    group the value is `d * w + m` and the sum is `sum((d * w + m) * x)`, which
    with `x = dx * q` is `d * dx * sum(w * q) + m * dx * sum(q)`. So the run
    carries the dot and the plain sum of the activation quants and converts
    twice at the end rather than multiplying and adding per value. A type with
    no minimum drops the second accumulator at compile time.

    Neither accumulator can overflow. The widest case is a run of eight signed
    byte weights at 127 against activation quants at 32767, which is 33 million,
    and it would still be inside an `Int32` at a run of a whole group.
    """
    var cols = Int(cols_dev)
    var stride = Int(stride_dev)
    var r = Int(block_idx.x)
    var t = Int(thread_idx.x)

    var row = r * stride
    var groups = cols // group
    var quants = w.unsafe_bitcast[Int8]()
    var packed = w
    var scales = w.unsafe_bitcast[Float32]()
    var quant_bytes = cols if form == QUANT_I8 else cols // 2
    var d_base = (row + quant_bytes) // 4
    var m_base = d_base + groups

    var aq = xq.unsafe_bitcast[Int16]()
    var ad = xq.unsafe_bitcast[Float32]()
    var a_base = Int(scale_base_dev)

    comptime shift = group_shift(group)
    comptime act_shift = group_shift(ACT_GROUP)

    var acc = Float32(0)
    var i = t * ACT_RUN
    while i < cols:
        var gi = i >> shift
        var ai = i >> act_shift
        var dot = Int32(0)
        var qsum = Int32(0)

        comptime if form == QUANT_I8:
            var k = 0
            while k < ACT_RUN:
                var q = Int32(quants[unsafe_offset=row + i + k])
                var a = Int32(aq[unsafe_offset=i + k])
                dot += q * a
                comptime if with_min:
                    qsum += a
                k += 1
        else:
            var at = row + (i >> 1)
            var k = 0
            while k < ACT_RUN // 2:
                var b = Int32(packed[unsafe_offset=at + k])
                var lo = b & 0xF
                var hi = (b >> 4) & 0xF
                comptime if form == QUANT_S4:
                    lo = (lo ^ 8) - 8
                    hi = (hi ^ 8) - 8
                var a0 = Int32(aq[unsafe_offset=i + k * 2])
                var a1 = Int32(aq[unsafe_offset=i + k * 2 + 1])
                dot += lo * a0 + hi * a1
                comptime if with_min:
                    qsum += a0 + a1
                k += 1

        var dx = ad[unsafe_offset=a_base + ai]
        acc += scales[unsafe_offset=d_base + gi] * dx * Float32(dot)
        comptime if with_min:
            acc += scales[unsafe_offset=m_base + gi] * dx * Float32(qsum)
        i += tile * ACT_RUN

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


def device_ready() -> Bool:
    """Whether this build has device code in it at all.

    A property of the machine that compiled the binary and not of the one
    running it, for the reason `molla.sys.device.build_targets_gpu` gives.
    """
    return has_accelerator()


def check_matvec(w: Tensor, x: Buffer, out_elements: Int) raises:
    """Everything a device matvec refuses, asked without a device.

    Separate from the launch so that the refusals are testable on the three
    machines in the fleet that have no accelerator. All four of these are
    conditions a caller can hit by wiring the engine up wrong, and the one that
    matters most is the residency check: a weight still sitting in a mapping has
    a host address, and a Metal kernel handed a host address does not fault, it
    reads zeros. Refusing here is the difference between a shape error and a
    model that runs at full speed and answers with noise.
    """
    check_matvec_shapes(w, x.elements(), out_elements)


def check_matvec_shapes(w: Tensor, in_elements: Int, out_elements: Int) raises:
    """The same checks, counting elements rather than holding buffers.

    Which is what the device to device path needs, since a `DeviceVec` is not
    copyable and a check that took one would have to borrow it through a
    signature that says nothing about why.
    """
    if w.layout != LAYOUT_PLANAR:
        raise Error(
            "the device matvec reads the planar layout and this weight is"
            " still in ggml blocks, so it has to be repacked first"
        )
    if group_size(w.kind) == 0:
        raise Error(
            "ggml type " + String(w.kind) + " has no planar form to read"
        )
    if w.place != WHERE_DEVICE:
        raise Error(
            "the device matvec wants a weight in a device pool and this one is"
            " in host memory, which a device kernel reads as zeros rather than"
            " as an error"
        )
    if in_elements != w.cols:
        raise Error(
            "the device matvec wants an input of "
            + String(w.cols)
            + " but got "
            + String(in_elements)
        )
    if out_elements != w.rows:
        raise Error(
            "the device matvec wants an output of "
            + String(w.rows)
            + " but got "
            + String(out_elements)
        )


def _launch[
    tile: Int, group: Int, with_min: Bool, form: Int
](
    ctx: DeviceContext,
    w: Tensor,
    x: DeviceVec,
    o: Pointer[Float32, MutAnyOrigin],
) raises:
    """One instantiation, launched. No transfer either side of it."""
    ctx.enqueue_function[planar_matvec_kernel[tile, group, with_min, form]](
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=w.device_address()),
        x.ptr(),
        o,
        Int32(w.cols),
        Int32(w.row_bytes()),
        grid_dim=(w.rows, 1, 1),
        block_dim=(tile, 1, 1),
    )


def device_matvec_into(
    ctx: DeviceContext, w: Tensor, x: DeviceVec, mut out: DeviceVec, at: Int = 0
) raises:
    """`out[at + r] = dot(row r of w, x)`, both sides already on the device.

    The dispatch is over the three things the layout varies by, which are how
    wide a quant is, the group size, and whether there is a minimum plane.
    Twelve combinations exist on paper and five exist in the type table:
    everything with a minimum groups by 32, the only type that groups by 16 is
    q6_k, which centres its quants and has no minimum at all, the two unsigned
    nibble types both group by 32 and both carry a minimum, and the one signed
    nibble type is q4_0, which centres and does not.

    The check is written out in full rather than inferred from the form,
    because a type added later that breaks one of those coincidences should
    fail to launch and say so, not quietly read the wrong plane.

    `at` is where the rows go and it defaults to the front, which is what every
    caller wanted until there was a cache. A key and a value projection write
    into the layer's cache at the slot this position occupies, and doing that
    here rather than into scratch and then copying is the same choice the host
    made for the same reason: the cache is where they are needed and this is the
    only place they are written.

    Queued and not synchronized. A block is a couple of dozen of these and
    waiting after each one would put the cost this exists to remove back in a
    different place, so the caller synchronizes once when it wants the answer.
    """
    if at < 0:
        raise Error("a matvec cannot write at a negative offset")
    if out.elements() < at + w.rows:
        raise Error(
            "the device matvec writes "
            + String(w.rows)
            + " rows at offset "
            + String(at)
            + " and the output ends at "
            + String(out.elements())
        )
    check_matvec_shapes(w, x.elements(), w.rows)
    comptime if not has_accelerator():
        raise Error(
            "this build has no device code in it, so there is no device matvec"
            " to run. Accelerator support is decided when molla is compiled,"
            " not when it is run"
        )
    else:
        var o = out.ptr_at(at)
        var g = group_size(w.kind)
        var carries_min = has_min(w.kind)
        var form = quant_form(w.kind)
        if form == QUANT_U4 and g == 32 and carries_min:
            _launch[TILE, 32, True, QUANT_U4](ctx, w, x, o)
        elif form == QUANT_S4 and g == 32 and not carries_min:
            _launch[TILE, 32, False, QUANT_S4](ctx, w, x, o)
        elif form == QUANT_I8 and g == 32 and carries_min:
            _launch[TILE, 32, True, QUANT_I8](ctx, w, x, o)
        elif form == QUANT_I8 and g == 32:
            _launch[TILE, 32, False, QUANT_I8](ctx, w, x, o)
        elif form == QUANT_I8 and g == 16 and not carries_min:
            _launch[TILE, 16, False, QUANT_I8](ctx, w, x, o)
        else:
            raise Error(
                "no device matvec is compiled for quant form "
                + String(form)
                + " with a group of "
                + String(g)
                + (" and a minimum plane" if carries_min else " and no minimum")
            )


def device_quantize(
    ctx: DeviceContext, x: DeviceVec, mut q: DeviceQuantVec
) raises:
    """`q` becomes `x` at a byte a value, on the device, queued.

    Called once per activation vector rather than once per matvec, because the
    query, key and value projections read one vector and up and gate read
    another. That is four of these a layer and one for the head, against about
    fourteen kernels a layer, so it is a 28 per cent rise in the launch count
    and on a model that is launch bound rather than kernel bound it may not pay
    for itself. Three of the four go away when the norm and the activation
    function write the quantized form as they go, which is #168.
    """
    if x.elements() != q.elements():
        raise Error(
            "quantizing "
            + String(x.elements())
            + " activations into room for "
            + String(q.elements())
        )
    comptime if not has_accelerator():
        raise Error(
            "this build has no device code in it, so there is nothing to"
            " quantize an activation vector on"
        )
    else:
        ctx.enqueue_function[act_quant_kernel[ACT_GROUP]](
            x.ptr(),
            q.ptr(),
            Int32(act_scale_offset(x.elements())),
            grid_dim=(x.elements() // ACT_GROUP, 1, 1),
            block_dim=(ACT_GROUP, 1, 1),
        )


def _launch_q8[
    tile: Int, group: Int, with_min: Bool, form: Int
](
    ctx: DeviceContext,
    w: Tensor,
    x: DeviceQuantVec,
    o: Pointer[Float32, MutAnyOrigin],
) raises:
    ctx.enqueue_function[planar_matvec_q8_kernel[tile, group, with_min, form]](
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=w.device_address()),
        x.ptr(),
        o,
        Int32(w.cols),
        Int32(w.row_bytes()),
        Int32(act_scale_offset(x.elements())),
        grid_dim=(w.rows, 1, 1),
        block_dim=(tile, 1, 1),
    )


def device_matvec_q8_into(
    ctx: DeviceContext,
    w: Tensor,
    x: DeviceQuantVec,
    mut out: DeviceVec,
    at: Int = 0,
) raises:
    """`device_matvec_into` with the activation already quantized.

    The same dispatch over the same five combinations, because the weight
    planes are read the same way and only the activation side changed. Both
    exist on purpose. The float one is the reference the corpus checks the
    layout against with no tolerance at all, and a kernel that is exact is worth
    keeping even once nothing on a token's path calls it.
    """
    if at < 0:
        raise Error("a matvec cannot write at a negative offset")
    if out.elements() < at + w.rows:
        raise Error(
            "the device matvec writes "
            + String(w.rows)
            + " rows at offset "
            + String(at)
            + " and the output ends at "
            + String(out.elements())
        )
    check_matvec_shapes(w, x.elements(), w.rows)
    comptime if not has_accelerator():
        raise Error(
            "this build has no device code in it, so there is no device matvec"
            " to run. Accelerator support is decided when molla is compiled,"
            " not when it is run"
        )
    else:
        var o = out.ptr_at(at)
        var g = group_size(w.kind)
        var carries_min = has_min(w.kind)
        var form = quant_form(w.kind)
        if form == QUANT_U4 and g == 32 and carries_min:
            _launch_q8[TILE, 32, True, QUANT_U4](ctx, w, x, o)
        elif form == QUANT_S4 and g == 32 and not carries_min:
            _launch_q8[TILE, 32, False, QUANT_S4](ctx, w, x, o)
        elif form == QUANT_I8 and g == 32 and carries_min:
            _launch_q8[TILE, 32, True, QUANT_I8](ctx, w, x, o)
        elif form == QUANT_I8 and g == 32:
            _launch_q8[TILE, 32, False, QUANT_I8](ctx, w, x, o)
        elif form == QUANT_I8 and g == 16 and not carries_min:
            _launch_q8[TILE, 16, False, QUANT_I8](ctx, w, x, o)
        else:
            raise Error(
                "no device matvec is compiled for quant form "
                + String(form)
                + " with a group of "
                + String(g)
                + (" and a minimum plane" if carries_min else " and no minimum")
            )


def device_matvec(
    ctx: DeviceContext, w: Tensor, x: Buffer, mut out: Buffer
) raises:
    """`out[r] = dot(row r of w, x)` with host activations either side.

    What the oracle and the tests call, and what nothing on a token's path
    should. Every call allocates two device buffers, uploads, launches,
    synchronizes and downloads, and at decode shapes the transfers cost more
    than the multiply does. It exists because a kernel that could only be
    called with device buffers could not be checked against the host at all
    until there was a block to run.
    """
    check_matvec(w, x, out.elements())
    comptime if not has_accelerator():
        raise Error(
            "this build has no device code in it, so there is no device matvec"
            " to run. Accelerator support is decided when molla is compiled,"
            " not when it is run"
        )
    else:
        var d_x = DeviceVec(ctx, w.cols)
        var d_o = DeviceVec(ctx, w.rows)
        d_x.upload(x)
        device_matvec_into(ctx, w, d_x, d_o)
        ctx.synchronize()
        d_o.download(out)
