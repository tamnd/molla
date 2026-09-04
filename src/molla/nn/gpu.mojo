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

from std.atomic import Atomic, Ordering
from std.gpu import block_idx, thread_idx
from std.gpu.primitives.warp import WARP_SIZE, lane_group_sum
from std.math import exp
from std.memory import AddressSpace, bitcast, stack_allocation
from std.sys.info import CompilationTarget, has_accelerator

from max.gpu import barrier
from max.gpu.compute.arch.mma_apple import (
    _mma_apple_8x8,
    apple_mma_load_8x8,
    apple_mma_store_8x8,
)
from max.gpu.host import DeviceBuffer, DeviceContext

from molla.nn.block import ACT_GELU, ACT_SILU
from molla.nn.repack import (
    LAYOUT_PLANAR,
    QUANT_I8,
    QUANT_S4,
    QUANT_S5,
    QUANT_S6,
    QUANT_U4,
    QUANT_U5,
    SCALE_BYTES,
    group_shift,
    group_size,
    has_min,
    quant_bias,
    quant_form,
    quant_high_bits,
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

The matvec launches at `MATVEC_TILE` and not at this, which is a narrower block
on one backend. Everything else here still launches at 128.
"""

comptime MATVEC_TILE = 32 if CompilationTarget.is_macos() else TILE
"""Threads per output row for the matvec, which is not `TILE` on Metal.

128 was never measured, it was the width every other kernel here uses. Sweeping
it in `scripts/matvec_probe.mojo` on an M4 says it is the wrong number for this
kernel at every shape a decode runs, and by a lot:

```text
q4_k attn  4096 by 4096      t32 216 us   t64 272 us   t128 444 us
q4_k down  4096 by 14336     t32 571 us   t64 622 us   t128 886 us
q8_0 head  49152 by 576      t32 1209 us  t64 2091 us  t128 3857 us
```

The floor for those three, a kernel that reads the same bytes and does no
arithmetic, is 206, 669 and 322 microseconds. So at 32 the two q4_K shapes are
at the floor and at 128 they are twice off it, and the head, which is the shape
a small model spends a fifth of its decode in, is three and a half times faster.

Of the two things that could cost at 128, only one does. A 576 column row at
eight values a thread is 72 threads of work, so 56 of the 128 arrive with
nothing in them, and neither that nor anything else here shows up on a 4096
column row, which is why this was invisible on the 8B. It is not the reduction:
a shuffle reduction under the narrower block was written and measured and is a
wash on both small models, so the width was never paying for the tree, it was
paying for the empty threads. See
[docs/validation/layout.md](../../../docs/validation/layout.md).

CUDA keeps 128, and that is now a measurement rather than a deferral. The same
sweep on a 4090 at load 0.06 finds the same pattern: the head is 1.85 times
faster at 32, the wide control prefers 128 by a quarter, and q4_K does not care.
End to end it is worth about one per cent, because a token there is 2.1 ms and
the head is 48 us of it. So 32 is very slightly better on both vendors and the
honest version of this constant keys off the row width rather than the target,
which is not worth writing without end to end evidence to write it from. The
reason there is none is that a SmolLM2 token on that card is 2.16 ms and the
matvec is not what it is spending it on: the model decodes through the fused
kernel, whose grid and barriers are about 1.9 ms of that token against 0.22 ms
of weights and 0.15 ms of submission. See #170.
"""


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


comptime EPI_NONE = 0
"""The matvec writes its row and nothing else, which is what it always did."""
comptime EPI_BIAS = 1
"""`o[r] = dot + aux[r]`, the projection bias folded into the row it belongs to.
"""
comptime EPI_ADD = 2
"""`o[r] = o[r] + dot`, the residual add folded into the projection.

The residual add is a whole kernel launch to do one flop per element on a vector
that the matvec that produced it had in a register a moment earlier. Folding it
in removes the launch, removes the round trip through device memory, and removes
the scratch vector it was going through. It is the same addition in the same
order, so it is exact.
"""
comptime EPI_GLU = 4
"""`o[r] = act(dot) * aux[r]`, the gate half of a gated MLP.

Only valid on the matvec that produces the gate, and only when the up projection
has already been written, which is the order `device_mlp` was already in.
"""

comptime ACT_BIT = 8
"""Set alongside `EPI_GLU` when the activation is gelu rather than silu."""


@always_inline
def activate[kind: Int](v: Float32) -> Float32:
    """The gate function, chosen at compile time.

    A parameter rather than a branch because it is the same value for every
    element of every layer of a model, so a runtime test would be a predictable
    branch executed fourteen thousand times per layer to reach the same side.

    It lives here rather than beside the elementwise kernels because the matvec
    epilogue applies it too and `gpu_ops` imports this file, not the other way
    round.
    """
    comptime if kind == ACT_SILU:
        return v / (Float32(1.0) + exp(-v))
    else:
        # The tanh approximation, which is the one the weights were trained
        # with. The exact form through the error function is a different
        # function by about a thousandth, and a model trained against one and
        # run against the other is a small consistent bias nobody ever finds.
        var c = Float32(0.7978845608028654)
        var inner = c * (v + Float32(0.044715) * v * v * v)
        var e = Float32(2.0) / (Float32(1.0) + exp(Float32(-2.0) * inner))
        return Float32(0.5) * v * (Float32(1.0) + (e - Float32(1.0)))


comptime MAGIC = UInt32(0x4B000000)
"""The bit pattern of `2^23` as a float32, which is where a small integer goes.

A float32 with an exponent of 23 has a mantissa step of exactly one, so the
representable numbers from `2^23` to `2^24` are the integers, and their bit
patterns are `MAGIC + n`. Nothing carries out of the mantissa for `n` under
`2^23`, so `MAGIC | n` and `MAGIC + n` are the same bits and the OR is what a
quant costs. Subtracting `2^23` back is exact because the result is small.

So an integer becomes a float in an OR and a subtract, both of which issue at
the full rate on both vendors, instead of in a convert, which on NVIDIA issues
on the transcendental unit at a quarter rate. Four converts a byte pair cost
more than sixteen multiplies before any multiplying happens, and that is what
`scripts/matvec_probe.mojo` was measuring when it found three quarters of this
kernel going into arithmetic on values that had already arrived.

The trick is only valid for a value that fits under the mantissa step and there
is no rounding anywhere in it, so it is exact rather than close. See
[docs/validation/layout.md](../../../docs/validation/layout.md).
"""


@always_inline
def nibble_float[form: Int](n: UInt32) -> Float32:
    """One four bit quant, as a float, without a convert.

    For a centred type the sign extension comes free: `n ^ 8` is `n + 8` below
    eight and `n - 8` above it, so biasing the subtraction by eight turns the
    same OR into `(n ^ 8) - 8`, which is the four bit two's complement value the
    repack wrote. An unsigned type subtracts the plain magic.
    """
    comptime if form == QUANT_S4:
        return bitcast[DType.float32, 1](MAGIC | (n ^ 8)) - Float32(8388616.0)
    else:
        return bitcast[DType.float32, 1](MAGIC | n) - Float32(8388608.0)


@always_inline
def byte_float(u: UInt32) -> Float32:
    """One signed byte quant, as a float, without a convert.

    Same shape as `nibble_float`, one bit wider. The quant plane of a byte wide
    type is signed two's complement, `u ^ 0x80` moves it to the unsigned range
    the mantissa can hold, and the bias takes it back.
    """
    return bitcast[DType.float32, 1](MAGIC | (u ^ 0x80)) - Float32(8388736.0)


@always_inline
def wide_float[form: Int](u: UInt32) -> Float32:
    """One five or six bit quant, already joined from its two planes, as a float.

    The same trick again with a different constant. A five or six bit value is
    stored unsigned, 0 to 31 or 0 to 63, and the offset that makes it signed
    comes off in the subtraction the magic number needed anyway, so a q5_0 value
    costs exactly what a q4_1 value costs once the bits are in hand and the only
    extra work the wider types do is joining the planes.
    """
    comptime bias = Float32(8388608.0) + Float32(quant_bias(form))
    return bitcast[DType.float32, 1](MAGIC | u) - bias


@always_inline
def _scale(s: Pointer[Float16, MutAnyOrigin], at: Int) -> Float32:
    """One group scale, widened on load.

    The plane is float16 and everything downstream of this is float32, and every
    GPU in the fleet widens in the load instruction rather than in an
    instruction after it, so the convert this looks like is not one. It is a
    function rather than a `.cast` at each of the sixteen sites because the
    width of a scale is one decision and it should read as one.
    """
    return s[unsafe_offset=at].cast[DType.float32]()


@always_inline
def high_shift[form: Int]() -> Int:
    """`log2` of how many values one byte of the second plane covers.

    A shift and not the count, because every use of it is a divide or a modulo
    by that count and both of those are signed operations on an `Int` that the
    Metal compiler keeps unless the constant is a power of two it can see. It is
    three at five bits and two at six.
    """
    return 3 if quant_high_bits(form) == 1 else 2


@always_inline
def planar_quant_stride[form: Int](cols: Int) -> Int:
    """Bytes both quant planes of one row take, which is where the scales start.

    Eight bits is a byte a value, four is half of that, five adds an eighth on
    top of the half and six adds a quarter. Written as eighths of a byte so
    there is one expression rather than a table, and exact for every row width
    molla accepts, since a row is a whole number of 32 or 256 value blocks.
    """
    comptime eighths = 8 if form == QUANT_I8 else 4 + quant_high_bits(form)
    return (cols * eighths) // 8


@always_inline
def write_epilogue(
    o: Pointer[Float32, MutAnyOrigin],
    aux: Pointer[Float32, MutAnyOrigin],
    epi: Int,
    out_at: Int,
    r: Int,
    v: Float32,
):
    """The tail of every matmul kernel: bias or gate, then store or accumulate.

    One copy rather than one per kernel. There are four kernels sharing this now
    and the epilogue is the only part of them that has to agree exactly, since
    a prefill and a decode of the same prompt are checked against each other to
    a relative 2e-4 and a difference here would be a difference in the answer
    rather than in the speed.

    `out_at` is the index into the output and `r` is the row, which for a bias
    is the index into `aux` and for a gate is not, because a gate is a whole
    other output tensor of the same shape.
    """
    var got = v
    if epi & EPI_BIAS != 0:
        got += aux[unsafe_offset=r]
    elif epi & EPI_GLU != 0:
        var g = aux[unsafe_offset=out_at]
        if epi & ACT_BIT != 0:
            got = activate[ACT_GELU](got) * g
        else:
            got = activate[ACT_SILU](got) * g
    if epi & EPI_ADD != 0:
        o[unsafe_offset=out_at] = o[unsafe_offset=out_at] + got
    else:
        o[unsafe_offset=out_at] = got


comptime _dev32 = Atomic[DType.int32, scope="device"]
"""A device scope atomic view of a float, for `coherent_load`."""


@always_inline
def coherent_load[
    coherent: Bool
](p: Pointer[Float32, MutAnyOrigin], i: Int) -> Float32:
    """Read an activation, from a buffer another block may have just written.

    `coherent` is false for every kernel that is launched once per operation,
    because the launch is the synchronisation and an ordinary load is correct.
    It is true inside the fused kernel, where the blocks that wrote this vector
    are the same blocks that are about to read it and the only thing between the
    two is a grid wide barrier.

    On Metal an ordinary load is not enough in that case, and the reason is not
    ordering. A block reads the same scratch at every record of every layer, so
    the second read hits a line its own L1 already holds and the barrier does
    not evict it, and the value that comes back is one round of the plan out of
    date. A relaxed device scope atomic load is what gets past that. On CUDA
    the barrier's release and acquire already cover it and this compiles to the
    same load either way.
    """
    comptime if coherent and CompilationTarget.is_macos():
        var q = p.unsafe_bitcast[Int32]()
        return bitcast[DType.float32, 1](
            _dev32.load[ordering=Ordering.RELAXED](
                Pointer[Int32, MutAnyOrigin](to=q[unsafe_offset=i])
            )
        )
    else:
        return p[unsafe_offset=i]


@always_inline
def planar_partial_dot[
    tile: Int, group: Int, with_min: Bool, form: Int, coherent: Bool = False
](
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    row: Int,
    cols: Int,
    t: Int,
) -> Float32:
    """Thread `t` of `tile`'s share of one planar row against one vector.

    Every loop that reads a weight in a decode goes through here, which is the
    point of it being its own function. It was the body of
    `planar_matvec_kernel` and a second copy of it lived in the fused kernel,
    and the copy went stale twice while the layout changed underneath it, in a
    way that builds and gives wrong answers rather than failing to compile.

    `tile` is the stride and not a block size. The matvec passes its block
    width and the fused kernel passes its own, and neither cares what the other
    uses, because a partial sum is defined by which values this thread took and
    not by who else is taking the rest.

    The scales are read through a float16 view of the same bytes rather than
    assembled a byte at a time. A planar row's quant plane is an even number of
    bytes wide for every row width molla accepts, so the scale planes are two
    byte aligned in every row of every tensor, and a device buffer starts at an
    alignment far larger than that.
    """
    var groups = cols // group
    # Two views of the same bytes rather than one view and a conversion. Every
    # quant is read as an unsigned byte and turned into a float by
    # `nibble_float` and `byte_float`, which carry the sign in the bias they
    # subtract rather than in the type they load through. That is why there is
    # no `Int8` view here any more: reading a signed byte only to convert it is
    # the instruction this kernel is trying not to issue, and the sign
    # extension a centred type needs is a constant in the subtraction either
    # way.
    var packed = w
    var scales = w.unsafe_bitcast[Float16]()
    var d_base = (row + planar_quant_stride[form](cols)) // SCALE_BYTES
    var m_base = d_base + groups

    # A shift and not `i // group`, which is a signed divide and which Metal
    # keeps. See `group_shift` for what it costs there.
    comptime shift = group_shift(group)

    var acc = Float32(0)
    comptime if form == QUANT_I8:
        var i = t * MATVEC_BYTES
        while i < cols:
            var gi = i >> shift
            var at = row + i
            var d = _scale(scales, d_base + gi)
            var m = Float32(0)
            comptime if with_min:
                m = _scale(scales, m_base + gi)
            comptime for k in range(MATVEC_BYTES):
                var q = byte_float(UInt32(packed[unsafe_offset=at + k]))
                var a = coherent_load[coherent](x, i + k)
                acc += d * q * a
                comptime if with_min:
                    acc += m * a
            i += tile * MATVEC_BYTES
    elif quant_high_bits(form) == 0:
        # A byte to a thread and both of its values, so a warp reads thirty two
        # contiguous bytes and gets sixty four values out of them. Both values
        # of a byte sit in one group, because every group size here is even, so
        # their scale and minimum are loaded once for the pair. The masking is
        # done with arithmetic rather than a branch on which nibble is which,
        # because that is a branch every warp would take both sides of, and the
        # high nibble needs no mask at all: the shift has already cleared
        # everything above it.
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
        # half its time on the group index, which is why that is a shift now,
        # and the largest thing left after that was the conversion of the quant,
        # which is why `nibble_float` exists.
        # See [docs/validation/layout.md](../../../docs/validation/layout.md).
        #
        # How many values a thread takes in one pass of the loop is
        # `MATVEC_STEP` and it is not the same number on the two backends. See
        # the comment on it for the measurement.
        var i = t * MATVEC_STEP
        while i < cols:
            var gi = i >> shift
            var at = row + (i >> 1)
            var d = _scale(scales, d_base + gi)
            var m = Float32(0)
            comptime if with_min:
                m = _scale(scales, m_base + gi)
            comptime for k in range(MATVEC_STEP // 2):
                var b = UInt32(packed[unsafe_offset=at + k])
                var lo = nibble_float[form](b & 0xF)
                var hi = nibble_float[form](b >> 4)
                var a0 = coherent_load[coherent](x, i + k * 2)
                var a1 = coherent_load[coherent](x, i + k * 2 + 1)
                acc += d * lo * a0
                acc += d * hi * a1
                comptime if with_min:
                    acc += m * a0
                    acc += m * a1
            i += tile * MATVEC_STEP
    else:
        # Five and six bit types, whose low four bits are the plane above and
        # whose remaining one or two are in a second plane after it. One byte of
        # that plane covers eight values at five bits and four at six, so the
        # step is at least that many and the byte is read once for all of them.
        # A step wider than one high byte reads several, which is what q6_K does
        # on Metal, where the step that pays is eight and a high byte is four
        # values.
        #
        # Nothing here converts. The two planes are joined with shifts and
        # masks into the unsigned value the repack wrote, and `wide_float` turns
        # that into a float and takes the type's offset off in the subtraction
        # the magic number was already doing. So a five bit value costs one
        # extra load per eight values and a handful of integer operations
        # against the four bit loop, and reads five eighths of the bytes the
        # byte wide fallback it replaces read.
        comptime hbits = quant_high_bits(form)
        comptime hshift = high_shift[form]()
        comptime hper = 1 << hshift
        comptime hmask = UInt32((1 << hbits) - 1)
        var high = row + (cols >> 1)
        var i = t * MATVEC_STEP
        while i < cols:
            var gi = i >> shift
            var at = row + (i >> 1)
            var d = _scale(scales, d_base + gi)
            var m = Float32(0)
            comptime if with_min:
                m = _scale(scales, m_base + gi)
            comptime for k in range(MATVEC_STEP // 2):
                var b = UInt32(packed[unsafe_offset=at + k])
                # `i` is a multiple of the step, so when the step covers a whole
                # high byte or more these two are constants after unrolling and
                # the load is hoisted out of the `k` loop for nothing. When it
                # covers less they are the index and the shift a narrower pass
                # needs. One loop rather than two shapes of it.
                var hi_at = (i + k * 2) >> hshift
                var s0 = UInt32(((i + k * 2) & (hper - 1)) * hbits)
                var h = UInt32(packed[unsafe_offset=high + hi_at])
                var lo = wide_float[form](
                    (b & 0xF) | (((h >> s0) & hmask) << 4)
                )
                var hv = wide_float[form](
                    (b >> 4) | (((h >> (s0 + UInt32(hbits))) & hmask) << 4)
                )
                var a0 = coherent_load[coherent](x, i + k * 2)
                var a1 = coherent_load[coherent](x, i + k * 2 + 1)
                acc += d * lo * a0
                acc += d * hv * a1
                comptime if with_min:
                    acc += m * a0
                    acc += m * a1
            i += tile * MATVEC_STEP
    return acc


comptime ACT_GROUP = 32
"""Activation values that share one quantization scale.

32 and not the weight's own group size, because a q4_K_M model has q4_K weights
grouped by 32 and q6_K weights grouped by 16 and both read the same activation
vector. A run of `ACT_RUN` is inside one group of either, since 8 divides both,
so the dot loop takes one weight scale and one activation scale for a whole run
whatever the weight type is.
"""

comptime ACT_RUN = 8
"""Values a thread multiplies before it converts and scales once.

The run is where the whole saving is. Eight integer multiplies into an `Int32`
and then one convert, against eight converts and sixteen floating point
operations for the same values in the float loop. It is a run rather than a
whole group so that a thread's share of a row stays contiguous and a warp still
reads consecutive bytes.
"""

comptime ACT_Q_MAX = 127
"""The largest magnitude an activation quant takes, which makes it a signed byte.
"""


def act_scale_words(n: Int) -> Int:
    """Float32 words in from the front of a quantized vector to its scales.

    The quants are one signed byte a value and the scales are one float32 a
    group, in one allocation, so the scale plane starts at the quant plane
    rounded up to a word. Every width this is used at is a multiple of 32 and
    the rounding is there for the widths that are not.
    """
    return (n + 3) // 4


struct DeviceQuantVec(Movable):
    """One activation vector as signed bytes and a scale a group, owned.

    The same two plane idea as a planar weight row, at the other end of the
    multiply. `DeviceVec` holds the float version and this holds what the
    integer matvec reads: `n` signed bytes, then one `Float32` a group of
    `ACT_GROUP`, in one buffer so a kernel takes one pointer and an offset
    rather than two pointers.

    Symmetric and not affine. The weight side already carries a minimum where
    its type has one, and a second offset here would put a cross term in every
    dot product that neither side wants to pay for.
    """

    var buf: DeviceBuffer[DType.uint8]
    var n: Int

    def __init__(out self, ctx: DeviceContext, n: Int) raises:
        if n <= 0:
            raise Error("a quantized vector needs a positive length")
        if n % ACT_GROUP != 0:
            raise Error(
                "a quantized vector is a whole number of groups of "
                + String(ACT_GROUP)
                + " and "
                + String(n)
                + " is not"
            )
        self.buf = ctx.enqueue_create_buffer[DType.uint8](
            act_scale_words(n) * 4 + (n // ACT_GROUP) * 4
        )
        self.n = n

    def elements(self) -> Int:
        return self.n

    def scale_words(self) -> Int:
        """Where the scales start, counted in float32 words."""
        return act_scale_words(self.n)

    def ptr(self) -> Pointer[UInt8, MutAnyOrigin]:
        return Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(self.buf.unsafe_ptr())
        )


def act_quant_kernel(
    x: Pointer[Float32, MutAnyOrigin],
    q: Pointer[UInt8, MutAnyOrigin],
    words_dev: Int32,
):
    """One group of activations to a signed byte each and one scale.

    A block a group and a thread a value, so the reduction that finds the
    group's largest magnitude is a block reduction over `ACT_GROUP` threads and
    nothing crosses a block. The scale is that magnitude over 127 and the quant
    is the value over the scale, rounded to nearest, which is what llama.cpp's
    `quantize_row_q8_1` does and is the arithmetic the error budget in
    [docs/validation/layout.md](../../../docs/validation/layout.md) is written
    against.

    A group of all zeros gets a scale of zero and quants of zero, and reads back
    as exactly zero rather than as a division by nothing.
    """
    var g = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var i = g * ACT_GROUP + t

    var v = x[unsafe_offset=i]
    var mag = v if v >= 0 else -v
    var part = stack_allocation[
        ACT_GROUP, Float32, address_space=AddressSpace.SHARED
    ]()
    part[unsafe_offset=t] = mag
    barrier()
    var step = ACT_GROUP // 2
    while step > 0:
        if t < step:
            var l = part[unsafe_offset=t]
            var r = part[unsafe_offset=t + step]
            part[unsafe_offset=t] = l if l > r else r
        barrier()
        step //= 2
    var top = part[unsafe_offset=0]

    var scale = top / Float32(ACT_Q_MAX)
    var inv = Float32(0) if top == Float32(0) else Float32(ACT_Q_MAX) / top
    var scaled = v * inv
    var rounded = scaled + (Float32(0.5) if scaled >= 0 else Float32(-0.5))
    q.unsafe_bitcast[Int8]()[unsafe_offset=i] = Int8(Int32(rounded))
    if t == 0:
        q.unsafe_bitcast[Float32]()[unsafe_offset=Int(words_dev) + g] = scale


def planar_partial_dot_q[
    tile: Int, group: Int, with_min: Bool, form: Int
](
    w: Pointer[UInt8, MutAnyOrigin],
    xq: Pointer[UInt8, MutAnyOrigin],
    row: Int,
    cols: Int,
    words: Int,
    t: Int,
) -> Float32:
    """`planar_partial_dot` with both sides integers until a run ends.

    Same row, same order of accumulation over groups, same planes read in the
    same way. What differs is everything between the load and the accumulator:
    a weight quant stays the integer the repack wrote, an activation is the
    signed byte beside it, and their products go into an `Int32` for a run of
    `ACT_RUN` before anything becomes a float.

    The offset a centred type carries is not paid per value. A value is
    `raw - bias`, so a run is `sum(raw * a) - bias * sum(a)`, and the plain sum
    of the activation quants is already being accumulated for the minimum. So a
    q4_0 or a q6_K run costs one extra integer multiply rather than one extra
    subtraction a value, and an unsigned type costs nothing at all.

    The minimum is the same algebra. Over a group the value is `d * raw + m` and
    the sum against `x = dx * q` is `d * dx * sum(raw * q) + m * dx * sum(q)`, so
    a run converts twice and scales twice at the end instead of multiplying and
    adding per value. A type with no minimum and no offset drops the second
    accumulator at compile time.

    Neither accumulator overflows. The widest run is eight six bit quants at 63
    against activation quants at 127, which is 64008, and a whole 4096 wide row
    of that would still be inside an `Int32`.
    """
    var groups = cols // group
    var packed = w
    var scales = w.unsafe_bitcast[Float16]()
    var d_base = (row + planar_quant_stride[form](cols)) // SCALE_BYTES
    var m_base = d_base + groups

    var aq = xq.unsafe_bitcast[Int8]()
    var ad = xq.unsafe_bitcast[Float32]()

    comptime shift = group_shift(group)
    comptime ashift = group_shift(ACT_GROUP)
    # What the reader subtracts to get the signed value, which is eight for the
    # centred nibble type, the type's own bias for the five and six bit ones,
    # and nothing for the unsigned nibbles or for the byte type, which is loaded
    # through a signed view and is already the value it means.
    comptime bias = (
        8 if form
        == QUANT_S4 else (
            0 if (
                form == QUANT_U4 or form == QUANT_U5 or form == QUANT_I8
            ) else quant_bias(form)
        )
    )
    comptime needs_sum = with_min or bias != 0

    var acc = Float32(0)
    var i = t * ACT_RUN
    while i < cols:
        var gi = i >> shift
        var ai = i >> ashift
        var dot = Int32(0)
        var qsum = Int32(0)

        comptime if form == QUANT_I8:
            var at = row + i
            var quants = w.unsafe_bitcast[Int8]()
            comptime for k in range(ACT_RUN):
                var a = Int32(aq[unsafe_offset=i + k])
                dot += Int32(quants[unsafe_offset=at + k]) * a
                comptime if needs_sum:
                    qsum += a
        elif quant_high_bits(form) == 0:
            var at = row + (i >> 1)
            comptime for k in range(ACT_RUN // 2):
                var b = Int32(packed[unsafe_offset=at + k])
                var a0 = Int32(aq[unsafe_offset=i + k * 2])
                var a1 = Int32(aq[unsafe_offset=i + k * 2 + 1])
                dot += (b & 0xF) * a0 + ((b >> 4) & 0xF) * a1
                comptime if needs_sum:
                    qsum += a0 + a1
        else:
            comptime hbits = quant_high_bits(form)
            comptime hshift = high_shift[form]()
            comptime hper = 1 << hshift
            comptime hmask = UInt32((1 << hbits) - 1)
            var high = row + (cols >> 1)
            var at = row + (i >> 1)
            comptime for k in range(ACT_RUN // 2):
                var b = UInt32(packed[unsafe_offset=at + k])
                var hi_at = (i + k * 2) >> hshift
                var s0 = UInt32(((i + k * 2) & (hper - 1)) * hbits)
                var h = UInt32(packed[unsafe_offset=high + hi_at])
                var v0 = Int32((b & 0xF) | (((h >> s0) & hmask) << 4))
                var v1 = Int32(
                    (b >> 4) | (((h >> (s0 + UInt32(hbits))) & hmask) << 4)
                )
                var a0 = Int32(aq[unsafe_offset=i + k * 2])
                var a1 = Int32(aq[unsafe_offset=i + k * 2 + 1])
                dot += v0 * a0 + v1 * a1
                comptime if needs_sum:
                    qsum += a0 + a1

        comptime if bias != 0:
            dot -= Int32(bias) * qsum

        var dx = ad[unsafe_offset=words + ai]
        acc += _scale(scales, d_base + gi) * dx * Float32(dot)
        comptime if with_min:
            acc += _scale(scales, m_base + gi) * dx * Float32(qsum)
        i += tile * ACT_RUN
    return acc


def planar_matvec_q_kernel[
    tile: Int, group: Int, with_min: Bool, form: Int
](
    w: Pointer[UInt8, MutAnyOrigin],
    xq: Pointer[UInt8, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    aux: Pointer[Float32, MutAnyOrigin],
    cols_dev: Int32,
    stride_dev: Int32,
    words_dev: Int32,
    epi_dev: Int32,
):
    """`planar_matvec_kernel` against a quantized activation vector.

    The reduction and the epilogue are the float kernel's, line for line, so the
    only thing this instantiation changes is which dot product runs.
    """
    var cols = Int(cols_dev)
    var stride = Int(stride_dev)
    var r = Int(block_idx.x)
    var t = Int(thread_idx.x)

    var acc = planar_partial_dot_q[tile, group, with_min, form](
        w, xq, r * stride, cols, Int(words_dev), t
    )

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
        write_epilogue(o, aux, Int(epi_dev), r, r, part[unsafe_offset=0])


def planar_matvec_kernel[
    tile: Int, group: Int, with_min: Bool, form: Int
](
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    aux: Pointer[Float32, MutAnyOrigin],
    cols_dev: Int32,
    stride_dev: Int32,
    epi_dev: Int32,
):
    """`o[r] = dot(planar row r of w, x)`, one thread block per row.

    The row is `planar_partial_dot` and what is here is the reduction over the
    block and the epilogue.

    `epi` says what happens to the row once it is reduced, and `aux` is the one
    other vector that needs, which is a bias plane or the up projection. It is a
    runtime argument and not a parameter on purpose. One thread of the block
    runs it, once, after everything else the block does, so making it a
    parameter would multiply five instantiations of the whole kernel by four to
    save one predicted branch executed once per block. `aux` is the output
    pointer when nothing reads it, so there is no null to check for.

    Shapes arrive as `Int32` because a kernel argument has to be device
    passable and `Int` is not. Nothing here is near two billion.
    """
    var cols = Int(cols_dev)
    var stride = Int(stride_dev)
    var r = Int(block_idx.x)
    var t = Int(thread_idx.x)

    var row = r * stride
    var acc = planar_partial_dot[tile, group, with_min, form](
        w, x, row, cols, t
    )

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
        # One token, so the index into the output and the row are the same
        # number, which is why a gate reads `aux[r]` here and `aux[out_at]`
        # there and both are the same read.
        write_epilogue(o, aux, Int(epi_dev), r, r, part[unsafe_offset=0])


comptime MATVEC_BYTES = 8 if CompilationTarget.is_macos() else 1
"""The same number for the byte wide path, where CUDA wants a narrower one.

Metal takes the widening here as well as on the nibble path. CUDA does not: at
two values a thread SmolLM2 135M, which is q8_0 and so is all of it, decoded
236.6 tokens a second against 262.3 at one, on a 4090 with everything else
equal. So this one stays where it was on that backend.
"""

comptime MATVEC_STEP = 8 if CompilationTarget.is_macos() else 2
"""How many four bit values one thread takes in one pass of the decode matvec.

The other number the two backends disagree about, and the disagreement is
larger here than anywhere else in this file. `scripts/matvec_probe.mojo` runs
the same loop at two, four and eight over the three shapes an 8B decode spends
its matvec time in, against a variant that does the loads and no arithmetic at
all, which is the floor the shape can reach.

On an M4 the loop goes 1377, 899 and 639 microseconds on the down projection at
two, four and eight, where the floor is 577. So eight is within ten per cent of
reading the bytes and doing nothing with them, and the other two shapes move by
the same fraction. Nothing about the arithmetic changed to get that, only how
many values a thread carries between two reads of the group scale.

On a 4090 the same three are 37, 37 and 52, so four is neutral and eight is a
regression of two fifths. A thread there already has enough of the row in
flight and widening the step costs occupancy instead of buying reuse.

The mask trick from the Metal q4_K matvec in llama.cpp, which #203 asks for and
which `V_MASK16` in the probe implements, is in that same table. Against the
loop at the same step width it is worth nothing on Metal and between nothing
and a fifth on CUDA depending on the shape, so the shifts are not what this
loop is paying for and the step width is.
"""

comptime SPAN = 16 if CompilationTarget.is_macos() else 8
"""How many tokens one group of threads in a matmul block carries at once.

The number that decides whether prefill is bandwidth bound or compute bound. A
block reads and dequantizes a weight value once and multiplies it into `SPAN`
accumulators, so the weight traffic and the conversion work for a chunk of `T`
tokens are `ceil(T / SPAN)` passes over the matrix rather than `T` of them. It
costs `SPAN` registers of accumulator, which is what stops it growing, and
`MM_GROUPS` is how the amortization goes past what the registers allow.

The two backends want different numbers and the difference is measured and
consistent rather than noise. Metal is 11 per cent faster at sixteen than at
eight and CUDA is 19 per cent faster at eight than at sixteen, on both models
that fit in cache. A macOS build targets Metal and a build anywhere else
targets CUDA, which is why the operating system is the thing asked here.
"""

comptime PREFILL_CHUNK = 256 if CompilationTarget.is_macos() else 64
"""How many prompt tokens go through the stack in one pass.

A prompt is a matrix rather than a run of decodes, and this is how much of it
is in flight at once. It trades launches against scratch: the launches for a
prompt fall by this factor, and the attention scores grow by it, which on a
thirty two head model at a four thousand context is 2 MiB a token. Sixty four
is 128 times fewer launches than a token at a time and 33 MiB of scores on an
8B, and at four passes for a five hundred token prompt the launches are no
longer what is left.

It is two numbers because the two backends want different ones, and by a lot.
On Metal the matrix core form covers thirty two tokens a block, so a chunk of
sixty four is two blocks of the token axis and the GPU runs out of work before
it runs out of rows: widening the chunk to 256 is 1611 tokens a second against
2142 on SmolLM2. On CUDA the same widening goes the other way, and not by a
little, 4990 to 3545 on Qwen and 378 to 181 on the 8B. That is #213, and a
chunk picked for the kernel that runs on the target is the honest thing to
write until it is answered.
"""

comptime MM_TILE = 32
"""Threads in a batched matmul block, which is not the matvec's tile.

A quarter of it, measured. A matmul block reduces `SPAN` accumulators rather
than one, so the reduction at the end of it costs `SPAN` times what the
matvec's does, and past a certain width the reduction is more work than the dot
product that fed it. Thirty two threads is the best of every width from sixteen
to five hundred and twelve on both backends and on all three models, and the
losses either side of it are large: on a 4090 a 514 token prompt through
SmolLM2 runs at 2734 tokens a second here, 2089 at 128 threads and 1034 at 512.

It is also a warp on both backends, which the reduction relies on and which
lets a group of threads whose tokens are all past the end of a short chunk
leave without stranding the rest of the block.
"""

comptime MM_BLOCK_TOKENS = 64
"""How many tokens one block of the ordinary matmul carries.

The chunk used to be this by construction and the two came apart when the
matrix core form arrived, which wants a much wider chunk than the sweeps below
found for this kernel. Sixty four is what those sweeps landed on and the grid
grows a block a chunk instead of a block growing with the chunk.
"""

comptime MM_GROUPS = MM_BLOCK_TOKENS // SPAN
"""How many groups of `SPAN` tokens one matmul block carries.

Amortization the accumulators cannot pay for. A block covers `SPAN` tokens per
group of threads because `SPAN` accumulators is as many as fit in registers, so
without this a chunk of 64 tokens reads the whole weight matrix eight times,
and on an 8B that was 52 GiB of reads a chunk and a prefill slower than
decoding the prompt a token at a time.

The groups share the weight row rather than the registers. Each one walks the
same columns and keeps its own accumulators, so the row is fetched from memory
once for the block and out of the L1 for the groups behind the first, and the
traffic for a chunk falls by this factor with no register cost at all.

A whole chunk to a block, which is the number the sweeps land on from either
side. Both backends want `SPAN * MM_GROUPS` to be the chunk exactly: on a 4090
the 8B runs at 388 tokens a second with eight groups of eight and 262 with four
of eight, and Metal falls off a cliff the other way when the product is twice
the chunk. Once a block is the whole chunk, the grid is one block to an output
row and the weight matrix is read once a chunk rather than once a block.
"""


def planar_matmul_kernel[
    tile: Int, group: Int, with_min: Bool, form: Int
](
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    aux: Pointer[Float32, MutAnyOrigin],
    cols_dev: Int32,
    stride_dev: Int32,
    epi_dev: Int32,
    rows_dev: Int32,
    tokens_dev: Int32,
):
    """`o[t][r] = dot(row r of w, x[t])` for `SPAN` tokens at a time.

    The same inner loop as `planar_matvec_kernel` with the accumulator widened
    from one float to `SPAN` of them. That is the entire difference and it is
    the entire point: the expensive part of a decode matvec is reading a weight
    byte and turning it into a float, and both of those happen once here for
    `SPAN` multiplies rather than once for one.

    The block is `MM_GROUPS` groups of `tile` threads. Every group walks the
    whole weight row and holds `SPAN` accumulators of its own, so the row is
    read from memory once for the block and out of the L1 for the groups behind
    the first, and a block covers `SPAN * MM_GROUPS` tokens for one pass over
    the weights. That is the difference between reading an 8B eight times a
    chunk and reading it twice.

    The grid is `(ceil(tokens / (SPAN * MM_GROUPS)), rows)` and the order is
    deliberate. Blocks that share an output row differ only in the token index,
    and they have to be co-resident for the L2 to serve that row once rather
    than once each, so the token index is the fast axis.

    A chunk is not a multiple of `SPAN` in general, and the tail block runs its
    dead lanes off the end of the chunk rather than branching around them. Every
    scratch vector a chunk uses is allocated with a block of rows of slack for
    exactly this, so those lanes read real memory, compute a dot product of
    whatever is in it, and never store it. It costs the tail block alone a
    fraction of one launch and it keeps the inner loop free of a test.

    See [docs/validation/prefill.md](../../../docs/validation/prefill.md).
    """
    var cols = Int(cols_dev)
    var stride = Int(stride_dev)
    var rows = Int(rows_dev)
    var tokens = Int(tokens_dev)
    var r = Int(block_idx.y)
    var g = Int(thread_idx.y)
    var base = (Int(block_idx.x) * MM_GROUPS + g) * SPAN
    var t = Int(thread_idx.x)

    var row = r * stride
    var groups = cols // group
    var packed = w
    var scales = w.unsafe_bitcast[Float16]()
    var d_base = (row + planar_quant_stride[form](cols)) // SCALE_BYTES
    var m_base = d_base + groups

    comptime shift = group_shift(group)

    var live = tokens - base
    if live > SPAN:
        live = SPAN
    # A group whose whole run is past the end of the chunk stops here rather
    # than walking the columns for nothing. It can, because a group is a warp of
    # its own and the reduction below is the only thing the lanes of a block
    # agree on, so there is no barrier left for the ones still working to wait
    # at. That holds only while a group is exactly a warp wide.
    if tile == WARP_SIZE and live <= 0:
        return

    # Every loop over `SPAN` below has a constant trip count so that the index
    # into `acc` is a constant after unrolling and the accumulators stay in
    # registers. A runtime bound would put all of them in local memory and undo
    # the whole change.
    #
    # The dead lanes of a tail block read past the last token rather than
    # clamping onto it, which is why every scratch vector a chunk uses is
    # allocated with `SPAN * MM_GROUPS` rows of slack. Clamping would need a
    # token index per lane held in a register for the length of the
    # accumulation, which is
    # `SPAN` registers taken from the accumulators for arithmetic that is
    # thrown away. Reading slack is an affine offset the address unit folds in
    # for free, and what it computes is discarded by the `t < live` below.
    var at0 = base * cols

    var acc = InlineArray[Float32, SPAN](fill=0)
    comptime if form == QUANT_I8:
        var i = t
        while i < cols:
            var gi = i >> shift
            var q = byte_float(UInt32(packed[unsafe_offset=row + i]))
            var d = _scale(scales, d_base + gi) * q
            comptime if with_min:
                var m = _scale(scales, m_base + gi)
                for k in range(SPAN):
                    var a = x[unsafe_offset=at0 + k * cols + i]
                    acc[k] += d * a + m * a
            else:
                for k in range(SPAN):
                    acc[k] += d * x[unsafe_offset=at0 + k * cols + i]
            i += tile
    elif quant_high_bits(form) == 0:
        var i = t * 2
        while i < cols:
            var gi = i >> shift
            var b = UInt32(packed[unsafe_offset=row + (i >> 1)])
            var d = _scale(scales, d_base + gi)
            var lo = d * nibble_float[form](b & 0xF)
            var hi = d * nibble_float[form](b >> 4)
            comptime if with_min:
                var m = _scale(scales, m_base + gi)
                for k in range(SPAN):
                    var a0 = x[unsafe_offset=at0 + k * cols + i]
                    var a1 = x[unsafe_offset=at0 + k * cols + i + 1]
                    acc[k] += lo * a0 + hi * a1 + m * (a0 + a1)
            else:
                for k in range(SPAN):
                    var a0 = x[unsafe_offset=at0 + k * cols + i]
                    var a1 = x[unsafe_offset=at0 + k * cols + i + 1]
                    acc[k] += lo * a0 + hi * a1
            i += tile * 2
    else:
        # The five and six bit forms, joined the same way the matvec joins them.
        # A step of one high byte and no more, because the accumulators here are
        # `SPAN` deep and a wider step would hold more of the row in registers
        # at the cost of the occupancy that keeps `SPAN` tokens in flight.
        comptime hbits = quant_high_bits(form)
        comptime hshift = high_shift[form]()
        comptime hper = 1 << hshift
        comptime hmask = UInt32((1 << hbits) - 1)
        var high = row + (cols >> 1)
        var i = t * 2
        while i < cols:
            var gi = i >> shift
            var b = UInt32(packed[unsafe_offset=row + (i >> 1)])
            var h = UInt32(packed[unsafe_offset=high + (i >> hshift)])
            var s0 = UInt32((i & (hper - 1)) * hbits)
            var d = _scale(scales, d_base + gi)
            var lo = d * wide_float[form](
                (b & 0xF) | (((h >> s0) & hmask) << 4)
            )
            var hv = d * wide_float[form](
                (b >> 4) | (((h >> (s0 + UInt32(hbits))) & hmask) << 4)
            )
            comptime if with_min:
                var m = _scale(scales, m_base + gi)
                for k in range(SPAN):
                    var a0 = x[unsafe_offset=at0 + k * cols + i]
                    var a1 = x[unsafe_offset=at0 + k * cols + i + 1]
                    acc[k] += lo * a0 + hv * a1 + m * (a0 + a1)
            else:
                for k in range(SPAN):
                    var a0 = x[unsafe_offset=at0 + k * cols + i]
                    var a1 = x[unsafe_offset=at0 + k * cols + i + 1]
                    acc[k] += lo * a0 + hv * a1
            i += tile * 2

    # A lane group reduction and not a tree through shared memory. A group is a
    # warp wide, so `SPAN` reductions are `SPAN` times five shuffles with no
    # shared memory, no barrier, and no occupancy given up to a scratch array
    # that is only touched in the last few instructions of the kernel. The tree
    # this replaced was more work than the dot product that fed it whenever the
    # matrix was narrow.
    #
    # Every lane ends up holding every total, so each one keeps the total for
    # the token it is about to write and drops the rest.
    var got = Float32(0)
    comptime for k in range(SPAN):
        var whole = lane_group_sum[num_lanes=tile](acc[k])
        if t == k:
            got = whole

    # One thread a token rather than one thread for the whole group, so the
    # tail is `SPAN` threads doing one epilogue each.
    if t < live:
        write_epilogue(o, aux, Int(epi_dev), (base + t) * rows + r, r, got)


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
    # A matvec reads one row from the front of its input, and since prefill a
    # scratch vector is a chunk of rows rather than one, so the width it has to
    # hold is a whole number of rows and at least one. Exact equality would
    # refuse the output head reading the last row of a prefill chunk, and a
    # width that is not a multiple of the row is still the wiring mistake this
    # check exists to catch.
    if in_elements < w.cols or in_elements % w.cols != 0:
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
    aux: Pointer[Float32, MutAnyOrigin],
    epi: Int,
) raises:
    """One instantiation, launched. No transfer either side of it."""
    ctx.enqueue_function[planar_matvec_kernel[tile, group, with_min, form]](
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=w.device_address()),
        x.ptr(),
        o,
        aux,
        Int32(w.cols),
        Int32(w.row_bytes()),
        Int32(epi),
        grid_dim=(w.rows, 1, 1),
        block_dim=(tile, 1, 1),
    )


def device_matvec_into(
    ctx: DeviceContext,
    w: Tensor,
    x: DeviceVec,
    mut out: DeviceVec,
    at: Int = 0,
    epi: Int = EPI_NONE,
    aux: Optional[Pointer[Float32, MutAnyOrigin]] = None,
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
        # No null to check for inside the kernel, so a caller that asked for
        # nothing gets the output pointer and a kernel that reads it is one that
        # was asked to.
        var a = aux.value() if aux and epi != EPI_NONE else o
        var g = group_size(w.kind)
        var carries_min = has_min(w.kind)
        var form = quant_form(w.kind)
        if form == QUANT_U4 and g == 32 and carries_min:
            _launch[MATVEC_TILE, 32, True, QUANT_U4](ctx, w, x, o, a, epi)
        elif form == QUANT_S4 and g == 32 and not carries_min:
            _launch[MATVEC_TILE, 32, False, QUANT_S4](ctx, w, x, o, a, epi)
        elif form == QUANT_U5 and g == 32 and carries_min:
            _launch[MATVEC_TILE, 32, True, QUANT_U5](ctx, w, x, o, a, epi)
        elif form == QUANT_S5 and g == 32 and not carries_min:
            _launch[MATVEC_TILE, 32, False, QUANT_S5](ctx, w, x, o, a, epi)
        elif form == QUANT_S6 and g == 16 and not carries_min:
            _launch[MATVEC_TILE, 16, False, QUANT_S6](ctx, w, x, o, a, epi)
        elif form == QUANT_I8 and g == 32 and not carries_min:
            _launch[MATVEC_TILE, 32, False, QUANT_I8](ctx, w, x, o, a, epi)
        else:
            raise Error(
                "no device matvec is compiled for quant form "
                + String(form)
                + " with a group of "
                + String(g)
                + (" and a minimum plane" if carries_min else " and no minimum")
            )


def device_quantize(
    ctx: DeviceContext, x: DeviceVec, mut out: DeviceQuantVec
) raises:
    """One activation vector into signed bytes and a scale a group.

    One launch, a block a group. It is a launch and not part of whatever
    produced the vector because the thing that produced it is a norm or an
    activation function that already walks the vector, and folding this into
    those is worth doing once this is known to pay rather than before.
    """
    if x.elements() != out.elements():
        raise Error(
            "quantizing "
            + String(x.elements())
            + " values into a vector of "
            + String(out.elements())
        )
    comptime if not has_accelerator():
        raise Error(
            "this build has no device code in it, so there is nothing to"
            " quantize an activation vector with"
        )
    else:
        ctx.enqueue_function[act_quant_kernel](
            x.ptr(),
            out.ptr(),
            Int32(out.scale_words()),
            grid_dim=(out.elements() // ACT_GROUP, 1, 1),
            block_dim=(ACT_GROUP, 1, 1),
        )


def _launch_q[
    tile: Int, group: Int, with_min: Bool, form: Int
](
    ctx: DeviceContext,
    w: Tensor,
    x: DeviceQuantVec,
    o: Pointer[Float32, MutAnyOrigin],
    aux: Pointer[Float32, MutAnyOrigin],
    epi: Int,
) raises:
    """One instantiation of the integer matvec, launched."""
    ctx.enqueue_function[planar_matvec_q_kernel[tile, group, with_min, form]](
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=w.device_address()),
        x.ptr(),
        o,
        aux,
        Int32(w.cols),
        Int32(w.row_bytes()),
        Int32(x.scale_words()),
        Int32(epi),
        grid_dim=(w.rows, 1, 1),
        block_dim=(tile, 1, 1),
    )


def device_matvec_q_into(
    ctx: DeviceContext,
    w: Tensor,
    x: DeviceQuantVec,
    mut out: DeviceVec,
    at: Int = 0,
    epi: Int = EPI_NONE,
    aux: Optional[Pointer[Float32, MutAnyOrigin]] = None,
) raises:
    """`device_matvec_into` against a vector that has already been quantized.

    The same dispatch over the same five shapes of row, because the weight side
    is untouched by this and a form that has no float kernel should not gain an
    integer one by accident.

    Quantizing is the caller's because one vector feeds several matvecs: the
    attention norm feeds three projections and the feed forward norm feeds two,
    so doing it here would quantize the same values twice and lose most of what
    this is for.
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
        var a = aux.value() if aux and epi != EPI_NONE else o
        var g = group_size(w.kind)
        var carries_min = has_min(w.kind)
        var form = quant_form(w.kind)
        if w.cols % ACT_RUN != 0:
            raise Error(
                "the integer matvec takes a run of "
                + String(ACT_RUN)
                + " values a thread and a row of "
                + String(w.cols)
                + " is not a whole number of runs"
            )
        if form == QUANT_U4 and g == 32 and carries_min:
            _launch_q[TILE, 32, True, QUANT_U4](ctx, w, x, o, a, epi)
        elif form == QUANT_S4 and g == 32 and not carries_min:
            _launch_q[TILE, 32, False, QUANT_S4](ctx, w, x, o, a, epi)
        elif form == QUANT_U5 and g == 32 and carries_min:
            _launch_q[TILE, 32, True, QUANT_U5](ctx, w, x, o, a, epi)
        elif form == QUANT_S5 and g == 32 and not carries_min:
            _launch_q[TILE, 32, False, QUANT_S5](ctx, w, x, o, a, epi)
        elif form == QUANT_S6 and g == 16 and not carries_min:
            _launch_q[TILE, 16, False, QUANT_S6](ctx, w, x, o, a, epi)
        elif form == QUANT_I8 and g == 32 and not carries_min:
            _launch_q[TILE, 32, False, QUANT_I8](ctx, w, x, o, a, epi)
        else:
            raise Error(
                "no integer device matvec is compiled for quant form "
                + String(form)
                + " with a group of "
                + String(g)
                + (" and a minimum plane" if carries_min else " and no minimum")
            )


comptime MMA_ROWS = 64
"""Output rows one matrix core block covers."""

comptime MMA_TOKENS = 32
"""Output tokens one matrix core block covers.

The same sixty four by thirty two `kernel_mul_mm` uses, and the same answer a
sweep here gives. Sixty four tokens is fewer passes over the weight matrix and
fewer barriers for the same arithmetic and it measured slower, because a block
that covers twice as much of the output is half as many blocks and this GPU has
ten cores to keep busy.
"""

comptime MMA_K = 32
"""How far down the reduction one staged tile reaches.

The group size of every quantized type molla repacks, which is what makes the
scale and the minimum of a row constant for the whole step and turns the
dequantization into one multiply and one add per value with no per value
lookup.
"""

comptime MMA_THREADS = 128
"""Threads in a matrix core block, which is four simdgroups."""

comptime MMA_SG_ROWS = MMA_ROWS // 2
comptime MMA_SG_TOKENS = MMA_TOKENS // 2
comptime MMA_FR = MMA_SG_ROWS // 8
comptime MMA_FT = MMA_SG_TOKENS // 8
"""What one simdgroup of the four covers, arranged two by two, in fragments.

Two by two rather than four by one because a square block of the output shares
both operands between the fragment loads, and `MMA_FR * MMA_FT` accumulators of
two floats a lane is what it costs.
"""

comptime MMA_ROW_THREADS = MMA_THREADS // MMA_ROWS
comptime MMA_KPT = MMA_K // MMA_ROW_THREADS
"""How the weight staging is split: threads to a row, and values to a thread.

A thread's run has to sit inside one quant group so the scale and the minimum
are loaded once for it, which holds for every group size and every tile width
here and is the reason the dequantization is two instructions a value.
"""

comptime MMA_XV = (MMA_TOKENS * MMA_K) // 8
"""Vector loads the activation tile takes, at eight floats each.

Staging with vector loads rather than element at a time is a factor of four and
a half on its own, measured in #201, which is more than the tile is worth
without it.
"""

comptime MMA_STAGE = MMA_TOKENS * MMA_K + MMA_K * MMA_ROWS
comptime MMA_POOL = MMA_STAGE if MMA_STAGE > MMA_TOKENS * MMA_ROWS else MMA_TOKENS * MMA_ROWS
"""Floats of threadgroup memory a block takes, once.

The two staged operands are live together and the output tile is live after
both are dead, so all three come out of one allocation with a barrier between
the phases. Kept apart they are half again as much, and threadgroup memory is
what decides how many blocks a core can hold at once.
"""


def planar_mma_kernel[
    group: Int, with_min: Bool, form: Int
](
    w: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    aux: Pointer[Float32, MutAnyOrigin],
    cols_dev: Int32,
    stride_dev: Int32,
    epi_dev: Int32,
    rows_dev: Int32,
    tokens_dev: Int32,
):
    """The same product as `planar_matmul_kernel`, on Apple simdgroup matrices.

    `o[t][r] = dot(row r of w, x[t])`, which is `X` times `W` transposed with
    the activations as the left operand, because that is the orientation whose
    result comes out in the order the output is written in.

    The multiply is `_mma_apple_8x8`, an eight by eight by eight fragment
    multiply accumulate that every Apple GPU since the M1 has. #201 measures a
    dense version of this tile at 3.04 TFLOP/s on an M4 against the 0.30 the
    same arrangement reaches with ordinary multiplies, and llama.cpp's Metal
    prefill on the same machine works out at 2.55, so the instruction is both
    the whole gap and enough of it.

    Both operands are staged in threadgroup memory and both are staged with
    vector loads. That second half is not a detail. The same tile with the same
    traffic loading four bytes at a time rather than thirty two measured 0.30
    against 1.34, so a scalar staging loop costs more than four times what the
    tile itself is worth.

    Everything stays float. The instruction takes half precision operands as
    well and runs them at the same rate on this hardware, 2.95 TFLOP/s against
    3.04, so half would buy half the threadgroup traffic and nothing else,
    while costing three orders of magnitude of agreement with the decode path:
    the same model measured 2.4e-7 of relative error in float and 4.5e-4 in
    half, against a numerics gate of 2e-4. Half precision activations are what
    llama.cpp does here and it is not what the rate on this GPU asks for.

    The weight is transposed on the way in, since the fragment multiply has no
    transposed form at this size and the weight arrives row major over the
    reduction. A thread takes one row and one quant group of it, so the scale
    and the minimum are read once for the whole run.

    The result goes to the staging tile before the epilogue rather than
    straight out. A fragment store writes a lane's two elements at whatever
    positions the hardware layout puts them, and the epilogue is per element
    work with a bias index and a gate index in it, so reading the tile back in
    the obvious order is what keeps the epilogue the same code as the other two
    kernels rather than a second version of it that depends on a private
    fragment layout. The staged operands are dead by then, which is why it
    costs no memory.
    """
    var cols = Int(cols_dev)
    var stride = Int(stride_dev)
    var rows = Int(rows_dev)
    var tokens = Int(tokens_dev)

    var pool = stack_allocation[
        MMA_POOL, Float32, address_space=AddressSpace.SHARED
    ]()
    var x_tile = pool
    var w_tile = pool.unsafe_offset(MMA_TOKENS * MMA_K)
    var o_tile = pool

    var tid = Int(thread_idx.x)
    var sg = tid // WARP_SIZE
    var sg_t = (sg // 2) * MMA_SG_TOKENS
    var sg_r = (sg % 2) * MMA_SG_ROWS

    var t0 = Int(block_idx.x) * MMA_TOKENS
    var r0 = Int(block_idx.y) * MMA_ROWS

    var wr = tid // MMA_ROW_THREADS
    var wk = (tid % MMA_ROW_THREADS) * MMA_KPT
    var live_row = r0 + wr < rows
    var row = (r0 + wr) * stride if live_row else 0
    var groups = cols // group
    var scales = w.unsafe_bitcast[Float16]()
    var d_base = (row + planar_quant_stride[form](cols)) // SCALE_BYTES
    var m_base = d_base + groups
    comptime shift = group_shift(group)

    var acc = InlineArray[SIMD[DType.float32, 2], MMA_FR * MMA_FT](fill=0)

    for k0 in range(0, cols, MMA_K):
        # The token tail reads the slack every scratch vector a prefill uses is
        # allocated with, for the reason `planar_matmul_kernel` gives. The row
        # tail cannot do that, since past the last row is past the tensor, so
        # it stages zeros and the epilogue drops them.
        for v in range(tid, MMA_XV, MMA_THREADS):
            var xt = v // (MMA_K // 8)
            var xk = (v % (MMA_K // 8)) * 8
            x_tile.unsafe_offset(xt * MMA_K + xk).unsafe_store(
                x.unsafe_offset((t0 + xt) * cols + k0 + xk).unsafe_load[
                    width=8
                ]()
            )

        var gi = (k0 + wk) >> shift
        var d = _scale(scales, d_base + gi) if live_row else Float32(0)
        var m = Float32(0)
        comptime if with_min:
            if live_row:
                m = _scale(scales, m_base + gi)
        comptime if form == QUANT_I8:
            var q = w.unsafe_offset(row + k0 + wk).unsafe_load[
                width=MMA_KPT
            ]() if live_row else SIMD[DType.uint8, MMA_KPT](0)
            comptime for j in range(MMA_KPT):
                w_tile[unsafe_offset=(wk + j) * MMA_ROWS + wr] = (
                    d * byte_float(UInt32(q[j])) + m
                )
        elif quant_high_bits(form) == 0:
            var q = w.unsafe_offset(row + ((k0 + wk) >> 1)).unsafe_load[
                width=MMA_KPT // 2
            ]() if live_row else SIMD[DType.uint8, MMA_KPT // 2](0)
            comptime for j in range(MMA_KPT // 2):
                var b = UInt32(q[j])
                w_tile[unsafe_offset=(wk + 2 * j) * MMA_ROWS + wr] = (
                    d * nibble_float[form](b & 0xF) + m
                )
                w_tile[unsafe_offset=(wk + 2 * j + 1) * MMA_ROWS + wr] = (
                    d * nibble_float[form](b >> 4) + m
                )
        else:
            # Two contiguous loads instead of one, the nibble plane and the
            # slice of the high plane that covers the same values. A thread
            # stages `MMA_KPT` values, which is sixteen, so that slice is two
            # bytes at five bits and four at six and both are a load of a width
            # the hardware has.
            comptime hbits = quant_high_bits(form)
            comptime hshift = high_shift[form]()
            comptime hper = 1 << hshift
            comptime hmask = UInt32((1 << hbits) - 1)
            var q = w.unsafe_offset(row + ((k0 + wk) >> 1)).unsafe_load[
                width=MMA_KPT // 2
            ]() if live_row else SIMD[DType.uint8, MMA_KPT // 2](0)
            var hq = w.unsafe_offset(
                row + (cols >> 1) + ((k0 + wk) >> hshift)
            ).unsafe_load[width=MMA_KPT >> hshift]() if live_row else SIMD[
                DType.uint8, MMA_KPT >> hshift
            ](
                0
            )
            comptime for j in range(MMA_KPT // 2):
                var b = UInt32(q[j])
                var h = UInt32(hq[(2 * j) >> hshift])
                var s0 = UInt32(((2 * j) & (hper - 1)) * hbits)
                w_tile[unsafe_offset=(wk + 2 * j) * MMA_ROWS + wr] = (
                    d * wide_float[form]((b & 0xF) | (((h >> s0) & hmask) << 4))
                    + m
                )
                w_tile[unsafe_offset=(wk + 2 * j + 1) * MMA_ROWS + wr] = (
                    d
                    * wide_float[form](
                        (b >> 4) | (((h >> (s0 + UInt32(hbits))) & hmask) << 4)
                    )
                    + m
                )
        barrier()

        comptime for kk in range(0, MMA_K, 8):
            var af = InlineArray[SIMD[DType.float32, 2], MMA_FT](fill=0)
            comptime for i in range(MMA_FT):
                af[i] = apple_mma_load_8x8[DType.float32](
                    x_tile.unsafe_offset((sg_t + i * 8) * MMA_K + kk), MMA_K
                )
            comptime for j in range(MMA_FR):
                var bf = apple_mma_load_8x8[DType.float32](
                    w_tile.unsafe_offset(kk * MMA_ROWS + sg_r + j * 8),
                    MMA_ROWS,
                )
                comptime for i in range(MMA_FT):
                    var got = SIMD[DType.float32, 2](0)
                    _mma_apple_8x8(got, af[i], bf, acc[i * MMA_FR + j])
                    acc[i * MMA_FR + j] = got
        barrier()

    comptime for i in range(MMA_FT):
        comptime for j in range(MMA_FR):
            apple_mma_store_8x8[DType.float32](
                o_tile.unsafe_offset((sg_t + i * 8) * MMA_ROWS + sg_r + j * 8),
                MMA_ROWS,
                acc[i * MMA_FR + j],
            )
    barrier()

    var epi = Int(epi_dev)
    for idx in range(tid, MMA_TOKENS * MMA_ROWS, MMA_THREADS):
        var t = idx // MMA_ROWS
        var r = idx % MMA_ROWS
        if t0 + t >= tokens or r0 + r >= rows:
            continue
        write_epilogue(
            o,
            aux,
            epi,
            (t0 + t) * rows + r0 + r,
            r0 + r,
            o_tile[unsafe_offset=idx],
        )


def _launch_mma[
    group: Int, with_min: Bool, form: Int
](
    ctx: DeviceContext,
    w: Tensor,
    x: Pointer[Float32, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    aux: Pointer[Float32, MutAnyOrigin],
    epi: Int,
    tokens: Int,
) raises:
    """One instantiation of the matrix core form, launched."""
    ctx.enqueue_function[planar_mma_kernel[group, with_min, form]](
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=w.device_address()),
        x,
        o,
        aux,
        Int32(w.cols),
        Int32(w.row_bytes()),
        Int32(epi),
        Int32(w.rows),
        Int32(tokens),
        grid_dim=(
            (tokens + MMA_TOKENS - 1) // MMA_TOKENS,
            (w.rows + MMA_ROWS - 1) // MMA_ROWS,
            1,
        ),
        block_dim=(MMA_THREADS, 1, 1),
    )


def _launch_mm[
    tile: Int, group: Int, with_min: Bool, form: Int
](
    ctx: DeviceContext,
    w: Tensor,
    x: Pointer[Float32, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    aux: Pointer[Float32, MutAnyOrigin],
    epi: Int,
    tokens: Int,
) raises:
    """One instantiation of the batched form, launched."""
    ctx.enqueue_function[planar_matmul_kernel[tile, group, with_min, form]](
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=w.device_address()),
        x,
        o,
        aux,
        Int32(w.cols),
        Int32(w.row_bytes()),
        Int32(epi),
        Int32(w.rows),
        Int32(tokens),
        grid_dim=(
            (tokens + SPAN * MM_GROUPS - 1) // (SPAN * MM_GROUPS),
            w.rows,
            1,
        ),
        block_dim=(tile, MM_GROUPS, 1),
    )


def device_matmul_into(
    ctx: DeviceContext,
    w: Tensor,
    x: DeviceVec,
    mut out: DeviceVec,
    tokens: Int,
    at: Int = 0,
    epi: Int = EPI_NONE,
    aux: Optional[Pointer[Float32, MutAnyOrigin]] = None,
) raises:
    """`out[at + t * w.rows + r] = dot(row r of w, x[t])`, for `tokens` of them.

    The batched form of `device_matvec_into`, with the same dispatch over the
    same five instantiations and the same epilogue. `x` holds `tokens` rows of
    `w.cols` laid out one after another and `out` holds `tokens` rows of
    `w.rows` the same way, which is what makes a key projection write a run of
    cache slots with no copy: the slots are contiguous and so are the rows.

    Queued and not synchronized, for the reason the single token form gives.
    """
    if tokens <= 0:
        raise Error("a batched matmul needs at least one token")
    if at < 0:
        raise Error("a matmul cannot write at a negative offset")
    if x.elements() < tokens * w.cols:
        raise Error(
            "the device matmul wants "
            + String(tokens)
            + " rows of "
            + String(w.cols)
            + " and the input holds "
            + String(x.elements())
        )
    if out.elements() < at + tokens * w.rows:
        raise Error(
            "the device matmul writes "
            + String(tokens)
            + " rows of "
            + String(w.rows)
            + " at offset "
            + String(at)
            + " and the output ends at "
            + String(out.elements())
        )
    check_matvec_shapes(w, w.cols, w.rows)
    comptime if not has_accelerator():
        raise Error(
            "this build has no device code in it, so there is no device matmul"
            " to run. Accelerator support is decided when molla is compiled,"
            " not when it is run"
        )
    else:
        var o = out.ptr_at(at)
        var a = aux.value() if aux and epi != EPI_NONE else o
        var p = x.ptr()
        var g = group_size(w.kind)
        var carries_min = has_min(w.kind)
        var form = quant_form(w.kind)
        # The matrix core form first where it exists, which today is Apple only.
        # `TensorCore` in MAX has no integer case, so the widest thing a 4090
        # offers here is the half precision tensor core, and #212 has the
        # measurement that says a tile built on it the same way this one is
        # loses to the ordinary kernel on two models out of three.
        #
        # A group of sixteen stages two groups to a step and is left on the
        # ordinary form until there is a model that wants it.
        comptime if CompilationTarget.is_macos():
            if g == 32:
                if form == QUANT_U4 and carries_min:
                    _launch_mma[32, True, QUANT_U4](
                        ctx, w, p, o, a, epi, tokens
                    )
                    return
                if form == QUANT_S4 and not carries_min:
                    _launch_mma[32, False, QUANT_S4](
                        ctx, w, p, o, a, epi, tokens
                    )
                    return
                if form == QUANT_U5 and carries_min:
                    _launch_mma[32, True, QUANT_U5](
                        ctx, w, p, o, a, epi, tokens
                    )
                    return
                if form == QUANT_S5 and not carries_min:
                    _launch_mma[32, False, QUANT_S5](
                        ctx, w, p, o, a, epi, tokens
                    )
                    return
                if form == QUANT_I8 and not carries_min:
                    _launch_mma[32, False, QUANT_I8](
                        ctx, w, p, o, a, epi, tokens
                    )
                    return
        if form == QUANT_U4 and g == 32 and carries_min:
            _launch_mm[MM_TILE, 32, True, QUANT_U4](
                ctx, w, p, o, a, epi, tokens
            )
        elif form == QUANT_S4 and g == 32 and not carries_min:
            _launch_mm[MM_TILE, 32, False, QUANT_S4](
                ctx, w, p, o, a, epi, tokens
            )
        elif form == QUANT_U5 and g == 32 and carries_min:
            _launch_mm[MM_TILE, 32, True, QUANT_U5](
                ctx, w, p, o, a, epi, tokens
            )
        elif form == QUANT_S5 and g == 32 and not carries_min:
            _launch_mm[MM_TILE, 32, False, QUANT_S5](
                ctx, w, p, o, a, epi, tokens
            )
        elif form == QUANT_S6 and g == 16 and not carries_min:
            _launch_mm[MM_TILE, 16, False, QUANT_S6](
                ctx, w, p, o, a, epi, tokens
            )
        elif form == QUANT_I8 and g == 32 and not carries_min:
            _launch_mm[MM_TILE, 32, False, QUANT_I8](
                ctx, w, p, o, a, epi, tokens
            )
        else:
            raise Error(
                "no device matmul is compiled for quant form "
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
