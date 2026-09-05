"""A whole decoder layer in one kernel launch, driven by a table in memory.

molla launches a kernel for every step of every layer. A Llama shaped layer is
twelve of them after #168 and #194 folded the biases, the residual adds and the
gate into the projections that were already running, so a thirty layer model is
363 launches for one decoded token. A launch is 4.82 microseconds on a 4090 and
20.10 on an M4, which is 1.75 ms and 7.30 ms of pure submission per token and a
ceiling of 571 tokens a second and 137 tokens a second before a single multiply
happens. That ceiling is the binding constraint on the models this milestone is
measured on, and no arrangement of kernels per layer clears it.

This is stage one of the answer, which is one kernel a layer. The design and the
measurements it rests on are
[docs/validation/fused.md](../../../docs/validation/fused.md) and the grid wide
sync section of [docs/validation/max.md](../../../docs/validation/max.md), and
the three findings that shape everything here are worth restating because the
code looks arbitrary without them.

There is no grid wide barrier in MAX and no cooperative launch. The one built
here is a sense reversing barrier over two device scope atomic words, and on
Metal it needs a threadgroup barrier asked for with the device memory flag on
either side of it because that target has no fence and rejects every ordering on
an atomic.

Ordinary global loads and stores are not coherent across a block on Metal, and
both sides need a relaxed device scope atomic. #170's probe said the store side
was enough on its own and this kernel does not reproduce that, for a reason the
probe could not have seen: there every block read an address it had never read
before, and here a block reads the same norm scratch at every record of every
layer, so the second read hits a line its own L1 already holds and the barrier
does not evict it. `_put` and `_get` are the two ends of that and neither is
optional.

A barrier round is not free on CUDA at full residency, 4.59 microseconds over
1536 blocks against a 4.82 microsecond launch, and it falls to 0.75 at 96 blocks
and is flat below that. So the grid is a few hundred blocks and each block walks
many output rows, which is `_grid_for` and is the part of this design most
likely to be wrong: a bandwidth bound matvec may want more threads in flight
than a few hundred blocks provide. That is measured rather than assumed.

## Why the steps are data

The kernel cannot call the host between layers, so everything the host passes as
an argument today has to be in device memory before the launch. That is one
table of fixed width records, built once when a model binds and walked by the
kernel, and it is built for the whole model rather than for a layer even though
stage one launches one layer at a time. Stage two is the same kernel over a
wider range of the same table, so writing a layer's steps out by hand in the
kernel would have been throwaway work and a second thing to keep correct.

What is not in a record is anything that changes between tokens. The position,
the cache slot and the number of keys are the same for every record in a pass,
so they are kernel arguments, and the table is never touched again after a model
loads.

A record names its operands as a space and an offset rather than as a pointer.
That is not a preference: a pointer rebuilt from an integer address loses its
address space on Metal and crashes the pipeline compiler the moment an atomic
touches it, which is exactly what `_put` does. So the five bases arrive as
kernel arguments and a record says which of them and how far in.

## What it agrees with

The same arithmetic in the same order as the kernels it replaces, so the answers
are identical in every digit rather than close. `tests/test_gpu_block.mojo`
checks a fused layer against the unfused one on a synthetic model, over the
logits, over both cache planes and over the residual stream a layer at a time,
and anything that is merely close there is a bug rather than a tolerance.
"""

from std.atomic import Atomic, Ordering
from std.gpu import block_idx, grid_dim, thread_idx
from std.gpu.primitives.warp import lane_group_max, lane_group_sum
from std.math import cos, exp, sin, sqrt
from std.memory import AddressSpace, bitcast, stack_allocation
from std.os.env import getenv
from std.sys import llvm_intrinsic
from std.sys.info import CompilationTarget, has_accelerator

from max.gpu import barrier
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext

from molla.nn.attention import AttnSpec
from molla.nn.block import ACT_GELU, ACT_SILU, BlockSpec
from molla.nn.gpu import (
    ACT_BIT,
    ALANES,
    EPI_ADD,
    EPI_BIAS,
    EPI_GLU,
    EPI_HALF,
    EPI_NONE,
    TILE,
    DeviceHalf,
    DeviceVec,
    activate,
    byte_float,
    coherent_load,
    coherent_load_half,
    key_dot,
    nibble_float,
    planar_partial_dot,
    planar_row_sum,
    row_takes_a_warp,
)
from molla.nn.gpu_ops import NEG_INF, _ramp, _reduce_angle, _tanh
from molla.nn.repack import (
    QUANT_I8,
    QUANT_K4,
    QUANT_K5,
    QUANT_S4,
    QUANT_S5,
    QUANT_S6,
    QUANT_U4,
    QUANT_U5,
    group_shift,
    group_size,
    has_min,
    quant_form,
)
from molla.nn.rope import RopeSpec, corr_range
from molla.nn.tensor import Tensor

comptime FTILE = TILE
"""Threads a block, which is the matvec's tile and for the matvec's reason.

The same number as the unfused kernels so that a fused layer and an unfused one
reduce over the same width in the same order and their answers agree bit for
bit. It is not a free parameter here even though it looks like one.
"""

comptime FSPLIT_MAX = 32
"""Most blocks that may share one head's keys in the attention step.

A head is the only unit of work attention has if a block has to own a whole
row, and models have between nine and thirty two of them, so a grid of a few
hundred blocks left almost all of itself idle for the one step whose cost grows
with the context. That is what made the fused path lose nine per cent on Qwen
at a 512 token prompt while winning fourteen at eight tokens.

So a head is cut into slices and a block takes one, in the flash attention way:
a slice reduces over its own keys with its own maximum, writes the running
total, the maximum and the weighted values, and a second pass over the head
folds the slices together. The bound is here because the partials are a work
vector field sized at load time, and thirty two slices is enough that every
model in the fleet fills the grid: nine heads become 288 blocks, fourteen 384
and thirty two the whole 1024 a large model would want.
"""

comptime PATIENCE = 100000000
"""Spins at the barrier before a block decides the rendezvous is broken.

A bound rather than forever, because forever is a hung GPU and on a laptop that
is the display. Nothing that works comes anywhere near it, so reaching it is a
grid that was not resident, which `fused_selftest` is meant to catch before a
model ever runs.
"""

comptime dev32 = Atomic[DType.int32, scope="device"]

comptime RMW = (
    Ordering.RELAXED if CompilationTarget.is_macos() else Ordering.ACQUIRE_RELEASE
)
comptime GET = (
    Ordering.RELAXED if CompilationTarget.is_macos() else Ordering.ACQUIRE
)
comptime PUT = (
    Ordering.RELAXED if CompilationTarget.is_macos() else Ordering.RELEASE
)
"""What carries the ordering at the rendezvous, per backend.

On CUDA it rides the atomic. An Apple GPU rejects `acquire`, `release` and
`acquire_release` on an atomic and crashes the Metal pipeline compiler on a
`fence` of every ordering, scoped or not, so there the atomics are relaxed and
`flush` carries the ordering instead.
"""


def flush():
    """A block barrier that also orders this block's device memory.

    On Apple this is `threadgroup_barrier(mem_device | mem_threadgroup)`, which
    is the instruction `barrier` already emits with flag 3 rather than flag 2,
    and it is the only way to order device memory on that target from here.
    Everywhere else the ordering is on the atomics and this is `barrier`.
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

    Every block adds one to a counter, the block that finds itself last resets
    the counter and moves the generation on, and everyone else waits for the
    generation to move. The counter is reset before the generation moves so that
    the next round starts from zero, which is safe because nobody is past the
    generation yet.

    One thread does the waiting and the other `FTILE - 1` wait for it at the
    `flush` on the way out, which is what makes this cost one atomic a block
    rather than one a thread.
    """
    flush()
    var ok = True
    if thread_idx.x == 0:
        var was = dev32.fetch_add[ordering=RMW](count, 1)
        if was == Int32(blocks - 1):
            dev32.store[ordering=Ordering.RELAXED](count, Int32(0))
            dev32.store[ordering=PUT](gen, seen + 1)
        else:
            var spins = 0
            while dev32.load[ordering=GET](gen) == seen:
                spins += 1
                if spins > PATIENCE:
                    ok = False
                    break
    flush()
    seen += 1
    return ok


@always_inline
def _put(p: Pointer[Float32, MutAnyOrigin], i: Int, v: Float32):
    """Write a float another block is going to read after a barrier.

    An ordinary store everywhere except Metal, where a plain store is invisible
    to the rest of the grid: the probe in `scripts/gridsync_probe.mojo` reads
    stale on 634 of 640 crossings and the barrier makes no difference, because
    the problem is not ordering but that the value never leaves the core that
    wrote it. A relaxed device scope atomic store fixes that end of it. `_get`
    is the other end and both are needed.

    `Pointer(to=...)` and not a pointer rebuilt from an integer address. The
    second form loses the address space and crashes the Metal pipeline compiler
    the moment an atomic touches it, which is also why a record names a base and
    an offset rather than carrying a pointer.
    """
    comptime if CompilationTarget.is_macos():
        var q = p.unsafe_bitcast[Int32]()
        dev32.store[ordering=Ordering.RELAXED](
            Pointer[Int32, MutAnyOrigin](to=q[unsafe_offset=i]),
            bitcast[DType.int32, 1](v),
        )
    else:
        p[unsafe_offset=i] = v


@always_inline
def _get(p: Pointer[Float32, MutAnyOrigin], i: Int) -> Float32:
    """Read a float another block may have written since the last barrier.

    `coherent_load` in `molla.nn.gpu` is the whole of it, and the reason it
    lives there rather than here is that the matvec loop is shared with the
    unfused path and takes this as a parameter. What it costs is the thing the
    design most wanted to avoid, because it is one instruction for every column
    of every row rather than one a row. See
    [docs/validation/fused.md](../../../docs/validation/fused.md).
    """
    return coherent_load[True](p, i)


@always_inline
def _put_pair(
    p: Pointer[Float32, MutAnyOrigin], j: Int, a: Float32, b: Float32
):
    """Write two adjacent halves of the cache as one word.

    `_put` needs a thirty two bit atomic to get a value off the core that wrote
    it, and the cache is made of sixteen bit values, so the unit a writer owns
    is the aligned pair and not the element. `j` is the index of that pair: the
    elements it covers are `2 * j` and `2 * j + 1`, `a` goes in the low one and
    `b` in the high one.

    Where `PAIRED` is true every writer of the cache in this kernel therefore
    owns whole pairs. The projections do it by taking rows two at a time and the
    rotations by owning a pair of dimensions rather than a pair of angles, and
    both are free to do so because a cache row is `kv_width` elements and every
    model molla accepts has an even one. Two threads owning the two halves of a
    word would be a read modify write on a value another core is holding, and
    there is no sixteen bit atomic on either backend to make that safe.
    """
    var word = bitcast[DType.int32, 1](
        SIMD[DType.float16, 2](Float16(a), Float16(b))
    )
    var q = p.unsafe_bitcast[Int32]()
    comptime if CompilationTarget.is_macos():
        dev32.store[ordering=Ordering.RELAXED](
            Pointer[Int32, MutAnyOrigin](to=q[unsafe_offset=j]), word
        )
    else:
        q[unsafe_offset=j] = word


comptime PAIRED = CompilationTarget.is_macos()
"""Whether a writer of the cache inside this kernel has to own an aligned pair.

Metal, and only Metal. `_put` needs a thirty two bit atomic to get a value off
the core that wrote it and a half is sixteen wide, so there the unit a writer
owns is the pair. Everywhere else a plain store is already visible to the rest
of the grid and a writer owns one element.

The difference is worth having rather than writing one path for both. Taking
rows two at a time gives half the warps nothing to do and the other half twice
as much, and the record is one grid wide step either way, so it costs the step
its whole second half. On an 8B on CUDA that measured 3.8 percent of a decode
for nothing at all.
"""


@always_inline
def _put_half(p: Pointer[Float32, MutAnyOrigin], i: Int, v: Float32):
    """Write element `i` of the cache, narrowed on the way in.

    The counterpart of `_put_pair` for a backend where the pair is not needed,
    which is what `PAIRED` decides. There is no atomic here because there is no
    backend that both needs one and has one this wide.
    """
    p.unsafe_bitcast[Float16]()[unsafe_offset=i] = Float16(v)


@always_inline
def _get_half(p: Pointer[Float32, MutAnyOrigin], i: Int) -> Float32:
    """Read element `i` of the cache, widened.

    A read has no pairing problem, so this is per element even though the write
    beside it is per pair. `coherent_load_half` in `molla.nn.gpu` is the whole of
    it and it loads the pair anyway, because the atomic it needs is thirty two
    bits wide whichever half is wanted.
    """
    return coherent_load_half[True](p.unsafe_bitcast[Float16](), i)


comptime SPACE_WORK = 0
"""The layer's intermediates: the norm, the query, the heads output, the
projection scratch, the gate, the up and the attention scores."""
comptime SPACE_RESID = 1
"""The residual stream, which is the one buffer a layer both starts from and
writes back into."""
comptime SPACE_KEYS = 2
comptime SPACE_VALS = 3
comptime SPACE_ARENA = 4
"""The small per layer weights, uploaded once: the norm gains, the projection
biases and the rope tables, concatenated so that a record can name one of them
as an offset."""

comptime OP_NORM = 0
comptime OP_MATVEC = 1
comptime OP_ROPE = 2
comptime OP_ATTEND = 3
comptime OP_ACT = 4
comptime OP_ADD = 5

comptime QK_U4 = 0
"""Unsigned nibbles, groups of 32, with a minimum plane. q4_1."""
comptime QK_S4 = 1
"""Centred nibbles, groups of 32, no minimum. q4_0."""
comptime QK_U5 = 2
"""Five bits in two planes, groups of 32, with a minimum. q5_1."""
comptime QK_S5 = 3
"""Centred five bits in two planes, groups of 32, no minimum. q5_0."""
comptime QK_S6 = 4
"""Centred six bits in two planes, groups of 16, block scaled. q6_K."""
comptime QK_I8 = 5
"""Bytes, groups of 32, no minimum. q8_0."""
comptime QK_K4 = 6
"""`QK_U4` with block scaled scale planes. q4_K."""
comptime QK_K5 = 7
"""`QK_U5` with block scaled scale planes. q5_K."""

comptime NEOX_BIT = 1
"""Set in a rope record's kind when the pairing is neox."""
comptime FACTOR_BIT = 2
"""Set when the rope reads a per pair frequency factor."""

comptime R_OP = 0
comptime R_SYNC = 1
"""Whether a grid barrier follows this record.

A property of the boundary rather than of the record, which is what lets the
three attention projections stay three records with one barrier after the third
and the up and the gate stay two records with one barrier after the gate.
Merging them into one record instead would have needed a record shape that
holds three weight matrices.
"""
comptime R_XS = 2
comptime R_XO = 3
comptime R_XK = 4
"""What to multiply the cache slot by before adding it to the input offset.

Zero for everything that does not move with the slot, and the key width for the
records that read or write this position's place in the cache. The slot is a
kernel argument because it changes every token and the table does not.
"""
comptime R_OS = 5
comptime R_OO = 6
comptime R_OK = 7
comptime R_AS = 8
comptime R_AO = 9
comptime R_AK = 10
comptime R_G = 11
"""Where in the arena this record's gain or rope step table starts."""
comptime R_H = 12
"""Where in the arena the rope frequency factors start."""
comptime R_W = 13
"""Where in the weight pool this record's matrix starts, in bytes."""
comptime R_COLS = 14
comptime R_ROWS = 15
comptime R_STRIDE = 16
comptime R_EPI = 17
comptime R_KIND = 18
comptime R_HEADS = 19
comptime R_HEAD_DIM = 20
comptime R_DIM = 21
comptime R_GROUP = 22
comptime R_WINDOW = 23
comptime R_SINKS = 24
comptime R_ROW = 25
comptime R_KV = 26
comptime R_N = 27
comptime R_RUNS = 28
comptime R_SPLIT = 29
"""Where in the work vector the attention partials start.

One row of `head_dim + 2` floats a head a slice, which is what lets more than
one block work on the same head. See `FSPLIT_MAX`.
"""
comptime REC_INTS = 32
"""Fields in a record, rounded up so that a record is a shift rather than a
multiply. Thirty are used and the table is a few tens of kilobytes for the
largest model in the fleet, so the rounding costs nothing worth counting."""

comptime R_EPS = 0
comptime R_SCALE = 1
comptime R_SOFTCAP = 2
comptime R_EXT = 3
comptime R_ATTN = 4
comptime R_LOW = 5
comptime R_HIGH = 6
comptime REC_FLOATS = 8


@always_inline
def _fi(plan: Pointer[Int64, MutAnyOrigin], rec: Int, field: Int) -> Int:
    return Int(plan[unsafe_offset=rec * REC_INTS + field])


@always_inline
def _ff(plan: Pointer[Float32, MutAnyOrigin], rec: Int, field: Int) -> Float32:
    return plan[unsafe_offset=rec * REC_FLOATS + field]


@always_inline
def _base(
    space: Int,
    off: Int,
    work: Pointer[Float32, MutAnyOrigin],
    resid: Pointer[Float32, MutAnyOrigin],
    keys: Pointer[Float16, MutAnyOrigin],
    values: Pointer[Float16, MutAnyOrigin],
    arena: Pointer[Float32, MutAnyOrigin],
) -> Pointer[Float32, MutAnyOrigin]:
    """One operand of a record, resolved to an address.

    A uniform branch over five bases, taken once a record rather than once an
    element, and it is a branch rather than an array of pointers because an
    array would have to live in device memory and a pointer loaded from device
    memory is the form that loses its address space on Metal.
    """
    if space == SPACE_RESID:
        return Pointer[Float32, MutAnyOrigin](to=resid[unsafe_offset=off])
    if space == SPACE_KEYS:
        # The cache is float16, so this is the address of element `off` of it
        # and not of a float32 vector, and every record that names one of these
        # two spaces knows that: a projection stores through an `EPI_HALF`
        # epilogue and a rotation through `_get_half` and `_put_half`. See
        # `DeviceHalf`.
        #
        # `unsafe_bitcast` and not a pointer rebuilt from an integer address,
        # for the reason `_put` gives: the second form loses the address space
        # and the first keeps it, and these two spaces are the ones an atomic
        # store lands on.
        return Pointer(to=keys[unsafe_offset=off]).unsafe_bitcast[Float32]()
    if space == SPACE_VALS:
        return Pointer(to=values[unsafe_offset=off]).unsafe_bitcast[Float32]()
    if space == SPACE_ARENA:
        return Pointer[Float32, MutAnyOrigin](to=arena[unsafe_offset=off])
    return Pointer[Float32, MutAnyOrigin](to=work[unsafe_offset=off])


@always_inline
def _tree_sum(value: Float32) -> Float32:
    """Sum one value a thread across the block, answer in every thread.

    The matvec's reduction and `gpu_ops`' `_block_sum` are the same tree and are
    written out again here rather than called, because both of those allocate
    their own shared array and a fused kernel that called all three would hold
    three of them for the whole launch. One allocation, reached from every
    record.
    """
    var part = stack_allocation[
        FTILE, Float32, address_space=AddressSpace.SHARED
    ]()
    var t = Int(thread_idx.x)
    part[unsafe_offset=t] = value
    barrier()
    var step = FTILE // 2
    while step > 0:
        if t < step:
            part[unsafe_offset=t] = (
                part[unsafe_offset=t] + part[unsafe_offset=t + step]
            )
        barrier()
        step //= 2
    var total = part[unsafe_offset=0]
    # Every thread reads slot zero before anything is allowed to overwrite it,
    # which matters because these run back to back inside one kernel and the
    # allocation is the same shared memory each time.
    barrier()
    return total


@always_inline
def _tree_max(value: Float32) -> Float32:
    """The largest value in the block, in every thread."""
    var part = stack_allocation[
        FTILE, Float32, address_space=AddressSpace.SHARED
    ]()
    var t = Int(thread_idx.x)
    part[unsafe_offset=t] = value
    barrier()
    var step = FTILE // 2
    while step > 0:
        if t < step:
            var other = part[unsafe_offset=t + step]
            if other > part[unsafe_offset=t]:
                part[unsafe_offset=t] = other
        barrier()
        step //= 2
    var top = part[unsafe_offset=0]
    barrier()
    return top


@always_inline
def _rope_turn(
    steps: Pointer[Float32, MutAnyOrigin],
    factors: Pointer[Float32, MutAnyOrigin],
    kind: Int,
    pos: Int,
    pair: Int,
    scale: Float32,
    ext: Float32,
    attn: Float32,
    low: Float32,
    high: Float32,
) -> SIMD[DType.float32, 2]:
    """The cosine and the sine of one rotation, with the attention factor in.

    The angle half of `device_rope`, lifted out of the record because the record
    now has two loops that want it and the arithmetic has to be the same in both
    to the digit. The returned pair is cosine first.
    """
    var freq = Float32(1.0)
    if kind & FACTOR_BIT != 0:
        freq = factors[unsafe_offset=pair]
    var step = steps[unsafe_offset=pair]
    var extrap = Float32(pos) * step / freq
    var interp = scale * extrap
    var theta = interp
    if ext != 0:
        var mix = _ramp(low, high, pair) * ext
        theta = interp * (Float32(1.0) - mix) + extrap * mix
    var turn = _reduce_angle(theta)
    return SIMD[DType.float32, 2](cos(turn) * attn, sin(turn) * attn)


@always_inline
def _epi_value(
    aux: Pointer[Float32, MutAnyOrigin],
    r: Int,
    epi: Int,
    total: Float32,
) -> Float32:
    """What a row is worth once it is reduced, before it is stored.

    The first two cases of the unfused matvec's epilogue in the same order: a
    bias, or a gated activation against the up projection. `aux` is the output
    pointer when nothing reads it, so there is no null to test for. Split from
    the store because a row bound for the cache is stored two at a time and this
    part of it is still one at a time.
    """
    var v = total
    if epi & EPI_BIAS != 0:
        v += aux[unsafe_offset=r]
    elif epi & EPI_GLU != 0:
        if epi & ACT_BIT != 0:
            v = activate[ACT_GELU](v) * _get(aux, r)
        else:
            v = activate[ACT_SILU](v) * _get(aux, r)
    return v


@always_inline
def _epilogue(
    o: Pointer[Float32, MutAnyOrigin],
    aux: Pointer[Float32, MutAnyOrigin],
    r: Int,
    epi: Int,
    total: Float32,
):
    """What happens to a row once it is reduced, run by one thread.

    The same four cases in the same order as the unfused matvec: a bias, or a
    gated activation against the up projection, and then a narrowing store to
    the cache, a residual add, or a plain store.

    The cache case is reached only where `PAIRED` is false. Where it is true a
    row bound for the cache never comes here, because it is stored with its
    neighbour by `_put_pair` and this writes one row at a time.
    """
    var v = _epi_value(aux, r, epi, total)
    if epi & EPI_HALF != 0:
        _put_half(o, r, v)
    elif epi & EPI_ADD != 0:
        _put(o, r, _get(o, r) + v)
    else:
        _put(o, r, v)


@always_inline
def _do_matvec[
    group: Int, with_min: Bool, form: Int
](
    packed: Pointer[UInt8, MutAnyOrigin],
    x: Pointer[Float32, MutAnyOrigin],
    o: Pointer[Float32, MutAnyOrigin],
    aux: Pointer[Float32, MutAnyOrigin],
    w_at: Int,
    cols: Int,
    stride: Int,
    rows: Int,
    epi: Int,
    b: Int,
    blocks: Int,
):
    """Every row this block owns, strided across the grid.

    A narrow row goes to a warp and a wide one to the whole block, which is
    `row_takes_a_warp`, and the unfused matvec asks the same question of the same
    weight so that the two keep adding a row up in the same order. `cols` is a
    property of the weight, so a record is all of one or all of the other and the
    barriers in the block half are still reached by all of a block or by none.

    Either way the rows are walked in a loop, because the grid is sized to
    synchronise cheaply rather than to the widest step: a 4096 row projection on
    96 blocks is 43 rows a block. The stride is the whole grid so that the
    workers on adjacent rows at any moment are adjacent, which is what keeps the
    weight reads of one round of the loop in one region of the pool.

    A projection whose output is the cache walks the same rows in the same
    order and, where `PAIRED` says it must, takes them two at a time, because a
    half is written as the aligned pair it sits in and `_put_pair` says why. The
    reduction of a row is untouched by that: which warp or which block adds a row
    up does not change the order the columns come in, so the value is the same
    one the unfused matvec gets and `tests/test_gpu_fused.mojo` still holds the
    two to every digit.
    """
    var t = Int(thread_idx.x)
    var half = PAIRED and epi & EPI_HALF != 0
    if row_takes_a_warp(cols):
        var lane = t % ALANES
        var warps = FTILE // ALANES
        var w = b * warps + t // ALANES
        if half:
            var pairs = rows // 2
            var pi = w
            while pi < pairs:
                var lo = planar_row_sum[group, with_min, form, coherent=True](
                    packed, x, w_at + (pi * 2) * stride, cols, lane
                )
                var hi = planar_row_sum[group, with_min, form, coherent=True](
                    packed, x, w_at + (pi * 2 + 1) * stride, cols, lane
                )
                if lane == 0:
                    _put_pair(
                        o,
                        pi,
                        _epi_value(aux, pi * 2, epi, lo),
                        _epi_value(aux, pi * 2 + 1, epi, hi),
                    )
                pi += blocks * warps
            return

        var rw = w
        while rw < rows:
            var total = planar_row_sum[group, with_min, form, coherent=True](
                packed, x, w_at + rw * stride, cols, lane
            )
            if lane == 0:
                _epilogue(o, aux, rw, epi, total)
            rw += blocks * warps
        return

    if half:
        var pairs = rows // 2
        var pi = b
        while pi < pairs:
            var alo = planar_partial_dot[
                FTILE, group, with_min, form, coherent=True
            ](packed, x, w_at + (pi * 2) * stride, cols, t)
            var lo = _tree_sum(alo)
            var ahi = planar_partial_dot[
                FTILE, group, with_min, form, coherent=True
            ](packed, x, w_at + (pi * 2 + 1) * stride, cols, t)
            var hi = _tree_sum(ahi)
            if t == 0:
                _put_pair(
                    o,
                    pi,
                    _epi_value(aux, pi * 2, epi, lo),
                    _epi_value(aux, pi * 2 + 1, epi, hi),
                )
            pi += blocks
        return

    var r = b
    while r < rows:
        var acc = planar_partial_dot[
            FTILE, group, with_min, form, coherent=True
        ](packed, x, w_at + r * stride, cols, t)
        var total = _tree_sum(acc)
        if t == 0:
            _epilogue(o, aux, r, epi, total)
        r += blocks


def fused_kernel(
    plan_i: Pointer[Int64, MutAnyOrigin],
    plan_f: Pointer[Float32, MutAnyOrigin],
    pool: Pointer[UInt8, MutAnyOrigin],
    arena: Pointer[Float32, MutAnyOrigin],
    work: Pointer[Float32, MutAnyOrigin],
    resid: Pointer[Float32, MutAnyOrigin],
    keys: Pointer[Float16, MutAnyOrigin],
    values: Pointer[Float16, MutAnyOrigin],
    sync: Pointer[Int32, MutAnyOrigin],
    first_dev: Int32,
    last_dev: Int32,
    blocks_dev: Int32,
    pos_dev: Int32,
    slot_dev: Int32,
    count_dev: Int32,
):
    """Records `first` through `last - 1` of the plan, in order, in one launch.

    `sync` is three words: the barrier's arrival count, its generation, and a
    flag a block sets if it ever gives up waiting. The first two are left in a
    state the next launch can start from, which is what the store of zero at the
    end is for: every block enters with a generation of zero, and a block still
    spinning when that store lands sees a value that is not the one it is
    waiting on and leaves, which is the exit it wanted.

    `pos`, `slot` and `count` are the three things that change between tokens
    and are therefore arguments rather than fields. Everything else a step needs
    is in the record.
    """
    var blocks = Int(blocks_dev)
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var pos = Int(pos_dev)
    var slot = Int(slot_dev)
    var count = Int(count_dev)
    var seen = Int32(0)
    var stuck = False

    var counter = Pointer[Int32, MutAnyOrigin](to=sync[unsafe_offset=0])
    var gen = Pointer[Int32, MutAnyOrigin](to=sync[unsafe_offset=1])

    for rec in range(Int(first_dev), Int(last_dev)):
        var op = _fi(plan_i, rec, R_OP)
        var x = _base(
            _fi(plan_i, rec, R_XS),
            _fi(plan_i, rec, R_XO) + slot * _fi(plan_i, rec, R_XK),
            work,
            resid,
            keys,
            values,
            arena,
        )
        var o = _base(
            _fi(plan_i, rec, R_OS),
            _fi(plan_i, rec, R_OO) + slot * _fi(plan_i, rec, R_OK),
            work,
            resid,
            keys,
            values,
            arena,
        )
        var aux = _base(
            _fi(plan_i, rec, R_AS),
            _fi(plan_i, rec, R_AO) + slot * _fi(plan_i, rec, R_AK),
            work,
            resid,
            keys,
            values,
            arena,
        )

        if op == OP_NORM:
            var n = _fi(plan_i, rec, R_N)
            var runs = _fi(plan_i, rec, R_RUNS)
            var gain = Pointer[Float32, MutAnyOrigin](
                to=arena[unsafe_offset=_fi(plan_i, rec, R_G)]
            )
            var eps = _ff(plan_f, rec, R_EPS)

            # A per head key norm runs in place on the cache, so this record has
            # the same two widths the rotation below has, chosen the same way
            # and uniformly across the grid. The sum is per element either way,
            # since reading a half needs no ownership of anything.
            var half = _fi(plan_i, rec, R_XS) == SPACE_KEYS
            var run = b
            while run < runs:
                var at = run * n
                var acc = Float32(0)
                var i = t
                while i < n:
                    var v = _get_half(x, at + i) if half else _get(x, at + i)
                    acc += v * v
                    i += FTILE
                var total = _tree_sum(acc)
                var scale = Float32(1.0) / sqrt(total / Float32(n) + eps)
                if half and PAIRED:
                    var wj = t
                    while wj * 2 + 1 < n:
                        var e = at + wj * 2
                        _put_pair(
                            o,
                            e >> 1,
                            _get_half(x, e)
                            * scale
                            * gain[unsafe_offset=wj * 2],
                            _get_half(x, e + 1)
                            * scale
                            * gain[unsafe_offset=wj * 2 + 1],
                        )
                        wj += FTILE
                    run += blocks
                    continue
                i = t
                while i < n:
                    if half:
                        _put_half(
                            o,
                            at + i,
                            _get_half(x, at + i)
                            * scale
                            * gain[unsafe_offset=i],
                        )
                    else:
                        _put(
                            o,
                            at + i,
                            _get(x, at + i) * scale * gain[unsafe_offset=i],
                        )
                    i += FTILE
                run += blocks

        elif op == OP_MATVEC:
            var cols = _fi(plan_i, rec, R_COLS)
            var rows = _fi(plan_i, rec, R_ROWS)
            var stride = _fi(plan_i, rec, R_STRIDE)
            var w_at = _fi(plan_i, rec, R_W)
            var epi = _fi(plan_i, rec, R_EPI)
            var kind = _fi(plan_i, rec, R_KIND)
            # The same eight combinations the unfused dispatch compiles,
            # chosen here by one uniform branch a record rather than by a
            # parameter on the whole kernel. That is the price of one kernel
            # source for every model: eight copies of the accumulation loop in
            # the binary and a branch executed once per projection.
            if kind == QK_U4:
                _do_matvec[32, True, QUANT_U4](
                    pool, x, o, aux, w_at, cols, stride, rows, epi, b, blocks
                )
            elif kind == QK_S4:
                _do_matvec[32, False, QUANT_S4](
                    pool, x, o, aux, w_at, cols, stride, rows, epi, b, blocks
                )
            elif kind == QK_U5:
                _do_matvec[32, True, QUANT_U5](
                    pool, x, o, aux, w_at, cols, stride, rows, epi, b, blocks
                )
            elif kind == QK_S5:
                _do_matvec[32, False, QUANT_S5](
                    pool, x, o, aux, w_at, cols, stride, rows, epi, b, blocks
                )
            elif kind == QK_S6:
                _do_matvec[16, False, QUANT_S6](
                    pool, x, o, aux, w_at, cols, stride, rows, epi, b, blocks
                )
            elif kind == QK_K4:
                _do_matvec[32, True, QUANT_K4](
                    pool, x, o, aux, w_at, cols, stride, rows, epi, b, blocks
                )
            elif kind == QK_K5:
                _do_matvec[32, True, QUANT_K5](
                    pool, x, o, aux, w_at, cols, stride, rows, epi, b, blocks
                )
            else:
                _do_matvec[32, False, QUANT_I8](
                    pool, x, o, aux, w_at, cols, stride, rows, epi, b, blocks
                )

        elif op == OP_ROPE:
            var heads = _fi(plan_i, rec, R_HEADS)
            var head_dim = _fi(plan_i, rec, R_HEAD_DIM)
            var dim = _fi(plan_i, rec, R_DIM)
            var kind = _fi(plan_i, rec, R_KIND)
            var steps = Pointer[Float32, MutAnyOrigin](
                to=arena[unsafe_offset=_fi(plan_i, rec, R_G)]
            )
            var factors = Pointer[Float32, MutAnyOrigin](
                to=arena[unsafe_offset=_fi(plan_i, rec, R_H)]
            )
            var scale = _ff(plan_f, rec, R_SCALE)
            var ext = _ff(plan_f, rec, R_EXT)
            var attn = _ff(plan_f, rec, R_ATTN)
            var low = _ff(plan_f, rec, R_LOW)
            var high = _ff(plan_f, rec, R_HIGH)
            var pairs = dim // 2

            # The key rotation runs on the cache and the query rotation on a
            # work vector, so this one branch is the difference between halves
            # and floats. It reads the same field for every block of the grid.
            var half = _fi(plan_i, rec, R_XS) == SPACE_KEYS
            var head = b
            while head < heads:
                var at = head * head_dim
                if half and PAIRED:
                    # Two rotations a thread rather than one, which makes the
                    # four elements it reads exactly the two words it writes.
                    # One rotation a thread would have it writing halves of two
                    # words whose other halves belong to a thread it cannot see,
                    # and `_put_pair` says why that is not allowed.
                    # `_rope_record` is where the even count of rotations this
                    # rests on is checked.
                    var two = t
                    while two * 2 + 1 < pairs:
                        var p0 = two * 2
                        var r0 = _rope_turn(
                            steps,
                            factors,
                            kind,
                            pos,
                            p0,
                            scale,
                            ext,
                            attn,
                            low,
                            high,
                        )
                        var r1 = _rope_turn(
                            steps,
                            factors,
                            kind,
                            pos,
                            p0 + 1,
                            scale,
                            ext,
                            attn,
                            low,
                            high,
                        )
                        if kind & NEOX_BIT == 0:
                            # Adjacent pairs, so a word is one rotation whole
                            # and a thread writes two of them.
                            var e = at + p0 * 2
                            var a0 = _get_half(x, e)
                            var b0 = _get_half(x, e + 1)
                            var a1 = _get_half(x, e + 2)
                            var b1 = _get_half(x, e + 3)
                            _put_pair(
                                o,
                                e >> 1,
                                a0 * r0[0] - b0 * r0[1],
                                a0 * r0[1] + b0 * r0[0],
                            )
                            _put_pair(
                                o,
                                (e >> 1) + 1,
                                a1 * r1[0] - b1 * r1[1],
                                a1 * r1[1] + b1 * r1[0],
                            )
                        else:
                            # Split halves, so a word is one side of two
                            # rotations and the two words are half a head apart.
                            var lo = at + p0
                            var hi = lo + pairs
                            var a0 = _get_half(x, lo)
                            var b0 = _get_half(x, hi)
                            var a1 = _get_half(x, lo + 1)
                            var b1 = _get_half(x, hi + 1)
                            _put_pair(
                                o,
                                lo >> 1,
                                a0 * r0[0] - b0 * r0[1],
                                a1 * r1[0] - b1 * r1[1],
                            )
                            _put_pair(
                                o,
                                hi >> 1,
                                a0 * r0[1] + b0 * r0[0],
                                a1 * r1[1] + b1 * r1[0],
                            )
                        two += FTILE
                    head += blocks
                    continue

                var pair = t
                while pair < pairs:
                    var rot = _rope_turn(
                        steps,
                        factors,
                        kind,
                        pos,
                        pair,
                        scale,
                        ext,
                        attn,
                        low,
                        high,
                    )
                    var c = rot[0]
                    var s = rot[1]
                    var lo = at + pair
                    var hi = at + pair + pairs
                    if kind & NEOX_BIT == 0:
                        lo = at + pair * 2
                        hi = lo + 1
                    if half:
                        var ha = _get_half(x, lo)
                        var hb = _get_half(x, hi)
                        _put_half(o, lo, ha * c - hb * s)
                        _put_half(o, hi, ha * s + hb * c)
                    else:
                        var a = _get(x, lo)
                        var bb = _get(x, hi)
                        _put(o, lo, a * c - bb * s)
                        _put(o, hi, a * s + bb * c)
                    pair += FTILE
                head += blocks

        elif op == OP_ATTEND:
            var head_dim = _fi(plan_i, rec, R_HEAD_DIM)
            var kv_width = _fi(plan_i, rec, R_KV)
            var group = _fi(plan_i, rec, R_GROUP)
            var window = _fi(plan_i, rec, R_WINDOW)
            var sinks = _fi(plan_i, rec, R_SINKS)
            var heads = _fi(plan_i, rec, R_HEADS)
            var scores = Pointer[Float32, MutAnyOrigin](
                to=work[unsafe_offset=_fi(plan_i, rec, R_ROW)]
            )
            var parts = Pointer[Float32, MutAnyOrigin](
                to=work[unsafe_offset=_fi(plan_i, rec, R_SPLIT)]
            )
            var scale = _ff(plan_f, rec, R_SCALE)
            var softcap = _ff(plan_f, rec, R_SOFTCAP)

            # How many blocks share a head, and how many keys each of them
            # takes. Every block computes the same two numbers from the same
            # three, which is what makes the pair index below agree across the
            # grid without anybody publishing anything.
            var splits = blocks // heads
            if splits < 1:
                splits = 1
            if splits > FSPLIT_MAX:
                splits = FSPLIT_MAX
            var span = (count + splits - 1) // splits
            if span < 1:
                span = 1
            # A short context does not need every slice, and an empty slice
            # would still cost the fold below a read, so the count comes back
            # from the span rather than the other way round.
            splits = (count + span - 1) // span
            var roww = head_dim + 2

            var p = b
            while p < heads * splits:
                var h = p // splits
                var kvh = h // group
                var qa = h * head_dim
                # One row of scores a head, `count` long, which is where the
                # scratch for a whole layer's attention comes from. The row
                # moves with the number of keys rather than with the context,
                # so a short sequence touches a short row. A slice only ever
                # writes and reads its own part of the row, so the row does not
                # cross a block even though the head does.
                var sa = h * count
                var lo = (p % splits) * span
                var hi = lo + span
                if hi > count:
                    hi = count
                # A warp to a key, the same `key_dot` the unfused kernels
                # call, with the coherent load the rest of this kernel uses.
                # The two paths have to agree in every bit and the reduction
                # order is part of that, so this is one function and not two.
                var lane = t % ALANES
                var team = t // ALANES
                var teams = FTILE // ALANES
                var mine = NEG_INF
                var j = lo + team
                while j < hi:
                    var visible = j < sinks or window <= 0 or j > pos - window
                    var s = NEG_INF
                    if visible:
                        var ka = j * kv_width + kvh * head_dim
                        s = (
                            key_dot[True](x, keys, qa, ka, head_dim, lane)
                            * scale
                        )
                        if softcap > 0:
                            s = softcap * _tanh(s / softcap)
                    if lane == 0:
                        scores[unsafe_offset=sa + j] = s
                        if s > mine:
                            mine = s
                    j += teams
                var top = _tree_max(mine)

                # A slice of a windowed model can be entirely outside the
                # window, and then the maximum is still minus infinity and the
                # exponent below would be a nan rather than a zero. Such a slice
                # publishes a total of zero and the fold leaves it out.
                var total = Float32(0)
                if top > NEG_INF:
                    var acc_sum = Float32(0)
                    j = lo + t
                    while j < hi:
                        var e = exp(scores[unsafe_offset=sa + j] - top)
                        scores[unsafe_offset=sa + j] = e
                        acc_sum += e
                        j += FTILE
                    total = _tree_sum(acc_sum)

                var pa = p * roww
                var d = t
                while d < head_dim:
                    var acc = Float32(0)
                    if total > 0:
                        for j2 in range(lo, hi):
                            var va = j2 * kv_width + kvh * head_dim
                            acc += scores[
                                unsafe_offset=sa + j2
                            ] * coherent_load_half[True](values, va + d)
                    _put(parts, pa + d, acc)
                    d += FTILE
                if t == 0:
                    _put(parts, pa + head_dim, top)
                    _put(parts, pa + head_dim + 1, total)
                p += blocks

            # The fold reads slices other blocks wrote, so it needs a rendezvous
            # of its own rather than the one the record boundary carries. It is
            # the only step in a layer that barriers in the middle, and it is
            # worth it: the alternative is one block a head.
            if not grid_barrier(counter, gen, blocks, seen):
                stuck = True
                break

            # A warp to an answer rather than a thread to one. There are only
            # `heads * head_dim` answers here, 576 on a small model against a
            # grid of `blocks * FTILE`, so a thread apiece left eighteen warps
            # on five blocks doing two serial passes over `splits` while the
            # other three hundred and seventy nine blocks waited at the barrier
            # below. Handing the slices to the lanes turns those passes into two
            # rounds and puts a hundred and forty four blocks to work.
            #
            # The lanes read `parts` a `roww` apart, which is the wrong stride
            # for a load and the opposite of what `key_dot` wants. It is still
            # the right trade here because `parts` is `heads * splits * roww`
            # floats, tens of kilobytes, and stays in L2 across the whole fold.
            # What this step was paying for was latency, not traffic.
            var flane = t % ALANES
            var fteams = FTILE // ALANES
            var e = b * fteams + t // ALANES
            var wide = blocks * fteams
            while e < heads * head_dim:
                var at = (e // head_dim) * splits * roww
                var d2 = e % head_dim
                var top2 = NEG_INF
                var s2 = flane
                while s2 < splits:
                    var m = _get(parts, at + s2 * roww + head_dim)
                    if m > top2:
                        top2 = m
                    s2 += ALANES
                # A lane with no slice of its own brings minus infinity to the
                # maximum and zero to the sums, which is what makes a single
                # slice come back through here unchanged in every bit. The
                # unfused `attend_merge_kernel` still folds one thread at a
                # time, and `tests/test_gpu_block.mojo` holds the two to an
                # exact match on a case that comes out at one slice.
                top2 = lane_group_max[num_lanes=ALANES](top2)
                var num = Float32(0)
                var den = Float32(0)
                s2 = flane
                while s2 < splits:
                    var base = at + s2 * roww
                    var l = _get(parts, base + head_dim + 1)
                    if l > 0:
                        # Every slice reduced against its own maximum, so the
                        # weight here is what it would have had against the
                        # whole row's maximum, and the same weight scales the
                        # slice's total and its values.
                        var wgt = exp(_get(parts, base + head_dim) - top2)
                        num += wgt * _get(parts, base + d2)
                        den += wgt * l
                    s2 += ALANES
                num = lane_group_sum[num_lanes=ALANES](num)
                den = lane_group_sum[num_lanes=ALANES](den)
                if flane == 0:
                    _put(o, e, num / den)
                e += wide

        elif op == OP_ACT:
            var n = _fi(plan_i, rec, R_N)
            var kind = _fi(plan_i, rec, R_KIND)
            var i = b * FTILE + t
            var stride = blocks * FTILE
            while i < n:
                if kind == ACT_GELU:
                    _put(o, i, activate[ACT_GELU](_get(x, i)))
                else:
                    _put(o, i, activate[ACT_SILU](_get(x, i)))
                i += stride

        else:
            var n = _fi(plan_i, rec, R_N)
            var i = b * FTILE + t
            var stride = blocks * FTILE
            while i < n:
                _put(o, i, _get(o, i) + _get(x, i))
                i += stride

        if _fi(plan_i, rec, R_SYNC) != 0:
            if not grid_barrier(counter, gen, blocks, seen):
                stuck = True
                break

    # One more rendezvous whatever the last record asked for, so that the
    # generation can be put back to zero for the next launch with nobody left
    # who could be confused by it. A block that is still spinning when the zero
    # lands is waiting for a value that is not zero, so it leaves, which is what
    # it was waiting to do.
    if not stuck:
        if not grid_barrier(counter, gen, blocks, seen):
            stuck = True
    if b == 0 and t == 0:
        dev32.store[ordering=Ordering.RELAXED](gen, Int32(0))
    if stuck and t == 0:
        _ = dev32.fetch_add(
            Pointer[Int32, MutAnyOrigin](to=sync[unsafe_offset=2]), Int32(1)
        )


def quant_kind(w: Tensor) raises -> Int:
    """Which of the eight compiled accumulation loops reads this weight.

    Written out in full rather than inferred from the form, for the reason the
    unfused dispatch gives: a type added later that breaks one of the
    coincidences between form, group and minimum plane should fail to build a
    plan and say so, not quietly read the wrong plane.
    """
    var g = group_size(w.kind)
    var carries_min = has_min(w.kind)
    var form = quant_form(w.kind)
    if form == QUANT_U4 and g == 32 and carries_min:
        return QK_U4
    if form == QUANT_S4 and g == 32 and not carries_min:
        return QK_S4
    if form == QUANT_U5 and g == 32 and carries_min:
        return QK_U5
    if form == QUANT_S5 and g == 32 and not carries_min:
        return QK_S5
    if form == QUANT_S6 and g == 16 and not carries_min:
        return QK_S6
    if form == QUANT_K4 and g == 32 and carries_min:
        return QK_K4
    if form == QUANT_K5 and g == 32 and carries_min:
        return QK_K5
    if form == QUANT_I8 and g == 32 and not carries_min:
        return QK_I8
    raise Error(
        "no fused matvec is compiled for quant form "
        + String(form)
        + " with a group of "
        + String(g)
        + (" and a minimum plane" if carries_min else " and no minimum")
    )


def rope_kind(spec: RopeSpec, use_factors: Bool) -> Int:
    var kind = 0
    if spec.neox:
        kind |= NEOX_BIT
    if use_factors:
        kind |= FACTOR_BIT
    return kind


struct StepPlan(Movable):
    """The step table on the host while it is being built, then on the device.

    Built once when a model binds and never touched again, which is what keeps
    this from being an upload a token. The two planes are separate because one
    of them holds addresses and counts that need sixty four bits and the other
    holds epsilons and rope constants that are float32 in the kernels they came
    from, and packing both into one plane would mean a bitcast at every read.
    """

    var ints: List[Int64]
    var floats: List[Float32]
    var records: Int

    def __init__(out self):
        self.ints = List[Int64]()
        self.floats = List[Float32]()
        self.records = 0

    def open(mut self, op: Int) -> Int:
        """Start a record, zeroed, and return its index."""
        for _ in range(REC_INTS):
            self.ints.append(Int64(0))
        for _ in range(REC_FLOATS):
            self.floats.append(Float32(0))
        var rec = self.records
        self.records += 1
        self.set(rec, R_OP, op)
        return rec

    def set(mut self, rec: Int, field: Int, value: Int):
        self.ints[rec * REC_INTS + field] = Int64(value)

    def setf(mut self, rec: Int, field: Int, value: Float32):
        self.floats[rec * REC_FLOATS + field] = value

    def sync(mut self, rec: Int):
        self.set(rec, R_SYNC, 1)

    def input(mut self, rec: Int, space: Int, off: Int, slot_mul: Int = 0):
        self.set(rec, R_XS, space)
        self.set(rec, R_XO, off)
        self.set(rec, R_XK, slot_mul)

    def output(mut self, rec: Int, space: Int, off: Int, slot_mul: Int = 0):
        self.set(rec, R_OS, space)
        self.set(rec, R_OO, off)
        self.set(rec, R_OK, slot_mul)

    def helper(mut self, rec: Int, space: Int, off: Int, slot_mul: Int = 0):
        self.set(rec, R_AS, space)
        self.set(rec, R_AO, off)
        self.set(rec, R_AK, slot_mul)


struct WorkPlan(Copyable, ImplicitlyCopyable, Movable):
    """Where each of a layer's intermediates sits in the one work vector.

    One allocation rather than seven, because a record names an operand as a
    base and an offset and the base has to be a kernel argument. Sized for one
    token, since stage one is a decode path and a chunk goes through the unfused
    kernels until stage three.
    """

    var norm: Int
    var q: Int
    var heads_out: Int
    var projected: Int
    var gate: Int
    var up: Int
    var scores: Int
    var splits: Int
    """The attention partials, one row of `head_dim + 2` a head a slice."""

    var elements: Int

    def __init__(out self, specs: List[BlockSpec], context: Int) raises:
        """Sized for the widest layer in the model rather than the first.

        They are the same layer in everything anybody ships, and taking the
        maximum costs one pass over a list at load time and removes a whole
        class of wrong answer from a model that ever stops being uniform.
        """
        if context <= 0:
            raise Error("a layer needs room for at least one position")
        if len(specs) == 0:
            raise Error("a model with no layers has no work to size")
        var width = 0
        var q_width = 0
        var hidden = 0
        var heads = 0
        var head_dim = 0
        for i in range(len(specs)):
            if specs[i].width > width:
                width = specs[i].width
            if specs[i].q_width() > q_width:
                q_width = specs[i].q_width()
            if specs[i].hidden > hidden:
                hidden = specs[i].hidden
            if specs[i].attn.heads > heads:
                heads = specs[i].attn.heads
            if specs[i].attn.head_dim > head_dim:
                head_dim = specs[i].attn.head_dim
        var at = 0
        self.norm = at
        at += width
        self.q = at
        at += q_width
        self.heads_out = at
        at += q_width
        self.projected = at
        at += width
        self.gate = at
        at += hidden
        self.up = at
        at += hidden
        self.scores = at
        at += heads * context
        self.splits = at
        # Sized for the widest grid rather than for the grid this machine will
        # pick, because the work vector is allocated at load time and the block
        # count is a launch argument. It is half a megabyte on an eight billion
        # parameter model and a few tens of kilobytes on a small one.
        at += heads * FSPLIT_MAX * (head_dim + 2)
        self.elements = at


comptime CUDA_BLOCKS = 384
"""How many blocks a fused launch uses on a backend with an occupancy query.

The sweep in max.md says a barrier round over 1536 blocks costs 4.59
microseconds, which is a launch, and that it falls to 1.26 at 384, 0.75 at 96
and is flat below that. The cost at the top is 1536 blocks contending on one
counter word rather than the rendezvous itself, so the useful grid is a few
hundred.

Decoding a real model says the same range and picks the top of it rather than
the bottom. SmolLM2 135M on a 4090 at 128 tokens decodes in 366 ms at 96
blocks, 239 at 256, 238 at 384 and 239 at 512, against 451 unfused, so
everything from 256 up is flat and 96 gives away half of what the path is
worth. Above that it stops working rather than getting slower: 640 blocks and
768 blocks never finish a token.

What it costs is arithmetic width: a 4096 row projection here is 43 rows a block
in a strided loop rather than one row a block. That is the number this design is
most likely to be wrong about on a large model and it is why the fused path is
measured against the unfused one rather than assumed to win.
"""

comptime METAL_PER_SM = 4
"""Blocks a multiprocessor on a backend with no occupancy query.

It was one, because one block a multiprocessor is the only count every Metal
device is known to hold, and a grid that is not fully resident hangs the GPU,
which on a laptop is the display. Four is what an M4 measures: Qwen 2.5 0.5B
decodes at 100 ms a token at ten blocks, 63 at twenty, 50 at forty, 49 at fifty
and 55 at sixty, and at seventy it stops finishing a token at all. The M4 has
ten multiprocessors, so the cliff is between six and seven blocks a
multiprocessor and four is the flat part with a margin under it.

The margin is the whole point, because the self test does not cover this. It
runs the barrier and nothing else, and a kernel that is only a barrier is
resident at counts the real one is not: the same M4 rendezvouses fine at a
hundred and sixty blocks, sixteen a multiprocessor, where the fused kernel
deadlocks at seventy. A probe cannot be the safety net for a grid whose
residency depends on the registers of the kernel that runs in it, so the number
is conservative instead.
"""

comptime CUDA_PER_SM = 3
"""And the same ceiling on a backend that does have the query, because the
query does not answer this question either.

`occupancy_max_active_blocks_per_multiprocessor` says how many blocks fit on a
multiprocessor, not how many the scheduler will run at once, and an ordinary
launch is not a cooperative one. On a 4090 the fused kernel deadlocks at 640
blocks, which is five a multiprocessor, so the query is an upper bound on the
answer rather than the answer. Three, so that a card with a tenth of the
multiprocessors gets a grid a tenth as wide instead of one that is over its own
cliff.
"""


def barrier_probe_kernel(
    sync: Pointer[Int32, MutAnyOrigin], blocks_dev: Int32, rounds_dev: Int32
):
    """Nothing but the rendezvous, so what it proves is the rendezvous.

    Run once when a session opens. A grid that is not resident deadlocks, and a
    deadlock inside a forward pass is a hung display rather than an error
    message, so the question is asked before a model runs and at the same grid
    the model is going to use.
    """
    var blocks = Int(blocks_dev)
    var seen = Int32(0)
    var counter = Pointer[Int32, MutAnyOrigin](to=sync[unsafe_offset=0])
    var gen = Pointer[Int32, MutAnyOrigin](to=sync[unsafe_offset=1])
    for _ in range(Int(rounds_dev)):
        if not grid_barrier(counter, gen, blocks, seen):
            if thread_idx.x == 0:
                _ = dev32.fetch_add(
                    Pointer[Int32, MutAnyOrigin](to=sync[unsafe_offset=2]),
                    Int32(1),
                )
            return
    if block_idx.x == 0 and thread_idx.x == 0:
        dev32.store[ordering=Ordering.RELAXED](gen, Int32(0))


comptime BLOCKS_ENV = "MOLLA_FUSED_BLOCKS"
"""An explicit fused grid width, for sweeping one.

The width is the whole trade this path makes, and the two ways of arriving at it
are a query on one backend and a constant on the other, so measuring what it is
worth means overriding both. `fused_selftest` still runs at whatever comes out
of here, so an override that is not resident refuses to start rather than
hanging, which is what makes this safe to hand to somebody sweeping it.
"""


def fused_grid(ctx: DeviceContext) raises -> Int:
    """How many blocks a fused launch gets on this device.

    Two backends and two ways of asking. CUDA answers
    `occupancy_max_active_blocks_per_multiprocessor`, so the resident bound is
    known and the grid is the smaller of that and `CUDA_BLOCKS`. Metal raises on
    the same query, so there the grid is the multiprocessor count times
    `METAL_PER_SM`, which is the floor every device holds.

    `MOLLA_FUSED_BLOCKS` overrides both.
    """
    var asked = getenv(BLOCKS_ENV)
    if asked != "":
        var n = 0
        var digits = asked.as_bytes()
        for i in range(len(digits)):
            var c = Int(digits[i])
            if c < 48 or c > 57:
                raise Error(BLOCKS_ENV + " is not a number: " + String(asked))
            n = n * 10 + (c - 48)
        if n < 1:
            raise Error("a fused launch needs at least one block")
        return n

    var sms: Int
    try:
        sms = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    except:
        sms = 0
    if sms <= 0:
        raise Error(
            "this device does not report a multiprocessor count, so there is no"
            " grid size a fused launch could be known to be resident at"
        )
    var per_sm: Int
    try:
        var f = ctx.compile_function[fused_kernel]()
        per_sm = f.occupancy_max_active_blocks_per_multiprocessor(FTILE, 0)
    except:
        per_sm = 0
    if per_sm <= 0:
        return sms * METAL_PER_SM
    if per_sm > CUDA_PER_SM:
        per_sm = CUDA_PER_SM
    var blocks = sms * per_sm
    if blocks > CUDA_BLOCKS:
        blocks = CUDA_BLOCKS
    return blocks


struct FusedPlan(Movable):
    """A model's step table on the device, plus the memory a launch needs.

    One of these per session, built when the model binds. It owns the two plan
    planes, the three sync words, the work vector every layer's intermediates
    live in, and the record range each layer occupies, which is what makes a
    launch a pair of indices.
    """

    var ints: DeviceBuffer[DType.int64]
    var floats: DeviceBuffer[DType.float32]
    var sync: DeviceBuffer[DType.int32]
    var work: DeviceVec
    var starts: List[Int]
    """Where each layer's records begin, with one past the end appended, so
    layer `i` is `starts[i]` through `starts[i + 1]`."""

    var pool: Int
    """The address every `R_W` in the table is measured from.

    A record carries a weight as a byte offset rather than as an address,
    because a pointer loaded out of device memory loses its address space on
    Metal, so the base arrives as a kernel argument and the record carries the
    difference. See `_pool_base` in `molla.nn.gpu_block`.
    """

    var blocks: Int
    var records: Int

    def __init__(out self, ctx: DeviceContext) raises:
        """A plan for a session that has decided against the fused path.

        Nothing is compiled and nothing is sized, which is the point of it: a
        session that will never launch the kernel should not pay what compiling
        it costs. `records` is zero, and a launch against a plan with no records
        in it raises rather than doing anything quietly.
        """
        self.pool = 0
        self.blocks = 0
        self.records = 0
        self.starts = List[Int]()
        self.work = DeviceVec(ctx, 1)
        self.ints = ctx.enqueue_create_buffer[DType.int64](1)
        self.floats = ctx.enqueue_create_buffer[DType.float32](1)
        self.sync = ctx.enqueue_create_buffer[DType.int32](3)
        ctx.enqueue_memset(self.sync, Int32(0))
        ctx.synchronize()

    def __init__(
        out self,
        ctx: DeviceContext,
        plan: StepPlan,
        starts: List[Int],
        shape: WorkPlan,
        pool: Int,
    ) raises:
        if plan.records == 0:
            raise Error("a fused plan with no steps in it is not a plan")
        if len(starts) < 2:
            raise Error("a fused plan needs at least one layer's range")
        self.pool = pool
        self.blocks = fused_grid(ctx)
        self.records = plan.records
        self.starts = starts.copy()
        self.work = DeviceVec(ctx, shape.elements)
        self.ints = ctx.enqueue_create_buffer[DType.int64](len(plan.ints))
        self.floats = ctx.enqueue_create_buffer[DType.float32](len(plan.floats))
        self.sync = ctx.enqueue_create_buffer[DType.int32](3)
        ctx.enqueue_copy(
            self.ints,
            Pointer[Int64, MutAnyOrigin](
                unsafe_from_address=Int(plan.ints.unsafe_ptr())
            ),
        )
        ctx.enqueue_copy(
            self.floats,
            Pointer[Float32, MutAnyOrigin](
                unsafe_from_address=Int(plan.floats.unsafe_ptr())
            ),
        )
        ctx.enqueue_memset(self.sync, Int32(0))
        ctx.synchronize()

    def stuck(self) raises -> Int:
        """How many blocks have ever given up at a barrier.

        Read when something asks rather than after every token, because reading
        it is a transfer and a synchronize and the thing it detects is a
        property of the grid rather than of the token.

        Into a plain list rather than a host buffer, for the reason
        `fused_selftest` gives: the first pinned allocation in a process costs
        1.2 GiB of resident memory and three integers do not need one.
        """
        var out = List[Int32]()
        for _ in range(3):
            out.append(0)
        self.sync.context().enqueue_copy(
            Pointer[Int32, MutAnyOrigin](
                unsafe_from_address=Int(out.unsafe_ptr())
            ),
            self.sync,
        )
        self.sync.context().synchronize()
        return Int(out[2])


def fused_selftest(ctx: DeviceContext, blocks: Int) raises:
    """Prove the grid can rendezvous before a model is asked to run in it.

    Sixteen rounds at the grid the layers will use. If a block gives up, the
    grid was not resident and every fused launch after this one would hang or
    answer with whatever the barrier let through, so this raises rather than
    letting a model start.

    The three words come back into a list rather than into a host buffer, and
    that is not a style choice. `enqueue_create_host_buffer` allocates pinned
    memory, and the first pinned allocation in a process costs 1.2 GiB of
    resident host memory whatever its size: on gpc a fused SmolLM2 session was
    1468 MiB with this line and 279 MiB without it, for three integers. That
    was the whole of #222, which had been read as the price of compiling the
    fused kernel and is not. An ordinary device to host copy of three words
    costs nothing and tells us the same thing.
    """
    if blocks < 1:
        raise Error("a fused launch needs at least one block")
    var sync = ctx.enqueue_create_buffer[DType.int32](3)
    ctx.enqueue_memset(sync, Int32(0))
    ctx.synchronize()
    ctx.enqueue_function[barrier_probe_kernel](
        Pointer[Int32, MutAnyOrigin](
            unsafe_from_address=Int(sync.unsafe_ptr())
        ),
        Int32(blocks),
        Int32(16),
        grid_dim=(blocks, 1, 1),
        block_dim=(FTILE, 1, 1),
    )
    var out = List[Int32]()
    for _ in range(3):
        out.append(0)
    ctx.enqueue_copy(
        Pointer[Int32, MutAnyOrigin](unsafe_from_address=Int(out.unsafe_ptr())),
        sync,
    )
    ctx.synchronize()
    if Int(out[2]) != 0:
        raise Error(
            "a grid of "
            + String(blocks)
            + " blocks of "
            + String(FTILE)
            + " threads cannot reach a barrier on this device, which means"
            " it is"
            " not fully resident. The fused path needs every block running at"
            " once"
        )


def launch_fused(
    ctx: DeviceContext,
    p: FusedPlan,
    arena: DeviceVec,
    mut resid: DeviceVec,
    mut keys: DeviceHalf,
    mut values: DeviceHalf,
    first: Int,
    last: Int,
    pos: Int,
    slot: Int,
    count: Int,
) raises:
    """One launch over records `first` through `last - 1`.

    Queued and not synchronized, like everything else on a token's path. A layer
    is one of these where it used to be twelve.
    """
    if first < 0 or last > p.records or first >= last:
        raise Error(
            "a fused launch was asked for records "
            + String(first)
            + " to "
            + String(last)
            + " of a plan that holds "
            + String(p.records)
        )
    comptime if not has_accelerator():
        raise Error(
            "this build has no device code in it, so there is no fused kernel"
            " to run. Accelerator support is decided when molla is compiled,"
            " not when it is run"
        )
    else:
        ctx.enqueue_function[fused_kernel](
            Pointer[Int64, MutAnyOrigin](
                unsafe_from_address=Int(p.ints.unsafe_ptr())
            ),
            Pointer[Float32, MutAnyOrigin](
                unsafe_from_address=Int(p.floats.unsafe_ptr())
            ),
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=p.pool),
            arena.ptr(),
            p.work.ptr(),
            resid.ptr(),
            keys.ptr(),
            values.ptr(),
            Pointer[Int32, MutAnyOrigin](
                unsafe_from_address=Int(p.sync.unsafe_ptr())
            ),
            Int32(first),
            Int32(last),
            Int32(p.blocks),
            Int32(pos),
            Int32(slot),
            Int32(count),
            grid_dim=(p.blocks, 1, 1),
            block_dim=(FTILE, 1, 1),
        )
