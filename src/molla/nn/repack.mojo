"""The layout the kernels actually want, and the transform into it.

A ggml block is a shipping format. It is compact, it is what the file has, and
every one of the eight quantized types packs its bits differently, so the fused
dot product in `molla.nn.kernel` is eight separate loops that each unpack a
different arrangement of nibbles and stray high bits before they can multiply
anything. That unpacking is real work and it happens on every token, against
every weight, forever, to recover numbers that were already known the first time
the file was read.

So read them once. This module defines one layout that all eight types repack
into, and it is deliberately the dullest thing that could work: the quants in
one plane, then the scales in planes at the end of the row.

    row = [ quants ][ groups float16 dscale ][ groups float16 mscale ]

with the mscale plane there only for the types that carry a minimum. A value is
`dscale[g] * q[i] + mscale[g]` where `g` is `i // group_size`, and that is the
whole format. The group is 32 values for everything except q6_k, which is 16.

The three k types keep the same two scales in a narrower pair of planes, because
what they have is not a float16 a group. A k type's group scale is a small
integer against one float16 for the whole 256 value block, six bits unsigned for
q4_k and q5_k and a signed byte for q6_k, and every one of those fits a byte
exactly. So they store the integer a byte a group and the float16 once a block:

    row = [ quants ][ groups dscale ][ groups mscale ]
          [ blocks float16 dfactor ][ blocks float16 mfactor ]

and a value is `dfactor[b] * dscale[g] * q[i] + mfactor[b] * mscale[g]` where
`b` is `g // block_groups`. That is a byte a group rather than two plus a
sixteenth, so a 4096 wide q4_k row is 2368 bytes against 2560, and it is
lossless where the float16 plane was not: the product that used to be rounded
into a float16 at repack time is now formed in float32 at read time out of the
two exact factors the file had.

A quant is stored at the width the file stores it, which `quant_form` decides
and nothing else does. The four bit plane is two values a byte, low nibble first
in the order the values appear, so position `i` is in byte `i // 2` and in the
low half when `i` is even. Five and six bits are that same nibble plane with a
second plane of one or two bits a value after it.

Two things fall out of it that are worth saying.

The centring folds into the byte. q4_0 subtracts eight, q5_0 subtracts sixteen
and q6_k subtracts thirty two, and every one of those is a constant the repack
can apply once instead of the kernel applying it per value per token. That is
why those types have no mscale plane at all rather than a plane full of zeros:
after centring they are a pure scale, and a plane of zeros would be a megabyte
of memory traffic to add nothing. The four types that do keep a minimum are
q4_1 and q5_1, where it is stored directly, and q4_k and q5_k, where the block
minimum and the group multiplier fold into one number.

Every quantized value fits in a signed byte and three of them fit in a nibble.
q6_k centred runs -32 to 31 and q5_1 runs 0 to 31, so those keep a byte. q4_k
and q4_1 run 0 to 15 and q4_0 centred runs -8 to 7, so those three keep a
nibble. That is not a coincidence, it is what being a sub eight bit format
means.

The nibble matters more than it looks. A q4_k weight is 4.5 bits per value in
the file, and it was 10 bits here until it was 5 plus a float32 scale plane,
which on an 8B was a repack cache twice the size of the model and twice the
bytes to read for every token of every run forever. Decode on a large model is
bandwidth bound, so those bytes were not a memory cost that bought speed, they
were a memory cost that spent it. `docs/validation/layout.md` has the
measurement.

The scale planes are float16, which is the width the file's own scales are, so
what a group costs here is a scale and no more. They were float32 until the
measurement said what that was worth. That change was not bit exact for the k
types, because a k type's scale is a product of a float16 and a small integer
and a product needs more mantissa than either factor, and storing the two
factors apart is what took the rounding back out.

Nothing here reads a file or knows what a model is. It takes packed bytes at an
address and writes unpacked bytes at another one, which is what lets
`molla.model.repack` own the cache and the disk without this module growing a
dependency on either.
"""

from std.memory import bitcast

from molla.nn.quant import (
    Q_BF16,
    Q_F16,
    Q_F32,
    Q_Q4_0,
    Q_Q4_1,
    Q_Q4_K,
    Q_Q5_0,
    Q_Q5_1,
    Q_Q5_K,
    Q_Q6_K,
    Q_Q8_0,
    _i8,
    _k_scale,
    _qh,
    _u8,
    block_bytes,
    block_elements,
    dequant_run,
    f16_at,
)
from molla.sys.mmap import RawPtr

comptime LAYOUT_GGML = 0
"""The bytes as the file has them. What a tensor is until it is repacked."""

comptime LAYOUT_PLANAR = 1
"""Quants in one plane and scales in another, as described above."""

comptime QUANT_I8 = 0
"""One signed byte a value. q8_0 alone."""

comptime QUANT_U4 = 1
"""Two unsigned nibbles a byte, values 0 to 15. q4_1 alone."""

comptime QUANT_S4 = 2
"""Two two's complement nibbles a byte, values -8 to 7. q4_0 alone.

Kept apart from `QUANT_U4` rather than shifted into it, even though q4_1 and
q4_k both have an mscale plane that could absorb an offset of eight. Folding it
there would change the terms the dot product sums, which changes the last bits
of every result, and there is no reason to spend that on a change whose whole
point is that the numbers do not move.
"""

comptime QUANT_U5 = 3
"""Five bits a value in two planes, values 0 to 31. q5_1 alone."""

comptime QUANT_S5 = 4
"""The same two planes, read sixteen lower, values -16 to 15. q5_0 alone."""

comptime QUANT_S6 = 5
"""Six bits a value in two planes, read thirty two lower, values -32 to 31.

q6_k alone. Kept apart from `QUANT_S5` rather than folded into one six bit form
with a parameter, because the two differ in the width of the second plane and
so in every index into it, and a parameter that changes an index is a runtime
shift where a separate form is a constant.
"""

comptime QUANT_K4 = 6
"""`QUANT_U4`'s quant plane with block scaled scale planes. q4_k alone."""

comptime QUANT_K5 = 7
"""`QUANT_U5`'s quant planes with block scaled scale planes. q5_k alone.

q4_k and q5_k read their quants exactly the way q4_1 and q5_1 do and differ
only in where the scale of a group comes from, and a form of their own is how
that gets said. The alternative was a second parameter on every kernel, which
would double the instantiations of all six forms to distinguish three, where
this adds two cases to a dispatch that already has six.
"""

comptime SCALE_BYTES = 2
"""Bytes one float16 scale takes in a planar row.

The width of every scale plane except the byte wide ones a block scaled form
keeps, and the width of those forms' block factors. The argument for it being
two rather than four is in `_put_scale` and measured in
docs/validation/layout.md.
"""

comptime ROW_ALIGN = 16
"""Bytes a planar row is rounded up to.

The width of the widest load any kernel makes over a row, which is the sixteen
bytes of the quant plane one thread stages in the Apple matrix core path. Rows
are laid out back to back, so this is the alignment of every row and not just
the first.
"""

comptime LAYOUT_VERSION = 5
"""Bumped whenever the meaning of a planar byte changes.

A cache file records this and a load that finds a different number treats the
cache as absent. That is the only safe behaviour: a repack is derived data, an
old one is not wrong so much as no longer meaningful, and a layout change that
silently reused it would produce a model that runs and answers badly.
"""


def repackable(kind: Int) -> Bool:
    """Whether this type has a planar form.

    The unquantized types do not. f32 is already the layout every kernel wants,
    and f16 and bf16 would lose precision going through an int8, which is not a
    trade worth making for tensors that are norms and biases and never the bulk
    of a model anyway.
    """
    if kind == Q_Q4_0 or kind == Q_Q4_1 or kind == Q_Q8_0:
        return True
    if kind == Q_Q5_0 or kind == Q_Q5_1:
        return True
    return kind == Q_Q4_K or kind == Q_Q5_K or kind == Q_Q6_K


def group_size(kind: Int) -> Int:
    """Values sharing one scale, or zero for a type with no planar form.

    32 for everything except q6_k. q6_k stores sixteen signed scales per 256
    value block where q4_k stores eight, so keeping 32 here would mean averaging
    two scales into one and changing the numbers, which is not a repack.
    """
    if not repackable(kind):
        return 0
    if kind == Q_Q6_K:
        return 16
    return 32


def group_shift(group: Int) -> Int:
    """The right shift that divides by `group`, or -1 if it is not a power of
    two.

    Both group sizes in the type table are powers of two, so a group index is a
    shift rather than a division, and the difference is worth having. `i //
    group` with `group` a compile time constant is a signed divide, and a signed
    divide has to correct for a negative numerator whatever the divisor is.
    NVVM strength reduces the whole thing away and Metal does not: measured with
    `scripts/matvec_probe.mojo` on an M4, a 4096 by 4096 q4_k matvec is 1916
    microseconds with the divide and 867 with the shift, so more than half of
    that kernel was the compiler being careful about a sign the index never has.

    -1 rather than a raise because the callers are device kernels, which cannot
    raise and evaluate this at compile time. `test_repack` checks that every
    kind in the table gives a shift that is really a shift, so a type added with
    a group of 24 fails there rather than reading the wrong scale.
    """
    var n = group
    var shift = 0
    while n > 1:
        if n & 1 != 0:
            return -1
        n >>= 1
        shift += 1
    return shift if n == 1 else -1


def has_min(kind: Int) -> Bool:
    """Whether the row carries an mscale plane.

    True for the two types that store a minimum in the block and the two K
    quants that store a per group one. False for the centred types, where the
    offset became part of the byte.
    """
    if kind == Q_Q4_1 or kind == Q_Q5_1:
        return True
    return kind == Q_Q4_K or kind == Q_Q5_K


def quant_form(kind: Int) -> Int:
    """How a value is stored in the quant plane.

    The one place that decides it. Everything that writes the plane, reads it,
    or measures it asks here, so adding a type or narrowing one is a line in
    this function and a case in the two readers, rather than a constant to find
    in six files.
    """
    if kind == Q_Q4_1:
        return QUANT_U4
    if kind == Q_Q4_0:
        return QUANT_S4
    if kind == Q_Q5_1:
        return QUANT_U5
    if kind == Q_Q5_0:
        return QUANT_S5
    if kind == Q_Q4_K:
        return QUANT_K4
    if kind == Q_Q5_K:
        return QUANT_K5
    if kind == Q_Q6_K:
        return QUANT_S6
    return QUANT_I8


def quant_bits(kind: Int) -> Int:
    """Bits a value occupies in the quant plane, 4, 5, 6 or 8."""
    var form = quant_form(kind)
    if form == QUANT_I8:
        return 8
    if form == QUANT_S6:
        return 6
    if form == QUANT_U5 or form == QUANT_S5 or form == QUANT_K5:
        return 5
    return 4


def quant_high_bits(form: Int) -> Int:
    """Bits of a value that live in the second plane, 0, 1 or 2.

    The four and eight bit forms have one plane and this is zero for them. A
    five bit value is a nibble and one bit, a six bit value is a nibble and two,
    and the odd bits go in a plane of their own rather than in a stream, because
    a stream costs a shift chain per value on the read side and the whole point
    of the layout is that the read is cheap.
    """
    if form == QUANT_S6:
        return 2
    if form == QUANT_U5 or form == QUANT_S5 or form == QUANT_K5:
        return 1
    return 0


def quant_bias(form: Int) -> Int:
    """What a form's reader subtracts after assembling the value, 0, 16 or 32.

    The four and eight bit forms carry their sign in the byte and subtract
    nothing here. The five and six bit forms store an unsigned value and take
    the offset off at the end, which is free: the reader turns the integer into
    a float by putting it in the mantissa of two to the twenty third and
    subtracting that constant back off, so an offset of sixteen is a different
    constant in a subtraction that was happening anyway.
    """
    if form == QUANT_S5:
        return 16
    if form == QUANT_S6:
        return 32
    return 0


def block_scaled(form: Int) -> Bool:
    """Whether this form keeps a group scale in a byte and a factor per block.

    True for the three k types and false for everything else, which is not a
    property of the quant plane at all but of what the file stores a scale as.
    The five non k types have a float16 a block and no per group integer, so
    there is nothing to split and their group scale is that float16 unchanged.
    """
    return form == QUANT_K4 or form == QUANT_K5 or form == QUANT_S6


def scale_signed(form: Int) -> Bool:
    """Whether a byte of a block scaled scale plane is two's complement.

    q6_k's group scales are signed bytes in the file and q4_k's and q5_k's are
    six bit unsigned integers, so the plane is written at the file's own
    signedness and read back the same way. Biasing q6_k into an unsigned plane
    would work and would cost a subtraction per group to undo, for nothing: a
    byte holds either range and the reader knows which form it has.
    """
    return form == QUANT_S6


def block_groups(form: Int) -> Int:
    """Groups sharing one float16 factor, or zero for a form with no factor.

    A k quant block is 256 values however wide its groups are, so this is 256
    over the group size: eight for q4_k and q5_k and sixteen for q6_k.
    """
    if form == QUANT_K4 or form == QUANT_K5:
        return 8
    if form == QUANT_S6:
        return 16
    return 0


def block_shift(form: Int) -> Int:
    """The right shift from a group index to a block index, or 0 for no factor.

    A shift for the reason `group_shift` gives, and it is one here because both
    counts are powers of two.
    """
    if not block_scaled(form):
        return 0
    return group_shift(block_groups(form))


def planar_low_bytes(kind: Int, cols: Int) raises -> Int:
    """Bytes the first plane of one row occupies.

    A byte a value at eight bits and a nibble a value at every other width, so
    the five and six bit types share their first plane with the four bit ones
    and only differ in what follows it.
    """
    if not repackable(kind):
        raise Error("ggml type " + String(kind) + " has no planar form")
    if quant_bits(kind) == 8:
        return cols
    if cols % 2 != 0:
        raise Error(
            "a row of "
            + String(cols)
            + " cannot be packed two values to a byte"
        )
    return cols // 2


def planar_quant_bytes(kind: Int, cols: Int) raises -> Int:
    """Bytes both quant planes of one row occupy.

    A whole number for every type and row width molla accepts, because a four
    bit type packs 32 or 256 values to a block and a row is a whole number of
    blocks, so `cols` is even long before it gets here. Checked anyway, because
    the alternative to checking is a row width that silently drops its last
    value.

    The second plane is `cols / 8` bytes at five bits and `cols / 4` at six, and
    both divide because a row is a whole number of 32 or 256 value blocks.
    """
    var low = planar_low_bytes(kind, cols)
    var high = quant_high_bits(quant_form(kind))
    if high == 0:
        return low
    var per = 8 // high
    if cols % per != 0:
        raise Error(
            "a row of "
            + String(cols)
            + " cannot be packed "
            + String(per)
            + " values to a byte"
        )
    return low + cols // per


def planar_groups(kind: Int, cols: Int) raises -> Int:
    var g = group_size(kind)
    if g == 0:
        raise Error("ggml type " + String(kind) + " has no planar form")
    if cols % g != 0:
        raise Error(
            "a row of "
            + String(cols)
            + " is not a whole number of "
            + String(g)
            + " value groups"
        )
    return cols // g


def planar_blocks(kind: Int, cols: Int) raises -> Int:
    """Groups of groups sharing one float16 factor, or zero for a plain form.

    Checked rather than assumed, for the reason `planar_groups` checks: a row
    that is not a whole number of blocks would give the last block a factor
    covering fewer groups than the reader is going to read with it.
    """
    var form = quant_form(kind)
    var per = block_groups(form)
    if per == 0:
        return 0
    var groups = planar_groups(kind, cols)
    if groups % per != 0:
        raise Error(
            "a row of "
            + String(groups)
            + " groups is not a whole number of "
            + String(per)
            + " group blocks"
        )
    return groups // per


def planar_scale_bytes(kind: Int, cols: Int) raises -> Int:
    """Bytes one row's scale planes occupy, block factors included.

    Two bytes a group for a plain form. One byte a group plus two bytes a block
    for a block scaled one, in both planes where the type carries a minimum.
    """
    var planes = 2 if has_min(kind) else 1
    var groups = planar_groups(kind, cols)
    if not block_scaled(quant_form(kind)):
        return planes * groups * SCALE_BYTES
    return planes * (groups + planar_blocks(kind, cols) * SCALE_BYTES)


def planar_row_bytes(kind: Int, cols: Int) raises -> Int:
    """Bytes one planar row occupies, rounded up to `ROW_ALIGN`.

    Rows sit back to back, so a row length is the alignment every row after the
    first gets, and the widest load a kernel makes over a row is sixteen bytes
    of the quant plane. Rounding up is how that is guaranteed rather than
    arrived at. It used to be arrived at: every combination of type and width in
    the models molla is run against happened to land on a multiple of sixteen
    except q8_0 at 896 columns, which is Qwen2.5 0.5B and which was reading a
    misaligned vector on the Apple matrix core path. The block scaled k types
    made that worse, since a q4_k row is 148 bytes a block and 148 is not a
    multiple of eight.

    What it costs is at most fifteen bytes a row against rows that are hundreds
    or thousands, and nothing at all for the shapes that were already aligned,
    which is most of them.
    """
    var used = planar_quant_bytes(kind, cols) + planar_scale_bytes(kind, cols)
    return (used + ROW_ALIGN - 1) & ~(ROW_ALIGN - 1)


def _put_scale(p: RawPtr, at: Int, value: Float32):
    """A little endian float16, byte at a time.

    The mirror of `molla.nn.quant.f16_at` and written the same way for the same
    reason: the destination is a byte address in a mapping, a store through a
    reinterpreted pointer would be an alignment assumption this code has not
    earned, and this runs once per group at repack time rather than per token.

    Nothing rounds here any more. Every value this is handed is a float16 out
    of a block widened on the way past, so it lands back on itself: for the
    five plain types that is the group's scale and for the three k types it is
    the block's factor, whose integer half goes to `_put_scale_byte` instead of
    being multiplied in.
    """
    var bits = bitcast[DType.uint16, 1](value.cast[DType.float16]())
    p.unsafe_store(at, UInt8(bits & 0xFF))
    p.unsafe_store(at + 1, UInt8((bits >> 8) & 0xFF))


def _put_scale_byte(p: RawPtr, at: Int, value: Int):
    """One group scale of a block scaled form, in a byte.

    Six bits unsigned for q4_k and q5_k and a signed byte for q6_k, and the mask
    is what lets one function write both: it is the two's complement of the
    signed one and does nothing at all to the other.
    """
    p.unsafe_store(at, UInt8(value & 0xFF))


def _put_i8(p: RawPtr, at: Int, value: Int):
    """A signed value in a byte. The mask is what makes it two's complement
    rather than an error on the way through UInt8."""
    p.unsafe_store(at, UInt8(value & 0xFF))


def _put_nibble(p: RawPtr, base: Int, i: Int, value: Int):
    """Value `i` of a four bit quant plane starting at `base`.

    Read, merge, write, which is only correct because every position in the
    plane is written exactly once by the loops below, so both halves of every
    byte end up set whatever the buffer held before. The scratch buffer a
    repack writes into is reused row after row and is not cleared between them,
    so relying on it being zero would be wrong on every row but the first.

    One thread owns that buffer for the length of a row, so there is no
    interleaving to worry about. Rows are a multiple of four bytes long, so two
    threads on two rows never share a byte either.
    """
    var at = base + (i >> 1)
    var b = Int(p.unsafe_load(at))
    if (i & 1) != 0:
        b = (b & 0x0F) | ((value & 0xF) << 4)
    else:
        b = (b & 0xF0) | (value & 0xF)
    p.unsafe_store(at, UInt8(b))


def _put_high(form: Int, p: RawPtr, hi: Int, i: Int, value: Int):
    """The bits of value `i` above the fourth, into the second plane.

    Read, merge, write, for the reason `_put_nibble` gives: the scratch buffer
    is reused row after row and is never cleared, so every position has to be
    written rather than assumed zero. Every position is, because the loops that
    call this walk a whole row.
    """
    var bits = quant_high_bits(form)
    var per = 8 // bits
    var at = hi + i // per
    var shift = (i % per) * bits
    var mask = ((1 << bits) - 1) << shift
    var b = Int(p.unsafe_load(at))
    p.unsafe_store(at, UInt8((b & ~mask) | (((value >> 4) << shift) & mask)))


def _put_quant(form: Int, p: RawPtr, base: Int, hi: Int, i: Int, value: Int):
    """Value `i` of a quant plane, at whichever width the type uses.

    `value` arrives already offset for the form: a `QUANT_S5` caller passes 0 to
    31 and not -16 to 15, because the offset is the reader's and taking it here
    would need the plane to hold a sign it has no bit for.
    """
    if form == QUANT_I8:
        _put_i8(p, base + i, value)
        return
    _put_nibble(p, base, i, value)
    if quant_high_bits(form) != 0:
        _put_high(form, p, hi, i, value)


def _quant_at(form: Int, p: RawPtr, base: Int, hi: Int, i: Int) -> Int:
    """Value `i` of a quant plane, back out.

    The sign extension for `QUANT_S4` is subtracting sixteen and not eight. The
    stored nibble is the value's low four bits in two's complement, so -8 is
    stored as 8 and -1 as 15, and it is the whole four bit word that wraps.

    The five and six bit forms are the other way round. Nothing about them
    wraps: the two planes assemble an unsigned value and the offset comes off
    afterwards, which is what `quant_bias` gives.
    """
    if form == QUANT_I8:
        return _i8(p, base + i)
    var b = Int(p.unsafe_load(base + (i >> 1)))
    var n = (b >> 4) if (i & 1) != 0 else (b & 0xF)
    if form == QUANT_S4:
        return n - 16 if n >= 8 else n
    var bits = quant_high_bits(form)
    if bits == 0:
        return n
    var per = 8 // bits
    var h = Int(p.unsafe_load(hi + i // per))
    n |= ((h >> ((i % per) * bits)) & ((1 << bits) - 1)) << 4
    return n - quant_bias(form)


def repack_row(
    kind: Int, src: RawPtr, at: Int, cols: Int, dst: RawPtr, to: Int
) raises:
    """Turn one row of blocks at `at` into one planar row at `to`.

    The block loops here are the same loops as in `molla.nn.quant`, walked for
    the same bytes in the same order, with the multiply left undone. Keeping
    them looking alike is deliberate. There is no way to write a q6_k unpacker
    that is obviously right, so the next best thing is two of them that are
    obviously the same, and a test that runs both over the corpus and compares.
    """
    var per = block_elements(kind)
    if not repackable(kind) or per == 0:
        raise Error("ggml type " + String(kind) + " has no planar form")
    if cols % per != 0:
        raise Error(
            "a row of "
            + String(cols)
            + " is not a whole number of "
            + String(per)
            + " element blocks"
        )
    var g = group_size(kind)
    var groups = cols // g
    var stride = block_bytes(kind)
    var per_groups = per // g
    var h_at = to + planar_low_bytes(kind, cols)
    var s_at = to + planar_quant_bytes(kind, cols)
    # A group scale is a float16 in a plain form and a byte in a block scaled
    # one, and the two planes sit back to back either way, so one width is all
    # it takes to place both.
    var each = 1 if block_scaled(quant_form(kind)) else SCALE_BYTES
    var d_at = s_at
    var m_at = s_at + groups * each
    # The factor planes, which a plain form has none of and never indexes.
    var planes = 2 if has_min(kind) else 1
    var fd_at = s_at + planes * groups * each
    var fm_at = fd_at + planar_blocks(kind, cols) * SCALE_BYTES
    # The bytes `ROW_ALIGN` rounded the row up by, which no plane covers. The
    # scratch buffer this writes into is reused row after row and is never
    # cleared, so leaving them would put whatever the last row left there into
    # the cache file. Nothing reads them, but a cache that is not a function of
    # the model is one that cannot be compared between two runs.
    var used = planar_quant_bytes(kind, cols) + planar_scale_bytes(kind, cols)
    for i in range(used, planar_row_bytes(kind, cols)):
        dst.unsafe_store(to + i, UInt8(0))
    # `q` counts values and not bytes, because the two stopped being the same
    # number when the four bit types started packing two to a byte. Every
    # packer takes the plane's base separately and adds its own offsets to it.
    for b in range(cols // per):
        var block = at + b * stride
        var q = b * per
        var d = d_at + b * per_groups * each
        var m = m_at + b * per_groups * each
        # One block of the type is one block of the factor planes, because the
        # factor is exactly the float16 the block already had.
        var fd = fd_at + b * SCALE_BYTES
        var fm = fm_at + b * SCALE_BYTES
        if kind == Q_Q4_0:
            _pack_q4_0(src, block, dst, to, q, d)
        elif kind == Q_Q4_1:
            _pack_q4_1(src, block, dst, to, q, d, m)
        elif kind == Q_Q5_0:
            _pack_q5_0(src, block, dst, to, h_at, q, d)
        elif kind == Q_Q5_1:
            _pack_q5_1(src, block, dst, to, h_at, q, d, m)
        elif kind == Q_Q8_0:
            _pack_q8_0(src, block, dst, to, q, d)
        elif kind == Q_Q4_K:
            _pack_q4_k(src, block, dst, to, q, d, m, fd, fm)
        elif kind == Q_Q5_K:
            _pack_q5_k(src, block, dst, to, h_at, q, d, m, fd, fm)
        else:
            _pack_q6_k(src, block, dst, to, h_at, q, d, fd)


def _pack_q4_0(src: RawPtr, at: Int, dst: RawPtr, to: Int, q: Int, d: Int):
    """One scale and thirty two nibbles, centred on the way past.

    Subtracting the eight here is the point of the whole exercise for this type.
    The kernel's q4_0 path carries a `sum(x)` accumulator purely to apply that
    offset at the end of the group, and once the value is centred there is no
    offset left to apply and the accumulator goes away.

    The centred value is -8 to 7, which is four bits, so it goes back into a
    nibble as its own two's complement. The plane is not the file's nibbles
    reordered: ggml puts value `l` and value `l + 16` in one byte and this puts
    value `l` and value `l + 1` in one, because the reader walks values in order
    and the writer is the only place that has to care.
    """
    _put_scale(dst, d, f16_at(src, at))
    for l in range(16):
        var b = _u8(src, at + 2 + l)
        _put_nibble(dst, to, q + l, (b & 0xF) - 8)
        _put_nibble(dst, to, q + l + 16, (b >> 4) - 8)


def _pack_q4_1(
    src: RawPtr, at: Int, dst: RawPtr, to: Int, q: Int, d: Int, m: Int
):
    _put_scale(dst, d, f16_at(src, at))
    _put_scale(dst, m, f16_at(src, at + 2))
    for l in range(16):
        var b = _u8(src, at + 4 + l)
        _put_nibble(dst, to, q + l, b & 0xF)
        _put_nibble(dst, to, q + l + 16, b >> 4)


def _pack_q5_0(
    src: RawPtr, at: Int, dst: RawPtr, to: Int, up: Int, q: Int, d: Int
):
    """Five bits a value, split four and one, with the offset left to the reader.

    The file stores 0 to 31 and the value is that minus sixteen. What goes in
    the planes is the stored 0 to 31 unchanged, and `QUANT_S5` takes the sixteen
    off on the way out. That is not a choice about where to put the subtraction:
    a plane four bits wide and a plane one bit wide have no room for a sign, so
    the offset has to survive the round trip.
    """
    _put_scale(dst, d, f16_at(src, at))
    var qh = _qh(src, at + 2)
    for l in range(16):
        var b = _u8(src, at + 6 + l)
        var hl = ((qh >> l) << 4) & 0x10
        var hh = (qh >> (l + 12)) & 0x10
        _put_quant(QUANT_S5, dst, to, up, q + l, (b & 0xF) | hl)
        _put_quant(QUANT_S5, dst, to, up, q + l + 16, (b >> 4) | hh)


def _pack_q5_1(
    src: RawPtr, at: Int, dst: RawPtr, to: Int, up: Int, q: Int, d: Int, m: Int
):
    _put_scale(dst, d, f16_at(src, at))
    _put_scale(dst, m, f16_at(src, at + 2))
    var qh = _qh(src, at + 4)
    for l in range(16):
        var b = _u8(src, at + 8 + l)
        var hl = ((qh >> l) << 4) & 0x10
        var hh = (qh >> (l + 12)) & 0x10
        _put_quant(QUANT_U5, dst, to, up, q + l, (b & 0xF) | hl)
        _put_quant(QUANT_U5, dst, to, up, q + l + 16, (b >> 4) | hh)


def _pack_q8_0(src: RawPtr, at: Int, dst: RawPtr, to: Int, q: Int, d: Int):
    """The identity transform, near enough.

    q8_0 is already a scale and thirty two signed bytes, so the quants are a
    copy and the only change is the float16 scale widening to a float32 in a
    plane of its own. It is here rather than being skipped because a planar
    tensor that is planar for seven types and ggml for the eighth is a branch in
    every kernel, and 1.06x on the one type that was already the biggest is a
    cheaper thing to carry than that branch.
    """
    _put_scale(dst, d, f16_at(src, at))
    for l in range(32):
        dst.unsafe_store(to + q + l, src.unsafe_load(at + 2 + l))


def _pack_q4_k(
    src: RawPtr,
    at: Int,
    dst: RawPtr,
    to: Int,
    q: Int,
    d: Int,
    m: Int,
    fd: Int,
    fm: Int,
):
    """Eight groups out of one 256 value block, with the two scales split.

    A q4_k value is `d * sc * v - dmin * mn`, so the block's dfactor is `d` and
    its mfactor is `-dmin`, and the group's two bytes are the six bit `sc` and
    `mn` the file has. Folding the negation into the block factor rather than
    keeping a subtract in the kernel is what lets the planar dot product be one
    shape for every type that has a minimum, and folding it there rather than
    into each group's byte is what keeps that byte the file's own integer.

    The group index needs checking rather than assuming, because this loop
    writes in halves of sixty four and the planes are indexed in groups of
    thirty two. Output position `p` is `half * 64 + l` for the low nibble and
    `half * 64 + l + 32` for the high one, so `p // 32` is `half * 2` and
    `half * 2 + 1`, which is exactly the pair of groups this iteration has the
    scales for.
    """
    _put_scale(dst, fd, f16_at(src, at))
    _put_scale(dst, fm, -f16_at(src, at + 2))
    var scales = at + 4
    var qs = at + 16
    for half in range(4):
        var lo = _k_scale(src, scales, half * 2)
        var hi = _k_scale(src, scales, half * 2 + 1)
        _put_scale_byte(dst, d + half * 2, Int(lo[0]))
        _put_scale_byte(dst, m + half * 2, Int(lo[1]))
        _put_scale_byte(dst, d + half * 2 + 1, Int(hi[0]))
        _put_scale_byte(dst, m + half * 2 + 1, Int(hi[1]))
        var b_at = qs + half * 32
        var base = q + half * 64
        for l in range(32):
            var b = _u8(src, b_at + l)
            _put_nibble(dst, to, base + l, b & 0xF)
            _put_nibble(dst, to, base + l + 32, b >> 4)


def _pack_q5_k(
    src: RawPtr,
    at: Int,
    dst: RawPtr,
    to: Int,
    up: Int,
    q: Int,
    d: Int,
    m: Int,
    fd: Int,
    fm: Int,
):
    """q4_k with the fifth bit plane merged into the byte.

    The whole reason q5_k is slow to read is that the fifth bit of a value lives
    in a different plane at a bit position that advances by two every sixty four
    values. Merging it here means the per token cost of that plane is zero.
    """
    _put_scale(dst, fd, f16_at(src, at))
    _put_scale(dst, fm, -f16_at(src, at + 2))
    var scales = at + 4
    var qh = at + 16
    var qs = at + 48
    for half in range(4):
        var lo = _k_scale(src, scales, half * 2)
        var hi = _k_scale(src, scales, half * 2 + 1)
        _put_scale_byte(dst, d + half * 2, Int(lo[0]))
        _put_scale_byte(dst, m + half * 2, Int(lo[1]))
        _put_scale_byte(dst, d + half * 2 + 1, Int(hi[0]))
        _put_scale_byte(dst, m + half * 2 + 1, Int(hi[1]))
        var u1 = 1 << (half * 2)
        var u2 = u1 << 1
        var b_at = qs + half * 32
        var base = q + half * 64
        for l in range(32):
            var b = _u8(src, b_at + l)
            var h = _u8(src, qh + l)
            _put_quant(
                QUANT_K5,
                dst,
                to,
                up,
                base + l,
                (b & 0xF) + (16 if (h & u1) != 0 else 0),
            )
            _put_quant(
                QUANT_K5,
                dst,
                to,
                up,
                base + l + 32,
                (b >> 4) + (16 if (h & u2) != 0 else 0),
            )


def _pack_q6_k(
    src: RawPtr, at: Int, dst: RawPtr, to: Int, up: Int, q: Int, d: Int, fd: Int
):
    """Sixteen groups of sixteen, and the awkward interleave paid off once.

    The four output positions this loop writes are `l`, `l + 32`, `l + 64` and
    `l + 96` within a half, and with a group of sixteen their group indices are
    `l // 16` plus zero, two, four and six, which is the same stride the scale
    bytes are read at in `molla.nn.quant`. That the two agree is why the group
    index can be the plain `i // 16` the kernel assumes.
    """
    _put_scale(dst, fd, f16_at(src, at + 208))
    var ql = at
    var qh = at + 128
    var sc = at + 192
    for gi in range(16):
        _put_scale_byte(dst, d + gi, _i8(src, sc + gi))
    for n in range(2):
        var lo = ql + n * 64
        var hb = qh + n * 32
        var base = q + n * 128
        for l in range(32):
            var h = _u8(src, hb + l)
            _put_quant(
                QUANT_S6,
                dst,
                to,
                up,
                base + l,
                (_u8(src, lo + l) & 0xF) | ((h & 3) << 4),
            )
            _put_quant(
                QUANT_S6,
                dst,
                to,
                up,
                base + l + 32,
                (_u8(src, lo + l + 32) & 0xF) | (((h >> 2) & 3) << 4),
            )
            _put_quant(
                QUANT_S6,
                dst,
                to,
                up,
                base + l + 64,
                (_u8(src, lo + l) >> 4) | (((h >> 4) & 3) << 4),
            )
            _put_quant(
                QUANT_S6,
                dst,
                to,
                up,
                base + l + 96,
                (_u8(src, lo + l + 32) >> 4) | (((h >> 6) & 3) << 4),
            )


def _scale_planes(kind: Int, at: Int, cols: Int) raises -> Tuple[Int, Int]:
    """Where one planar row keeps its dscale plane and its mscale plane.

    Byte offsets into the mapping and not group counts, because the two planes
    are a float16 a group in a plain form and a byte a group in a block scaled
    one and the callers should not have to know which.
    """
    var groups = planar_groups(kind, cols)
    var each = 1 if block_scaled(quant_form(kind)) else SCALE_BYTES
    var d_at = at + planar_quant_bytes(kind, cols)
    return (d_at, d_at + groups * each)


def _factor_planes(kind: Int, at: Int, cols: Int) raises -> Tuple[Int, Int]:
    """Where it keeps the two float16 factor planes, if it has any.

    Zeroes for a plain form, which never asks: `_group_scale` reads a factor
    only where `block_scaled` said there is one.
    """
    if not block_scaled(quant_form(kind)):
        return (0, 0)
    var planes = 2 if has_min(kind) else 1
    var fd_at = (
        at + planar_quant_bytes(kind, cols) + planes * planar_groups(kind, cols)
    )
    return (fd_at, fd_at + planar_blocks(kind, cols) * SCALE_BYTES)


def _group_scale(kind: Int, p: RawPtr, base: Int, fac: Int, gi: Int) -> Float32:
    """One group's scale, out of whichever planes this form keeps it in.

    The float32 multiply is the whole of what a block scaled form costs to
    read, and it happens once a group rather than once a value.
    """
    var form = quant_form(kind)
    if not block_scaled(form):
        return f16_at(p, base + gi * SCALE_BYTES)
    var q = _i8(p, base + gi) if scale_signed(form) else _u8(p, base + gi)
    var b = gi >> block_shift(form)
    return f16_at(p, fac + b * SCALE_BYTES) * Float32(q)


def planar_row_dot(
    kind: Int, p: RawPtr, at: Int, x: List[Float32], to: Int, n: Int
) raises -> Float32:
    """The dot product of one planar row with a float32 vector.

    The counterpart of `molla.nn.kernel.row_dot` and the reason this layout
    exists. Eight per type unpacking loops collapse into two loops that differ
    only in whether there is an mscale term, and the inner one is a signed byte
    widened and multiplied, which is the shape every machine has an instruction
    for.

    The accumulation order matches the fused ggml paths group for group, so the
    two agree closely but not bit for bit, for the same reason those already do
    not agree bit for bit with dequantizing first: the terms are the same and
    the association is not.
    """
    var g = group_size(kind)
    if g == 0:
        raise Error("ggml type " + String(kind) + " has no planar form")
    if n % g != 0:
        raise Error(
            "a row of "
            + String(n)
            + " is not a whole number of "
            + String(g)
            + " value groups"
        )
    var groups = n // g
    var form = quant_form(kind)
    var h_at = at + planar_low_bytes(kind, n)
    var planes = _scale_planes(kind, at, n)
    var facs = _factor_planes(kind, at, n)
    var total = Float32(0)
    if has_min(kind):
        for gi in range(groups):
            var qb = gi * g
            var xb = to + gi * g
            var qx = Float32(0)
            var sx = Float32(0)
            for l in range(g):
                var a = x[xb + l]
                qx += Float32(_quant_at(form, p, at, h_at, qb + l)) * a
                sx += a
            total += _group_scale(kind, p, planes[0], facs[0], gi) * qx
            total += _group_scale(kind, p, planes[1], facs[1], gi) * sx
        return total
    for gi in range(groups):
        var qb = gi * g
        var xb = to + gi * g
        var qx = Float32(0)
        for l in range(g):
            qx += Float32(_quant_at(form, p, at, h_at, qb + l)) * x[xb + l]
        total += _group_scale(kind, p, planes[0], facs[0], gi) * qx
    return total


def planar_run(
    kind: Int, p: RawPtr, at: Int, count: Int, mut out: List[Float32], to: Int
) raises:
    """Decode a whole planar row back to float32.

    Not part of a forward pass any more than `molla.nn.quant.dequant_run` is,
    with the one exception that is the embedding lookup: reading a token's row
    out of `token_embd` is a copy and not a dot product, and it has to work
    whether that tensor was repacked or not. Everything else that calls this is
    a test.
    """
    var g = group_size(kind)
    if g == 0:
        raise Error("ggml type " + String(kind) + " has no planar form")
    if count % g != 0:
        raise Error(
            "asked for "
            + String(count)
            + " values, which is not a whole number of "
            + String(g)
            + " value groups"
        )
    if to + count > len(out):
        raise Error("unpack would write past the end of the output")
    var groups = count // g
    var form = quant_form(kind)
    var h_at = at + planar_low_bytes(kind, count)
    var planes = _scale_planes(kind, at, count)
    var facs = _factor_planes(kind, at, count)
    var carries_min = has_min(kind)
    for gi in range(groups):
        var d = _group_scale(kind, p, planes[0], facs[0], gi)
        var m = _group_scale(
            kind, p, planes[1], facs[1], gi
        ) if carries_min else Float32(0)
        var qb = gi * g
        var ob = to + gi * g
        for l in range(g):
            out[ob + l] = d * Float32(_quant_at(form, p, at, h_at, qb + l)) + m


def unpack_run(
    kind: Int,
    layout: Int,
    p: RawPtr,
    at: Int,
    count: Int,
    mut out: List[Float32],
    to: Int,
) raises:
    """`dequant_run` or `planar_run`, whichever the tensor is in.

    One place that knows the answer, so a caller holding a tensor does not have
    to. Every kernel that reads a whole run rather than dotting it goes through
    here.
    """
    if layout == LAYOUT_PLANAR:
        planar_run(kind, p, at, count, out, to)
        return
    dequant_run(kind, p, at, count, out, to)
