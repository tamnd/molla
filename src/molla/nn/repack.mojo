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

    row = [ cols quants ][ groups float32 dscale ][ groups float32 mscale ]

with the mscale plane there only for the types that carry a minimum. A value is
`dscale[g] * q[i] + mscale[g]` where `g` is `i // group_size`, and that is the
whole format. The group is 32 values for everything except q6_k, which is 16.

A quant is four bits for the three four bit types and a signed byte for the
rest, which `quant_form` decides and nothing else does. The four bit plane is
two values a byte, low nibble first in the order the values appear, so position
`i` is in byte `i // 2` and in the low half when `i` is even.

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

What the layout still spends is the scale planes, which are float32 where the
file's are float16, and that is 2 bits a value against 0.5 in the file. Whether
that is worth narrowing is its own question and its own change.

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
    f32_at,
)
from molla.sys.mmap import RawPtr

comptime LAYOUT_GGML = 0
"""The bytes as the file has them. What a tensor is until it is repacked."""

comptime LAYOUT_PLANAR = 1
"""Quants in one plane and scales in another, as described above."""

comptime QUANT_I8 = 0
"""One signed byte a value. q5_0, q5_1, q5_k, q6_k and q8_0."""

comptime QUANT_U4 = 1
"""Two unsigned nibbles a byte, values 0 to 15. q4_1 and q4_k."""

comptime QUANT_S4 = 2
"""Two two's complement nibbles a byte, values -8 to 7. q4_0 alone.

Kept apart from `QUANT_U4` rather than shifted into it, even though q4_1 and
q4_k both have an mscale plane that could absorb an offset of eight. Folding it
there would change the terms the dot product sums, which changes the last bits
of every result, and there is no reason to spend that on a change whose whole
point is that the numbers do not move.
"""

comptime LAYOUT_VERSION = 2
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
    if kind == Q_Q4_1 or kind == Q_Q4_K:
        return QUANT_U4
    if kind == Q_Q4_0:
        return QUANT_S4
    return QUANT_I8


def quant_bits(kind: Int) -> Int:
    """Bits a value occupies in the quant plane, 4 or 8."""
    return 8 if quant_form(kind) == QUANT_I8 else 4


def planar_quant_bytes(kind: Int, cols: Int) raises -> Int:
    """Bytes the quant plane of one row occupies.

    A whole number for every type and row width molla accepts, because a four
    bit type packs 32 or 256 values to a block and a row is a whole number of
    blocks, so `cols` is even long before it gets here. Checked anyway, because
    the alternative to checking is a row width that silently drops its last
    value.
    """
    if not repackable(kind):
        raise Error("ggml type " + String(kind) + " has no planar form")
    var bits = quant_bits(kind)
    if bits == 8:
        return cols
    if cols % 2 != 0:
        raise Error(
            "a row of "
            + String(cols)
            + " cannot be packed two values to a byte"
        )
    return cols // 2


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


def planar_row_bytes(kind: Int, cols: Int) raises -> Int:
    """Bytes one planar row occupies.

    A multiple of four, because the quant plane is and the scale planes are
    whole float32s. That matters more than it looks: rows sit back to back, so
    if one row were not a multiple of four then every row after the first would
    put its scale planes at an address the loads cannot be aligned to.

    The quant plane is a multiple of four for both widths. At a byte a value it
    is `cols`, which is a multiple of the group size and so of 16. At a nibble
    it is `cols // 2`, which is a multiple of 8, because a four bit type blocks
    by 32 or 256 and a row is whole blocks.
    """
    var groups = planar_groups(kind, cols)
    var planes = 2 if has_min(kind) else 1
    return planar_quant_bytes(kind, cols) + planes * groups * 4


def _put_f32(p: RawPtr, at: Int, value: Float32):
    """A little endian float32, byte at a time.

    The mirror of `molla.nn.quant.f32_at` and written the same way for the same
    reason: the destination is a byte address in a mapping, a store through a
    reinterpreted pointer would be an alignment assumption this code has not
    earned, and this runs once per group at repack time rather than per token.
    """
    var bits = bitcast[DType.uint32, 1](value)
    p.unsafe_store(at, UInt8(bits & 0xFF))
    p.unsafe_store(at + 1, UInt8((bits >> 8) & 0xFF))
    p.unsafe_store(at + 2, UInt8((bits >> 16) & 0xFF))
    p.unsafe_store(at + 3, UInt8((bits >> 24) & 0xFF))


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


def _put_quant(form: Int, p: RawPtr, base: Int, i: Int, value: Int):
    """Value `i` of a quant plane, at whichever width the type uses."""
    if form == QUANT_I8:
        _put_i8(p, base + i, value)
        return
    _put_nibble(p, base, i, value)


def _quant_at(form: Int, p: RawPtr, base: Int, i: Int) -> Int:
    """Value `i` of a quant plane, back out.

    The sign extension for `QUANT_S4` is subtracting sixteen and not eight. The
    stored nibble is the value's low four bits in two's complement, so -8 is
    stored as 8 and -1 as 15, and it is the whole four bit word that wraps.
    """
    if form == QUANT_I8:
        return _i8(p, base + i)
    var b = Int(p.unsafe_load(base + (i >> 1)))
    var n = (b >> 4) if (i & 1) != 0 else (b & 0xF)
    if form == QUANT_S4 and n >= 8:
        return n - 16
    return n


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
    var d_at = to + planar_quant_bytes(kind, cols)
    var m_at = d_at + groups * 4
    # `q` counts values and not bytes, because the two stopped being the same
    # number when the four bit types started packing two to a byte. Every
    # packer takes the plane's base separately and adds its own offsets to it.
    for b in range(cols // per):
        var block = at + b * stride
        var q = b * per
        var d = d_at + b * per_groups * 4
        var m = m_at + b * per_groups * 4
        if kind == Q_Q4_0:
            _pack_q4_0(src, block, dst, to, q, d)
        elif kind == Q_Q4_1:
            _pack_q4_1(src, block, dst, to, q, d, m)
        elif kind == Q_Q5_0:
            _pack_q5_0(src, block, dst, to, q, d)
        elif kind == Q_Q5_1:
            _pack_q5_1(src, block, dst, to, q, d, m)
        elif kind == Q_Q8_0:
            _pack_q8_0(src, block, dst, to, q, d)
        elif kind == Q_Q4_K:
            _pack_q4_k(src, block, dst, to, q, d, m)
        elif kind == Q_Q5_K:
            _pack_q5_k(src, block, dst, to, q, d, m)
        else:
            _pack_q6_k(src, block, dst, to, q, d)


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
    _put_f32(dst, d, f16_at(src, at))
    for l in range(16):
        var b = _u8(src, at + 2 + l)
        _put_nibble(dst, to, q + l, (b & 0xF) - 8)
        _put_nibble(dst, to, q + l + 16, (b >> 4) - 8)


def _pack_q4_1(
    src: RawPtr, at: Int, dst: RawPtr, to: Int, q: Int, d: Int, m: Int
):
    _put_f32(dst, d, f16_at(src, at))
    _put_f32(dst, m, f16_at(src, at + 2))
    for l in range(16):
        var b = _u8(src, at + 4 + l)
        _put_nibble(dst, to, q + l, b & 0xF)
        _put_nibble(dst, to, q + l + 16, b >> 4)


def _pack_q5_0(src: RawPtr, at: Int, dst: RawPtr, to: Int, q: Int, d: Int):
    _put_f32(dst, d, f16_at(src, at))
    var qh = _qh(src, at + 2)
    for l in range(16):
        var b = _u8(src, at + 6 + l)
        var hl = ((qh >> l) << 4) & 0x10
        var hh = (qh >> (l + 12)) & 0x10
        _put_i8(dst, to + q + l, ((b & 0xF) | hl) - 16)
        _put_i8(dst, to + q + l + 16, ((b >> 4) | hh) - 16)


def _pack_q5_1(
    src: RawPtr, at: Int, dst: RawPtr, to: Int, q: Int, d: Int, m: Int
):
    _put_f32(dst, d, f16_at(src, at))
    _put_f32(dst, m, f16_at(src, at + 2))
    var qh = _qh(src, at + 4)
    for l in range(16):
        var b = _u8(src, at + 8 + l)
        var hl = ((qh >> l) << 4) & 0x10
        var hh = (qh >> (l + 12)) & 0x10
        _put_i8(dst, to + q + l, (b & 0xF) | hl)
        _put_i8(dst, to + q + l + 16, (b >> 4) | hh)


def _pack_q8_0(src: RawPtr, at: Int, dst: RawPtr, to: Int, q: Int, d: Int):
    """The identity transform, near enough.

    q8_0 is already a scale and thirty two signed bytes, so the quants are a
    copy and the only change is the float16 scale widening to a float32 in a
    plane of its own. It is here rather than being skipped because a planar
    tensor that is planar for seven types and ggml for the eighth is a branch in
    every kernel, and 1.06x on the one type that was already the biggest is a
    cheaper thing to carry than that branch.
    """
    _put_f32(dst, d, f16_at(src, at))
    for l in range(32):
        dst.unsafe_store(to + q + l, src.unsafe_load(at + 2 + l))


def _pack_q4_k(
    src: RawPtr, at: Int, dst: RawPtr, to: Int, q: Int, d: Int, m: Int
):
    """Eight groups out of one 256 value block, with the two scales folded.

    A q4_k value is `d * sc * v - dmin * mn`, so the group's dscale is `d * sc`
    and its mscale is `-(dmin * mn)`. Folding the negation into the stored
    number rather than keeping a subtract in the kernel is what lets the planar
    dot product be one shape for every type that has a minimum.

    The group index needs checking rather than assuming, because this loop
    writes in halves of sixty four and the planes are indexed in groups of
    thirty two. Output position `p` is `half * 64 + l` for the low nibble and
    `half * 64 + l + 32` for the high one, so `p // 32` is `half * 2` and
    `half * 2 + 1`, which is exactly the pair of groups this iteration has the
    scales for.
    """
    var d0 = f16_at(src, at)
    var dmin = f16_at(src, at + 2)
    var scales = at + 4
    var qs = at + 16
    for half in range(4):
        var lo = _k_scale(src, scales, half * 2)
        var hi = _k_scale(src, scales, half * 2 + 1)
        _put_f32(dst, d + half * 8, d0 * lo[0])
        _put_f32(dst, m + half * 8, -(dmin * lo[1]))
        _put_f32(dst, d + half * 8 + 4, d0 * hi[0])
        _put_f32(dst, m + half * 8 + 4, -(dmin * hi[1]))
        var b_at = qs + half * 32
        var base = q + half * 64
        for l in range(32):
            var b = _u8(src, b_at + l)
            _put_nibble(dst, to, base + l, b & 0xF)
            _put_nibble(dst, to, base + l + 32, b >> 4)


def _pack_q5_k(
    src: RawPtr, at: Int, dst: RawPtr, to: Int, q: Int, d: Int, m: Int
):
    """q4_k with the fifth bit plane merged into the byte.

    The whole reason q5_k is slow to read is that the fifth bit of a value lives
    in a different plane at a bit position that advances by two every sixty four
    values. Merging it here means the per token cost of that plane is zero.
    """
    var d0 = f16_at(src, at)
    var dmin = f16_at(src, at + 2)
    var scales = at + 4
    var qh = at + 16
    var qs = at + 48
    for half in range(4):
        var lo = _k_scale(src, scales, half * 2)
        var hi = _k_scale(src, scales, half * 2 + 1)
        _put_f32(dst, d + half * 8, d0 * lo[0])
        _put_f32(dst, m + half * 8, -(dmin * lo[1]))
        _put_f32(dst, d + half * 8 + 4, d0 * hi[0])
        _put_f32(dst, m + half * 8 + 4, -(dmin * hi[1]))
        var u1 = 1 << (half * 2)
        var u2 = u1 << 1
        var b_at = qs + half * 32
        var base = q + half * 64
        for l in range(32):
            var b = _u8(src, b_at + l)
            var h = _u8(src, qh + l)
            _put_i8(
                dst, to + base + l, (b & 0xF) + (16 if (h & u1) != 0 else 0)
            )
            _put_i8(
                dst,
                to + base + l + 32,
                (b >> 4) + (16 if (h & u2) != 0 else 0),
            )


def _pack_q6_k(src: RawPtr, at: Int, dst: RawPtr, to: Int, q: Int, d: Int):
    """Sixteen groups of sixteen, and the awkward interleave paid off once.

    The four output positions this loop writes are `l`, `l + 32`, `l + 64` and
    `l + 96` within a half, and with a group of sixteen their group indices are
    `l // 16` plus zero, two, four and six, which is the same stride the scale
    bytes are read at in `molla.nn.quant`. That the two agree is why the group
    index can be the plain `i // 16` the kernel assumes.
    """
    var d0 = f16_at(src, at + 208)
    var ql = at
    var qh = at + 128
    var sc = at + 192
    for gi in range(16):
        _put_f32(dst, d + gi * 4, d0 * Float32(_i8(src, sc + gi)))
    for n in range(2):
        var lo = ql + n * 64
        var hi = qh + n * 32
        var base = to + q + n * 128
        for l in range(32):
            var h = _u8(src, hi + l)
            _put_i8(
                dst, base + l, ((_u8(src, lo + l) & 0xF) | ((h & 3) << 4)) - 32
            )
            _put_i8(
                dst,
                base + l + 32,
                ((_u8(src, lo + l + 32) & 0xF) | (((h >> 2) & 3) << 4)) - 32,
            )
            _put_i8(
                dst,
                base + l + 64,
                ((_u8(src, lo + l) >> 4) | (((h >> 4) & 3) << 4)) - 32,
            )
            _put_i8(
                dst,
                base + l + 96,
                ((_u8(src, lo + l + 32) >> 4) | (((h >> 6) & 3) << 4)) - 32,
            )


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
    var d_at = at + planar_quant_bytes(kind, n)
    var m_at = d_at + groups * 4
    var total = Float32(0)
    if has_min(kind):
        for gi in range(groups):
            var qb = gi * g
            var xb = to + gi * g
            var qx = Float32(0)
            var sx = Float32(0)
            for l in range(g):
                var a = x[xb + l]
                qx += Float32(_quant_at(form, p, at, qb + l)) * a
                sx += a
            total += f32_at(p, d_at + gi * 4) * qx
            total += f32_at(p, m_at + gi * 4) * sx
        return total
    for gi in range(groups):
        var qb = gi * g
        var xb = to + gi * g
        var qx = Float32(0)
        for l in range(g):
            qx += Float32(_quant_at(form, p, at, qb + l)) * x[xb + l]
        total += f32_at(p, d_at + gi * 4) * qx
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
    var d_at = at + planar_quant_bytes(kind, count)
    var m_at = d_at + groups * 4
    var carries_min = has_min(kind)
    for gi in range(groups):
        var d = f32_at(p, d_at + gi * 4)
        var m = f32_at(p, m_at + gi * 4) if carries_min else Float32(0)
        var qb = gi * g
        var ob = to + gi * g
        for l in range(g):
            out[ob + l] = d * Float32(_quant_at(form, p, at, qb + l)) + m


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
