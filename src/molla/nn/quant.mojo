"""Turning ggml blocks back into float32.

Every weight in a quantized model arrives as a block: a small group of values
sharing one or two scales, packed into fewer bytes than the values would take.
Reading one is bit unpacking and nothing more, but the formats are undocumented
outside the C that writes them, and each one has at least one place where the
obvious reading is wrong. The six digit K quant scale, packed six bits at a time
across twelve bytes, is the worst of them.

This module is the slow, obvious version: one block in, a run of float32 out.
Nothing in a forward pass should call it on a whole tensor, because dequantizing
a 4 GB model into float32 is 15 GB of host memory to do arithmetic that could
have been done against the packed bytes. What it is for is being right. The
fused kernels that read a block and multiply in the same pass are checked
against this, and this is checked against the `gguf` package, so a fast path
that drifts has something to fail against that is not a model quietly getting
worse.

The layouts are ggml's `block_q*` structs. Byte for byte:

    q4_0    d:f16, qs[16]                                 18 bytes,  32 values
    q4_1    d:f16, m:f16, qs[16]                          20 bytes,  32 values
    q5_0    d:f16, qh:u32, qs[16]                         22 bytes,  32 values
    q5_1    d:f16, m:f16, qh:u32, qs[16]                  24 bytes,  32 values
    q8_0    d:f16, qs[32]:i8                              34 bytes,  32 values
    q4_k    d:f16, dmin:f16, scales[12], qs[128]         144 bytes, 256 values
    q5_k    d:f16, dmin:f16, scales[12], qh[32], qs[128] 176 bytes, 256 values
    q6_k    ql[128], qh[64], scales[16]:i8, d:f16        210 bytes, 256 values

Note that q6_k puts its scale last and the others put it first. There is no
reason for that beyond the order the formats were written in, and a decoder
that assumes the scale is at offset zero reads sixteen signed scales as one
float16 and produces numbers that are wrong without being obviously wrong.

The trailing digit is not a version. A format ending in zero centres its
integers, so a q4_0 nibble of eight is zero and the range is symmetric about it.
A format ending in one carries a minimum instead and its integers run from zero
upwards, so a nibble of zero is the minimum rather than the bottom of a
symmetric range. Reading a q4_1 as a centred format gives every weight in the
tensor an offset of eight scales, which is a model that still produces words.
"""

from std.memory import bitcast

from molla.sys.mmap import RawPtr

comptime Q_F32 = 0
comptime Q_F16 = 1
comptime Q_Q4_0 = 2
comptime Q_Q4_1 = 3
comptime Q_Q5_0 = 6
comptime Q_Q5_1 = 7
comptime Q_Q8_0 = 8
comptime Q_Q4_K = 12
comptime Q_Q5_K = 13
comptime Q_Q6_K = 14
comptime Q_BF16 = 30
"""ggml type numbers, which are what a GGUF tensor directory stores. The same
numbers `molla.model.spec.encoding_of` knows the block geometry for."""

comptime QK_K = 256
"""Elements per block in every K quant. ggml calls it QK_K and so does this."""


def supported(kind: Int) -> Bool:
    """Whether `dequant_block` can read this type.

    Everything a Llama 3 or Qwen 3 q4_K_M file contains is here, plus the
    formats next to them that cost nothing to add. The IQ types and the sub four
    bit K quants are not, because each one needs its own lookup tables and none
    of them appears in a model molla runs today. `Gguf` can still read the
    directory of a file full of them and `molla spec` will still say what it is,
    which is the point of keeping the geometry table and the decoder separate.
    """
    if kind == Q_F32 or kind == Q_F16 or kind == Q_BF16:
        return True
    if kind == Q_Q4_0 or kind == Q_Q4_1 or kind == Q_Q8_0:
        return True
    if kind == Q_Q5_0 or kind == Q_Q5_1:
        return True
    return kind == Q_Q4_K or kind == Q_Q5_K or kind == Q_Q6_K


def block_elements(kind: Int) -> Int:
    """Values per block, or zero for a type this module cannot read."""
    if kind == Q_F32 or kind == Q_F16 or kind == Q_BF16:
        return 1
    if kind == Q_Q4_0 or kind == Q_Q4_1 or kind == Q_Q8_0:
        return 32
    if kind == Q_Q5_0 or kind == Q_Q5_1:
        return 32
    if kind == Q_Q4_K or kind == Q_Q5_K or kind == Q_Q6_K:
        return QK_K
    return 0


def block_bytes(kind: Int) -> Int:
    """Bytes per block, or zero for a type this module cannot read."""
    if kind == Q_F32:
        return 4
    if kind == Q_F16 or kind == Q_BF16:
        return 2
    if kind == Q_Q4_0:
        return 18
    if kind == Q_Q4_1:
        return 20
    if kind == Q_Q5_0:
        return 22
    if kind == Q_Q5_1:
        return 24
    if kind == Q_Q8_0:
        return 34
    if kind == Q_Q4_K:
        return 144
    if kind == Q_Q5_K:
        return 176
    if kind == Q_Q6_K:
        return 210
    return 0


def _u8(p: RawPtr, at: Int) -> Int:
    return Int(p.unsafe_load(at))


def _i8(p: RawPtr, at: Int) -> Int:
    """A byte read as signed, which is what ggml stores q8_0 and q6_k scales
    as."""
    var v = Int(p.unsafe_load(at))
    if v >= 128:
        return v - 256
    return v


def f16_at(p: RawPtr, at: Int) -> Float32:
    """A little endian float16 at this offset, widened to float32."""
    var bits = UInt16(p.unsafe_load(at)) | (UInt16(p.unsafe_load(at + 1)) << 8)
    return bitcast[DType.float16, 1](bits).cast[DType.float32]()


def bf16_at(p: RawPtr, at: Int) -> Float32:
    """A little endian bfloat16 at this offset, widened to float32.

    A bfloat16 is the top sixteen bits of a float32, so this is a shift rather
    than a conversion. Going through `BFloat16` would work too and would be one
    line shorter, but it is a software conversion on every target we build for
    and this is called once per element on an unquantized tensor.
    """
    var bits = UInt32(p.unsafe_load(at)) | (UInt32(p.unsafe_load(at + 1)) << 8)
    return bitcast[DType.float32, 1](bits << 16)


def f32_at(p: RawPtr, at: Int) -> Float32:
    var bits = UInt32(p.unsafe_load(at))
    bits |= UInt32(p.unsafe_load(at + 1)) << 8
    bits |= UInt32(p.unsafe_load(at + 2)) << 16
    bits |= UInt32(p.unsafe_load(at + 3)) << 24
    return bitcast[DType.float32, 1](bits)


def _k_scale(p: RawPtr, at: Int, j: Int) -> Tuple[Float32, Float32]:
    """One of eight scale and minimum pairs out of a twelve byte K quant field.

    This is ggml's `get_scale_min_k4` and it is the single densest piece of bit
    packing in any of these formats. Eight scales and eight minimums, six bits
    each, is 96 bits, and they are stored in twelve bytes with no padding. The
    first four pairs are the low six bits of bytes 0 to 3 and 4 to 7. The last
    four are the low four bits of bytes 8 to 11 for the low half, with the high
    two bits borrowed from the top of the first eight bytes.

    Both come back unsigned and are multiplied by the block scale and the block
    minimum by the caller.
    """
    if j < 4:
        return (
            Float32(_u8(p, at + j) & 63),
            Float32(_u8(p, at + j + 4) & 63),
        )
    var d = (_u8(p, at + j + 4) & 0xF) | ((_u8(p, at + j - 4) >> 6) << 4)
    var m = (_u8(p, at + j + 4) >> 4) | ((_u8(p, at + j) >> 6) << 4)
    return (Float32(d), Float32(m))


def dequant_block(
    kind: Int, p: RawPtr, at: Int, mut out: List[Float32], to: Int
) raises:
    """Decode one block at `at` into `out` starting at `to`.

    `out` is written into rather than appended to, so a caller decoding a row
    sizes the list once and hands the same one to every block. Appending here
    would grow the list a block at a time and reallocate through the whole row.
    """
    if to + block_elements(kind) > len(out):
        raise Error("dequant would write past the end of the output")

    if kind == Q_F32:
        out[to] = f32_at(p, at)
        return
    if kind == Q_F16:
        out[to] = f16_at(p, at)
        return
    if kind == Q_BF16:
        out[to] = bf16_at(p, at)
        return
    if kind == Q_Q4_0:
        _q4_0(p, at, out, to)
        return
    if kind == Q_Q4_1:
        _q4_1(p, at, out, to)
        return
    if kind == Q_Q5_0:
        _q5_0(p, at, out, to)
        return
    if kind == Q_Q5_1:
        _q5_1(p, at, out, to)
        return
    if kind == Q_Q8_0:
        _q8_0(p, at, out, to)
        return
    if kind == Q_Q4_K:
        _q4_k(p, at, out, to)
        return
    if kind == Q_Q5_K:
        _q5_k(p, at, out, to)
        return
    if kind == Q_Q6_K:
        _q6_k(p, at, out, to)
        return
    raise Error("ggml type " + String(kind) + " has no dequantizer")


def _q4_0(p: RawPtr, at: Int, mut out: List[Float32], to: Int):
    """Sixteen bytes of nibble pairs and one scale, centred on eight.

    The two nibbles of a byte are not neighbours in the output. The low nibble
    of byte l is element l and the high nibble is element l plus sixteen, which
    puts the two halves of the block sixteen apart rather than interleaved. Get
    that backwards and every value is still in range and the model still runs.
    """
    var d = f16_at(p, at)
    for l in range(16):
        var b = _u8(p, at + 2 + l)
        out[to + l] = Float32((b & 0xF) - 8) * d
        out[to + l + 16] = Float32((b >> 4) - 8) * d


def _q4_1(p: RawPtr, at: Int, mut out: List[Float32], to: Int):
    """The same nibbles as q4_0 with a minimum instead of a fixed centre.

    q4_0 subtracts eight from every value, which assumes the weights in a block
    are spread evenly either side of zero. q4_1 stores the minimum the block
    actually had, so a block whose values are all positive does not spend half
    its range on numbers that are not there. Two bytes more per thirty two
    values, and the nibbles are still sixteen apart in the output.
    """
    var d = f16_at(p, at)
    var m = f16_at(p, at + 2)
    for l in range(16):
        var b = _u8(p, at + 4 + l)
        out[to + l] = Float32(b & 0xF) * d + m
        out[to + l + 16] = Float32(b >> 4) * d + m


def _qh(p: RawPtr, at: Int) -> Int:
    """The thirty two bit field of fifth bits, little endian."""
    var out = 0
    for i in range(4):
        out |= _u8(p, at + i) << (i * 8)
    return out


def _q5_0(p: RawPtr, at: Int, mut out: List[Float32], to: Int):
    """Four bits per value in a nibble and the fifth somewhere else entirely.

    A five bit value does not divide a byte, so ggml keeps the low four bits
    where q4_0 keeps them and gathers the thirty two fifth bits into one word
    at the front of the block. Bit `l` of that word belongs to element `l` and
    bit `l + 16` belongs to element `l + 16`, which matches the nibble order
    rather than fighting it.
    """
    var d = f16_at(p, at)
    var qh = _qh(p, at + 2)
    for l in range(16):
        var b = _u8(p, at + 6 + l)
        var hl = ((qh >> l) << 4) & 0x10
        var hh = (qh >> (l + 12)) & 0x10
        out[to + l] = Float32(((b & 0xF) | hl) - 16) * d
        out[to + l + 16] = Float32(((b >> 4) | hh) - 16) * d


def _q5_1(p: RawPtr, at: Int, mut out: List[Float32], to: Int):
    """q5_0's bit layout with q4_1's minimum, so nothing is centred."""
    var d = f16_at(p, at)
    var m = f16_at(p, at + 2)
    var qh = _qh(p, at + 4)
    for l in range(16):
        var b = _u8(p, at + 8 + l)
        var hl = ((qh >> l) << 4) & 0x10
        var hh = (qh >> (l + 12)) & 0x10
        out[to + l] = Float32((b & 0xF) | hl) * d + m
        out[to + l + 16] = Float32((b >> 4) | hh) * d + m


def _q8_0(p: RawPtr, at: Int, mut out: List[Float32], to: Int):
    var d = f16_at(p, at)
    for l in range(32):
        out[to + l] = Float32(_i8(p, at + 2 + l)) * d


def _q4_k(p: RawPtr, at: Int, mut out: List[Float32], to: Int):
    """Four bit values with a per block scale and a per block minimum.

    Two levels of scaling. The block has one float16 scale and one float16
    minimum, and then each group of thirty two values has its own six bit
    multiplier for each of those. So a value is `d * sc * q - dmin * m`, which
    is an affine map and not just a scale, and dropping the minimum term gives
    numbers that look plausible and are shifted.
    """
    var d = f16_at(p, at)
    var dmin = f16_at(p, at + 2)
    var scales = at + 4
    var qs = at + 16
    for half in range(4):
        var lo = _k_scale(p, scales, half * 2)
        var hi = _k_scale(p, scales, half * 2 + 1)
        var d1 = d * lo[0]
        var m1 = dmin * lo[1]
        var d2 = d * hi[0]
        var m2 = dmin * hi[1]
        var q = qs + half * 32
        var base = to + half * 64
        for l in range(32):
            var b = _u8(p, q + l)
            out[base + l] = d1 * Float32(b & 0xF) - m1
            out[base + l + 32] = d2 * Float32(b >> 4) - m2


def _q5_k(p: RawPtr, at: Int, mut out: List[Float32], to: Int):
    """q4_k with a fifth bit for each value kept in a separate plane.

    The extra bit is not next to the other four. It lives in a thirty two byte
    plane where byte l holds the fifth bit of eight different values, one per
    bit position, and the position advances by two for every sixty four values.
    That is why the mask is a shifting `1 << (half * 2)` rather than an index.
    """
    var d = f16_at(p, at)
    var dmin = f16_at(p, at + 2)
    var scales = at + 4
    var qh = at + 16
    var qs = at + 48
    for half in range(4):
        var lo = _k_scale(p, scales, half * 2)
        var hi = _k_scale(p, scales, half * 2 + 1)
        var d1 = d * lo[0]
        var m1 = dmin * lo[1]
        var d2 = d * hi[0]
        var m2 = dmin * hi[1]
        var u1 = 1 << (half * 2)
        var u2 = u1 << 1
        var q = qs + half * 32
        var base = to + half * 64
        for l in range(32):
            var b = _u8(p, q + l)
            var h = _u8(p, qh + l)
            var v1 = (b & 0xF) + (16 if (h & u1) != 0 else 0)
            var v2 = (b >> 4) + (16 if (h & u2) != 0 else 0)
            out[base + l] = d1 * Float32(v1) - m1
            out[base + l + 32] = d2 * Float32(v2) - m2


def _q6_k(p: RawPtr, at: Int, mut out: List[Float32], to: Int):
    """Six bits per value, four low in one plane and two high in another.

    The interleave here is the awkward one. Within each run of 128 values the
    output positions are l, l+32, l+64 and l+96 for a single l, and the four
    values at those positions come from two different bytes of the low plane
    and two different bit pairs of the same byte of the high plane. Writing it
    as four separate loops over l gives the same answer and reads the high plane
    four times, so it is one loop that writes four places.

    The scales are signed here and unsigned in q4_k and q5_k, and there is no
    minimum. A q6_k value is a pure scale of a number centred on thirty two.
    """
    var d = f16_at(p, at + 208)
    var ql = at
    var qh = at + 128
    var sc = at + 192
    for n in range(2):
        var lo = ql + n * 64
        var hi = qh + n * 32
        var scale = sc + n * 8
        var base = to + n * 128
        for l in range(32):
            var group = l // 16
            var h = _u8(p, hi + l)
            var q1 = ((_u8(p, lo + l) & 0xF) | (((h >> 0) & 3) << 4)) - 32
            var q2 = ((_u8(p, lo + l + 32) & 0xF) | (((h >> 2) & 3) << 4)) - 32
            var q3 = ((_u8(p, lo + l) >> 4) | (((h >> 4) & 3) << 4)) - 32
            var q4 = ((_u8(p, lo + l + 32) >> 4) | (((h >> 6) & 3) << 4)) - 32
            out[base + l] = d * Float32(_i8(p, scale + group)) * Float32(q1)
            out[base + l + 32] = (
                d * Float32(_i8(p, scale + group + 2)) * Float32(q2)
            )
            out[base + l + 64] = (
                d * Float32(_i8(p, scale + group + 4)) * Float32(q3)
            )
            out[base + l + 96] = (
                d * Float32(_i8(p, scale + group + 6)) * Float32(q4)
            )


def dequant_run(
    kind: Int, p: RawPtr, at: Int, count: Int, mut out: List[Float32], to: Int
) raises:
    """Decode `count` consecutive values starting at byte `at`.

    `count` has to be a whole number of blocks. A partial block is not something
    ggml can write, and rounding down here would silently return fewer values
    than the caller sized its buffer for.
    """
    var per = block_elements(kind)
    if per == 0:
        raise Error("ggml type " + String(kind) + " has no dequantizer")
    if count % per != 0:
        raise Error(
            "asked for "
            + String(count)
            + " values, which is not a whole number of "
            + String(per)
            + " element blocks"
        )
    var stride = block_bytes(kind)
    var blocks = count // per
    for i in range(blocks):
        dequant_block(kind, p, at + i * stride, out, to + i * per)
