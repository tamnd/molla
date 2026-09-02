"""Tests for the ggml block decoders.

What proves these decoders right is `scripts/quant_oracle.mojo`, which runs them
over half a million values of random block bytes and compares every one against
the `gguf` package. That needs numpy and a Python interpreter, so it is a
conformance job rather than a suite check, and it is written up in
`docs/validation/quant.md`.

What is here is the part that does not need an oracle: the geometry table, the
error paths, and a handful of blocks built by hand so that the expected value is
something a person worked out rather than something another program said. The
hand built ones are chosen to pin the two mistakes that an oracle would catch
but that nothing else would explain, which are where a nibble lands in the
output and whether a scale is signed.
"""

from harness import Suite

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
    bf16_at,
    block_bytes,
    block_elements,
    dequant_block,
    dequant_run,
    f16_at,
    f32_at,
    supported,
)
from molla.sys.mmap import RawPtr


def _ptr(ref bytes: List[UInt8]) -> RawPtr:
    return RawPtr(unsafe_from_address=Int(bytes.unsafe_ptr()))


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


def _bytes(n: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for _ in range(n):
        out.append(0)
    return out^


def _put16(mut b: List[UInt8], at: Int, value: Int):
    b[at] = UInt8(value & 0xFF)
    b[at + 1] = UInt8((value >> 8) & 0xFF)


comptime HALF_ONE = 0x3C00
"""1.0 as a float16, which is the scale every hand built block below uses so
that the decoded value is the packed integer and nothing else."""


def run(mut suite: Suite) raises:
    _check_geometry(suite)
    _check_scalars(suite)
    _check_q4_0(suite)
    _check_q4_1(suite)
    _check_q5_0(suite)
    _check_q5_1(suite)
    _check_q8_0(suite)
    _check_q4_k(suite)
    _check_q5_k(suite)
    _check_q6_k(suite)
    _check_errors(suite)


def _check_geometry(mut suite: Suite) raises:
    suite.group("quant geometry")

    suite.check(block_bytes(Q_F32) == 4, "an f32 is four bytes for one value")
    suite.check(block_elements(Q_F32) == 1, "and is its own block")
    suite.check(block_bytes(Q_F16) == 2, "an f16 is two")
    suite.check(block_bytes(Q_BF16) == 2, "and so is a bf16")
    suite.check(
        block_bytes(Q_Q4_0) == 18 and block_elements(Q_Q4_0) == 32,
        "q4_0 packs 32 values into 18 bytes",
    )
    suite.check(
        block_bytes(Q_Q4_1) == 20 and block_elements(Q_Q4_1) == 32,
        "q4_1 is the same 32 in 20, two more bytes for the minimum",
    )
    suite.check(
        block_bytes(Q_Q5_0) == 22 and block_elements(Q_Q5_0) == 32,
        "q5_0 is 22, four of them the fifth bits",
    )
    suite.check(
        block_bytes(Q_Q5_1) == 24 and block_elements(Q_Q5_1) == 32,
        "and q5_1 is 24, carrying both",
    )
    suite.check(
        block_bytes(Q_Q8_0) == 34 and block_elements(Q_Q8_0) == 32,
        "q8_0 packs 32 into 34",
    )
    suite.check(
        block_bytes(Q_Q4_K) == 144 and block_elements(Q_Q4_K) == 256,
        "q4_k packs 256 into 144",
    )
    suite.check(
        block_bytes(Q_Q5_K) == 176 and block_elements(Q_Q5_K) == 256,
        "q5_k packs 256 into 176",
    )
    suite.check(
        block_bytes(Q_Q6_K) == 210 and block_elements(Q_Q6_K) == 256,
        "and q6_k packs 256 into 210",
    )

    # The geometry here has to agree with the table `molla.model.spec` uses to
    # add up a tensor directory. Two tables that disagree means a tensor whose
    # length is computed one way and read the other.
    from molla.model.spec import encoding_of

    var kinds = [
        Q_F32,
        Q_F16,
        Q_BF16,
        Q_Q4_0,
        Q_Q4_1,
        Q_Q5_0,
        Q_Q5_1,
        Q_Q8_0,
        Q_Q4_K,
        Q_Q5_K,
        Q_Q6_K,
    ]
    var agree = True
    for i in range(len(kinds)):
        var enc = encoding_of(kinds[i])
        if enc.block != block_elements(kinds[i]):
            agree = False
        if enc.block_bytes != block_bytes(kinds[i]):
            agree = False
    suite.check(agree, "and every one agrees with the directory size table")

    suite.check(
        block_bytes(19) == 0 and block_elements(19) == 0,
        "a type with no decoder reports no geometry rather than guessing",
    )
    suite.check(not supported(19), "and says it is not supported")
    suite.check(supported(Q_Q4_K), "while one with a decoder says it is")


def _check_scalars(mut suite: Suite) raises:
    suite.group("quant scalars")

    var b = _bytes(8)
    _put16(b, 0, HALF_ONE)
    suite.check(f16_at(_ptr(b), 0) == 1.0, "0x3C00 as a float16 is one")
    _put16(b, 0, 0xBC00)
    suite.check(f16_at(_ptr(b), 0) == -1.0, "and the sign bit is the top one")
    _put16(b, 0, 0x3F80)
    suite.check(bf16_at(_ptr(b), 0) == 1.0, "0x3F80 as a bfloat16 is one")
    _put16(b, 0, 0xC000)
    suite.check(bf16_at(_ptr(b), 0) == -2.0, "and 0xC000 is minus two")

    b[0] = 0
    b[1] = 0
    b[2] = 0x80
    b[3] = 0x3F
    suite.check(
        f32_at(_ptr(b), 0) == 1.0, "and a float32 is read low byte first"
    )

    # Reading a value at an offset rather than at zero is the case that matters,
    # since every real read is into the middle of a block.
    _put16(b, 4, HALF_ONE)
    suite.check(f16_at(_ptr(b), 4) == 1.0, "at an offset as well as at zero")


def _check_q4_0(mut suite: Suite) raises:
    suite.group("quant q4_0")

    var b = _bytes(18)
    _put16(b, 0, HALF_ONE)
    b[2] = 0x91  # low nibble 1, high nibble 9
    b[3] = 0x08  # low nibble 8, high nibble 0

    var out = _zeros(32)
    dequant_block(Q_Q4_0, _ptr(b), 0, out, 0)

    suite.check(out[0] == -7.0, "a low nibble of one is seven below centre")
    suite.check(
        out[16] == 1.0,
        (
            "and the high nibble of the same byte lands sixteen along, not next"
            " to it"
        ),
    )
    suite.check(
        out[1] == 0.0, "a nibble of eight is the centre and decodes to zero"
    )
    suite.check(
        out[17] == -8.0, "and a nibble of zero is the bottom of the range"
    )


def _check_q4_1(mut suite: Suite) raises:
    suite.group("quant q4_1")

    var b = _bytes(20)
    _put16(b, 0, HALF_ONE)
    _put16(b, 2, 0xC000)  # a minimum of minus two
    b[4] = 0x91
    b[5] = 0x00

    var out = _zeros(32)
    dequant_block(Q_Q4_1, _ptr(b), 0, out, 0)

    suite.check(
        out[0] == -1.0,
        "a q4_1 nibble is scaled and then shifted by the minimum",
    )
    suite.check(out[16] == 7.0, "and the high nibble is still sixteen along")
    suite.check(
        out[1] == -2.0,
        "a nibble of zero is the minimum, because nothing here is centred",
    )


def _check_q5_0(mut suite: Suite) raises:
    suite.group("quant q5_0")

    var b = _bytes(22)
    _put16(b, 0, HALF_ONE)
    # The fifth bits: element 0 is bit 0, element 16 is bit 16, nothing else.
    b[2] = 0x01
    b[4] = 0x01
    b[6] = 0x10  # low nibble 0, high nibble 1
    b[7] = 0x0F  # low nibble 15, high nibble 0

    var out = _zeros(32)
    dequant_block(Q_Q5_0, _ptr(b), 0, out, 0)

    suite.check(
        out[0] == 0.0,
        (
            "the fifth bit is worth sixteen, so a nibble of zero with it set is"
            " the centre"
        ),
    )
    suite.check(
        out[16] == 1.0,
        (
            "element sixteen takes its fifth bit from bit twenty eight, not bit"
            " twelve"
        ),
    )
    suite.check(
        out[1] == -1.0, "and a nibble of fifteen without one is one below"
    )
    suite.check(
        out[17] == -16.0, "as is a zero nibble without one at the floor"
    )


def _check_q5_1(mut suite: Suite) raises:
    suite.group("quant q5_1")

    var b = _bytes(24)
    _put16(b, 0, HALF_ONE)
    _put16(b, 2, 0xC000)  # a minimum of minus two
    b[4] = 0x01  # the fifth bit of element zero
    b[8] = 0x10  # low nibble 0, high nibble 1

    var out = _zeros(32)
    dequant_block(Q_Q5_1, _ptr(b), 0, out, 0)

    suite.check(
        out[0] == 14.0,
        "a q5_1 value runs from zero to thirty one and is never centred",
    )
    suite.check(
        out[16] == -1.0, "and the minimum applies to the high nibble as well"
    )
    suite.check(out[1] == -2.0, "an empty pair decodes to the minimum")


def _check_q8_0(mut suite: Suite) raises:
    suite.group("quant q8_0")

    var b = _bytes(34)
    _put16(b, 0, HALF_ONE)
    b[2] = 1
    b[3] = 127
    b[4] = 128
    b[5] = 255

    var out = _zeros(32)
    dequant_block(Q_Q8_0, _ptr(b), 0, out, 0)

    suite.check(out[0] == 1.0, "a q8_0 value is the byte times the scale")
    suite.check(out[1] == 127.0, "up to 127")
    suite.check(
        out[2] == -128.0,
        "and 128 is minus 128, because the bytes are signed and not unsigned",
    )
    suite.check(out[3] == -1.0, "so 255 is minus one")


def _scales_of_one() -> List[UInt8]:
    """Twelve bytes that make every K quant scale one and every minimum zero.

    Worked out from the packing rather than found by trying: the first four
    pairs read the low six bits of bytes 0 to 3 for the scale and 4 to 7 for the
    minimum, and the last four read the nibbles of bytes 8 to 11 with two more
    bits from the top of the first eight. Leaving those top bits clear and
    setting the low nibbles to one gives one and zero for all eight.
    """
    var out = _bytes(12)
    for i in range(4):
        out[i] = 1
        out[8 + i] = 1
    return out^


def _check_q4_k(mut suite: Suite) raises:
    suite.group("quant q4_k")

    var b = _bytes(144)
    _put16(b, 0, HALF_ONE)  # d
    _put16(b, 2, 0)  # dmin, so the affine term drops out
    var scales = _scales_of_one()
    for i in range(12):
        b[4 + i] = scales[i]
    b[16] = 0x53  # first byte of qs
    b[16 + 32] = 0x27  # first byte of the second group of 64

    var out = _zeros(256)
    dequant_block(Q_Q4_K, _ptr(b), 0, out, 0)

    suite.check(out[0] == 3.0, "the low nibble of the first byte is value zero")
    suite.check(
        out[32] == 5.0,
        "and its high nibble is value 32, a group along rather than next to it",
    )
    suite.check(
        out[64] == 7.0,
        "the next 32 bytes of qs are the second group of 64 values",
    )
    suite.check(out[96] == 2.0, "with their high nibbles 32 further on")

    # The minimum is subtracted, not added, and it is scaled by dmin and not by
    # d. Setting dmin without setting d proves both.
    var c = _bytes(144)
    _put16(c, 0, 0)
    _put16(c, 2, HALF_ONE)
    for i in range(12):
        c[4 + i] = 3  # scale 3, minimum 3, in both halves of the packing
    var shifted = _zeros(256)
    dequant_block(Q_Q4_K, _ptr(c), 0, shifted, 0)
    suite.check(
        shifted[0] == -3.0,
        (
            "a q4_k value is d times sc times q minus dmin times m, not a bare"
            " scale"
        ),
    )


def _check_q5_k(mut suite: Suite) raises:
    suite.group("quant q5_k")

    var scales = _scales_of_one()

    # The same low nibbles in a q4_k block and a q5_k block whose high bit plane
    # is empty have to decode to the same values. The fifth bit adds sixteen and
    # never does anything else, so this pins that it is additive and that the
    # two formats agree on everything below it.
    var four = _bytes(144)
    _put16(four, 0, HALF_ONE)
    for i in range(12):
        four[4 + i] = scales[i]
    var five = _bytes(176)
    _put16(five, 0, HALF_ONE)
    for i in range(12):
        five[4 + i] = scales[i]
    for l in range(128):
        four[16 + l] = UInt8((l * 37) & 0xFF)
        five[48 + l] = UInt8((l * 37) & 0xFF)

    var a = _zeros(256)
    var e = _zeros(256)
    dequant_block(Q_Q4_K, _ptr(four), 0, a, 0)
    dequant_block(Q_Q5_K, _ptr(five), 0, e, 0)
    var same = True
    for i in range(256):
        if a[i] != e[i]:
            same = False
    suite.check(
        same, "with an empty high plane a q5_k block decodes like a q4_k one"
    )

    # Bit zero of the high plane belongs to value zero and bit one belongs to
    # value 32, and the pair advances by two for every 64 values. Setting one
    # byte of the plane to all ones lifts four values by sixteen and nothing
    # else.
    five[16] = 0xFF
    var lifted = _zeros(256)
    dequant_block(Q_Q5_K, _ptr(five), 0, lifted, 0)
    suite.check(lifted[0] == a[0] + 16.0, "and a set high bit adds sixteen")
    suite.check(lifted[32] == a[32] + 16.0, "to the value 32 along as well")
    suite.check(
        lifted[64] == a[64] + 16.0,
        "and the bit pair moves on by two for the next group of 64",
    )
    suite.check(lifted[1] == a[1], "while the neighbouring value is untouched")


def _check_q6_k(mut suite: Suite) raises:
    suite.group("quant q6_k")

    var b = _bytes(210)
    for i in range(16):
        b[192 + i] = 1  # scales, signed, all one
    _put16(b, 208, HALF_ONE)  # d, which lives at the end and not the start
    b[0] = 0x50  # ql[0]: low nibble 0, high nibble 5
    b[32] = 0x03  # ql[32]
    b[128] = 0b01000001  # qh[0]: bit pairs 1, 0, 0, 1 from the bottom up

    var out = _zeros(256)
    dequant_block(Q_Q6_K, _ptr(b), 0, out, 0)

    suite.check(
        out[0] == -16.0,
        (
            "a q6_k value is the six bits minus 32, low four from ql and high"
            " two from qh"
        ),
    )
    suite.check(
        out[32] == -29.0, "with value 32 taking the byte 32 along in ql"
    )
    suite.check(
        out[64] == -27.0, "and value 64 taking the high nibble of ql[0]"
    )
    suite.check(out[96] == -16.0, "and value 96 the high nibble 32 along")

    # The scales are signed. An unsigned reading of 0xFF is 255 and a signed one
    # is minus one, and every value in the group changes sign between them.
    b[192] = 0xFF
    var flipped = _zeros(256)
    dequant_block(Q_Q6_K, _ptr(b), 0, flipped, 0)
    suite.check(
        flipped[0] == 16.0,
        "and a q6_k scale is signed, so 0xFF is minus one and not 255",
    )


def _check_errors(mut suite: Suite) raises:
    suite.group("quant errors")

    var b = _bytes(210)
    var out = _zeros(256)

    var raised = False
    try:
        dequant_block(19, _ptr(b), 0, out, 0)
    except:
        raised = True
    suite.check(
        raised, "a type with no decoder raises rather than writing zeros"
    )

    raised = False
    try:
        var small = _zeros(4)
        dequant_block(Q_Q4_K, _ptr(b), 0, small, 0)
    except:
        raised = True
    suite.check(raised, "and a block that would not fit the output raises")

    raised = False
    try:
        dequant_run(Q_Q4_K, _ptr(b), 0, 300, out, 0)
    except:
        raised = True
    suite.check(
        raised,
        (
            "a run that is not a whole number of blocks raises rather than"
            " rounding"
        ),
    )

    # A run of two blocks has to land the second block after the first, at the
    # element stride and not the byte stride. Getting that wrong overwrites the
    # first block and the values are all still plausible.
    var two = _bytes(36 * 2)
    _put16(two, 0, HALF_ONE)
    two[2] = 5
    _put16(two, 34, HALF_ONE)
    two[36] = 7
    var run_out = _zeros(64)
    dequant_run(Q_Q8_0, _ptr(two), 0, 64, run_out, 0)
    suite.check(run_out[0] == 5.0, "the first block of a run lands at zero")
    suite.check(
        run_out[32] == 7.0, "and the second lands 32 values on, not 34 bytes on"
    )
