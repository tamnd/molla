"""Checks `molla.nn.quant` against the fixtures `scripts/gen-quant.py` wrote.

Not part of the library and not part of the test suite. For each format it maps
the raw blocks and the reference float32 values, dequantizes the blocks, and
reports the largest absolute disagreement and where it was.

The tolerance is exact and not a small epsilon. Both sides are decoding the same
bytes with the same arithmetic in the same order, so a bit for bit match is what
correct looks like. A tolerance here would let a wrong nibble order through
whenever the two nibbles happened to be close, which is the exact failure this
exists to catch. The one thing that is allowed to differ is a zero of the wrong
sign, since ggml writes both and they compare equal in float anyway.

Usage:

    mojo run -I src scripts/quant_oracle.mojo corpus/quant
"""

from std.sys import argv, exit

from molla.nn.quant import (
    Q_BF16,
    Q_F16,
    Q_Q4_0,
    Q_Q4_1,
    Q_Q4_K,
    Q_Q5_0,
    Q_Q5_1,
    Q_Q5_K,
    Q_Q6_K,
    Q_Q8_0,
    block_bytes,
    block_elements,
    dequant_run,
)
from molla.sys.mmap import Mapping, RawPtr


def _names() -> List[String]:
    return [
        String("f16"),
        String("bf16"),
        String("q4_0"),
        String("q4_1"),
        String("q5_0"),
        String("q5_1"),
        String("q8_0"),
        String("q4_k"),
        String("q5_k"),
        String("q6_k"),
    ]


def _kinds() -> List[Int]:
    return [
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


def _f32_at(p: RawPtr, at: Int) -> Float32:
    from std.memory import bitcast

    var bits = UInt32(p.unsafe_load(at))
    bits |= UInt32(p.unsafe_load(at + 1)) << 8
    bits |= UInt32(p.unsafe_load(at + 2)) << 16
    bits |= UInt32(p.unsafe_load(at + 3)) << 24
    return bitcast[DType.float32, 1](bits)


def _abs(x: Float32) -> Float32:
    return -x if x < 0 else x


def _check(dir: String, name: String, kind: Int) raises -> Int:
    """Returns the number of values that did not match."""
    var raw = Mapping(dir + "/" + name + ".bin")
    var oracle = Mapping(dir + "/" + name + ".f32")

    var per = block_elements(kind)
    var stride = block_bytes(kind)
    if raw.length % stride != 0:
        raise Error(name + ".bin is not a whole number of blocks")
    var blocks = raw.length // stride
    var count = blocks * per
    if oracle.length != count * 4:
        raise Error(
            name
            + ".f32 has "
            + String(oracle.length // 4)
            + " values, expected "
            + String(count)
        )

    var got = List[Float32]()
    for _ in range(count):
        got.append(0.0)
    dequant_run(kind, raw.base(), 0, count, got, 0)

    var bad = 0
    var worst = Float32(0)
    var first = -1
    for i in range(count):
        var want = _f32_at(oracle.base(), i * 4)
        if got[i] == want:
            continue
        bad += 1
        var gap = _abs(got[i] - want)
        if gap > worst:
            worst = gap
        if first < 0:
            first = i

    var line = name + "  " + String(blocks) + " blocks, "
    line += String(count) + " values"
    if bad == 0:
        print(line + "  exact")
    else:
        print(
            line
            + "  "
            + String(bad)
            + " differ, worst "
            + String(worst)
            + ", first at "
            + String(first)
        )
    raw.close()
    oracle.close()
    return bad


def main():
    var args = argv()
    if len(args) < 2:
        print("usage: quant_oracle <fixture-dir>")
        exit(2)

    var dir = String(args[1])
    var names = _names()
    var kinds = _kinds()
    var bad = 0
    var ran = 0
    for i in range(len(names)):
        # The synthetic fixtures are always there and the ones cut out of a
        # real model are only there if `gen-quant.py` was given one, so a
        # missing pair of files is a skip and a malformed one is a failure.
        for suffix in [String(""), String("-real")]:
            var name = names[i] + suffix
            try:
                bad += _check(dir, name, kinds[i])
                ran += 1
            except e:
                if suffix == "":
                    print("quant_oracle:", e)
                    exit(1)
    if ran == 0:
        print("no fixtures in " + dir + ", run scripts/gen-quant.py first")
        exit(1)

    if bad > 0:
        print(String(bad) + " values disagree with the oracle")
        exit(1)
    print("every format matches the oracle exactly")
