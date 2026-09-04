"""Checks the device matvec against the host one over the quantization corpus.

Not part of the library and not part of the test suite, for the same reason the
other oracles are not: it needs fixtures that are generated rather than
committed, and this one also needs a GPU, which three of the five machines in
the fleet do not have.

For every ggml type that has a planar form it reads the blocks
`scripts/gen-quant.py` wrote, views them as a matrix, multiplies by a fixed
vector three ways, and prints how far apart the answers are.

    ggml on the host     `molla.nn.kernel.matvec` reading the file's blocks
    planar on the host   the same, after `molla.nn.repack` has rewritten them
    planar on the device `molla.nn.gpu.device_matvec`

The middle one is there to split the error in two. A disagreement between the
first and the second is the repack, which is host arithmetic and has nothing to
do with any GPU, and a disagreement between the second and the third is the
kernel. Reporting one number for both would mean every device failure came with
a suspect that could not be ruled out.

The gate is on the first against the third, because that is the claim: the same
weights give the same answer on a device as they do on a host. It is stated
against the peak magnitude of the output rather than per element, because a dot
product over hundreds of terms lands near zero often enough that a relative
error per element means nothing there.

Usage:

    mojo run -I src scripts/kernel_oracle.mojo corpus/quant
"""

from std.memory import alloc
from std.sys import argv, exit
from std.sys.info import has_accelerator

from max.gpu.host import DeviceContext

from molla.nn.gpu import device_matvec
from molla.nn.kernel import matvec
from molla.nn.quant import (
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
)
from molla.nn.repack import LAYOUT_PLANAR, planar_row_bytes, repack_row
from molla.nn.tensor import WHERE_DEVICE, Buffer, Tensor
from molla.sys.device import build_target_arch
from molla.sys.mem import keep
from molla.sys.mmap import Mapping, RawPtr

comptime TOLERANCE = 1e-5
"""Allowed disagreement as a fraction of the peak magnitude in the reference.

The same gate the M0 spike used for a float32 output, and it is loose on purpose
in one direction and tight in the other. Float32 accumulation over a few hundred
terms costs a few times 1e-7 whatever order it happens in, so this leaves two
orders of magnitude of headroom for a reduction tree that sums the row in a
different order from the host loop. It is nowhere near loose enough to hide a
swapped nibble, a scale read from the wrong group or a row stride that is off,
which are the failures this is looking for and all of which move a result by a
recognisable fraction of the whole.
"""

comptime K_TOLERANCE = 1e-3
"""The same thing for the three k types, which do not round trip exactly.

Their scale planes hold a product of a float16 and a small integer and the
product is rounded back to float16 once, at repack time, so every term of the
dot product can be off by 2^-11 of itself, which is 4.9e-4. The row's total can
be smaller than the sum of the magnitudes it is made of, so the error as a
fraction of the total can be larger than the error in any one term, which is why
this is two of them rather than one. What the corpus actually produces is in
docs/validation/logits.md and it is well inside this.

It is still nowhere near loose enough to hide the failures the tighter number is
looking for. A swapped nibble or a scale read from the wrong group moves a
result by a fraction of the whole and not by a thousandth of it.
"""


def _lossy(kind: Int) -> Bool:
    """Whether this type loses anything on the way into the planar layout.

    The three k types do, for the reason `K_TOLERANCE` gives. The other five
    carry a float16 scale out of the block into a float16 plane and nothing
    about it moves.
    """
    return kind == Q_Q4_K or kind == Q_Q5_K or kind == Q_Q6_K


def _names() -> List[String]:
    return [
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
    return [Q_Q4_0, Q_Q4_1, Q_Q5_0, Q_Q5_1, Q_Q8_0, Q_Q4_K, Q_Q5_K, Q_Q6_K]


def _abs(x: Float32) -> Float32:
    return -x if x < 0 else x


def _input(cols: Int) -> Buffer:
    """A fixed activation vector, signed and never zero.

    Signed matters. A kernel that dropped the sign of the quants, or that read
    the minimum plane with the wrong sign, still agrees with the host to several
    digits when every input is positive and the errors cancel across a row.
    """
    var x = Buffer(cols)
    for i in range(cols):
        x.data[i] = (Float32((i * 37) % 101) - 50.0) / 50.0
    return x^


def _worst(a: Buffer, b: Buffer) -> Float32:
    var worst = Float32(0)
    for i in range(a.elements()):
        var gap = _abs(a.data[i] - b.data[i])
        if gap > worst:
            worst = gap
    return worst


def _peak(a: Buffer) -> Float32:
    var peak = Float32(0)
    for i in range(a.elements()):
        var m = _abs(a.data[i])
        if m > peak:
            peak = m
    return peak


def _check(
    ctx: DeviceContext, dir: String, name: String, kind: Int
) raises -> Bool:
    """One format, three matvecs. Returns whether it is inside tolerance."""
    var raw = Mapping(dir + "/" + name + ".bin")
    var per = block_elements(kind)
    var stride = block_bytes(kind)
    if raw.length % stride != 0:
        raise Error(name + ".bin is not a whole number of blocks")
    var blocks = raw.length // stride

    # Two blocks to a row, which makes every fixture in the corpus a matrix of a
    # few hundred rows. More than one block per row so that a row stride bug has
    # somewhere to show itself, and a shape that does not divide the tile width
    # evenly so that the tail of the strided walk is exercised rather than
    # assumed.
    if blocks < 2 or blocks % 2 != 0:
        raise Error(name + ".bin does not hold an even number of blocks")
    var cols = per * 2
    var rows = blocks // 2

    var x = _input(cols)
    var ggml = Tensor(raw.address, kind, cols, rows)
    var want = Buffer(rows)
    matvec(ggml, x, want)

    var row_bytes = planar_row_bytes(kind, cols)
    var total = row_bytes * rows
    var host_bytes = alloc[UInt8](total)
    var host_planar = RawPtr(unsafe_from_address=Int(host_bytes))
    var src_stride = ggml.row_bytes()
    for r in range(rows):
        repack_row(
            kind,
            raw.base(),
            r * src_stride,
            cols,
            host_planar,
            r * row_bytes,
        )

    var planar = Tensor(Int(host_bytes), kind, cols, rows, LAYOUT_PLANAR)
    var on_host = Buffer(rows)
    matvec(planar, x, on_host)

    var pool = ctx.enqueue_create_buffer[DType.uint8](total)
    with pool.map_to_host() as h:
        for i in range(total):
            h[i] = host_bytes[i]
    var resident = Tensor(
        Int(pool.unsafe_ptr()),
        kind,
        cols,
        rows,
        LAYOUT_PLANAR,
        WHERE_DEVICE,
    )
    var on_device = Buffer(rows)
    device_matvec(ctx, resident, x, on_device)
    # The tensor holds the pool's address rather than the pool, so the last use
    # of `pool` the compiler can see is the line above that read its pointer,
    # and without this it is freed before the kernel that reads it has run. The
    # symptom is a device result of exactly zero for every row, which looks like
    # a kernel that never launched rather than like a lifetime bug. Same hazard
    # as the one in `docs/validation/threading.md`, third time it has come up.
    keep(pool)

    var peak = _peak(want)
    var repack_gap = _worst(want, on_host)
    var device_gap = _worst(want, on_device)
    var kernel_gap = _worst(on_host, on_device)
    var relative = device_gap / peak if peak > 0 else Float32(0)
    var tol = Float32(K_TOLERANCE) if _lossy(kind) else Float32(TOLERANCE)

    var line = name + "  " + String(rows) + " by " + String(cols)
    line += "  peak " + String(peak)
    line += "  repack " + String(repack_gap)
    line += "  kernel " + String(kernel_gap)
    line += "  device " + String(device_gap)
    line += " (" + String(relative) + " of peak, allowed " + String(tol) + ")"
    print(line)

    raw.close()
    return relative <= tol


def main():
    var args = argv()
    if len(args) < 2:
        print("usage: kernel_oracle <fixture-dir>")
        exit(2)
    var dir = String(args[1])

    comptime if not has_accelerator():
        print(
            "this build has no device code in it, so there is no device matvec"
            " to check. Build on a machine with a GPU"
        )
        exit(2)
    else:
        try:
            var ctx = DeviceContext()
            print(
                "built for",
                build_target_arch(),
                "running on",
                ctx.api(),
                ctx.name(),
            )
            print(
                "tolerance",
                TOLERANCE,
                "of the peak magnitude, and",
                K_TOLERANCE,
                "for the k types",
            )
            var names = _names()
            var kinds = _kinds()
            var bad = 0
            var ran = 0
            for i in range(len(names)):
                # The synthetic fixtures are always there and the ones cut out
                # of a real model are only there if `gen-quant.py` was given
                # one, so a missing pair of files is a skip and a malformed one
                # is a failure.
                for suffix in [String(""), String("-real")]:
                    var name = names[i] + suffix
                    try:
                        if not _check(ctx, dir, name, kinds[i]):
                            bad += 1
                        ran += 1
                    except e:
                        if suffix == "":
                            print("kernel_oracle:", e)
                            exit(1)
            if ran == 0:
                print(
                    "no fixtures in " + dir + ", run scripts/gen-quant.py first"
                )
                exit(1)
            if bad > 0:
                print(String(bad) + " formats are outside tolerance")
                exit(1)
            print("every format agrees with the host inside tolerance")
        except e:
            print("kernel_oracle:", e)
            exit(1)
