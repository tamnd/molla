"""Call the int4 GEMM from max/kernels on an NVIDIA GPU.

Separate from qmatmul_gpu.mojo because nothing about it is portable. It wants
Q4_0 weights repacked by a second kernel, bfloat16 in and out, and it will not
compile for anything but an NVIDIA target. The point of running it is to show
that the NVIDIA half of the quantised GPU story in max/kernels does work, so
the gap on Apple is a gap and not a general breakage.

Shapes are compile time constants because the repack kernel reads N and K out
of the layout at compile time. K has to be a multiple of 1024, which is the K
tile the kernel steps in. The full 4864 wide tensor this fixture comes from is
not, and it fails to compile with a constraint error several layers down in the
layout code that never mentions K.
"""

from std.memory import alloc

from layout import Layout, LayoutTensor, lt_to_tt
from max.gpu.host import DeviceContext
from quantization.qmatmul_gpu import gpu_qint4_repack_Q4_0, matmul_gpu_qint4

comptime M = 64
comptime N = 896
comptime K = 4096
comptime KB = K // 32 * 18
"""Q4_0 stores 32 values per block as one float16 scale and 16 packed bytes."""

comptime A_LAYOUT = Layout.row_major(M, K)
comptime B_LAYOUT = Layout.row_major(N, KB)
comptime C_LAYOUT = Layout.row_major(M, N)


def read_file(path: String) raises -> List[UInt8]:
    with open(path, "r") as f:
        return f.read_bytes()


def main() raises:
    var ctx = DeviceContext()
    print("api", ctx.api(), "name", ctx.name(), "cc", ctx.compute_capability())

    var a_bytes = read_file("a_bf16.bin")
    var b_bytes = read_file("b_q40.bin")
    if a_bytes.__len__() != M * K * 2:
        raise Error("a_bf16.bin is the wrong size")
    if b_bytes.__len__() != N * KB:
        raise Error("b_q40.bin is the wrong size")

    var h_a = alloc[BFloat16](M * K)
    var h_a_raw = h_a.bitcast[UInt8]()
    for i in range(M * K * 2):
        h_a_raw[i] = a_bytes[i]
    var h_b = alloc[UInt8](N * KB)
    for i in range(N * KB):
        h_b[i] = b_bytes[i]
    var h_c = alloc[BFloat16](M * N)

    var d_a = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var d_b = ctx.enqueue_create_buffer[DType.uint8](N * KB)
    var d_bp = ctx.enqueue_create_buffer[DType.uint8](N * KB)
    var d_c = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    ctx.enqueue_copy(d_a, h_a)
    ctx.enqueue_copy(d_b, h_b)

    var a = LayoutTensor[DType.bfloat16, A_LAYOUT](d_a.unsafe_ptr())
    var b = LayoutTensor[DType.uint8, B_LAYOUT](d_b.unsafe_ptr())
    var bp = LayoutTensor[DType.uint8, B_LAYOUT](d_bp.unsafe_ptr())
    var c = LayoutTensor[DType.bfloat16, C_LAYOUT](d_c.unsafe_ptr())

    gpu_qint4_repack_Q4_0["gpu"](lt_to_tt(b), lt_to_tt(bp), ctx)
    matmul_gpu_qint4[32, "gpu"](lt_to_tt(c), lt_to_tt(a), lt_to_tt(bp), ctx)

    ctx.enqueue_copy(h_c, d_c)
    ctx.synchronize()

    var out = List[UInt8]()
    out.reserve(M * N * 2)
    var c_raw = h_c.bitcast[UInt8]()
    for i in range(M * N * 2):
        out.append(c_raw[i])
    with open("c_nv.bin", "w") as f:
        f.write_bytes(Span(out))
    print("wrote c_nv.bin")
