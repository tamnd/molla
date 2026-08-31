"""A Q4_K dequantise and multiply kernel written from scratch, run on whatever
GPU the machine has.

This exists because max/kernels does not have one. The K quant matmul in
max/kernels is CPU only, the 4 bit GPU matmul there is NVIDIA only and wants a
repacked weight layout, and the Apple GPU quantised matmuls are a separate
source tree that targets NVFP4 and int8 and refuses to launch on anything below
an M5. So the portability claim gets tested with our own kernel instead.

The kernel is deliberately naive. One thread per output element, no shared
memory, no tiling, weights dequantised on the fly straight out of the GGUF
block layout. It is not fast and it is not meant to be. The question it answers
is whether one Mojo source compiles and produces the same numbers on Metal and
on sm_89, not how many teraflops we can reach.
"""

from std.gpu import block_idx, thread_idx
from std.math import ceildiv
from std.memory import bitcast, alloc
from max.gpu.host import DeviceContext

comptime QK_K = 256
comptime BLOCK_BYTES = 144
"""Q4_K on disk: base scale as float16, base min as float16, twelve bytes of
six bit group scales and mins, then 128 bytes holding 256 four bit values."""

comptime TPB = 64


def load_f16(p: UnsafePointer[UInt8, ImmutAnyOrigin], off: Int) -> Float32:
    """Read a little endian float16 out of the block header.

    Assembled a byte at a time rather than cast in place because the block is
    only two byte aligned inside the row and a float16 load off an odd address
    is undefined on at least one of the two targets.
    """
    var lo = UInt16(Int(p[off]))
    var hi = UInt16(Int(p[off + 1]))
    return Float32(bitcast[DType.float16, 1](lo | (hi << 8)))


def scale_min(scales: UnsafePointer[UInt8, ImmutAnyOrigin], j: Int) -> Tuple[Int, Int]:
    """Unpack the six bit scale and min for group j.

    This is get_scale_min_k4 from llama.cpp. The first four groups keep their
    six bits in one byte each. The last four borrow their top two bits from the
    high end of the first four, which is how sixteen six bit values fit in
    twelve bytes.
    """
    if j < 4:
        return (Int(scales[j]) & 63, Int(scales[j + 4]) & 63)
    var d = (Int(scales[j + 4]) & 15) | ((Int(scales[j - 4]) >> 6) << 4)
    var m = (Int(scales[j + 4]) >> 4) | ((Int(scales[j]) >> 6) << 4)
    return (d, m)


def q4k_matmul_kernel(
    a: UnsafePointer[Float32, ImmutAnyOrigin],
    b: UnsafePointer[UInt8, ImmutAnyOrigin],
    c: UnsafePointer[Float32, MutAnyOrigin],
    m_dev: Int32,
    n_dev: Int32,
    k_dev: Int32,
    kb_dev: Int32,
):
    var M = Int(m_dev)
    var N = Int(n_dev)
    var K = Int(k_dev)
    var k_bytes = Int(kb_dev)

    var n = block_idx.x * TPB + thread_idx.x
    var m = block_idx.y
    if n >= N or m >= M:
        return

    var row = b + n * k_bytes
    var a_row = a + m * K
    var acc = Float32(0)
    var blocks = K // QK_K

    for kb in range(blocks):
        var blk = row + kb * BLOCK_BYTES
        var d = load_f16(blk, 0)
        var dmin = load_f16(blk, 2)
        var scales = blk + 4
        var qs = blk + 16
        var base = kb * QK_K

        for pair in range(4):
            var lo = scale_min(scales, pair * 2)
            var hi = scale_min(scales, pair * 2 + 1)
            var d1 = d * Float32(lo[0])
            var m1 = dmin * Float32(lo[1])
            var d2 = d * Float32(hi[0])
            var m2 = dmin * Float32(hi[1])
            var q = qs + pair * 32
            var off = base + pair * 64

            for l in range(32):
                var byte = Int(q[l])
                acc += a_row[off + l] * (d1 * Float32(byte & 15) - m1)
                acc += a_row[off + 32 + l] * (d2 * Float32(byte >> 4) - m2)

    c[m * N + n] = acc


def read_file(path: String) raises -> List[UInt8]:
    with open(path, "r") as f:
        return f.read_bytes()


def main() raises:
    var shape = read_file("shape.txt")
    var text = String(StringSpan(unsafe_from_utf8=shape))
    var parts = text.strip().split(" ")
    var M = Int(parts[0])
    var N = Int(parts[1])
    var K = Int(parts[2])
    var k_bytes = Int(parts[3])

    var ctx = DeviceContext()
    print("api", ctx.api(), "name", ctx.name(), "cc", ctx.compute_capability())
    print("M", M, "N", N, "K", K, "k_bytes", k_bytes)

    var a_bytes = read_file("a.bin")
    var b_bytes = read_file("b.bin")

    var h_a = alloc[Float32](M * K)
    var h_a_raw = h_a.bitcast[UInt8]()
    for i in range(M * K * 4):
        h_a_raw[i] = a_bytes[i]
    var h_b = alloc[UInt8](N * k_bytes)
    for i in range(N * k_bytes):
        h_b[i] = b_bytes[i]
    var h_c = alloc[Float32](M * N)

    var d_a = ctx.enqueue_create_buffer[DType.float32](M * K)
    var d_b = ctx.enqueue_create_buffer[DType.uint8](N * k_bytes)
    var d_c = ctx.enqueue_create_buffer[DType.float32](M * N)
    ctx.enqueue_copy(d_a, h_a)
    ctx.enqueue_copy(d_b, h_b)

    ctx.enqueue_function[q4k_matmul_kernel](
        d_a,
        d_b,
        d_c,
        Int32(M),
        Int32(N),
        Int32(K),
        Int32(k_bytes),
        grid_dim=(ceildiv(N, TPB), M, 1),
        block_dim=(TPB, 1, 1),
    )
    ctx.enqueue_copy(h_c, d_c)
    ctx.synchronize()

    var out = List[UInt8]()
    out.reserve(M * N * 4)
    var c_raw = h_c.bitcast[UInt8]()
    for i in range(M * N * 4):
        out.append(c_raw[i])
    with open("c_gpu.bin", "w") as f:
        f.write_bytes(Span(out))
    print("wrote c_gpu.bin")
