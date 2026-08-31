"""Run one Q4_K matmul from max/kernels against weights taken out of a real
GGUF file.

The inputs and the two NumPy references are produced by gen_data.py. This reads
the raw bytes, calls the kernel, and writes the result back out for check.py to
compare. Nothing here is molla code. It exists to answer one question, which is
whether a plain Mojo binary can call these kernels at all.
"""

from std.math import ceildiv
from std.memory import alloc
from std.sys import size_of
from std.utils.index import Index

from layout import Layout, LayoutTensor, RuntimeLayout, lt_to_tt
from quantization.qmatmul_k import (
    _block_Q4_K,
    matmul_Q4_K,
    matmul_Q4_K_pack_b,
)

comptime RM2 = Layout.row_major[2]()


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
    print("M", M, "N", N, "K", K, "k_bytes", k_bytes)

    var a_bytes = read_file("a.bin")
    var b_bytes = read_file("b.bin")
    if a_bytes.__len__() != M * K * 4:
        raise Error("a.bin is the wrong size")
    if b_bytes.__len__() != N * k_bytes:
        raise Error("b.bin is the wrong size")

    var a_ptr = alloc[Float32](M * K)
    var a_raw = a_ptr.bitcast[UInt8]()
    for i in range(M * K * 4):
        a_raw[i] = a_bytes[i]

    var b_ptr = alloc[UInt8](N * k_bytes)
    for i in range(N * k_bytes):
        b_ptr[i] = b_bytes[i]

    var c_ptr = alloc[Float32](M * N)
    for i in range(M * N):
        c_ptr[i] = 0.0

    var b_packed_ptr = alloc[UInt8](N * k_bytes)

    var a = LayoutTensor[DType.float32, RM2](
        a_ptr, RuntimeLayout[RM2].row_major(Index(M, K))
    )
    var b = LayoutTensor[DType.uint8, RM2](
        b_ptr, RuntimeLayout[RM2].row_major(Index(N, k_bytes))
    )
    var b_packed = LayoutTensor[DType.uint8, RM2](
        b_packed_ptr, RuntimeLayout[RM2].row_major(Index(N, k_bytes))
    )
    var c = LayoutTensor[DType.float32, RM2](
        c_ptr, RuntimeLayout[RM2].row_major(Index(M, N))
    )

    matmul_Q4_K_pack_b(lt_to_tt(b), lt_to_tt(b_packed))
    matmul_Q4_K(lt_to_tt(a), lt_to_tt(b_packed), lt_to_tt(c))

    var out = List[UInt8]()
    out.reserve(M * N * 4)
    var c_raw = c_ptr.bitcast[UInt8]()
    for i in range(M * N * 4):
        out.append(c_raw[i])
    with open("c_cpu.bin", "w") as f:
        f.write_bytes(Span(out))
    print("wrote c_cpu.bin, block size", size_of[_block_Q4_K]())
