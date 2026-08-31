"""Second fixture, for the NVIDIA only int4 GEMM in max/kernels.

That kernel wants Q4_0 weights and bfloat16 activations, neither of which the
Q4_K fixture has, so the same weight matrix is dequantised and requantised to
Q4_0 and the activations are rounded to bfloat16. The reference is computed
from exactly those rounded inputs, so any difference is the kernel rather than
the rounding.

K defaults to 4096 rather than the full 4864 because the kernel tiles K in
blocks of 1024 and a K that is not a multiple of 1024 fails to compile, deep
inside the layout code and with no message that says so.

    python gen_q40.py [model.gguf] [M] [K]
"""

import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize, quantize
from gguf.constants import GGMLQuantizationType as T

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-0.5b-instruct-q4_k_m.gguf"
M = int(sys.argv[2]) if len(sys.argv) > 2 else 64
K = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
TENSOR = "blk.11.ffn_down.weight"


def to_bf16(x):
    """Round float32 to bfloat16, half to even, keeping the bits in a uint16.

    NumPy has no bfloat16 so this works on the bit pattern. Add the rounding
    bias, which is 0x7fff plus the low bit of the surviving mantissa, then drop
    the bottom sixteen bits.
    """
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


reader = GGUFReader(MODEL)
tensor = next(x for x in reader.tensors if x.name == TENSOR)
b_k = dequantize(np.ascontiguousarray(tensor.data), T.Q4_K).astype(np.float32)
b_k = np.ascontiguousarray(b_k[:, :K])
N = b_k.shape[0]

b_q40 = quantize(b_k, T.Q4_0)
b_deq = dequantize(b_q40, T.Q4_0).astype(np.float32)
assert b_q40.shape == (N, K // 32 * 18), b_q40.shape

rng = np.random.default_rng(20260831)
a = rng.standard_normal((M, K), dtype=np.float32)
a_bf = to_bf16(a)

# The kernel accumulates in float32 and writes bfloat16, so the reference is
# rounded the same way at the end.
ref = from_bf16(a_bf).astype(np.float64) @ b_deq.astype(np.float64).T
ref_bf = from_bf16(to_bf16(ref.astype(np.float32)))

open("a_bf16.bin", "wb").write(a_bf.tobytes())
open("b_q40.bin", "wb").write(b_q40.tobytes())
open("ref_q40.bin", "wb").write(ref_bf.tobytes())
open("shape_q40.txt", "w").write("%d %d %d %d\n" % (M, N, K, K // 32 * 18))
print("tensor %s, M %d, N %d, K %d, weight bytes %d"
      % (TENSOR, M, N, K, b_q40.nbytes))
print("peak |ref| %.4f" % float(np.abs(ref_bf).max()))
