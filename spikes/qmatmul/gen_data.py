"""Pull one Q4_K weight matrix out of a real GGUF file and write the inputs and
two references for the Q4_K half of the spike.

Everything is written as raw little endian bytes so the Mojo side can read it
with a plain file read and no parsing. Shapes go in a text file next to them.

Needs the `gguf` package, which is a build time tool for this spike only and
never a molla dependency.

    python gen_data.py [model.gguf]
"""

import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType as T

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-0.5b-instruct-q4_k_m.gguf"
TENSOR = "blk.11.ffn_down.weight"
M = 8
QK_K = 256

reader = GGUFReader(MODEL)
tensor = next(x for x in reader.tensors if x.name == TENSOR)
assert tensor.tensor_type == T.Q4_K, tensor.tensor_type

b_raw = np.ascontiguousarray(tensor.data)
N, k_bytes = b_raw.shape
K = k_bytes // 144 * QK_K
assert k_bytes % 144 == 0 and K % QK_K == 0

b_deq = dequantize(b_raw, T.Q4_K).astype(np.float32)
assert b_deq.shape == (N, K), b_deq.shape

rng = np.random.default_rng(20260831)
a = rng.standard_normal((M, K), dtype=np.float32)

# Reference one, full precision. A stays float32 and B is dequantised the way
# llama.cpp does it. This is the number a caller would want to be true. It is
# not what matmul_Q4_K computes, because that kernel quantises A first.
ref_float = (a.astype(np.float64) @ b_deq.astype(np.float64).T).astype(np.float32)

# Reference two, the arithmetic matmul_Q4_K actually performs. A is quantised
# to int8 with one scale per 256 element super block per row, rounding half to
# even, which is what roundeven_to_int32 does in the kernel.
a_blocks = a.reshape(M, K // QK_K, QK_K)
amax = np.abs(a_blocks).max(axis=2)
mult = np.where(amax != 0.0, 127.0 / amax, 0.0).astype(np.float32)
scale = (amax / 127.0).astype(np.float32)
aq = np.rint(a_blocks * mult[:, :, None]).astype(np.int32).clip(-128, 127)
a_hat = (aq * scale[:, :, None]).reshape(M, K).astype(np.float64)
ref_exact = (a_hat @ b_deq.astype(np.float64).T).astype(np.float32)

open("a.bin", "wb").write(a.tobytes())
open("b.bin", "wb").write(b_raw.tobytes())
open("ref_float.bin", "wb").write(ref_float.tobytes())
open("ref_exact.bin", "wb").write(ref_exact.tobytes())
open("shape.txt", "w").write("%d %d %d %d\n" % (M, N, K, k_bytes))

diff = np.abs(ref_float - ref_exact)
peak = float(np.abs(ref_float).max())
print("tensor %s, M %d, N %d, K %d, weight bytes %d" % (TENSOR, M, N, K, b_raw.nbytes))
print("peak |ref_float| %.4f" % peak)
print("cost of quantising the activations: max abs %.6f, %.3g of peak"
      % (float(diff.max()), float(diff.max()) / peak))
