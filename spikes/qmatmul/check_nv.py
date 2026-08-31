"""Compare the NVIDIA int4 GEMM output against its bfloat16 NumPy reference.

The kernel writes bfloat16, which has eight mantissa bits, so the error is
reported in units of one bfloat16 step at the magnitude of the value rather
than as a fraction. A fraction would be dominated by the output format and
would say nothing about the kernel.
"""

import sys
import numpy as np

M, N, K, kb = (int(x) for x in open("shape_q40.txt").read().split())


def from_bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


c = from_bf16(np.fromfile("c_nv.bin", dtype=np.uint16)).reshape(M, N)
ref = np.fromfile("ref_q40.bin", dtype=np.float32).reshape(M, N)

d = np.abs(c.astype(np.float64) - ref.astype(np.float64))
# One bfloat16 step at the peak magnitude of the output, not at the magnitude
# of each element. A dot product over thousands of terms lands near zero often
# enough that a per element step count is meaningless there, the same reason
# check.py measures against the peak.
peak = float(np.abs(ref).max())
ulp = 2.0 ** (np.floor(np.log2(peak)) - 8)
steps = d / ulp

TOL = 4.0
"""Maximum error in bfloat16 steps. The kernel accumulates in float32 and
rounds once at the end, so a correct implementation lands within a couple of
steps of a float64 reference and a wrong nibble order or a dropped scale lands
thousands away. Four leaves room for accumulation order without leaving room
for a bug."""

print("cuda-4090  peak |ref| %.4f, one bfloat16 step there is %.6f" % (peak, ulp))
print("cuda-4090  max abs %.6g  max steps %.2f  mean steps %.4f  exact %d of %d" % (
    d.max(), steps.max(), steps.mean(), int((d == 0).sum()), d.size))
ok = steps.max() < TOL
print("cuda-4090  tolerance %.1f bfloat16 steps, %s" % (TOL, "pass" if ok else "FAIL"))
sys.exit(0 if ok else 1)
