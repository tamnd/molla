"""Compare a kernel result against the two references written by gen_data.py.

Usage: check.py <result.bin> <label> <exact|float>

The third argument picks which reference is the gate, because the two kernels
compute different things on purpose. matmul_Q4_K from max/kernels quantises the
activations to int8 before it multiplies, so the reference it has to match is
ref_exact, which does the same quantisation in NumPy. Our own GPU kernel keeps
the activations in float32, so the reference it has to match is ref_float. Each
one is also reported against the other reference, and that second number is the
cost of quantising activations rather than an error.
"""

import sys
import numpy as np

path, label, gate = sys.argv[1], sys.argv[2], sys.argv[3]
M, N, K, k_bytes = (int(x) for x in open("shape.txt").read().split())

c = np.fromfile(path, dtype=np.float32).reshape(M, N)
refs = {
    "exact": np.fromfile("ref_exact.bin", dtype=np.float32).reshape(M, N),
    "float": np.fromfile("ref_float.bin", dtype=np.float32).reshape(M, N),
}
peak = float(np.abs(refs["float"]).max())

TOLERANCE = 1e-5
"""Maximum absolute error divided by the largest magnitude in the reference.
Stated relative to the peak and not per element because a dot product over 4864
terms lands near zero often enough that a per element relative error is
meaningless there. Float32 accumulation over that many terms costs a few times
1e-7, so 1e-5 leaves two orders of magnitude of headroom and is still tight
enough to catch a wrong scale, a swapped nibble or a mis-ordered group."""

worst = {}
for name, ref in refs.items():
    d = np.abs(c.astype(np.float64) - ref.astype(np.float64))
    worst[name] = d.max() / peak
    mark = "gate" if name == gate else "    "
    print("%-12s %s vs ref_%-6s max abs %.6g  over peak %.3g  rms %.6g" % (
        label, mark, name, d.max(), d.max() / peak, float(np.sqrt((d * d).mean()))))

ok = worst[gate] < TOLERANCE
print("%-12s      peak |ref| %.4f, gate %s, tolerance %g, %s" % (
    label, peak, gate, TOLERANCE, "pass" if ok else "FAIL"))
sys.exit(0 if ok else 1)
