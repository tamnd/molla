#!/usr/bin/env python3
"""Write the quantization conformance fixtures.

For each ggml block format this writes two files into the output directory: a
`.bin` of raw block bytes, and a `.f32` of the little endian float32 values the
`gguf` package decodes those bytes to. `scripts/quant_oracle.mojo` reads both
and reports the largest disagreement.

The block bytes are random rather than taken from a model. That is deliberate.
Weights out of a real file are well behaved, so a decoder that gets a nibble
order or a sign wrong on the rare paths still matches on real data and fails
later inside a matmul, which is a much worse place to find out. Random bytes hit
every branch of the bit unpacking on the first block.

The one thing that cannot be random is a scale, because a random pair of bytes
read as a float16 is a NaN or an infinity about one time in every hundred and
twenty, and comparing NaN to NaN says nothing. So the scale fields are written
as ordinary float16 values in a sane range and everything else in the block is
whatever the generator produced.

Deterministic from a fixed seed, so CI regenerates the same fixtures rather
than carrying a few megabytes of binary in the repository.

Give it a GGUF file as a third argument and it writes a second set of fixtures
named `<type>-real`, taken out of that model. Random bytes prove the decoder
reads the format; real bytes prove it is pointed at the right offset in a file
somebody else wrote, which is a different mistake.

Usage:

    scripts/gen-quant.py corpus/quant [blocks-per-type] [model.gguf]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from gguf.constants import GGMLQuantizationType as T
from gguf.quants import dequantize

# Name, ggml type, bytes per block, elements per block, and the byte offsets
# inside a block that hold a float16 scale.
FORMATS = [
    ("f16", T.F16, 2, 1, [0]),
    ("bf16", T.BF16, 2, 1, []),
    ("q4_0", T.Q4_0, 18, 32, [0]),
    ("q4_1", T.Q4_1, 20, 32, [0, 2]),
    ("q5_0", T.Q5_0, 22, 32, [0]),
    ("q5_1", T.Q5_1, 24, 32, [0, 2]),
    ("q8_0", T.Q8_0, 34, 32, [0]),
    ("q4_k", T.Q4_K, 144, 256, [0, 2]),
    ("q5_k", T.Q5_K, 176, 256, [0, 2]),
    ("q6_k", T.Q6_K, 210, 256, [208]),
]


def scales(rng: np.random.Generator, count: int) -> np.ndarray:
    """Float16 values that are neither denormal nor absurd.

    A model scale is a small positive number, but negative ones are legal and
    the sign has to survive the decode, so both signs are generated here.
    """
    mag = np.exp(rng.uniform(np.log(1e-3), np.log(2.0), size=count))
    sign = rng.choice(np.array([-1.0, 1.0]), size=count)
    return (mag * sign).astype(np.float16)


def blocks(rng: np.random.Generator, fmt, count: int) -> np.ndarray:
    name, _, block_bytes, _, scale_at = fmt
    if name == "f16":
        return scales(rng, count).view(np.uint8).reshape(count, 2)
    if name == "bf16":
        # A bfloat16 is the top half of a float32, so generating float32 values
        # and dropping the low two bytes gives every bit pattern that can occur
        # in a real tensor and none of the ones that cannot.
        wide = scales(rng, count).astype(np.float32)
        return wide.view(np.uint8).reshape(count, 4)[:, 2:4].copy()
    raw = rng.integers(0, 256, size=(count, block_bytes), dtype=np.uint8)
    for at in scale_at:
        half = scales(rng, count).view(np.uint8).reshape(count, 2)
        raw[:, at : at + 2] = half
    return raw


def write(out: pathlib.Path, name: str, raw: np.ndarray, ref: np.ndarray) -> None:
    (out / f"{name}.bin").write_bytes(raw.tobytes())
    (out / f"{name}.f32").write_bytes(ref.astype("<f4").tobytes())


def from_model(out: pathlib.Path, path: str, count: int) -> int:
    """Fixtures cut out of a real model, one per type the file happens to use.

    The first `count` blocks of the first tensor of each type, which is enough
    to cover the format and small enough that generating them is not a second
    load of the model.
    """
    from gguf import GGUFReader

    reader = GGUFReader(path)
    sizes = {kind: (b, e) for _, kind, b, e, _ in FORMATS}
    seen = set()
    for tensor in reader.tensors:
        kind = tensor.tensor_type
        if kind in seen or kind not in sizes:
            continue
        block_bytes, block_elems = sizes[kind]
        raw = tensor.data.reshape(-1)
        if raw.dtype != np.uint8:
            raw = raw.view(np.uint8)
        take = min(count, raw.size // block_bytes)
        if take == 0:
            continue
        raw = raw[: take * block_bytes].reshape(take, block_bytes)
        ref = dequantize(raw, kind).astype(np.float32).reshape(-1)
        name = f"{kind.name.lower()}-real"
        write(out, name, raw, ref)
        print(f"{name}  {take} blocks from {tensor.name}  {ref.size} values")
        seen.add(kind)
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    out = pathlib.Path(sys.argv[1])
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260902)
    for fmt in FORMATS:
        name, kind, block_bytes, block_elems, _ = fmt
        raw = blocks(rng, fmt, count)
        ref = dequantize(raw, kind).astype(np.float32).reshape(-1)
        want = count * block_elems
        if ref.size != want:
            print(
                f"{name}: decoded {ref.size} values, expected {want}",
                file=sys.stderr,
            )
            return 1
        if not np.isfinite(ref).all():
            print(f"{name}: reference has a NaN or an infinity", file=sys.stderr)
            return 1
        write(out, name, raw, ref)
        print(f"{name}  {count} blocks  {raw.nbytes} bytes  {ref.size} values")

    if len(sys.argv) > 3:
        return from_model(out, sys.argv[3], count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
