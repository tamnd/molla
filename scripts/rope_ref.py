#!/usr/bin/env python3
"""Rope reference values, transcribed from ggml.

This is `rope_yarn` and the rope loop out of ggml's CPU backend, written out in
Python with nothing rearranged and nothing simplified. It is where the expected
numbers in `tests/test_rope.mojo` came from.

It is a transcription and not an independent derivation, and that is
deliberate. The question is whether molla rotates a vector the same way
llama.cpp does, because that is what a model's weights were trained against. A
second derivation from the YaRN paper would answer a different question and
would miss any place where ggml and the paper disagree.

Needs nothing but the standard library. Run it and read the numbers.
"""

import math


def yarn_ramp(low, high, i0):
    y = (i0 / 2 - low) / max(0.001, high - low)
    return 1 - min(1, max(0, y))


def corr_dim(n_dims, n_ctx_orig, n_rot, base):
    return (
        n_dims
        * math.log(n_ctx_orig / (n_rot * 2 * math.pi))
        / (2 * math.log(base))
    )


def corr_dims(n_dims, n_ctx_orig, base, beta_fast, beta_slow):
    start = math.floor(corr_dim(n_dims, n_ctx_orig, beta_fast, base))
    end = math.ceil(corr_dim(n_dims, n_ctx_orig, beta_slow, base))
    return max(0, start), min(n_dims - 1, end)


def rope_yarn(theta_extrap, freq_scale, cd, i0, ext_factor, mscale):
    theta_interp = freq_scale * theta_extrap
    theta = theta_interp
    if ext_factor != 0:
        ramp_mix = yarn_ramp(cd[0], cd[1], i0) * ext_factor
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix
        mscale *= 1.0 + 0.1 * math.log(1.0 / freq_scale)
    return math.cos(theta) * mscale, math.sin(theta) * mscale


def rope(
    x,
    pos,
    n_dims,
    base,
    freq_scale=1.0,
    ext_factor=0.0,
    attn_factor=1.0,
    beta_fast=32.0,
    beta_slow=1.0,
    n_ctx_orig=0,
    neox=True,
    ff=None,
):
    out = list(x)
    theta_scale = base ** (-2.0 / n_dims)
    cd = (
        corr_dims(n_dims, n_ctx_orig, base, beta_fast, beta_slow)
        if ext_factor
        else (0, 0)
    )
    theta = float(pos)
    for i0 in range(0, n_dims, 2):
        f = ff[i0 // 2] if ff else 1.0
        c, s = rope_yarn(theta / f, freq_scale, cd, i0, ext_factor, attn_factor)
        if neox:
            ic = i0 // 2
            a, b = x[ic], x[ic + n_dims // 2]
            out[ic] = a * c - b * s
            out[ic + n_dims // 2] = a * s + b * c
        else:
            a, b = x[i0], x[i0 + 1]
            out[i0] = a * c - b * s
            out[i0 + 1] = a * s + b * c
        theta *= theta_scale
    return out


def show(label, values):
    print(label + ":", [round(v, 7) for v in values])


def main():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    show("plain neox pos3 base10000", rope(x, 3, 8, 10000.0))
    show("plain norm pos3 base10000", rope(x, 3, 8, 10000.0, neox=False))
    show("linear f=4 neox pos8", rope(x, 8, 8, 10000.0, freq_scale=0.25))
    print("ntk base:", 10000.0 * (4.0 ** (8 / 6)))
    show(
        "yarn f=4 ctx=2048 pos8",
        rope(
            x,
            8,
            8,
            10000.0,
            freq_scale=0.25,
            ext_factor=1.0,
            n_ctx_orig=2048,
        ),
    )
    print("yarn corr dims:", corr_dims(8, 2048, 10000.0, 32.0, 1.0))
    print("yarn attn factor:", 1.0 + 0.1 * math.log(1 / 0.25))
    show(
        "factors 1 2 4 8 neox pos3",
        rope(x, 3, 8, 10000.0, ff=[1.0, 2.0, 4.0, 8.0]),
    )
    step = 10000.0 ** (-2 / 8)
    print("pair 1 at pos 3:", math.cos(3 * step), math.sin(3 * step))


if __name__ == "__main__":
    main()
