# qmatmul spike

The M0 kernel spike for issue #5. It asks whether `max/kernels` is usable directly from our own Mojo code on an Apple GPU and an NVIDIA GPU without the MAX runtime, and it answers no. The full writeup with the numbers and what it means is in [docs/validation/kernels.md](../../docs/validation/kernels.md). This file is just how to run it.

Nothing here is molla code and nothing here is built by `pixi run build` at the repo root. It has its own manifest because it depends on `max-core`, which is `LicenseRef-Modular-Proprietary`, and that dependency is the finding rather than a choice. Keeping it in a separate directory with a separate manifest is what stops it leaking into the shipped build.

## What is in here

| File | What it does |
| --- | --- |
| `gen_data.py` | Pulls a Q4_K weight matrix out of a real GGUF file, makes an activation matrix, writes two NumPy references |
| `gen_q40.py` | Requantises the same weights to Q4_0 and rounds the activations to bfloat16, for the NVIDIA kernel |
| `qmatmul_cpu.mojo` | Calls `matmul_Q4_K` from `max/kernels`, which is CPU only |
| `qmatmul_gpu.mojo` | Our own Q4_K dequantise and multiply kernel, one source, runs on Metal and on CUDA |
| `qmatmul_nv.mojo` | Calls `matmul_gpu_qint4` from `max/kernels`, which is NVIDIA only |
| `appleprobe.mojo` | Asks `max/kernels` for its Apple GPU quantised matmul and prints what it says |
| `check.py` | Compares a Q4_K result against both references |
| `check_nv.py` | Compares the Q4_0 result against its bfloat16 reference |

## Running it

The generators need Python with `numpy` and `gguf`, which is tooling for this spike and not a molla dependency. Put a GGUF file with Q4_K tensors next to them, or pass the path.

```console
python gen_data.py ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
pixi run cpu && python check.py c_cpu.bin cpu exact
pixi run gpu && python check.py c_gpu.bin gpu float
```

The two gates differ on purpose. `matmul_Q4_K` quantises the activations to int8 before it multiplies, so it is checked against `ref_exact`, which does the same quantisation in NumPy. Our kernel keeps the activations in float32, so it is checked against `ref_float`. Each result is also printed against the other reference, and that second number is the cost of quantising activations rather than an error in anything.

The NVIDIA path needs its own fixture and an NVIDIA GPU:

```console
python gen_q40.py ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
pixi run nv && python check_nv.py
```

## Things that will bite you

The build is noisy. `max-core` 26.5.0 and Mojo 1.0.0 are mid migration on `alloc`, `bitcast` and pointer indexing, so both this code and `max/kernels` itself produce a page of deprecation warnings. They are warnings and the results are correct.

`K` for the NVIDIA kernel has to be a multiple of 1024. The failure when it is not is a constraint error inside `layout_tensor.mojo` about depth-1 layouts that never mentions K.

The Apple GPU quantised matmuls in `max/kernels` require an M5. On an M4 they raise at launch, which is why `qmatmul_gpu.mojo` exists at all.
