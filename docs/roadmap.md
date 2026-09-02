# Roadmap

Ordered so the riskiest assumptions get tested first and something runs end to end as early as possible. Every milestone has acceptance criteria you can check, and closes only when it has passed on at least two device classes with one of them a GPU.

Durations are rough calendar estimates for a small team. The ordering matters more than the numbers.

## M0: feasibility spike, 2 to 3 weeks

This is the go or no go. It exists to test the pure Mojo bet before anything is built on top of it, and it is designed to be cheap to fail.

| Deliverable | Acceptance |
| --- | --- |
| Toolchain on M4, the 4090 box, and one Linux server | `mojo build` produces a running binary on all three |
| TCP echo server on `std.ffi` with epoll and kqueue | 1000 concurrent connections for 60 seconds, no leaks, on macOS and Linux |
| Minimal HTTP/1.1 parse and respond | at least 5000 requests per second for a trivial handler on the M4 |
| mmap a GGUF and read its metadata | tensor directory printed correctly for three real files |
| One `qmatmul_k` from `max/kernels` on the M4 GPU and the 4090 | matches a NumPy reference within tolerance |
| TLS through dlopen | pull a blob from ghcr.io over HTTPS on both platforms |

If HTTP throughput, FFI stability, or GPU dispatch fails, D1 is reversed here, before any dependent work exists.

## M1: systems layer, 6 to 8 weeks

The whole edge: `sys`, `io`, `net`, `http`, `json`, `tls`, threading, config, logging, metrics. Plus `/molla/version`, `/molla/health`, and `/molla/metrics`.

Acceptance: systems suite green on macOS and Linux, zero allocation parse path verified, a one hour soak with mixed keep alive and streaming traffic stays stable, SSE and NDJSON writers stay correct under slow readers, and shutdown drains cleanly.

## M2: first tokens, 8 to 10 weeks

GGUF reader to model spec, a Llama and Qwen dense architecture, weight loading, single sequence prefill and decode on CPU and one GPU, sampling, the tokenizer, the Jinja engine, and `/v1/chat/completions`, `/v1/completions`, and `/v1/models` with streaming.

Acceptance: a real q4_K_M model generates coherent text on Linux CPU, macOS and Windows. Logits agree with llama.cpp within tolerance on the host kernels. The template and tokenizer suites are green for the covered families. The OpenAI Python SDK streams successfully.

This is the milestone that turns the project from a document into something people can try.

The acceptance line above used to ask for the M4 GPU and the 4090 as well, and the GPU half of it moved. Nothing in M2's scope wrote a device kernel, so the milestone was being asked to pass on three backends while having one. That work is M2b, which is the next section and did not exist when this was first written.

M2 does not close on the strength of that rewrite. The rule at the top of this page is that a milestone closes only when it has passed on at least two device classes with one of them a GPU, and it is D8 in the design, and a milestone is not the place to make an exception to it. M2's checklist is complete and M2 stays open until a real model generates text on the M4 GPU and on the 4090, which is the point where the fourth item of M2b lands.

## M2b: kernels on the GPU, 6 to 8 weeks

The milestone where molla stops being a CPU program. Device buffers and weight residency, a quantized matvec written for Metal and CUDA, the rest of a block beside it, the KV cache in device memory, and a backend that is chosen and printed rather than assumed.

`max-core` is already required at runtime for every token molla generates, which is D6, and `molla.sys.device` already enumerates through it. A kernel launch is the part of that dependency the engine has not used yet. `max/kernels` has no quantized matmul that will launch on an Apple GPU below an M5 and the M4 is the reference machine, so the Apple GPU kernels are ours and the same source has to compile for the 4090.

Acceptance: `molla generate` produces the same greedy text on the host, the M4 GPU and the 4090 for the same model, prompt and seed. The fourteen case logit corpus from M2 passes on all three backends against the same committed reference files, with tolerances stated per target. `docs/validation/bench.md` has a table per fleet machine with molla, llama.cpp and Ollama in it.

The bench table is not a gate here. Being faster than llama.cpp is an M7 gate with a number attached. What this milestone owes is knowing the distance, because a performance number first taken at the end is a number nobody can bisect.

## M3: serving properly, 8 to 10 weeks

Paged KV cache, continuous batching, chunked prefill, prefix caching, admission control, runner lifecycle and keep alive, fit estimation, and eviction. Anthropic `/v1/messages` with the full event sequence. Embeddings.

Acceptance: 16 concurrent streams on the 4090 without latency collapse, the prefix cache hit path is at least 10 times faster than recompute, preempt and resume produces identical greedy output, the `claude` CLI completes a multi turn session, and performance baselines are recorded on every fleet device.

## M4: open ecosystem, 6 to 8 weeks

OCI and ModelPack pull and push, the Hugging Face resolver, store garbage collection and verification, import from an Ollama store, and `molla build`. The MCP server over Streamable HTTP and stdio. `/v1/responses`. The Ollama compatibility adapter. Tool calling with declarative parser descriptors. Structured outputs and grammars.

Acceptance: pull the same model from ghcr, Hugging Face, and a local zot registry. Import an Ollama store by hardlink with zero copies. MCP conformance tooling green. Tool calling round trips through both the OpenAI and Anthropic surfaces. Grammar tests pass for soundness and for completeness, since a grammar that masks out valid continuations degrades quality silently.

## M5: depth, 8 to 10 weeks

Reasoning with budget enforcement, the MCP client, LoRA serving with batched adapters, speculative decoding, native quantizers, multi GPU replica mode, and the operator tooling: `molla fit`, `molla why`, `molla trace`, `molla bench`, and the terminal UI.

Acceptance: speculation passes a distribution test against non speculative sampling, several LoRAs are served concurrently from one base model, the quantizer is bit exact against GGML reference vectors, and reasoning text never leaks into `content` on any supported family.

## M6: multimodal and audio, 8 to 10 weeks

Vision with the projector and patch embedding and placeholder accounting, audio transcription and speech, reranking, fp8 paths, and tensor parallel multi GPU.

Acceptance: a vision language model answers about the same image identically on the M4 and the 4090, placeholder count mismatches are hard errors rather than silent sequence corruption, and audio round trips through the OpenAI SDK.

## M7: 1.0, ongoing

Performance work against the published gates, promoting AMD if we get hardware, calibrated quantization, packaging, documentation, and the first stability guarantee: no breaking changes to the API surface or the artifact format within a major version.

For 1.0 specifically, decode throughput has to be within 1.5 times of llama.cpp on the same model, quantization, and device across every tier 1 target, and there has to be a published validation table for every machine in the fleet.

## Why this order

The riskiest thing goes first, so M0 tests the pure Mojo bet in weeks rather than months, and M2 tests the own the engine bet before batching complexity is added on top of it.

Something usable arrives at M2. A single stream OpenAI compatible server on real hardware is already useful, and it makes every later milestone measurable against real usage instead of against benchmarks we picked.

Ecosystem work comes after correctness. OCI, MCP, and the compatibility adapter are valuable but additive, and shipping them before generation is correct would produce a very well connected system that generates the wrong tokens.

Multimodal comes late because it multiplies the correctness surface and depends on everything else being stable.

Kernels, test corpora, and documentation can run in parallel from M1 onward. The critical path of systems to engine to serving to ecosystem is strictly sequential, because each stage's acceptance depends on the previous one being real.
