# molla

A local inference server written in pure Mojo. It speaks OpenAI, Anthropic, and MCP, so anything that already talks to those APIs works against it by changing one environment variable.

> **Status: early. Nothing works yet.** This repo currently holds the design, the build scaffolding, and the milestone plan. The first milestone (M0) is a spike to check that the whole idea is viable in Mojo at all. If it fails we will say so here and change course.

## What this is

molla loads local models and serves them over APIs other people already specified. There is no molla client library, no molla wire format, and no molla registry, because every one of those would be a reason you could not leave.

Concretely:

- **OpenAI HTTP**: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/responses`, `/v1/audio/*`
- **Anthropic Messages**: `/v1/messages` with the full streaming event sequence
- **MCP** (revision 2026-07-28): molla runs as an MCP server, and can act as an MCP client so tool calling works from any chat client
- **Models** are OCI artifacts following CNCF ModelPack, plus Hugging Face repos, plus plain GGUF and safetensors files on disk
- **Prompts** come from the model's own Jinja2 `chat_template` and `tokenizer.json`, not from renderers we hand write per model family

An optional Ollama compatibility adapter exists behind `--compat ollama`. It is off by default. It is there so existing tools connect, not because we are cloning anything.

## Why bother

There are good local runners already. The gap this fills is a runner with no proprietary dependency anywhere in the stack, that runs the same source on CPU, NVIDIA, AMD, and Apple GPUs, and that never asks you to sign in.

Mojo's compiler and standard library are Apache-2.0, and so is `max/kernels`, which is where the hard math lives (attention, paged KV cache, quantized matmul, MoE, collectives). The MAX runtime sits under a separate license, so molla does not use it. We write the engine and take the kernels. That one split is the reason a permissively licensed inference server in this stack is possible.

Five things we commit to, and test in CI:

1. Nothing molla produces is only readable by molla.
2. Nothing molla consumes requires molla.
3. No account, no telemetry, no network traffic you did not ask for.
4. No required proprietary dependency.
5. The exit is one command and it is tested.

## What it costs

Being honest about this up front. Mojo 1.0 has no async, no sockets, no HTTP, no JSON, no TLS, and no regex in the standard library. So before molla generates a single token we have to write, in Mojo:

| Layer | Why |
| --- | --- |
| Sockets and an event loop over epoll and kqueue | no `std.net` |
| HTTP/1.1 with chunked, SSE, and NDJSON framing | no HTTP stack |
| A SIMD JSON parser and serializer | no `std.json` |
| A Jinja2 subset for chat templates | no template engine |
| BPE, Unigram, and WordPiece tokenizers | no tokenizer |
| GGUF and safetensors readers | no model loaders |
| An OCI client, which needs TLS through FFI | no HTTP client |
| A paged KV cache batching executor | `max/kernels` gives kernels, not an engine |

That is a foundation project, not a wrapper project. M0 exists to find out in three weeks whether it is a reasonable one, and there is a written condition under which we move the network edge to Rust and keep Mojo for the engine and kernels.

## Design decisions

Each of these has a full writeup with the alternatives we rejected and the evidence that would reverse it.

| | Decision |
| --- | --- |
| D1 | Everything in Mojo. FFI limited to libc and the platform TLS library. No Python at runtime. |
| D2 | Open APIs are the product surface: OpenAI, Anthropic, MCP. |
| D3 | Ollama compatibility is an optional adapter, off by default. |
| D4 | Model semantics come from the model's own artifacts, not from per family code we maintain. |
| D5 | Distribution is OCI. No molla registry, ever. |
| D6 | The engine is ours, built on `max/kernels`. MAX runtime interop is optional. |
| D7 | One kernel source per op, four targets, numerics asserted on each. |
| D8 | Every milestone is validated on at least two device classes, one of them a GPU. |
| D9 | Local, offline, and accountless by default. |

See [docs/design.md](docs/design.md) for the reasoning and [docs/roadmap.md](docs/roadmap.md) for the plan.

## Hardware

| Target | Backend | Tier |
| --- | --- | --- |
| CPU x86-64 | AVX2 and AVX-512 | 1 |
| CPU aarch64 | NEON | 1 |
| NVIDIA, sm_89 primary | Mojo GPU to PTX | 1 |
| Apple GPU, M1 to M5 | Mojo GPU to Metal | 1 |
| AMD, gfx11xx | ROCm | 2 |

Tier 1 means every kernel is tuned, numerics run in CI on real hardware, and performance regressions block merges. Tier 2 means correct and tested at milestone boundaries, but not performance gated. AMD is tier 2 because we do not have the hardware, which is a real gap and not a hedge. Apple GPU is tier 1 but carries the most risk, because the toolchain supports it while upstream nightly CI does not cover it.

Windows is WSL only, matching the toolchain. That is not a hypothetical: the only NVIDIA GPU we develop against is an RTX 4090 in a Windows machine, reached through WSL2. So every CUDA result molla publishes is a WSL2 result until someone puts a card in a Linux box, and we label them that way rather than rounding up to "Linux CUDA". The machines are listed in [docs/validation/toolchain.md](docs/validation/toolchain.md).

## Building

You need [pixi](https://pixi.sh). It pulls the pinned Mojo toolchain, so there is nothing else to install.

```console
pixi install
pixi run build
pixi run test
./build/molla version
```

That builds and runs today on macOS arm64, Linux x86_64, and Linux arm64. It does not serve anything yet. `molla version` prints the toolchain and what it detected about your machine, which is all M0 promises. See [docs/validation/toolchain.md](docs/validation/toolchain.md) for the pinned version, the machines it has actually been run on, and the notes on what Mojo 1.0 turned out to look like in practice.

## Contributing

Issues are grouped by milestone and labelled by area. Anything tagged `good first issue` is genuinely self contained. Read [CONTRIBUTING.md](CONTRIBUTING.md) first, and for anything that changes a decision in `docs/design.md`, open an issue before the pull request.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
