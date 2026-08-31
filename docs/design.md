# Design

This is the short version. Each decision below records what we chose, why, what we rejected, and what evidence would make us change our mind. A decision without a reversal condition is a belief, not an engineering choice.

## D1: everything in Mojo

All of molla is Mojo. The HTTP server, JSON, TLS bindings, Jinja engine, tokenizers, GGUF and safetensors readers, OCI client, scheduler, executor, and kernels. Foreign code is limited to the C ABI through `std.ffi`, in three places only:

| Allowed | Why | Isolated in |
| --- | --- | --- |
| libc and libSystem | sockets, mmap, epoll and kqueue, threads, clocks, dlopen | `molla.sys` |
| Platform TLS | HTTPS for registry pulls | `molla.tls` |
| Vendor GPU runtimes through Mojo's `gpu` layer | device dispatch | `molla.device` |

Banned: `std.python` at runtime, any C, C++, Rust, or Go source in the build, any runtime dependency on the MAX runtime or a Python interpreter.

One language means one debugger, one build, one performance model, one binary, and the same source compiling to CPU, CUDA, ROCm, and Metal. It is also the only configuration where molla is Apache-2.0 the whole way down.

We rejected a Go control plane, and then a Rust one. Rust remains the documented fallback because `axum`, `minijinja`, the Hugging Face `tokenizers` crate, and `oci-client` are all mature.

**Decided at the M0 gate, 2026-08-31.** D1 holds. Throughput cleared the gate by at least 8x on the worst measurement taken, the TLS binding works in both directions across three libraries and two operating systems, and the borrow checker was not the problem anybody expected it to be. One condition could not be tested, because Mojo 1.0 has no threading module, and it moves to the M1 gate. Record and numbers in [adr/0001-network-edge-stays-in-mojo.md](adr/0001-network-edge-stays-in-mojo.md).

**Reversal condition, decided at the M1 gate.** If sustained HTTP throughput on the M4 is under 5000 requests per second for a trivial handler, or the TLS FFI binding proves unstable across macOS and Linux, or memory safety diagnostics make a non blocking multi threaded socket server impractical in Mojo 1.0, then the network edge moves to Rust and Mojo keeps the engine and kernels. The module boundaries are drawn so that swap costs weeks rather than a rewrite.

**TLS, answered at M0.** The binding holds. `molla pull` fetches and verifies a blob from ghcr.io through OpenSSL 3.x, OpenSSL 1.1.1 and Secure Transport, all dlopened, with no C in the build and no bundled CA list. One limit came out of it: Secure Transport has no TLS 1.3, and the Apple framework that does is built on Objective-C blocks, which Mojo cannot emit, so macOS caps at TLS 1.2 until someone solves that. Details in [validation/tls.md](validation/tls.md).

## D2: open APIs are the product surface

molla speaks OpenAI HTTP, Anthropic Messages, and MCP revision 2026-07-28, as both an MCP server and an MCP client. These are the interfaces the ecosystem already targets. Implementing them means every SDK, IDE plugin, and agent runtime works without anyone adopting a molla specific client.

We rejected a molla native API. Not being a clone means having no proprietary surface of our own, not inventing a third one. The only molla specific routes are under `/molla/*` for operations that no standard covers, and none of them are required for inference.

## D3: Ollama compatibility is an optional adapter

A translation layer maps a subset of `/api/*` onto the open core, enabled only with `--compat ollama`. Deliberately excluded: sign in and account endpoints, cloud passthrough, Modelfile as canonical config, and the experimental endpoints. Those are the parts that tie you to a vendor.

The adapter never gets its own engine path. Anything it cannot express is an error, not a second implementation.

**Reversal condition.** If it costs more than about a day a quarter to maintain, or forces a compromise in the open core, it is frozen and marked legacy.

## D4: model semantics come from the model's own artifacts

Prompt formatting uses the model's Jinja2 `chat_template`. Tokenization uses `tokenizer.json`. Architecture comes from `config.json` or GGUF metadata.

The alternative is hand written renderers per model family, which means a new renderer with every notable model release. Using the artifact the model author shipped is more open and less work, and a brand new model works on day zero without a molla release.

The cost is real and we are not hiding it. Jinja2 is a large language and chat templates use awkward corners of it. Our answer is that the supported subset is pinned, and an unsupported construct is a named error at model load time rather than a silent misrender in the middle of someone's conversation. Misrendering must be impossible. Failing must be loud.

## D5: distribution is OCI

Models are OCI artifacts following CNCF ModelPack, stored locally in a content addressed store, pullable from any registry, from Hugging Face, or from local files. molla can also read an existing Ollama store for migration, by hardlink, so it costs zero bytes.

This makes model distribution a solved infrastructure problem. Deduplication, resumable pulls, sigstore signing, mirroring, and air gapped installs all come for free. A molla registry would be exactly the vendor move we are rejecting.

## D6: the engine is ours, built on max/kernels

molla implements its own executor: weight loading, layer composition, paged KV cache, continuous batching, sampling, speculative decoding, and LoRA. It calls kernels from `max/kernels`, which is Apache-2.0, and adds its own where there are gaps. The MAX runtime is an optional backend and never required.

`max/kernels` supplies the hard math under a permissive license. The engine around it is systems work we want to own anyway, because it defines scheduling behaviour and memory policy. Owning it is also what keeps molla clear of the Modular Community License.

**Reversal condition.** If our executor is more than twice as slow as MAX on the same hardware six months in, we flip the default to the MAX backend where it is available, keep the native engine as the portable path, and say so publicly.

**Status: under review.** The M0 kernel spike found that the second paragraph above is wrong. `max/kernels` is Apache-2.0 source that does not compile or run without `max-core`, which is proprietary, and that is true of its CPU kernels as well as its GPU ones. It also found that molla already links a proprietary support library out of the compiler package, so the licensing split this decision was built on does not exist. `max/kernels` also has no quantized matmul that will launch on an Apple GPU below an M5. The evidence is in [docs/validation/kernels.md](validation/kernels.md) and the decision is issue #7.

## D7: one kernel source, four targets, numerics asserted

Every operation has a single Mojo source compiled for CPU, NVIDIA, AMD, and Apple GPU. Divergence lives in tile parameters and small compile time branches inside one function, never in separate files, so a fix cannot be applied to three targets and forgotten on the fourth.

Every operation has a naive reference implementation, an external oracle where one exists, and numerics tests that run on each target with tolerances stated per dtype. Portable is a claim that decays silently. Only per target tests keep it true.

A target that cannot pass numerics is listed as unsupported. It does not ship as probably fine.

**Status: achievable, not inherited.** The M0 kernel spike wrote one naive Q4_K matmul with no compile time branches at all, compiled it for Metal and for sm_89, and got byte identical output from an M4 and a 4090. So the claim holds for code we write. It is not how `max/kernels` is organised, which has three separate quantized matmul implementations in three separate places with different entry points, weight layouts and dtypes. The first two tolerances the second paragraph promised are also now stated, in [docs/validation/kernels.md](validation/kernels.md), because this decision asked for them without giving any.

## D8: real hardware gates every milestone

No milestone closes without passing on at least two device classes, one of which is a GPU. The fleet is an M4 MacBook Air, three CPU only Linux servers, and an RTX 4090 reached through WSL2 on a Windows machine. CI runs on the Linux servers, with M4 and 4090 runs required at milestone boundaries. See [docs/validation/toolchain.md](validation/toolchain.md) for what each machine actually is.

Two consequences of that fleet are worth stating plainly rather than discovering later. Every CUDA number we produce is a WSL2 number, because that is the only NVIDIA GPU we have, and WSL2 does not have the same transfer or pinned memory behaviour as a bare metal Linux host. We label those results as WSL2 results. And three of the five machines have no GPU at all, which keeps the CPU path honest but also means CPU is the path that gets the most incidental testing while the GPU paths get the least.

This exists because Apple GPU support is documented but not covered by upstream nightly CI, and consumer NVIDIA is described upstream as known compatible for development rather than tested for serving. Those are exactly the conditions where paper support and real support drift apart.

## D9: local, offline, accountless

No telemetry. No account. No outbound connection except pulls you asked for. Loopback bind by default. Every network destination is user configured and printable with `molla config sources`. The point is that you can verify it with `tcpdump`.

## Architecture

One process, one language, layered so each layer could be replaced. Arrows point downward only. The API layer never touches the engine directly, it goes through the scheduler, and the engine never parses HTTP or JSON. A build lint checks this, because layering that is not enforced does not exist.

```text
edge        sys, tls, io, net, http, json, uri, time, log, metrics
api         openai, anthropic, mcp, admin, compat.ollama
semantics   jinja, chat, tokenizer, parse, grammar
model       gguf, safetensors, store, oci, hub, modelspec
engine      device, tensor, kernels, model, kv, batch, sample, spec, lora
control     sched, config, cli, tui
```

Mojo has no async, so concurrency is explicit. One acceptor, a pool of I/O workers running epoll or kqueue, a pool of request workers for templating and tokenization, one runner thread per loaded model which owns its device context exclusively, and one housekeeping thread. Handoff is over bounded queues with backpressure, so a full queue returns 503 rather than growing memory. No thread ever touches a device context it does not own, which removes the largest source of GPU driver misbehaviour.

A connection is a state machine struct rather than a suspended stack, since there are no coroutines to suspend. That is more code than async would be, but it gives predictable memory per connection and no scheduler overhead. When Mojo ships async this layer can be rewritten behind the same interface and nothing above it changes.

## Non-goals

1. Training and fine tuning. Serving LoRA adapters yes, producing them no.
2. A desktop GUI. A terminal UI ships. GUIs are other people's clients talking OpenAI or MCP.
3. A hosted service or a model registry. Both are the lock-in we are removing.
4. Windows native. WSL only, matching the toolchain.
5. Multi node clusters. Multiple GPUs in one host is in scope. Racks are not.
6. Byte for byte parity with any specific vendor API, including Ollama's.
7. A curated model library. We point at OCI registries and Hugging Face.

## Where molla is less open than the alternatives

Stating this is part of the point, since a one sided comparison is the thing we are objecting to.

The toolchain is single vendor. Mojo is Apache-2.0 but developed primarily by Modular, so a change of direction affects molla in a way it would not affect a Rust or C++ project. We mitigate with pinned toolchain versions, a vendored kernel snapshot, and keeping D1's reversal path alive, but the risk is real.

Worse than that, and only found once we actually installed it: the Mojo compiler we build with is not the open source one. The language and standard library are Apache-2.0, but the prebuilt package on Modular's conda channel is distributed under `LicenseRef-Modular-Proprietary`, and that is what `pixi install` pulls. So molla's source and molla's dependencies are Apache-2.0 while its compiler is not, which makes the README's "Apache-2.0 all the way down" true of the artifact and not yet true of the build. Building the toolchain from the open source tree closes the gap. We have not done it. Until we do, this is the largest hole in the openness charter and it is tracked in `docs/validation/toolchain.md`.

Apple GPU support depends on a path that upstream nightly CI does not cover. AMD is tier 2 for lack of hardware. And we do not host a registry, which avoids lock-in but does mean molla is less turnkey than a tool with a curated library. Refusing to default to any registry is a deliberate convenience cost.
