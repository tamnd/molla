# Changelog

Notable changes per release. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [0.1.1] - 2026-08-31

Two layers of the standard library Mojo 1.0 does not have. The OS boundary, and the memory the request path will live in. Nothing above them exists yet, so nothing here changes what molla can do, and both are the kind of thing that is much cheaper to get right before there is a server on top of it.

The sys layer is every OS call molla makes, in one module, returning one result type that carries errno from the call site. Files, threads, mutexes, condition variables and signals, all tested against a real kernel on three machines and both architectures.

The io layer is buffers, rings and arenas, with the growth policy of each written down next to the code rather than left to be inferred, and an allocation counter underneath so the zero allocation claim in M1 can be a number instead of a promise.

### Added

- `molla.sys` grows the rest of the OS boundary. `result.mojo` holds the one type every wrapper returns, carrying a value and the errno captured at the call site. `file.mojo` covers open, read, write, seek, truncate, sync, stat, unlink, rename and directory listing. `thread.mojo` covers pthreads, mutexes and condition variables, which is what Mojo 1.0 has no threading module for. `signal.mojo` covers dispositions, masks and a self pipe that turns a signal into a readable descriptor.
- `tests/test_sys.mojo`, which runs every wrapper against the real OS. FFI mistakes show up as memory corruption somewhere else entirely, so they are caught at the boundary or not at all.
- `access` behind `exists`, `writev` behind `write_vectored`, `socketpair`, unix domain sockets and `shutdown` in `molla.sys.socket`.
- `docs/validation/sys.md`, which records what the boundary covers, what ran green on which machine, and the four platform traps that cost a session each.
- `molla.io`, the memory layer the request path is built on. `buffer.mojo` is an owned growable buffer with a written down growth policy, doubling to 64 kB and then a fixed step. `ring.mojo` is the per connection output ring, so a short write costs two integer updates instead of a memmove, and it hands `writev` its one or two pieces directly. `arena.mojo` is a bump allocator with a per request lifetime, freed in constant time. `bytes.mojo` compares, searches, trims and parses spans without allocating.
- `molla.sys.mem`, which is where every allocation in molla goes, and the allocation counter that makes "this request allocated nothing" a number rather than a claim. Issue #17 is what the counter is for.
- `tests/test_io.mojo`, 116 checks over the growth policy, the ring wrap, the arena and the byte helpers.

### Changed

- `molla.sys.mmap` opens and closes through `molla.sys.file` instead of declaring `openat` a second time. Two declarations of one C symbol with different argument counts in the same build fail to lower, and the error points at the standard library rather than at either file that caused it.

## [0.1.0] - 2026-08-31

M0 is done. The question it existed to answer was whether Mojo 1.0 can hold a socket, parse HTTP fast enough, map a model file, call a kernel on a GPU and reach a TLS library, and the answer is yes on all five, with numbers rather than opinions behind each one. Two decisions were taken at the gate and both are recorded with the measurements next to them.

D1 holds. The network edge stays in Mojo, and the Rust fallback stays documented and untaken. On the M4 a trivial handler runs between 43705 and 249896 requests per second depending on how loaded the machine was, against a gate of 5000. A thousand concurrent connections held for sixty seconds on kqueue and on epoll with flat memory. The TLS binding pulls the same blob from ghcr.io through three different libraries on four machines. One condition, the multi threaded one, cannot be tested because Mojo 1.0 has no threading module, so it moves to the M1 gate rather than being rounded up.

D6 does not. `max/kernels` needs proprietary `max-core` at runtime, its CPU kernels included, so the promise of an optional MAX runtime was describing a seam that does not exist. molla accepts the dependency, which means running molla means installing a proprietary runtime, and the README says so instead of claiming otherwise.

This release is still a foundation. It does not serve a model. M2 is the first one that does.

### Added

- `docs/adr/`, for decisions taken at a gate against measurements, with `0001-network-edge-stays-in-mojo.md` and `0002-accept-max-core.md`, the two M0 gate records

### Changed

- D1 in `docs/design.md` records the M0 gate outcome. The network edge stays in Mojo. The multi threaded half of the third reversal condition moves to the M1 gate, because Mojo 1.0 has no threading module to test it with.
- D6 is rewritten. `max-core` is a required dependency at runtime rather than an optional backend, because `max/kernels` does not run without it and its CPU kernels do not either. Running molla now means installing a proprietary runtime, and the README names both proprietary packages and what each is needed for.
- D7 is marked load bearing. `max/kernels` has no quantized matmul that will launch on an Apple GPU below an M5, so every Apple GPU kernel is one molla writes, and D7's per target numerics tests are what keep the portability claim honest.

## [0.0.3] - 2026-08-31

The M0 kernel spike ran, and it changed what the README is allowed to claim. The TLS spike ran after it, and molla can now pull a blob from ghcr.io over HTTPS on macOS and Linux.

### Added

- `molla.tls`, client TLS over OpenSSL 3.x and 1.1.1 on Linux and Secure Transport on macOS, both loaded with dlopen so a machine without a TLS library still runs molla and only loses HTTPS
- `molla.http.client`, a GET only HTTPS client with redirect following and chunked bodies, and `molla.registry.ghcr`, enough of the OCI distribution protocol to fetch a blob and check its digest
- `molla tls <host>` prints the backend, protocol, cipher and certificate chain, and `molla pull <ref>` pulls a blob from ghcr.io and verifies it
- `molla.sys.dns` for `getaddrinfo`, `molla.sys.sha256`, `molla.sys.cstr`, and `dial` in `molla.sys.socket` for a blocking socket with timeouts
- `MOLLA_LIBSSL` and `MOLLA_LIBCRYPTO` to point at a specific OpenSSL, which is also how the 1.1 fallback gets tested on a machine that has 3.x
- `docs/validation/tls.md` with the results from four machines and three TLS libraries
- `spikes/qmatmul/`, the kernel spike for issue #5, with its own pixi manifest so its proprietary dependency stays out of the root build
- `docs/validation/kernels.md` with the licence audit, the numbers from six machines, and the three options for what molla does next
- Numerics tolerances for Q4_K matmul on CPU and on GPU, which D7 asked for and never gave

### Changed

- The README no longer claims there is no proprietary dependency in the stack, because there is, and it names it
- D6 in `docs/design.md` is marked under review, since `max/kernels` does not build or run without proprietary `max-core`, its CPU kernels included
- D7 in `docs/design.md` is marked achievable but not inherited, since one source did compile to Metal and sm_89 with byte identical output, but `max/kernels` is not organised that way

### Known issues

- Carried over from 0.0.2 unchanged: the compiler is proprietary, so releases are source only. Build with `pixi run build`.
- `max/kernels` has no quantized matmul that will launch on an Apple GPU below an M5. On an M4 it raises at launch. The spike wrote its own kernel to get a Metal number at all.
- What molla actually does about the licence finding is not decided here. That is issue #7.
- TLS on macOS caps at 1.2. Secure Transport has no TLS 1.3, and the framework that does is built on Objective-C blocks, which Mojo cannot emit. Linux gets 1.3 through OpenSSL.
- The HTTPS client is IPv4 only, opens a connection per request, and reads bodies into memory whole. None of that is suitable for pulling a model and M3 replaces it.

## [0.0.2] - 2026-08-31

Four of the seven M0 spikes are done. molla can now map a model file and read what is in it, though it still cannot read a tensor.

### Added

- `molla.sys.mmap`, a read only whole file memory map
- `molla.model.gguf`, a GGUF v2 and v3 metadata reader that walks the header, the key value block and the tensor directory in place without copying the file, and `molla gguf <path>` to dump one
- `docs/validation/gguf.md` with the comparison against `gguf-dump` on four models covering bert, llama, gemma3 and qwen2, and what the zero copy read is actually worth

### Known issues

- Carried over from 0.0.1 unchanged: the compiler is proprietary, so releases are source only. Build with `pixi run build`.
- Nothing reads a tensor. The GGUF reader records where each one is and what type it is, and stops there.
- Metadata arrays are measured and skipped rather than decoded, so there is no way to read a tokenizer vocabulary yet.

## [0.0.1] - 2026-08-31

First tagged release. Three of the seven M0 spikes are done: the toolchain is pinned across the fleet, sockets and the event loop work on epoll and kqueue, and HTTP/1.1 parse and respond clears the throughput gate. Nothing serves a model yet.

### Added

- Pixi workspace with the Mojo toolchain pinned to 1.0.0, locked for macOS arm64, Linux x86_64, and Linux arm64
- A `molla` binary with `version` and `help`, reporting the toolchain and detected host
- A test runner, since Mojo 1.0 has no `mojo test`
- CI builds and tests on all three platforms for real, and smoke tests the binary
- `docs/validation/toolchain.md` recording the pin, the machines validated so far, and what Mojo 1.0 actually looks like against the release notes
- `molla.sys`, the libc boundary: errno, descriptors, IPv4 TCP sockets, and one `Poller` over kqueue and epoll with read and write interest
- `molla.net.echo`, a non blocking edge triggered TCP echo server, and `molla echo` to run it
- `molla soak`, which holds a thousand connections for sixty seconds and checks for descriptor and memory leaks
- `docs/validation/sockets.md` with the soak results on all three platforms and what the spike says about D1
- `molla.http`, a zero copy HTTP/1.1 request parser and a prebuilt response with an in place `Date` field, and `molla http` to run the throughput spike
- `docs/validation/http.md` with the M0 throughput measurements, the fleet results, and the two allocation and socket problems that cost more than the parser did
- Design document, roadmap, and milestone plan
- CI with docs linting, workflow linting, CodeQL on workflow definitions, OpenSSF Scorecard, and dependency review
- Release pipeline with SBOM, build provenance attestation, and keyless signing
- `scripts/check-action-pins.sh`, run in CI, which fails if an action is pinned to an annotated tag object rather than a commit

### Changed

- The toolchain version lives only in `pixi.toml` now, rather than also in a CI environment variable

### Fixed

- Six actions were pinned to annotated tag object SHAs instead of commit SHAs, which made the OpenSSF Scorecard workflow fail on publish with an imposter commit error even though the scan itself succeeded

### Known issues

- The Mojo compiler we build with comes from Modular's conda channel under a proprietary license, so the build is not yet Apache-2.0 end to end even though the source is. See `docs/design.md`.
- Releases are source only for the same reason. A Mojo 1.0 binary links `libKGENCompilerRTShared` and two other runtime libraries that ship only as shared objects under `LicenseRef-Modular-Proprietary`, and the linker bakes RUNPATH to the pixi directory that built it, so a bare binary does not start anywhere else. Publishing a working tarball would mean redistributing Modular's runtime inside molla's own artifacts. Build with `pixi run build` until that changes.

molla answers HTTP requests as of the M0 spike, but every path returns the same fixed body. The first milestone that serves a model is M2.
