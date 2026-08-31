# Changelog

Notable changes per release. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

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

### Changed

- The toolchain version lives only in `pixi.toml` now, rather than also in a CI environment variable

### Known issues

- The Mojo compiler we build with comes from Modular's conda channel under a proprietary license, so the build is not yet Apache-2.0 end to end even though the source is. See `docs/design.md`.

molla answers HTTP requests as of the M0 spike, but every path returns the same fixed body. The first milestone that serves a model is M2.
