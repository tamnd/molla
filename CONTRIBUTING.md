# Contributing

Thanks for looking. molla is early, so the most useful contributions right now are in the systems layer and the test corpora rather than in features.

## Before you write code

Open an issue first for anything that is not a typo or an obvious bug fix. For anything that changes a decision in `docs/design.md`, open the issue and let it settle before the pull request, because those decisions have reversal conditions attached and changing one is a design change rather than a code change.

Issues are grouped by milestone. Work that belongs to a later milestone will usually sit until the milestone before it lands, because each milestone's acceptance depends on the previous one being real. That is not a judgement on the idea.

## Ground rules

**Everything is Mojo.** No C, C++, Rust, or Go in the build. FFI is limited to libc and the platform TLS library, and it lives in `molla.sys` and `molla.tls` only. Python is allowed in tests as an oracle and is banned at runtime.

**Layering is enforced.** The API layer never calls the engine directly, it goes through the scheduler. The engine never parses HTTP or JSON. There is a build lint for this.

**No new required dependency** without a decision record explaining the tradeoff.

**Errors are documentation.** Every user facing error carries a machine readable code, a measurement, and at least one concrete next command. An error that says only what went wrong is incomplete.

## Tests

A pull request needs tests that fail without the change. Beyond that:

- Kernels need a naive reference, an external oracle where one exists, and numerics on every target with tolerances stated per dtype
- Anything touching prompt rendering or tokenization is compared against a Python oracle for exact string or exact token equality, not approximate agreement
- Anything touching the engine or kernels gets run on at least two device classes with one of them a GPU, using the `needs: hardware` label to queue it on the fleet
- Performance sensitive paths have a benchmark, and a 5 percent regression on a tier 1 target blocks the merge

If a test is slow, make it a nightly. If a test is flaky, fix it or delete it. A skipped test is a lie in the test count.

## Commits and pull requests

Conventional commit prefixes: `feat`, `fix`, `perf`, `docs`, `test`, `refactor`, `ci`, `chore`. Keep the subject under 72 characters and write it in the imperative.

Keep pull requests small enough to review in one sitting. If a change has a mechanical part and a thinking part, send them separately.

## Style

Plain English in docs and comments. Short sentences. Say what the code does and why it is not obvious, and skip comments that restate the line below them.

Comment the surprising parts. If you spent an hour working out why something has to be done a particular way, that hour belongs in a comment.

## Local setup

Once M0 lands:

```console
pixi install
pixi run build
pixi run test
```

Until then there is nothing to build, and the CI build job skips on purpose rather than failing.

## Reporting security issues

Do not open a public issue. Use the private advisory link in `SECURITY.md`.
