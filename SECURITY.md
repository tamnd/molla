# Security

## Reporting

Report vulnerabilities privately through [GitHub Security Advisories](https://github.com/tamnd/molla/security/advisories/new). Do not open a public issue.

You should get a first response within 3 working days. We aim to ship a fix and publish an advisory within 90 days of the report, and we will credit you unless you would rather we did not.

## Supported versions

molla is pre-release. Until 1.0 only the latest tag gets fixes.

## What we consider a vulnerability

molla binds to loopback by default and executes no model supplied code, so the interesting surface is narrower than it looks. Things we treat as vulnerabilities:

- Anything that lets a request reach a device context, a file, or a network destination it should not
- Template escapes. Chat templates are untrusted input pulled from the internet, and the Jinja evaluator is sandboxed with no filesystem or network access, a step budget, and an output cap. An escape from that is a vulnerability
- DNS rebinding or Origin and Host handling failures that let a browser page reach a local molla
- Memory safety failures reachable from a request, including through the FFI boundary in `molla.sys`
- Anything that causes molla to make a network connection the user did not ask for. This is commitment 3 in the openness charter and we treat a breach of it as a security bug, not a feature regression
- Signature or digest verification that can be bypassed on a model pull

## What we do not

- Denial of service from a client that is already authorised to use the server. Bounded queues return 503 by design
- Running molla on a public interface without auth. It warns loudly and refuses without `--insecure`, and past that point it is your network
- Model output. Prompt injection is a property of the model, not of the server, with one exception below

## The MCP client is the sharp edge

molla can act as an MCP client, which means it executes tool calls on the model's behalf. That is the one place molla acts on model output, and it is off unless you configure it. Per server allow lists and a per request call budget are required settings, not suggestions. If you find a way around either, report it.
