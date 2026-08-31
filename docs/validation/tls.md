# Client TLS through dlopen

Two issues, in order. Issue #6 is the M0 spike: can molla do HTTPS by loading the platform's TLS library at runtime rather than linking it, on macOS and on Linux, with the spike done when a real blob comes down from ghcr.io over HTTPS on both. It can. The same 1744 byte blob, with the same SHA-256, on four machines and three different TLS libraries. Issue #14 is the M1 job of making that fit to use, which is the insecure flag being per registry rather than global, and a machine with no TLS library starting anyway and losing only HTTPS. That part starts at [Productionising](#productionising-issue-14).

There is one finding that is not good news and it is in the macOS section: the only usable Apple TLS API caps at TLS 1.2.

## What was built

| Module | What it is |
| --- | --- |
| `molla.sys.cstr` | NUL terminated strings in and out of C |
| `molla.sys.dns` | `getaddrinfo`, IPv4 only |
| `molla.sys.sha256` | SHA-256, because a pulled blob has to be checked against the digest that named it |
| `molla.sys.socket` | `dial`, a blocking connected socket with send and receive timeouts |
| `molla.tls.openssl` | OpenSSL 3.x and 1.1.1 through dlopen |
| `molla.tls.darwin` | Secure Transport and CoreFoundation through dlopen |
| `molla.tls.policy` | Which hosts skip verification, by name, added by #14 |
| `molla.tls.client` | One `TlsClient` over both, the backend probe, and `molla tls <host>` |
| `molla.http.client` | GET, redirects, `Content-Length` and chunked bodies |
| `molla.registry.ghcr` | Token, manifest, index, blob, digest check |
| `molla.registry.pull` | `molla pull <ref>` |

Two new commands. `molla tls <host>` connects and prints the backend, the negotiated protocol and cipher, and the certificate chain the peer sent. `molla pull <ref>` walks the registry protocol and prints every step. Both exist for the same reason: when this fails on a machine we do not have, the output has to say which of the six or seven things went wrong.

## Why dlopen

A linked binary does not start on a machine whose OpenSSL is a different soname. Not "cannot do HTTPS", does not start, exits before `main` with a loader error. Ubuntu 24.04 has `libssl.so.3` and Ubuntu 20.04 has `libssl.so.1.1`, and both are still in the field, so a linked molla would need one build per soname.

A dlopened molla starts everywhere and loses exactly one thing when the library is missing. That is measurable rather than a claim:

```text
$ MOLLA_LIBSSL=libssl.so.99 molla version
molla 0.0.2
  mojo       1.0.0
  target     linux x86_64 (avx2)
  ...

$ MOLLA_LIBSSL=libssl.so.99 molla pull linuxcontainers/alpine
pull ghcr.io/linuxcontainers/alpine reference latest
molla pull: no usable libssl, tried libssl.so.99. Install OpenSSL 3 or 1.1 to enable HTTPS.
```

`MOLLA_LIBSSL` and `MOLLA_LIBCRYPTO` replace the candidate list rather than joining the front of it. An override that does not load is an error, because quietly using a different library than the one that was asked for is how you spend an afternoon debugging a machine that is not running what you told it to run. It is also how the 1.1 fallback gets tested on a machine that has 3.x, which is the only way to know the fallback works before meeting a host that needs it.

## Linux, OpenSSL

The symbol set is the intersection of 1.1.1 and 3.x. Nothing in it was added after 1.1.0, so one binding drives both and the only difference is which file dlopen found.

Three things were worth learning here.

`SSL_CTX_set_min_proto_version` is a macro over `SSL_CTX_ctrl`, so there is no symbol to dlsym. Same for the SNI setter. Both go through `SSL_ctrl` with the command number written out, 123 and 55.

`SSL_get1_peer_certificate` is the natural way to get the leaf on 3.x and it does not exist on 1.1, where it is spelled `SSL_get_peer_certificate`. Reading the chain instead avoids the split, because `SSL_get_peer_cert_chain` exists on both and on a client it includes the leaf.

Verification is the platform's. `SSL_CTX_set_default_verify_paths` points at whatever CA bundle the distribution installed, `SSL_set1_host` turns on hostname checking, and `SSL_VERIFY_PEER` makes a verification failure fail the handshake rather than being something the caller has to remember to check. molla ships no CA list and should not.

## macOS, and the TLS 1.2 problem

macOS has two TLS APIs and neither is a good answer.

Network.framework is the supported one. It is built on dispatch queues and Objective-C blocks, and a block is not a C function pointer, it is a struct with an invoke pointer and a captured environment that the runtime knows how to copy. Mojo can emit a C function pointer. It cannot emit a block, and faking one means hand building `_NSConcreteStackBlock` layouts against an ABI Apple documents as an implementation detail. That is not a foundation for the network edge.

Secure Transport is deprecated, present in every macOS since 10.2, and is a plain C API with function pointer callbacks. So this spike uses it, and it works, and here is the cost:

**Secure Transport does not do TLS 1.3.** `kTLSProtocol13` does not exist. macOS negotiates TLS 1.2 and Linux negotiates TLS 1.3 against the same server, which is visible in the results table below and is not a configuration mistake. ghcr.io accepts TLS 1.2 today and so does everything else on the public internet, but this is a floor that will rise, and when it does the macOS side needs Network.framework and Network.framework needs blocks.

That is a real item for M1, recorded here rather than discovered later. It does not change the M0 answer, because the M0 question is whether the FFI approach holds, and it does.

The callbacks were the part most likely to sink this. Secure Transport does not own the socket. It calls back for bytes, `SSLSetIOFuncs` takes two C function pointers, and the contract is that the length argument is bytes wanted on the way in and bytes moved on the way out, with a full transfer returning `noErr`, a partial one returning `errSSLWouldBlock`, and end of stream returning `errSSLClosedGraceful`. Mojo 1.0 can do this. A function declared `def name(...) abi("C") -> ret` converts to a function pointer alias by assignment, `var reader: IoFunc = st_read`. Construction, `IoFunc(st_read)`, does not work. That one line is what made the macOS side possible.

Verification is Apple's. `SSLSetPeerDomainName` turns on hostname checking against the system trust store and `SSLHandshake` fails with `errSSLXCertChainInvalid` or `errSSLHostNameMismatch` rather than succeeding quietly. Chain printing goes through `SSLCopyPeerTrust`, `SecTrustCopyCertificateChain` and `SecCertificateCopySubjectSummary`, which is why the macOS chain below shows a summary and the Linux chain shows a full subject line. Same certificates, different accessor.

## The registry side

Public repositories on ghcr.io still need a token. They just hand one out to anybody who asks, at `https://ghcr.io/token?service=ghcr.io&scope=repository:<repo>:pull`.

A blob request is answered with 307 to a signed URL on `pkg-containers.githubusercontent.com`, so following redirects across hosts is not optional. The bearer token is dropped when the host changes. That is deliberate and it matters: a redirect names a host, and sending a registry credential to whatever host a response asked for is handing the credential away.

There is no JSON parser. There is a scanner that finds `"key":"value"` at or after a byte offset, which is enough for digests, sizes and media types and would be wrong for anything nested or escaped. Selecting a platform out of an index takes one forward pass that remembers the last digest it saw and returns it when the architecture and os match, because an index entry writes its digest before its platform block. That handles both the compact and the pretty printed forms, which registries disagree on.

The digest check is the point of the whole exercise. Everything above the blob fetch trusts TLS and trusts the registry. The digest makes that trust unnecessary, because the bytes either hash to the name they were fetched under or they are thrown away.

## Results

`molla pull linuxcontainers/alpine:latest` on every machine. The path is a token, then the index, then the linux/amd64 manifest `sha256:d22cb65d578e`, then its config blob `sha256:1dd0f00d536d`, 1744 bytes, hashed and compared.

| Machine | Backend | Protocol | Cipher | Blob verified |
| --- | --- | --- | --- | --- |
| macbook, M4, macOS 15.8 | Secure Transport | TLSv1.2 | 0xc02f | yes |
| server1, Ubuntu 24.04 | OpenSSL 3.6.4 via libssl.so.3 | TLSv1.3 | TLS_AES_128_GCM_SHA256 | yes |
| server1, Ubuntu 24.04, forced | OpenSSL 1.1.1f via libssl.so.1.1 | TLSv1.3 | TLS_AES_128_GCM_SHA256 | yes |
| server2, Ubuntu 24.04 | OpenSSL 3.6.4 via libssl.so.3 | TLSv1.3 | TLS_AES_128_GCM_SHA256 | yes |
| server3, Ubuntu 24.04 | OpenSSL 3.6.4 via libssl.so.3 | TLSv1.3 | TLS_AES_128_GCM_SHA256 | yes |

The 1.1.1f row is the same machine with `MOLLA_LIBSSL=libssl.so.1.1`, which is how the fallback path gets exercised. 1.1.1f went out of support in 2023 and still negotiates TLS 1.3, because 1.1.1 was the release that added it.

The test suite was 204 checks at that point and passed on macOS, server1, server2 and server3. Fifty of those are new here: four SHA-256 vectors including one that crosses a block boundary, seven for DNS, ten for URL splitting, and the rest for the registry scanner against canned index and manifest bodies.

## Productionising, issue #14

Issue #14 asks for three things the spike did not have: an insecure flag that is per registry and never global, a binary that starts on a host with no TLS library and only loses HTTPS, and all of it behind one narrow interface. The interface was already there, so this is the other two.

### The insecure flag is a host, not a switch

`molla.tls.policy.TlsPolicy` holds the names of the hosts whose certificate is not checked. There is no boolean anywhere that means "do not verify", and that is the design rather than a preference.

A pull is not one connection. ghcr.io answers a blob request with a 307 to a signed URL on `pkg-containers.githubusercontent.com`, so a process wide flag would also turn verification off for a host named by the response rather than by the operator. Naming the host makes that impossible to write by accident. It is the same reasoning that drops the bearer token on a cross host redirect, one layer down.

The effect is visible in the output. Four connections say they are not verifying and the fifth, the one the redirect chose, says nothing:

```text
$ molla pull --insecure linuxcontainers/alpine:latest
pull ghcr.io/linuxcontainers/alpine reference latest
  insecure   not verifying the certificate for ghcr.io
  token      60 bytes
  insecure   not verifying the certificate for ghcr.io
  manifest   5413 bytes
  platform   linux/amd64 d22cb65d578e
  insecure   not verifying the certificate for ghcr.io
  manifest   670 bytes
  config     1dd0f00d536d 1744 bytes
  insecure   not verifying the certificate for ghcr.io
  blob       1744 bytes
  digest     sha256:1dd0f00d536d... verified
```

Matching is exact and case folded. No wildcards and no suffix rules, because a pattern language here is a way to turn off more verification than anybody intended, and the case that actually comes up is one registry on one name with a certificate the platform does not trust.

Every insecure connection says so, once, on stdout. Four lines for one pull is noisy and it is deliberate. An insecure connection nobody can see is the one that is still insecure a year later.

### Two different ways to not verify

OpenSSL is the easy side. `SSL_VERIFY_NONE` and no `SSL_set1_host`, both rather than one, because setting the host name and then not verifying leaves a check in the source that silently does nothing.

Secure Transport has no call that turns verification off. What it has is `kSSLSessionOptionBreakOnServerAuth`, which stops when the server's certificate arrives and hands control back, so evaluating the chain becomes the application's job and the application declines to do it. `SSLHandshake` then returns `errSSLPeerAuthCompleted`, and calling it again is how you say carry on. That return is an error when verifying, since nothing asked for the break.

One thing worth knowing when reading a failure on macOS: Secure Transport reports a name mismatch as `errSSLXCertChainInvalid`, the same code as an untrusted root. `wrong.host.badssl.com` and `self-signed.badssl.com` fail with the identical message there, and with different ones on Linux, where OpenSSL hands back the X509 code as well.

### A machine with no TLS library

`probe()` loads the library and reports what it found without opening a socket, and `molla version` prints it. That turns a promise into a line of output, one of these three:

```text
  tls        OpenSSL 3.6.4 25 Aug 2026 via libssl.so.3 up to TLS 1.3
  tls        Secure Transport up to TLS 1.2
  tls        unavailable, HTTPS is off: no usable libssl, tried libssl.so.99. Install OpenSSL 3 or 1.1 to enable HTTPS.
```

The `up to` is the TLS 1.2 cap from the macOS section above, printed on every machine rather than left in a document, because the machine where it matters is the one somebody is looking at on the day a server stops accepting TLS 1.2.

Security.framework now takes `MOLLA_SECURITY` and `MOLLA_COREFOUNDATION` overrides, matching `MOLLA_LIBSSL` and `MOLLA_LIBCRYPTO` on the other side. Nobody moves Security.framework, so these exist for the opposite reason to the Linux ones: pointing them at a path that does not load is the only way to see what molla does on a machine with no TLS library, and that behaviour is half of what the issue promises. The suite uses them, so every machine that runs the tests runs the missing library case.

### Where it was run

Every row is a real run of the commands rather than a claim, and the pull column is the same 1744 byte blob with the same digest as the M0 table.

| Machine | Backend | ghcr.io | expired.badssl.com | the same with --insecure | pull |
| --- | --- | --- | --- | --- | --- |
| macbook, M4, macOS 15.8 | Secure Transport, TLS 1.2 | verified | refused | connects | verified |
| server1, Ubuntu 24.04 | OpenSSL 3.6.4, TLS 1.3 | verified | refused | connects | verified |
| server2, Ubuntu 24.04 | OpenSSL 3.6.4, TLS 1.3 | verified | refused | connects | verified |
| gpc, WSL2 on Windows | OpenSSL 3.6.4, TLS 1.3 | verified | refused | connects | verified |

server1 ran it again with `MOLLA_LIBSSL=libssl.so.1.1`, which still negotiates TLS 1.3 and still verifies, so the 1.1 fallback has not rotted.

With the library taken away, all four print the unavailable line from `molla version` and exit 0, and `molla pull` fails with the same message and exits 1. That is the acceptance criterion of the issue, and it is also in the suite, so it is a test rather than a screenshot.

The suite is 827 checks on Linux and 826 on macOS. gpc fails one of them, which is issue #87, the reactor backpressure test under WSL2, and it fails the same way on main.

## What is missing

Recorded so nobody finds these by surprise.

IPv4 only, because the socket layer under this is IPv4 only. A host with only an AAAA record does not resolve.

Blocking sockets with a thirty second timeout on each side, not the event loop from the socket spike. A registry pull is a handful of sequential requests and the loop buys nothing until pulls run alongside serving, which is M4.

Bodies are read into memory whole. A model file will not fit that way. M3 needs a streaming reader and this one is explicitly not it.

One connection per request, `Connection: close`, no reuse. Five TCP connections and five handshakes for one blob pull, counting the redirect. That is slower than it needs to be and it makes the body framing two cases instead of six, which was the right trade for a spike and is the wrong one for a puller.

No client certificates, no session resumption, no proxy support, no revocation checking beyond whatever the platform does on its own.

No way to add a CA. Trust is the platform's, which is the right default and leaves one gap: a registry behind a private CA either has that CA in the machine's trust store, which is where it belongs, or it needs `--insecure`, which is a bigger hammer than the job wants. A per host CA file is the obvious fix and nothing needs it yet.

No server side TLS, deliberately, and it is not planned for v1. molla binds loopback by default under D9, and terminating TLS for an exposed deployment is a job for whatever reverse proxy is already in front of it. Saying that is more honest than shipping a weak TLS server.

Still TLS 1.2 on macOS. Unchanged by #14 and unchangeable without Network.framework, which needs Objective-C blocks. `molla version` now prints the cap on every machine, which is the mitigation available today.

Only ghcr.io. The protocol is the OCI distribution spec and Docker Hub speaks the same one, so widening this is mostly about where the token comes from.

## What this means for D1

D1 allows foreign code in three places through the C ABI and TLS is the second of them. The spike says that boundary holds. Two libraries, four opaque handles, one interface of connect, read, write, close and peer chain, and the calling code above `molla.tls.client` does not know which platform it is on.

The pattern that made it work is worth naming, because it is the same one D7 asks for in kernels: one struct, one method per operation, and a `comptime if` inside each method rather than two types selected at compile time. Two types was tried first. A `comptime` alias that picks a type with a ternary does not work in Mojo 1.0, it resolves only the move constructor, and even if it had worked the pattern is worse: a fix that lands in one branch and not the other is a bug found on the other platform months later, and keeping both branches inside one function makes that hard to do by accident.
