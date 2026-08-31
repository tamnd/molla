# Client TLS through dlopen

The M0 TLS spike. Issue #6 asks whether molla can do HTTPS by loading the platform's TLS library at runtime rather than linking it, on macOS and on Linux, and it says the spike is done when a real blob comes down from ghcr.io over HTTPS on both. It does. The same 1744 byte blob, with the same SHA-256, on four machines and three different TLS libraries.

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
| `molla.tls.client` | One `TlsClient` over both, and `molla tls <host>` |
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

The test suite is 204 checks and passes on macOS, server1, server2 and server3. Fifty of those are new here: four SHA-256 vectors including one that crosses a block boundary, seven for DNS, ten for URL splitting, and the rest for the registry scanner against canned index and manifest bodies.

## What is missing

Recorded so nobody finds these by surprise.

IPv4 only, because the socket layer under this is IPv4 only. A host with only an AAAA record does not resolve.

Blocking sockets with a thirty second timeout on each side, not the event loop from the socket spike. A registry pull is a handful of sequential requests and the loop buys nothing until pulls run alongside serving, which is M4.

Bodies are read into memory whole. A model file will not fit that way. M3 needs a streaming reader and this one is explicitly not it.

One connection per request, `Connection: close`, no reuse. Five TCP connections and five handshakes for one blob pull, counting the redirect. That is slower than it needs to be and it makes the body framing two cases instead of six, which was the right trade for a spike and is the wrong one for a puller.

No client certificates, no session resumption, no proxy support, no revocation checking beyond whatever the platform does on its own.

Only ghcr.io. The protocol is the OCI distribution spec and Docker Hub speaks the same one, so widening this is mostly about where the token comes from.

## What this means for D1

D1 allows foreign code in three places through the C ABI and TLS is the second of them. The spike says that boundary holds. Two libraries, four opaque handles, one interface of connect, read, write, close and peer chain, and the calling code above `molla.tls.client` does not know which platform it is on.

The pattern that made it work is worth naming, because it is the same one D7 asks for in kernels: one struct, one method per operation, and a `comptime if` inside each method rather than two types selected at compile time. Two types was tried first. A `comptime` alias that picks a type with a ternary does not work in Mojo 1.0, it resolves only the move constructor, and even if it had worked the pattern is worse: a fix that lands in one branch and not the other is a bug found on the other platform months later, and keeping both branches inside one function makes that hard to do by accident.
