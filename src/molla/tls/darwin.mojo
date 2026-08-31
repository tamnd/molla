"""TLS on macOS through Secure Transport.

macOS has three ways to do TLS and none of them is comfortable. Network.framework
is the supported one and it is built on dispatch queues and Objective-C blocks,
neither of which Mojo can produce. CFNetwork's stream API has the same problem.
Secure Transport is a flat C API that takes two function pointers for its byte
transport, which is the one shape Mojo can hand a library, so that is what this
uses.

Secure Transport has been deprecated since macOS 10.15. That is a real cost and
worth saying plainly: the symbols are all still exported on macOS 15 and they
work, but Apple is not obliged to keep them. The mitigation is that this file is
the whole binding, so replacing it means writing one file rather than tracing
TLS calls through the codebase. Doing that properly means teaching molla to call
Objective-C, which is a much larger piece of work than issue #6.

Everything is loaded with dlopen rather than linked, so a molla binary starts on
a machine that for some reason has no Security.framework and only loses HTTPS.
The framework is part of the OS and always present in practice, but the Linux
side genuinely needs the dlopen and having the two platforms behave the same way
is worth more than saving a call here.

Certificate verification is the platform's. `SSLSetPeerDomainName` gives Secure
Transport the name to check, and `SSLHandshake` evaluates the chain against the
system trust store and fails the handshake if it does not like it. molla ships no
CA list, which is the point.
"""

from std.ffi import OwnedDLHandle, c_int
from std.memory import stack_allocation

from molla.sys.cstr import c_string, from_c_string
from molla.sys.errno import errno_name, get_errno
from molla.sys.socket import recv, send

comptime SECURITY_PATH = (
    "/System/Library/Frameworks/Security.framework/Security"
)
comptime COREFOUNDATION_PATH = (
    "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
)

comptime K_SSL_CLIENT_SIDE = 1
comptime K_SSL_STREAM_TYPE = 0
comptime K_TLS_PROTOCOL_12 = 8

comptime NO_ERR = 0
comptime ERR_SSL_WOULD_BLOCK = -9803
comptime ERR_SSL_CLOSED_GRACEFUL = -9805
comptime ERR_SSL_CLOSED_ABORT = -9806

comptime K_CF_STRING_ENCODING_UTF8 = 0x08000100

comptime EINTR = 4
comptime EAGAIN_MAC = 35
"""EAGAIN and EWOULDBLOCK are both 35 on macOS. The sockets here are blocking so
this should never come back, but a signal or a socket timeout can produce it and
turning that into a fatal error would be wrong."""


def status_name(status: Int) -> String:
    """A readable name for the OSStatus values TLS actually produces.

    Not a full table. Security.framework has hundreds of these and the ones a
    client handshake can hit are a short list, so the rest print as a number and
    can be looked up.
    """
    if status == 0:
        return "noErr"
    elif status == -9803:
        return "errSSLWouldBlock"
    elif status == -9804:
        return "errSSLPeerBadCert"
    elif status == -9805:
        return "errSSLClosedGraceful"
    elif status == -9806:
        return "errSSLClosedAbort"
    elif status == -9807:
        return "errSSLXCertChainInvalid"
    elif status == -9808:
        return "errSSLBadCert"
    elif status == -9812:
        return "errSSLCertExpired"
    elif status == -9813:
        return "errSSLUnknownRootCert"
    elif status == -9814:
        return "errSSLNoRootCert"
    elif status == -9824:
        return "errSSLPeerHandshakeFail"
    elif status == -9836:
        return "errSSLProtocol"
    elif status == -9843:
        return "errSSLHostNameMismatch"
    return "OSStatus " + String(status)


def _transfer(
    fd: Int,
    buf: Pointer[UInt8, MutAnyOrigin],
    length: Pointer[UInt64, MutAnyOrigin],
    sending: Bool,
) -> Int32:
    """The shared body of the two Secure Transport callbacks.

    The contract is unusual and getting it wrong produces a handshake that hangs
    rather than one that fails. `*length` comes in as the number of bytes Secure
    Transport wants moved and must go out as the number actually moved. Moving
    all of them is `noErr`. Moving some of them is `errSSLWouldBlock` with the
    count set, and Secure Transport will ask again for the rest. Reading zero
    bytes at end of stream is `errSSLClosedGraceful`.
    """
    var want = Int(length.unsafe_load(0))
    var done = 0
    var closed = False
    while done < want:
        var n: Int
        if sending:
            n = send(fd, buf.unsafe_offset(done), want - done)
        else:
            n = recv(fd, buf.unsafe_offset(done), want - done)
        if n > 0:
            done += n
        elif n == 0:
            closed = True
            break
        else:
            var code = get_errno()
            if code == EINTR:
                continue
            if code == EAGAIN_MAC:
                break
            length.unsafe_store(0, UInt64(done))
            return Int32(ERR_SSL_CLOSED_ABORT)
    length.unsafe_store(0, UInt64(done))
    if done == want:
        return Int32(NO_ERR)
    if closed:
        return Int32(ERR_SSL_CLOSED_GRACEFUL)
    return Int32(ERR_SSL_WOULD_BLOCK)


def st_read(
    connection: Int,
    data: Pointer[UInt8, MutAnyOrigin],
    length: Pointer[UInt64, MutAnyOrigin],
) abi("C") -> Int32:
    """Secure Transport's read callback.

    `connection` is whatever was handed to `SSLSetConnection`, and molla hands
    it the file descriptor as an integer rather than a pointer to one. Secure
    Transport treats the value as opaque and never dereferences it, so there is
    nothing to keep alive and nothing to free.
    """
    return _transfer(connection, data, length, sending=False)


def st_write(
    connection: Int,
    data: Pointer[UInt8, MutAnyOrigin],
    length: Pointer[UInt64, MutAnyOrigin],
) abi("C") -> Int32:
    """Secure Transport's write callback."""
    return _transfer(connection, data, length, sending=True)


comptime IoFunc = def(
    Int, Pointer[UInt8, MutAnyOrigin], Pointer[UInt64, MutAnyOrigin]
) thin abi("C") -> Int32


def open_security() raises -> OwnedDLHandle:
    """dlopen Security.framework."""
    return OwnedDLHandle(SECURITY_PATH)


def open_corefoundation() raises -> OwnedDLHandle:
    """dlopen CoreFoundation, which is where the certificate chain accessors and
    CFRelease live."""
    return OwnedDLHandle(COREFOUNDATION_PATH)


def create_context(security: OwnedDLHandle) raises -> Int:
    """A new client side stream context. Held as an integer address."""
    var ctx = security.get_function[Int]("SSLCreateContext")(
        Int(0), c_int(K_SSL_CLIENT_SIDE), c_int(K_SSL_STREAM_TYPE)
    )
    if ctx == 0:
        raise Error("SSLCreateContext returned null")
    return ctx


def configure(
    security: OwnedDLHandle, ctx: Int, fd: Int, host: StringSpan
) raises:
    """Wire the context to the socket and tell it who it is talking to."""
    var reader: IoFunc = st_read
    var writer: IoFunc = st_write
    var rc = Int(
        security.get_function[Int32]("SSLSetIOFuncs")(ctx, reader, writer)
    )
    if rc != NO_ERR:
        raise Error("SSLSetIOFuncs: " + status_name(rc))

    rc = Int(security.get_function[Int32]("SSLSetConnection")(ctx, fd))
    if rc != NO_ERR:
        raise Error("SSLSetConnection: " + status_name(rc))

    # This is the line that makes verification mean anything. Without it the
    # chain is still checked but the name on the certificate is not, and a valid
    # certificate for any host would be accepted for this one.
    var name = c_string(host)
    rc = Int(
        security.get_function[Int32]("SSLSetPeerDomainName")(
            ctx, name.unsafe_ptr(), UInt64(host.byte_length())
        )
    )
    if rc != NO_ERR:
        raise Error("SSLSetPeerDomainName: " + status_name(rc))

    rc = Int(
        security.get_function[Int32]("SSLSetProtocolVersionMin")(
            ctx, c_int(K_TLS_PROTOCOL_12)
        )
    )
    if rc != NO_ERR:
        raise Error("SSLSetProtocolVersionMin: " + status_name(rc))


def handshake(security: OwnedDLHandle, ctx: Int) raises:
    """Run the handshake, including trust evaluation.

    `errSSLWouldBlock` comes back whenever the callbacks moved a short count,
    which on a blocking socket means a partial record arrived, so the loop just
    asks again.
    """
    while True:
        var rc = Int(security.get_function[Int32]("SSLHandshake")(ctx))
        if rc == NO_ERR:
            return
        if rc == ERR_SSL_WOULD_BLOCK:
            continue
        raise Error("TLS handshake failed: " + status_name(rc))


def read(
    security: OwnedDLHandle,
    ctx: Int,
    buf: Pointer[UInt8, MutAnyOrigin],
    count: Int,
) raises -> Int:
    """Read up to `count` plaintext bytes. Returns 0 at end of stream."""
    var moved = stack_allocation[1, UInt64]()
    moved.unsafe_store(0, UInt64(0))
    var rc = Int(
        security.get_function[Int32]("SSLRead")(ctx, buf, UInt64(count), moved)
    )
    var n = Int(moved.unsafe_load(0))
    if rc == NO_ERR or rc == ERR_SSL_WOULD_BLOCK:
        return n
    if rc == ERR_SSL_CLOSED_GRACEFUL or rc == ERR_SSL_CLOSED_ABORT:
        # A registry that closes without a close_notify is common enough that
        # treating an abort as end of stream is the only workable choice. The
        # content length check upstream is what catches a truncated body.
        return n
    raise Error("SSLRead: " + status_name(rc))


def write(
    security: OwnedDLHandle,
    ctx: Int,
    buf: Pointer[UInt8, MutAnyOrigin],
    count: Int,
) raises -> Int:
    """Write `count` plaintext bytes."""
    var moved = stack_allocation[1, UInt64]()
    moved.unsafe_store(0, UInt64(0))
    var rc = Int(
        security.get_function[Int32]("SSLWrite")(ctx, buf, UInt64(count), moved)
    )
    if rc != NO_ERR:
        raise Error("SSLWrite: " + status_name(rc))
    return Int(moved.unsafe_load(0))


def shutdown(security: OwnedDLHandle, ctx: Int):
    """Send close_notify and drop the context.

    Failures are swallowed on purpose. There is nothing useful to do about a
    goodbye that did not arrive, and raising out of teardown would hide whatever
    error caused the teardown in the first place.
    """
    try:
        _ = security.get_function[Int32]("SSLClose")(ctx)
        _ = security.get_function[NoneType]("CFRelease")(ctx)
    except:
        pass


def protocol_name(security: OwnedDLHandle, ctx: Int) raises -> String:
    """Which TLS version was negotiated."""
    var out = stack_allocation[1, Int32]()
    out.unsafe_store(0, Int32(0))
    var rc = Int(
        security.get_function[Int32]("SSLGetNegotiatedProtocolVersion")(
            ctx, out
        )
    )
    if rc != NO_ERR:
        raise Error("SSLGetNegotiatedProtocolVersion: " + status_name(rc))
    var v = Int(out.unsafe_load(0))
    if v == 8:
        return "TLSv1.2"
    elif v == 10:
        return "TLSv1.3"
    return "SSLProtocol " + String(v)


def cipher_name(security: OwnedDLHandle, ctx: Int) raises -> String:
    """The negotiated cipher suite, as its IANA number.

    Secure Transport has no call that turns a suite number into a name, so this
    prints the number. The names are in RFC 8446 and the IANA registry, and
    inventing a lookup table that goes stale is worse than a number you can
    search for.
    """
    var out = stack_allocation[1, UInt32]()
    out.unsafe_store(0, UInt32(0))
    var rc = Int(
        security.get_function[Int32]("SSLGetNegotiatedCipher")(ctx, out)
    )
    if rc != NO_ERR:
        raise Error("SSLGetNegotiatedCipher: " + status_name(rc))
    return "cipher suite " + hex(Int(out.unsafe_load(0)))


def peer_chain(
    security: OwnedDLHandle, corefoundation: OwnedDLHandle, ctx: Int
) raises -> List[String]:
    """Subject summaries for the certificates the peer sent, leaf first.

    This is after the handshake, so the chain has already been accepted. It is
    here because issue #6 asks for it and because being able to print what you
    are actually trusting is worth having before anyone debugs a corporate proxy.
    """
    var out = List[String]()

    var trust_out = stack_allocation[1, UInt64]()
    trust_out.unsafe_store(0, UInt64(0))
    var rc = Int(
        security.get_function[Int32]("SSLCopyPeerTrust")(ctx, trust_out)
    )
    if rc != NO_ERR:
        raise Error("SSLCopyPeerTrust: " + status_name(rc))
    var trust = Int(trust_out.unsafe_load(0))
    if trust == 0:
        return out^

    # SecTrustCopyCertificateChain rather than SecTrustGetCertificateAtIndex.
    # The indexed one is deprecated and, on recent macOS, returns null unless
    # the trust object has been evaluated in the way it expects.
    var chain = security.get_function[Int]("SecTrustCopyCertificateChain")(
        trust
    )
    if chain == 0:
        _ = corefoundation.get_function[NoneType]("CFRelease")(trust)
        return out^

    var count = corefoundation.get_function[Int]("CFArrayGetCount")(chain)
    var buf = stack_allocation[512, UInt8]()
    for i in range(count):
        var cert = corefoundation.get_function[Int]("CFArrayGetValueAtIndex")(
            chain, i
        )
        if cert == 0:
            continue
        var summary = security.get_function[Int](
            "SecCertificateCopySubjectSummary"
        )(cert)
        if summary == 0:
            out.append("(no subject summary)")
            continue
        var ok = Int(
            corefoundation.get_function[UInt8]("CFStringGetCString")(
                summary, buf, Int(512), UInt32(K_CF_STRING_ENCODING_UTF8)
            )
        )
        if ok != 0:
            out.append(from_c_string(buf, 512))
        else:
            out.append("(subject summary too long)")
        _ = corefoundation.get_function[NoneType]("CFRelease")(summary)

    _ = corefoundation.get_function[NoneType]("CFRelease")(chain)
    _ = corefoundation.get_function[NoneType]("CFRelease")(trust)
    return out^
