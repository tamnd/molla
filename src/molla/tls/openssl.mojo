"""TLS on Linux through OpenSSL, loaded with dlopen.

dlopen rather than link, for one reason: a linked molla will not start at all on
a machine whose OpenSSL is a different soname, and a dlopened molla starts and
tells you HTTPS is unavailable. That matters because OpenSSL 1.1 and 3.x are
both still in the field and distributions do not agree on which one is present.

The symbol set here is deliberately the intersection of 1.1.1 and 3.x. Nothing
in it was added after 1.1.0, so the same code drives both and the only difference
is which file dlopen found. `SSL_get1_peer_certificate` would be the natural way
to get the leaf certificate on 3.x and it does not exist on 1.1, so this reads
the chain instead, which does exist on both and includes the leaf on the client
side anyway.

Both libssl and libcrypto get loaded. libssl pulls libcrypto in as a dependency
so the second dlopen is cheap, but the error and stack helpers live in libcrypto
and are not reachable through the libssl handle.

Certificate verification is the platform's. `SSL_CTX_set_default_verify_paths`
points OpenSSL at whatever CA bundle the distribution installed, `SSL_set1_host`
turns on hostname checking, and `SSL_VERIFY_PEER` makes a failure fail the
handshake instead of being something the caller has to remember to check. molla
ships no CA list.
"""

from std.ffi import OwnedDLHandle, c_int, c_long, c_size_t
from std.memory import stack_allocation
from std.os.env import getenv

from molla.sys.cstr import c_string, from_c_string

comptime SSL_VERIFY_PEER = 1

comptime SSL_CTRL_SET_TLSEXT_HOSTNAME = 55
comptime TLSEXT_NAMETYPE_HOST_NAME = 0
comptime SSL_CTRL_SET_MIN_PROTO_VERSION = 123
comptime TLS1_2_VERSION = 0x0303

comptime SSL_ERROR_NONE = 0
comptime SSL_ERROR_SSL = 1
comptime SSL_ERROR_WANT_READ = 2
comptime SSL_ERROR_WANT_WRITE = 3
comptime SSL_ERROR_ZERO_RETURN = 6
comptime SSL_ERROR_SYSCALL = 5


def _open_first(
    candidates: List[String], what: String, mut chosen: String
) raises -> OwnedDLHandle:
    """dlopen the first candidate that loads, and report which one it was.

    The name comes back through a `mut` argument rather than in a pair with the
    handle. Returning both would mean moving the handle out of a struct that has
    a destructor, which Mojo will not allow, and an out parameter is less
    machinery than working around that.
    """
    var tried = String("")
    for i in range(len(candidates)):
        try:
            var lib = OwnedDLHandle(candidates[i])
            chosen = candidates[i]
            return lib^
        except:
            if i > 0:
                tried += ", "
            tried += candidates[i]
    raise Error(
        "no usable "
        + what
        + ", tried "
        + tried
        + ". Install OpenSSL 3 or 1.1 to enable HTTPS."
    )


comptime LIBSSL_ENV = "MOLLA_LIBSSL"
comptime LIBCRYPTO_ENV = "MOLLA_LIBCRYPTO"


def _forced(name: String) -> String:
    """An environment override, or an empty string.

    Two uses. Somebody with OpenSSL in a nonstandard prefix can point molla at
    it without touching the loader path for everything else on the machine, and
    the 1.1 path can be tested on a machine that also has 3.x, which is the only
    way to know the fallback works before meeting a host that needs it.

    An override replaces the candidate list rather than joining the front of it.
    Naming a library that does not load is a mistake worth reporting, and
    quietly using a different one instead is how you end up debugging a machine
    that is not running what you told it to run.
    """
    return getenv(name)


def open_ssl(mut soname: String) raises -> OwnedDLHandle:
    """Load libssl.

    Newest soname first, because a machine with both 3.x and 1.1 should use
    3.x, and the bare `libssl.so` last because that is usually a development
    symlink rather than the runtime library.
    """
    var candidates = List[String]()
    var forced = _forced(LIBSSL_ENV)
    if forced.byte_length() > 0:
        candidates.append(forced)
    else:
        candidates.append("libssl.so.3")
        candidates.append("libssl.so.1.1")
        candidates.append("libssl.so")
    return _open_first(candidates, "libssl", soname)


def open_crypto() raises -> OwnedDLHandle:
    """Load libcrypto, where the error strings and the stack accessors live."""
    var candidates = List[String]()
    var forced = _forced(LIBCRYPTO_ENV)
    if forced.byte_length() > 0:
        candidates.append(forced)
    else:
        candidates.append("libcrypto.so.3")
        candidates.append("libcrypto.so.1.1")
        candidates.append("libcrypto.so")
    var chosen = String("")
    return _open_first(candidates, "libcrypto", chosen)


def library_version(crypto: OwnedDLHandle) raises -> String:
    """What OpenSSL calls itself, or a placeholder if the call is missing."""
    var p = crypto.get_function[Pointer[UInt8, ImmutAnyOrigin]](
        "OpenSSL_version"
    )(c_int(0))
    return from_c_string(p, 128)


def last_error(crypto: OwnedDLHandle) raises -> String:
    """Drain OpenSSL's error queue into one string.

    The queue matters. OpenSSL pushes a chain of errors and the useful one is
    usually not the first, so reporting only `ERR_get_error` once regularly
    produces "internal error" when the real message two entries down says the
    certificate has expired.
    """
    var out = String("")
    var buf = stack_allocation[256, UInt8]()
    var seen = 0
    while seen < 8:
        var code = Int(crypto.get_function[UInt64]("ERR_get_error")())
        if code == 0:
            break
        _ = crypto.get_function[NoneType]("ERR_error_string_n")(
            UInt64(code), buf, c_size_t(256)
        )
        if seen > 0:
            out += "; "
        out += from_c_string(buf, 256)
        seen += 1
    if seen == 0:
        return "no OpenSSL error queued"
    return out^


def create_context(ssl_lib: OwnedDLHandle, crypto: OwnedDLHandle) raises -> Int:
    """A client SSL_CTX with the platform trust store and a TLS 1.2 floor."""
    var method = ssl_lib.get_function[Int]("TLS_client_method")()
    if method == 0:
        raise Error("TLS_client_method returned null")

    var ctx = ssl_lib.get_function[Int]("SSL_CTX_new")(method)
    if ctx == 0:
        raise Error("SSL_CTX_new: " + last_error(crypto))

    if (
        Int(
            ssl_lib.get_function[c_int]("SSL_CTX_set_default_verify_paths")(ctx)
        )
        != 1
    ):
        raise Error(
            "no system CA store OpenSSL will accept: " + last_error(crypto)
        )

    # SSL_VERIFY_PEER with a null callback. A verification failure then aborts
    # the handshake rather than being reported through SSL_get_verify_result,
    # which is a call it is very easy to forget.
    _ = ssl_lib.get_function[NoneType]("SSL_CTX_set_verify")(
        ctx, c_int(SSL_VERIFY_PEER), Int(0)
    )

    # SSL_CTX_set_min_proto_version is a macro over SSL_CTX_ctrl, so there is no
    # symbol of that name to look up.
    _ = ssl_lib.get_function[c_long]("SSL_CTX_ctrl")(
        ctx,
        c_int(SSL_CTRL_SET_MIN_PROTO_VERSION),
        c_long(TLS1_2_VERSION),
        Int(0),
    )
    return ctx


def create_connection(
    ssl_lib: OwnedDLHandle,
    crypto: OwnedDLHandle,
    ctx: Int,
    fd: Int,
    host: StringSpan,
) raises -> Int:
    """A new SSL bound to `fd`, with SNI and hostname verification set up."""
    var ssl = ssl_lib.get_function[Int]("SSL_new")(ctx)
    if ssl == 0:
        raise Error("SSL_new: " + last_error(crypto))

    if Int(ssl_lib.get_function[c_int]("SSL_set_fd")(ssl, c_int(fd))) != 1:
        raise Error("SSL_set_fd: " + last_error(crypto))

    var name = c_string(host)

    # SNI. Without it a registry behind a shared front end serves whichever
    # certificate is the default, and the name check below then fails with a
    # mismatch that looks like the server's fault.
    _ = ssl_lib.get_function[c_long]("SSL_ctrl")(
        ssl,
        c_int(SSL_CTRL_SET_TLSEXT_HOSTNAME),
        c_long(TLSEXT_NAMETYPE_HOST_NAME),
        name.unsafe_ptr(),
    )

    # Hostname verification. Separate from SNI, and the one that actually
    # protects anything.
    if (
        Int(
            ssl_lib.get_function[c_int]("SSL_set1_host")(ssl, name.unsafe_ptr())
        )
        != 1
    ):
        raise Error("SSL_set1_host: " + last_error(crypto))

    return ssl


def handshake(ssl_lib: OwnedDLHandle, crypto: OwnedDLHandle, ssl: Int) raises:
    """Run the client handshake, including verification."""
    var rc = Int(ssl_lib.get_function[c_int]("SSL_connect")(ssl))
    if rc == 1:
        return
    var err = Int(ssl_lib.get_function[c_int]("SSL_get_error")(ssl, c_int(rc)))
    var verify = Int(ssl_lib.get_function[c_long]("SSL_get_verify_result")(ssl))
    var detail = last_error(crypto)
    if verify != 0:
        detail = (
            "certificate rejected, X509 code " + String(verify) + ": " + detail
        )
    raise Error(
        "TLS handshake failed, SSL_get_error " + String(err) + ": " + detail
    )


def read(
    ssl_lib: OwnedDLHandle,
    crypto: OwnedDLHandle,
    ssl: Int,
    buf: Pointer[UInt8, MutAnyOrigin],
    count: Int,
) raises -> Int:
    """Read up to `count` plaintext bytes. Returns 0 at end of stream."""
    var n = Int(ssl_lib.get_function[c_int]("SSL_read")(ssl, buf, c_int(count)))
    if n > 0:
        return n
    var err = Int(ssl_lib.get_function[c_int]("SSL_get_error")(ssl, c_int(n)))
    if err == SSL_ERROR_ZERO_RETURN:
        return 0
    if err == SSL_ERROR_SYSCALL:
        # A peer that closes the socket without close_notify lands here with an
        # empty error queue. Registries and their CDNs do this, so treating it
        # as end of stream is the only thing that works. The content length
        # check upstream is what catches a genuinely truncated body.
        return 0
    raise Error(
        "SSL_read failed, SSL_get_error "
        + String(err)
        + ": "
        + last_error(crypto)
    )


def write(
    ssl_lib: OwnedDLHandle,
    crypto: OwnedDLHandle,
    ssl: Int,
    buf: Pointer[UInt8, MutAnyOrigin],
    count: Int,
) raises -> Int:
    """Write plaintext bytes. Returns how many went."""
    var n = Int(
        ssl_lib.get_function[c_int]("SSL_write")(ssl, buf, c_int(count))
    )
    if n > 0:
        return n
    var err = Int(ssl_lib.get_function[c_int]("SSL_get_error")(ssl, c_int(n)))
    raise Error(
        "SSL_write failed, SSL_get_error "
        + String(err)
        + ": "
        + last_error(crypto)
    )


def shutdown(ssl_lib: OwnedDLHandle, ssl: Int, ctx: Int):
    """close_notify, then free both objects.

    Failures are swallowed on purpose. There is nothing useful to do about a
    goodbye that did not arrive, and raising out of teardown would hide whatever
    error caused the teardown in the first place.
    """
    try:
        if ssl != 0:
            _ = ssl_lib.get_function[c_int]("SSL_shutdown")(ssl)
            _ = ssl_lib.get_function[NoneType]("SSL_free")(ssl)
        if ctx != 0:
            _ = ssl_lib.get_function[NoneType]("SSL_CTX_free")(ctx)
    except:
        pass


def protocol_name(ssl_lib: OwnedDLHandle, ssl: Int) raises -> String:
    """Which TLS version was negotiated."""
    var p = ssl_lib.get_function[Pointer[UInt8, ImmutAnyOrigin]](
        "SSL_get_version"
    )(ssl)
    return from_c_string(p, 32)


def cipher_name(ssl_lib: OwnedDLHandle, ssl: Int) raises -> String:
    """The negotiated cipher suite, by name. OpenSSL has the table, unlike
    Secure Transport, so this side gets to print something readable."""
    var cipher = ssl_lib.get_function[Int]("SSL_get_current_cipher")(ssl)
    if cipher == 0:
        return "unknown"
    var p = ssl_lib.get_function[Pointer[UInt8, ImmutAnyOrigin]](
        "SSL_CIPHER_get_name"
    )(cipher)
    return from_c_string(p, 64)


def peer_chain(
    ssl_lib: OwnedDLHandle, crypto: OwnedDLHandle, ssl: Int
) raises -> List[String]:
    """Subject lines for the certificates the peer sent, leaf first.

    `SSL_get_peer_cert_chain` on a client includes the peer's own certificate,
    which is not true on a server, and is the reason this does not also call
    `SSL_get_peer_certificate` and end up printing the leaf twice.
    """
    var out = List[String]()
    var chain = ssl_lib.get_function[Int]("SSL_get_peer_cert_chain")(ssl)
    if chain == 0:
        return out^

    var count = Int(crypto.get_function[c_int]("OPENSSL_sk_num")(chain))
    var buf = stack_allocation[512, UInt8]()
    for i in range(count):
        var cert = crypto.get_function[Int]("OPENSSL_sk_value")(chain, c_int(i))
        if cert == 0:
            continue
        var name = crypto.get_function[Int]("X509_get_subject_name")(cert)
        if name == 0:
            out.append("(no subject)")
            continue
        _ = crypto.get_function[Pointer[UInt8, ImmutAnyOrigin]](
            "X509_NAME_oneline"
        )(name, buf, c_int(512))
        out.append(from_c_string(buf, 512))
    return out^
