"""One TLS interface over two very different libraries.

The two backends have nothing in common at the C level. OpenSSL owns the socket
and reads it itself, Secure Transport hands the bytes back through callbacks.
OpenSSL has a context and a connection, Secure Transport has one object that is
both. OpenSSL knows the names of cipher suites, Secure Transport does not.

What they do have in common is their shape once loaded: two dynamic libraries and
two opaque handles. So `TlsClient` holds exactly that, and each method is a
compile time branch to one binding or the other. This is the pattern D7 asks for
in kernels, applied here for the same reason: a fix that lands in one branch and
not the other is a bug you find on the other platform months later, and keeping
both branches in one function makes that hard to do by accident.

Everything is blocking. See `dial` in `molla.sys.socket` for why.
"""

from std.ffi import OwnedDLHandle
from std.sys.info import CompilationTarget

from molla.sys.dns import format_ipv4, resolve_ipv4
from molla.sys.fd import close
from molla.sys.socket import dial
from molla.tls import darwin, openssl

comptime DEFAULT_TIMEOUT_SECONDS = 30
comptime DEFAULT_PORT: UInt16 = 443


struct TlsClient(Movable):
    """One verified TLS connection to one host."""

    var primary: OwnedDLHandle
    """libssl on Linux, Security.framework on macOS."""

    var secondary: OwnedDLHandle
    """libcrypto on Linux, CoreFoundation on macOS. Both platforms keep the
    error and certificate helpers in a second library."""

    var backend: String
    """What was actually loaded, for the version line and for error messages
    that would otherwise be impossible to tell apart."""

    var fd: Int
    var ctx: Int
    """SSL_CTX on Linux. Unused on macOS, where one object does both jobs."""

    var conn: Int
    """SSL on Linux, SSLContextRef on macOS."""

    var host: String
    var open: Bool

    def __init__(out self, host: String, port: UInt16 = DEFAULT_PORT) raises:
        """Resolve, connect, handshake, verify. Any failure raises and leaves
        nothing open."""
        self.host = host
        self.fd = 0
        self.ctx = 0
        self.conn = 0
        self.open = False

        comptime if CompilationTarget.is_macos():
            self.primary = darwin.open_security()
            self.secondary = darwin.open_corefoundation()
            self.backend = "Secure Transport"
        else:
            var soname = String("")
            self.primary = openssl.open_ssl(soname)
            self.secondary = openssl.open_crypto()
            self.backend = (
                openssl.library_version(self.secondary) + " via " + soname
            )

        var addr = resolve_ipv4(host)
        self.fd = dial(addr, port, DEFAULT_TIMEOUT_SECONDS)

        try:
            comptime if CompilationTarget.is_macos():
                self.conn = darwin.create_context(self.primary)
                darwin.configure(self.primary, self.conn, self.fd, host)
                darwin.handshake(self.primary, self.conn)
            else:
                self.ctx = openssl.create_context(self.primary, self.secondary)
                self.conn = openssl.create_connection(
                    self.primary, self.secondary, self.ctx, self.fd, host
                )
                openssl.handshake(self.primary, self.secondary, self.conn)
        except e:
            self._teardown()
            raise Error(
                "https://" + host + " (" + format_ipv4(addr) + "): " + String(e)
            )

        self.open = True

    def _teardown(mut self):
        """Free whatever got as far as existing. Safe to call twice."""
        comptime if CompilationTarget.is_macos():
            if self.conn != 0:
                darwin.shutdown(self.primary, self.conn)
        else:
            openssl.shutdown(self.primary, self.conn, self.ctx)
        self.conn = 0
        self.ctx = 0
        if self.fd != 0:
            _ = close(self.fd)
            self.fd = 0
        self.open = False

    def close(mut self):
        """Shut the connection down. Explicit rather than in a destructor,
        matching `Poller` and the servers."""
        self._teardown()

    def write_all(mut self, data: Span[UInt8, _]) raises:
        """Write every byte or raise. Short writes are normal and not an error.
        """
        var sent = 0
        var n = len(data)
        var base = Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(data.unsafe_ptr())
        )
        while sent < n:
            var wrote: Int
            comptime if CompilationTarget.is_macos():
                wrote = darwin.write(
                    self.primary, self.conn, base.unsafe_offset(sent), n - sent
                )
            else:
                wrote = openssl.write(
                    self.primary,
                    self.secondary,
                    self.conn,
                    base.unsafe_offset(sent),
                    n - sent,
                )
            if wrote <= 0:
                raise Error("TLS write made no progress")
            sent += wrote

    def read(mut self, buf: Pointer[UInt8, _], count: Int) raises -> Int:
        """Read up to `count` plaintext bytes. Zero means end of stream.

        The origin is widened here rather than at every call site. Both
        bindings hand the buffer straight to C, which has no idea what an
        origin is, so there is nothing for the checker to protect and forcing
        callers to allocate with one particular origin would only mean the cast
        happens in more places.
        """
        var wide = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(buf))
        comptime if CompilationTarget.is_macos():
            return darwin.read(self.primary, self.conn, wide, count)
        else:
            return openssl.read(
                self.primary, self.secondary, self.conn, wide, count
            )

    def protocol(self) raises -> String:
        """The negotiated TLS version."""
        comptime if CompilationTarget.is_macos():
            return darwin.protocol_name(self.primary, self.conn)
        else:
            return openssl.protocol_name(self.primary, self.conn)

    def cipher(self) raises -> String:
        """The negotiated cipher suite. A name on Linux, a number on macOS,
        because Secure Transport has no call that names one."""
        comptime if CompilationTarget.is_macos():
            return darwin.cipher_name(self.primary, self.conn)
        else:
            return openssl.cipher_name(self.primary, self.conn)

    def peer_chain(self) raises -> List[String]:
        """The certificates the peer sent, leaf first, already verified."""
        comptime if CompilationTarget.is_macos():
            return darwin.peer_chain(self.primary, self.secondary, self.conn)
        else:
            return openssl.peer_chain(self.primary, self.secondary, self.conn)


def run_tls(host: String, port: UInt16 = DEFAULT_PORT) raises:
    """The `molla tls` command: connect, verify, print what was negotiated.

    This is the first thing to run on a machine where a pull fails. It
    separates the three ways TLS goes wrong on a new host, which are no library
    to load, a library that loads but cannot verify, and a peer that verifies
    but presents a chain nobody expected.
    """
    var conn = TlsClient(host, port)
    print("tls", host, "port", port)
    print("  backend  ", conn.backend)
    print("  protocol ", conn.protocol())
    print("  cipher   ", conn.cipher())
    var chain = conn.peer_chain()
    for i in range(len(chain)):
        print("  cert", i, " ", chain[i])
    conn.close()
