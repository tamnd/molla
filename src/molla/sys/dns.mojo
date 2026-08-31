"""Name resolution through getaddrinfo.

The echo and HTTP spikes only ever talked to 127.0.0.1, so molla has never
needed a resolver. Pulling from a registry does, and writing one would mean
reimplementing /etc/hosts, nsswitch, mDNS and whatever the platform does about
VPNs. getaddrinfo already knows all of that, so this is a wrapper and nothing
more.

The annoying part is `struct addrinfo`. It is 48 bytes on both platforms and
the first five fields agree, and then macOS puts `ai_canonname` before `ai_addr`
while Linux puts it after. Getting that backwards gives you a pointer to a
string where a sockaddr should be, which does not crash, it just resolves every
hostname to a nonsense address. Fields are read by offset with the difference
resolved at compile time, the same way `write_sockaddr_in` handles the sockaddr
layout split.

IPv4 only. molla will need IPv6 before it can claim to work on a modern
network, but the socket layer underneath this is IPv4 only too, so widening
here alone would buy nothing.
"""

from std.ffi import c_int, external_call
from std.memory import stack_allocation
from std.sys.info import CompilationTarget

from molla.sys.cstr import c_string, from_c_string
from molla.sys.socket import AF_INET, SOCK_STREAM

comptime ADDRINFO_SIZE = 48
"""sizeof(struct addrinfo) on macOS arm64 and on Linux x86_64 and aarch64."""

comptime AI_FAMILY_OFF = 4
comptime AI_SOCKTYPE_OFF = 8


def _ai_addr_off() -> Int:
    comptime if CompilationTarget.is_macos():
        # macOS: flags, family, socktype, protocol, addrlen, canonname, addr.
        return 32
    else:
        # Linux: flags, family, socktype, protocol, addrlen, addr, canonname.
        return 24


comptime AI_ADDR_OFF = _ai_addr_off()
comptime AI_NEXT_OFF = 40

comptime CharPtr = Pointer[UInt8, ImmutAnyOrigin]
"""What libc calls `char *`."""

comptime NullableCharPtr = OptionalPointer[UInt8, ImmutAnyOrigin]
"""A Mojo `Pointer` is not nullable, so anything that can come back NULL from C
has to be spelled as an optional. Passing `NullableCharPtr()` is how a null
argument reaches libc."""


def _addr_at(base: Int, offset: Int) -> Int:
    """Load a pointer sized field out of a C struct and keep it as an integer.

    Pointer fields are read as integers rather than pointers all the way through
    this file. `unsafe_load` only handles scalar pointees, so a pointer to a
    pointer cannot be loaded directly, and holding an address as an `Int` also
    means the NULL check is an integer comparison rather than an optional dance.
    """
    var p = CharPtr(unsafe_from_address=base + offset)
    return Int(p.unsafe_bitcast[UInt64]().unsafe_load(0))


def resolve_ipv4(host: StringSpan) raises -> UInt32:
    """The first IPv4 address for `host`, in host byte order.

    Host order because that is what `connect` and `write_sockaddr_in` take, and
    having exactly one place that flips byte order is worth the extra swap.
    """
    var node = c_string(host)

    # Hints: IPv4, stream. Everything else zero, which means no AI_ADDRCONFIG.
    # AI_ADDRCONFIG would be the polite flag to set but it makes resolution
    # depend on which interfaces happen to be up, so it stays off until there is
    # a reason for it.
    var hints = stack_allocation[ADDRINFO_SIZE, UInt8]()
    for i in range(ADDRINFO_SIZE):
        hints.unsafe_store(i, UInt8(0))
    var hints_i32 = hints.unsafe_bitcast[c_int]()
    hints_i32.unsafe_store(AI_FAMILY_OFF // 4, c_int(AF_INET))
    hints_i32.unsafe_store(AI_SOCKTYPE_OFF // 4, c_int(SOCK_STREAM))

    var result = stack_allocation[1, UInt64]()
    result.unsafe_store(0, UInt64(0))

    var rc = Int(
        external_call["getaddrinfo", c_int](
            node.unsafe_ptr(), NullableCharPtr(), hints, result
        )
    )
    if rc != 0:
        var detail = String("error ") + String(rc)
        var msg = external_call["gai_strerror", NullableCharPtr](c_int(rc))
        if msg:
            detail = from_c_string(msg.value(), 256)
        raise Error("cannot resolve " + String(host) + ": " + detail)

    var head = Int(result.unsafe_load(0))
    if head == 0:
        raise Error("cannot resolve " + String(host) + ": no addresses")

    # Walk the list rather than trusting the first entry. The hints ask for
    # AF_INET so in practice the first one is right, but a resolver is allowed
    # to return more than was asked for, and reading a sockaddr_in6 as a
    # sockaddr_in would silently produce the wrong address rather than fail.
    var cursor = head
    var found = False
    var addr: UInt32 = 0
    while cursor != 0:
        var entry = CharPtr(unsafe_from_address=cursor)
        var family = Int(
            entry.unsafe_bitcast[c_int]().unsafe_load(AI_FAMILY_OFF // 4)
        )
        if family == AF_INET:
            var sa_addr = _addr_at(cursor, AI_ADDR_OFF)
            if sa_addr != 0:
                # sockaddr_in bytes 4 through 7 hold the address, big endian, at
                # the same offset on both platforms.
                var sa = CharPtr(unsafe_from_address=sa_addr)
                addr = (
                    (UInt32(sa.unsafe_load(4)) << 24)
                    | (UInt32(sa.unsafe_load(5)) << 16)
                    | (UInt32(sa.unsafe_load(6)) << 8)
                    | UInt32(sa.unsafe_load(7))
                )
                found = True
                break
        cursor = _addr_at(cursor, AI_NEXT_OFF)

    _ = external_call["freeaddrinfo", NoneType](
        CharPtr(unsafe_from_address=head)
    )

    if not found:
        raise Error("cannot resolve " + String(host) + ": no IPv4 address")
    return addr


def format_ipv4(addr_host_order: UInt32) -> String:
    """Dotted quad, for log lines and error messages."""
    return (
        String((addr_host_order >> 24) & 0xFF)
        + "."
        + String((addr_host_order >> 16) & 0xFF)
        + "."
        + String((addr_host_order >> 8) & 0xFF)
        + "."
        + String(addr_host_order & 0xFF)
    )
