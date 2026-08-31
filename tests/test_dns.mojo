"""Tests for name resolution.

Only localhost is resolved, because a test that needs DNS is a test that fails
on a laptop on a train. That still exercises the whole getaddrinfo path
including the addrinfo field offsets, which is the part that differs between
macOS and Linux and the part that would silently return a wrong address.
"""

from harness import Suite

from molla.sys.dns import format_ipv4, resolve_ipv4


def run(mut suite: Suite) raises:
    suite.group("dns")

    suite.check(format_ipv4(0x7F000001) == "127.0.0.1", "loopback formats")
    suite.check(format_ipv4(0) == "0.0.0.0", "all zeroes formats")
    suite.check(
        format_ipv4(0xFFFFFFFF) == "255.255.255.255", "all ones formats"
    )
    suite.check(
        format_ipv4(0x14CDF3A4) == "20.205.243.164", "high bit in each octet"
    )

    var local = resolve_ipv4(String("localhost"))
    suite.check(local == 0x7F000001, "localhost resolves to 127.0.0.1")

    var literal = resolve_ipv4(String("93.184.216.34"))
    suite.check(literal == 0x5DB8D822, "a dotted quad resolves to itself")

    var failed = False
    try:
        _ = resolve_ipv4(String("this-host-does-not-exist.invalid"))
    except:
        failed = True
    suite.check(failed, "an unresolvable name raises")
