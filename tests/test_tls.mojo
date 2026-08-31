"""Tests for the TLS policy and the backend probe, without the network.

Two things are checked here and neither needs a socket. The policy decides
which hosts skip verification, and it is worth testing on its own because the
property that matters is a negative one: naming a registry must not name the
CDN it redirects to. That is exactly the kind of rule that looks right in the
source and is wrong in practice.

The probe is the other half of issue #14, the promise that a machine with no
TLS library still runs molla and only loses HTTPS. The way to test a promise
about a missing library is to make one go missing, so these tests point the
loader at a name that does not exist and check that the answer is a report
rather than a crash. The environment is put back afterwards.

Connecting to a real host is not done here. `molla tls` and `molla pull` are
the tests for that and they are run by hand on the fleet, because a test suite
that needs the internet is a test suite that fails on a train.
"""

from std.ffi import c_int, external_call

from harness import Suite

from molla.sys.cstr import c_string
from molla.tls.client import probe
from molla.tls.policy import TlsPolicy


def _setenv(name: String, value: String):
    """Set an environment variable through libc, so `getenv` in the loader path
    sees it. Mojo has no setter of its own in 1.0."""
    var n = c_string(name)
    var v = c_string(value)
    _ = external_call["setenv", c_int](n.unsafe_ptr(), v.unsafe_ptr(), c_int(1))


def _unsetenv(name: String):
    var n = c_string(name)
    _ = external_call["unsetenv", c_int](n.unsafe_ptr())


def _test_policy(mut suite: Suite):
    suite.group("tls policy")

    var strict = TlsPolicy()
    suite.check(strict.verifies("ghcr.io"), "a fresh policy verifies")
    suite.check(
        not strict.any_insecure(), "a fresh policy has nothing named in it"
    )

    var named = TlsPolicy()
    named.allow_insecure("ghcr.io")
    suite.check(
        not named.verifies("ghcr.io"), "the named host stops being verified"
    )

    # The property the whole type exists for. ghcr answers a blob request with
    # a redirect to this host, so a flag that covered it would be turning
    # verification off for a host the response chose.
    suite.check(
        named.verifies("pkg-containers.githubusercontent.com"),
        "the host a redirect names is still verified",
    )
    suite.check(
        named.verifies("ghcr.io.evil.example"),
        "a longer name that starts the same is not the named host",
    )
    suite.check(
        named.verifies("evil.ghcr.io"),
        "a subdomain of the named host is not the named host",
    )

    var upper = TlsPolicy()
    upper.allow_insecure("GHCR.IO")
    suite.check(
        not upper.verifies("ghcr.io"), "the host is matched case insensitively"
    )
    suite.check(
        not upper.verifies("Ghcr.Io"), "and so is the host being looked up"
    )

    var twice = TlsPolicy()
    twice.allow_insecure("ghcr.io")
    twice.allow_insecure("ghcr.io")
    twice.allow_insecure("example.test")
    suite.check(
        twice.insecure_list() == "ghcr.io, example.test",
        "naming a host twice records it once",
    )
    suite.check(twice.any_insecure(), "a policy with a host in it says so")


def _test_probe(mut suite: Suite):
    suite.group("tls backend")

    var here = probe()
    suite.check(
        here.available or here.detail.byte_length() > 0,
        "the probe either loads a backend or says why not",
    )
    if here.available:
        suite.check(
            here.backend.byte_length() > 0, "an available backend is named"
        )
        suite.check(
            here.max_protocol.startswith("TLS"),
            "and says how high it can negotiate",
        )
    else:
        # Not a failure. A machine can legitimately have no TLS library, which
        # is the case this whole module is about, and molla still has to run.
        print("            no TLS backend here:", here.detail)


def _test_missing_library(mut suite: Suite):
    """Point the loader at a name that does not exist and check the answer.

    This is the acceptance criterion of issue #14 as a test: a binary on a host
    with no TLS library starts and only loses HTTPS. Everything above the probe
    is still running when it comes back false, which is the assertion.
    """
    suite.group("tls without a library")

    var linux_key = String("MOLLA_LIBSSL")
    var macos_key = String("MOLLA_SECURITY")
    var bogus = String("molla-no-such-tls-library.so.99")

    var before = probe()

    _setenv(linux_key, bogus)
    _setenv(macos_key, bogus)

    var broken = probe()
    suite.check(
        not broken.available, "a library that does not load is not available"
    )
    suite.check(
        broken.detail.find(bogus) >= 0,
        "and the message names what it tried to load",
    )
    suite.check(
        broken.detail.find("HTTPS") >= 0 or broken.detail.find("OpenSSL") >= 0,
        "and says what is lost",
    )

    _unsetenv(linux_key)
    _unsetenv(macos_key)

    var restored = probe()
    suite.check(
        restored.available == before.available,
        "removing the override puts the backend back",
    )


def run(mut suite: Suite):
    _test_policy(suite)
    _test_probe(suite)
    _test_missing_library(suite)
