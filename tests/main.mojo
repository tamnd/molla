"""Entry point for the test suite. Add new test modules here.

Modules that can raise should be wrapped in try and except, reporting through
`suite.fail` so one bad module does not hide the rest of the run. Mojo warns on
an unreachable except, so only wrap the ones that actually raise.
"""

from harness import Suite, finish

import test_client
import test_dns
import test_gguf
import test_host
import test_http
import test_net
import test_registry
import test_sha256
import test_sys


def main():
    var suite = Suite()

    test_host.run(suite)
    test_net.run(suite)
    test_http.run(suite)
    test_sha256.run(suite)
    try:
        test_gguf.run(suite)
    except e:
        suite.fail("test_gguf", String(e))
    try:
        test_dns.run(suite)
    except e:
        suite.fail("test_dns", String(e))
    try:
        test_client.run(suite)
    except e:
        suite.fail("test_client", String(e))
    try:
        test_registry.run(suite)
    except e:
        suite.fail("test_registry", String(e))
    try:
        test_sys.run(suite)
    except e:
        suite.fail("test_sys", String(e))

    finish(suite)
