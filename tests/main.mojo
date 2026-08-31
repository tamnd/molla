"""Entry point for the test suite. Add new test modules here.

Modules that can raise should be wrapped in try and except, reporting through
`suite.fail` so one bad module does not hide the rest of the run. Mojo warns on
an unreachable except, so only wrap the ones that actually raise.
"""

from harness import Suite, finish

import test_gguf
import test_host
import test_http
import test_net


def main():
    var suite = Suite()

    test_host.run(suite)
    test_net.run(suite)
    test_http.run(suite)
    try:
        test_gguf.run(suite)
    except e:
        suite.fail("test_gguf", String(e))

    finish(suite)
