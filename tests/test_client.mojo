"""Tests for the HTTPS client, without the network.

URL splitting is the part most likely to be wrong in a way that only shows up
against one particular registry, so it gets the cases that a real redirect
produces: a long query string, a port, a path that is only a slash.
"""

from harness import Suite

from molla.http.client import parse_url


def run(mut suite: Suite) raises:
    suite.group("http client")

    var plain = parse_url(
        String("https://ghcr.io/v2/owner/name/blobs/sha256:ab")
    )
    suite.check(plain.host == "ghcr.io", "host is split off")
    suite.check(plain.port == 443, "port defaults to 443")
    suite.check(
        plain.path == "/v2/owner/name/blobs/sha256:ab",
        "a colon in the path is not a port",
    )

    var bare = parse_url(String("https://example.com"))
    suite.check(bare.path == "/", "a missing path becomes a slash")

    var ported = parse_url(String("https://localhost:8443/x"))
    suite.check(ported.host == "localhost", "host without the port")
    suite.check(ported.port == 8443, "port is parsed")
    suite.check(
        ported.text() == "https://localhost:8443/x",
        "a non default port round trips",
    )

    var query = parse_url(
        String(
            "https://pkg-containers.githubusercontent.com/a/b?se=2026&sig=x%2Fy"
        )
    )
    suite.check(
        query.path == "/a/b?se=2026&sig=x%2Fy", "the query stays on the path"
    )

    var refused = False
    try:
        _ = parse_url(String("http://ghcr.io/v2/"))
    except:
        refused = True
    suite.check(refused, "plain http is refused")

    var rejected = False
    try:
        _ = parse_url(String("ghcr.io/v2/"))
    except:
        rejected = True
    suite.check(rejected, "a URL without a scheme is rejected")
