"""Tests for the registry scanner, against canned responses.

The bodies here are trimmed copies of what ghcr.io actually answered for
linuxcontainers/alpine, with the entry list cut down. Two of them are indented
and two are not, because registries differ on that and the scanner has to not
care.
"""

from harness import Suite

from molla.registry.ghcr import (
    config_digest,
    config_size,
    is_index,
    select_platform,
)
from molla.registry.pull import split_reference

comptime INDEX = String(
    '{"schemaVersion":2,"mediaType":"application/vnd.oci.image.index.v1+json",'
    '"manifests":['
    '{"mediaType":"application/vnd.oci.image.manifest.v1+json",'
    '"digest":"sha256:aaaa","size":670,'
    '"platform":{"architecture":"arm64","os":"linux"}},'
    '{"mediaType":"application/vnd.oci.image.manifest.v1+json",'
    '"digest":"sha256:bbbb","size":670,'
    '"platform":{"architecture":"amd64","os":"linux"}},'
    '{"mediaType":"application/vnd.oci.image.manifest.v1+json",'
    '"digest":"sha256:cccc","size":670,'
    '"platform":{"architecture":"ppc64le","os":"linux"}}]}'
)

comptime INDEX_SPACED = String(
    '{"manifests": [ { "digest": "sha256:dddd", '
    '"platform": { "architecture": "amd64", "os": "linux" } } ] }'
)

comptime MANIFEST = String(
    '{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json",'
    '"config":{"mediaType":"application/vnd.oci.image.config.v1+json",'
    '"digest":"sha256:1dd0f00d","size":1744},'
    '"layers":[{"mediaType":"application/vnd.oci.image.layer.v1.tar+gzip",'
    '"digest":"sha256:9999","size":3400000}]}'
)


def run(mut suite: Suite) raises:
    suite.group("registry")

    suite.check(is_index(INDEX), "an index is recognised")
    suite.check(not is_index(MANIFEST), "a manifest is not an index")

    suite.check(
        select_platform(INDEX, String("linux"), String("amd64"))
        == "sha256:bbbb",
        "the amd64 entry is picked out of the middle",
    )
    suite.check(
        select_platform(INDEX, String("linux"), String("arm64"))
        == "sha256:aaaa",
        "the first entry is picked when it matches",
    )
    suite.check(
        select_platform(INDEX_SPACED, String("linux"), String("amd64"))
        == "sha256:dddd",
        "a pretty printed index parses",
    )

    var missing = False
    try:
        _ = select_platform(INDEX, String("linux"), String("riscv64"))
    except:
        missing = True
    suite.check(missing, "an absent platform raises")

    var wrong_os = False
    try:
        _ = select_platform(INDEX, String("windows"), String("amd64"))
    except:
        wrong_os = True
    suite.check(wrong_os, "the os has to match too")

    suite.check(
        config_digest(MANIFEST) == "sha256:1dd0f00d",
        "the config digest is read, not the layer digest",
    )
    suite.check(config_size(MANIFEST) == 1744, "the config size is read")

    var tagged = split_reference(String("linuxcontainers/alpine:3.20"))
    suite.check(tagged[0] == "linuxcontainers/alpine", "repo before the tag")
    suite.check(tagged[1] == "3.20", "tag after the colon")

    var untagged = split_reference(String("owner/name"))
    suite.check(untagged[1] == "latest", "a missing tag defaults to latest")

    var pinned = split_reference(String("owner/name@sha256:abcd"))
    suite.check(pinned[0] == "owner/name", "repo before the at sign")
    suite.check(pinned[1] == "sha256:abcd", "digest after the at sign")

    var ported = split_reference(String("registry.local:5000/owner/name"))
    suite.check(
        ported[0] == "registry.local:5000/owner/name",
        "a registry port is not mistaken for a tag",
    )
    suite.check(ported[1] == "latest", "and the tag still defaults")
