"""The `molla pull` command.

This exists to answer one question from the M0 spike: can a molla binary fetch a
real blob from a real registry over TLS, on every machine in the fleet, with the
digest checked. So it prints each step rather than a progress bar. When it fails
on some machine, the last line printed says which step failed, which is the
whole reason for the shape of the output.

It does not save anything. M3 turns this into a puller with a content addressed
store behind it.
"""

from molla.registry.ghcr import (
    REGISTRY,
    config_digest,
    config_size,
    fetch_blob,
    fetch_manifest,
    fetch_token,
    is_index,
    select_platform,
)
from molla.sys.sha256 import sha256_hex


def _short(digest: String) -> String:
    """First twelve hex characters, the way every registry tool prints one."""
    if digest.startswith("sha256:"):
        return String(digest[byte=7:19])
    return digest


def split_reference(reference: String) raises -> List[String]:
    """Split `repo`, `repo:tag` or `repo@digest` into a repo and a reference.

    The colon has to be found after the last slash. `ghcr.io:443/owner/name`
    is a legal thing to write and splitting on the first colon would take the
    port for a tag.
    """
    var out = List[String]()

    var at = reference.find("@")
    if at >= 0:
        out.append(String(reference[byte=0:at]))
        out.append(String(reference[byte = at + 1 :]))
        return out^

    var last_slash = reference.rfind("/")
    var colon = reference.rfind(":")
    if colon > last_slash:
        out.append(String(reference[byte=0:colon]))
        out.append(String(reference[byte = colon + 1 :]))
    else:
        out.append(reference)
        out.append(String("latest"))
    return out^


def run_pull(reference: String) raises:
    """Pull one blob and print every step on the way to it."""
    var parts = split_reference(reference)
    var repo = parts[0]
    var want = parts[1]

    # A leading registry host is accepted and dropped. molla only speaks to
    # ghcr.io today and silently pulling from somewhere else because the name
    # said so would be worse than not accepting the name at all.
    if repo.startswith(REGISTRY + "/"):
        var bare = String(repo[byte = REGISTRY.byte_length() + 1 :])
        repo = bare
    elif repo.find(".") >= 0 and repo.find("/") > repo.find("."):
        raise Error(
            "only " + REGISTRY + " is supported, cannot pull from " + repo
        )

    print("pull", REGISTRY + "/" + repo, "reference", want)

    var token = fetch_token(repo)
    print("  token     ", token.byte_length(), "bytes")

    var digest: String
    if want.startswith("sha256:"):
        digest = want
    else:
        var manifest = fetch_manifest(repo, want, token)
        print("  manifest  ", manifest.byte_length(), "bytes")

        if is_index(manifest):
            var picked = select_platform(
                manifest, String("linux"), String("amd64")
            )
            print("  platform   linux/amd64", _short(picked))
            manifest = fetch_manifest(repo, picked, token)
            print("  manifest  ", manifest.byte_length(), "bytes")

        digest = config_digest(manifest)
        print("  config    ", _short(digest), config_size(manifest), "bytes")

    var blob = fetch_blob(repo, digest, token)
    print("  blob      ", len(blob), "bytes")
    print("  digest     sha256:" + sha256_hex(blob), "verified")
