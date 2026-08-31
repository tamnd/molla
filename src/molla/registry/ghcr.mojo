"""Enough of the OCI distribution protocol to fetch one blob from ghcr.io.

The flow is fixed and short. Ask `ghcr.io/token` for a pull token, ask for the
tag, get back an index listing one manifest per platform, ask for the manifest
that matches, read the config digest out of it, and pull that blob. Every step
is a GET, every response is JSON except the blob, and the blob is checked
against the digest that named it.

Two things about ghcr are worth knowing before reading this. Public repositories
still need a token, they just hand one out to anybody who asks. And a blob
request answers 307 to a signed URL on a completely different host, so the
client has to follow redirects across hosts and has to drop the bearer token
when it does.

There is no JSON parser here and there should not be one until something needs
more than a few fields. What is here is a scanner that finds `"key":"value"`
pairs, which is enough for digests and sizes and would be wrong for anything
nested or escaped. It is honest about that: every lookup names the key it wants
and fails loudly when the key is not there.
"""

from molla.http.client import get
from molla.sys.sha256 import sha256_hex

comptime REGISTRY = "ghcr.io"

comptime MANIFEST_ACCEPT = (
    "application/vnd.oci.image.index.v1+json,"
    "application/vnd.oci.image.manifest.v1+json,"
    "application/vnd.docker.distribution.manifest.list.v2+json,"
    "application/vnd.docker.distribution.manifest.v2+json"
)
"""Ask for all four because registries store what they were pushed. A repo
pushed with an old Docker client is still a Docker manifest list years later,
and asking only for the OCI types gets a 404 that looks like a missing tag."""


def _string_field(
    body: String, key: String, from_byte: Int = 0
) raises -> String:
    """The value of the first `"key":"..."` at or after `from_byte`.

    Whitespace between the colon and the value is allowed because registries
    differ on whether they pretty print. Escapes are not handled, which is fine
    for digests and media types and would not be for a description field.
    """
    var needle = String('"') + key + '"'
    var at = body.find(needle, from_byte)
    if at < 0:
        raise Error("no " + key + " in registry response")

    var p = body.unsafe_ptr()
    var i = at + needle.byte_length()
    var n = body.byte_length()
    while i < n and (p.unsafe_load(i) == 32 or p.unsafe_load(i) == 9):
        i += 1
    if i >= n or p.unsafe_load(i) != 58:
        raise Error("no colon after " + key + " in registry response")
    i += 1
    while i < n and (p.unsafe_load(i) == 32 or p.unsafe_load(i) == 9):
        i += 1
    if i >= n or p.unsafe_load(i) != 34:
        raise Error(key + " is not a string in registry response")
    i += 1

    var start = i
    while i < n and p.unsafe_load(i) != 34:
        i += 1
    if i >= n:
        raise Error("unterminated " + key + " in registry response")
    return String(body[byte=start:i])


def _int_field(body: String, key: String, from_byte: Int = 0) raises -> Int:
    var needle = String('"') + key + '"'
    var at = body.find(needle, from_byte)
    if at < 0:
        raise Error("no " + key + " in registry response")

    var p = body.unsafe_ptr()
    var i = at + needle.byte_length()
    var n = body.byte_length()
    while i < n and p.unsafe_load(i) != 58:
        i += 1
    i += 1
    while i < n and (p.unsafe_load(i) == 32 or p.unsafe_load(i) == 9):
        i += 1

    var start = i
    while i < n and p.unsafe_load(i) >= 48 and p.unsafe_load(i) <= 57:
        i += 1
    if i == start:
        raise Error(key + " is not a number in registry response")
    return Int(body[byte=start:i])


def fetch_token(repo: String) raises -> String:
    """A pull scoped bearer token for `repo`.

    Anonymous. molla has no credential store yet and public images do not need
    one. Private repositories will 401 here and that is the correct failure
    until M3 adds a keychain.
    """
    var url = (
        String("https://")
        + REGISTRY
        + "/token?service="
        + REGISTRY
        + "&scope=repository:"
        + repo
        + ":pull"
    )
    var response = get(url, String("application/json"))
    if response.status != 200:
        raise Error(
            "token request for " + repo + " returned " + String(response.status)
        )
    return _string_field(response.body_text(), String("token"))


def _manifest_url(repo: String, reference: String) -> String:
    return (
        String("https://")
        + REGISTRY
        + "/v2/"
        + repo
        + "/manifests/"
        + reference
    )


def fetch_manifest(
    repo: String, reference: String, token: String
) raises -> String:
    """The manifest or index for a tag or a digest, as text."""
    var response = get(
        _manifest_url(repo, reference),
        MANIFEST_ACCEPT,
        String("Bearer ") + token,
    )
    if response.status != 200:
        raise Error(
            repo
            + ":"
            + reference
            + " manifest returned "
            + String(response.status)
        )
    return response.body_text()


def select_platform(index: String, os: String, arch: String) raises -> String:
    """The digest of the manifest for one platform inside an index.

    An index entry puts its digest before its platform block, so the scan
    remembers the last digest it saw and returns it when the platform matches.
    That is one forward pass and it does not care how the entry is indented,
    which a bracket matcher would.
    """
    var want = String('"architecture":"') + arch + '"'
    var want_spaced = String('"architecture": "') + arch + '"'
    var want_os = String('"os":"') + os + '"'
    var want_os_spaced = String('"os": "') + os + '"'

    var digest = String("")
    var i = 0
    var n = index.byte_length()
    while i < n:
        var d = index.find('"digest"', i)
        var a = index.find(want, i)
        if a < 0:
            a = index.find(want_spaced, i)

        if a < 0:
            break
        if d >= 0 and d < a:
            digest = _string_field(index, String("digest"), d)
            i = d + 8
            continue

        # The architecture matched. The os sits right after it in every
        # registry we have seen, so check the next 64 bytes rather than the
        # rest of the document, which would match the next entry along.
        var stop = a + 64
        if stop > n:
            stop = n
        var window = String(index[byte=a:stop])
        if window.find(want_os) >= 0 or window.find(want_os_spaced) >= 0:
            if digest.byte_length() == 0:
                raise Error(
                    "index entry for " + os + "/" + arch + " has no digest"
                )
            return digest
        i = a + 1

    raise Error("no manifest for " + os + "/" + arch + " in this index")


def fetch_blob(
    repo: String, digest: String, token: String
) raises -> List[UInt8]:
    """Pull one blob and check it against the digest that named it.

    The check is the whole point. Everything above this line trusts TLS and the
    registry, and the digest is what makes that trust unnecessary: the bytes
    either hash to the name they were fetched under or they are thrown away.
    """
    if not digest.startswith("sha256:"):
        raise Error("only sha256 digests are supported: " + digest)

    var url = String("https://") + REGISTRY + "/v2/" + repo + "/blobs/" + digest
    var response = get(url, String(""), String("Bearer ") + token)
    if response.status != 200:
        raise Error("blob " + digest + " returned " + String(response.status))

    var want = String(digest[byte=7:])
    var got = sha256_hex(response.body)
    if got != want:
        raise Error(
            "digest mismatch for "
            + digest
            + ": got sha256:"
            + got
            + " over "
            + String(len(response.body))
            + " bytes"
        )
    return response^.take_body()


def is_index(body: String) -> Bool:
    """Whether a manifest response is a multi platform index.

    By the presence of a `manifests` array rather than by media type, because
    the media type header and the media type in the body do not always agree
    and the body is what we are about to read.
    """
    return body.find('"manifests"') >= 0


def config_digest(manifest: String) raises -> String:
    """The digest of the image config blob.

    The config is the smallest blob in any image and it is a real one, which
    makes it the right thing to pull when the question is whether pulling
    works at all.
    """
    var at = manifest.find('"config"')
    if at < 0:
        raise Error("manifest has no config section")
    return _string_field(manifest, String("digest"), at)


def config_size(manifest: String) raises -> Int:
    var at = manifest.find('"config"')
    if at < 0:
        raise Error("manifest has no config section")
    return _int_field(manifest, String("size"), at)
