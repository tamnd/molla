"""Pulling from an OCI registry.

M0 only asks whether the TLS path works end to end, so this package goes as far
as fetching a token, resolving a tag to a manifest, and pulling one blob with
its digest checked. It does not write anything to disk, does not resume, and
does not stream. M3 owns the real puller and will rewrite most of this.

`ghcr.mojo` is the only backend for now. The protocol is the OCI distribution
spec and Docker Hub speaks the same one, so widening this later is mostly a
matter of where the token comes from.
"""
