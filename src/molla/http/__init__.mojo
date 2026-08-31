"""HTTP/1.1 for molla.

`request` parses, `response` builds, `server` runs the loop. Nothing in here
calls `external_call`; everything that touches the kernel goes through
`molla.sys`, which is D1 in `docs/design.md`.

This is the M0 spike, so the server answers every path with the same body. What
it is proving is the parse and respond path, not routing.
"""
