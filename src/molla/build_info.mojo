"""Constants describing this build.

Version is bumped by the release process. Keep it in step with the tag, because
`molla version` is the first thing anyone will paste into a bug report.
"""

comptime VERSION = "0.0.2"

comptime MOJO_PIN = "1.0.0"
"""The Mojo toolchain version pinned in pixi.toml. Duplicated here so a binary
built against a drifted toolchain is visible without reading the manifest."""
