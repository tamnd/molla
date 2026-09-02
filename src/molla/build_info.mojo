"""Constants describing this build.

Version is bumped by the release process, along with pixi.toml and the newest
CHANGELOG heading. `scripts/check-version.sh` fails CI when the three disagree,
which they did for two releases before anybody noticed, because `molla version`
is the first thing anyone pastes into a bug report and it was reporting a
version that was two tags old.
"""

comptime VERSION = "0.2.11"

comptime MOJO_PIN = "1.0.0"
"""The Mojo toolchain version pinned in pixi.toml. Duplicated here so a binary
built against a drifted toolchain is visible without reading the manifest."""
