#!/usr/bin/env bash
# Fail if the three places that hold molla's version number disagree.
#
# There are three: pixi.toml, which is the package version, VERSION in
# src/molla/build_info.mojo, which is what `molla version` prints, and the
# newest section heading in CHANGELOG.md. A release bumps all three and the
# tag is named after them.
#
# This exists because it already went wrong. 0.1.3 and 0.1.4 both shipped with
# build_info.mojo still saying 0.1.2, so every binary built from those tags
# reported a version that was two releases old. Nothing failed, because nothing
# was looking, and `molla version` is the first thing anybody pastes into a bug
# report.

set -euo pipefail

cd "$(dirname "$0")/.."

pixi_version=$(sed -n 's/^version = "\(.*\)"$/\1/p' pixi.toml | head -1)
build_version=$(sed -n 's/^comptime VERSION = "\(.*\)"$/\1/p' src/molla/build_info.mojo | head -1)
changelog_version=$(sed -n 's/^## \[\(.*\)\] - .*$/\1/p' CHANGELOG.md | head -1)

printf '  pixi.toml       %s\n' "$pixi_version"
printf '  build_info.mojo %s\n' "$build_version"
printf '  CHANGELOG.md    %s\n' "$changelog_version"

if [ -z "$pixi_version" ] || [ -z "$build_version" ] || [ -z "$changelog_version" ]; then
    echo
    echo "could not read a version out of one of the three files"
    exit 1
fi

if [ "$pixi_version" != "$build_version" ] || [ "$pixi_version" != "$changelog_version" ]; then
    echo
    echo "the three versions disagree. A release bumps all three:"
    echo "  version in pixi.toml"
    echo "  comptime VERSION in src/molla/build_info.mojo"
    echo "  the newest ## [x.y.z] heading in CHANGELOG.md"
    exit 1
fi
