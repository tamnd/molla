#!/usr/bin/env bash
# Parse every workflow and config YAML file, and fail if any of them is invalid.
#
# This exists because a syntax error in ci.yml cannot be caught by ci.yml. A
# broken workflow file does not fail a job, it fails the whole run before any
# job starts, and GitHub shows the run named after the file path rather than the
# workflow. That is easy to miss. Run this before pushing workflow changes.
#
# The specific mistake that prompted it: an unquoted `if:` expression containing
# the label name 'needs: hardware'. A colon followed by a space ends a plain
# YAML scalar, so the rest of the line became a syntax error.

set -euo pipefail

cd "$(dirname "$0")/.."

status=0

check() {
    local file="$1"
    if python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" "$file" 2>/tmp/yamlerr; then
        printf '  ok      %s\n' "$file"
    else
        printf '  BROKEN  %s\n' "$file"
        sed 's/^/            /' /tmp/yamlerr
        status=1
    fi
}

shopt -s nullglob
for file in .github/workflows/*.yml .github/workflows/*.yaml \
    .github/dependabot.yml .github/ISSUE_TEMPLATE/*.yml; do
    check "$file"
done

if [ "$status" -ne 0 ]; then
    echo
    echo "at least one YAML file is invalid"
fi

exit "$status"
