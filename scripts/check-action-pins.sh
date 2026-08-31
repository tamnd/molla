#!/usr/bin/env bash
# Check that every action is pinned to a commit SHA and not to a tag object SHA.
#
# Pinning to a 40 character hex string looks the same either way, and zizmor and
# Scorecard both accept it, so this mistake is invisible until something else
# trips over it. What tripped over it here was Scorecard's publish step, which
# verifies the action it ran as and rejected ours with "imposter commit: ... does
# not belong to ossf/scorecard-action". The scan had run fine and only the upload
# failed, so the workflow went red for a reason that had nothing to do with the
# result.
#
# The cause is that `git rev-parse v2.4.2` on an annotated tag gives you the tag
# object, not the commit it points at. You have to dereference it, either with
# `v2.4.2^{commit}` or by reading `.object.sha` from the tags API. Six of our
# pins were tag objects, and the ones that were are exactly the actions that
# publish annotated tags for moving majors like v3 and v4.
#
# Needs a GitHub token, so it runs in CI rather than in `pixi run check`.

set -euo pipefail

cd "$(dirname "$0")/.."

status=0

while read -r ref; do
    [ -n "$ref" ] || continue
    sha="${ref##*@}"
    path="${ref%@*}"
    repo="$(echo "$path" | cut -d/ -f1,2)"

    if gh api "repos/$repo/git/commits/$sha" --jq '.sha' >/dev/null 2>&1; then
        printf '  ok        %s\n' "$ref"
        continue
    fi

    deref="$(gh api "repos/$repo/git/tags/$sha" --jq '.object.sha' 2>/dev/null || true)"
    if [ -n "$deref" ]; then
        printf '  TAG OBJ   %s\n' "$ref"
        printf '            pin the commit instead: %s\n' "$deref"
    else
        printf '  UNKNOWN   %s\n' "$ref"
        printf '            not a commit or a tag in %s\n' "$repo"
    fi
    status=1
done < <(grep -rhoE 'uses: [^ ]+@[0-9a-f]{40}' .github/workflows/ | sed 's/uses: //' | sort -u)

if [ "$status" -ne 0 ]; then
    echo
    echo "at least one action is pinned to something that is not a commit"
fi

exit "$status"
