#!/usr/bin/env bash
# Run the shutdown test in a loop and stop on the first failure.
#
# Issue #15 asks for a shutdown under load that drains cleanly every time in a
# hundred runs, and every time is the part a single run cannot tell you. A race
# between the signal, the workers and the last flush shows up at one run in
# thirty, which is invisible in CI and obvious here.
#
# Each run starts a server on a fresh port, loads every connection with a
# pipelined batch it cannot possibly have answered yet, sends itself SIGTERM,
# and checks that every client got every answer whole. Anything other than a
# clean exit stops the loop with the output of the run that failed.
#
# Usage: scripts/drain-loop.sh [runs] [connections] [deadline_ms]

set -uo pipefail

cd "$(dirname "$0")/.."

runs=${1:-100}
connections=${2:-32}
deadline=${3:-5000}

if [ ! -x build/molla ]; then
    echo "drain-loop: build/molla is missing, run 'pixi run build' first" >&2
    exit 1
fi

echo "drain-loop: $runs runs of $connections connections, ${deadline}ms deadline"

start=$SECONDS
for i in $(seq 1 "$runs"); do
    if ! output=$(./build/molla drain "$connections" "$deadline" 2>&1); then
        echo "drain-loop: run $i failed"
        echo "$output"
        exit 1
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "  $i/$runs ok"
    fi
done

echo "drain-loop: $runs runs, all clean, $((SECONDS - start))s"
