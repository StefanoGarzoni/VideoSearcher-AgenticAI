#!/usr/bin/env bash
# Prune old session directories under ./data/runs/, keeping the N most recent.
# Usage: ./scripts/cleanup_runs.sh [N]   (default: keep 3)
set -euo pipefail

KEEP=${1:-3}
RUN_DIR="$(dirname "$0")/../data/runs"

if [ ! -d "$RUN_DIR" ]; then
    echo "No runs directory at $RUN_DIR — nothing to clean."
    exit 0
fi

TO_DELETE=$(ls -dt "$RUN_DIR"/20* 2>/dev/null | tail -n +$((KEEP + 1)))

if [ -z "$TO_DELETE" ]; then
    echo "Nothing to delete (≤${KEEP} session dirs found)."
    exit 0
fi

echo "$TO_DELETE" | xargs rm -rf
echo "Kept ${KEEP} most recent sessions; deleted $(echo "$TO_DELETE" | wc -l | tr -d ' ') older ones."
