#!/usr/bin/env bash
# Full 66-scenario comparison for the two NEW viable orchestrators (mistral-small,
# falcon3-10b). The existing strong 4 (gemma-E2B/12b, gpt-oss, qwen) are reused from
# run 2026-06-20T09-49-27 — not re-run. Afterwards, consolidate all runs into one
# CSV and rebuild the charts. Memory-gated so it's safe to leave unattended.
# Resolve the repo root from this script's location so it works for any user/clone.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "=== FINAL COMPARISON (mistral-small + falcon3-10b) started $(date) ==="
python3 scripts/run_model_comparison.py \
    --models mistral-small,falcon3-10b \
    --min-free-gb 40 --ttft-samples 3
rc=$?
echo "=== run_model_comparison exit=$rc; consolidating $(date) ==="
python3 scripts/consolidate_metrics.py || true
echo "=== building charts $(date) ==="
python3 scripts/visualise_results.py --charts || true
echo "=== FINAL COMPARISON done $(date) ==="
