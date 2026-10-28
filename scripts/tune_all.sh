#!/usr/bin/env bash
# tune_all.sh — run Bayesian sampling-parameter tuning for all 5 comparison
# models, sequentially (they share the GPU + port 9000). Each model is fully
# resumable: rerunning this script skips completed trials via the per-config
# cache in tests/results/tuning/<model>/ and resumes the same wandb run.
#
# Usage:
#   ./scripts/tune_all.sh                       # full suite/levels, 5+10 trials
#   TUNE_LEVELS=clear ./scripts/tune_all.sh     # 1/3 of the runtime
#   TUNE_INIT=4 TUNE_ITER=8 ./scripts/tune_all.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$PROJECT_ROOT/.venv-tuning/bin/python"

MODELS=(gemma gemma4-12b gpt-oss-120b qwen36-27b qwen36-a3b)
LEVELS="${TUNE_LEVELS:-vague,clear,algorithmic}"
N_INIT="${TUNE_INIT:-5}"
N_ITER="${TUNE_ITER:-10}"

for m in "${MODELS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  TUNING: $m   (levels=$LEVELS, budget=$N_INIT+$N_ITER)"
    echo "════════════════════════════════════════════════════════"
    "$PY" "$SCRIPT_DIR/tune_orchestrator.py" \
        --model "$m" --levels "$LEVELS" \
        --n-initial "$N_INIT" --n-iterations "$N_ITER"
done

echo ""
echo "ALL MODELS TUNED — per-model best configs printed above;"
echo "histories in tests/results/tuning/<model>/history.csv; live dashboards on wandb."
