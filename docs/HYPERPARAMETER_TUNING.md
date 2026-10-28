# Hyperparameter Tuning — Orchestrator Sampling Parameters

Bayesian optimization of the orchestrator LLM's **sampling parameters**
(`temperature`, `top_p`, `top_k`), maximizing the test suite's **Overall pass %**
(handoff ∧ vision ∧ result over all 66 runs). The optimizer is **d-MALIBOO**
(GP + Expected Improvement) — the same engine used by the
`Repetti_VLM-Testing` framework — driven by `scripts/tune_orchestrator.py`
with live **wandb** logging and crash-resumable checkpoints.

The VQA model is held constant; the tuned sampling flags are applied at the
`llama-server` level, so they govern **every LLM call in the stack**
(orchestrator + all specialized agents). The untuned baseline is llama-server's
defaults: `temp 0.8 / top-p 0.95 / top-k 40` — the config the
[model comparison](MODEL_COMPARISON.md) was measured at.

---

## Search space

Discrete grids (the optimizer snaps its proposals onto these values):

| Parameter | Values |
|---|---|
| `temperature` | 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| `top_p` | 0.65, 0.70, 0.77, 0.84, 0.91, 0.98 |
| `top_k` | 10, 20, 30, 40, 50, 60, 70, 80 |

384 possible combos; a budget of 20 trials (5 random init + 15 BO-guided)
explores ~5% of the space and typically converges within the first 10–15.

## One trial =

1. Restart `llama-server` with the model's GGUF + `--temp/--top-p/--top-k`
   (per-model launch config comes from the `REGISTRY` in
   `run_model_comparison.py`: gpt-oss gets `reasoning_effort=low`, Qwen3.6 gets
   `enable_thinking=false` — automatic).
2. Run the full scenario suite (22 scenarios × 3 levels = 66 runs) inside the
   `agents-orchestrator` container — the same runner and ground-truth scoring as
   the benchmark.
3. Parse the result JSON → `overall_pct` (the objective) + handoff/vision/result
   %, latency, tokens.
4. Log everything to wandb, checkpoint the trial, feed `−overall%` back to the
   optimizer.

Typical trial wall-times (66 runs + server restart + per-trial VQA reload):
gemma-E2B ~65 min · gpt-oss-120b ~50 min · qwen36-a3b ~46 min ·
gemma4-12b ~105 min · qwen36-27b ~100 min.

---

## Prerequisites (once per machine)

```bash
# 1. Full stack must be up (12 MCP servers + orchestrator + VQA):
./local-ai-service/run_agentic_system.sh --orchestrator-model <any-key>

# 2. d-MALIBOO next to the repo root (the optimizer engine; gitignored):
git clone https://github.com/RobertoSala98/dMALIBOO.git dMALIBOO

# 3. Tuning venv (gitignored):
python3 -m venv .venv-tuning
.venv-tuning/bin/pip install numpy pandas scipy scikit-learn xgboost matplotlib wandb

# 4. wandb login (once per machine/user):
.venv-tuning/bin/wandb login          # paste API key from https://wandb.ai/authorize
```

## Run — single model

```bash
.venv-tuning/bin/python scripts/tune_orchestrator.py --model qwen36-a3b \
    --n-initial 5 --n-iterations 15          # 20 trials total
```

`--model` accepts registry keys **or** the comparison-table labels
(`gemma-E2B→gemma`, `gemma-12b→gemma4-12b`, `qwen3.6-27B→qwen36-27b`,
`qwen3.6-35B-A3B→qwen36-a3b`).

Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--n-initial` / `--n-iterations` | 5 / 10 | random seeding / BO iterations (total trials = sum) |
| `--levels` | all three | e.g. `--levels clear` cuts trial time to ~1/3 |
| `--scenarios` | all 22 | comma-separated subset (smoke: `--scenarios SC-01 --levels clear`) |
| `--fresh` | — | discard the model's cached trials and start over |
| `--wandb-project` | `vlm-orchestrator-tuning` | wandb destination |
| `--no-ml-af` | — | plain EI acquisition (skip d-MALIBOO's ML-on-target NN) |
| `--suite-timeout` | 10800 s | per-trial suite timeout |

Smoke-test the loop without login: `WANDB_MODE=offline … --scenarios SC-01 --levels clear --n-initial 2 --n-iterations 1`.

## Run — multi-model queue (detached)

```bash
./scripts/tune_all.sh                              # all 5 models, sequential
TUNE_LEVELS=clear TUNE_INIT=4 TUNE_ITER=8 ./scripts/tune_all.sh

# custom queue that survives logout:
setsid nohup bash -c '
  .venv-tuning/bin/python scripts/tune_orchestrator.py --model gemma        --n-initial 5 --n-iterations 15
  .venv-tuning/bin/python scripts/tune_orchestrator.py --model gpt-oss-120b --n-initial 5 --n-iterations 15
' > tuning_run.log 2>&1 &
```

Models share the GPU and port 9000 → **one model at a time per machine**. For
real parallelism, split models across machines (see below).

---

## wandb: what gets logged

One resumable run per model (`tune-<model>`), project `vlm-orchestrator-tuning`:

- **`tuning/*` per trial**, charted against the `trial` axis (`define_metric`):
  temperature, top_p, top_k, overall_pct, handoff/vision/result_pct,
  avg_latency_s, tokens_per_task, trial_wall_s, best_overall_pct, cached.
- **`live/scenarios_done`** — progress ticker every 30 s during a trial.
- **`summary`** — best config / best overall % / trials done, updated live.
- **`tables/trials`** — full trials table, rebuilt every trial.

> ⚠ Implementation note: never log per-trial rows with an explicit low
> `step=` — on resumed runs the internal step counter is already higher and
> wandb **silently drops** those rows. `tuning/*` is pinned to the `trial`
> axis via `define_metric` precisely to avoid this.

Recommended panels: parallel-coordinates over
`tuning/temperature, tuning/top_p, tuning/top_k → tuning/overall_pct`, plus
`tuning/best_overall_pct` vs `trial`.

## Checkpointing & resume

Every completed trial is written to
`tests/results/tuning/<model>/<cfg>.json` (e.g. `t0.70_p0.77_k70.json`), plus
`wandb_run_id.txt` and a final `history.csv`.

- **Resume after crash/stop:** rerun the *same command*. Same seed → the
  optimizer regenerates the same trajectory; completed configs return from
  cache instantly (`[cache]` in the log) and the same wandb run continues.
  A trial interrupted mid-suite fails the `n_records` guard and is re-run.
- **Fresh wandb run, keep compute:** delete `wandb_run_id.txt` only — cached
  trials are replayed into the new run's history.
- **Migrate a model to another machine:** rsync its
  `tests/results/tuning/<model>/` dir — the cached trials carry over.

## Two-machine split (Castor/Pollux)

Each model's optimization is independent → partition models across machines.
A second machine needs the full stack replica: docker images (`docker save |
ssh | docker load`), **both** named volumes (`…_vqa_hf_cache` *and*
`…_vlm_weights` — auto-created volumes start empty and the segmentation server
needs the SAM2/DINO weights), the GGUFs of *its* models, the
`llama.cpp/build-cuda/bin/` binaries, the repo (code + fixtures + ground truth),
the venv, and a wandb login. See the `castor-pollux-split-tuning` memory note /
session logs for the exact procedure used.

## Final results — 100-trial campaign (2026-07-17)

Run on the twin GB10s (finished on Pollux) with `--n-initial 5 --n-iterations 95`
per model, levels `vague,clear,algorithmic` (66 runs/trial). Trial counts exceed
100 where machine reboots forked the BO trajectory — every extra config is a
valid full-suite sample. **Winners come from `scripts/tuning_report.py`**, which
counts only completed 66-record trials: a suite that dies mid-run yields a
partial file whose score is inflated by the easy early scenarios, so partials
are rejected at trial time and excluded from reporting (`REPOSITORY_NOTES.md`
F13/F15 — one such partial faked a 93.8% "winner" for gpt-oss before the guard).

| Model | Valid trials | Best config | Tuned Overall % | Default (0.8/0.95/40) | Δ |
|---|---|---|---|---|---|
| gemma-E2B | 97 | `t0.40 p0.65 k10` (7-way tie at 95.5: also `t0.30/p0.84/k10`, `t0.40/p0.77/k10`, `t0.60/p0.70/k10`, `t0.60/p0.77/k20`, `t0.70/p0.65/k30`, `t0.70/p0.91/k30`) | **95.5** | 90.9 | **+4.6** |
| gpt-oss-120b | 98 | `t0.30 p0.98 k20` (tie: `t0.30/p0.70/k20`) | **92.4** | 87.9 | **+4.5** |
| qwen3.6-35B-A3B | 107 | `t0.90 p0.77 k20` (7-way tie at 92.4, latency-sorted) | 92.4 | **95.5** | −3.1 |

Reading the plateaus: both improvable models optimize in the **low-temperature,
small-top-k region**; qwen3.6-A3B is sampling-insensitive (its 107 configs span
only 87.9–92.4%).

### Confirmation runs — the single-shot winners are noise-inflated

`scripts/confirm_winners.py` re-runs each winner twice (a3b on defaults). Results
(2026-07-17):

| Config | Single-shot | Repeat 1 | Repeat 2 | Mean | Spread | Δ vs single-shot |
|---|---|---|---|---|---|---|
| gemma-E2B `t0.40/p0.65/k10` | 95.5 | 93.9 | 92.4 | **93.2** | 1.5 | −2.3 |
| gpt-oss-120b `t0.30/p0.98/k20` | 92.4 | 84.8 | 87.9 | **86.3** | 3.1 | −6.1 |
| qwen3.6-A3B defaults | 95.5 | 89.4 | 90.9 | **90.2** | 1.5 | −5.3 |

**Interpretation (important methodological caveat).** The objective is the suite
Overall %, evaluated **once** per config. That evaluation is stochastic — the
orchestrator LLM samples its tool-routing decisions, so the same config scores
differently across runs (observed spread 1.5–3.1 points here, and gpt-oss's
single-shot 92.4 was a ~+6 outlier). Bayesian optimization *maximizes*, so it
preferentially surfaces configs that got a lucky high draw: the reported best is
biased upward (max-of-noisy-samples > true mean). The confirmation means are the
honest performance estimates; the tuning "winners" mark good *regions*, not
exact optima. A rigorous redo would average k≥3 evaluations per trial (k× cost)
or add a noise model to the GP — out of scope here, but the confirmation runs
quantify the bias so the numbers aren't over-claimed.

## Outputs

- Best config + score printed at the end of each model's run, and in the wandb
  run summary.
- `tests/results/tuning/<model>/history.csv` — the trials from the final BO run's
  in-memory history with params + all metrics (Recovery-style CSV, same spirit as
  `Repetti_VLM-Testing`). Note: after a crash/resume the campaign can accumulate MORE
  per-config caches (`t*.json`) than the last run's `history.csv` rows — the per-config
  `t*.json` files are the complete, authoritative record; `scripts/tuning_report.py`
  reads those.
- Raw per-trial suite results in `tests/results/tune-<model>-<cfg>_*.json`.
