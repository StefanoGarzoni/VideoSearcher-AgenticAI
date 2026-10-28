#!/usr/bin/env python3
"""
tune_orchestrator.py — Bayesian hyperparameter tuning of the orchestrator LLM's
sampling parameters (temperature / top-p / top-k), driven by the d-MALIBOO
optimizer from the Repetti_VLM-Testing framework, with live wandb logging and
crash-resumable checkpoints.

Search space (discrete, as specified):
    temperature: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    top_p:       0.65, 0.70, 0.77, 0.84, 0.91, 0.98
    top_k:       10, 20, 30, 40, 50, 60, 70, 80

One TRIAL = restart llama-server with the model + sampling flags, run the
orchestrator test suite inside the agents container (same runner as the model
comparison), score it, log to wandb. The objective maximized is the suite's
overall pass % (handoff AND vision AND result gates).

Architecture notes:
  * The Agents SDK sends no sampling params (ModelSettings defaults are None),
    so llama-server's --temp/--top-p/--top-k govern every LLM call in the stack.
    The reference/baseline config is llama-server's defaults: 0.8 / 0.95 / 40.
  * Optimizer = d-MALIBOO (GP + EI acquisition), invoked with the exact call
    pattern used by Repetti_VLM-Testing/src/BayesianOptimizer.py. We drive the
    engine directly instead of importing that class because it pulls torch via
    TaskExecutor — unnecessary on the host (our "executor" is the docker suite).
  * Checkpoint/resume: every completed trial is written to
    tests/results/tuning/<model>/<cfg>.json. On restart with the same seed the
    optimizer regenerates the same trajectory; cached configs return instantly,
    so the run effectively resumes where it crashed (Recovery.py-style).
    The wandb run resumes too (stable run id + resume="allow").

Usage:
    .venv-tuning/bin/python scripts/tune_orchestrator.py --model qwen36-a3b
    .venv-tuning/bin/python scripts/tune_orchestrator.py --model gemma-12b \
        --n-initial 5 --n-iterations 10 --levels clear
    WANDB_MODE=offline .venv-tuning/bin/python scripts/tune_orchestrator.py \
        --model gemma --scenarios SC-01 --levels clear --n-initial 2 --n-iterations 1

Model keys accept both registry keys and the comparison-table labels:
    gemma-E2B→gemma, gemma-12b→gemma4-12b, qwen3.6-27B→qwen36-27b,
    qwen3.6-35B-A3B→qwen36-a3b, gpt-oss-120b→gpt-oss-120b
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DMALIBOO_SRC = PROJECT_ROOT / "dMALIBOO" / "src"
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
TUNING_DIR = RESULTS_DIR / "tuning"

sys.path.insert(0, str(DMALIBOO_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import wandb  # noqa: E402

# d-MALIBOO engine — same modules Repetti_VLM-Testing's BayesianOptimizer uses.
from GaussianProcess import GP  # noqa: E402
from AcquisitionFunction import AF  # noqa: E402
from BayesianOptimization import BO  # noqa: E402
from Logger import Logger  # noqa: E402

# Reuse the comparison harness: REGISTRY + llama-server lifecycle + suite exec.
_spec = importlib.util.spec_from_file_location(
    "rmc", PROJECT_ROOT / "scripts" / "run_model_comparison.py")
rmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmc)

# ── Search space (exact values requested) ────────────────────────────────────

SPACE: Dict[str, List[float]] = {
    "temperature": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "top_p":       [0.65, 0.70, 0.77, 0.84, 0.91, 0.98],
    "top_k":       [10, 20, 30, 40, 50, 60, 70, 80],
}
PARAM_NAMES = list(SPACE.keys())

# Comparison-table labels → registry keys.
MODEL_ALIASES = {
    "gemma-e2b": "gemma", "gemma-12b": "gemma4-12b",
    "qwen3.6-27b": "qwen36-27b", "qwen3.6-35b-a3b": "qwen36-a3b",
}


def resolve_model_key(name: str) -> str:
    key = MODEL_ALIASES.get(name.lower(), name)
    if key not in rmc.REGISTRY:
        raise SystemExit(
            f"Unknown model '{name}'. Registry keys: {list(rmc.REGISTRY)}; "
            f"aliases: {list(MODEL_ALIASES)}")
    return key


def cfg_key(params: Dict[str, float]) -> str:
    """Stable, filesystem-safe key for one sampling config."""
    return (f"t{params['temperature']:.2f}"
            f"_p{params['top_p']:.2f}"
            f"_k{int(round(params['top_k']))}")


def wait_llama_healthy(timeout_s: int = 900) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if rmc._llama_healthy():
            return True
        time.sleep(3)
    return False


# ── Objective ────────────────────────────────────────────────────────────────

class SuiteObjective:
    """Callable objective for d-MALIBOO: params → −overall_pass_fraction."""

    def __init__(self, model_key: str, model: dict, args, cache_dir: Path,
                 wb_run, expected_records: int):
        self.model_key = model_key
        self.model = model
        self.args = args
        self.cache_dir = cache_dir
        self.wb = wb_run
        self.expected_records = expected_records
        self.trial_idx = 0          # counts objective evaluations (incl. cached)
        self.best = None            # best overall_pct so far
        self.history: List[dict] = []

    # -- cache ---------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cached(self, key: str) -> Optional[dict]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            entry = json.loads(p.read_text())
        except Exception:
            return None
        if entry.get("n_records", 0) < self.expected_records:
            return None  # incomplete trial (crashed mid-suite) → rerun
        return entry

    # -- suite execution -----------------------------------------------------

    def _run_trial(self, params: Dict[str, float], key: str) -> dict:
        sampling_args = [
            "--temp", f"{params['temperature']:.2f}",
            "--top-p", f"{params['top_p']:.2f}",
            "--top-k", str(int(round(params["top_k"]))),
        ]
        reasoning = "low" if self.model.get("gpt_oss") else None

        rmc.stop_llama_server()
        rmc.start_llama_server(self.model, reasoning,
                               extra_server_args=sampling_args)
        if not wait_llama_healthy():
            raise RuntimeError("llama-server did not become healthy")

        run_id = f"tune-{self.model_key}-{key}"
        slug = rmc.slugify(self.model["name"])
        result_file = RESULTS_DIR / f"{run_id}_{slug}.json"
        if result_file.exists():
            result_file.unlink()  # partial leftovers from a crashed trial

        # Live progress → wandb while the suite runs in the container.
        stop_evt = threading.Event()

        def _poll_progress():
            while not stop_evt.wait(30):
                try:
                    n = len(json.loads(result_file.read_text()))
                except Exception:
                    n = 0
                try:
                    self.wb.log({"live/scenarios_done": n,
                                 "live/trial": self.trial_idx})
                except Exception:
                    pass

        poller = threading.Thread(target=_poll_progress, daemon=True)
        poller.start()
        t0 = time.monotonic()
        try:
            rmc.run_suite_in_container(
                self.model, run_id,
                self.args.scenarios, self.args.levels,
                self.args.suite_timeout,
                # Warm HF cache per trial → a healthy prewarm takes ~4 min; the
                # benchmark's 2 h cold-load default would let a wedged GPU burn
                # hours per trial before the suite even starts.
                extra_env={"VQA_PREWARM_TIMEOUT_S": "1800"})
        except subprocess.TimeoutExpired:
            # The timeout kills only the docker-exec CLIENT; run_tests.py keeps
            # running inside the container. Kill it so it can't overlap the next
            # trial/model (two concurrent suites + a swapped llama-server GGUF
            # would contaminate results), then fail this trial loudly.
            subprocess.run(["docker", "exec", rmc.ORCH_CONTAINER,
                            "pkill", "-f", "run_tests.py"], check=False)
            raise RuntimeError(
                f"suite timed out after {self.args.suite_timeout}s for {key}")
        finally:
            stop_evt.set()
            poller.join(timeout=5)
        wall_s = time.monotonic() - t0

        records = json.loads(result_file.read_text())
        n = len(records)
        if n < self.expected_records:
            # A partial suite (transient death mid-run — e.g. a co-tenant grabbing
            # the GPU) writes fewer than expected records. run_tests.py exits
            # CLEANLY with partial output (no exception/timeout), so without this
            # guard _run_trial would score just those few — usually the EASY early
            # audio scenarios (SC-01..) — and cache a spuriously high result. On
            # 2026-07-15 a 16/66 partial scored 93.8% and beat gpt-oss's true
            # 66/66 best of 92.4%. Treat it like a failed trial: raise → caught in
            # __call__ (neutral value, NOT cached, re-evaluated on resume).
            raise RuntimeError(
                f"partial suite for {key}: {n}/{self.expected_records} records "
                f"(suite died mid-run); refusing to score a partial")

        def pct(field):
            vals = [r.get(field) for r in records if r.get(field) is not None]
            return round(100.0 * sum(bool(v) for v in vals) / len(vals), 1) if vals else None

        overall = round(100.0 * sum(bool(r.get("overall_pass")) for r in records) / max(n, 1), 1)
        lat = [r.get("execution_time_ms") for r in records if r.get("execution_time_ms")]
        toks = [r.get("total_tokens") for r in records if r.get("total_tokens")]

        return {
            "params": {"temperature": params["temperature"],
                       "top_p": params["top_p"],
                       "top_k": int(round(params["top_k"]))},
            "cfg_key": key,
            "run_id": run_id,
            "n_records": n,
            "overall_pct": overall,
            "handoff_pct": pct("handoff_match"),
            "vision_pct": pct("vision_pass"),
            "result_pct": pct("result_pass"),
            "avg_latency_s": round(sum(lat) / len(lat) / 1000.0, 1) if lat else None,
            "tokens_per_task": round(sum(toks) / len(toks)) if toks else None,
            "trial_wall_s": round(wall_s, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H-%M-%S"),
        }

    # -- d-MALIBOO objective interface ----------------------------------------

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        out = []
        for x in X:
            params = {name: float(x[i]) for i, name in enumerate(PARAM_NAMES)}
            key = cfg_key(params)
            cached = self._load_cached(key)
            if cached is not None:
                entry, src = cached, "cache"
            else:
                print(f"\n=== trial {self.trial_idx}: {key} "
                      f"(model={self.model_key}) ===", flush=True)
                try:
                    entry = self._run_trial(params, key)
                    self._cache_path(key).write_text(json.dumps(entry, indent=2))
                    src = "run"
                except Exception as exc:
                    # A failed/hung trial must not kill the whole optimization
                    # (on 2026-07-11 one 3h-hung gemma trial crashed the entire
                    # 3-model 100-trial queue). Kill any leftover suite, feed the
                    # GP a NEUTRAL value (median of real scores — neither rewards
                    # nor demonizes this config), and do NOT cache it, so a
                    # later resume re-evaluates this config for real.
                    subprocess.run(["docker", "exec", rmc.ORCH_CONTAINER,
                                    "pkill", "-f", "run_tests.py"], check=False)
                    real = [h["overall_pct"] for h in self.history
                            if h.get("source") != "failed"]
                    neutral = statistics.median(real) if real else 50.0
                    print(f"[trial-FAILED] {key}: {exc} → neutral {neutral}%, "
                          f"not cached (will re-run on resume)", flush=True)
                    entry = {
                        "params": {"temperature": params["temperature"],
                                   "top_p": params["top_p"],
                                   "top_k": int(round(params["top_k"]))},
                        "cfg_key": key, "run_id": f"tune-{self.model_key}-{key}",
                        "n_records": 0, "overall_pct": neutral,
                        "handoff_pct": None, "vision_pct": None,
                        "result_pct": None, "avg_latency_s": None,
                        "tokens_per_task": None, "trial_wall_s": None,
                        "failed": True, "error": str(exc)[:300],
                        "timestamp": time.strftime("%Y-%m-%dT%H-%M-%S"),
                    }
                    src = "failed"

            overall = entry["overall_pct"]
            # Only a COMPLETED full-suite, non-failed trial can define "best" —
            # never a failed trial's imputed neutral value nor a partial suite
            # (belt-and-suspenders; the _run_trial guard already blocks partials).
            is_full = entry.get("n_records") == self.expected_records
            if src != "failed" and is_full and (self.best is None or overall > self.best):
                self.best = overall
            self.history.append({"trial": self.trial_idx, "source": src, **entry})

            # All per-trial telemetry lives under tuning/* and charts against the
            # "trial" axis (define_metric in main). NEVER pass an explicit low
            # `step=` here: on resumed runs the internal step counter is already
            # past it (live/ logs auto-increment), and wandb silently DROPS
            # history rows with step < max — which is exactly how per-trial rows
            # vanished from the dashboard in the first runs.
            row = {
                "trial": self.trial_idx,
                "tuning/temperature": entry["params"]["temperature"],
                "tuning/top_p": entry["params"]["top_p"],
                "tuning/top_k": entry["params"]["top_k"],
                "tuning/overall_pct": overall,
                "tuning/handoff_pct": entry.get("handoff_pct"),
                "tuning/vision_pct": entry.get("vision_pct"),
                "tuning/result_pct": entry.get("result_pct"),
                "tuning/avg_latency_s": entry.get("avg_latency_s"),
                "tuning/tokens_per_task": entry.get("tokens_per_task"),
                "tuning/trial_wall_s": entry.get("trial_wall_s"),
                "tuning/best_overall_pct": self.best,
                "tuning/cached": int(src == "cache"),
                "tuning/failed": int(src == "failed"),
            }
            try:
                self.wb.log(row)
                # Run-page summary updates live (best config so far), and the
                # trials table is rebuilt each trial so it exists mid-run.
                real_hist = [h for h in self.history
                             if h.get("source") != "failed"
                             and h.get("n_records") == self.expected_records] or self.history
                best_entry = max(real_hist, key=lambda h: h["overall_pct"])
                self.wb.summary["best_overall_pct"] = self.best
                self.wb.summary["best_config"] = best_entry["cfg_key"]
                self.wb.summary["trials_done"] = self.trial_idx + 1
                cols = ["trial", "cfg", "temperature", "top_p", "top_k",
                        "overall_pct", "handoff_pct", "vision_pct", "result_pct",
                        "avg_latency_s", "cached", "failed"]
                data = [[h["trial"], h["cfg_key"], h["params"]["temperature"],
                         h["params"]["top_p"], h["params"]["top_k"],
                         h["overall_pct"], h.get("handoff_pct"),
                         h.get("vision_pct"), h.get("result_pct"),
                         h.get("avg_latency_s"),
                         int(h.get("source") == "cache"),
                         int(h.get("source") == "failed")] for h in self.history]
                self.wb.log({"tables/trials": wandb.Table(columns=cols, data=data)})
            except Exception as exc:
                print(f"[warn] wandb.log failed: {exc}")

            print(f"    {key} → overall {overall}% "
                  f"(handoff {entry.get('handoff_pct')}%, "
                  f"vision {entry.get('vision_pct')}%, "
                  f"result {entry.get('result_pct')}%) "
                  f"[{src}] best={self.best}%", flush=True)

            self.trial_idx += 1
            out.append(-overall / 100.0)   # d-MALIBOO minimizes
        return np.array(out)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Registry key or table label (see --help header).")
    p.add_argument("--n-initial", type=int, default=5,
                   help="Random initialization points (default 5).")
    p.add_argument("--n-iterations", type=int, default=10,
                   help="Bayesian-optimization iterations after init (default 10).")
    p.add_argument("--scenarios", default="",
                   help="Comma-separated scenario IDs (default: all 22).")
    p.add_argument("--levels", default="vague,clear,algorithmic",
                   help="Prompt levels (default: all three).")
    p.add_argument("--suite-timeout", type=int, default=10800,
                   help="Per-trial suite timeout, seconds (default 3 h).")
    p.add_argument("--wandb-project", default="vlm-orchestrator-tuning")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--no-ml-af", action="store_true",
                   help="Plain EI acquisition (skip d-MALIBOO's ML-on-target NN).")
    p.add_argument("--fresh", action="store_true",
                   help="Discard cached trials for this model and start over.")
    args = p.parse_args()

    model_key = resolve_model_key(args.model)
    model = rmc.REGISTRY[model_key]

    # Pre-flight (same guarantees as the comparison driver).
    if not rmc.LLAMA_BIN.is_file():
        raise SystemExit(f"llama-server binary not found: {rmc.LLAMA_BIN}")
    if not rmc.gguf_complete(model["path"]):
        raise SystemExit(f"GGUF missing/incomplete for {model_key}: {model['path']}")
    if not rmc._container_running():
        raise SystemExit("agents-orchestrator container is not running — "
                         "start the stack first (run_agentic_system.sh).")

    # wandb mode / login guard.
    mode = os.getenv("WANDB_MODE", "online")
    if mode == "online" and not os.getenv("WANDB_API_KEY"):
        netrc = Path.home() / ".netrc"
        if not (netrc.exists() and "api.wandb.ai" in netrc.read_text()):
            raise SystemExit(
                "wandb is not logged in. Run `wandb login` (or set WANDB_API_KEY, "
                "or use WANDB_MODE=offline) and retry.")

    cache_dir = TUNING_DIR / model_key
    if args.fresh and cache_dir.exists():
        print(f"--fresh: removing {cache_dir}")
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_scen = len([s for s in args.scenarios.split(",") if s.strip()]) or 22
    n_lvls = len([l for l in args.levels.split(",") if l.strip()]) or 3
    expected_records = n_scen * n_lvls

    # Stable wandb run id → resume="allow" continues the same run after a crash.
    id_file = cache_dir / "wandb_run_id.txt"
    if id_file.exists():
        run_id = id_file.read_text().strip()
    else:
        run_id = f"tune-{model_key}-{int(time.time())}"
        id_file.write_text(run_id)

    wb = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        id=run_id, resume="allow",
        name=f"tune-{model_key}", group=model_key, job_type="bayes-tuning",
        config={
            "model_key": model_key, "model_name": model["name"],
            "space": SPACE, "n_initial": args.n_initial,
            "n_iterations": args.n_iterations, "levels": args.levels,
            "scenarios": args.scenarios or "all", "seed": args.seed,
            "optimizer": "d-MALIBOO (GP + EI)",
            "baseline_defaults": {"temperature": 0.8, "top_p": 0.95, "top_k": 40},
        },
    )
    # Chart all tuning/* metrics against the trial number, not the internal
    # step counter — makes per-trial rows land correctly even on resumed runs.
    wb.define_metric("trial")
    wb.define_metric("tuning/*", step_metric="trial")

    objective = SuiteObjective(model_key, model, args, cache_dir, wb,
                               expected_records)
    n_cached = len(list(cache_dir.glob("t*.json")))
    print(f"Model: {model_key} ({model['name']})")
    print(f"Space: {PARAM_NAMES} — "
          f"{len(SPACE['temperature'])}x{len(SPACE['top_p'])}x{len(SPACE['top_k'])} "
          f"= 384 combos; budget {args.n_initial}+{args.n_iterations}")
    print(f"Cache: {cache_dir} ({n_cached} completed trials found)")
    print(f"Suite: scenarios={args.scenarios or 'all'} levels={args.levels} "
          f"({expected_records} runs/trial)")

    bounds_array = np.array([[min(SPACE[p_]), max(SPACE[p_])] for p_ in PARAM_NAMES])
    discrete_values = [np.array(SPACE[p_], dtype=float) for p_ in PARAM_NAMES]

    # d-MALIBOO wiring — mirrors Repetti_VLM-Testing/src/BayesianOptimizer.optimize()
    gp = GP(kernel_name="Matern", random_state=args.seed, length_scale=0.5)()
    if args.no_ml_af:
        af = AF(kind="ei", kappa=args.kappa, bounds=bounds_array,
                random_state=args.seed, ml_on_bounds=False, ml_on_target=False)
    else:
        af = AF(kind="ei", kappa=args.kappa, bounds=bounds_array,
                random_state=args.seed, ml_on_bounds=False,
                ml_on_target=True,
                ml_on_target_parameters={
                    "name": "nn", "task": "classification",
                    "hidden_layer_sizes": (100, 50), "max_iter": 500,
                })
    logger = Logger(dim=len(PARAM_NAMES))
    bo = BO(objective_function=objective,
            domain_bounds=bounds_array,
            gp=gp, af=af, constraint_functions=None,
            initial_points=args.n_initial,
            random_state=args.seed, logger=logger,
            discrete_values=discrete_values, discrete_refine=True)

    print(f"\nRandom initialization ({args.n_initial} points)")
    bo.initialize()
    print(f"\nBayesian optimization ({args.n_iterations} iterations)")
    x_best, y_best = bo.run(n_iterations=args.n_iterations, n_restarts=5,
                            verbose=True)

    # d-MALIBOO's argmax (x_best/y_best) can land on a PARTIAL trial: run_tests.py
    # may exit cleanly having scored only the easy early scenarios, giving the best
    # (lowest) objective value even though it's not a real full-suite result. The
    # _run_trial guard now prevents new partials, but derive the canonical winner
    # from COMPLETED (full-suite, non-failed) trials only — and fall back to BO's
    # argmax only if somehow none exist.
    full_hist = [h for h in objective.history
                 if h.get("source") != "failed"
                 and h.get("n_records") == objective.expected_records]
    if full_hist:
        best_h = max(full_hist, key=lambda h: h["overall_pct"])
        best_params = {"temperature": best_h["params"]["temperature"],
                       "top_p": best_h["params"]["top_p"],
                       "top_k": int(best_h["params"]["top_k"])}
        best_pct = best_h["overall_pct"]
    else:
        best_params = {name: float(x_best[i]) for i, name in enumerate(PARAM_NAMES)}
        best_params["top_k"] = int(round(best_params["top_k"]))
        best_pct = -y_best * 100.0

    # Persist history (Recovery.py-compatible CSV) + wandb summary/table.
    hist_df = pd.DataFrame(objective.history)
    hist_csv = cache_dir / "history.csv"
    hist_df.to_csv(hist_csv, index=False)
    try:
        wb.summary["best_overall_pct"] = round(best_pct, 1)
        wb.summary["best_temperature"] = best_params["temperature"]
        wb.summary["best_top_p"] = best_params["top_p"]
        wb.summary["best_top_k"] = best_params["top_k"]
    except Exception as exc:
        print(f"[warn] wandb summary failed: {exc}")
    wb.finish()

    print("\n" + "=" * 60)
    print(f"BEST CONFIG for {model_key}: "
          f"temp={best_params['temperature']:.2f} "
          f"top_p={best_params['top_p']:.2f} top_k={best_params['top_k']} "
          f"→ overall {best_pct:.1f}%")
    print(f"History: {hist_csv}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
