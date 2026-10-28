#!/usr/bin/env python3
"""
confirm_winners.py — repeat-run confirmation of the tuned sampling winners.

Re-runs the full 66-run suite N times per (model, config) to measure run-to-run
noise around the tuning campaign's winners (docs/HYPERPARAMETER_TUNING.md
"Final results"). a3b is confirmed on llama-server DEFAULTS (0.8/0.95/40),
since tuning found no config beating its default-run benchmark score.

Same execution path as tune_orchestrator._run_trial (llama-server restart with
sampling flags → suite in the agents container → score), including the
partial-suite guard: a run with fewer than the expected records is recorded as
"partial" and excluded from the confirmation statistics.

Usage: .venv-tuning/bin/python scripts/confirm_winners.py [--repeats 2]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"

_spec = importlib.util.spec_from_file_location(
    "rmc", PROJECT_ROOT / "scripts" / "run_model_comparison.py")
rmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmc)

LEVELS = "vague,clear,algorithmic"
EXPECTED = 66
SUITE_TIMEOUT = 10800

# (registry key, human label, sampling flags or None for llama-server defaults)
PLAN = [
    ("gemma", "t0.40_p0.65_k10",
     ["--temp", "0.40", "--top-p", "0.65", "--top-k", "10"]),
    ("gpt-oss-120b", "t0.30_p0.98_k20",
     ["--temp", "0.30", "--top-p", "0.98", "--top-k", "20"]),
    ("qwen36-a3b", "defaults_t0.80_p0.95_k40", None),
]


def wait_healthy(timeout_s: int = 900) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if rmc._llama_healthy():
            return True
        time.sleep(3)
    return False


def run_once(model_key: str, cfg_label: str, flags, rep: int) -> dict:
    model = rmc.REGISTRY[model_key]
    reasoning = "low" if model.get("gpt_oss") else None
    rmc.stop_llama_server()
    rmc.start_llama_server(model, reasoning, extra_server_args=flags)
    if not wait_healthy():
        raise RuntimeError("llama-server did not become healthy")

    run_id = f"confirm-{model_key}-{cfg_label}-r{rep}"
    slug = rmc.slugify(model["name"])
    result_file = RESULTS_DIR / f"{run_id}_{slug}.json"
    if result_file.exists():
        result_file.unlink()

    t0 = time.monotonic()
    try:
        rmc.run_suite_in_container(model, run_id, "", LEVELS, SUITE_TIMEOUT,
                                   extra_env={"VQA_PREWARM_TIMEOUT_S": "1800"})
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "exec", rmc.ORCH_CONTAINER,
                        "pkill", "-f", "run_tests.py"], check=False)
        raise RuntimeError(f"suite timed out for {run_id}")
    wall_s = time.monotonic() - t0

    records = json.loads(result_file.read_text())
    n = len(records)
    entry = {
        "model": model_key, "config": cfg_label, "repeat": rep,
        "n_records": n,
        "overall_pct": round(
            100.0 * sum(bool(r.get("overall_pass")) for r in records)
            / max(n, 1), 1),
        "wall_s": round(wall_s, 1),
        "partial": n < EXPECTED,
        "timestamp": time.strftime("%Y-%m-%dT%H-%M-%S"),
    }
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    out_path = RESULTS_DIR / "tuning" / "confirmations.json"
    results = (json.loads(out_path.read_text())
               if out_path.exists() else [])
    done = {(r["model"], r["config"], r["repeat"])
            for r in results if not r.get("partial")}

    for model_key, cfg_label, flags in PLAN:
        for rep in range(1, args.repeats + 1):
            if (model_key, cfg_label, rep) in done:
                print(f"[skip] {model_key} {cfg_label} r{rep} (already done)",
                      flush=True)
                continue
            print(f"\n=== confirm {model_key} {cfg_label} r{rep} ===",
                  flush=True)
            try:
                entry = run_once(model_key, cfg_label, flags, rep)
            except Exception as exc:
                # Kill any zombie suite so the next run can't be contaminated,
                # record the failure, keep going.
                subprocess.run(["docker", "exec", rmc.ORCH_CONTAINER,
                                "pkill", "-f", "run_tests.py"], check=False)
                print(f"[confirm-FAILED] {model_key} {cfg_label} r{rep}: {exc}",
                      flush=True)
                entry = {"model": model_key, "config": cfg_label,
                         "repeat": rep, "n_records": 0, "overall_pct": None,
                         "partial": True, "error": str(exc)[:300],
                         "timestamp": time.strftime("%Y-%m-%dT%H-%M-%S")}
            results.append(entry)
            out_path.write_text(json.dumps(results, indent=2))
            tag = " (PARTIAL — excluded)" if entry["partial"] else ""
            print(f"    → overall {entry['overall_pct']}% "
                  f"({entry['n_records']}/{EXPECTED}){tag}", flush=True)

    rmc.stop_llama_server()
    print("\n" + "=" * 60)
    print("CONFIRMATION SUMMARY (full-suite runs only)")
    for model_key, cfg_label, _ in PLAN:
        vals = [r["overall_pct"] for r in results
                if r["model"] == model_key and r["config"] == cfg_label
                and not r.get("partial")]
        print(f"  {model_key:14s} {cfg_label:26s} runs={vals}")
    print("CONFIRM COMPLETE")


if __name__ == "__main__":
    main()
