#!/usr/bin/env python3
"""
Authoritative tuning-winner report.

Derives the best sampling config per model from the per-config cache in
tests/results/tuning/<model>/*.json, counting ONLY completed full-suite trials
(n_records == EXPECTED). This is the source of truth for the tuned winners —
it is independent of the tuner process's own stdout/wandb "BEST CONFIG", which
d-MALIBOO's argmax can pin to a partial trial (see FIXES_TRACKING.md F15).

Usage: python scripts/tuning_report.py [--expected 66] [--top 5]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

MODELS = ["gemma", "gpt-oss-120b", "qwen36-a3b"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", type=int, default=66,
                    help="records for a complete suite (scenarios × levels)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--root", default="tests/results/tuning")
    args = ap.parse_args()

    for model in MODELS:
        files = glob.glob(os.path.join(args.root, model, "t*.json"))
        full, partial = [], []
        for f in files:
            d = json.load(open(f))
            row = (d.get("overall_pct"), d.get("cfg_key"), d.get("n_records"),
                   d.get("handoff_pct"), d.get("vision_pct"), d.get("result_pct"))
            (full if d.get("n_records") == args.expected else partial).append(row)
        full.sort(key=lambda r: (r[0] is not None, r[0]), reverse=True)

        print(f"\n{'=' * 66}\n{model}: {len(full)} complete / {len(files)} total"
              f"{f'  ⚠ {len(partial)} partial (excluded)' if partial else ''}")
        if not full:
            print("  (no completed full-suite trials yet)")
            continue
        o, k, n, h, v, r = full[0]
        print(f"  WINNER  {o}%  {k}   handoff={h} vision={v} result={r}")
        for o, k, n, h, v, r in full[1:args.top]:
            print(f"    ...   {o}%  {k}")
        if partial:
            for o, k, n, *_ in partial:
                print(f"    excluded partial: {o}% {k} (n_records={n})")


if __name__ == "__main__":
    main()
