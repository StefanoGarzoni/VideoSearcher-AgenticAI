"""
visualise_results.py — render a test-run JSON into a human-readable Markdown file.

Usage:
    python3 scripts/visualise_results.py <path-to-result.json>

Output:
    tests/results/markdowns/<input_stem>.md
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import groupby
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

MARKDOWNS_DIR = (
    Path(__file__).parent.parent / "tests" / "results" / "markdowns"
)
RESULTS_DIR = Path(__file__).parent.parent / "tests" / "results"

_LEVEL_ORDER = {"vague": 0, "clear": 1, "algorithmic": 2}

# Metric direction for normalisation: True = higher is better, False = lower is better.
_METRIC_DIR = {
    "overall_pct": True, "handoff_pct": True, "vision_pct": True, "result_pct": True,
    "error_handling_pct": True, "pass_vague_pct": True, "pass_clear_pct": True,
    "pass_algo_pct": True, "throughput_tps": True, "passes_per_kwh": True,
    "passes_per_min": True,
    "extra_calls_per_task": False, "avg_latency_s": False, "ttft_ms": False,
    "tokens_per_task": False, "energy_kwh": False,
}


# ── Pass logic ───────────────────────────────────────────────────────────────

def overall_pass(r: dict) -> bool:
    # Support both old format (no result_pass) and new format
    if not r.get("handoff_match", False):
        return False
    if r.get("vision_pass") is not None and not r["vision_pass"]:
        return False
    if r.get("result_pass") is not None and not r["result_pass"]:
        return False
    return True


def cell_label(r: dict) -> str:
    if not r["success"]:
        return "✗ error"
    if not r.get("handoff_match", False):
        return "✗ handoff"
    if r.get("vision_pass") is False:
        return "✗ vision"
    if r.get("result_pass") is False:
        score = r.get("result_score")
        score_str = f"={score:.2f}" if score is not None else ""
        return f"✗ result{score_str}"
    # All checks passed
    badges = []
    if r.get("vision_pass") is True:
        badges.append("👁")
    if r.get("result_pass") is True:
        score = r.get("result_score")
        score_str = f"{score:.2f}" if score is not None else ""
        badges.append(f"📊{score_str}")
    return "✓ " + " ".join(badges) if badges else "✓"


# ── Markdown helpers ─────────────────────────────────────────────────────────

def _blockquote(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(f"> {line}" if line.strip() else ">" for line in lines)


def _render_conversation(conversation: list[dict]) -> str:
    parts: list[str] = [f"**Conversation trace** ({len(conversation)} steps)\n"]
    for step in conversation:
        seq = step.get("sequence", "?")
        itype = step.get("item_type", "")
        content = step.get("content") or ""
        tool_name = step.get("tool_name") or ""
        arguments = step.get("arguments") or {}
        agent_role = step.get("agent_role") or ""

        if itype == "message":
            role_str = f" *({agent_role})*" if agent_role else ""
            parts.append(f"**[{seq}] message**{role_str}")
            if content:
                parts.append(_blockquote(content))

        elif itype == "tool_call":
            parts.append(f"**[{seq}] tool_call**: `{tool_name}`")
            if arguments:
                parts.append("```json\n" + json.dumps(arguments, indent=2) + "\n```")
            else:
                parts.append("*(no arguments)*")

        elif itype == "tool_output":
            parts.append(f"**[{seq}] tool_output**:")
            if content:
                parts.append(_blockquote(content))

        else:
            parts.append(f"**[{seq}] {itype}**")
            if content:
                parts.append(_blockquote(content))

        parts.append("")  # blank line between steps

    return "\n".join(parts)


def _render_record(r: dict) -> str:
    level = r["prompt_level"]
    elapsed = r["execution_time_ms"] / 1000
    status = "✓ PASS" if overall_pass(r) else "✗ FAIL"

    ttft = r.get("ttft_ms")
    ttft_str = f" · TTFT {ttft:.0f} ms" if ttft is not None else ""

    lines: list[str] = []
    lines.append(f"### `{level}` — {status}  _({elapsed:.1f} s{ttft_str})_\n")

    # Expected vs actual
    expected = " → ".join(r["expected_agents"]) if r["expected_agents"] else "(none)"
    actual = " → ".join(r["handoff_sequence"]) if r["handoff_sequence"] else "(none)"
    lines.append(f"**Expected agents**: `{expected}`")
    lines.append(f"**Actual handoff sequence**: `{actual}`")

    hm = "Yes" if r.get("handoff_match") else "No"
    extra_checks = []
    if r.get("vision_pass") is not None:
        extra_checks.append(f"Vision: {'Pass' if r['vision_pass'] else 'Fail'}")
    if r.get("result_pass") is not None:
        score = r.get("result_score")
        score_str = f" ({score:.3f})" if score is not None else ""
        extra_checks.append(f"Result: {'Pass' if r['result_pass'] else 'Fail'}{score_str}")
    if extra_checks:
        lines.append(f"**Handoff match**: {hm} | {' | '.join(extra_checks)}\n")
    else:
        lines.append(f"**Handoff match**: {hm}\n")

    # First divergence (pipeline diagnosis)
    fd = r.get("first_divergence")
    if fd:
        lines.append(f"**First divergence**: `{fd}` (first expected agent not matched)\n")

    # Result detail (metric breakdown)
    rd = r.get("result_detail")
    if rd:
        lines.append(f"**Result detail**: {rd}\n")

    # Error
    if r.get("error"):
        lines.append("**Error**:")
        lines.append("```")
        lines.append(r["error"])
        lines.append("```\n")

    # Orchestrator output
    output = (r.get("orchestrator_output") or "").strip()
    if output:
        lines.append("**Orchestrator output**:")
        lines.append(_blockquote(output))
        lines.append("")

    # Conversation
    conversation = r.get("conversation") or []
    if conversation:
        lines.append(_render_conversation(conversation))

    return "\n".join(lines)


# ── Section builders ─────────────────────────────────────────────────────────

def build_header(records: list[dict]) -> str:
    r0 = records[0]
    run_id = r0["run_id"]
    model = r0["model_name"]
    scenarios = len(set(r["scenario_id"] for r in records))
    levels = " / ".join(sorted(_LEVEL_ORDER, key=_LEVEL_ORDER.get))
    total = len(records)

    return (
        f"# Test Run: {run_id}\n\n"
        f"**Model**: {model}  \n"
        f"**Scenarios tested**: {scenarios} | "
        f"**Prompt levels**: {levels} | "
        f"**Total records**: {total}\n"
    )


def build_summary(records: list[dict]) -> str:
    total = len(records)
    n_pass = sum(1 for r in records if overall_pass(r))
    n_hm = sum(1 for r in records if r.get("handoff_match"))
    vision_records = [r for r in records if r.get("vision_pass") is not None]
    n_vp = sum(1 for r in vision_records if r["vision_pass"])
    result_records = [r for r in records if r.get("result_pass") is not None]
    n_rp = sum(1 for r in result_records if r["result_pass"])
    n_success = sum(1 for r in records if r["success"])

    times = [(r["execution_time_ms"] / 1000, r["scenario_id"], r["prompt_level"]) for r in records]
    avg_s = sum(t for t, _, _ in times) / len(times)
    slowest = max(times)
    fastest = min(times)

    ttfts = [r["ttft_ms"] for r in records if r.get("ttft_ms") is not None]
    ttft_line = f"| TTFT (model, median) | {ttfts[0]:.0f} ms |" if ttfts else None

    # Per-scenario table
    by_scenario: dict[str, dict[str, dict]] = {}
    for r in records:
        by_scenario.setdefault(r["scenario_id"], {})[r["prompt_level"]] = r

    table_rows: list[str] = []
    for sc_id in sorted(by_scenario):
        levels_map = by_scenario[sc_id]
        name = next(iter(levels_map.values()))["scenario_name"]
        cells = []
        for level in ("vague", "clear", "algorithmic"):
            if level in levels_map:
                cells.append(cell_label(levels_map[level]))
            else:
                cells.append("—")
        table_rows.append(f"| {sc_id} {name} | {' | '.join(cells)} |")

    table_header = (
        "| Scenario | Vague | Clear | Algorithmic |\n"
        "|----------|-------|-------|-------------|"
    )

    pct = lambda n, d: f"{n} / {d} ({100 * n // d}%)"

    lines = [
        "## Summary\n",
        "### Pass rates\n",
        "| Metric | Result |",
        "|--------|--------|",
        f"| Overall pass (handoff + vision + result) | {pct(n_pass, total)} |",
        f"| Handoff match | {pct(n_hm, total)} |",
        f"| Vision pass | {n_vp} / {len(vision_records)} evaluated |",
        f"| Result pass (correctness) | {n_rp} / {len(result_records)} evaluated |",
        f"| Completed without error | {pct(n_success, total)} |",
        "",
        "### Execution time\n",
        "| | Time |",
        "|-|------|",
        f"| Average | {avg_s:.1f} s |",
        f"| Slowest | {slowest[1]} {slowest[2]} — {slowest[0]:.1f} s |",
        f"| Fastest | {fastest[1]} {fastest[2]} — {fastest[0]:.1f} s |",
        *( [ttft_line] if ttft_line else [] ),
        "",
        "### Results by scenario\n",
        table_header,
        *table_rows,
    ]
    return "\n".join(lines)


def build_detail(records: list[dict]) -> str:
    sorted_records = sorted(
        records,
        key=lambda r: (r["scenario_id"], _LEVEL_ORDER.get(r["prompt_level"], 99)),
    )

    sections: list[str] = ["## Detailed Results"]

    for sc_id, group in groupby(sorted_records, key=lambda r: r["scenario_id"]):
        group_list = list(group)
        sc_name = group_list[0]["scenario_name"]

        sections.append(f"\n---\n\n## {sc_id} — {sc_name}\n")
        for r in group_list:
            sections.append(_render_record(r))
            sections.append("")

    return "\n".join(sections)


# ── Entry point ──────────────────────────────────────────────────────────────

def build_charts(run_id: str | None = None, models: list[str] | None = None) -> int:
    """Generate the 3 presentation figures from tests/results/all_metrics.csv:
    a model×metric heatmap (seaborn), a radar chart (plotly), and an
    accuracy-vs-energy Pareto scatter (matplotlib). Output → tests/results/.

    Default: chart one run (run_id, or auto-picked). MERGE mode (`models` given):
    chart a curated set of models, each taken from its latest FULL run — used to
    combine reused models (run 09-49-27) with newly-run ones into one figure."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go

    csv = RESULTS_DIR / "all_metrics.csv"
    if not csv.exists():
        print("ERROR: tests/results/all_metrics.csv not found — run "
              "`python3 scripts/consolidate_metrics.py` first.", file=sys.stderr)
        return 1
    df = pd.read_csv(csv)

    full = df[df["n_total"] == 66]
    if models:
        # MERGE mode: pick each requested model's latest FULL run, across runs.
        rows = []
        for name in models:
            cand = full[full["model"].str.contains(name, case=False, regex=False)]
            if len(cand):
                rows.append(cand.sort_values("run_id").iloc[-1])
        if not rows:
            print("ERROR: no matching models for merge", file=sys.stderr)
            return 1
        sub = pd.DataFrame(rows).drop_duplicates("model").set_index("model")
        tag = "merged"
        print(f"Merged charts ({len(sub)} models): {list(sub.index)}")
    else:
        if run_id is None:
            # Auto-pick the full run with the most models scoring >0 overall
            # (skips broken all-zero runs, e.g. tool-call parsing failures).
            valid = full[full["overall_pct"] > 0]
            run_id = (valid.groupby("run_id").size().idxmax()
                      if len(valid) else (full["run_id"].max() if len(full) else df["run_id"].max()))
        sub = df[df["run_id"] == run_id].drop_duplicates("model").set_index("model")
        tag = run_id
        if sub.empty:
            print(f"ERROR: no rows for run {run_id}", file=sys.stderr)
            return 1
        print(f"Charts for run {run_id} ({len(sub)} models)")

    metrics = [m for m in _METRIC_DIR if m in sub.columns and sub[m].notna().any()]

    # Normalise each metric to 0–1 and flip lower-is-better so 1 = best everywhere.
    norm = pd.DataFrame(index=sub.index)
    for m in metrics:
        col = sub[m].astype(float)
        lo, hi = col.min(), col.max()
        n = (col - lo) / (hi - lo) if hi > lo else col * 0 + 0.5
        norm[m] = n if _METRIC_DIR[m] else 1 - n

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    short = [i.replace(".gguf", "") for i in sub.index]

    # 1) Heatmap — color = normalised goodness, annotation = raw value.
    plt.figure(figsize=(max(10, len(metrics) * 0.95), max(4, len(sub) * 0.7)))
    sns.heatmap(norm[metrics].set_axis(short), annot=sub[metrics].set_axis(short).round(2),
                fmt="g", cmap="RdYlGn", vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={"label": "normalised (1 = best)"})
    plt.title(f"Model × Metric matrix — {tag}\n(color = goodness, cells = raw values)")
    plt.xlabel("metric"); plt.ylabel("model")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    heatmap_path = RESULTS_DIR / f"charts_{tag}_heatmap.png"
    plt.savefig(heatmap_path, dpi=150); plt.close()

    # 2) Radar — key metrics, normalised (outer = better).
    radar_metrics = [m for m in ["overall_pct", "vision_pct", "result_pct",
                                 "error_handling_pct", "throughput_tps", "passes_per_kwh",
                                 "avg_latency_s", "tokens_per_task"] if m in norm.columns]
    fig = go.Figure()
    for model, label in zip(sub.index, short):
        vals = [float(norm.loc[model, m]) for m in radar_metrics]
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]],
                                      theta=radar_metrics + [radar_metrics[0]],
                                      fill="toself", name=label))
    fig.update_layout(title=f"Model comparison radar — {tag} (outer = better)",
                      polar=dict(radialaxis=dict(range=[0, 1])))
    radar_path = RESULTS_DIR / f"charts_{tag}_radar.html"
    fig.write_html(str(radar_path))

    # 3) Pareto — accuracy vs energy (best = top-left: high accuracy, low energy).
    pareto_path = None
    if sub["energy_kwh"].notna().any():
        x = sub["energy_kwh"].astype(float); y = sub["overall_pct"].astype(float)
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, s=140, c="steelblue", edgecolors="k")
        for model, label in zip(sub.index, short):
            plt.annotate(label, (x[model], y[model]), fontsize=8,
                         xytext=(6, 4), textcoords="offset points")
        plt.xlabel("Energy per run (kWh)  ← lower is better")
        plt.ylabel("Overall pass %  ↑ higher is better")
        plt.title(f"Accuracy vs Energy (Pareto) — {tag}\nbest = top-left")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        pareto_path = RESULTS_DIR / f"charts_{tag}_pareto.png"
        plt.savefig(pareto_path, dpi=150); plt.close()

    print(f"  heatmap → {heatmap_path}")
    print(f"  radar   → {radar_path}")
    print(f"  pareto  → {pareto_path}" if pareto_path else "  pareto  → skipped (no energy data)")
    return 0


# Final comparison roster (label, model-name substring) + the 14 metrics, used by
# --table. Order = descending overall; phi4-mini* kept as the 0% baseline.
_TABLE_ROSTER = [
    ("gemma-12b", "gemma-4-12b"), ("gemma-E2B", "gemma-4-E2B"),
    ("gpt-oss-120b", "gpt-oss-120b"), ("qwen2.5-7B", "Qwen2"),
    ("qwen3.6-27B", "Qwen3.6-27B"), ("qwen3.6-35B-A3B", "Qwen3.6-35B-A3B"),
    ("mistral-24B", "Mistral-Small"), ("falcon3-10B", "Falcon3-10B"),
    ("phi4-mini*", "Phi-4-mini"),
]
_TABLE_METRICS = [
    ("1 · Overall %", "overall_pct"), ("2 · Handoff %", "handoff_pct"),
    ("3 · Vision %", "vision_pct"), ("4 · Result %", "result_pct"),
    ("5 · Tool-call eff. (extra/task)", "extra_calls_per_task"),
    ("6 · Error-handling %", "error_handling_pct"),
    ("7a · Robust vague %", "pass_vague_pct"), ("7b · Robust clear %", "pass_clear_pct"),
    ("7c · Robust algo %", "pass_algo_pct"), ("8 · Avg latency (s)", "avg_latency_s"),
    ("9 · TTFT (ms)", "ttft_ms"), ("10 · Throughput (tok/s)", "throughput_tps"),
    ("11 · Tokens/task", "tokens_per_task"), ("12 · Energy/run (kWh)", "energy_kwh"),
    ("13 · Passes/kWh", "passes_per_kwh"), ("14 · Passes/min", "passes_per_min"),
]


def build_table() -> int:
    """Render the final cross-model comparison as a transposed Markdown table
    (metrics = rows, models = columns), each model taken from its latest full
    66-scenario run. Reads all_metrics.csv → markdowns/final_metrics_table.md."""
    import csv as _csv
    csvp = RESULTS_DIR / "all_metrics.csv"
    if not csvp.exists():
        print("ERROR: all_metrics.csv not found — run consolidate_metrics.py first.", file=sys.stderr)
        return 1
    rows = [r for r in _csv.DictReader(csvp.open()) if r.get("n_total") == "66"]
    pick = {}
    for lbl, sub in _TABLE_ROSTER:
        cand = [r for r in rows if sub.lower() in r["model"].lower()]
        if cand:
            pick[lbl] = sorted(cand, key=lambda r: r["run_id"])[-1]
    labels = [l for l, _ in _TABLE_ROSTER if l in pick]
    if not labels:
        print("ERROR: no roster models found in all_metrics.csv", file=sys.stderr)
        return 1

    def cell(v):
        return v if v not in (None, "", "None") else "–"

    out = ["| Metric | " + " | ".join(labels) + " |", "|" + "---|" * (len(labels) + 1)]
    out.append("| _Country_ | " + " | ".join(cell(pick[l]["country"]) for l in labels) + " |")
    out.append("| _Size · Arch_ | " + " | ".join(
        f"{cell(pick[l]['size_b'])} {'MoE' if pick[l]['arch'] == 'MoE' else 'd'}" for l in labels) + " |")
    for mlabel, col in _TABLE_METRICS:
        out.append(f"| {mlabel} | " + " | ".join(cell(pick[l].get(col)) for l in labels) + " |")
    md = "\n".join(out)

    hdr = ("# Final model comparison — all 14 metrics\n\n"
           "VQA held constant (`Phi-4-reasoning-vision`). Each model from its latest full "
           "66-scenario run. `phi4-mini*` = broken-parsing baseline (fix in registry, but too "
           "weak to complete chains). `–` = metric not recorded for that run.\n\n")
    MARKDOWNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MARKDOWNS_DIR / "final_metrics_table.md"
    out_path.write_text(hdr + md + "\n")
    print(md)
    print(f"\nsaved → {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a test-run JSON into Markdown, or build comparison charts."
    )
    parser.add_argument("json_path", nargs="?", help="Path to a result JSON file (Markdown mode).")
    parser.add_argument("--charts", action="store_true",
                        help="Build heatmap/radar/Pareto figures from all_metrics.csv into tests/results/.")
    parser.add_argument("--run", default=None,
                        help="run_id to chart (default: auto-pick the best full run).")
    parser.add_argument("--merge", nargs="?", const="DEFAULT", default=None,
                        help="Build ONE figure set merging a curated model set across runs "
                             "(each from its latest full run). Optional comma-list of model "
                             "name substrings; defaults to the final viable roster.")
    parser.add_argument("--table", action="store_true",
                        help="Render the final 14-metric comparison table (Markdown) to "
                             "markdowns/final_metrics_table.md.")
    args = parser.parse_args()

    if args.table:
        return build_table()

    if args.merge is not None:
        roster = ["gemma-4-E2B", "gemma-4-12b", "gpt-oss-120b", "Qwen2", "Mistral-Small", "Falcon3-10B"]
        models = roster if args.merge == "DEFAULT" else [m.strip() for m in args.merge.split(",")]
        return build_charts(models=models)

    if args.charts:
        return build_charts(args.run)

    if not args.json_path:
        parser.error("provide a JSON path, or use --charts")

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"ERROR: file not found: {json_path}", file=sys.stderr)
        return 1

    try:
        with open(json_path) as f:
            records: list[dict] = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(records, list) or not records:
        print("ERROR: JSON must be a non-empty array of test records.", file=sys.stderr)
        return 1

    MARKDOWNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MARKDOWNS_DIR / (json_path.stem + ".md")

    parts = [
        build_header(records),
        "",
        build_summary(records),
        "",
        build_detail(records),
    ]
    out_path.write_text("\n".join(parts), encoding="utf-8")

    print(f"Written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
