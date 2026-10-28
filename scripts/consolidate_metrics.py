"""
consolidate_metrics.py — gather every model-comparison run's metrics into ONE
tidy CSV, ready for plotting (one row per run × model, all 14 metrics numeric).

Sources:
  - tests/results/<run_id>_<model>.json  → correctness, latency, TTFT, tokens,
    efficiency, robustness-by-level (computed numerically from the records)
  - tests/results/markdowns/comparison_<run_id>.md → Throughput + Energy/run
    (these are probe/attributed values, not stored in the per-record JSON)

Output: tests/results/all_metrics.csv   (re-run anytime; idempotent)
Usage:  python3 scripts/consolidate_metrics.py
"""
from __future__ import annotations
import csv, glob, json, os, re, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "tests" / "results"
MD = RESULTS / "markdowns"
OUT = RESULTS / "all_metrics.csv"

# model file-name → (company, country, size_b, arch)
META = {
    "gemma-4-E2B-it-Q8_0":            ("Google",    "USA",    5,   "dense"),
    "gemma-4-12b-it-Q8_0":            ("Google",    "USA",    12,  "dense"),
    "Phi-4-mini-instruct-Q4_K_M":     ("Microsoft", "USA",    3.8, "dense"),
    "Qwen2.5-7B-Instruct-Q8_0":       ("Alibaba",   "China",  7,   "dense"),
    "Qwen2_5-7B-Instruct-Q8_0":       ("Alibaba",   "China",  7,   "dense"),
    "gpt-oss-120b-mxfp4":             ("OpenAI",    "USA",    120, "MoE"),
    "Meta-Llama-3.1-8B-Instruct-Q8_0":("Meta",      "USA",    8,   "dense"),
    "Meta-Llama-3_1-8B-Instruct-Q8_0":("Meta",      "USA",    8,   "dense"),
    "DeepSeek-V2-Lite-Chat-Q4_K_M":   ("DeepSeek",  "China",  16,  "MoE"),
    "Mistral-Small-24B-Instruct-2501-Q4_K_M": ("Mistral", "France", 24, "dense"),
    "c4ai-command-r-08-2024-Q4_K_M":  ("Cohere",    "Canada", 32,  "dense"),
    "Falcon3-10B-Instruct-q5_k_m":    ("TII",       "UAE",    10,  "dense"),
    "c4ai-command-r7b-12-2024-Q6_K":  ("Cohere",    "Canada", 7,   "dense"),
    # Qwen3.6 (https://huggingface.co/collections/Qwen/qwen36). Dot/underscore variants
    # cover both the raw filename and the slugified label. Adjust the quant suffix to
    # match the GGUF you actually run.
    "Qwen3.6-27B-Q4_K_M":             ("Alibaba",   "China",  27,  "dense"),
    "Qwen3_6-27B-Q4_K_M":             ("Alibaba",   "China",  27,  "dense"),
    "Qwen3.6-35B-A3B-Q4_K_M":         ("Alibaba",   "China",  35,  "MoE"),
    "Qwen3_6-35B-A3B-Q4_K_M":         ("Alibaba",   "China",  35,  "MoE"),
}


def overall_pass(r: dict) -> bool:
    if not r.get("handoff_match"):
        return False
    if r.get("vision_pass") is not None and not r["vision_pass"]:
        return False
    if r.get("result_pass") is not None and not r["result_pass"]:
        return False
    return True


def is_error_scenario(r: dict) -> bool:
    exp = " ".join(str(a) for a in r.get("expected_agents", [])).lower()
    return "reject" in exp or "clarification" in exp


def pct(n, d):
    return round(100.0 * n / d, 1) if d else None


def parse_md_energy_throughput(run_id: str) -> dict:
    """Return {model_stem: (throughput_tps, energy_kwh)} from ONLY the '## Headline'
    table of comparison_<run_id>.md (the md has several tables; the others share the
    same model rows, so we must not read them). Both None when 'n/a'."""
    f = MD / f"comparison_{run_id}.md"
    out = {}
    if not f.exists():
        return out

    def num(x):
        try: return float(x)
        except ValueError: return None

    in_headline = False
    for line in f.read_text().splitlines():
        if line.startswith("## "):
            in_headline = line.strip() == "## Headline"
            continue
        if not in_headline or not line.startswith("| ") or ".gguf" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 7:   # Model|Overall|Avg latency|TTFT|Throughput|Energy|Tokens
            continue
        out[cells[0].replace(".gguf", "")] = (num(cells[4]), num(cells[5]))
    return out


def metrics_for(records: list, tput, energy) -> dict:
    n = len(records)
    n_overall = sum(1 for r in records if overall_pass(r))
    matched = [r for r in records if r.get("handoff_match")]
    extras = [max(0, len(r.get("handoff_sequence", [])) - len(r.get("expected_agents", [])))
              for r in matched]
    vis = [r for r in records if r.get("vision_pass") is not None]
    res = [r for r in records if r.get("result_pass") is not None]
    err = [r for r in records if is_error_scenario(r)]
    toks = [r["total_tokens"] for r in records if r.get("total_tokens") is not None]
    ttfts = [r["ttft_ms"] for r in records if r.get("ttft_ms") is not None]
    lvl = lambda L: pct(sum(1 for r in records if r.get("prompt_level") == L and overall_pass(r)),
                        sum(1 for r in records if r.get("prompt_level") == L))
    total_min = sum(r["execution_time_ms"] for r in records) / 60000.0
    return {
        "n_total": n,
        "n_overall": n_overall,
        "overall_pct": pct(n_overall, n),
        "handoff_pct": pct(sum(1 for r in records if r.get("handoff_match")), n),
        "vision_pct": pct(sum(1 for r in vis if r["vision_pass"]), len(vis)),
        "result_pct": pct(sum(1 for r in res if r["result_pass"]), len(res)),
        "extra_calls_per_task": round(statistics.mean(extras), 3) if extras else None,
        "error_handling_pct": pct(sum(1 for r in err if overall_pass(r)), len(err)),
        "pass_vague_pct": lvl("vague"),
        "pass_clear_pct": lvl("clear"),
        "pass_algo_pct": lvl("algorithmic"),
        "avg_latency_s": round(statistics.mean(r["execution_time_ms"] / 1000 for r in records), 1),
        "ttft_ms": round(statistics.median(ttfts), 1) if ttfts else None,
        "throughput_tps": tput,
        "tokens_per_task": round(statistics.mean(toks)) if toks else None,
        "energy_kwh": energy,
        "passes_per_kwh": round(n_overall / energy, 1) if energy else None,
        "passes_per_min": round(n_overall / total_min, 2) if total_min else None,
    }


def main():
    cols = ["run_id", "model", "company", "country", "size_b", "arch",
            "n_total", "n_overall", "overall_pct", "handoff_pct", "vision_pct",
            "result_pct", "extra_calls_per_task", "error_handling_pct",
            "pass_vague_pct", "pass_clear_pct", "pass_algo_pct",
            "avg_latency_s", "ttft_ms", "throughput_tps", "tokens_per_task",
            "energy_kwh", "passes_per_kwh", "passes_per_min"]
    rows = []
    for f in sorted(glob.glob(str(RESULTS / "*.json"))):
        base = os.path.basename(f)
        if base.endswith("_ttft.json") or base == "all_metrics.csv":
            continue
        m = re.match(r"(\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d)_(.+)_gguf\.json$", base)
        if not m:
            continue
        run_id, model_stem = m.group(1), m.group(2)
        try:
            records = json.loads(Path(f).read_text())
        except Exception:
            continue
        if not records:
            continue
        et = parse_md_energy_throughput(run_id).get(model_stem.replace("_gguf", ""), (None, None))
        # match by the model_name field too (more reliable than the slug)
        name = records[0].get("model_name", model_stem)
        et = parse_md_energy_throughput(run_id).get(name.replace(".gguf", ""), et)
        meta = META.get(model_stem) or META.get(name.replace(".gguf", "")) or ("?", "?", None, "?")
        row = {"run_id": run_id, "model": name,
               "company": meta[0], "country": meta[1], "size_b": meta[2], "arch": meta[3]}
        row.update(metrics_for(records, et[0], et[1]))
        rows.append(row)

    rows.sort(key=lambda r: (r["run_id"], -(r["n_overall"] or 0)))
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {OUT}")
    # quick console preview of the latest run
    if rows:
        latest = max(r["run_id"] for r in rows)
        print(f"\nLatest run {latest}:")
        for r in rows:
            if r["run_id"] == latest:
                print(f"  {r['model']:34s} overall={r['overall_pct']}%  ttft={r['ttft_ms']}  "
                      f"tput={r['throughput_tps']}  tokens={r['tokens_per_task']}  energy={r['energy_kwh']}")


if __name__ == "__main__":
    main()
