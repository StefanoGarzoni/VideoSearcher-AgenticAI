#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TUNING = ROOT / "tests" / "results" / "tuning"
OUT = ROOT / "presentation_plots"
OUT.mkdir(exist_ok=True)

# ── palette (reference instance, light mode) ─────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
NEUTRAL = "#f0efec"          # diverging midpoint gray — used for "not sampled"
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP_BLUE = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

# fixed entity colors (categorical slots 1..3; magenta carries direct labels — relief rule)
MODELS = ["gemma", "gpt-oss-120b", "qwen36-a3b"]
MCOLOR = {"gemma": "#2a78d6", "gpt-oss-120b": "#008300", "qwen36-a3b": "#e87ba4"}
MLABEL = {"gemma": "gemma-E2B", "gpt-oss-120b": "gpt-oss-120b", "qwen36-a3b": "qwen3.6-A3B"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"], "text.color": INK,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.frameon": False, "legend.fontsize": 11,
})


def savefig(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote presentation_plots/{name}")


# ── data loading ─────────────────────────────────────────────────────────────
def load_history(model):
    """Final BO run trajectory, full-66 rows only (drops pre-F15 fake partials)."""
    rows = []
    with open(TUNING / model / "history.csv") as f:
        for r in csv.DictReader(f):
            if int(float(r["n_records"] or 0)) != 66:
                continue
            p = ast.literal_eval(r["params"])
            rows.append({"trial": int(r["trial"]), "source": r["source"],
                         "overall": float(r["overall_pct"]),
                         "temp": float(p["temperature"]),
                         "top_p": float(p["top_p"]), "top_k": int(p["top_k"])})
    return sorted(rows, key=lambda r: r["trial"])


def load_caches(model):
    """Every completed full-66 config of the whole campaign."""
    out = []
    for fp in glob.glob(str(TUNING / model / "t*.json")):
        d = json.load(open(fp))
        if d.get("n_records") == 66:
            p = d["params"]
            out.append({"overall": d["overall_pct"], "temp": p["temperature"],
                        "top_p": p["top_p"], "top_k": int(p["top_k"]),
                        "cfg": d["cfg_key"]})
    return out


HIST = {m: load_history(m) for m in MODELS}
CACHE = {m: load_caches(m) for m in MODELS}
CONF = json.load(open(TUNING / "confirmations.json"))

BENCH = [r for r in csv.DictReader(open(ROOT / "tests/results/all_metrics.csv"))
         if r["run_id"] == "2026-07-01T21-35-33"]
BENCH_NAME = {  # pretty names for the 5 benchmark models
    "gemma-4-E2B-it-Q8_0.gguf": "gemma-E2B",
    "gemma-4-12b-it-Q8_0.gguf": "gemma-12b",
    "gpt-oss-120b-mxfp4.gguf": "gpt-oss-120b",
    "Qwen3.6-27B-Q4_K_M.gguf": "qwen3.6-27B",
    "Qwen3.6-35B-A3B-Q4_K_M.gguf": "qwen3.6-A3B",
}

TEMPS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TOPPS = [0.65, 0.70, 0.77, 0.84, 0.91, 0.98]


# ── P1 · convergence ─────────────────────────────────────────────────────────
def p1():
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    ax.axvspan(-0.5, 4.5, color=NEUTRAL, zorder=0)
    ax.text(2.0, 96.6, "random\ninit (5)", ha="center", va="top",
            fontsize=10, color=INK2)
    ends = []
    for m in MODELS:
        rows = HIST[m]
        t = [r["trial"] for r in rows]
        best = np.maximum.accumulate([r["overall"] for r in rows])
        ax.plot(t, best, lw=2.2, color=MCOLOR[m], solid_capstyle="round",
                label=MLABEL[m], zorder=3)
        ax.scatter(t, [r["overall"] for r in rows], s=9, color=MCOLOR[m],
                   alpha=0.30, linewidths=0, zorder=2)
        ends.append((m, t[-1], float(best[-1])))
    # dodge end labels: gpt-oss and a3b both end at 92.4 → stack, don't overlap
    ends.sort(key=lambda e: -e[2])
    label_y = []
    for m, tx, val in ends:
        y = val
        if label_y and y > label_y[-1] - 1.0:
            y = label_y[-1] - 1.0
        label_y.append(y)
        ax.annotate(f"{MLABEL[m]}  {val:.1f}%", xy=(tx, y),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=11.5, fontweight="bold",
                    color=MCOLOR[m])
    ax.set_xlim(-1, 118)
    ax.set_ylim(76, 97.5)
    ax.set_xlabel("trial (d-MALIBOO iteration)")
    ax.set_ylabel("Overall pass %")
    ax.set_title("d-MALIBOO convergence — running best over 100 guided trials")
    ax.text(0, -0.13, "dots = per-trial scores · line = best-so-far · "
            "objective = full 66-run suite per trial",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    ax.legend(loc="lower right", ncols=3)
    savefig(fig, "p1_convergence.png")


# ── P2 · search-space landscape ──────────────────────────────────────────────
def p2():
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.9), sharey=True)
    vmin, vmax = 78, 96
    for ax, m in zip(axes, MODELS):
        grid = np.full((len(TOPPS), len(TEMPS)), np.nan)
        for c in CACHE[m]:
            i = TOPPS.index(c["top_p"]); j = TEMPS.index(c["temp"])
            v = grid[i, j]
            grid[i, j] = c["overall"] if np.isnan(v) else max(v, c["overall"])
        masked = np.ma.masked_invalid(grid)
        ax.set_facecolor(NEUTRAL)                      # not-sampled cells
        im = ax.imshow(masked, origin="lower", aspect="auto",
                       cmap=CMAP_BLUE, vmin=vmin, vmax=vmax)
        best = max(CACHE[m], key=lambda c: c["overall"])
        bi, bj = TOPPS.index(best["top_p"]), TEMPS.index(best["temp"])
        ax.scatter([bj], [bi], marker="*", s=260, color=INK, zorder=5)
        ax.set_title(f"{MLABEL[m]} — best {best['overall']:.1f}%", fontsize=13)
        ax.set_xticks(range(len(TEMPS)), [f"{t:g}" for t in TEMPS])
        ax.set_yticks(range(len(TOPPS)), [f"{p:g}" for p in TOPPS])
        ax.set_xlabel("temperature")
        ax.grid(False)
    axes[0].set_ylabel("top-p")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.015)
    cb.set_label("best Overall % (max over top-k)", fontsize=11)
    cb.outline.set_visible(False)
    fig.suptitle("Sampling search space — 384 combos, ~100 guided trials per model",
                 fontsize=15, fontweight="bold", y=1.04)
    fig.text(0.125, -0.045, "★ = campaign winner · gray cells = never sampled by the BO "
             "(low expected improvement)", fontsize=10, color=MUTED)
    savefig(fig, "p2_search_space.png")


# ── P3 · exploration path (winner model) ─────────────────────────────────────
def p3():
    m = "gemma"
    rows = HIST[m]
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    xs, ys, cs = [], [], []
    for r in rows:
        xs.append(TEMPS.index(r["temp"]) + rng.uniform(-0.22, 0.22))
        ys.append(TOPPS.index(r["top_p"]) + rng.uniform(-0.22, 0.22))
        cs.append(r["trial"])
    sc = ax.scatter(xs, ys, c=cs, cmap=CMAP_BLUE, s=64, linewidths=0.6,
                    edgecolors=SURFACE, zorder=3)
    init = [i for i, r in enumerate(rows) if r["trial"] < 5]
    ax.scatter([xs[i] for i in init], [ys[i] for i in init], marker="s", s=90,
               facecolors="none", edgecolors=INK2, linewidths=1.4, zorder=4,
               label="random init")
    best = max(CACHE[m], key=lambda c: c["overall"])
    bx, by = TEMPS.index(best["temp"]), TOPPS.index(best["top_p"])
    ax.scatter([bx], [by], marker="*", s=380, color=INK, zorder=5)
    ax.annotate(f"winner  t{best['temp']:g} / p{best['top_p']:g} / k{best['top_k']}"
                f"  →  {best['overall']:.1f}%",
                xy=(bx, by), xytext=(bx + 0.35, by + 1.45), fontsize=11.5,
                fontweight="bold", color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_xticks(range(len(TEMPS)), [f"{t:g}" for t in TEMPS])
    ax.set_yticks(range(len(TOPPS)), [f"{p:g}" for p in TOPPS])
    ax.set_xlabel("temperature")
    ax.set_ylabel("top-p")
    ax.set_title("BO exploration path — gemma-E2B (top-k jittered out of view)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("trial index (early → late)", fontsize=11)
    cb.outline.set_visible(False)
    ax.legend(loc="lower right")
    ax.grid(False)
    savefig(fig, "p3_exploration.png")


# ── P4 · noise bias (single-shot vs confirmations) ───────────────────────────
def p4():
    single = {"gemma": 95.5, "gpt-oss-120b": 92.4, "qwen36-a3b": 95.5}
    cfg = {"gemma": "tuned  t0.40 / p0.65 / k10",
           "gpt-oss-120b": "tuned  t0.30 / p0.98 / k20",
           "qwen36-a3b": "llama-server defaults (0.8 / 0.95 / 40)"}
    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    ypos = {m: i for i, m in enumerate(reversed(MODELS))}
    for m in MODELS:
        y = ypos[m]
        runs = [r["overall_pct"] for r in CONF
                if r["model"] == m and not r.get("partial")]
        mean = float(np.mean(runs))
        ax.plot([min(runs + [single[m]]), max(runs + [single[m]])], [y, y],
                color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter([single[m]], [y], s=150, color=MCOLOR[m], zorder=3)
        ax.scatter(runs, [y] * len(runs), s=110, facecolors=SURFACE,
                   edgecolors=MCOLOR[m], linewidths=2.0, zorder=3)
        ax.plot([mean, mean], [y - 0.16, y + 0.16], color=INK, lw=2.6, zorder=4)
        ax.annotate(f"Δ {mean - single[m]:+.1f}", xy=(mean, y),
                    xytext=(0, 15), textcoords="offset points", ha="center",
                    fontsize=11.5, fontweight="bold", color=INK)
        # config caption UNDER the row, clear of the range line and markers
        ax.annotate(cfg[m], xy=(83.7, y - 0.34), ha="left", fontsize=9.5,
                    color=INK2)
    # model names as colored y-tick labels (outside the data area — no collisions)
    ax.set_yticks([ypos[m] for m in MODELS], [MLABEL[m] for m in MODELS],
                  fontsize=12.5, fontweight="bold")
    for tick, m in zip(ax.get_yticklabels(), MODELS):
        tick.set_color(MCOLOR[m])
    ax.set_xlim(83.5, 97)
    ax.set_ylim(-0.6, 2.75)
    ax.set_xlabel("Overall pass % (full 66-run suite)")
    ax.set_title("Optimizing on noisy single evaluations inflates the winner")
    handles = [
        Line2D([], [], marker="o", ls="", color=INK2, ms=11,
               label="single-shot (tuning / benchmark)"),
        Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE,
               markeredgecolor=INK2, markeredgewidth=2, ms=10,
               label="repeat runs (confirmation)"),
        Line2D([], [], marker="|", ls="", color=INK, ms=14, markeredgewidth=2.6,
               label="confirmation mean"),
    ]
    ax.legend(handles=handles, loc="lower left", ncols=3,
              bbox_to_anchor=(0, -0.24))
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    savefig(fig, "p4_noise_bias.png")


# ── P5 · sampling sensitivity (distributions) ────────────────────────────────
def p5():
    defaults = {"gemma": 90.9, "gpt-oss-120b": 87.9, "qwen36-a3b": 95.5}
    rng = np.random.default_rng(11)
    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    ypos = {m: i for i, m in enumerate(reversed(MODELS))}
    for m in MODELS:
        y = ypos[m]
        vals = np.array([c["overall"] for c in CACHE[m]])
        jitter = rng.uniform(-0.17, 0.17, len(vals))
        ax.scatter(vals, y + jitter, s=26, color=MCOLOR[m], alpha=0.5,
                   linewidths=0, zorder=2)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([q1, q3], [y - 0.30, y - 0.30], color=INK, lw=2.4,
                solid_capstyle="round", zorder=3)
        ax.plot([med, med], [y - 0.36, y - 0.24], color=INK, lw=2.4, zorder=4)
        ax.scatter([defaults[m]], [y + 0.30], marker="D", s=90, color=INK,
                   zorder=5)
        # label ABOVE the dot band so it never sits on low-score dots
        ax.annotate(f"{MLABEL[m]}   (n={len(vals)}, span "
                    f"{vals.min():.1f}–{vals.max():.1f})",
                    xy=(76.6, y + 0.46), ha="left", va="center",
                    fontsize=12, fontweight="bold", color=MCOLOR[m])
    ax.set_yticks([])
    ax.set_xlim(76, 97.5)
    ax.set_ylim(-0.75, 2.8)
    ax.set_xlabel("Overall pass % — one dot per completed config (66 runs each)")
    ax.set_title("Sampling sensitivity — qwen3.6-A3B barely moves, its default wins")
    handles = [
        Line2D([], [], marker="D", ls="", color=INK, ms=9,
               label="default sampling (benchmark run)"),
        Line2D([], [], marker="|", ls="", color=INK, ms=13, markeredgewidth=2.4,
               label="median · IQR"),
    ]
    ax.legend(handles=handles, loc="lower left", ncols=2,
              bbox_to_anchor=(0, -0.22))
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    savefig(fig, "p5_sensitivity.png")


# ── P6 · Pareto accuracy vs energy ───────────────────────────────────────────
def p6():
    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    ARCH_COLOR = {"dense": "#2a78d6", "MoE": "#008300"}
    pts = []
    for r in BENCH:
        pts.append({"name": BENCH_NAME[r["model"]],
                    "wh": float(r["energy_kwh"]) * 1000.0,
                    "acc": float(r["overall_pct"]),
                    "arch": r["arch"], "size": r["size_b"]})
    # Pareto frontier: lower energy AND higher accuracy dominate
    frontier = [p for p in pts
                if not any(q["wh"] <= p["wh"] and q["acc"] >= p["acc"] and q is not p
                           for q in pts)]
    frontier.sort(key=lambda p: p["wh"])
    ax.plot([p["wh"] for p in frontier], [p["acc"] for p in frontier],
            ls=(0, (4, 3)), color=INK2, lw=1.6, zorder=2)
    # frontier label beside the line's midpoint, in empty space left of it
    fx = min(p["wh"] for p in frontier)
    fy = float(np.mean([p["acc"] for p in frontier]))
    ax.annotate("Pareto frontier", xy=(fx - 0.8, fy), ha="right", va="center",
                fontsize=10.5, color=INK2, style="italic")
    for p in pts:
        onf = p in frontier
        ax.scatter([p["wh"]], [p["acc"]], s=170 if onf else 130,
                   color=ARCH_COLOR[p["arch"]], zorder=4,
                   edgecolors=INK if onf else SURFACE,
                   linewidths=1.6 if onf else 0.8)
        dy = 10 if p["name"] != "qwen3.6-27B" else -16
        ax.annotate(f"{p['name']}  ({p['size']}B {p['arch']})",
                    xy=(p["wh"], p["acc"]), xytext=(0, dy),
                    textcoords="offset points", ha="center", fontsize=11,
                    fontweight="bold" if onf else "normal", color=INK)
    ax.set_xlabel("energy per full 66-run suite (Wh)")
    ax.set_ylabel("Overall pass %")
    ax.set_xlim(20, 102)
    ax.set_ylim(85, 98)
    ax.set_title("Accuracy vs energy — the MoEs own the frontier")
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=11, label=a)
               for a, c in ARCH_COLOR.items()]
    ax.legend(handles=handles, loc="lower right", title="architecture")
    savefig(fig, "p6_pareto.png")


# ── P7 · robustness by prompt level ──────────────────────────────────────────
def p7():
    LV = [("vague", "pass_vague_pct", "#2a78d6"),
          ("clear", "pass_clear_pct", "#008300"),
          ("algorithmic", "pass_algo_pct", "#e87ba4")]
    rows = sorted(BENCH, key=lambda r: -float(r["overall_pct"]))
    fig, ax = plt.subplots(figsize=(11.6, 6.0))
    for i, r in enumerate(rows):
        vals = [float(r[k]) for _, k, _ in LV]
        ax.plot(vals, [i] * 3, color=GRID, lw=5, solid_capstyle="round", zorder=1)
        # dodge exact ties vertically so no dot hides another (e.g. gemma-12b
        # vague == clear == 90.9)
        offs = [0.0] * 3
        for a in range(3):
            for b in range(a + 1, 3):
                if vals[a] == vals[b]:
                    offs[a], offs[b] = 0.11, -0.11
        for (name, k, c), v, dy in zip(LV, vals, offs):
            ax.scatter([v], [i + dy], s=130, color=c, zorder=3,
                       edgecolors=SURFACE, linewidths=1.2)
    ax.set_yticks(range(len(rows)), [BENCH_NAME[r["model"]] for r in rows],
                  fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(78, 102)
    ax.set_xlabel("pass % at prompt level (22 scenarios each)")
    ax.set_title("Robustness by prompt level — vague prompts separate the models")
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=11, label=n)
               for n, _, c in LV]
    ax.legend(handles=handles, loc="lower left", ncols=3,
              bbox_to_anchor=(0, -0.20))
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    savefig(fig, "p7_prompt_levels.png")


if __name__ == "__main__":
    print("generating presentation plots →", OUT)
    p1(); p2(); p3(); p4(); p5(); p6(); p7()
    print("done.")
