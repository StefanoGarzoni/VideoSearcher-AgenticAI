> **Availability:** Full-stack reruns require private code absent from this export. [Details](PRIVATE_COMPONENTS.md).

# Model Comparison — Orchestrator LLM Benchmark

This benchmark answers one question: **which LLM should drive the orchestrator?**
The VQA model is held constant (`microsoft/Phi-4-reasoning-vision-15B`); only the
orchestrator "brain" changes. Each candidate runs the full 66-scenario suite
(22 scenarios × 3 prompt levels) and is scored on 14 metrics.

All of this is driven by three host-side scripts in `scripts/`:

| Script | Role |
|---|---|
| `run_model_comparison.py` | Load each model on `llama-server`, probe TTFT/throughput, run the suite in the orchestrator container, attribute energy, write per-model result JSON + a comparison markdown. |
| `consolidate_metrics.py` | Scan every run's result JSON + comparison md → one tidy `tests/results/all_metrics.csv` (one row per run × model, all 14 metrics). |
| `visualise_results.py` | Render results: per-run Markdown, or `--charts` / `--merge` figures, or `--table` (the final 14-metric table). |
| `tune_orchestrator.py` | Bayesian tuning of the winner's sampling params (temp/top-p/top-k) over the same suite — see [HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md). |

---

## Host-side setup (fresh clone)

The scripts run on the **host** and talk to `llama-server` + the orchestrator container.

```bash
# 1. The full stack (see top-level README / local-ai-service/).
echo "Private component not distributed; see docs/PRIVATE_COMPONENTS.md"                       # VLMs-Testing-Framework

# 2. llama.cpp with CUDA — gitignored, so build it locally (provides llama-server).
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build-cuda -DGGML_CUDA=ON
cmake --build llama.cpp/build-cuda --config Release -j --target llama-server
#   (override the binary path with LLAMA_SERVER_BIN if you build elsewhere.)

# 3. Python deps for the chart/merge modes (the rest is stdlib).
pip install -r scripts/requirements.txt --break-system-packages

# 4. Download each model's GGUF — see local-ai-service/ADDING_MODELS.md.
#    Missing models are skipped automatically by run_model_comparison.py.
```

Chat templates are vendored in `scripts/chat_templates/` (not taken from the
gitignored `llama.cpp/`), so the per-model overrides work in a fresh clone.

---

## Quick start

```bash
# 1. Run the comparison for one or more models (registry keys, comma-separated).
#    --min-free-gb makes it wait for GPU headroom before each model (co-tenant safe).
python3 scripts/run_model_comparison.py --models qwen36-27b,qwen36-a3b --min-free-gb 40

# 2. Consolidate every run into one CSV.
python3 scripts/consolidate_metrics.py            # → tests/results/all_metrics.csv

# 3a. Merged figures across runs (heatmap + radar + Pareto):
python3 scripts/visualise_results.py --merge      # → tests/results/charts_merged_*

# 3b. The final 14-metric table (Markdown):
python3 scripts/visualise_results.py --table      # → tests/results/markdowns/final_metrics_table.md
```

`--models` keys come from the `REGISTRY` in `run_model_comparison.py` (see
`--list-models`). Already-computed models are reused via the CSV — no need to re-run.

---

## The 14 metrics

| # | Metric | CSV column | Rule |
|---|--------|-----------|------|
| 1 | Overall pass | `overall_pct` | all gates pass |
| 2 | Handoff match | `handoff_pct` | right agents, right order |
| 3 | Vision pass | `vision_pct` | keyword contract |
| 4 | Result pass | `result_pct` | ground-truth scorer |
| 5 | Tool-call efficiency | `extra_calls_per_task` | extra handoffs beyond expected chain |
| 6 | Error-handling | `error_handling_pct` | pass rate on reject/clarification scenarios |
| 7 | Robustness by prompt level | `pass_vague/clear/algo_pct` | pass rate per level |
| 8 | Avg latency | `avg_latency_s` | mean wall-clock |
| 9 | TTFT | `ttft_ms` | time-to-first-token (streaming probe) |
| 10 | Throughput | `throughput_tps` | decode tokens/sec |
| 11 | Tokens/task | `tokens_per_task` | SDK usage |
| 12 | Energy/run | `energy_kwh` | CodeCarbon, L1+L2+L3 (see [CODECARBON.md](CODECARBON.md)) |
| 13 | Passes per kWh | `passes_per_kwh` | accuracy ÷ energy |
| 14 | Passes per min | `passes_per_min` | accuracy ÷ time |

---

## Tool-calling: not every model "just works"

The orchestrator delegates via **tool calls** (handoffs). A model is only usable
if `llama-server` can parse its tool-call format into OpenAI `tool_calls`, AND the
model is capable enough not to loop in multi-step chains.

**Plug-and-play** (just `--jinja`, no overrides): models whose GGUF template uses
the Hermes-style `<tool_call>{...}</tool_call>` format, or that are
Gemma/Qwen/Llama/Falcon family, or have a dedicated parser in this `llama.cpp`
build (gemma4, gpt-oss). Examples here: **gemma-4, gpt-oss, Qwen2.5, Qwen3.6, Falcon3-10B.**

**Needs a per-model override** in the `REGISTRY` (`chat_template`, `extra_stop`,
and/or `chat_template_kwargs`):

| Model | Fix | Why |
|---|---|---|
| mistral-small-2501 | `chat_template` → `Mistral-Small-3.2-relaxed.jinja` | GGUF ships no tool template; the stock 3.2 template `raise_exception`s on the SDK's tool-call IDs — the *relaxed* copy sanitizes IDs to 9 chars. |
| phi4-mini | Hermes template + `extra_stop=["</tool_call>"]` | default template renders no tools ("I can't access tools"). |
| command-r (32B & 7B) | Hermes + `extra_stop` | native Cohere `Action:` format is unparseable here. |
| Qwen3.6 (27B & 35B-A3B) | `chat_template_kwargs={"enable_thinking": false}` | native template parses tool calls fine, but Qwen3 defaults to *thinking mode* — a long reasoning trace before every tool call that times out the suite (same failure mode as Gemma's old thinking default). The kwarg is merged into `--chat-template-kwargs` at server launch. |

`extra_stop` is passed to the orchestrator via the `LLM_EXTRA_STOP` env var, which
`main_agent.py` turns into `ModelSettings(extra_args={"stop": [...]})`.

> **Pre-download check:** if a model's chat template mentions `<tool_call>`, or it's
> a Qwen/Llama/Gemma/Falcon derivative, it will plug in clean. Exotic formats
> (`Action:`, recipient `>>>`, raw `[TOOL_CALLS]`) mean template work.

**Dropped as non-viable** (parse OK but loop/ramble in multi-turn → timeouts):
command-r 32B & 7B (Cohere), deepseek-v2-lite (too small), and phi4-mini (parses
now, but rambles 6k-token replies — kept only as a 0% baseline).

See [`local-ai-service/ADDING_MODELS.md`](../local-ai-service/ADDING_MODELS.md) for
the full add-a-model recipe, and the `model_tool_call_viability` memory note.

---

## Current result — 5 models on the F-35 suite (run `2026-07-01T21-35-33`)

The suite's vision half now runs on the **F-35 fixtures** (`F-28..F-37`) with
ground-truth–backed result scoring (`tests/ground_truth/videos/f-35/`). 2 countries,
3 companies, 5B→120B, dense + MoE. Generated by `--table`
(full precision in `tests/results/markdowns/final_metrics_table.md`):

| Metric | gemma-12b | gemma-E2B | gpt-oss-120b | qwen3.6-27B | qwen3.6-35B-A3B |
|---|---|---|---|---|---|
| Country | USA | USA | USA | China | China |
| Size · Arch | 12 d | 5 d | 120 MoE | 27 d | 35 MoE (~3B active) |
| Overall % | 93.9 | 90.9 | 87.9 | 87.9 | **95.5** |
| Handoff % | **97.0** | 93.9 | 87.9 | 89.4 | 95.5 |
| Vision % | 100 | 88.9 | 88.9 | 100 | 100 |
| Result % | 92.6 | 92.6 | 96.3 | 96.3 | **100** |
| Error-handling % | 100 | 100 | 100 | 73.3 | 93.3 |
| TTFT (ms) | 433 | 91 | 204 | 426 | 135 |
| Throughput (tok/s) | 16.8 | 83.2 | 53.5 | 11.6 | 74.1 |
| Energy/run (kWh) | 0.093 | 0.035 | 0.028 | 0.087 | 0.029 |
| Passes/kWh | 668 | 1701 | 2050 | 670 | **2163** |

**Reads:** **qwen3.6-35B-A3B** wins both accuracy (95.5%) and efficiency
(2163 passes/kWh) — the MoE's ~3B active params keep decode fast (74 tok/s) at
27B-beating quality. **gemma-4-12b** has the most precise routing (97% handoff,
0.09 extra calls/task) but ~3× the energy of the MoEs. The dense **qwen3.6-27B**
disappoints: slowest decode (11.6 tok/s on this GPU), weakest error handling
(11/15) — its MoE sibling dominates it on every metric. Failure patterns worth
knowing: `SC-06/vague` (full 6-step pipeline from a vague prompt) fails **all
five** models; `F-33` (5-step cross-modal) fails only gpt-oss-120b, at all three
prompt levels.

---

## Legacy result — 7 models on the lionandgorilla suite (June 2026)

Preserved for the record. **Not comparable with the table above**: the vision
scenarios (then `SC-28..SC-37`), fixtures (`lionandgorilla.mp4`/`tedtalk.mp4`,
since lost — see `data/fixtures/raw/DOWNLOADS.md`), and vision scoring
(keyword-only, no ground truth) all differ.

| Metric | gemma-12b | gemma-E2B | gpt-oss-120b | qwen2.5-7B | mistral-24B | falcon3-10B | phi4-mini* |
|---|---|---|---|---|---|---|---|
| Country | USA | USA | USA | China | France | UAE | USA |
| Overall % | **93.9** | 89.4 | 87.9 | 60.6 | 48.5 | 4.5 | 0.0 |
| Handoff % | 95.5 | 92.4 | 87.9 | 72.7 | 56.1 | 15.2 | 0.0 |
| TTFT (ms) | 398 | 92 | 306 | 82 | 575 | 94 | 58 |
| Throughput (tok/s) | 16.2 | 78.4 | 52.7 | 28.2 | 14.1 | 26.7 | 73.8 |
| Energy/run (kWh) | 0.108 | 0.034 | 0.044 | 0.082 | 0.070 | 0.013 | 0.001 |
| Passes/kWh | 576 | **1712** | 1333 | 486 | 456 | 232 | 0 |

`*phi4-mini` = broken-parsing baseline. **Reads (then):** gemma-4-12b best
accuracy; gemma-4-E2B the efficiency sweet spot; qwen2.5-7B over-called
(1.17 extra handoffs/task); mistral-small and falcon3-10B trailed.
