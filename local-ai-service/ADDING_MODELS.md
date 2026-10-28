# Adding a New Orchestrator Model

The launcher script (`run_agentic_system.sh`) uses a **model registry**: a two-entry associative array that maps a short key to a GGUF file path and a display name. Adding a new model is three steps.

---

## Prerequisites

- The GGUF file must be compatible with `llama.cpp` (any standard GGUF).
- The GPU must have enough unified memory for the new model **plus** the always-present services:
  - VQA model (Phi-4-reasoning-vision-15B) ≈ 30 GB
  - vlm-segmentation (GroundingDINO + SAM2) ≈ 3 GB
  - MCP tool containers ≈ 2 GB
  - **Available headroom on GB10 (128 GB total): ≈ 60 GB** for the orchestrator GGUF.
- A model's on-disk size in Q8_0 ≈ its VRAM footprint. A 7B Q8 = ~8 GB, 32B Q4 ≈ 20 GB.

---

## Step 1 — Download the GGUF

Create a directory under `local-ai-service/models/` and download the file there.

```bash
# Example: Qwen2.5-7B-Instruct-Q8_0 from bartowski's GGUF repo
mkdir -p local-ai-service/models/qwen25-7b

hf download bartowski/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q8_0.gguf \
    --local-dir local-ai-service/models/qwen25-7b/
```

> `hf` is the HuggingFace CLI tool (`pip install huggingface-hub --break-system-packages`).  
> `huggingface-cli` is deprecated; use `hf` instead.

The directory layout should look like:

```
local-ai-service/models/
├── gemma4/
│   └── gemma-4-E2B-it-Q8_0.gguf
├── phi4/
│   └── Phi-4-mini-instruct-Q4_K_M.gguf
└── qwen25-7b/              ← new
    └── Qwen2.5-7B-Instruct-Q8_0.gguf
```

---

## Download the registry models from HuggingFace

All registry models are publicly available. The **source of truth is the official model**
(Google / Qwen / OpenAI); note that the official orgs mostly release base **safetensors** (Qwen
3.6 also `-FP8`), while the project's registry uses specific **GGUF quants** — so you either
convert the official weights yourself (path A, the "original models") or pull a ready-made
community GGUF (path B). The launcher checks for the **exact** filename in `ORCH_MODEL_PATHS`, so
land the `.gguf` at that path (rename, or edit the registry line — `--list-models` prints it).

Only the default (`gemma`) is required to start the system; the rest are for the model comparison.

Official collections: **Gemma 4** → <https://huggingface.co/collections/google/googles-gemma-models-family> ·
**Qwen3.6** → <https://huggingface.co/collections/Qwen/qwen36>

| Key | Registry file | Official model (source of truth) | Official GGUF? |
|-----|---------------|----------------------------------|----------------|
| `gemma` *(default)* | `gemma-4-E2B-it-Q8_0.gguf` | `google/gemma-4-E2B-it` | only QAT `google/gemma-4-E2B-it-qat-q4_0-gguf`; Q8_0 = quantize (A) or community (B) |
| `gemma4-12b` | `gemma-4-12b-it-Q8_0.gguf` | `google/gemma-4-12B-it` | same (QAT-only official) |
| `qwen36-27b` | `Qwen3.6-27B-Q4_K_M.gguf` | `Qwen/Qwen3.6-27B` (+ `-FP8`) | **no** → quantize (A) or community (B) |
| `qwen36-a3b` | `Qwen3.6-35B-A3B-Q4_K_M.gguf` | `Qwen/Qwen3.6-35B-A3B` (+ `-FP8`) | **no** → quantize (A) or community (B) |
| `gpt-oss-120b` | `gpt-oss-120b-mxfp4-*.gguf` | `openai/gpt-oss-120b` (native MXFP4) | yes → `ggml-org/gpt-oss-120b-GGUF` |
| `qwen25-7b`, `phi4-mini`, `llama31-8b`, `deepseek-v2-lite`, `mistral-small`, `command-r` | *(see registry)* | `Qwen/…`, `microsoft/…`, `meta-llama/…`, `deepseek-ai/…`, `mistralai/…`, `CohereForAI/…` | community GGUF (B): `bartowski/…`, `unsloth/…`, `lmstudio-community/…` |

**Path A — from the official weights (produces the exact registry GGUF).** Uses the llama.cpp you
built in Quick Start step 5.1:

```bash
# hf = HuggingFace CLI:  pip install huggingface-hub --break-system-packages
hf download Qwen/Qwen3.6-35B-A3B --local-dir /tmp/qwen36-a3b-hf         # official safetensors
python llama.cpp/convert_hf_to_gguf.py /tmp/qwen36-a3b-hf \
    --outfile /tmp/qwen36-a3b-f16.gguf --outtype f16                    # → GGUF (f16)
./llama.cpp/build-cuda/bin/llama-quantize /tmp/qwen36-a3b-f16.gguf \
    local-ai-service/models/qwen36-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf Q4_K_M   # → registry path
```

Same three steps for any official model — swap the repo, `--outtype` and the final quant
(`Q8_0` for the gemma keys, `Q4_K_M` for the qwen keys).

**Path B — ready-made community GGUF (quicker; not the official org).**

```bash
# Default gemma (Q8_0) — filename already matches the registry, no rename
hf download ggml-org/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q8_0.gguf \
    --local-dir local-ai-service/models/gemma4/

# qwen36-a3b — bartowski prefixes files with "Qwen_", so rename to the registry name
hf download bartowski/Qwen_Qwen3.6-35B-A3B-GGUF --include "*Q4_K_M*" \
    --local-dir local-ai-service/models/qwen36-a3b/
mv local-ai-service/models/qwen36-a3b/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf \
   local-ai-service/models/qwen36-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf

# gpt-oss-120b — official MXFP4 conversion (sharded); needs an MXFP4-capable llama.cpp build
hf download ggml-org/gpt-oss-120b-GGUF --include "*mxfp4*" \
    --local-dir local-ai-service/models/gpt-oss-120b/
```

> `--include "*QUANT*"` fetches the right quant regardless of the repo's exact filename; just make
> sure the resulting `.gguf` lands at the path the registry expects.

---

## Step 2 — Register the model

Open `run_agentic_system.sh` and find the two registry arrays (around line 46). Add **one line** to each:

```bash
# BEFORE
declare -A ORCH_MODEL_PATHS=(
    [gemma]="$LOCAL_AI_DIR/models/gemma4/gemma-4-E2B-it-Q8_0.gguf"
    [phi4-mini]="$LOCAL_AI_DIR/models/phi4/Phi-4-mini-instruct-Q4_K_M.gguf"
)
declare -A ORCH_MODEL_NAMES=(
    [gemma]="gemma-4-E2B-it-Q8_0.gguf"
    [phi4-mini]="Phi-4-mini-instruct-Q4_K_M.gguf"
)

# AFTER — add one line to each array (same key in both)
declare -A ORCH_MODEL_PATHS=(
    [gemma]="$LOCAL_AI_DIR/models/gemma4/gemma-4-E2B-it-Q8_0.gguf"
    [phi4-mini]="$LOCAL_AI_DIR/models/phi4/Phi-4-mini-instruct-Q4_K_M.gguf"
    [qwen25-7b]="$LOCAL_AI_DIR/models/qwen25-7b/Qwen2.5-7B-Instruct-Q8_0.gguf"
)
declare -A ORCH_MODEL_NAMES=(
    [gemma]="gemma-4-E2B-it-Q8_0.gguf"
    [phi4-mini]="Phi-4-mini-instruct-Q4_K_M.gguf"
    [qwen25-7b]="Qwen2.5-7B-Instruct-Q8_0.gguf"
)
```

Rules:
- The **key** (e.g. `qwen25-7b`) is what you pass to `--orchestrator-model`.
- The **path** must resolve to the `.gguf` file. Use `$LOCAL_AI_DIR/models/<dir>/<file>.gguf`.
- The **name** is a display string that appears in test result labels — use the filename.
- Nothing else in the script changes.

Verify the registry:

```bash
./local-ai-service/run_agentic_system.sh --list-models
```

---

## Step 3 — Start the stack with the new model

```bash
# First time: use --fresh to tear down the previous orchestrator container
# (needed because LLM_MODEL env var must change)
./local-ai-service/run_agentic_system.sh --orchestrator-model qwen25-7b --fresh

# Subsequent restarts (model already registered, no container change needed):
./local-ai-service/run_agentic_system.sh --orchestrator-model qwen25-7b
```

The script will:
1. Validate the GGUF file exists.
2. Kill any running llama-server and relaunch with the new model on the GPU.
3. Start/restart all MCP tool containers.
4. Pre-warm the VQA model (Phi-4) — this takes 2–3 minutes on first load.
5. Print `ALL SYSTEMS READY` with a service status table.

---

## Running tests with the new model

```bash
# Full 66-scenario suite (writes to a dedicated log)
docker exec -d agents-orchestrator bash -c \
  "python tests/orchestrator/run_tests.py \
   > /data/runs/logs/test_run_qwen25_7b.log 2>&1 \
   && echo DONE >> /data/runs/logs/test_run_qwen25_7b.log"

# Monitor progress
docker exec agents-orchestrator tail -f /data/runs/logs/test_run_qwen25_7b.log

# Targeted scenarios only
docker exec agents-orchestrator python tests/orchestrator/run_tests.py \
  --scenarios SC-05,SC-06,SC-08,SC-09 --levels vague,clear,algorithmic
```

The model name (from `ORCH_MODEL_NAMES`) is embedded in every test result line, so results from different models can coexist in the same log or be compared side by side.

---

## Model recommendations

| Key | Model | VRAM | Best for |
|-----|-------|------|----------|
| `gemma` | gemma-4-E2B Q8 | ~5 GB | Fast baseline, low memory |
| `phi4-mini` | Phi-4-mini Q4 | ~3 GB | Microsoft instruction-tuned baseline |
| `gemma4-12b` | gemma-4-12B Q8 | ~13 GB | **Recommended**: same family as gemma, 6× params |
| `qwen25-7b` | Qwen2.5-7B Q8 | ~8 GB | Avoid — parallel tool-call fan-out issues |
| `qwen36-27b` | Qwen3.6-27B Q4_K_M | ~17 GB | Dense Qwen3.6; needs thinking disabled (see note below) |
| `qwen36-a3b` | Qwen3.6-35B-A3B Q4_K_M | ~21 GB | **Best benchmark result (95.5%)** — MoE, ~3B active; thinking disabled |

> **Note on `--reasoning-budget 0`**: this llama.cpp flag is for DeepSeek-R1 / "thinking" models only.
> It suppresses EOS tokens on standard instruct models (Gemma, Qwen, Llama), causing runaway
> generation that fills the full context window. The default `LLAMA_EXTRA_ARGS` uses `--n-predict 2048`
> instead, which works correctly with all models.

> **Note on hybrid-thinking models (Qwen3 / Qwen3.6)**: these default to a long internal
> reasoning trace before every tool call, which times out the test suite. Disable it via the
> chat template: set `chat_template_kwargs={"enable_thinking": False}` on the model's
> `REGISTRY` entry in `scripts/run_model_comparison.py` (merged into `--chat-template-kwargs`
> at launch), or add `--chat-template-kwargs '{"enable_thinking":false}'` to
> `LLAMA_EXTRA_ARGS` for interactive use. No chat-template file override is needed —
> Qwen3.6's native template parses tool calls cleanly with `--jinja`.

For **function-calling / multi-step chain tasks** (this project's workload), see the
full benchmark in [`docs/MODEL_COMPARISON.md`](../docs/MODEL_COMPARISON.md): gemma-4
and gpt-oss lead; Mistral-Small (France) and Falcon3-10B (UAE) are usable; Qwen2.5
over-calls; Cohere Command-R and DeepSeek-V2-Lite loop and were dropped.

---

## Adding a model to the comparison harness (with tool-call overrides)

The benchmark harness (`scripts/run_model_comparison.py`) has its **own** `REGISTRY`,
separate from `run_agentic_system.sh`. Add a row there to include a model in
`--models`. Crucially, some GGUFs need a **tool-template override** so `llama-server`
can parse their tool calls into OpenAI `tool_calls`:

```python
# scripts/run_model_comparison.py → REGISTRY
"falcon3-10b": {                       # plug-and-play: native <tool_call> format
    "path": LOCAL_AI_DIR / "models/falcon3-10b/Falcon3-10B-Instruct-q5_k_m.gguf",
    "name": "Falcon3-10B-Instruct-q5_k_m.gguf", "ctx": 16384,
},
"mistral-small": {                     # GGUF has no tool template → supply one
    "path": LOCAL_AI_DIR / "models/mistral-small/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
    "name": "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf", "ctx": 16384,
    "chat_template": TEMPLATES_DIR / "Mistral-Small-3.2-relaxed.jinja",
},
"command-r7b": {                       # unparseable native format → Hermes + stop
    "path": LOCAL_AI_DIR / "models/command-r7b/c4ai-command-r7b-12-2024-Q6_K.gguf",
    "name": "c4ai-command-r7b-12-2024-Q6_K.gguf", "ctx": 16384,
    "chat_template": TEMPLATES_DIR / "NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja",
    "extra_stop": ["</tool_call>"],
},
```

- `chat_template` → adds `--chat-template-file` to the `llama-server` launch.
- `extra_stop` → passed to the orchestrator via `LLM_EXTRA_STOP` → `ModelSettings`.
- **Decide which (if any) you need by probing first:** load the model, send a
  `/v1/chat/completions` request with a `tools` array, and check the response has
  `message.tool_calls` (not tool-call text in `content`, and not HTTP 500). If it's
  text/500, try the bundled Hermes template (+ `stop=["</tool_call>"]` if it loops).
  Bundled templates live in `llama.cpp/models/templates/`.

See [`docs/MODEL_COMPARISON.md`](../docs/MODEL_COMPARISON.md#tool-calling-not-every-model-just-works)
for the plug-and-play rule and the full per-model fix table.

---

## Troubleshooting

**`Permission denied` on llama-server log**: the default log path `/tmp/llama-server.log` is writable by all users. If you see this with a custom path, override via env:
```bash
LLAMA_LOG=/tmp/my-llama.log ./local-ai-service/run_agentic_system.sh --orchestrator-model qwen25-7b
```

**llama-server never becomes ready**: check the log at `$LLAMA_LOG` (default `/tmp/llama-server.log`) for CUDA errors or OOM messages. Reduce context size if memory is tight:
```bash
LLAMA_CTX_SIZE=8192 ./local-ai-service/run_agentic_system.sh --orchestrator-model qwen25-7b
```

**`unknown orchestrator model` error**: the key passed to `--orchestrator-model` is not in the registry. Run `--list-models` to see registered keys.
