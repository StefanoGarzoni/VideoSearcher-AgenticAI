#!/usr/bin/env python3
"""
run_model_comparison.py — benchmark several ORCHESTRATOR LLMs across the full
test suite and collect, per model: the existing metrics (handoff_match,
vision_pass, result_pass) plus TTFT (time-to-first-token).

Why a host-side driver?  All local GGUFs are served by ONE llama-server on
port 9000, so the models cannot be tested simultaneously.  This script, for
each model, (re)starts llama-server with that GGUF on the host, measures TTFT
with a direct streaming probe, runs the scenario suite INSIDE the orchestrator
container (which already computes every answer-based metric), stamps the
measured TTFT onto each result record, then moves to the next model.

The VQA model (microsoft/Phi-4-reasoning-vision-15B) is held constant — it is
loaded once inside the orchestrator container and reused across every run.

Usage:
    python3 scripts/run_model_comparison.py --models gemma,phi4-mini,qwen25-7b,gemma4-12b,gpt-oss-120b
    python3 scripts/run_model_comparison.py --models gemma,qwen25-7b --scenarios SC-01,SC-29 --levels clear
    python3 scripts/run_model_comparison.py --models gpt-oss-120b --reasoning low
    python3 scripts/run_model_comparison.py --list-models

Per-model results land in tests/results/<run_id>_<model_slug>.json (one file
per model, shared run_id) plus a cross-model summary in
tests/results/markdowns/comparison_<run_id>.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Paths / constants ────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_AI_DIR = PROJECT_ROOT / "local-ai-service"
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
VISUALISE = PROJECT_ROOT / "scripts" / "visualise_results.py"

LLAMA_BIN = Path(
    os.getenv("LLAMA_SERVER_BIN", str(PROJECT_ROOT / "llama.cpp" / "build-cuda" / "bin" / "llama-server"))
)
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "9000"))
LLAMA_LOG = Path(os.getenv("LLAMA_LOG", "/tmp/llama-server-comparison.log"))
# Tool-use chat templates VENDORED into the repo (llama.cpp/ is gitignored, so its
# bundled templates aren't in a fresh clone). Some GGUFs ship without a tool template
# (or one this llama.cpp build's autoparser can't parse), so we override with a
# template whose tool-call format the server CAN parse into OpenAI tool_calls.
TEMPLATES_DIR = PROJECT_ROOT / "scripts" / "chat_templates"
ORCH_CONTAINER = "agents-orchestrator"

# ── Orchestrator model registry (mirrors local-ai-service/run_agentic_system.sh) ─
# To add a model: drop its .gguf under local-ai-service/models/<dir>/ and add a row.
REGISTRY: dict[str, dict] = {
    "phi4-mini": {
        "path": LOCAL_AI_DIR / "models/phi4/Phi-4-mini-instruct-Q4_K_M.gguf",
        "name": "Phi-4-mini-instruct-Q4_K_M.gguf",
        "ctx": 16384,
        # Default template renders no tools (model replies "I can't access tools"),
        # which caused its 0% in the old run → Hermes + stop fixes parsing.
        "chat_template": TEMPLATES_DIR / "NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja",
        "extra_stop": ["</tool_call>"],
    },
    "gemma": {
        "path": LOCAL_AI_DIR / "models/gemma4/gemma-4-E2B-it-Q8_0.gguf",
        "name": "gemma-4-E2B-it-Q8_0.gguf",
        "ctx": 16384,
    },
    "qwen25-7b": {
        "path": LOCAL_AI_DIR / "models/qwen25-7b/Qwen2.5-7B-Instruct-Q8_0.gguf",
        "name": "Qwen2.5-7B-Instruct-Q8_0.gguf",
        "ctx": 16384,
    },
    "gemma4-12b": {
        "path": LOCAL_AI_DIR / "models/gemma4-12b/gemma-4-12b-it-Q8_0.gguf",
        "name": "gemma-4-12b-it-Q8_0.gguf",
        "ctx": 16384,
    },
    "gpt-oss-120b": {
        "path": LOCAL_AI_DIR / "models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf",
        "name": "gpt-oss-120b-mxfp4.gguf",
        "ctx": 8192,            # smaller KV cache: 120B + VQA must share 128 GB
        "gpt_oss": True,        # supports the reasoning-effort knob
    },
    # Diversity set — different companies / countries / sizes / speed.
    "llama31-8b": {            # Meta · USA · 8B dense · fastest
        "path": LOCAL_AI_DIR / "models/llama31-8b/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        "name": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        "ctx": 16384,
    },
    "deepseek-v2-lite": {      # DeepSeek · China · 16B MoE (~2.4B active) · fast
        "path": LOCAL_AI_DIR / "models/deepseek-v2-lite/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf",
        "name": "DeepSeek-V2-Lite-Chat-Q4_K_M.gguf",
        "ctx": 16384,
        # GGUF has no tool template → Hermes <tool_call> format (parses cleanly).
        "chat_template": TEMPLATES_DIR / "NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja",
    },
    "mistral-small": {         # Mistral AI · France · 24B dense · medium
        "path": LOCAL_AI_DIR / "models/mistral-small/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
        "name": "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
        "ctx": 16384,
        # GGUF default template has no tools → same-family v3.2 [TOOL_CALLS] template,
        # patched ("relaxed") to sanitize the SDK's tool-call IDs to 9 chars instead
        # of raising (the stock template hard-requires 9-char IDs and 500s otherwise).
        "chat_template": TEMPLATES_DIR / "Mistral-Small-3.2-relaxed.jinja",
    },
    "command-r": {             # Cohere · Canada · 32B dense · tool-use specialist
        "path": LOCAL_AI_DIR / "models/command-r/c4ai-command-r-08-2024-Q4_K_M.gguf",
        "name": "c4ai-command-r-08-2024-Q4_K_M.gguf",
        "ctx": 16384,
        # Native "Action:" tool format is unparseable by this build's autoparser;
        # Hermes parses, but Command-R loops without a stop at the tool-call close.
        "chat_template": TEMPLATES_DIR / "NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja",
        "extra_stop": ["</tool_call>"],
    },
    "falcon3-10b": {           # TII · UAE · 10B dense · native <tool_call> format
        "path": LOCAL_AI_DIR / "models/falcon3-10b/Falcon3-10B-Instruct-q5_k_m.gguf",
        "name": "Falcon3-10B-Instruct-q5_k_m.gguf",
        "ctx": 16384,
        # Embedded template already uses Hermes-style <tool_call> → parses cleanly,
        # no override or stop needed (verified via probe).
    },
    "command-r7b": {           # Cohere · Canada · 7B dense · newer (12-2024) model
        "path": LOCAL_AI_DIR / "models/command-r7b/c4ai-command-r7b-12-2024-Q6_K.gguf",
        "name": "c4ai-command-r7b-12-2024-Q6_K.gguf",
        "ctx": 16384,
        # Like the 32B, its native Action format is unparseable here → Hermes + stop.
        "chat_template": TEMPLATES_DIR / "NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja",
        "extra_stop": ["</tool_call>"],
    },
    # ── Qwen3.6 candidates — added for the orchestrator comparison ───────────────
    # Models: Qwen/Qwen3.6-27B (dense) and Qwen/Qwen3.6-35B-A3B (MoE, ~3B active),
    # from https://huggingface.co/collections/Qwen/qwen36 . Qwen ships the originals
    # as safetensors (base + FP8) only — llama-server needs GGUF, so `path` points at
    # a GGUF quant of those SAME weights (e.g. bartowski/Qwen_Qwen3.6-*-GGUF or
    # unsloth), exactly as every other model in this table is a GGUF quant. Drop the
    # file under local-ai-service/models/<dir>/ (dir = the key) and match `name` to it.
    # Requires a recent llama.cpp build (Qwen3.6 + MTP support, merged upstream).
    # enable_thinking=false suppresses Qwen3's thinking trace, which would otherwise
    # time out every tool call (Gemma's old 0% failure mode). If this build rejects
    # the kwarg, drop it and disable thinking via the template / --reasoning-budget 0.
    "qwen36-27b": {            # Alibaba · China · 27B dense
        "path": LOCAL_AI_DIR / "models/qwen36-27b/Qwen3.6-27B-Q4_K_M.gguf",
        "name": "Qwen3.6-27B-Q4_K_M.gguf",
        "ctx": 16384,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "qwen36-a3b": {            # Alibaba · China · 35B-A3B MoE (~3B active)
        "path": LOCAL_AI_DIR / "models/qwen36-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf",
        "name": "Qwen3.6-35B-A3B-Q4_K_M.gguf",
        "ctx": 16384,
        "chat_template_kwargs": {"enable_thinking": False},
    },
}

# ── Tiny logging helpers ─────────────────────────────────────────────────────

def info(msg: str) -> None:
    print(f"\033[0;36m[INFO]\033[0m  {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"\033[0;32m[ OK ]\033[0m  {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"\033[1;33m[WARN]\033[0m  {msg}", flush=True)


def err(msg: str) -> None:
    print(f"\033[0;31m[FAIL]\033[0m  {msg}", file=sys.stderr, flush=True)


def slugify(model_name: str) -> str:
    """Mirror ModelConfig.slug() so we can predict the result filename."""
    return model_name.replace("/", "_").replace(" ", "-").replace(".", "_")


_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def gguf_complete(path: Path) -> bool:
    """True if the GGUF is fully present. For split files (…-00001-of-00003.gguf)
    every shard must exist — a partial download leaves only the first shards."""
    m = _SPLIT_RE.search(path.name)
    if not m:
        return path.is_file()
    total = int(m.group(2))
    stem = path.name[: m.start()]
    return all(
        (path.parent / f"{stem}-{i:05d}-of-{total:05d}.gguf").is_file()
        for i in range(1, total + 1)
    )


# ── llama-server lifecycle ───────────────────────────────────────────────────

def stop_llama_server() -> None:
    """Kill any running llama-server and wait for the port to free."""
    subprocess.run(["pkill", "-f", "llama-server"], check=False)
    subprocess.run(["pkill", "-f", "run_server.sh"], check=False)
    for _ in range(20):
        if not _llama_healthy():
            time.sleep(1)
            return
        time.sleep(0.5)


def _llama_healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{LLAMA_PORT}/health", timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return data.get("status") == "ok"
    except Exception:
        return False


def loaded_model_path() -> Optional[Path]:
    """Return the GGUF path the live llama-server currently serves (via /props), or None."""
    try:
        with urllib.request.urlopen(f"http://localhost:{LLAMA_PORT}/props", timeout=3) as resp:
            mp = json.loads(resp.read().decode()).get("model_path")
        return Path(mp) if mp else None
    except Exception:
        return None


# ── Autonomous resource gating ───────────────────────────────────────────────

def available_memory_gb() -> float:
    """Free unified (CPU+GPU) memory in GB, read from /proc/meminfo MemAvailable.
    On the GB10/DGX unified-memory box, GPU allocations are drawn from system RAM,
    so MemAvailable tracks GPU pressure — e.g. a co-tenant ollama job holding 80 GB
    shows up here. (`nvidia-smi --query-gpu=memory.used` reports N/A on GB10.)"""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return 0.0


def wait_for_memory(min_free_gb: float, poll_s: int = 30) -> None:
    """Block until >= min_free_gb of unified memory is free, so the suite never
    runs GPU-starved. This is what makes the pipeline autonomous: if another job
    (e.g. ollama) is hogging the GB10, we wait it out instead of producing slow,
    timed-out runs with corrupted latency/throughput/energy. Polls quietly; logs
    a status line each poll while it waits so progress is visible in the log."""
    if min_free_gb <= 0:
        return
    waited = False
    while available_memory_gb() < min_free_gb:
        free = available_memory_gb()
        info(f"⏳ Waiting for memory: {free:.0f} GB free, need {min_free_gb:.0f} GB "
             f"(co-tenant GPU job busy) — re-checking in {poll_s}s…")
        waited = True
        time.sleep(poll_s)
    if waited:
        ok(f"Memory available: {available_memory_gb():.0f} GB free ≥ {min_free_gb:.0f} GB — proceeding.")


def ensure_server(model: dict, reasoning: Optional[str]) -> None:
    """Make llama-server serve `model`. If it is ALREADY serving it, reuse the
    live process untouched (this is how we avoid killing a server we don't need
    to restart, e.g. a gemma already running)."""
    if _llama_healthy():
        cur = loaded_model_path()
        if cur is not None and cur.resolve() == model["path"].resolve():
            ok(f"Reusing the already-loaded server for {model['name']} — not restarting.")
            return
    stop_llama_server()
    start_llama_server(model, reasoning)


def start_llama_server(
    model: dict, reasoning: Optional[str],
    extra_server_args: Optional[list] = None,
) -> subprocess.Popen:
    """Launch llama-server for one model; block until /health == ok.

    extra_server_args: appended verbatim to the llama-server command line —
    used by scripts/tune_orchestrator.py to inject sampling overrides
    (--temp / --top-p / --top-k) per Bayesian-optimization trial.
    """
    cmd = [
        str(LLAMA_BIN),
        "--host", "0.0.0.0",
        "--port", str(LLAMA_PORT),
        "--model", str(model["path"]),
        "--ctx-size", str(model.get("ctx", 16384)),
        "--n-gpu-layers", "-1",
        "--jinja",
        "--n-predict", "2048",
    ]
    if model.get("chat_template"):
        # Override the GGUF's embedded template with one whose tool-call format
        # this build's autoparser can parse into OpenAI tool_calls.
        cmd += ["--chat-template-file", str(model["chat_template"])]
    # Per-model chat-template kwargs, merged into a single --chat-template-kwargs flag:
    #   - gpt-oss takes a reasoning-effort knob via the harmony template.
    #   - Qwen3 (and other hybrid-thinking models) take enable_thinking=false to
    #     suppress the internal reasoning trace that otherwise times out every tool
    #     call — the same failure mode as Gemma's thinking default (see DEVELOPMENT.md).
    template_kwargs = dict(model.get("chat_template_kwargs", {}))
    if model.get("gpt_oss") and reasoning:
        template_kwargs["reasoning_effort"] = reasoning
    if template_kwargs:
        cmd += ["--chat-template-kwargs", json.dumps(template_kwargs)]
    if extra_server_args:
        cmd += [str(a) for a in extra_server_args]

    info(f"Launching llama-server: {model['name']} (ctx={model.get('ctx', 16384)})")
    logf = open(LLAMA_LOG, "a")
    logf.write(f"\n\n===== {datetime.now().isoformat()} :: {model['name']} =====\n")
    logf.flush()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, start_new_session=True)

    info("Waiting for llama-server to be ready (no timeout — large models load slowly)...")
    while True:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited early (code {proc.returncode}). See {LLAMA_LOG}"
            )
        if _llama_healthy():
            ok("llama-server ready.")
            return proc
        time.sleep(3)


# ── TTFT probe ───────────────────────────────────────────────────────────────

def measure_ttft(served_name: str, samples: int) -> dict:
    """Measure time-to-first-token via a direct streaming call to llama-server.

    Returns {median_ms, mean_ms, min_ms, samples:[...]} — TTFT is a model-serving
    property (independent of scenario), so one measurement applies to the model.
    """
    url = f"http://localhost:{LLAMA_PORT}/v1/chat/completions"
    prompts = [
        "Reply with a single short sentence.",
        "Name one primary colour.",
        "Say hello in one word.",
    ]
    times: list[float] = []
    for i in range(samples):
        payload = {
            "model": served_name,
            "messages": [{"role": "user", "content": prompts[i % len(prompts)]}],
            "stream": True,
            "max_tokens": 16,
            "temperature": 0,
        }
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0].get("delta", {})
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    # First token in ANY channel (gpt-oss emits reasoning first).
                    if delta.get("content") or delta.get("reasoning_content"):
                        times.append((time.perf_counter() - t0) * 1000)
                        break
        except Exception as exc:
            warn(f"TTFT sample {i + 1} failed: {exc}")
    if not times:
        return {"median_ms": None, "mean_ms": None, "min_ms": None, "samples": []}
    return {
        "median_ms": round(statistics.median(times), 1),
        "mean_ms": round(statistics.mean(times), 1),
        "min_ms": round(min(times), 1),
        "samples": [round(t, 1) for t in times],
    }


def measure_throughput(served_name: str, max_tokens: int = 128) -> Optional[float]:
    """Decode throughput (tokens/sec) via a streaming generation, measured from
    the first token to the last (excludes prefill, which TTFT already covers)."""
    url = f"http://localhost:{LLAMA_PORT}/v1/chat/completions"
    payload = {
        "model": served_name,
        "messages": [{"role": "user", "content": "Write a detailed paragraph about how video processing pipelines work."}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    t_first = t_last = None
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n += 1
    except Exception as exc:
        warn(f"Throughput probe failed: {exc}")
        return None
    if t_first is None or t_last <= t_first or n < 2:
        return None
    return round((n - 1) / (t_last - t_first), 1)


# ── Suite execution (inside the orchestrator container) ──────────────────────

def run_suite_in_container(
    model: dict, run_id: str, scenarios: str, levels: str, timeout_s: int,
    extra_env: Optional[dict] = None,
) -> None:
    """Run the full scenario suite for ONE model via docker exec.

    extra_env: additional container env vars (e.g. the tuner caps
    VQA_PREWARM_TIMEOUT_S per trial — the cache is warm there, so the
    benchmark's generous cold-load default would let a wedged GPU burn hours).
    """
    models_json = json.dumps([{
        "model_name": model["name"],
        "provider": "openai-chat",
        "base_url": "http://host.docker.internal:9000/v1",
        "api_key": "not-needed",
        "display_name": model["name"],
    }])
    env_flags = [
        "-e", f"TEST_RUN_ID={run_id}",
        "-e", f"TEST_MODELS_JSON={models_json}",
        "-e", "DATA_DIR=/data",
    ]
    if scenarios:
        env_flags += ["-e", f"TEST_SCENARIO_IDS={scenarios}"]
    if levels:
        env_flags += ["-e", f"TEST_PROMPT_LEVELS={levels}"]
    if model.get("extra_stop"):
        env_flags += ["-e", f"LLM_EXTRA_STOP={json.dumps(model['extra_stop'])}"]
    for k, v in (extra_env or {}).items():
        env_flags += ["-e", f"{k}={v}"]

    cmd = ["docker", "exec", *env_flags, ORCH_CONTAINER,
           "python", "tests/orchestrator/run_tests.py"]
    info(f"Running suite in container for {model['name']} ...")
    # NOT check=True: run_tests.py exits non-zero when scenarios error (e.g. a few
    # timeouts), but the per-scenario results ARE still written. Raising here would
    # drop the model's TTFT/energy/summary entirely, so we tolerate non-zero and
    # let the caller stamp + summarise from the result file that exists.
    try:
        proc = subprocess.run(cmd, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        # `subprocess` timeout kills only the docker-exec CLIENT; the suite keeps
        # running INSIDE the container. Kill it, or it overlaps the next model —
        # whose ensure_server() swaps the llama-server GGUF — and corrupts that
        # model's early scenarios (two orchestrators, wrong weights). This exact
        # cascade killed the 100-trial run on 2026-07-11 (gemma trial-64 hang →
        # zombie → gpt-oss/a3b first trials found no result file → crashed).
        subprocess.run(["docker", "exec", ORCH_CONTAINER, "pkill", "-f", "run_tests.py"],
                       check=False)
        raise
    if proc.returncode != 0:
        warn(f"run_tests exited {proc.returncode} for {model['name']} "
             f"(some scenarios errored) — proceeding with the results it wrote.")


_EMISSIONS_CSV = PROJECT_ROOT / "logs" / "emissions" / "emissions.csv"


def container_now() -> datetime:
    """Current time on the ORCHESTRATOR CONTAINER's clock. CodeCarbon emissions
    timestamps are written in container-local time, which can differ from the
    host timezone — so energy windows must be bracketed in container time."""
    try:
        out = subprocess.run(
            ["docker", "exec", ORCH_CONTAINER, "date", "+%Y-%m-%d %H:%M:%S"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return datetime.strptime(out, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now().replace(tzinfo=None)


def attribute_energy_kwh(start: datetime, end: datetime) -> float:
    """Sum CodeCarbon energy_consumed (kWh) for all emissions rows whose timestamp
    falls in [start, end). This attributes per-model energy by run time-window —
    covering tool energy (L2) and, on instrumented runs, the orchestrator request
    (L1 LLM + L3 overhead) and VQA (L1)."""
    if not _EMISSIONS_CSV.exists():
        return 0.0
    total = 0.0
    try:
        with _EMISSIONS_CSV.open(newline="") as f:
            for row in csv.DictReader(f):
                ts = (row.get("timestamp") or "").replace("T", " ").split(".")[0]
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                if start <= dt < end:
                    try:
                        total += float(row.get("energy_consumed") or 0.0)
                    except ValueError:
                        pass
    except Exception as exc:
        warn(f"Energy attribution failed: {exc}")
    return total


def _chown_to_host(path: Path) -> None:
    """Result files are written by the container as ROOT; chown them back to the
    host user (via the container, which is root) so host-side post-processing can
    rewrite them. Best-effort — never raises."""
    try:
        subprocess.run(
            ["docker", "exec", ORCH_CONTAINER, "chown",
             f"{os.getuid()}:{os.getgid()}", f"/app/tests/results/{path.name}"],
            check=False, capture_output=True,
        )
    except Exception:
        pass


def stamp_ttft(run_id: str, model: dict, ttft: dict) -> Optional[Path]:
    """Inject the per-model TTFT into each record of that model's result file.

    Returns the result path even if stamping fails (the scenario data is still
    valid) so downstream rendering/summary always proceed."""
    result_path = RESULTS_DIR / f"{run_id}_{slugify(model['name'])}.json"
    if not result_path.exists():
        warn(f"Result file not found, cannot stamp TTFT: {result_path}")
        return None
    _chown_to_host(result_path)  # container writes as root; make it host-writable
    try:
        records = json.loads(result_path.read_text())
        for r in records:
            r["ttft_ms"] = ttft.get("median_ms")
        result_path.write_text(json.dumps(records, indent=2))
        (RESULTS_DIR / f"{run_id}_{slugify(model['name'])}_ttft.json").write_text(
            json.dumps({"model": model["name"], **ttft}, indent=2)
        )
    except Exception as exc:
        warn(f"Could not stamp TTFT into {result_path.name} ({exc}); scenario data is still valid.")
    return result_path


# ── Cross-model comparison summary ───────────────────────────────────────────

def _overall_pass(r: dict) -> bool:
    if not r.get("handoff_match"):
        return False
    if r.get("vision_pass") is not None and not r["vision_pass"]:
        return False
    if r.get("result_pass") is not None and not r["result_pass"]:
        return False
    return True


def _is_error_scenario(r: dict) -> bool:
    """Error-handling scenarios expect a graceful reject/clarification."""
    exp = " ".join(str(a) for a in r.get("expected_agents", [])).lower()
    return "reject" in exp or "clarification" in exp


def compute_model_metrics(records: list[dict], ttft_ms, throughput_tps, energy_kwh) -> dict:
    """All comparison metrics for one model, derived from its result records
    plus the externally-measured TTFT / throughput / energy."""
    total = len(records)
    pct = lambda n, d: f"{n}/{d} ({100 * n // d}%)" if d else "n/a"

    n_overall = sum(1 for r in records if _overall_pass(r))
    n_hand = sum(1 for r in records if r.get("handoff_match"))
    vis = [r for r in records if r.get("vision_pass") is not None]
    res = [r for r in records if r.get("result_pass") is not None]
    avg_latency = statistics.mean(r["execution_time_ms"] / 1000 for r in records)

    # #1 Tool-call efficiency: extra agent calls beyond the expected chain,
    # over records that DID call the right agents (handoff_match=True).
    matched = [r for r in records if r.get("handoff_match")]
    extras = [max(0, len(r.get("handoff_sequence", [])) - len(r.get("expected_agents", [])))
              for r in matched]
    avg_extra = round(statistics.mean(extras), 2) if extras else None

    # #4 Robustness by prompt level: overall pass rate at each level.
    by_level = {}
    for lvl in ("vague", "clear", "algorithmic"):
        lv = [r for r in records if r.get("prompt_level") == lvl]
        by_level[lvl] = pct(sum(1 for r in lv if _overall_pass(r)), len(lv))

    # #7 Error-handling accuracy: pass rate on reject/clarification scenarios.
    err = [r for r in records if _is_error_scenario(r)]
    err_acc = pct(sum(1 for r in err if _overall_pass(r)), len(err))

    # #8 Tokens per task (present only on instrumented runs).
    toks = [r["total_tokens"] for r in records if r.get("total_tokens") is not None]
    avg_tokens = round(statistics.mean(toks)) if toks else None

    # #14 Efficiency frontier: correctness per unit cost.
    total_min = sum(r["execution_time_ms"] for r in records) / 60000.0
    pass_per_kwh = round(n_overall / energy_kwh, 1) if energy_kwh else None
    pass_per_min = round(n_overall / total_min, 2) if total_min else None

    return {
        "model": records[0]["model_name"],
        "n_overall": n_overall,
        "overall": pct(n_overall, total),
        "handoff": pct(n_hand, total),
        "vision": pct(sum(1 for r in vis if r["vision_pass"]), len(vis)),
        "result": pct(sum(1 for r in res if r["result_pass"]), len(res)),
        "avg_latency_s": avg_latency,
        "ttft_ms": ttft_ms,
        "throughput_tps": throughput_tps,
        "energy_kwh": energy_kwh,
        "avg_extra_calls": avg_extra,        # #1
        "by_level": by_level,                # #4
        "err_handling": err_acc,             # #7
        "avg_tokens": avg_tokens,            # #8
        "pass_per_kwh": pass_per_kwh,        # #14
        "pass_per_min": pass_per_min,        # #14
    }


def build_comparison(run_id: str, per_model: list[dict]) -> Path:
    md_dir = RESULTS_DIR / "markdowns"
    md_dir.mkdir(parents=True, exist_ok=True)
    out = md_dir / f"comparison_{run_id}.md"

    def cell(v, fmt="{}"):
        return fmt.format(v) if v is not None else "n/a"

    lines = [
        f"# Model Comparison — run {run_id}",
        "",
        "VQA model held constant: `microsoft/Phi-4-reasoning-vision-15B`.",
        "",
        "## Headline",
        "",
        "| Model | Overall pass | Avg latency | TTFT (ms) | Throughput (tok/s) | Energy/run (kWh) | Tokens/task |",
        "|-------|--------------|-------------|-----------|--------------------|------------------|-------------|",
    ]
    for m in per_model:
        lines.append(
            f"| {m['model']} | {m['overall']} | {m['avg_latency_s']:.1f} s | "
            f"{cell(m.get('ttft_ms'))} | {cell(m.get('throughput_tps'))} | "
            f"{cell(m.get('energy_kwh'), '{:.6f}')} | {cell(m.get('avg_tokens'))} |"
        )

    lines += [
        "",
        "## Correctness & efficiency",
        "",
        "| Model | Handoff | Vision | Result | Extra calls/task (eff.) | Error-handling |",
        "|-------|---------|--------|--------|-------------------------|----------------|",
    ]
    for m in per_model:
        lines.append(
            f"| {m['model']} | {m['handoff']} | {m['vision']} | {m['result']} | "
            f"{cell(m.get('avg_extra_calls'))} | {m.get('err_handling', 'n/a')} |"
        )

    lines += [
        "",
        "## Robustness by prompt level (overall pass rate)",
        "",
        "| Model | Vague | Clear | Algorithmic |",
        "|-------|-------|-------|-------------|",
    ]
    for m in per_model:
        bl = m.get("by_level", {})
        lines.append(
            f"| {m['model']} | {bl.get('vague','n/a')} | {bl.get('clear','n/a')} | "
            f"{bl.get('algorithmic','n/a')} |"
        )

    lines += [
        "",
        "## Efficiency frontier (correctness per cost)",
        "",
        "| Model | Passes / kWh | Passes / min |",
        "|-------|--------------|--------------|",
    ]
    for m in per_model:
        lines.append(
            f"| {m['model']} | {cell(m.get('pass_per_kwh'))} | {cell(m.get('pass_per_min'))} |"
        )

    lines += [
        "",
        "---",
        "_**Extra calls/task** = avg agent handoffs beyond the expected chain (lower = leaner). "
        "**Error-handling** = pass rate on reject/clarification scenarios. "
        "**TTFT / Throughput** = model-serving latency / decode speed (streaming probe). "
        "**Energy/run** = CodeCarbon energy attributed by run time-window (L1 LLM + L2 tools + L3 overhead). "
        "**Tokens/task** = avg orchestrator tokens (instrumented runs only). "
        "**Passes/kWh** & **Passes/min** = the accuracy-vs-cost frontier._",
    ]
    out.write_text("\n".join(lines))
    return out


def summarise_model_file(run_id: str, model: dict, ttft: dict,
                         throughput_tps=None, energy_kwh=None) -> Optional[dict]:
    result_path = RESULTS_DIR / f"{run_id}_{slugify(model['name'])}.json"
    if not result_path.exists():
        return None
    records = json.loads(result_path.read_text())
    if not records:
        return None
    return compute_model_metrics(records, ttft.get("median_ms"), throughput_tps, energy_kwh)


# ── Main ─────────────────────────────────────────────────────────────────────

def _container_running() -> bool:
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", ORCH_CONTAINER],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() == "true"
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="", help="Comma-separated registry keys (see --list-models).")
    p.add_argument("--scenarios", default="", help="Comma-separated scenario IDs (default: all).")
    p.add_argument("--levels", default="", help="Comma-separated prompt levels (default: all three).")
    p.add_argument("--reasoning", default=None, choices=["low", "medium", "high"],
                   help="gpt-oss reasoning effort (ignored by other models). Recommend 'low' for routing.")
    p.add_argument("--ttft-samples", type=int, default=5, help="TTFT probe repetitions per model.")
    p.add_argument("--suite-timeout", type=int, default=36000, help="Per-model suite timeout (s).")
    p.add_argument("--min-free-gb", type=float, default=0.0,
                   help="Wait until this much unified memory is free before loading each "
                        "model (autonomous gating around co-tenant GPU jobs). 0 = no wait.")
    p.add_argument("--keep-server", action="store_true",
                   help="Leave the last-run model loaded at the end (default: restore the model that was running before the comparison).")
    p.add_argument("--list-models", action="store_true", help="Print the registry and exit.")
    args = p.parse_args()

    if args.list_models:
        print("Available orchestrator models (--models KEY[,KEY...]):")
        for k, m in REGISTRY.items():
            present = "✓" if gguf_complete(m["path"]) else "✗ (not downloaded / incomplete)"
            print(f"  {k:14s} {present:32s} {m['path']}")
        return 0

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    if not keys:
        err("No --models given. Use --list-models to see options.")
        return 1
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        err(f"Unknown model key(s): {unknown}. Use --list-models.")
        return 1

    # Pre-flight
    if not LLAMA_BIN.is_file():
        err(f"llama-server binary not found: {LLAMA_BIN}")
        return 1
    if not _container_running():
        err(f"Container '{ORCH_CONTAINER}' is not running. Start the stack first "
            f"(local-ai-service/run_agentic_system.sh).")
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    # Reuse-friendly ordering: if a server is already running one of the requested
    # models, run THAT model first off the live process (so it is never killed
    # just to be reloaded). We also remember it to restore at the end.
    original = loaded_model_path() if _llama_healthy() else None
    if original is not None:
        for k in list(keys):
            if REGISTRY[k]["path"].resolve() == original.resolve():
                keys = [k] + [x for x in keys if x != k]
                info(f"A server is already serving '{k}' — running it first (reused, not killed).")
                break

    info(f"Comparison run_id = {run_id}")
    info(f"Models (in run order): {keys}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_model_summaries: list[dict] = []
    for key in keys:
        model = REGISTRY[key]
        print("\n" + "=" * 70)
        info(f"MODEL: {key}  ({model['name']})")
        print("=" * 70)

        if not gguf_complete(model["path"]):
            warn(f"GGUF missing or incomplete (still downloading?), skipping {key}: {model['path']}")
            continue

        # Autonomous gate: don't load until the GPU has room (waits out co-tenants).
        wait_for_memory(args.min_free_gb)

        try:
            ensure_server(model, args.reasoning)
        except Exception as exc:
            err(f"Could not start llama-server for {key}: {exc}")
            continue

        try:
            info(f"Measuring TTFT ({args.ttft_samples} samples) + throughput...")
            ttft = measure_ttft(model["name"], args.ttft_samples)
            tput = measure_throughput(model["name"])
            ok(f"TTFT median = {ttft['median_ms']} ms | throughput = {tput} tok/s")

            # Bracket the suite to attribute per-model energy by time-window.
            # Use container time (matches CodeCarbon's emissions timestamps, which
            # can be in a different timezone than the host).
            win_start = container_now()
            run_suite_in_container(model, run_id, args.scenarios, args.levels, args.suite_timeout)
            win_end = container_now()
            stamped = stamp_ttft(run_id, model, ttft)

            if stamped:
                subprocess.run([sys.executable, str(VISUALISE), str(stamped)], check=False)
                energy_kwh = attribute_energy_kwh(win_start, win_end)
                s = summarise_model_file(run_id, model, ttft, tput, energy_kwh)
                if s:
                    per_model_summaries.append(s)
                    ok(f"{key}: overall {s['overall']} | extra-calls {s['avg_extra_calls']} | "
                       f"err-handling {s['err_handling']} | TTFT {s['ttft_ms']}ms | "
                       f"{tput} tok/s | energy {energy_kwh:.6f} kWh")
        except subprocess.TimeoutExpired:
            err(f"Suite timed out for {key}.")
        except Exception as exc:
            err(f"Run failed for {key}: {exc}")
        # Note: no stop here — the next model's ensure_server() swaps the server,
        # and the originally-loaded model is restored after the loop.

    # Restore the environment to how we found it (unless asked to keep the last one).
    if not args.keep_server and original is not None:
        restore = next(
            (m for m in REGISTRY.values() if m["path"].resolve() == original.resolve()),
            {"path": original, "name": original.name, "ctx": 16384},
        )
        if loaded_model_path() != original:
            info(f"Restoring originally-loaded model: {restore['name']}")
            try:
                stop_llama_server()
                start_llama_server(restore, None)
                ok(f"Restored — {restore['name']} serving on :{LLAMA_PORT}.")
            except Exception as exc:
                warn(f"Could not restore original model: {exc}")

    if per_model_summaries:
        comp = build_comparison(run_id, per_model_summaries)
        print("\n" + "=" * 70)
        ok(f"Comparison complete. Cross-model summary: {comp}")
        print("=" * 70)
        for s in per_model_summaries:
            print(f"  {s['model']:34s} overall={s['overall']:14s} TTFT={s['ttft_ms']}ms")
    else:
        warn("No models produced results.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        err("Interrupted — stopping llama-server.")
        stop_llama_server()
        sys.exit(130)
