#!/usr/bin/env bash
# run_agentic_system.sh — start the FULL VideoSearcher AgenticAI stack and block
# until the system is COMPLETELY up.
#
# "Completely up" means ALL of: the orchestrator GGUF served on the GPU, every MCP
# tool server healthy, the orchestrator container running, AND the VQA model loaded
# into GPU memory. The VQA model load is a MANDATORY part of startup (the agentic
# system cannot answer visual questions without it) — the script only prints
# "ALL SYSTEMS READY" once it has finished loading. There is no way to skip it.
#
# Usage:
#   ./run_agentic_system.sh [--build] [--fresh]
#                           [--orchestrator-model KEY] [--vqa-model HF_ID] [--list-models]
#
#   --build                  Pass --build to docker compose up (rebuild images)
#   --fresh                  Tear down existing containers before starting (volumes kept)
#   --orchestrator-model KEY Which local GGUF to serve for the orchestrator (default: gemma).
#                            See the model registry below / run --list-models.
#   --vqa-model HF_ID        HuggingFace model id for the in-process VQA agent
#                            (default: microsoft/Phi-4-reasoning-vision-15B)
#   --list-models            Print the orchestrator model registry and exit.
#
# The orchestrator GGUF is always served on the GPU via the CUDA llama-server build.
# Once it reports ready, use the system with:
#    docker exec -it agents-orchestrator python videosearcher_cli.py
set -Eeuo pipefail

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_AI_DIR="$SCRIPT_DIR"

LLAMA_PORT=9000
# Per-checkout by default, not shared /tmp — a fixed shared-/tmp path can already
# be owned by a different user on a multi-user host, which fails writes silently
# and (pre-fix) hung this phase forever. logs/ lives inside this checkout, so it's
# never shared across users/clones the way /tmp is.
LLAMA_LOG="${LLAMA_LOG:-$PROJECT_ROOT/logs/llama-server.log}"
ORCHESTRATOR_CONTAINER="agents-orchestrator"

# Fast MCP servers (HTTP 200 + any status field is sufficient)
MCP_PORTS=(8000 8001 8002 8003 8004 8005 8006 8009 8010 8016 8025)
# Slow GPU server: requires model_loaded=true in the JSON body
SEG_PORT=8008

# ── Model registry ───────────────────────────────────────────────────────────
# To add an orchestrator model: drop its .gguf under local-ai-service/models/<dir>/
# and add ONE line to each array below (path + display name). Nothing else changes.
declare -A ORCH_MODEL_PATHS=(
    [gemma]="$LOCAL_AI_DIR/models/gemma4/gemma-4-E2B-it-Q8_0.gguf"
    [phi4-mini]="$LOCAL_AI_DIR/models/phi4/Phi-4-mini-instruct-Q4_K_M.gguf"
    [qwen25-7b]="$LOCAL_AI_DIR/models/qwen25-7b/Qwen2.5-7B-Instruct-Q8_0.gguf"
    [gemma4-12b]="$LOCAL_AI_DIR/models/gemma4-12b/gemma-4-12b-it-Q8_0.gguf"
    [gpt-oss-120b]="$LOCAL_AI_DIR/models/gpt-oss-120b/gpt-oss-120b-mxfp4-00001-of-00003.gguf"
    [llama31-8b]="$LOCAL_AI_DIR/models/llama31-8b/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
    [deepseek-v2-lite]="$LOCAL_AI_DIR/models/deepseek-v2-lite/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
    [mistral-small]="$LOCAL_AI_DIR/models/mistral-small/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf"
    [command-r]="$LOCAL_AI_DIR/models/command-r/c4ai-command-r-08-2024-Q4_K_M.gguf"
    # Qwen3.6 candidates (orchestrator comparison): Qwen3.6-27B (dense) + 35B-A3B (MoE),
    # https://huggingface.co/collections/Qwen/qwen36 . Qwen ships safetensors/FP8 only —
    # download a GGUF quant of the same weights (bartowski/unsloth) into the dirs below.
    # Needs a recent llama.cpp build (Qwen3.6 + MTP). Qwen3 defaults to thinking mode;
    # for interactive use add  --chat-template-kwargs '{"enable_thinking":false}'  to
    # LLAMA_EXTRA_ARGS, or benchmark via scripts/run_model_comparison.py (sets it
    # automatically). Otherwise every tool call times out.
    [qwen36-27b]="$LOCAL_AI_DIR/models/qwen36-27b/Qwen3.6-27B-Q4_K_M.gguf"
    [qwen36-a3b]="$LOCAL_AI_DIR/models/qwen36-a3b/Qwen3.6-35B-A3B-Q4_K_M.gguf"
)
# Name string reported to the orchestrator (appears in test-result labels).
declare -A ORCH_MODEL_NAMES=(
    [gemma]="gemma-4-E2B-it-Q8_0.gguf"
    [phi4-mini]="Phi-4-mini-instruct-Q4_K_M.gguf"
    [qwen25-7b]="Qwen2.5-7B-Instruct-Q8_0.gguf"
    [gemma4-12b]="gemma-4-12b-it-Q8_0.gguf"
    [gpt-oss-120b]="gpt-oss-120b-mxfp4.gguf"
    [llama31-8b]="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
    [deepseek-v2-lite]="DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
    [mistral-small]="Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf"
    [command-r]="c4ai-command-r-08-2024-Q4_K_M.gguf"
    [qwen36-27b]="Qwen3.6-27B-Q4_K_M.gguf"
    [qwen36-a3b]="Qwen3.6-35B-A3B-Q4_K_M.gguf"
)
ORCH_MODEL_DEFAULT="gemma"
VQA_MODEL_DEFAULT="microsoft/Phi-4-reasoning-vision-15B"

# CUDA-enabled llama-server binary — offloads all layers to the GB10 GPU.
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$PROJECT_ROOT/llama.cpp/build-cuda/bin/llama-server}"
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-16384}"
LLAMA_NGL="${LLAMA_NGL:--1}"                              # -1 = offload every layer
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:---jinja --n-predict 2048}"

# ── Argument parsing ─────────────────────────────────────────────────────────

BUILD_FLAG=""
FRESH=false
ORCH_MODEL="$ORCH_MODEL_DEFAULT"
VQA_MODEL_NAME_SEL="$VQA_MODEL_DEFAULT"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)              BUILD_FLAG="--build"; shift ;;
        --fresh)              FRESH=true;           shift ;;
        --orchestrator-model) ORCH_MODEL="${2:?--orchestrator-model needs a value}"; shift 2 ;;
        --vqa-model)          VQA_MODEL_NAME_SEL="${2:?--vqa-model needs a value}";  shift 2 ;;
        --list-models)
            echo "Orchestrator models (use: --orchestrator-model KEY):"
            for k in "${!ORCH_MODEL_PATHS[@]}"; do
                printf "  %-12s %s\n" "$k" "${ORCH_MODEL_PATHS[$k]}"
            done
            echo "Default: $ORCH_MODEL_DEFAULT"
            exit 0
            ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0
            ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Resolve the selected orchestrator model from the registry.
# (Helpers like die() are defined below, so use a plain guard here.)
MODEL_PATH="${ORCH_MODEL_PATHS[$ORCH_MODEL]:-}"
[[ -n "$MODEL_PATH" ]] || {
    echo "ERROR: unknown orchestrator model '$ORCH_MODEL'. Run with --list-models." >&2
    exit 1
}
ORCH_MODEL_NAME="${ORCH_MODEL_NAMES[$ORCH_MODEL]:-$(basename "$MODEL_PATH")}"

# ── Helpers ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[ OK ]${RESET}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
die()     { echo -e "${RED}[FAIL]${RESET}  $*" >&2; exit 1; }

# Returns 0 if llama-server /health responds with status=ok
llama_ready() {
    local resp
    resp=$(curl -sf --max-time 3 "http://localhost:${LLAMA_PORT}/health" 2>/dev/null) || return 1
    echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('status')=='ok' else 1)" 2>/dev/null
}

# Returns 0 if a generic MCP /health responds with HTTP 200 and any status field
mcp_ready() {
    local port=$1
    local resp
    resp=$(curl -sf --max-time 5 "http://localhost:${port}/health" 2>/dev/null) || return 1
    echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('status') in ('ok','healthy') else 1)" 2>/dev/null
}

# Returns 0 if vlm-segmentation /health responds with model_loaded=true
seg_ready() {
    local resp
    resp=$(curl -sf --max-time 5 "http://localhost:${SEG_PORT}/health" 2>/dev/null) || return 1
    echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('model_loaded') else 1)" 2>/dev/null
}

# ── Phase 1: Pre-flight checks ───────────────────────────────────────────────

echo ""
info "=== Phase 1: Pre-flight checks ==="

command -v docker >/dev/null 2>&1 || die "docker not found in PATH."
docker info >/dev/null 2>&1       || die "Docker daemon is not running."
ok "Docker daemon running."

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 \
    || die "nvidia-smi failed — GPU or NVIDIA drivers not accessible."
ok "GPU accessible (nvidia-smi ok)."

[[ -f "$PROJECT_ROOT/agents-container/.env" ]] \
    || die "agents-container/.env not found. Copy .env.example and fill in your keys."
ok ".env file present."

# Selected orchestrator GGUF (resolved from the registry above).
[[ -f "$MODEL_PATH" ]] \
    || die "GGUF model not found: $MODEL_PATH\nAdd the file or pick another with --orchestrator-model."
ok "Orchestrator model '$ORCH_MODEL' → $MODEL_PATH (served as '$ORCH_MODEL_NAME')"

[[ -x "$LLAMA_SERVER_BIN" ]] \
    || die "CUDA llama-server not found/executable: $LLAMA_SERVER_BIN\nBuild it (llama.cpp with -DGGML_CUDA=ON) or set LLAMA_SERVER_BIN."
ok "CUDA llama-server: $LLAMA_SERVER_BIN"

info "VQA model: $VQA_MODEL_NAME_SEL"

[[ -d "$PROJECT_ROOT/data/fixtures/raw" ]] \
    || die "data/fixtures/raw/ directory not found — test fixtures and input videos are expected there."
ok "data/fixtures/raw/ directory exists."

# The proprietary VLM framework is intentionally not distributed with this
# source snapshot. It must be supplied separately by its rights-holder before
# a full-stack build can be attempted. Do not offer a broken git submodule clone.
VLM_FRAMEWORK_SRC="$PROJECT_ROOT/VLMs-Testing-Framework/src"
if [[ ! -d "$VLM_FRAMEWORK_SRC" ]] || [[ -z "$(ls -A "$VLM_FRAMEWORK_SRC" 2>/dev/null)" ]]; then
    die "Full-stack build unavailable: private VLM framework source is not included.\nThe segmentation MCP server and in-process VQA agent require it.\nSee docs/PRIVATE_COMPONENTS.md for architecture and access limitations.\nNo public clone or source code for this dependency is provided."
fi
ok "Private VLM framework source supplied independently."

# ── Phase 2: llama-server ────────────────────────────────────────────────────

echo ""
info "=== Phase 2: llama-server (port $LLAMA_PORT) ==="

mkdir -p "$(dirname "$LLAMA_LOG")" 2>/dev/null || true

# Always (re)launch the selected model on the GPU so the served model is
# guaranteed to match --orchestrator-model. The CUDA build loads in seconds.
info "Stopping any existing llama-server on port $LLAMA_PORT..."
pkill -f "llama-server" 2>/dev/null || true
sleep 2

info "Launching llama-server on GPU: model='$ORCH_MODEL_NAME' ngl=$LLAMA_NGL ctx=$LLAMA_CTX_SIZE"
# shellcheck disable=SC2086
nohup "$LLAMA_SERVER_BIN" \
    --host 0.0.0.0 --port "$LLAMA_PORT" \
    --model "$MODEL_PATH" \
    --ctx-size "$LLAMA_CTX_SIZE" \
    --n-gpu-layers "$LLAMA_NGL" \
    $LLAMA_EXTRA_ARGS \
    >> "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!
info "llama-server pid $LLAMA_PID. Logs: $LLAMA_LOG"

# Fail fast if the process died immediately — e.g. LLAMA_LOG isn't writable (shared
# /tmp, permission denied, read-only mount). A redirection failure kills the
# backgrounded process instantly and silently; without this check, the readiness
# loop below would poll a server that will never exist, forever, with no error.
sleep 1
kill -0 "$LLAMA_PID" 2>/dev/null \
    || die "llama-server exited immediately after launch (pid $LLAMA_PID is gone).\nThis usually means LLAMA_LOG isn't writable: $LLAMA_LOG\nOverride it: LLAMA_LOG=/path/you/can/write.log ./run_agentic_system.sh ..."

info "Waiting for llama-server to be ready..."
LLAMA_READY_TIMEOUT=180
LLAMA_WAITED=0
until llama_ready; do
    kill -0 "$LLAMA_PID" 2>/dev/null \
        || die "llama-server process died while waiting for readiness. Check $LLAMA_LOG for details."
    echo -n "."
    sleep 3
    LLAMA_WAITED=$((LLAMA_WAITED + 3))
    [[ "$LLAMA_WAITED" -ge "$LLAMA_READY_TIMEOUT" ]] \
        && die "llama-server did not become ready within ${LLAMA_READY_TIMEOUT}s. Check $LLAMA_LOG for details."
done
echo ""
ok "llama-server ready (GPU)."

# ── Phase 3: MCP tool servers ────────────────────────────────────────────────

echo ""
info "=== Phase 3: MCP tool servers ==="

cd "$PROJECT_ROOT"

if $FRESH; then
    info "--fresh: tearing down MCP containers (volumes kept)..."
    docker compose -f docker-compose.yml down
fi

info "Starting MCP containers..."
# shellcheck disable=SC2086
docker compose -f docker-compose.yml up -d $BUILD_FLAG

# Parallel health pollers — each writes its port to a temp file when ready
TMPDIR_READY="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_READY"' EXIT

_poll_mcp() {
    local port=$1
    until mcp_ready "$port"; do
        sleep 5
    done
    touch "$TMPDIR_READY/$port"
}

_poll_seg() {
    until seg_ready; do
        sleep 10
    done
    touch "$TMPDIR_READY/$SEG_PORT"
}

info "Polling health endpoints in parallel..."
POLLER_PIDS=()
for p in "${MCP_PORTS[@]}"; do
    _poll_mcp "$p" &
    POLLER_PIDS+=($!)
done
_poll_seg &
POLLER_PIDS+=($!)

# Wait for all pollers and show progress
ALL_PORTS=("${MCP_PORTS[@]}" "$SEG_PORT")
while true; do
    ready_count=$(ls "$TMPDIR_READY" 2>/dev/null | wc -l)
    total_count=${#ALL_PORTS[@]}
    echo -ne "\r  ${ready_count}/${total_count} servers ready...   "
    [[ "$ready_count" -ge "$total_count" ]] && break
    sleep 2
done
echo ""
# Reap ONLY the poller jobs. A bare `wait` would also block on the llama-server
# background job started in Phase 2, which never exits.
wait "${POLLER_PIDS[@]}" 2>/dev/null || true

ok "All ${#ALL_PORTS[@]} MCP servers healthy."

# ── Phase 4: Orchestrator container ─────────────────────────────────────────

echo ""
info "=== Phase 4: Orchestrator container ==="

if $FRESH; then
    info "--fresh: tearing down orchestrator container..."
    docker compose -f docker-compose.agents.yml down
fi

# Pass the selected models to the orchestrator container. docker-compose.agents.yml
# reads these via ${VAR:-default} substitution; changing a value recreates the container.
export LLM_MODEL="$ORCH_MODEL_NAME"
export LLM_ORCHESTRATOR_MODEL="$ORCH_MODEL_NAME"
export VQA_MODEL_NAME="$VQA_MODEL_NAME_SEL"

info "Starting orchestrator container (orchestrator='$ORCH_MODEL_NAME', vqa='$VQA_MODEL_NAME_SEL')..."
# shellcheck disable=SC2086
docker compose -f docker-compose.agents.yml up -d $BUILD_FLAG

info "Waiting for container to be running..."
until [[ "$(docker inspect "$ORCHESTRATOR_CONTAINER" --format '{{.State.Running}}' 2>/dev/null)" == "true" ]]; do
    echo -n "."
    sleep 2
done
echo ""
ok "Container '$ORCHESTRATOR_CONTAINER' is running."

# ── Phase 5: VQA model load (Phi-4) — MANDATORY ─────────────────────────────
# The system is not "up" until the VQA model is loaded in GPU memory. This is a
# required phase: a failure here fails the whole startup.

echo ""
info "=== Phase 5: VQA model load (Phi-4) — required ==="
info "Loading Phi-4 into GPU VRAM — this may take several minutes on first run..."
info "(No timeout: the script waits however long the model needs.)"

vqa_output=$(docker exec -i "$ORCHESTRATOR_CONTAINER" python3 - <<'PYEOF'
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/vlm_src')
from videosearcher_agents.specialized.vqa_agent import _get_vlm
_get_vlm()
print('VQA_READY')
PYEOF
)

echo "$vqa_output" | grep -q "VQA_READY" \
    || die "VQA model failed to load — system is NOT ready. Check container logs:\n  docker logs $ORCHESTRATOR_CONTAINER"

ok "Phi-4 VQA model loaded and ready."

# ── Phase 6: Final readiness check + banner ──────────────────────────────────

echo ""
info "=== Phase 6: Final readiness verification ==="

FINAL_FAIL=false

_check_final_llama() {
    if llama_ready; then echo "ok"; else echo "FAIL"; fi
}
_check_final_mcp() {
    local port=$1
    if mcp_ready "$port"; then echo "ok"; else echo "FAIL"; fi
}
_check_final_seg() {
    if seg_ready; then echo "ok"; else echo "FAIL"; fi
}

L_STATUS=$(_check_final_llama)
S8000=$(_check_final_mcp 8000); S8001=$(_check_final_mcp 8001)
S8002=$(_check_final_mcp 8002); S8003=$(_check_final_mcp 8003)
S8004=$(_check_final_mcp 8004); S8005=$(_check_final_mcp 8005)
S8006=$(_check_final_mcp 8006); S8008=$(_check_final_seg)
S8009=$(_check_final_mcp 8009)
S8010=$(_check_final_mcp 8010); S8016=$(_check_final_mcp 8016); S8025=$(_check_final_mcp 8025)
VQA_STATUS="ok"   # Phase 5 already die()d if the VQA model failed to load

_mark() { [[ "$1" == "ok" ]] && echo "✓" || { echo "✗"; FINAL_FAIL=true; }; }

echo ""
info "Orchestrator model : $ORCH_MODEL_NAME (GPU, key='$ORCH_MODEL')"
info "VQA model          : $VQA_MODEL_NAME_SEL"
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   VideoSearcher AgenticAI — SYSTEM STATUS            ║"
echo "╠══════════════════════════════════════════════════════╣"
printf "║  llama-server    http://localhost:%-5s  %-2s           ║\n" "$LLAMA_PORT" "$(_mark "$L_STATUS")"
printf "║  ffmpeg-0        http://localhost:%-5s  %-2s           ║\n" "8000"        "$(_mark "$S8000")"
printf "║  librosa         http://localhost:%-5s  %-2s           ║\n" "8001"        "$(_mark "$S8001")"
printf "║  ffmpeg-1        http://localhost:%-5s  %-2s           ║\n" "8002"        "$(_mark "$S8002")"
printf "║  ffmpeg-2        http://localhost:%-5s  %-2s           ║\n" "8003"        "$(_mark "$S8003")"
printf "║  transcribe      http://localhost:%-5s  %-2s           ║\n" "8004"        "$(_mark "$S8004")"
printf "║  grep            http://localhost:%-5s  %-2s           ║\n" "8005"        "$(_mark "$S8005")"
printf "║  ffmpeg-frames   http://localhost:%-5s  %-2s           ║\n" "8006"        "$(_mark "$S8006")"
printf "║  vlm-segment     http://localhost:%-5s  %-2s           ║\n" "8008"        "$(_mark "$S8008")"
printf "║  video-segment   http://localhost:%-5s  %-2s           ║\n" "8009"        "$(_mark "$S8009")"
printf "║  web-search      http://localhost:%-5s  %-2s           ║\n" "8010"        "$(_mark "$S8010")"
printf "║  summarize       http://localhost:%-5s  %-2s           ║\n" "8016"        "$(_mark "$S8016")"
printf "║  scene-detect    http://localhost:%-5s  %-2s           ║\n" "8025"        "$(_mark "$S8025")"
printf "║  VQA (Phi-4)     in-process            %-2s           ║\n" "$(_mark "$VQA_STATUS")"
echo "╠══════════════════════════════════════════════════════╣"

if $FINAL_FAIL; then
    echo "║  ✗  One or more services failed final check.         ║"
    echo "╚══════════════════════════════════════════════════════╝"
    die "System not fully ready. Check service logs above."
fi

echo "║                                                      ║"
echo "║  ALL SYSTEMS READY — start the CLI with:            ║"
echo "║                                                      ║"
echo "║  docker exec -it agents-orchestrator \\              ║"
echo "║      python videosearcher_cli.py                     ║"
echo "║                                                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
