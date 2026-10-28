#!/usr/bin/env bash
# stop_agentic_system.sh — stop the full VideoSearcher AgenticAI stack and free
# every resource it uses: the orchestrator + all MCP containers, the llama-server
# process, the GPU VRAM they hold, and TCP port 9000.
#
# This is the counterpart to run_agentic_system.sh.
#
# Usage:
#   ./stop_agentic_system.sh [--volumes]
#
#   --volumes   Also remove the docker named volumes (vqa_hf_cache, vlm_weights).
#               WARNING: this deletes cached model weights — the next start will
#               re-download / re-build them. Default keeps volumes (and ./data).

set -Eeuo pipefail

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_PORT=9000
ORCHESTRATOR_CONTAINER="agents-orchestrator"

# ── Argument parsing ─────────────────────────────────────────────────────────

VOL_FLAG=""
REMOVE_VOLUMES=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --volumes) REMOVE_VOLUMES=true; VOL_FLAG="--volumes"; shift ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RESET='\033[0m'
info() { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()   { echo -e "${GREEN}[ OK ]${RESET}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET}  $*"; }

port_in_use() {
    local port=$1
    if command -v ss >/dev/null 2>&1; then
        ss -ltnH "sport = :${port}" 2>/dev/null | grep -q . && return 0 || return 1
    fi
    curl -sf --max-time 2 "http://localhost:${port}/health" >/dev/null 2>&1
}

# ── Pre-flight ───────────────────────────────────────────────────────────────

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found in PATH." >&2; exit 1; }
cd "$PROJECT_ROOT"

echo ""
info "=== Stopping VideoSearcher AgenticAI stack ==="
$REMOVE_VOLUMES && warn "--volumes set: named volumes (cached model weights) will be removed."

# ── Phase 1: Orchestrator container ──────────────────────────────────────────

echo ""
info "=== Phase 1: Orchestrator container ==="
docker compose -f docker-compose.agents.yml down --remove-orphans $VOL_FLAG 2>&1 | sed 's/^/  /' || warn "agents compose down reported an issue."
ok "Orchestrator stopped."

# ── Phase 2: MCP tool servers ────────────────────────────────────────────────

echo ""
info "=== Phase 2: MCP tool servers ==="
docker compose -f docker-compose.yml down --remove-orphans $VOL_FLAG 2>&1 | sed 's/^/  /' || warn "MCP compose down reported an issue."
ok "MCP servers stopped."

# Sweep up any leftover containers from THIS checkout only — scoped by the
# com.docker.compose.project.working_dir label (set to $PROJECT_ROOT), never by
# name. All service/container names in this repo are fixed (not namespaced by
# Compose project), so a second checkout of this repo on the same host — e.g. a
# different user's clone — produces containers with the IDENTICAL names. A
# name-only sweep would forcibly remove that other checkout's containers too,
# including ones actively running. Scoping by working_dir makes that impossible:
# this only ever touches containers this exact directory created.
LEFTOVER=$(docker ps -a --filter "label=com.docker.compose.project.working_dir=$PROJECT_ROOT" --format '{{.Names}}' || true)
if [[ -n "$LEFTOVER" ]]; then
    warn "Removing leftover containers from this checkout ($PROJECT_ROOT):"
    echo "$LEFTOVER" | sed 's/^/    /'
    echo "$LEFTOVER" | xargs -r docker rm -f >/dev/null 2>&1 || true
fi

# Remove the shared bridge network this stack creates (docker-compose.yml owns
# it; docker-compose.agents.yml only references it as `external: true`, so
# neither `down` above reliably removes it). Only remove it if nothing is
# actually attached — if another checkout's containers are currently using it,
# leave it alone rather than pull the network out from under a live stack.
NETWORK_NAME="videosearcher_agenticai_videosearcher"
if docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
    ATTACHED=$(docker network inspect "$NETWORK_NAME" --format '{{len .Containers}}' 2>/dev/null || echo 0)
    if [[ "$ATTACHED" == "0" ]]; then
        docker network rm "$NETWORK_NAME" >/dev/null 2>&1 \
            && ok "Removed shared network '$NETWORK_NAME' (no containers attached)." \
            || warn "Could not remove network '$NETWORK_NAME' — leaving it."
    else
        warn "Network '$NETWORK_NAME' still has $ATTACHED container(s) attached — leaving it (likely another checkout is using it)."
    fi
fi

# ── Phase 3: llama-server ────────────────────────────────────────────────────
# LLAMA_PORT (9000) is a fixed host port, same for every checkout of this repo —
# there's no per-checkout namespacing for a native (non-Docker) process. So this
# phase is careful to only kill a llama-server this exact checkout launched,
# identified by its absolute binary path, and never blindly kill "whatever is on
# port 9000" — that could be a different user's live server.

echo ""
info "=== Phase 3: llama-server (port $LLAMA_PORT) ==="

OUR_LLAMA_BIN="${LLAMA_SERVER_BIN:-$PROJECT_ROOT/llama.cpp/build-cuda/bin/llama-server}"
# Match on the exact absolute path, not a bare substring — "build-cuda/bin/llama-server"
# alone would also match another checkout's binary at a different absolute path.
pkill -f "^${OUR_LLAMA_BIN}" 2>/dev/null || true
sleep 2

if port_in_use "$LLAMA_PORT"; then
    # Something is still on the port. Only kill it if it's demonstrably ours
    # (same binary path) — otherwise it's most likely another user's live server.
    HOLDER_PID=""
    if command -v ss >/dev/null 2>&1; then
        HOLDER_PID=$(ss -ltnpH "sport = :${LLAMA_PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
    fi
    HOLDER_EXE=""
    [[ -n "$HOLDER_PID" ]] && HOLDER_EXE=$(readlink -f "/proc/$HOLDER_PID/exe" 2>/dev/null || true)

    if [[ -n "$HOLDER_EXE" && "$HOLDER_EXE" == "$(readlink -f "$OUR_LLAMA_BIN" 2>/dev/null)" ]]; then
        kill "$HOLDER_PID" 2>/dev/null || true
        sleep 2
    elif [[ -n "$HOLDER_PID" ]]; then
        warn "Port $LLAMA_PORT is held by pid $HOLDER_PID ($HOLDER_EXE), which is NOT this checkout's llama-server."
        warn "Leaving it running — it's most likely another user's active server. Not killing it."
    fi
fi

if port_in_use "$LLAMA_PORT"; then
    if [[ -n "${HOLDER_EXE:-}" && "$HOLDER_EXE" != "$(readlink -f "$OUR_LLAMA_BIN" 2>/dev/null)" ]]; then
        info "Port $LLAMA_PORT still in use by another user's process — this is expected, not an error."
    else
        warn "Port $LLAMA_PORT is STILL in use. Inspect manually (ss -ltnp | grep $LLAMA_PORT)."
    fi
else
    ok "llama-server stopped; port $LLAMA_PORT is free."
fi

# ── Phase 4: Verification ────────────────────────────────────────────────────

echo ""
info "=== Phase 4: Verification ==="
REMAIN=$(docker ps --filter "label=com.docker.compose.project.working_dir=$PROJECT_ROOT" --format '{{.Names}}' || true)
if [[ -z "$REMAIN" ]]; then
    ok "No containers from this checkout running."
else
    warn "Still running (from this checkout):"; echo "$REMAIN" | sed 's/^/    /'
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)
    info "GPU memory in use now: ${USED:-unknown} (model VRAM is freed once processes exit)."
fi

echo ""
ok "Stack stopped. Resources freed (containers, llama-server, port $LLAMA_PORT)."
$REMOVE_VOLUMES && ok "Named volumes removed." || info "Named volumes and ./data kept. Use --volumes to also drop cached weights."
echo ""
