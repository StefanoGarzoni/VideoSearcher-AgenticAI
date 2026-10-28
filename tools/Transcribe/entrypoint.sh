#!/bin/bash
set -e

echo "=========================================="
echo "Faster Whisper MCP Server"
echo "=========================================="

# Environment configuration
HF_CACHE="${HF_HOME:-/app/.cache/huggingface}"
# Must match the model actually loaded in transcribe_main.py ("small",
# multilingual) — the check used to validate "small.en", so it reported a
# green cache hit while the real model was absent, then failed offline.
MODEL_NAME="models--Systran--faster-whisper-small"
MODEL_PATH="${HF_CACHE}/hub/${MODEL_NAME}"

echo "[*] Checking Hugging Face model cache..."
echo "    Cache directory: ${HF_CACHE}"

# Check if model exists in cache
if [ -d "${MODEL_PATH}" ]; then
    echo "[✓] Model found in cache: ${MODEL_NAME}"
    echo "    Location: ${MODEL_PATH}"
    ls -lh "${MODEL_PATH}/snapshots" 2>/dev/null || echo "    (snapshot details unavailable)"
else
    echo "[!] Model NOT found in cache"
    echo ""
    echo "    The Faster Whisper model will be downloaded on first use."
    echo "    This requires internet access and may take a few minutes."
    echo ""
    echo "    To avoid this, mount a pre-populated cache:"
    echo "    volumes:"
    echo "      - ~/.cache/huggingface:/app/.cache/huggingface"
    echo ""
fi

echo "[*] Starting MCP server on port ${PORT:-8004}..."
echo "=========================================="

# Start the server
exec uvicorn transcribe_server:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8004}" \
    --log-level "${UVICORN_LEVEL:-info}" \
    ${UVICORN_ACCESS_LOG:+--access-log}
