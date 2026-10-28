# Setup / verification

## Supported from this source export: static code and metric tests

You need Python 3.10+ (3.11+ recommended for the local test command) and pytest. From the repository root:

```bash
python -m pip install pytest
python -m pytest -q tests/unit/test_metrics.py
python -m compileall -q agents-container scripts tests tools
```

These checks exercise evaluation metrics and syntax. **They do not launch the agentic video system or replicate experimental results.**

## Full stack: not self-contained

The original deployment requires at least:

1. A separately owned **VLM testing framework**, integrated at `/vlm_src` in the VQA and segmentation containers; not distributed here.
2. NVIDIA GPU/CUDA environment compatible with the local model server and VLMs; exact software versions and weights depend on the experiment environment.
3. The required local GGUF checkpoints, `llama-server`, Docker / Compose and third-party packages for each MCP service.
4. The experiment-specific media fixtures and access to any services invoked by extension tools.

The launcher will explicitly refuse a full-stack boot when the private framework source is absent. It would be misleading to publish a 'one-click install' script for this export. For service layout, Docker configuration and historical commands, see [Architecture](ARCHITECTURE.md) and [Technical guide](TECHNICAL_GUIDE.md).

## Configuration safety

Use [`agents-container/.env.example`](../agents-container/.env.example) as a reference. Keep actual `.env` files, API keys, GGUF weights, logs, and videos **out of Git**. Never place third-party/private source into this repository, even if local access is later granted.

## Results / presentations without GPU

See [Results](RESULTS.md), the measured CSV / curated tuning records under `tests/results/`, and the slides in [`docs/presentations/`](presentations/).
