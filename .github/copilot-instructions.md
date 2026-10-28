# VideoSearcher repository guidance

This is a student research source snapshot, **not** a self-contained executable distribution.

- Current entry point: root `README.md`. Architecture: `docs/ARCHITECTURE.md`.
- Project limitations: `docs/PRIVATE_COMPONENTS.md` and `docs/REPRODUCIBILITY.md`.
- Present code lives in `agents-container/videosearcher_agents/`, `tools/`, `tests/`, `scripts/` and `local-ai-service/`.
- Treat `docs/TECHNICAL_GUIDE.md` as archived historical material; do not copy its private-submodule setup instructions.
- Do not hallucinate the missing private code, interfaces, fixtures, model files, or benchmark reruns.
- `tests/unit/test_metrics.py` is CPU-only; other integration tests need the full stack, external models, GPU and fixtures.
- Avoid exposing API keys or example `.env` values in commits; use `.env.example`.
