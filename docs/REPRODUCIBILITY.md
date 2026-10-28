# Reproducibility statement

| Capability | From this export? | Evidence / limitation |
|---|---|---|
| Browse source code and architecture | Yes | Python, Docker, Markdown, scripts |
| Run CPU-only evaluation metric tests | Yes | `tests/unit/test_metrics.py` |
| Reinspect curated benchmark tables and plots | Yes | `tests/results/` and `docs/assets/` |
| Recreate all plots from fully preserved raw traces | Not guaranteed | Per-trial raw outputs were not retained in the curated Git package; tables and tuning histories are retained |
| Build all containers | **No** | Private VLM framework required by two Dockerfiles |
| Perform VQA/segmentation inference | **No** | Private framework + model weights + GPU |
| Rerun full 66-task experiments | **No** | Private framework, trained weights, original runtime and fixture media needed |

## Experimental conditions (as reported)

- Full suite: 22 scenarios × three prompt levels = 66 per model.
- VQA held constant while orchestrator LLM varied.
- Model-serving used local GGUF weights and `llama-server`, with CUDA on DGX GB10-class hardware according to slides.
- Energy measurement used CodeCarbon with an experiment-specific multi-layer methodology.
- Separate Bayesian optimization and two-repeat confirmation phase; see [Results](RESULTS.md).

**No experiments were rerun during repository preparation.** The included data represent prior recorded observations rather than newly verified system performance.

## Fixture provenance

The original `data/fixtures/raw/DOWNLOADS.md` documents a source archive and fixture list. Fixture videos have **not** been copied into this Git snapshot; their usage and redistribution rights should be checked before downloading or rehosting them.
