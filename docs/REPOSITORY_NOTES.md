# Repository preparation notes (28 October 2026)

Starting point: a detached ZIP delivery containing a nested source ZIP, an experimental result archive, two PDF presentations and a source PPTX. **No prior Git history was supplied**.

Curated changes:

1. Prepared a clean repo root, concise README and documentation index; retained the original comprehensive technical README under `TECHNICAL_GUIDE.md`, marked as historical.
2. Documented **unavailable private components** and their integration boundaries; removed the detached `.gitmodules` references because no gitlink objects or private source are part of this snapshot. Removed obsolete automatic submodule setup instructions from the active setup path.
3. Replaced the original launcher error with an explicit message that the external VLM framework is intentionally not distributed; no fake inference implementation was added.
4. Corrected the Grep evaluation metric to accept both `results: [{"match_count": N}]` and the original `results: {"match_count": N}` structure.
5. Preserved benchmark evidence and clarified distinction between best observed single-run scores and repeat means. Documentation uses default configuration `t0.80/p0.95/k40` for Qwen3.6-A3B.
6. Organized PDFs/PPTX and plot assets under `docs/`; the corrected benchmark PDF was regenerated from the supplied editable PPTX to avoid a stale configuration statement in the outer ZIP's PDF.
7. Excluded scratch JSON trials, model files and other local outputs from Git indexing; kept curated baseline table, tuning summaries/confirmation records and plots.
8. Omitted unrelated legacy course handout PDF and removed a global Git identity setup helper which unnecessarily read `.env` and changed global user config.
9. Updated GitHub Copilot context to the actual present codebase, and added CPU-only checks in GitHub Actions.

Known limitations remain intentional: missing private framework(s), missing model weights, external media and inability to reproduce the full GPU pipeline from this archive. There is **no license grant** and no claim of per-contributor code ownership.
