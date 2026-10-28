# Private components and unavailable source

This repository is built from the **provided project export only**. It intentionally does **not** contain the separately maintained libraries below. They are excluded for confidentiality, access control and third-party intellectual-property reasons. This absence is not a failed download and does not need to be remedied by contacting or cloning any private repository.

## 1. VLM testing framework — required for complete execution

The current code references a framework historically called `VLMs-Testing-Framework`. Its **source was not supplied** and therefore cannot be copied, rewritten, tested or licensed here.

Based solely on call sites, it provided:

- `ModelInterfaces`: model loading and prediction adapter used by the `VqaAgent` in [`agents-container/videosearcher_agents/specialized/vqa_agent.py`](../agents-container/videosearcher_agents/specialized/vqa_agent.py).
- `SegmentationPipelines`: image segmentation integration with GroundingDINO and SAM2 used in [`tools/vlm-segmentation-mcp/`](../tools/vlm-segmentation-mcp/).
- Shared setup at `/vlm_src` copied by two Dockerfiles: [`agents-container/Dockerfile`](../agents-container/Dockerfile) and [`tools/vlm-segmentation-mcp/Dockerfile`](../tools/vlm-segmentation-mcp/Dockerfile).

**Consequences:** Both images depend on `VLMs-Testing-Framework/src` at build time, and the VQA / segmentation workflows cannot be executed from this export. `local-ai-service/run_agentic_system.sh` checks this prerequisite and stops with an explicit error. No stub package is included because it could misleadingly imply working inference.

## 2. Repetti VLM-Testing framework — referenced by the research

An external project previously referenced as `Repetti_VLM-Testing` informed the Bayesian optimizer workflow. **Its source is not part of this export**, and no methods or files from it are claimed as original here. The included `scripts/tune_orchestrator.py` implements its own integration around a *separate* `dMALIBOO` optimizer dependency; its exact requirements are described in [hyperparameter tuning](HYPERPARAMETER_TUNING.md). Running this script still requires the main system and other research dependencies.

## What remains available

The repository contains its own orchestrator/agent code, MCP service implementations, test harness and **structured** ground-truth files, benchmark summaries, selected experimental records, plots, and presentations. Media-derived JPEG frames and historical TED transcript/derived JSON have been excluded from the public tree pending rights verification.

## What the documentation means

- **Available source** ≠ **self-contained runnable stack**.
- Documented experimental measurements are historical results, **not newly reproduced**.
- Private components are documented as dependencies and boundaries only; this repository does not grant rights to their source or to distribute it.
- Historical commands referencing Git submodules in the archived technical guide are **not** instructions for this detached export.
