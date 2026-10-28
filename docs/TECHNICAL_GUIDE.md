> **Archived guide (July 2026).** This was written for an environment with access to private dependencies.
> **This repository is a standalone source export, not an executable full-stack distribution.**
> For the current status and limitations, see [Private components](PRIVATE_COMPONENTS.md),
> [Setup](SETUP.md), and [Results](RESULTS.md). Historical commands below are preserved for reference only.

# VideoSearcher — Technical Guide (archived documentation)

A modular, multi-agent video search and analysis system. An orchestrator agent coordinates a fleet of specialised tool servers and a local vision-language model to answer natural-language questions about video content.

---

## Documentation


| Document                                                                 | Description                                                                                                                        |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **This file**                                                            | System overview, quick start, CLI usage, configuration                                                                             |
| [docs/ARCHITECTURE.md](ARCHITECTURE.md)                           | Complete architecture: all agents, MCP servers, data flow, GPU setup                                                               |
| [docs/ORCHESTRATOR_DESIGN.md](ORCHESTRATOR_DESIGN.md)             | Orchestrator pipeline chains, routing rules, example request traces                                                                |
| [docs/DESIGN_DECISIONS.md](DESIGN_DECISIONS.md)                   | Why the system is built the way it is — tradeoffs and rationale                                                                    |
| [docs/VLM_COMPONENTS.md](VLM_COMPONENTS.md)                       | All AI/ML models: Phi-4 VQA, GroundingDINO+SAM2, Faster-Whisper, Gemma                                                             |
| [docs/EXTENSION_TOOLS.md](EXTENSION_TOOLS.md)                     | Web search, transcript summarization, and scene detection agents                                                                   |
| [tests/README.md](../tests/README.md)                                     | Test scenarios, pass rates, how to run and extend tests                                                                            |
| [docs/MODEL_COMPARISON.md](MODEL_COMPARISON.md)                   | Orchestrator-LLM benchmark: the 14 metrics, how to run it, per-model tool-call fixes, results                                      |
| [docs/HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md)         | Bayesian tuning of the orchestrator's sampling params (temp/top-p/top-k): search space, how to run, wandb dashboards, crash-resume |
| [local-ai-service/ADDING_MODELS.md](../local-ai-service/ADDING_MODELS.md) | How to add new GGUF models to the local LLM registry (incl. tool-template overrides)                                               |
| [Repository notes](REPOSITORY_NOTES.md)                                       | Project development history: phases, bugs found, key decisions                                                                     |


---



## Technology Stack


| Area                 | Technologies used                                                                                                                                                                                                                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Agent orchestration  | Python, OpenAI Agents SDK, LiteLLM, Pydantic                                                                                                                                                                                                                                                              |
| LLM backends         | Local GGUF models via CUDA `llama.cpp` / `llama-server` on port 9000; optional OpenAI-compatible APIs such as OpenAI and Gemini                                                                                                                                                                           |
| Local model registry | `gemma`, `gemma4-12b`, `gpt-oss-120b`, `qwen36-27b`, `qwen36-a3b` + legacy-comparison models (`qwen25-7b`, `phi4-mini`, `mistral-small`, `falcon3-10b`, …) in `local-ai-service/run_agentic_system.sh` and `scripts/run_model_comparison.py` — see [docs/MODEL_COMPARISON.md](MODEL_COMPARISON.md) |
| Tool protocol        | Official Python MCP SDK (`mcp[cli]`) over Streamable HTTP (`/mcp`) with `/health` checks                                                                                                                                                                                                                  |
| Runtime isolation    | Docker Compose, per-tool Python containers, shared `./data` volume mounted as `/data`                                                                                                                                                                                                                     |
| Video/audio tools    | FFmpeg, `ffmpeg-python`, Librosa, Faster-Whisper for Whisper STT, Python regex/fuzzy matching for Grep                                                                                                                                                                                                    |
| Vision tools         | Phi-4 VQA in-process, GroundingDINO + SAM2 segmentation, PySceneDetect                                                                                                                                                                                                                                    |
| Development          | Git/GitHub, Python tests with pytest, optional GitHub Codespaces-compatible workflow                                                                                                                                                                                                                      |


---



## Orchestrator Model Comparison

We benchmarked 5 orchestrator LLMs — including both **Qwen3.6** releases — with the VQA model held constant at `Phi-4-reasoning-vision`, across the full 66-run suite (22 scenarios × 3 prompt levels; vision half on the F-35 fixtures with ground-truth scoring) on 14 metrics. Run `2026-07-01`. Methodology, per-model tool-call fixes, and the plug-and-play rule are in [docs/MODEL_COMPARISON.md](MODEL_COMPARISON.md); regenerate with `python3 scripts/visualise_results.py --table`.


| Metric                          | gemma-12b | gemma-E2B | gpt-oss-120b | qwen3.6-27B | qwen3.6-35B-A3B |
| ------------------------------- | --------- | --------- | ------------ | ----------- | --------------- |
| *Country*                       | USA       | USA       | USA          | China       | China           |
| *Size · Arch*                   | 12 d      | 5 d       | 120 MoE      | 27 d        | 35 MoE          |
| **1 · Overall %**               | 93.9      | 90.9      | 87.9         | 87.9        | **95.5**        |
| 2 · Handoff %                   | **97.0**  | 93.9      | 87.9         | 89.4        | 95.5            |
| 3 · Vision %                    | 100       | 88.9      | 88.9         | 100         | 100             |
| 4 · Result %                    | 92.6      | 92.6      | 96.3         | 96.3        | **100**         |
| 5 · Tool-call eff. (extra/task) | **0.09**  | 0.15      | 0.16         | 0.19        | 0.18            |
| 6 · Error-handling %            | 100       | 100       | 100          | 73.3        | 93.3            |
| 7a · Robust vague %             | 90.9      | 86.4      | 81.8         | 86.4        | 90.9            |
| 7b · Robust clear %             | 90.9      | 90.9      | 90.9         | 86.4        | 95.5            |
| 7c · Robust algo %              | 100       | 95.5      | 90.9         | 90.9        | 100             |
| 8 · Avg latency (s)             | 92.1      | 37.1      | 30.2         | 77.1        | 32.0            |
| 9 · TTFT (ms)                   | 432.9     | 90.9      | 204.3        | 426.3       | 134.8           |
| 10 · Throughput (tok/s)         | 16.8      | 83.2      | 53.5         | 11.6        | 74.1            |
| 11 · Tokens/task                | 14991     | 14991     | 12103        | 15962       | 16508           |
| 12 · Energy/run (kWh)           | 0.093     | 0.035     | 0.028        | 0.087       | 0.029           |
| 13 · Passes/kWh                 | 668       | 1701      | 2050         | 670         | **2163**        |
| 14 · Passes/min                 | 0.61      | 1.47      | 1.74         | 0.68        | **1.79**        |


An earlier 7-model comparison (incl. `qwen2.5-7B`, `mistral-24B`, `falcon3-10B`, and the `phi4-mini` baseline) ran on the legacy lionandgorilla vision suite and is preserved in [docs/MODEL_COMPARISON.md](MODEL_COMPARISON.md) — the fixtures and vision scoring changed, so those numbers are not directly comparable with this table.

**Takeaways:** **qwen3.6-35B-A3B** now leads *both* axes — best accuracy (95.5%, perfect Vision and Result) *and* best efficiency (2163 passes/kWh, 1.79 passes/min): a 35B MoE with only ~3B active parameters. **gemma-4-12b** remains the routing-precision leader (97% handoff, leanest tool-calling) but pays ~3× the energy of the MoEs. **gpt-oss-120b** stays the token/energy-lean option (12.1k tokens/task) but trails on handoff accuracy. The dense **qwen3.6-27B** is the surprise laggard: slowest decode on this GPU (11.6 tok/s), weakest error handling (73%) — its A3B MoE sibling beats it on every metric. Both Qwen3.6 models run with thinking disabled (`enable_thinking=false`) and need no chat-template override.

---



## Architecture

```
User (CLI)
    └── agents-orchestrator  [GPU · CUDA 12.4 · agents-container]
            │
            ├── audio/text pipeline ──► FFmpeg-0  (audio extraction  · :8000)
            │                          Librosa    (audio analysis     · :8001)
            │                          FFmpeg-1   (video splitting    · :8002)
            │                          FFmpeg-2   (compression        · :8003)
            │                          Transcribe (speech-to-text     · :8004)
            │                          Grep       (transcript search  · :8005)
            │
            ├── vision pipeline ──────► ffmpeg-frames      (frame extract             · :8006)
            │                           vlm-segmentation   (image/frame segmentation  · :8008)
            │                                             GroundingDINO + SAM2
            │                           video-segmentation (video segmentation        · :8009)
            │                                             calls vlm-segmentation
            │
            ├── VQA ─────────────────► Phi-4-reasoning-vision-15B
            │                           loaded on GPU inside the
            │                           orchestrator container itself
            │                           (no separate MCP server)
            │
            └── extension tools ──────► summarize-mcp   (transcript summary    · :8016)
                                        web-search-mcp  (DuckDuckGo / Tavily   · :8010)
                                        scene-detect-mcp (PySceneDetect cuts   · :8025)
```



### How Components Interact

```
System (Docker network + shared volume)
┌──────────────────────────────────────────────────────────────────────────┐
│ Agent container (agents-orchestrator) [GPU · CUDA 12.4]                  │
│ ┌───────────────────────────┐  ┌──────────────────────────────────────┐  │
│ │ videosearcher_cli.py      │  │ Orchestrator agent                   │  │
│ └───────────────────────────┘  └──────────────────────────────────────┘  │
│ ┌───────────────────────────┐  ┌──────────────────────────────────────┐  │
│ │ Audio extraction agent    │  │ Audio analysis agent                 │  │
│ └───────────────────────────┘  └──────────────────────────────────────┘  │
│ ┌───────────────────────────┐  ┌──────────────────────────────────────┐  │
│ │ Video splitting agent     │  │ Compression agent                    │  │
│ └───────────────────────────┘  └──────────────────────────────────────┘  │
│ ┌───────────────────────────┐  ┌──────────────────────────────────────┐  │
│ │ Transcription agent       │  │ Information finder agent             │  │
│ └───────────────────────────┘  └──────────────────────────────────────┘  │
│ ┌───────────────────────────┐  ┌──────────────────────────────────────┐  │
│ │ Frame extraction agent    │  │ Segmentation agent                   │  │
│ └───────────────────────────┘  └──────────────────────────────────────┘  │
│ ┌──────────────────────────────────────────────────────────────────────┐  │
│ │ VQA agent  (in-process GPU inference — no MCP server)                │  │
│ └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────┬─────────────────────────────────────────────────────┘
                     │ HTTP /mcp
┌────────────────────▼─────────────────────────────────────────────────────┐
│ Tool containers (MCP servers)                                             │
│ ┌───────────────────┐  ┌───────────────────┐  ┌─────────────────────┐   │
│ │ ffmpeg-0 (8000)   │  │ librosa (8001)    │  │ ffmpeg-1 (8002)     │   │
│ │ audio extraction  │  │ audio analysis    │  │ video splitting     │   │
│ └───────────────────┘  └───────────────────┘  └─────────────────────┘   │
│ ┌───────────────────┐  ┌───────────────────┐  ┌─────────────────────┐   │
│ │ ffmpeg-2 (8003)   │  │ transcribe (8004) │  │ grep (8005)         │   │
│ │ compression       │  │ speech-to-text    │  │ search             │   │
│ └───────────────────┘  └───────────────────┘  └─────────────────────┘   │
│ ┌───────────────────┐  ┌──────────────────────────────────────────────┐  │
│ │ ffmpeg-frames     │  │ vlm-segmentation (8008) [GPU]               │  │
│ │ (8006)            │  │ image/frame segmentation                     │  │
│ └───────────────────┘  │ GroundingDINO + SAM2                         │  │
│                        └──────────────────────────────────────────────┘  │
│ ┌──────────────────────────────────────────────────────────────────────┐  │
│ │ video-segmentation (8009)                                            │  │
│ │ video → segmented video OR segmented frames                          │  │
│ │ calls vlm-segmentation-mcp-server for each frame                     │  │
│ └──────────────────────────────────────────────────────────────────────┘  │
│ ┌───────────────────┐  ┌───────────────────┐  ┌─────────────────────┐   │
│ │ summarize (8016)  │  │ web-search (8010) │  │ scene-detect (8025) │   │
│ │ transcript summary│  │ DuckDuckGo/Tavily │  │ PySceneDetect cuts  │   │
│ └───────────────────┘  └───────────────────┘  └─────────────────────┘   │
└────────────────────┬─────────────────────────────────────────────────────┘
                     │ shared volumes: /data/fixtures (ro), /data/runs (rw)
┌────────────────────▼─────────────────────────────────────────────────────┐
│ Data layout (host ./data -> container /data)                              │
│ ┌───────────────────────────────────────────────────────────────────────┐ │
│ │ INPUTS  ./data/fixtures  -> /data/fixtures  (read-only)                │ │
│ │   raw/ · audio_extraction/ · transcription/ · video_clips/            │ │
│ │                                                                       │ │
│ │ OUTPUTS ./data/runs      -> /data/runs      (read-write)              │ │
│ │   <session-id>/  (e.g. 2026-06-21T09-50-16_test_0768/)               │ │
│ │     audio_extraction -> transcription -> search_results              │ │
│ │     audio_extraction -> audio_analysis -> video_clips                │ │
│ │     audio_extraction -> compression                                  │ │
│ │     frames -> segmentation (masks OR segmented image)                │ │
│ │     video  -> segmented video OR segmented frames                    │ │
│ │     scenes/ · summarization/ · web_search/ · logs/                   │ │
│ └───────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```



### MCP Tool Servers


| Service                         | Port | Purpose                                                                                                  | Base image                               |
| ------------------------------- | ---- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `ffmpeg-0-mcp-server`           | 8000 | Extract audio from video                                                                                 | `python:3.11-slim`                       |
| `librosa-mcp-server`            | 8001 | Silence detection / timestamps                                                                           | `python:3.11-slim`                       |
| `ffmpeg-1-mcp-server`           | 8002 | Split video into clips                                                                                   | `python:3.11-slim`                       |
| `ffmpeg-2-mcp-server`           | 8003 | Compress video/audio                                                                                     | `python:3.11-slim`                       |
| `speech-to-text-mcp-server`     | 8004 | Transcribe audio with Whisper. Writes per-segment `[MM:SS.mmm] text` timestamps for downstream chaining. | `python:3.11-slim`                       |
| `grep-mcp-server`               | 8005 | Search transcripts with normalization, flexible regex, and fuzzy fallback                                | `python:3.11-slim`                       |
| `ffmpeg-frames-mcp-server`      | 8006 | Extract frames from video time windows                                                                   | `python:3.11-slim`                       |
| `vlm-segmentation-mcp-server`   | 8008 | Text-guided image/frame segmentation. Can return masks or a segmented image.                             | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` |
| `video-segmentation-mcp-server` | 8009 | Text-guided video segmentation. Returns either a segmented video or segmented frames.                    | `python:3.11-slim`                       |
| `summarize-mcp-server`          | 8016 | LLM-based transcript summarization                                                                       | `python:3.11-slim`                       |
| `web-search-mcp-server`         | 8010 | Web search via DuckDuckGo (default), Tavily, Serper, or Brave                                            | `python:3.11-slim`                       |
| `scene-detect-mcp-server`       | 8025 | Shot/scene boundary detection via PySceneDetect                                                          | `python:3.11-slim`                       |




### Tools and Agent Mapping


| Tool / MCP server  | Port | Used by agent            | Typical input → output                      |
| ------------------ | ---- | ------------------------ | ------------------------------------------- |
| FFmpeg-0           | 8000 | Audio extraction agent   | raw video → extracted audio                 |
| Librosa            | 8001 | Audio analysis agent     | audio → silence timestamps                  |
| FFmpeg-1           | 8002 | Video splitting agent    | audio analysis → video clips                |
| FFmpeg-2           | 8003 | Compression agent        | audio/video → compressed outputs            |
| Transcribe         | 8004 | Transcription agent      | audio → timestamped transcript              |
| Grep               | 8005 | Information finder agent | transcript → search results                 |
| ffmpeg-frames      | 8006 | Frame extraction agent   | raw video/time window → frames              |
| vlm-segmentation   | 8008 | Segmentation agent       | image/frame → masks or segmented image      |
| video-segmentation | 8009 | Segmentation agent       | video → segmented video or segmented frames |
| in-process VQA     | —    | VQA agent                | frames/images → natural-language answer     |
| summarize          | 8016 | Summarization agent      | transcription archive → summary text        |
| web-search         | 8010 | Web search agent         | query string → search results               |
| scene-detect       | 8025 | Scene detection agent    | video → shot boundary timestamps            |




### Segmentation Output Modes

The segmentation system is split into two MCP servers.

#### Image/frame segmentation — port 8008

`vlm-segmentation-mcp-server` exposes:

- `segment_frame`
- `segment_image`

It supports two output modes:


| output_mode       | Behavior                                                                                      | Typical use                                                |
| ----------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `masks`           | Saves and returns individual mask PNG paths. This preserves the old behavior.                 | Internal pipelines, VQA, post-processing, object reasoning |
| `segmented_image` | Saves and returns one visual segmented image with overlay. It does not save individual masks. | User-facing image/frame segmentation                       |


Examples:

```text
Image/frame used as an intermediate step:
  output_mode="masks"
  returns: mask_paths, scores, labels, boxes

Image/frame requested as final segmentation:
  output_mode="segmented_image"
  returns: segmented_image_path, scores, labels, boxes
```



#### Video segmentation — port 8009

`video-segmentation-mcp-server` exposes:

- `segment_video`

It supports two output modes:


| output_mode | Behavior                                                                                  |
| ----------- | ----------------------------------------------------------------------------------------- |
| `video`     | Saves and returns only the final segmented video. Temporary extracted frames are removed. |
| `frames`    | Saves and returns only the segmented frames. Temporary source frames are removed.         |


The video server does not load another segmentation model. It extracts frames with FFmpeg and calls `vlm-segmentation-mcp-server` with `output_mode="segmented_image"` for each frame.

### VQA Agent (in-process)

The `VqaAgent` runs inside the orchestrator container — no MCP server. On the first VQA call the model is downloaded and loaded on GPU; subsequent calls reuse the in-memory instance.

- **Default model**: `microsoft/Phi-4-reasoning-vision-15B`
- **Configurable**: set `VQA_MODEL_NAME` in `docker-compose.agents.yml` and rebuild
- **Cache**: model weights persist in a named Docker volume (`vqa_hf_cache`)



### Data flow (flat `/data/` layout)

```
/data/
  raw/               ← drop source videos here
  audio_extraction/  ← extracted .wav files
  audio_analysis/    ← silence timestamps
  video_clips/       ← split clips
  compression/       ← compressed outputs
  transcription/     ← .txt transcripts
  search_results/    ← grep matches
  frames/            ← extracted JPEG frames
  segmentation/      ← image masks, segmented images, segmented videos, segmented frames
  logs/              ← orchestrator run logs (JSON)
```

Typical segmentation outputs:

```
/data/segmentation/
  <image_or_frame>_<run_id>/
    mask_000.png
    mask_001.png
    ...

  <image_or_frame>_<run_id>/
    segmented_image.png

  videos/
    <video>_<run_id>/
      <video>_segmented.mp4

    <video>_<run_id>/
      segmented_frames/
        frame_000001.jpg
        frame_000002.jpg
        ...
```

---



## Architecture Notes

The following components extend the base 6-server audio/text pipeline with vision capabilities.

### Frame Extraction Server — port 8006

`tools/ffmpeg-frames-mcp/` extracts JPEG frames from a video time window at a configurable FPS. It exposes the `extract_frames` MCP tool consumed by `FrameExtractionAgent`.

### VLM Image/Frame Segmentation Server — port 8008

`tools/vlm-segmentation-mcp/` replaces the old placeholder proxy on port 8007. It runs GroundingDINO + SAM2 directly via the `VLMs-Testing-Framework` library.

Given an image/frame path and a text query, for example `"person, car"`, it can return:

- per-object mask PNGs with confidence scores, labels, and boxes
- or one user-facing segmented image with overlay

This is controlled by `output_mode="masks"` or `output_mode="segmented_image"`.

### Video Segmentation Server — port 8009

`tools/video-segmentation-mcp/` segments full videos by extracting frames with FFmpeg and calling the image/frame segmentation server on each frame.

It supports:

- `output_mode="video"`: return only the segmented `.mp4`
- `output_mode="frames"`: return only the segmented frames

The server uses `vlm-segmentation-mcp-server` for the actual segmentation and does not load a second GPU model.

### VQA Agent

`agents-container/videosearcher_agents/specialized/vqa_agent.py` answers open-ended visual questions about frames using a local vision-language model. Unlike all MCP-based agents, this one does not delegate to a separate MCP server; inference runs inside the orchestrator process on the same GPU.

### VLMs-Testing-Framework submodule

The external VLM framework is not included in this standalone export. Both the segmentation server and the VQA agent expect imports from its `src/` directory.

When the upstream framework releases new commits, update with:

```bash
echo "Not available in this source export"
# then rebuild affected images, see Quick Start
```



### agents-container CUDA upgrade

The orchestrator container uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04` to support in-process GPU inference for VQA. Its `docker-compose.agents.yml` requests one NVIDIA GPU.

### Timestamp-preserving transcription

`tools/Transcribe/transcribe_main.py` writes Whisper output as one segment per line with a leading `[MM:SS.mmm]` timestamp, for example:

```text
[00:29.000] They're talking to you, random person, scrolling Facebook.
```

This is backwards compatible with plain-text search while allowing the orchestrator to parse timestamps and chain transcript matches into frame extraction, segmentation, and VQA. The transcription model (`Systran/faster-whisper-small`) is **multilingual with language auto-detection** — it transcribes the audio's actual language rather than forcing English.

### Flexible transcript search — grep v2

`tools/Grep/grep_main.py` chooses its matching strategy from the query:

1. **Real regex** — if the pattern contains regex metacharacters (`| . * + ? [] () \ …`) and compiles, it is used verbatim, so `engine|thrust` matches either term (the `case_sensitive` flag is honored).
2. **Word-level flexible match** — otherwise the query words are normalized (punctuation stripped, whitespace collapsed) and joined by `\W+`, so punctuation/whitespace variants still match.
3. **RapidFuzz** `WRatio` **fallback** — used when the above returns zero results.

Matches are tagged with `match_type: "literal" | "fuzzy"`, and fuzzy matches include a `fuzzy_score` field.

### Tool-server correctness

The 12 MCP tool servers underwent a correctness sweep (2026-07-17): ~20 verified fixes covering silent success-on-failure, path-quoting/`shell=true` injection, honoring the `case_sensitive`/regex grep contract, per-archive summary accumulation, multilingual transcription, and safe output-path handling. Each fix — with its failure scenario and rationale — is catalogued in [Repository notes](REPOSITORY_NOTES.md); rebuild the affected images (see [Useful rebuild commands](#useful-rebuild-commands)) to pick them up.

---



## Prerequisites

- Docker + Docker Compose v2
- NVIDIA Container Toolkit **registered as a Docker runtime** — the segmentation and orchestrator
  containers declare `runtime: nvidia`, so the toolkit must be installed **and** wired into the
  Docker daemon. On a fresh machine, after installing the toolkit:
  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  docker info | grep -i runtimes      # must list "nvidia"
  ```
  If `docker info` doesn't list `nvidia`, Phase 3 of the launcher fails with
  `unknown or invalid runtime name: nvidia`.
- NVIDIA GPU with ≥ 29 GB VRAM (required for the VQA agent — Phi-4 in bfloat16)
- ≥ 33 GB VRAM recommended when running segmentation (GroundingDINO + SAM2) and VQA simultaneously
- Segmentation-only workflows (no VQA): ≥ 8 GB VRAM
- The local LLM orchestrator (Gemma-4-E2B via llama-server) can run on CPU if GPU VRAM is limited
- An API key for the orchestrator LLM, for example Gemini or OpenAI (not required when using the local LLM path)
- **External private dependency source**, not included in this export; no setup procedure provided here

---



## Quick Start

> **Follow the steps in order.** On a clean machine, top-to-bottom, they take you from a fresh
> clone to a passing test with the **default `gemma` model**. Steps 2–5 put in place exactly
> what `run_agentic_system.sh` checks for at startup (llama-server binary → GGUF →
> `data/fixtures/` → `.env`), so by step 6 the launcher has everything it needs. For any other
> model, only step 3 changes (download a different GGUF) — see the registry table in step 6.



### 1. Source-snapshot limitation (supersedes original Git-submodule procedure)

The original delivery had links to external private repositories. This detached
snapshot contains **no private dependency source and no Git submodule objects**.
No public clone operation can restore the VLM framework. The system requires
independent authorization and provisioning by the relevant rightsholder; see
[PRIVATE_COMPONENTS.md](PRIVATE_COMPONENTS.md).

The remainder of this section is preserved as historical infrastructure
information; it is **not** a complete or currently runnable installation guide.
Use [SETUP.md](SETUP.md) for current supported verification steps.

### 2. Build llama.cpp (CUDA)

`run_agentic_system.sh` **always** starts a local llama-server (even when the orchestrator uses a
cloud API), so this is required. llama.cpp is gitignored; build it once (requires the host CUDA
toolkit — `nvcc`) — it provides the `llama-server` binary the launcher expects at
`llama.cpp/build-cuda/bin/llama-server` (override with `LLAMA_SERVER_BIN` if you build elsewhere):

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build-cuda -DGGML_CUDA=ON
cmake --build llama.cpp/build-cuda --config Release -j --target llama-server
```



### 3. Download the default orchestrator model

The default `gemma` key maps to `local-ai-service/models/gemma4/gemma-4-E2B-it-Q8_0.gguf`
(gitignored). Download it from HuggingFace:

```bash
pip install huggingface-hub --break-system-packages    # provides the `hf` CLI
hf download ggml-org/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q8_0.gguf \
    --local-dir local-ai-service/models/gemma4/
```

Every registry model is public on HuggingFace (official `google/…`, `Qwen/…`, `openai/…` weights
plus ready-made community GGUFs). The full per-key download table is in
[ADDING_MODELS.md](../local-ai-service/ADDING_MODELS.md#download-the-registry-models-from-huggingface);
`./local-ai-service/run_agentic_system.sh --list-models` prints the expected filename per key.



### 4. Download the input fixtures + create output dirs

The launcher requires `data/fixtures/raw/` to exist; the test suite and demos read the videos from
there. They ship as one public Google Drive archive (~161 MB, md5-verified):

```bash
pip install gdown --break-system-packages
RAW_ZIP="$(mktemp --suffix=.zip)"     # avoids clobbering /tmp/raw.zip on a shared machine
gdown 1BvCoujwL3YydSe8qLM_gwXUfoRhZVx0f -O "$RAW_ZIP"
unzip "$RAW_ZIP" -d data/fixtures/ -x "__MACOSX/*" "raw/DOWNLOADS.md"
rm -f "$RAW_ZIP"

# output + log dirs (the only other host paths the containers use)
mkdir -p data/runs logs/orchestrator logs/emissions
```

`data/fixtures` (read-only, at `/data/fixtures`) and `data/runs` (outputs, at `/data/runs`) are the
only host dirs bind-mounted into the containers. See
[`data/fixtures/raw/DOWNLOADS.md`](../data/fixtures/raw/DOWNLOADS.md) for the file list.

> On a shared machine, a hardcoded path like `/tmp/raw.zip` can already be owned by another user,
> which fails with `Permission denied` after the download completes. `mktemp` sidesteps this by
> always generating a fresh, uniquely-owned filename.



### 5. Configure environment

```bash
cp agents-container/.env.example agents-container/.env
```

`.env.example` ships with **every option commented out**, so a bare copy configures no provider.
For the default fully-local path, **uncomment Option D** in `agents-container/.env` (or paste the
block below):

```bash
LLM_ORCHESTRATOR_PROVIDER=openai-chat
LLM_ORCHESTRATOR_BASE_URL=http://host.docker.internal:9000/v1
LLM_ORCHESTRATOR_MODEL=gemma-4-E2B-it-Q8_0.gguf
LLM_ORCHESTRATOR_API_KEY=not-needed
LLM_MODEL=gemma-4-E2B-it-Q8_0.gguf
```

What matters here is the **provider** (`openai-chat` → the local llama-server on port 9000) and
`API_KEY=not-needed`. You don't have to keep the two model-name lines in sync when you switch
models: `run_agentic_system.sh` injects the served model name from `--orchestrator-model`
(default `gemma`), overriding them. (To use a cloud orchestrator instead — Gemini/OpenAI,
Options A/B — uncomment those and set the key; the local llama-server from step 2 still starts.)



### 6. Start the full stack

> `run_agentic_system.sh` **is the only supported way to start the system.** It starts the
> llama-server GPU process, all MCP containers, the orchestrator container, and pre-warms the VQA
> model — in the correct order, with health polling. Do not run `docker compose up` directly.

```bash
# FIRST run — build the images (this is where VLMs-Testing-Framework/src is needed):
./local-ai-service/run_agentic_system.sh --build

# later runs (images already built):
./local-ai-service/run_agentic_system.sh
```

It blocks until every service is healthy and prints `ALL SYSTEMS READY`.

> **First-ever start on a clean machine:** the segmentation checkpoints (GroundingDINO +
> SAM2, ~156 MB) are **baked into the image at build time** (step 6 `--build`), so there is
> **no runtime download** — Phase 3 only loads them onto the GPU (~1 min). The one genuine
> one-time download is:
> - **Phase 5** — the VQA model (Phi-4, **~38 GB**) is downloaded from HuggingFace into the
>   `vqa_hf_cache` volume — allow tens of minutes depending on bandwidth (the launcher waits
>   up to 2 h). On every later start it loads from the cached volume and pre-warming takes
>   **~2–3 min**.
>
> **Troubleshooting `11/12 servers ready` that never completes:** the SAM2 checkpoint is baked
> into the image, so this only happens if a **stale `vlm_weights` volume** from an image built
> *before* the weights were baked in is shadowing the new image (Docker never repopulates a
> volume that already has content — and `--fresh` keeps volumes). Reset the volume and rebuild:
> ```bash
> docker compose -f docker-compose.yml down
> docker volume rm guarnizoscappaturagarzoni_vlm_interaction_vlm_weights
> ./local-ai-service/run_agentic_system.sh --build
> ```
> `docker logs vlm-segmentation-mcp-server` prints the same instructions if it detects the
> checkpoint missing.
>
> **Missing submodule content:** Phase 1 checks `VLMs-Testing-Framework/src` before doing any
> build work and stops immediately with an explicit fix-it message if it's missing or empty —
> see step 1a/1b above. This replaces what used to be a much later, harder-to-diagnose failure
> deep inside `docker build`.
>
> **`llama-server` won't come up / script hangs on Phase 2 with no error:** the launcher writes
> `llama-server`'s own log to `/tmp/llama-server.log` by default. If that path isn't writable
> (shared machine, another user's file already there, read-only `/tmp`), the server process dies
> instantly on launch. Phase 2 now detects this and fails fast (and times out after 180s
> regardless), printing which log to check. Point it elsewhere with:
> ```bash
> LLAMA_LOG=/path/you/can/write.log ./local-ai-service/run_agentic_system.sh --build
> ```
>
> **On a shared machine, container names collide across separate clones of this repo:**
> `docker-compose.yml` uses fixed `container_name:` values (e.g. `summarize-mcp-server`), which
> are global to the Docker daemon, not scoped per Compose project — `COMPOSE_PROJECT_NAME` alone
> does not avoid this. If `docker compose up` reports `Conflict. The container name "..." is
> already in use`, another checkout of this repo already created containers under those names on
> the same host; free them with `docker rm <name>...` (safe once they're stopped) or coordinate
> with whoever owns them.

Common flags:

| Flag                       | Purpose                                                                  |
| -------------------------- | ------------------------------------------------------------------------ |
| `--build`                  | Rebuild Docker images before starting (use on the first run)             |
| `--fresh`                  | Tear down existing containers first (volumes kept)                       |
| `--orchestrator-model KEY` | Pick the local GGUF (default: `gemma`)                                   |
| `--vqa-model HF_ID`        | Override the VQA model (default: `microsoft/Phi-4-reasoning-vision-15B`) |
| `--list-models`            | Print the model registry                                                 |

**Orchestrator model registry** (select with `--orchestrator-model KEY`; for any non-default key,
redo step 3 with that model's GGUF):

| KEY                 | Model                          | Size / quant                                                  |
| ------------------- | ------------------------------ | ------------------------------------------------------------- |
| `phi4-mini`         | Phi-4-mini-instruct-Q4_K_M     | 2.4 GB · Q4_K_M                                               |
| `gemma` *(default)* | gemma-4-E2B-it-Q8_0            | 4.7 GB · Q8_0                                                 |
| `qwen25-7b`         | Qwen2.5-7B-Instruct-Q8_0       | 7.6 GB · Q8_0                                                 |
| `gemma4-12b`        | gemma-4-12b-it-Q8_0            | 12 GB · Q8_0                                                  |
| `gpt-oss-120b`      | gpt-oss-120b (MXFP4, 3 shards) | ~60 GB · MoE; use `--reasoning low`                           |
| `qwen36-27b`        | Qwen3.6-27B-Q4_K_M             | 17 GB · Q4_K_M; thinking disabled via `enable_thinking=false` |
| `qwen36-a3b`        | Qwen3.6-35B-A3B-Q4_K_M         | 21 GB · Q4_K_M · MoE (~3B active); thinking disabled          |

Four additional legacy-comparison keys are also registered — `llama31-8b`, `deepseek-v2-lite`,
`mistral-small`, `command-r` — for 11 total. The **VQA model**
(`microsoft/Phi-4-reasoning-vision-15B`, in-process) is held constant and is never an orchestrator
option — only the orchestrator LLM varies.

To stop the full stack:

```bash
./local-ai-service/stop_agentic_system.sh            # keep volumes
./local-ai-service/stop_agentic_system.sh --volumes  # also drop cached weights
```



### 7. Run a test with the default model

With the stack up (`ALL SYSTEMS READY`), run one scenario end-to-end to confirm the default model
works:

```bash
# one audio scenario at one prompt level (~1–2 min)
docker exec agents-orchestrator python tests/orchestrator/run_tests.py \
    --scenarios SC-01 --levels clear

# the full 66-run suite (22 scenarios × 3 levels)
docker exec agents-orchestrator python tests/orchestrator/run_tests.py
```

Per-scenario results are printed and the JSON result file is written under **`tests/results/`**
(bind-mounted, so it appears on the host); tool artifacts (frames, clips, audio) land under
`data/runs/`. See [`tests/README.md`](../tests/README.md) for the scenario list and scoring.

Or drive it interactively:

```bash
docker exec -it agents-orchestrator python videosearcher_cli.py
```

Example requests (inputs from `/data/fixtures/raw/...`, outputs to `/data/runs/...`):

```text
# Audio/text pipeline
> Extract audio from /data/fixtures/raw/sample_video.mp4 and transcribe it
> Find all mentions of "neural network" in /data/fixtures/raw/sample_video.mp4
> Split /data/fixtures/raw/sample_video.mp4 into clips at natural break points

# Vision pipeline
> Extract frames from /data/fixtures/raw/f-35.mp4 between 0 and 10 seconds at 2fps, then segment for "aircraft"
> Segment /data/fixtures/raw/f-35.mp4 for "aircraft" and return the segmented video
> What is happening in /data/fixtures/raw/f-35.mp4 at 5 seconds?

# Extension tools
> Summarize what is discussed in /data/fixtures/raw/tedtalk.mp4
> Detect all scene changes in /data/fixtures/raw/f-35.mp4
> Search the web for information about the F-35 aircraft
```

---



## Cross-modal example

The grep tool's timestamped context and the orchestrator's timestamp-aware instructions enable queries that chain audio, vision, segmentation, and VQA in a single prompt. This is exactly what test scenario **F-33** validates on the F-35 fixture:

```text
> Search /data/fixtures/raw/f-35.mp4 for the moment the word "helmet" is spoken, extract a frame at that timestamp, and describe what you see.
```

Handoff sequence:

1. `audio_extraction_handoff` → ffmpeg-0 extracts audio from the raw video.
2. `transcription_handoff` → Whisper writes a timestamped transcript.
3. `information_finder_handoff` → grep fuzzy-matches "helmet" and returns timestamped context.
4. Orchestrator parses the timestamp.
5. `extract_frames_handoff` → ffmpeg-frames extracts one or more frames around that timestamp.
6. `vqa_handoff` → local Phi-4 answers the visual question (here: the pilot holding a helmet).

Same idea with segmentation:

```text
> Find the moment in /data/fixtures/raw/f-35.mp4 when "helmet" is spoken, extract a frame there, segment the pilot from the background, and describe what the pilot is doing
```

In this case, segmentation can run in `output_mode="masks"` because the segmented object is an intermediate artifact used by the visual pipeline.

For a user-facing segmentation request:

```text
> Segment the person in /data/frames/frame_000001.jpg
```

The segmentation agent should call image segmentation with `output_mode="segmented_image"` and return `segmented_image_path`.

---



## Configuration Reference

All LLM settings live in `agents-container/.env`.


| Variable                    | Description                          | Example                               |
| --------------------------- | ------------------------------------ | ------------------------------------- |
| `GEMINI_API_KEY`            | Gemini API key                       | `AIza...`                             |
| `OPENAI_API_KEY`            | OpenAI API key                       | `sk-...`                              |
| `LLM_MODEL`                 | Model for all agents                 | `gemini/gemini-2.0-flash`             |
| `LLM_ORCHESTRATOR_MODEL`    | Override model for orchestrator only | `gemini/gemini-2.0-flash-thinking`    |
| `LLM_ORCHESTRATOR_BASE_URL` | Local LLM server URL                 | `http://host.docker.internal:9000/v1` |


MCP server URLs are set in `docker-compose.agents.yml`:


| Variable                            | Description                                | Default                                         |
| ----------------------------------- | ------------------------------------------ | ----------------------------------------------- |
| `FRAMES_MCP_SERVER_URL`             | Frame extraction MCP server                | `http://ffmpeg-frames-mcp-server:8006/mcp`      |
| `SEGMENTATION_MCP_SERVER_URL`       | Backward-compatible image segmentation URL | `http://vlm-segmentation-mcp-server:8008/mcp`   |
| `IMAGE_SEGMENTATION_MCP_SERVER_URL` | Image/frame segmentation MCP server        | `http://vlm-segmentation-mcp-server:8008/mcp`   |
| `VIDEO_SEGMENTATION_MCP_SERVER_URL` | Video segmentation MCP server              | `http://video-segmentation-mcp-server:8009/mcp` |


---



## Swapping the VQA Model

To use a different HuggingFace vision-language model:

1. Edit `VQA_MODEL_NAME` in `docker-compose.agents.yml`:
  ```yaml
   VQA_MODEL_NAME: your-org/your-model-name
  ```
2. Rebuild and restart the orchestrator:
  ```bash
   docker compose -f docker-compose.agents.yml build agents-orchestrator
   docker compose -f docker-compose.agents.yml up -d
  ```

Model weights are cached in the `vqa_hf_cache` Docker volume and reused across rebuilds.

---



## Testing Framework

An automated test suite lives in `tests/` with 22 scenarios — SC-01..SC-09 (orchestrator /
audio-text, `sample_video.mp4`), F-28..F-37 (vision, F-35 jet fixtures scored against
`tests/ground_truth/videos/f-35/`), SC-40/42/47 (extension tools) — across three prompt
difficulty levels (22 × 3 = 66 runs per model).


| Level       | Description                                             |
| ----------- | ------------------------------------------------------- |
| Vague       | Under-specified requests — tests orchestrator reasoning |
| Clear       | Explicit step-by-step instructions                      |
| Algorithmic | Structured, pipeline-oriented prompts                   |


Run all tests:

```bash
docker exec -it agents-orchestrator python tests/orchestrator/run_tests.py
```

Run specific scenarios:

```bash
docker exec -it agents-orchestrator python tests/orchestrator/run_tests.py --scenarios F-29,F-33
```

Full scenario catalog, how to add new tests, current pass rates, and VQA-specific guidance are documented in [tests/README.md](../tests/README.md).

Orchestrator run logs are saved as JSON in:

```text
data/logs/orchestrator/
```



### Benchmarking, TTFT, and energy

`scripts/run_model_comparison.py` benchmarks several orchestrator models against the full
suite in one run (it swaps `llama-server` per model — they share port 9000 — measures
**TTFT** via a streaming probe, and attributes per-model **energy**):

```bash
python3 scripts/run_model_comparison.py \
    --models gemma,gemma4-12b,gpt-oss-120b,qwen36-27b,qwen36-a3b --reasoning low
```

It writes a multi-section cross-model table to `tests/results/markdowns/comparison_<run_id>.md`.
The VQA model is held constant across all models, so the orchestrator LLM is the only variable.
Metrics gathered per model:


| Metric                         | What it captures                                                                         |
| ------------------------------ | ---------------------------------------------------------------------------------------- |
| **Overall pass**               | handoff_match AND vision AND result, over all 66 runs                                    |
| **Handoff / Vision / Result**  | the three correctness sub-checks                                                         |
| **Extra calls / task**         | avg agent handoffs beyond the expected chain — tool-call efficiency (lower = leaner)     |
| **Error-handling**             | pass rate on the reject/clarification scenarios (does it fail gracefully vs hallucinate) |
| **Robustness by prompt level** | overall pass rate at vague / clear / algorithmic (degradation under under-specification) |
| **Avg latency**                | mean wall-clock per scenario                                                             |
| **TTFT** (`ttft_ms`)           | time-to-first-token, model-serving latency (streaming probe)                             |
| **Throughput**                 | decode speed, tokens/sec (streaming probe)                                               |
| **Tokens / task**              | avg orchestrator token usage per scenario (`input/output/total_tokens` on each record)   |
| **Energy/run**                 | CodeCarbon energy attributed by run time-window (L1 LLM + L2 tools + L3 overhead)        |
| **Passes / kWh, Passes / min** | the accuracy-vs-cost efficiency frontier                                                 |


Render any single result JSON to readable Markdown:

```bash
python3 scripts/visualise_results.py tests/results/<run>.json
```

**Energy tracking (CodeCarbon)** is built in across three layers — all 12 MCP tool servers
(Layer 2), the orchestrator LLM and VQA model (Layer 1), and orchestration overhead
(Layer 3). Aggregate it with:

```bash
python3 scripts/aggregate_emissions.py
```

See [docs/CODECARBON.md](CODECARBON.md) for the full layer/project_name breakdown.

### Sampling-parameter tuning (Bayesian)

`scripts/tune_orchestrator.py` tunes the orchestrator LLM's `temperature` /
`top_p` / `top_k` with Bayesian optimization (d-MALIBOO, the engine from
`Repetti_VLM-Testing`), **maximizing the suite's Overall pass %**. Each trial
restarts `llama-server` with the candidate sampling flags and runs the full
66-run suite; progress and per-trial metrics stream live to **wandb**, and every
trial is checkpointed so a crashed or stopped run resumes with the same command.

```bash
# one model, 20 trials (5 random + 15 BO-guided)
.venv-tuning/bin/python scripts/tune_orchestrator.py --model qwen36-a3b \
    --n-initial 5 --n-iterations 15

# all registry models, sequential, detached-friendly
./scripts/tune_all.sh
```

Setup (venv, d-MALIBOO clone, wandb login), all flags, the resume/checkpoint
semantics, and the two-machine split procedure are documented in
[docs/HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md).

**Results — 100-trial campaign.** Winners derived from completed full-suite trials only (`scripts/tuning_report.py`; partial trials are
excluded — see `REPOSITORY_NOTES.md` F15). Baseline = llama-server defaults
(temp 0.8 / top-p 0.95 / top-k 40) from the comparison table above.


| Model           | Valid trials | Best sampling config                               | Tuned Overall % | Baseline % | Δ        |
| --------------- | ------------ | -------------------------------------------------- | --------------- | ---------- | -------- |
| gemma-E2B       | 97           | temp 0.4 · top-p 0.65 · top-k 10 *(7 configs tie)* | **95.5**        | 90.9       | **+4.6** |
| gpt-oss-120b    | 98           | temp 0.3 · top-p 0.98 · top-k 20                   | **92.4**        | 87.9       | **+4.5** |
| qwen3.6-35B-A3B | 107          | temp 0.9 · top-p 0.77 · top-k 20 *(7 configs tie)* | 92.4            | **95.5**   | −3.1     |


**Takeaways:** both winners sit at **low temperature with a small top-k**, and
qwen3.6-A3B is **insensitive to sampling** (107 configs span 87.9–92.4%). Wide
tie plateaus at each model's optimum mean the exact values matter less than the
low-temp/low-k *region*.

**Confirmation runs (repeat measurements) — the single-shot scores are noise-inflated.** Re-running each winner twice shows the suite has real run-to-run variance (stochastic orchestrator sampling → different handoff decisions), and because Bayesian optimization maximizes over the *max* of noisy
single evaluations, the reported winner scores are optimistically biased:


| Config                         | Single-shot | Repeat runs | Mean     | Δ    |
| ------------------------------ | ----------- | ----------- | -------- | ---- |
| gemma-E2B `t0.40/p0.65/k10`    | 95.5        | 93.9 · 92.4 | **93.2** | −2.3 |
| gpt-oss-120b `t0.30/p0.98/k20` | 92.4        | 84.8 · 87.9 | **86.3** | −6.1 |
| qwen3.6-A3B defaults           | 95.5        | 89.4 · 90.9 | **90.2** | −5.3 |


The *direction* holds (low-temp/small-top-k is a good region; gemma-E2B remains
the strongest orchestrator at ~93%), but absolute winner percentages should be
read as **mean ± ~3–6 points**, not exact. Reporting a tuned optimum from a
single noisy evaluation without repeat confirmation overstates the gain — the
+4.5-point "improvements" largely wash out once each config is measured more than
once. See [docs/HYPERPARAMETER_TUNING.md](HYPERPARAMETER_TUNING.md).

---



## Useful rebuild commands

> After rebuilding any image, use `run_agentic_system.sh --build` (or `--fresh --build`)
> to restart the stack cleanly. Only use the bare `docker compose` commands below when
> you need to rebuild a **single** container without restarting the whole stack.

Rebuild only the image/frame segmentation server:

```bash
docker compose -f docker-compose.yml build vlm-segmentation-mcp-server
docker compose -f docker-compose.yml up -d vlm-segmentation-mcp-server
```

Rebuild only the video segmentation server:

```bash
docker compose -f docker-compose.yml build video-segmentation-mcp-server
docker compose -f docker-compose.yml up -d video-segmentation-mcp-server
```

Rebuild the agent container:

```bash
docker compose -f docker-compose.agents.yml build agents-orchestrator
docker compose -f docker-compose.agents.yml up -d agents-orchestrator
```

Rebuild and restart everything (preferred):

```bash
./local-ai-service/stop_agentic_system.sh
./local-ai-service/run_agentic_system.sh --build
```

---

