# VideoSearcher-AgenticAI — Architecture

## Architecture Principles

- Most specialized agents communicate with tool servers through MCP (Model Context Protocol) over HTTP; VQA is an explicit in-process exception.
- Each MCP tool server runs in its own Docker container with isolated dependencies.
- Agents never import tool code directly — they rely on MCP tool schemas auto-generated from type hints.
- The orchestrator and all specialized agents use the same local OpenAI-compatible LLM endpoint (served by `llama-server`).
- GPU-heavy components are isolated:
  - VQA runs inside the `agents-orchestrator` container (in-process, no MCP server).
  - Image/frame segmentation runs inside `vlm-segmentation-mcp-server`.
  - Video segmentation reuses the image/frame server instead of loading a second GPU model.

---

## High-Level Architecture

```
User (CLI)
  └─► Orchestrator (OpenAI Agents SDK + LiteLLM)
        │
        ├─► Core audio/text pipeline
        │     ├─► AudioExtractionAgent     → ffmpeg-0-mcp      :8000
        │     ├─► AudioAnalysisAgent       → librosa-mcp       :8001
        │     ├─► VideoSplittingAgent      → ffmpeg-1-mcp      :8002
        │     ├─► CompressionAgent         → ffmpeg-2-mcp      :8003
        │     ├─► TranscriptionAgent       → transcribe-mcp    :8004
        │     └─► InformationFinderAgent   → grep-mcp          :8005
        │
        ├─► Vision pipeline
        │     ├─► FrameExtractionAgent     → ffmpeg-frames-mcp :8006
        │     ├─► SegmentationAgent        → vlm-segmentation  :8008
        │     │                            → video-segmentation :8009
        │     └─► VqaAgent                → in-process Phi-4-reasoning-vision (GPU)
        │
        └─► Extension tools
              ├─► SummarizationAgent      → summarize-mcp     :8016
              ├─► WebSearchAgent          → web-search-mcp    :8010
              └─► SceneDetectAgent        → scene-detect-mcp  :8025
```

---

## Project File Structure

```
agents-container/
  Dockerfile
  videosearcher_cli.py              ← interactive CLI entry point
  videosearcher_agents/
    orchestrator/
      main_agent.py                 ← OrchestratorAgent, all handoff tool registrations
    specialized/
      audio_extraction_agent.py
      audio_analysis_agent.py
      video_splitting_agent.py
      compression_agent.py
      transcription_agent.py
      information_finder_agent.py
      frame_extraction_agent.py
      segmentation_agent.py
      vqa_agent.py
      summarize_agent.py
      web_search_agent.py
      scene_detect_agent.py
    llm_utils.py                    ← LiteLLM config resolution
    mcp_streamable_client.py        ← StreamableHTTP MCP client (with schema caching)
    run_logger.py                   ← orchestrator run trace logging

local-ai-service/
  run_agentic_system.sh             ← single entry point to start/stop the full stack
  stop_agentic_system.sh
  ADDING_MODELS.md                  ← how to add GGUF models to the registry
  models/                           ← 5-model orchestrator registry (--orchestrator-model KEY)
    gemma4/gemma-4-E2B-it-Q8_0.gguf       (gemma, default)
    phi4/Phi-4-mini-instruct-Q4_K_M.gguf  (phi4-mini)
    qwen25-7b/Qwen2.5-7B-Instruct-Q8_0.gguf
    gemma4-12b/gemma-4-12b-it-Q8_0.gguf
    gpt-oss-120b/gpt-oss-120b-mxfp4-*.gguf (3 shards, MXFP4)

tools/
  ffmpeg-0-mcp/                     ← audio extraction
  ffmpeg-1-mcp/                     ← video splitting
  ffmpeg-2-mcp/                     ← compression
  Librosa/                          ← silence detection
  Transcribe/                       ← Whisper STT
  Grep/                             ← transcript search
  ffmpeg-frames-mcp/                ← frame extraction
  vlm-segmentation-mcp/             ← GroundingDINO + SAM2
  video-segmentation-mcp/           ← per-frame video segmentation
  summarize-mcp/                    ← LLM transcript summarization
  web-search-mcp/                   ← DuckDuckGo / Tavily web search
  scene-detect-mcp/                 ← PySceneDetect shot boundaries
  (+ 16 inactive prototype tools)

tests/
  README.md
  conftest.py
  scenarios/
    base.py                         ← OrchestratorScenario, VisionSuccessCriteria, TestResult (+ ttft_ms)
    orchestrator_scenarios.py       ← SC-01..SC-09
    vision_scenarios.py             ← F-28..F-37 (f-35 fixtures; legacy SC-28..SC-37 commented out)
    extension_scenarios.py          ← SC-40 / SC-42 / SC-47  (22 scenarios total)
  orchestrator/
    runner.py
    run_tests.py
    test_orchestrator.py
  unit/
    test_vqa.py
    test_segmentation.py
    test_frame_extraction.py
    test_video_segmentation.py
  ground_truth/                     ← hand-labeled reference data
  results/                          ← JSON result files (gitignored)
    markdowns/                      ← rendered Markdown + comparison_<run_id>.md

scripts/
  run_model_comparison.py           ← benchmark orchestrator models × full suite (TTFT + energy)
  visualise_results.py              ← render a result JSON → Markdown
  aggregate_emissions.py            ← aggregate CodeCarbon emissions (cpu/gpu/ram)

data/                               ← shared Docker volume (./data → /data in all containers)
  raw/                              ← source videos (place inputs here)
  fixtures/                        ← test fixture videos
  runs/                            ← session-scoped output (one dir per orchestrator run)

logs/emissions/                     ← shared CodeCarbon volume (→ /app/emissions in every
                                       tool server + orchestrator); energy CSVs per project_name

VLMs-Testing-Framework/             ← external private dependency (not included): SegmentationPipelines, ModelInterfaces

docker-compose.yml                  ← MCP tool servers
docker-compose.agents.yml          ← agents-orchestrator + VQA GPU container
```

---

## Container Roles

### Local AI service (runs on host, not in Docker)

`local-ai-service/run_agentic_system.sh` starts `llama-server` (CUDA build) on the host:

```text
http://localhost:9000      ← accessible from host
http://host.docker.internal:9000/v1  ← accessible from Docker containers
```

This endpoint is OpenAI-compatible. Both the orchestrator and all specialized agents call it for LLM reasoning.

### Agent container (`agents-orchestrator`)

- Built from `agents-container/Dockerfile`
- Runs `videosearcher_cli.py` (interactive) or the test runner
- Hosts the orchestrator and all 12 specialized agents
- Connects to MCP servers via URLs from environment variables
- Runs VQA locally in-process (loads Phi-4-reasoning-vision-15B on GPU)
- Declared with `runtime: nvidia` in `docker-compose.agents.yml`

### MCP tool containers

- FastMCP servers exposing `POST /mcp` and `GET /health`
- One container per tool
- All tool outputs written to the shared `/data` volume
- The vlm-segmentation container also declares `runtime: nvidia` (GroundingDINO + SAM2)

### Shared data volume

All containers mount `./data` at `/data`. Intermediate artifacts (audio, transcripts, frames, masks) are referenced by absolute path through MCP calls — no serialization of file content over HTTP.

---

## Communication Flow

```
User request
  → CLI (videosearcher_cli.py)
     → OrchestratorAgent.process_request()
        → Runner.run(agent, prompt, max_turns=15)
           → LLM reasons → calls handoff tool
              → Specialized agent (e.g. TranscriptionAgent)
                 → mcp_streamable_client.call_tool()
                    → POST /mcp → JSON-RPC → MCP server handler
                       → tool executes (e.g. Whisper)
                          → writes output to /data/...
                             → returns output_path
```

For LLM reasoning:
```
Agent container → http://host.docker.internal:9000/v1
  → llama-server (host, CUDA)
     → GGUF model → response
```

---

## MCP Server Reference

| Container name | Port | Purpose |
|---|---:|---|
| `ffmpeg-0-mcp-server` | 8000 | Audio extraction from video |
| `librosa-mcp-server` | 8001 | Silence detection, timestamp generation |
| `ffmpeg-1-mcp-server` | 8002 | Video splitting at silence points |
| `ffmpeg-2-mcp-server` | 8003 | Video/audio compression |
| `speech-to-text-mcp-server` | 8004 | Whisper speech-to-text, timestamped output |
| `grep-mcp-server` | 8005 | 3-layer transcript search (literal → regex → fuzzy) |
| `ffmpeg-frames-mcp-server` | 8006 | Frame extraction from video time windows |
| `vlm-segmentation-mcp-server` | 8008 | Text-guided segmentation: GroundingDINO + SAM2 |
| `video-segmentation-mcp-server` | 8009 | Per-frame video segmentation (calls port 8008) |
| `summarize-mcp-server` | 8016 | LLM-based transcript summarization |
| `web-search-mcp-server` | 8010 | DuckDuckGo / Tavily web search |
| `scene-detect-mcp-server` | 8025 | PySceneDetect shot boundary detection |

---

## GPU Passthrough

Two containers need direct GPU access:

| Container | Compose file | Purpose |
|---|---|---|
| `agents-orchestrator` | `docker-compose.agents.yml` | In-process VQA inference (Phi-4, ~30 GB VRAM) |
| `vlm-segmentation-mcp-server` | `docker-compose.yml` | GroundingDINO + SAM2 (~3 GB VRAM) |

Both declare `runtime: nvidia` in their compose entry. This requires `nvidia-container-toolkit` installed on the host. No changes to `/etc/docker/daemon.json` are needed.

Verify GPU access:
```bash
docker exec agents-orchestrator python -c "import torch; print(torch.cuda.is_available())"
# Expected: True
```

---

## Data Flow (Pipeline Stages)

All outputs are written under the session root `/data/runs/{SESSION_ID}/` and named by stage:

```
/data/runs/{SESSION_ID}/
  audio_extraction/    ← .tar.gz archives (video.mp4 + audio.wav)
  audio_analysis/      ← .tar.gz archives (video.mp4 + timestamps.txt)
  video_clips/         ← folders of split .mp4 clips
  compression/         ← compressed .tar.gz archives
  transcription/       ← .tar.gz archives (video.mp4 + transcript.txt)
  search_results/      ← search_results.json
  frames/              ← JPEG frames from extract_frames
  segmentation/        ← masks, segmented images, segmented videos
  scenes/              ← scene boundary timestamps
  summarization/       ← summary.txt archives
  web_search/          ← search result JSON
  logs/orchestrator/   ← full run trace (JSON)
```

### Core pipeline paths

```
Audio/text search:
raw video → audio_extraction → transcription → search_results

Video splitting:
raw video → audio_extraction → audio_analysis → video_clips

Cross-modal QA:
raw video → transcription → search_results (timestamp) → frames → VQA answer

Image/frame segmentation:
image/frame → vlm-segmentation → masks OR segmented_image

Video segmentation:
video → video-segmentation → (calls vlm-segmentation per frame) → segmented_video OR segmented_frames
```

---

## Transcript Format

`tools/Transcribe/transcribe_main.py` writes one Whisper segment per line with a `[MM:SS.mmm]` prefix:

```text
[00:00.000] They're talking to you, random person, scrolling Facebook.
[00:08.600] The F-35 is a fifth-generation fighter jet.
[00:29.680] At an altitude of approximately 50,000 feet.
```

This format enables cross-modal chaining: the grep tool preserves the `[MM:SS.mmm]` prefix in every search match's `context` field, and the orchestrator parses it to drive `extract_frames_handoff` at the exact timestamp.

---

## Three-Layer Transcript Search

The grep MCP server uses three matching layers (each runs only if the previous returned zero results):

| Layer | Strategy | Catches |
|---|---|---|
| 1 | Lowercase + strip punctuation + collapse whitespace | Case, punctuation, spacing |
| 2 | Word-level regex with `\W+` between words | Token boundary differences |
| 3 | RapidFuzz `WRatio` fuzzy (threshold 80) | Typos, dropped/added words |

Matches include `match_type: "literal" | "fuzzy"` and (for fuzzy) `fuzzy_score`. The timestamp prefix is preserved in all layers.

---

## Segmentation Architecture

### Image/frame segmentation (port 8008)

`vlm-segmentation-mcp-server` exposes `segment_frame` and `segment_image`. Output modes:

| `output_mode` | Output | Use case |
|---|---|---|
| `masks` | Individual mask PNG paths + scores + labels + boxes | Internal pipelines, VQA, mask-based reasoning |
| `segmented_image` | One annotated image path + scores + labels + boxes | User-facing segmentation result |

### Video segmentation (port 8009)

`video-segmentation-mcp-server` exposes `segment_video`. It extracts frames via FFmpeg and calls port 8008 with `output_mode="segmented_image"` per frame. Output modes:

| `output_mode` | Output |
|---|---|
| `video` | Segmented `.mp4` |
| `frames` | Folder of segmented JPEG frames |

The video server does not load a second GPU model.

---

## VQA Compatibility Notes

Phi-4-reasoning-vision-15B ships custom code targeting older `transformers` versions. Three workarounds in `vqa_agent.py` patch these at load time:

1. `transformers >= 5.0, < 5.4` pinned in `requirements.txt` (prevents import errors)
2. `_ensure_phi4_cache_patched()` patches `past_key_values` handling (incompatible with `DynamicCache`)
3. `_phi4_predict()` uses tokenizer-level chat template (processor lacks `apply_chat_template`)

These workarounds are idempotent and self-remove if Microsoft updates the model.
