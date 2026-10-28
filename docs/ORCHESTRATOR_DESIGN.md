# Orchestrator Agent Design

## Overview

The orchestrator is the brain of the system. It receives natural-language user requests, reasons about which specialized agents to invoke and in which order, and delegates work via handoff tools. It does not execute tool logic directly.

The system uses:
- OpenAI Agents SDK (Handoff pattern) for multi-agent coordination
- LiteLLM for provider-agnostic LLM calls
- A local `llama-server` endpoint (port 9000) as the default LLM backend
- MCP StreamableHTTP for agent→tool communication

---

## LLM Configuration

The orchestrator LLM is configured via environment variables in `agents-container/.env`:

```bash
# Local LLM (default)
LLM_ORCHESTRATOR_PROVIDER=openai-chat
LLM_ORCHESTRATOR_BASE_URL=http://host.docker.internal:9000/v1
LLM_ORCHESTRATOR_MODEL=gemma-4-E2B-it-Q8_0.gguf
LLM_ORCHESTRATOR_API_KEY=not-needed

# Cloud LLM (alternative — change these in .env)
# LLM_MODEL=gemini/gemini-2.0-flash
# GEMINI_API_KEY=AIza...
```

The model is started by `local-ai-service/run_agentic_system.sh` and serves an OpenAI-compatible API on port 9000. The Docker container reaches it via `http://host.docker.internal:9000/v1`.

`gemma-4-E2B-it-Q8_0.gguf` shown above is the **default**, but `run_agentic_system.sh`
ships a **multi-model registry** — `gemma`, `gemma4-12b`, `gpt-oss-120b`, `qwen36-27b`,
`qwen36-a3b`, plus legacy keys (`phi4-mini`, `qwen25-7b`, …) — selectable at launch with `--orchestrator-model KEY` (it exports
`LLM_ORCHESTRATOR_MODEL`, overriding `.env`). `--list-models` prints the registry. The VQA
model (`microsoft/Phi-4-reasoning-vision-15B`) is held constant and is never an
orchestrator option.

### Benchmarking & metrics

`scripts/run_model_comparison.py` swaps each registry model through port 9000 and runs the
full 66-run suite per model, writing a multi-section table to
`tests/results/markdowns/comparison_<run_id>.md`. Metrics per model:

- **Correctness** — overall pass, handoff match, vision pass, result pass.
- **Tool-call efficiency** — avg extra agent handoffs beyond the expected chain.
- **Error-handling** — pass rate on the reject/clarification scenarios (graceful failure vs hallucination).
- **Robustness by prompt level** — overall pass rate at vague / clear / algorithmic.
- **Latency** — avg wall-clock, **TTFT** (`ttft_ms`, time-to-first-token) and **throughput** (tokens/sec), both via a streaming probe to llama-server.
- **Tokens/task** — `input/output/total_tokens` from the OpenAI Agents SDK usage, per scenario.
- **Energy** — CodeCarbon `energy_consumed` attributed by run time-window (L1 LLM + L2 tools + L3 overhead).
- **Efficiency frontier** — passes per kWh and passes per minute.

`scripts/visualise_results.py` renders any single result JSON to Markdown. Energy is tracked
end-to-end via CodeCarbon — the orchestrator request itself is measured as
`orchestrator_request` (its GPU energy ≈ the LLM, its CPU+RAM ≈ orchestration overhead);
see [`CODECARBON.md`](CODECARBON.md).

---

## Handoff Tools

The orchestrator has 14 registered handoff tools. Each tool wraps a specialized agent via `agent.as_tool()` or a `@function_tool` decorator:

### Core audio/text pipeline

| Tool name | Input | Output |
|---|---|---|
| `audio_extraction_handoff` | `.mp4` path | archive (video.mp4 + audio.wav) |
| `audio_analysis_handoff` | archive from audio_extraction | archive (video.mp4 + timestamps.txt) |
| `video_splitting_handoff` | archive from audio_analysis | folder of `.mp4` clips |
| `compression_handoff` | `.mp4` path OR folder of clips | archive(s) (compressed .mp4 + .wav) |
| `transcription_handoff` | archive from audio_extraction | archive (transcript.txt) |
| `information_finder_handoff` | archive from transcription + search query | search_results.json |

### Vision pipeline

| Tool name | Input | Output |
|---|---|---|
| `extract_frames_handoff` | video path + time window | `first_frame_path` + output folder |
| `segmentation_handoff` | image/frame/video path + text query + `output_mode` | masks / segmented image / segmented video |
| `vqa_handoff` | `frame_path` + `question` | text answer |

### Extension tools

| Tool name | Input | Output |
|---|---|---|
| `summarize_handoff` | transcription archive path | archive with summary.txt |
| `web_search_handoff` | search query string | ranked results (title, url, snippet) |
| `scene_detect_handoff` | `.mp4` path or archive | scene boundary timestamps |

### Control tools

| Tool name | When to use |
|---|---|
| `reject_handoff` | Call as the VERY FIRST action when the initial request is impossible (wrong file format, unsupported operation). Never mid-pipeline. |
| `request_clarification_handoff` | Call as the VERY FIRST action ONLY when no file path is provided and the intent is irrecoverably ambiguous. |

---

## Pipeline Chains

The orchestrator system prompt defines named chains that the LLM should recognize and execute:

### Chain A — Timestamps only
```
audio_extraction_handoff → audio_analysis_handoff
```
Use when: "find timestamps", "detect pauses", "sentence breaks".

### Chain B — Split video into clips
```
audio_extraction_handoff → audio_analysis_handoff → video_splitting_handoff
```
Use when: "split into clips", "clip at silence".

### Chain C — Search keyword in pre-split clips
```
compression_handoff → transcription_handoff → information_finder_handoff
```
Use when: input is already a folder of clips.

### Chain D — Full pipeline (split + compress + search)
```
audio_extraction_handoff → audio_analysis_handoff → video_splitting_handoff
  → compression_handoff → transcription_handoff → information_finder_handoff
```
Use when: "full pipeline", "run all steps", "split by sentence then search".

### Chain E — Quick search (no splitting)
```
audio_extraction_handoff → transcription_handoff → information_finder_handoff
```
Use when: "quick search", "just transcribe and find".

### Chain F — Visual question answering on a video
```
extract_frames_handoff → vqa_handoff(frame_path=first_frame_path)
```
Or with segmentation:
```
extract_frames_handoff → segmentation_handoff(image_path=first_frame_path, output_mode=...)
```

### Chain G — Cross-modal: find phrase → frame → answer
```
audio_extraction_handoff → transcription_handoff
  → information_finder_handoff (returns [MM:SS.mmm] timestamp)
  → extract_frames_handoff (t_start = parsed timestamp)
  → vqa_handoff(frame_path=first_frame_path)
```
Use when: "What does X look like when the speaker says Y?".

### Chain H — Segmentation as intermediate step → VQA
```
segmentation_handoff(output_mode="masks") → vqa_handoff
```

### Extension Chain I — Summarize transcript
```
audio_extraction_handoff → transcription_handoff → summarize_handoff
```

### Extension Chain II — Web fact-check
```
[transcription_handoff →] web_search_handoff(query string)
```

---

## Routing Rules

### Path normalization
```
data/raw/demo.mp4 → /data/raw/demo.mp4
```
Always normalize relative paths to absolute before passing to tools.

### Segmentation output mode selection

| User asks for | `output_mode` |
|---|---|
| Bounding boxes, labels, positions, object locations | `"masks"` |
| Visual annotated image (user-facing) | `"segmented_image"` |
| Segmented video | `"video"` |
| Segmented frames of a video | `"frames"` |
| Internal step before VQA | `"masks"` |

Never use `output_mode="both"` — not supported by the video server.

### Cross-modal timestamp parsing

Parse `[MM:SS.mmm]` from `information_finder_handoff` result's `context` field:
```
[00:33.500] → t_start = 33.5 seconds
```
Pass `t_start` (and `t_end = t_start + 1`, `fps = 1`) to `extract_frames_handoff`.

### VQA input

- User provides image path (`.jpg`, `.jpeg`, `.png`) + visual question → call `vqa_handoff` directly.
- Input is a video → call `extract_frames_handoff` first, then pass `first_frame_path` to `vqa_handoff`.
- Input is an audio file (`.wav`, `.mp3`) → call `reject_handoff`. Audio has no visual content.

### PATH PASS-THROUGH RULE

Every tool result contains `output_path=<absolute_path>`. The orchestrator must:
1. Extract the exact path by reading after `output_path=`
2. Copy it verbatim to the next tool call
3. Never construct, abbreviate, or modify paths

Each pipeline step produces a new path with a new unique hash. Guessing or reconstructing paths from previous step hashes is always wrong.

---

## Example Request Traces

### Audio/text search
```
"Find mentions of AI in /data/raw/lecture.mp4"
  → audio_extraction_handoff(video_path=/data/raw/lecture.mp4)
  → transcription_handoff(archive=<output_path from above>)
  → information_finder_handoff(archive=<output_path from above>, query="AI")
```

### Cross-modal visual question
```
"What colour is the jacket of the person who says 'scrolling Facebook'?"
  → audio_extraction_handoff(video_path=/data/raw/tedtalk.mp4)
  → transcription_handoff(archive=<output_path>)
  → information_finder_handoff(archive=<output_path>, query="scrolling Facebook")
     → result context: "[00:33.500] ...scrolling Facebook..."
  → extract_frames_handoff(video_path=/data/raw/tedtalk.mp4, t_start=33.5, t_end=34.5, fps=1)
     → result: first_frame_path=/data/runs/.../frames/tedtalk_abc123/frame_000001.jpg
  → vqa_handoff(frame_path=<first_frame_path>, question="What colour is the jacket?")
```

### Segment a frame (user-facing)
```
"Segment the person in /data/frames/frame_000001.jpg"
  → segmentation_handoff(image_path=/data/frames/frame_000001.jpg,
                         text_query="person",
                         output_mode="segmented_image")
```

### Segment a video
```
"Segment /data/raw/demo.mp4 for person and return the segmented video"
  → segmentation_handoff(video_path=/data/raw/demo.mp4,
                         text_query="person",
                         output_mode="video")
```

### Summarize a video
```
"Summarize what is said in /data/raw/lecture.mp4"
  → audio_extraction_handoff(video_path=/data/raw/lecture.mp4)
  → transcription_handoff(archive=<output_path>)
  → summarize_handoff(archive=<output_path>)
```

---

## Constraints

- Call EXACTLY ONE tool per turn. Wait for the result before calling the next.
- Never ask the user for clarification mid-pipeline.
- Never retry a failed tool call. Report the error and stop.
- Never call the same agent twice for the same input.
- Always return exact output file paths in the final answer.
- Execute ALL steps of the chosen chain. Never stop mid-chain unless an agent errors.

---

## Session Isolation

Each call to `OrchestratorAgent.__init__()` creates a unique session root:
```
/data/runs/{YYYY-MM-DDTHH-MM-SS}_{hex6}/
  audio_extraction/, audio_analysis/, video_clips/, compression/,
  transcription/, search_results/, frames/, segmentation/,
  scenes/, summarization/, web_search/, logs/orchestrator/
```

The `SESSION_OUTPUT_ROOT` environment variable is set by the first orchestrator instance and reused by subsequent instances in the same process (used by the test runner to share a single session across scenarios).

---

## Health Checks

```bash
# Check all 12 MCP servers
for port in 8000 8001 8002 8003 8004 8005 8006 8008 8009 8010 8016 8025; do
  echo -n "Port $port: "
  curl -s http://localhost:$port/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))"
done

# Check the local LLM server
curl http://localhost:9000/v1/models

# Verify GPU access inside the orchestrator container
docker exec agents-orchestrator python -c "import torch; print(torch.cuda.is_available())"
```
