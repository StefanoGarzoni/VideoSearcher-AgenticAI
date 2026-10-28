# VLM Components — VideoSearcher AgenticAI

This document covers the machine learning models used in the system: what each model does, why it was chosen, how it is integrated, and its known limitations. This is the reference for understanding the AI/ML layer of the project.

---

## 1. Orchestrator LLM — Gemma-4-E2B (default)

### What it is

Gemma-4-E2B-it-Q8_0 is a 2-billion-parameter instruction-tuned language model from Google, quantized to 8-bit precision (Q8_0). It is served locally using `llama.cpp`'s `llama-server` on port 9000 and exposes an OpenAI-compatible API.

### Role in the system

The orchestrator LLM receives the user's natural-language request and decides which specialized agents to invoke and in which order. It does NOT directly execute any tools — it only calls handoff functions (e.g., `audio_extraction_handoff`, `transcription_handoff`) that delegate to specialized agents.

### Why this model

- **Small footprint.** At Q8_0, the 2B model occupies approximately 5 GB of GPU VRAM, leaving room for the VQA model (~30 GB) and segmentation model (~3 GB) on the same GPU.
- **Instruction-tuned.** The `-it` (instruction-tuned) variant follows explicit step-by-step instructions, which is required because the orchestrator's system prompt contains named pipeline chains (A–H) with explicit ordering rules.
- **Fast inference.** With `--reasoning-budget 0` (thinking mode disabled), each tool call decision takes approximately 8 seconds.

### Known limitations

- **Multi-step pipeline failures.** The 2B model occasionally stops mid-pipeline after completing one step, treating it as the final answer. This is a model capability limit, not a code bug. The system prompt's pipeline chains (A–H) mitigate but do not eliminate this.
- **Non-determinism.** On identical prompts, the model may sometimes call the same agent twice or in a different order. `max_turns=15` prevents infinite loops, and the test runner's subsequence matching tolerates duplicate calls.

### Alternative models

| Key | Model | VRAM | Notes |
|-----|-------|------|-------|
| `gemma` | gemma-4-E2B Q8 | ~5 GB | Default — fast, low memory |
| `phi4-mini` | Phi-4-mini Q4 | ~3 GB | Microsoft instruct baseline; too small to emit tool calls reliably |
| `qwen25-7b` | Qwen2.5-7B Q8 | ~8 GB | Better reasoning, but parallel tool-call issues observed |
| `gemma4-12b` | gemma-4-12b Q8 | ~12 GB | Largest dense option; strongest accuracy |
| `gpt-oss-120b` | gpt-oss-120b MXFP4 | ~60 GB | 120B MoE, 3 shards, registry ctx 8192; run with `--reasoning low` |

Pick one with `run_agentic_system.sh --orchestrator-model KEY`. The VQA model below is
**held constant** and is never an orchestrator option — only the orchestrator LLM varies
(e.g. when benchmarking with `scripts/run_model_comparison.py`). See
`local-ai-service/ADDING_MODELS.md` for how to add new models to the registry.

---

## 2. VQA Model — Phi-4-reasoning-vision-15B

### What it is

`microsoft/Phi-4-reasoning-vision-15B` is a 15-billion-parameter vision-language model (VLM) that can answer open-ended questions about images. It takes a JPEG/PNG frame and a natural-language question, and returns a text answer.

### Role in the system

The `VqaAgent` loads this model in-process inside the `agents-orchestrator` container. When the orchestrator calls `vqa_handoff`, it passes a frame path and a question. The VQA agent loads the image, encodes it into the model's visual format, and generates a text answer.

Typical usage:

```text
Input: frame_path=/data/runs/<session>/frames/f-35_abc123/frame_000001.jpg
       question="What aircraft is visible?"
Output: "An F-35 fighter jet is shown in flight against a clear sky."
```

### Why this model

- **Vision-reasoning.** The "reasoning" variant is specifically designed for visual question answering with chain-of-thought reasoning steps, producing more reliable and detailed answers than base VLMs.
- **Microsoft's official model.** Well-documented, with known integration patterns for HuggingFace Transformers.
- **Local inference.** No cloud API cost; answers are generated entirely on the local GPU.

### VRAM requirements

- ~30 GB in bfloat16 (half-precision float)
- Requires NVIDIA GPU with ≥30 GB VRAM (e.g., the GB10 with 128 GB unified memory)
- Model weights are cached in a named Docker volume (`vqa_hf_cache`) — first download is ~15 GB and takes 15–20 minutes on a cold cache

### Known compatibility issues and workarounds

Phi-4's HuggingFace implementation targets older versions of the `transformers` library. Three workarounds are applied in `vqa_agent.py`:

| Issue | Fix applied |
|-------|-------------|
| `Phi4VisionRProcessor` imports from `siglip2.image_processing_siglip2`, removed in `transformers >= 5.4` | `requirements.txt` pins `transformers >= 5.0, < 5.4` |
| Modeling code uses `past_key_values[-1][-1].shape[-2]` which fails on `DynamicCache` (new in transformers 5.x) | `_ensure_phi4_cache_patched()` patches the cached model file on every load (idempotent) |
| `HuggingFaceVLM.predict()` calls `processor.apply_chat_template()` which Phi-4's processor lacks | `_phi4_predict()` uses tokenizer-level template + `<image>\n{question}` format from Phi-4's own helper |

Each workaround becomes a no-op if Microsoft updates the model repository on HuggingFace Hub.

### How VQA integrates with frame extraction

1. `extract_frames_handoff` extracts frames and injects `first_frame_path` (absolute path to the first frame) directly into the tool result.
2. The orchestrator's system prompt instructs it to pass `first_frame_path` verbatim to `vqa_handoff`.
3. `vqa_agent.py` also reads `/tmp/last_extracted_frame.txt` as a fallback when the given path does not exist (written by the frame extraction tool on every successful extraction).

This double-mechanism was needed because the orchestrator LLM (2B parameters) occasionally mangles hash-named directory paths when reconstructing them from string output.

### Energy tracking

VQA inference is energy-tracked via CodeCarbon as `vqa_phi4_inference`: the tracker wraps
`model.generate()` and is started in `answer_visual_question_tool` on the event-loop thread
(it cannot start inside the worker thread, where CodeCarbon's signal handlers fail). See
[`CODECARBON.md`](CODECARBON.md). Model-serving latency for the orchestrator LLM is reported
separately as the `ttft_ms` (time-to-first-token) metric on each `TestResult`.

---

## 3. Faster-Whisper — Speech-to-Text

### What it is

OpenAI Whisper is an automatic speech recognition (ASR) model. This system runs it through Faster-Whisper, a CTranslate2-based implementation optimized for practical inference. The default model is multilingual `small` (~244M parameters); `small.en` (English-only) is available for a small accuracy boost on English audio.

### Role in the system

`tools/Transcribe/transcribe_main.py` wraps Faster-Whisper and exposes it as the `speech-to-text-mcp-server` on port 8004. Given an audio `.wav` file, it produces a timestamped transcript in the format:

```text
[00:00.000] They're talking to you, random person, scrolling Facebook.
[00:08.600] The F-35 is a fifth-generation fighter jet.
[00:29.680] At an altitude of approximately 50,000 feet.
```

The `[MM:SS.mmm]` prefix on every line is the key design choice that enables cross-modal chaining: the grep tool preserves this prefix in search matches, and the orchestrator parses it to drive frame extraction at the exact timestamp.

### Whisper model variants

| Variant | Language | VRAM | Use when |
|---------|----------|------|---------|
| `small.en` | English only | CPU-compatible | Default — fastest, English audio |
| `small` | Multilingual | CPU-compatible | Non-English audio (Italian F-35 video) |
| `medium` | Multilingual | ~5 GB GPU | Better accuracy at higher cost |
| `large-v3` | Multilingual | ~10 GB GPU | Reference-quality transcription |

The Faster-Whisper model is not loaded on GPU in the default setup — it runs on CPU inside the `speech-to-text-mcp-server` container. This is intentional: GPU resources are reserved for VQA and segmentation.

---

## 4. GroundingDINO — Open-Vocabulary Object Detection

### What it is

GroundingDINO is a zero-shot object detection model that accepts a text query (e.g., `"person, car"`) and returns bounding boxes for matching objects in an image. Unlike YOLO or Faster R-CNN, it does not require pre-defined class labels — any noun phrase in the query is a valid target.

### Role in the system

GroundingDINO runs inside `vlm-segmentation-mcp-server` (port 8008). Given an image and a text query, it produces:
- Bounding boxes (normalized coordinates)
- Confidence scores
- Labels (matched to the query phrase)

These are passed to SAM2 for precise pixel-level masks.

### Why GroundingDINO

- **Open-vocabulary.** The orchestrator can pass any user-specified query (e.g., `"lion"`, `"F-35 aircraft"`, `"person with red jacket"`) without retraining or reconfiguring the model.
- **Text-grounded detection.** Bridges the gap between natural language and visual objects — exactly what an agentic system needs.

---

## 5. SAM2 — Segment Anything Model v2

### What it is

SAM2 (Segment Anything Model 2) by Meta produces high-quality pixel-level segmentation masks given a prompt (bounding box or point). It is a foundation model trained on a massive dataset of masks and generalizes well to unseen objects.

### Role in the system

SAM2 runs alongside GroundingDINO inside `vlm-segmentation-mcp-server`. After GroundingDINO identifies bounding boxes for the queried objects, SAM2 produces precise masks for each detected region.

### Two-stage pipeline

```text
User query: "lion"
         ↓
GroundingDINO: image + "lion" → bounding boxes [x1,y1,x2,y2], scores
         ↓
SAM2: image + bounding boxes → segmentation masks (PNG files)
         ↓
Output: mask_paths / segmented_image_path
```

### Output modes

| Mode | Output | Use case |
|------|--------|---------|
| `masks` | Individual PNG mask files per object | Internal pipelines, VQA post-processing, spatial reasoning |
| `segmented_image` | One visual image with colored overlays | User-facing segmentation result |

### Video segmentation

`video-segmentation-mcp-server` (port 8009) does NOT load a second model. It extracts frames with FFmpeg, then calls `vlm-segmentation-mcp-server` with `output_mode="segmented_image"` for each frame. The frames are assembled into a segmented video (or returned as a set of segmented frames).

### VRAM requirements

GroundingDINO + SAM2 together require approximately 3 GB VRAM when loaded. The `vlm-segmentation-mcp-server` declares `runtime: nvidia` in `docker-compose.yml`.

### Private VLM framework (not distributed)

Both `vlm-segmentation-mcp-server` and `VqaAgent` expect imports from a separately maintained, unavailable VLM framework (`VLMs-Testing-Framework/src/`). Its original interfaces provided:
- `SegmentationPipelines` — GroundingDINO + SAM2 combined pipeline
- `ModelInterfaces` — unified HuggingFace model loading for VQA
- `Samples` — sample data utilities

This component is not distributed or downloadable through this source snapshot; this command is intentionally disabled:
```bash
echo "Private component not distributed"
# then rebuild affected images
```

---

## 6. Summary Table

| Model | Parameters | VRAM | Where it runs | Purpose |
|-------|-----------|------|--------------|---------|
| Gemma-4-E2B Q8 | 2B | ~5 GB | host (llama-server) | Orchestrator reasoning & routing |
| Phi-4-reasoning-vision-15B | 15B | ~30 GB | agents-orchestrator (in-process) | Visual question answering |
| Faster-Whisper small | 244M | CPU | speech-to-text-mcp-server | Speech-to-text transcription |
| GroundingDINO | ~230M | ~1 GB | vlm-segmentation-mcp-server | Text-guided bounding box detection |
| SAM2 | ~308M | ~2 GB | vlm-segmentation-mcp-server | Pixel-level segmentation masks |

Total VRAM on a simultaneous full-stack run: ~38 GB (VQA + segmentation + orchestrator LLM).
