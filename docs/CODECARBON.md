# CodeCarbon and Local Energy Tracking

This guide explains how energy usage is tracked in the VideoSearcher system.

Because almost all inference is local (orchestrator LLM via `llama-server`, in-process
VQA, GPU segmentation, FFmpeg/Whisper tools), the footprint to track is local CPU/GPU
energy. Tracking is **built in**: each component starts its own `EmissionsTracker`
and writes to a shared emissions directory — there is no external wrapper to run.

---

## Three tracking layers

| Layer | What it measures | project_name(s) |
|---|---|---|
| **Layer 2 — Tool usage** | Each MCP tool's CPU/GPU work | one per tool (see table below) |
| **Layer 1 — LLM use** | Orchestrator LLM (GPU) + in-process VQA | `orchestrator_request` (gpu), `vqa_phi4_inference` |
| **Layer 3 — Orchestration overhead** | Client-side CPU/RAM of a request | `orchestrator_request` (cpu + ram) |

A single `orchestrator_request` tracker wraps each orchestrator request
(`process_request_with_items` in `main_agent.py`); CodeCarbon's component split
gives Layer 1 (its `gpu_energy`, dominated by the `llama-server` process on the same
GPU) and Layer 3 (its `cpu_energy + ram_energy`). The VQA model is tracked separately
via `vqa_phi4_inference`, started in `answer_visual_question_tool` on the event-loop
thread (CodeCarbon cannot start inside the worker thread, where its signal handlers fail).

---

## Layer 2 — all 12 MCP tool servers

Every tool server wraps its handler with a guarded `_energy_tracker()` helper that
instantiates `EmissionsTracker(project_name=..., output_dir="/app/emissions", log_level="error")`,
starts it before the work, and stops it in a `finally`:

| project_name | Port | Tool |
|---|---|---|
| `ffmpeg_audio_extraction` | 8000 | audio extraction |
| `librosa_audio_analysis` | 8001 | silence/audio analysis |
| `ffmpeg_video_split` | 8002 | video splitting |
| `ffmpeg_compression` | 8003 | compression |
| `whisper_transcription` | 8004 | speech-to-text |
| `grep_search` | 8005 | transcript search |
| `frame_extraction` | 8006 | frame extraction |
| `image_segmentation` | 8008 | GroundingDINO + SAM2 (GPU) |
| `video_segmentation` | 8009 | video segmentation (calls 8008 per frame) |
| `web_search` | 8010 | web search |
| `summarization` | 8016 | summarization |
| `scene_detection` | 8025 | scene detection |

`video_segmentation` calls the 8008 image server once per frame, so a video run also
emits several `image_segmentation` rows. The `codecarbon` dependency must be present in
each server's `requirements.txt` (otherwise `_energy_tracker()` degrades to a no-op).

Example pattern (as used in every tool server):

```python
_tracker = _energy_tracker("whisper_transcription")
try:
    ...  # tool logic
    return result
finally:
    if _tracker is not None:
        try:
            _tracker.stop()
        except Exception:
            pass
```

---

## Docker shared emissions directory

All instrumented containers (the 12 tool servers **and** the orchestrator) mount:

```yaml
volumes:
  - ./logs/emissions:/app/emissions
```

so every tracker appends to one shared host folder (`EMISSIONS_DIR=/app/emissions`).

---

## Aggregating emissions

```bash
python3 scripts/aggregate_emissions.py
```

This sums every CSV in `logs/emissions/` by `project_name` and also breaks out the
component energy, so Layer 1 vs Layer 3 is visible:

```text
- orchestrator_request: 0.00142 kg, 0.00429 kWh across 10 records
    [gpu=0.00251 kWh (L1 LLM) | cpu+ram=0.00178 kWh (L3 overhead)]
```

Output is written to `logs/emissions/aggregated_emissions.csv`. When multiple
containers write at once CodeCarbon may create `emissions.csv`, `emissions.csv_2`,
`emissions.csv.bak`, etc. — this is expected; aggregate after the run.

Per-model energy for a benchmarking run is captured automatically by
`scripts/run_model_comparison.py`, which attributes the emissions in each model's run
window to that model (the **Energy/run** column of `comparison_<run_id>.md`).

---

## Caveats

- **Shared GPU:** `orchestrator_request.gpu_energy` is whole-GPU draw during the
  request window — dominated by `llama-server`, but not perfectly isolated when other
  GPU work runs concurrently.
- **VQA double-count:** a request that triggers a VQA call has that GPU energy counted
  in *both* `orchestrator_request.gpu` and `vqa_phi4_inference` — do not naively sum
  those two.
- **Remote providers:** if you switch the orchestrator back to a hosted provider
  (Gemini/OpenAI), CodeCarbon cannot measure data-center energy; only a token-based
  estimate is possible. The default local setup is fully measured.
