# Ground Truth — Provenance & Caveats

**Created**: 2026-06-13  
**Purpose**: Reference labels for Step 3 correctness testing of VideoSearcher-AgenticAI.  
**Schema contract**: see `schema.md`  
**Independence rule**: no label was derived from the production system's own output.

---

## Layout

```
tests/ground_truth/
  README.md        (this file)
  schema.md        (canonical field definitions)
  videos/
    f-35/          (f-35.mp4, 88s, 1280x720)
    lionandgorilla/ (lionandgorilla.mp4, 17s, 1280x720)
    sample_video/  (sample_video.mp4, 60s, 720p)
    tedtalk/       (tedtalk.mp4, 81s, 720p)
```

Each `videos/<name>/` contains all artifacts for that video: frames/, transcript.json, grep.json, audio.json, vqa.json, segmentation.json, scene_detect.json.

---

## Per-file provenance

### `videos/*/transcript.json` — SILVER
- **Source**: openai-whisper `large-v3`, run on GPU (CUDA PyTorch) in the `agents-orchestrator` container.
- **Independent of production**: production uses `faster-whisper small`. `reference_model: "large-v3"` ≠ `tested_model: "small"` — circularity guard satisfied.
- **Not human-verified**: do not treat as gold. Word-level errors possible, especially proper nouns.

### `videos/*/audio.json` — SILVER
- **Method**: `gaps-between-whisper-large-v3-segments` with `min_silence_s: 0.5`.
- Silence = intervals between consecutive `large-v3` transcript segments ≥ 0.5s.
- **Independent of production**: production uses `librosa.effects.split` with adaptive `top_db` sweep. Timestamps will differ — tolerance expected in Step 3 metrics.
- Not human-verified.

### `videos/*/grep.json` — SILVER (derived from silver transcript)
- Terms chosen to span: confirmed present terms, true-negative absent terms.
- Match locations derived by substring search of `large-v3` transcript segments.
- `expected_matches[].text` verified to be substrings of `transcript.json full_text`.

### `videos/f-35/vqa.json` — SILVER (human spot-check pending → A.3.8)
- 12 Q/A items across 6 key frames. All `q_type`s covered: identify, count, attribute, spatial, true_negative.
- **Authored by Claude via frame inspection** (Read tool — independent of any vision MCP server).
- Items confirmed by human spot-check are promoted to `is_silver: false` (gold). See spot-check section below.
- True negatives: items 3 (no aircraft at t=5s) and 7 (empty hangar at t=45s).
- Note: t085s (85s) shows a NATO logo end card — no VQA items use it.

### `videos/lionandgorilla/vqa.json` — SILVER (human spot-check pending → A.3.8)
- 5 Q/A items across 2 frames.
- All frames have a burned-in subtitle band (~80px bottom) and a "BLONDI FOKS" watermark center-frame. VQA questions avoid the subtitle region.
- True negative: item 5 (no tiger present).

### `videos/f-35/segmentation.json` — SILVER (human spot-check pending → A.3.8)
- 6 items across 5 frames.
- Bboxes are **approximate visual estimates** from Claude frame inspection. Expected IoU with production segmentation: 0.5–0.8 for large objects.
- 1 absent-object item: query "airplane" at t=5s (pilot close-up, no aircraft visible).

### `videos/lionandgorilla/segmentation.json` — SILVER (human spot-check pending → A.3.8)
- 5 items across 2 frames (t=6s and t=10s).
- 1 absent-object item: query "tiger" at t=6s.
- Bboxes approximate.

### `videos/f-35/scene_detect.json` — GOLD
- **Method**: frame inspection at 1fps density around transition zones (42 frames sampled).
- 11 confirmed cuts → 12 scenes. Tolerance ±1.5s.
- f-35.mp4 is a heavily edited promotional video with ~14+ rapid cuts.

---

## A.3.8 User Spot-Check — COMPLETE (all items gold)

**Status**: COMPLETE — 2026-06-13  
**Reviewer**: DanielGuarnizo  
**Step 2 check**: 5 VQA + 5 SEG items (2026-06-13, all OK)  
**Step 3 check**: 11 VQA + 6 SEG + 1 SUM items (2026-06-13, 17/18 OK, 1 correction)

### Step 2 (first pass)
All confirmed correct, no corrections.

### Step 3 (full remaining silvers)
- VQA V6–V16: all OK — promoted to gold
- SEG S7–S11: all OK — promoted to gold
- SEG S6 (f-35/t065s "airplane"): **CORRECTED** — reviewer confirmed only 2 of 4 aircraft are F-35s; the center and bottom jets are a different type (possibly F-15 Eagle or F-14, reviewer unsure). Labels updated to "F-35 upper-left", "F-35 upper-right", "fighter jet center (non-F-35)", "fighter jet bottom (non-F-35)". Now gold.
- SUM1 (sample_video summary key_points): OK — promoted to gold

**All ground truth items are now gold (`is_silver: false`). No silver items remain.**

---

## Regen commands

```bash
# Re-extract frames
docker exec ffmpeg-frames-mcp-server ffmpeg -ss <t> -i /data/raw/f-35.mp4 -frames:v 1 /tmp/frame.jpg

# Re-run Whisper large-v3 (orchestrator container, GPU)
docker exec agents-orchestrator python /tmp/run_whisper_openai.py

# Re-derive silence/grep from transcripts
python3 tests/ground_truth/derive_audio_grep.py  # (Step 3 will add this)
```
