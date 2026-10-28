# Ground Truth Schema

Canonical field definitions. Step 3 test code reads these files — treat this as the contract.

## Storage Layout

```
tests/ground_truth/
  README.md           — provenance + caveats per file
  schema.md           — this file (canonical field definitions)
  videos/
    <video-name>/
      frames/         — deterministically extracted frames (committed)
        t<NNN>s.jpg   — frame at NNN seconds (zero-padded to 3 digits)
      transcript.json
      grep.json
      audio.json
      vqa.json
      segmentation.json
      scene_detect.json
```

Each video's artifacts live entirely within `videos/<video-name>/`. Adding a new video = add a new directory.
Frame filenames: `t<NNN>s.jpg` where NNN is zero-padded timestamp in seconds (e.g. `t005s.jpg` = 5s).

---

## VQA — `videos/<video>/vqa.json`

```json
{
  "video": "<video-name>",
  "items": [
    {
      "frame_path": "videos/<video>/frames/t<NNN>s.jpg",
      "timestamp": 5,
      "question": "...",
      "q_type": "identify | count | attribute | spatial | true_negative",
      "reference_answer": "...",
      "required_facts": ["fact1", "fact2"],
      "acceptable_keywords": ["kw1", "kw2"],
      "is_true_negative": false,
      "min_answer_length": 3
    }
  ]
}
```

- `required_facts` — ALL must appear in model answer (AND logic)
- `acceptable_keywords` — any subset that captures the answer (OR logic)
- `is_true_negative` — question asks about something NOT present
- Absence of `is_silver` (or `is_silver: true`) — see silver convention below

---

## Segmentation — `videos/<video>/segmentation.json`

```json
{
  "video": "<video-name>",
  "items": [
    {
      "frame_path": "videos/<video>/frames/t<NNN>s.jpg",
      "timestamp": 5,
      "query": "airplane",
      "image_w": 1280,
      "image_h": 720,
      "objects": [
        {
          "label": "airplane",
          "bbox": [x1, y1, x2, y2],
          "present": true
        }
      ],
    }
  ]
}
```

- `bbox` — pixel coordinates `[x1, y1, x2, y2]`, origin top-left
- `present: false` — object is absent (true-negative query); bbox should be null/omitted
- Bboxes are approximate (Claude-authored visual inspection); documented in README

---

## Transcript (silver) — `videos/<video>/transcript.json`

```json
{
  "video": "<video-name>",
  "language": "en",
  "source": "whisper-large-v3",
  "reference_model": "large-v3",
  "tested_model": "small",
  "segments": [
    {"start": 0.0, "end": 3.5, "text": "..."}
  ],
  "full_text": "...",
  "is_silver": true
}
```

- `reference_model` must differ from `tested_model` (circularity guard)
- `is_silver: true` always — machine-generated, not hand-verified

---

## Grep — `videos/<video>/grep.json`

```json
{
  "video": "<video-name>",
  "items": [
    {
      "term": "elimination",
      "transcript": "<video-name>",
      "expected_matches": [
        {"segment_idx": 3, "start": 12.1, "text": "...elimination..."}
      ],
      "total_count": 2,
      "is_true_negative": false
    }
  ]
}
```

- `expected_matches[].text` must be a substring of `transcript.json full_text`
- `is_true_negative: true` — term does not appear; `expected_matches: []`, `total_count: 0`

---

## Audio/Silence (silver, ASR-derived) — `videos/<video>/audio.json`

```json
{
  "video": "<video-name>",
  "method": "gaps-between-whisper-large-v3-segments",
  "min_silence_s": 0.5,
  "silence_segments": [
    {"start": 0.0, "end": 1.2}
  ],
  "speech_segments": [
    {"start": 1.2, "end": 4.8}
  ],
  "is_silver": true
}
```

- Derived from gaps between consecutive Whisper `large-v3` segments
- Independent of production `librosa.effects.split`

---

## Scene-Detect — `videos/<video>/scene_detect.json`

```json
{
  "video": "<video-name>",
  "expected_cuts": [5.0, 22.5, 40.0],
  "scene_count": 4,
  "tolerance_s": 1.0,
  "method": "hand-verified frame inspection",
}
```

- `tolerance_s` — acceptable offset for predicted cut timestamps
- `scene_count` — number of distinct scenes (= len(expected_cuts) + 1)

---

## Silver / Gold Convention

- **Gold** (no `is_silver` field): human-verified. Test failures against gold items are definitive.
- **Silver** (`"is_silver": true`): machine-generated or not yet human-verified. Test failures are
  informative but should trigger a manual spot-check before promotion to gold.
- `"is_silver": false` is deprecated — use absence of the field to mean gold.

The `grep.json` items carry `is_silver` at the top level (one flag per file).
VQA, segmentation, and scene_detect carry it per-item where each item may have independent confidence.
