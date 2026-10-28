# Fixture Binary Files

These files are required to run the test suite but are excluded from git due to size.

## Download (public Google Drive archive)

All 11 files below ship as a single public archive — `raw.zip`, ~161 MB, verified
byte-identical (md5) to the fixtures every committed result was produced with:

**<https://drive.google.com/file/d/1BvCoujwL3YydSe8qLM_gwXUfoRhZVx0f/view?usp=sharing>**

```bash
# from the repo root — CLI download (gdown handles Drive's large-file confirm page):
pip install gdown
gdown 1BvCoujwL3YydSe8qLM_gwXUfoRhZVx0f -O /tmp/raw.zip

# the zip contains a top-level raw/ folder → extract into data/fixtures/
# (exclude the zip's own stale DOWNLOADS.md copy so this file isn't overwritten)
unzip /tmp/raw.zip -d data/fixtures/ -x "__MACOSX/*" "raw/DOWNLOADS.md"
ls data/fixtures/raw/   # → the 11 files in the table below
```

(Or download from the browser link above and unzip so the files land in
`data/fixtures/raw/`.)

| File | Description | Used by |
|------|-------------|---------|
| `sample_video.mp4` | 60 s lecture intro (variable elimination) | SC-01, SC-04, SC-06, SC-09 |
| `sample_video_long.mp4` | Same lecture, 409 s | SC-40 source material (summarization archive) |
| `f-35.mp4` | Pilot + F-35 fighter jet, ~88 s, 1280×720, with speech | F-28, F-30, F-32, F-33, F-37 (vision suite), SC-47 (scene detection) |
| `f-35_clip.mp4` | ~15 s clip of `f-35.mp4` (≈12–27 s, in-flight jet) | F-34 (video segmentation) |
| `f-35_t05.jpg` | Reference frame at t=5 s (pilot holding helmet; no aircraft) — copied from `tests/ground_truth/videos/f-35/frames/` | GT-aligned VQA/segmentation |
| `f-35_t15.jpg` | Reference frame at t=15 s (F-35 in flight) — copied from GT frames | F-29 (VQA), F-31 (segmentation) |
| `f-35_t65.jpg` | Reference frame at t=65 s (4 jets in formation) — copied from GT frames | extra scenarios F-38..F-42 (commented out) |
| `f-35.wav` | Audio track of `f-35.mp4` (first 10 s) | F-35 (error: VQA on audio) |
| `lionandgorilla_frame0.jpg` | Any still image (content irrelevant) | SC-07 (error: wrong file format) |
| `tedtalk.mp4` | TED talk video | interactive demos (summarize example) |
| `chimpanzee.mp4` | Wildlife footage | spare — no active scenario |

Derivation commands (clip/wav/frames from the primary fixture):

```bash
ffmpeg -ss 12 -t 15 -i f-35.mp4 -c copy f-35_clip.mp4
ffmpeg -i f-35.mp4 -t 10 -vn -acodec pcm_s16le f-35.wav
# reference frames: copy verbatim from tests/ground_truth/videos/f-35/frames/
cp ../../../tests/ground_truth/videos/f-35/frames/t005s.jpg f-35_t05.jpg
cp ../../../tests/ground_truth/videos/f-35/frames/t015s.jpg f-35_t15.jpg
cp ../../../tests/ground_truth/videos/f-35/frames/t065s.jpg f-35_t65.jpg
```

> **Legacy note.** The original vision fixtures (`lionandgorilla.mp4`,
> `lionandgorilla_15s.mp4`, `test_video.mp4`, `sample_video.wav`) were lost; the vision
> suite migrated to the F-35 set above (scenarios F-28..F-37). The legacy scenario
> definitions are preserved commented-out in `tests/scenarios/vision_scenarios.py`.
