# Historical fixture manifest — media not distributed

This file documents the media used by the original academic evaluation. **It is not a download instruction.** The original delivery referenced an externally hosted ZIP, but that link has been removed from the public working tree because redistribution rights were not established. The host's access controls must be changed separately by its owner if the archive should not remain publicly accessible.

Original benchmark numbers were obtained with the original fixture set. Those figures are retained as historical measurements; this repository cannot reproduce the full suite without the separately owned VLM framework, models, hardware and properly licensed media.

| Historical filename | Role in the experimental setup |
|---|---|
| `sample_video.mp4` | Short lecture excerpt for speech and retrieval tasks |
| `sample_video_long.mp4` | Long lecture fixture for historical summarization scenarios |
| `f-35.mp4` | Aircraft video for vision, audio and scene tasks |
| `f-35_clip.mp4` | Short derived aircraft clip for segmentation |
| `f-35_t05.jpg`, `f-35_t15.jpg`, `f-35_t65.jpg` | Previously derived reference frames |
| `f-35.wav` | Derived audio excerpt |
| `lionandgorilla_frame0.jpg` | Historical format-error scenario |
| `tedtalk.mp4` | Historical interactive demonstration — **not provided or cleared for AI reuse** |
| `chimpanzee.mp4` | Historical spare wildlife fixture |

## Reproducing the setup with authorized materials

Use media you are entitled to process and redistribute, or create your own replacement videos. Place them under `data/fixtures/raw/` locally (ignored by Git). If you substitute media, ground truths, task expectations and any reported results may no longer match and must be recomputed; **do not present replacement-media outcomes as the original benchmark**.

The original fixtures and third-party video-derived frames are not included. TED specifically restricts AI-related uses without appropriate separate permission; consult the current [TED Terms of Use](https://www.ted.com/about/our-organization/our-policies-terms/ted-com-terms-of-use) before considering any such use.
