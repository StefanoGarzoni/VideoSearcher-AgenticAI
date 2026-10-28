"""
Audio segment count metric.

Compares the number of speech segments returned by the librosa silence-detection
tool against the GT speech_segment count (derived from large-v3 ASR gaps).

Pass iff |returned_count - gt_count| <= threshold (default 3).
Score = 1 - |diff| / gt_count (clamped to [0, 1]).
"""
from __future__ import annotations

import glob
import re

from .utils import extract_data_paths, read_text_from_archive, scan_tool_outputs


def evaluate(
    new_items: list,
    final_output: str,
    gt_data: dict,
    threshold: float = 3.0,
    **_kw,
) -> tuple[bool, float | None, str]:
    """
    Check that librosa segment count is within threshold of the GT speech count.
    Returns (passed, score, detail).
    """
    gt_speech = gt_data.get("speech_segments", [])
    gt_count = len(gt_speech)
    if gt_count == 0:
        return False, None, "gt has no speech_segments"

    # Find librosa tool output: has 'segments_count'
    # Enhanced scan_tool_outputs also finds backtick-quoted dicts in sub-agent text output.
    returned_count: int | None = None
    for d in scan_tool_outputs(new_items):
        if "segments_count" in d and d.get("success"):
            returned_count = int(d["segments_count"])
            break

    # Archive fallback: count non-empty lines in timestamps.txt
    if returned_count is None:
        for p in extract_data_paths(new_items, "audio_analysis"):
            if p.endswith(".tar.gz"):
                txt = read_text_from_archive(p, "timestamps.txt")
                if txt:
                    returned_count = len([ln for ln in txt.splitlines() if ln.strip()])
                    break

    # Session-root fallback: derive session from any /data/runs/ID/ path and
    # glob for silence_detection_*.tar.gz in the session's audio_analysis dir.
    # This handles cases where the agent/orchestrator hallucinate a wrong path.
    if returned_count is None:
        all_text = final_output + " ".join(
            str(getattr(item, "output", "") or "") for item in new_items
        )
        m = re.search(r"/data/runs/([^/]+)/", all_text)
        if m:
            session_root = f"/data/runs/{m.group(1)}"
            for p in sorted(glob.glob(f"{session_root}/audio_analysis/silence_detection_*.tar.gz")):
                txt = read_text_from_archive(p, "timestamps.txt")
                if txt:
                    returned_count = len([ln for ln in txt.splitlines() if ln.strip()])
                    break

    if returned_count is None:
        return False, None, "no segments_count found in tool outputs (librosa response missing)"

    diff = abs(returned_count - gt_count)
    passed = diff <= threshold
    score = max(0.0, 1.0 - diff / gt_count)
    detail = (
        f"segments: returned={returned_count}, gt={gt_count} (diff={diff}, tol±{int(threshold)}); "
        f"note: methods differ (librosa vs large-v3 ASR gaps)"
    )
    return passed, round(score, 4), detail
