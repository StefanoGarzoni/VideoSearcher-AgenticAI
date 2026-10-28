"""
Word Error Rate (WER) metric — agreement vs large-v3 silver reference.

Lower is better. threshold is the MAXIMUM allowed WER (e.g. 0.30 for a lecture).
"""
from __future__ import annotations

import re
import tarfile

from .utils import extract_data_paths, find_all_outputs_with_key


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _strip_timestamps(text: str) -> str:
    """Remove Whisper timestamp markers like [00:00.000] from transcript.txt lines."""
    return re.sub(r"\[\d+:\d+\.\d+\]", "", text).strip()


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word error rate via DP edit distance. Returns value in [0, ∞)."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    n = len(ref)
    if n == 0:
        return 0.0
    m = len(hyp)
    # Space-efficient: keep only prev and curr row
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[m] / n


def read_transcript_from_archive(archive_path: str) -> str:
    """Extract the transcript text from a .tar.gz archive containing transcript.txt."""
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            # Prefer files named 'transcript.txt'
            for member in members:
                if "transcript" in member.name and member.name.endswith(".txt"):
                    f = tar.extractfile(member)
                    if f:
                        return f.read().decode("utf-8", errors="replace").strip()
            # Fallback: any .txt
            for member in members:
                if member.name.endswith(".txt"):
                    f = tar.extractfile(member)
                    if f:
                        return f.read().decode("utf-8", errors="replace").strip()
    except Exception:
        pass
    return ""


def evaluate(
    new_items: list,
    final_output: str,
    gt_data: dict,
    threshold: float,
    **_kw,
) -> tuple[bool, float | None, str]:
    """
    Locate a transcription archive in new_items, compute WER vs gt_data['full_text'].
    Returns (passed, wer_score, detail).
    """
    ref_text = gt_data.get("full_text", "")
    if not ref_text:
        return False, None, "gt missing full_text"

    # Find the transcription tool output: success=True + output_path to a .tar.gz
    archive_path: str | None = None
    for d in find_all_outputs_with_key(new_items, "output_path"):
        if not d.get("success"):
            continue
        path = d.get("output_path", "")
        if isinstance(path, str) and "transcription" in path and path.endswith(".tar.gz"):
            archive_path = path
            break

    # Fallback: extract path from sub-agent text outputs ("DONE: output_path=...")
    if not archive_path:
        for p in extract_data_paths(new_items, "transcription"):
            if p.endswith(".tar.gz"):
                archive_path = p
                break

    if not archive_path:
        return False, None, "no transcription archive found in tool outputs"

    hyp_text = read_transcript_from_archive(archive_path)
    if not hyp_text:
        return False, None, f"could not extract transcript text from {archive_path}"

    hyp_text = _strip_timestamps(hyp_text)
    score = compute_wer(ref_text, hyp_text)
    passed = score <= threshold
    detail = (
        f"WER={score:.3f} (threshold≤{threshold:.2f}); "
        f"agreement vs large-v3 silver"
    )
    return passed, round(score, 4), detail
