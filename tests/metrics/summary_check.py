"""
Summary key-point coverage metric.

Score = fraction of gt key_points that appear (case-insensitive substring) in
the summary text. Pass if score >= threshold (e.g. 0.60).
"""
from __future__ import annotations

from .utils import extract_data_paths, read_text_from_archive, scan_tool_outputs


def evaluate(
    new_items: list,
    final_output: str,
    gt_data: dict,
    threshold: float = 0.6,
    **_kw,
) -> tuple[bool, float | None, str]:
    """
    Check key-point coverage in the orchestrator's final output or the summarize
    tool's 'summary' field (whichever is longer).
    Returns (passed, coverage_score, detail).
    """
    key_points: list[str] = gt_data.get("key_points", [])
    if not key_points:
        return False, None, "gt has no key_points"

    # Prefer the summarize tool output's 'summary' field; fall back to final_output
    summary_text = final_output.strip()
    for d in scan_tool_outputs(new_items):
        candidate = d.get("summary", "")
        if isinstance(candidate, str) and len(candidate) > len(summary_text):
            summary_text = candidate.strip()
            break

    # Archive fallback: summarize agent returns "DONE: output_path=..." pointing to
    # a .tar.gz that contains summary.txt with the actual summary content.
    for p in extract_data_paths(new_items, "summarization"):
        if p.endswith(".tar.gz"):
            txt = read_text_from_archive(p, "summary.txt")
            if txt and len(txt) > len(summary_text):
                summary_text = txt
            break

    if not summary_text:
        return False, 0.0, "no summary text found"

    lower = summary_text.lower()
    covered = [kp for kp in key_points if kp.lower() in lower]
    n_total = len(key_points)
    n_covered = len(covered)
    score = n_covered / n_total

    passed = score >= threshold
    detail = (
        f"covered {n_covered}/{n_total} key_points "
        f"({score:.0%} ≥ {threshold:.0%}?{'yes' if passed else 'no'}): "
        f"found={covered}; missing={[kp for kp in key_points if kp not in covered]}"
    )
    return passed, round(score, 4), detail
