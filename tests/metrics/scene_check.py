"""
Scene-detect metric.

Checks two things:
1. Scene count: |predicted_count - gt_count| <= count_tol (default 3)
2. Cut recall: fraction of GT cuts that have a predicted cut within tolerance_s seconds.

'threshold' is used as tolerance_s (seconds).
Pass iff count check AND cut_recall >= 0.7.
"""
from __future__ import annotations

from .utils import extract_data_paths, read_json_from_path, scan_tool_outputs


def _parse_scenes(d: dict) -> tuple[int, list[float]]:
    """Extract num_scenes and cut timestamps from a scene dict."""
    count = int(d.get("num_scenes", 0))
    scenes = d.get("scenes", [])
    cuts: list[float] = []
    if isinstance(scenes, list):
        cuts = [
            float(s["start_seconds"])
            for s in scenes
            if isinstance(s, dict) and s.get("start_seconds", 0) > 0
        ]
    return count, cuts


def evaluate(
    new_items: list,
    final_output: str,
    gt_data: dict,
    threshold: float = 1.5,
    count_tol: int = 3,
    cut_recall_min: float = 0.7,
    **_kw,
) -> tuple[bool, float | None, str]:
    """
    Compare detected scene count and cut timestamps vs ground truth.
    Returns (passed, cut_recall_score, detail).
    """
    gt_count: int = gt_data.get("scene_count", 0)
    gt_cuts: list[float] = gt_data.get("expected_cuts", [])
    tolerance_s: float = gt_data.get("tolerance_s", threshold)

    # Find scene detection tool output from scan_tool_outputs
    pred_count: int | None = None
    pred_cuts: list[float] = []

    for d in scan_tool_outputs(new_items):
        if "num_scenes" in d:
            pred_count, pred_cuts = _parse_scenes(d)
            break

    # Path-extraction fallback: scene detect agent outputs "DONE: output_path=<path>"
    # where path is a scenes.json file (e.g. /data/scenes/scenes_XXXX/scenes.json)
    if pred_count is None:
        for p in extract_data_paths(new_items, "scenes"):
            if p.endswith(".json"):
                d = read_json_from_path(p)
                if d and "num_scenes" in d:
                    pred_count, pred_cuts = _parse_scenes(d)
                    break

    if pred_count is None:
        return False, None, "no scene detection output found in tool outputs"

    # Count check
    count_diff = abs(pred_count - gt_count)
    count_ok = count_diff <= count_tol

    # Cut recall
    if not gt_cuts:
        cut_recall = 1.0
    else:
        matched = 0
        for gt_cut in gt_cuts:
            if any(abs(gt_cut - pc) <= tolerance_s for pc in pred_cuts):
                matched += 1
        cut_recall = matched / len(gt_cuts)

    cut_ok = cut_recall >= cut_recall_min
    passed = count_ok and cut_ok

    detail = (
        f"scenes: pred={pred_count} vs gt={gt_count} (diff={count_diff}, tol±{count_tol}={'OK' if count_ok else 'FAIL'}); "
        f"cut_recall={cut_recall:.2f} (≥{cut_recall_min:.1f}?{'OK' if cut_ok else 'FAIL'}); "
        f"tolerance={tolerance_s}s"
    )
    return passed, round(cut_recall, 4), detail
