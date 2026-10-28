"""
Grep match metric — recall vs expected match count.

For true-positive terms: pass iff returned_count >= expected_count (recall = 1.0).
For true-negative terms: pass iff returned_count == 0.
threshold is the minimum required recall (typically 1.0).
"""
from __future__ import annotations

from .utils import extract_data_paths, read_json_from_archive, scan_tool_outputs


def evaluate(
    new_items: list,
    final_output: str,
    gt_data: dict,
    threshold: float = 1.0,
    item_index: int | None = None,
    **_kw,
) -> tuple[bool, float | None, str]:
    """
    Compare returned match count against the GT item specified by item_index.
    Returns (passed, recall_score, detail).
    """
    items = gt_data.get("items", [])
    if not items:
        return False, None, "gt has no items"
    if item_index is not None and item_index < len(items):
        gt_item = items[item_index]
    else:
        gt_item = items[0]

    term = gt_item.get("term", "?")
    expected_count = gt_item.get("total_count", 0)
    is_tn = gt_item.get("is_true_negative", False)

    # Extract match count from tool outputs
    returned_count: int | None = None
    for d in scan_tool_outputs(new_items):
        # Grep server returns: {"success": true, "results": [{"match_count": N, ...}], ...}
        results_block = d.get("results")
        if isinstance(results_block, list) and results_block:
            first = results_block[0]
            if isinstance(first, dict) and "match_count" in first:
                returned_count = int(first["match_count"])
                break
        # Grep outputs may contain either a list of result records or one
        # summary dictionary (older tool-server format).
        if isinstance(results_block, dict) and "match_count" in results_block:
            returned_count = int(results_block["match_count"])
            break
        # Fallback: top-level match_count (e.g. from reading search_results.json directly)
        if "match_count" in d:
            returned_count = int(d["match_count"])
            break

    # Path-extraction fallback: find grep archive from sub-agent text output
    # e.g. "FOUND: 'X' at ... Output archive: /data/search_results/grep_XXXX.tar.gz"
    if returned_count is None:
        for p in extract_data_paths(new_items, "search_results"):
            if p.endswith(".tar.gz"):
                data = read_json_from_archive(p, "search_results.json")
                if data and "match_count" in data:
                    returned_count = int(data["match_count"])
                    break

    if returned_count is None:
        return False, None, f"no match_count found in tool outputs (term='{term}')"

    if is_tn:
        passed = returned_count == 0
        score = 1.0 if passed else 0.0
        detail = (
            f"TN term='{term}': returned={returned_count} "
            f"(expected 0) → {'PASS' if passed else 'FAIL'}"
        )
        return passed, score, detail

    # True positive: recall = min(returned, expected) / expected
    if expected_count == 0:
        return False, None, f"gt item for term='{term}' has expected_count=0 but is_true_negative=False"

    recall = min(returned_count, expected_count) / expected_count
    passed = recall >= threshold
    detail = (
        f"term='{term}': returned={returned_count}, expected≥{expected_count}; "
        f"recall={recall:.2f} (threshold≥{threshold:.2f})"
    )
    return passed, round(recall, 4), detail
