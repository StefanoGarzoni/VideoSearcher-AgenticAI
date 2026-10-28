"""
Bounding-box IoU metric for segmentation outputs.

For each GT object with present=True: find the predicted box with highest IoU;
pass if best IoU >= threshold.
For GT objects with present=False (true-negatives): pass iff no boxes returned.
Score = mean IoU across all GT objects with present=True.
"""
from __future__ import annotations

from .utils import scan_tool_outputs


def _compute_iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between [x1,y1,x2,y2] boxes. Returns 0 on invalid input."""
    try:
        ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    except (IndexError, TypeError, ValueError):
        return 0.0
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate(
    new_items: list,
    final_output: str,
    gt_data: dict,
    threshold: float = 0.5,
    item_index: int | None = None,
    **_kw,
) -> tuple[bool, float | None, str]:
    """
    Compare predicted boxes/labels from tool outputs against GT objects.
    Returns (passed, mean_iou, detail).
    """
    items = gt_data.get("items", [])
    if not items:
        return False, None, "gt has no items"

    # Collect GT items to evaluate
    if item_index is not None:
        if item_index >= len(items):
            return False, None, f"item_index={item_index} out of range (len={len(items)})"
        eval_items = [items[item_index]]
    else:
        # Default: first item with present=True (most conservative)
        eval_items = [items[0]]

    # Extract predicted boxes from tool output
    pred_boxes: list[list[float]] = []
    pred_labels: list[str] = []
    for d in scan_tool_outputs(new_items):
        if "boxes" in d:
            boxes = d["boxes"]
            labels = d.get("labels", [])
            if isinstance(boxes, list) and len(boxes) > 0:
                pred_boxes = [list(b) for b in boxes]
                pred_labels = [str(lbl) for lbl in labels] if labels else ["?"] * len(boxes)
                break

    no_predictions = len(pred_boxes) == 0

    iou_scores: list[float] = []
    details: list[str] = []

    for gt_item in eval_items:
        gt_objects = gt_item.get("objects", [])
        query = gt_item.get("query", "?")
        frame = gt_item.get("frame_path", "?").split("/")[-1]

        for obj in gt_objects:
            present = obj.get("present", True)
            gt_bbox = obj.get("bbox")
            gt_label = obj.get("label", query)

            if not present:
                # True-negative: pass if nothing detected
                tn_pass = no_predictions
                iou_scores.append(1.0 if tn_pass else 0.0)
                details.append(
                    f"TN '{gt_label}'@{frame}: "
                    f"{'no boxes ✓' if tn_pass else f'{len(pred_boxes)} boxes detected ✗'}"
                )
                continue

            if not gt_bbox:
                details.append(f"'{gt_label}'@{frame}: gt bbox missing, skip")
                continue

            if no_predictions:
                iou_scores.append(0.0)
                details.append(f"'{gt_label}'@{frame}: no predictions returned")
                continue

            # Find best IoU among all predictions
            best_iou = max(_compute_iou(gt_bbox, pb) for pb in pred_boxes)
            iou_scores.append(best_iou)
            details.append(f"'{gt_label}'@{frame}: best_IoU={best_iou:.3f}")

    if not iou_scores:
        return False, None, "no GT objects to evaluate"

    mean_iou = sum(iou_scores) / len(iou_scores)
    passed = all(s >= threshold for s in iou_scores)
    detail = f"threshold≥{threshold}; " + "; ".join(details)
    return passed, round(mean_iou, 4), detail
