"""
Detection-presence metric — frame-independent segmentation capability check.

WHY THIS EXISTS
---------------
Chain scenarios such as "extract a frame → segment it" do NOT let us assume the
frame the orchestrator extracts equals any ground-truth frame. The exact frame
depends on the model (it may extract t=14, t=15, a 1-fps range, …) AND on the
video's native resolution (f-35.mp4 is 1920×1080, while the GT frames were
annotated at a downscaled 1280×720). A fixed-bbox IoU — or any check anchored to
GT pixel dimensions — is therefore invalid for these scenarios: it compares boxes
drawn on frame A (native res) against numbers authored on frame B (downscaled).
That violates "a GT comparison is only valid when prediction and GT describe the
same artifact", and produces false negatives driven by model luck / resolution,
not behaviour.

WHAT THIS ASSERTS INSTEAD
-------------------------
The capability claim that holds for ANY legitimately-extracted frame:

  pass iff the segmentation tool returned >= 1 detection whose box is
  geometrically plausible — valid coordinates, inside the *actual processed
  frame's* bounds, and area fraction within a sane band [threshold, 0.98].

Crucially, the frame dimensions are taken from the artifact under test itself —
read from the segmented frame the tool reports (frame_path / image_path), NOT
from the ground truth. So the verdict is independent of the orchestrator model
and of the GT annotation resolution. It is not gameable: the system must really
run the detector and localise an object; it fails on empty output, malformed
boxes, or degenerate / whole-image boxes. Localisation *accuracy* (strict IoU)
stays on single-image scenarios (e.g. F-31), where the artifact IS the GT frame.

PARAMETERS
----------
gt_data    used only as a last-resort source of image dimensions.
threshold  minimum plausible-box area fraction (default 0.005). Upper bound is
           fixed at 0.98 to reject whole-image boxes. Area-fraction and bounds
           checks are applied only when the frame's true dimensions are known;
           if they cannot be determined, the metric degrades to a presence +
           valid-geometry check (never a false negative from unknown dims).
"""
from __future__ import annotations

from .utils import scan_tool_outputs

_AREA_HI = 0.98          # reject boxes covering ~the whole frame
_OOB_TOL = 0.02          # allow boxes 2% outside the frame (rounding slack)


def _frame_dims(box_dict: dict, gt_item: dict) -> tuple[int | None, int | None, str]:
    """
    Resolve the processed frame's (width, height) from the most authoritative
    source available: explicit tool dims → the segmented image file → GT dims.
    Returns (w, h, source).
    """
    # 1) explicit dimensions on the tool output (future-proofing; current server
    #    does not emit these, but other tools might)
    for wk, hk in (("image_w", "image_h"), ("width", "height"), ("img_w", "img_h")):
        if box_dict.get(wk) and box_dict.get(hk):
            return int(box_dict[wk]), int(box_dict[hk]), f"tool:{wk}"

    # 2) open the actual segmented frame the tool reports
    path = box_dict.get("frame_path") or box_dict.get("image_path")
    if path:
        try:
            from PIL import Image  # available in the orchestrator container
            with Image.open(path) as im:
                return int(im.width), int(im.height), "frame_file"
        except Exception:
            pass

    # 3) last resort: GT annotation dims (may differ in resolution — used only if
    #    nothing better exists)
    if gt_item.get("image_w") and gt_item.get("image_h"):
        return int(gt_item["image_w"]), int(gt_item["image_h"]), "gt(fallback)"

    return None, None, "unknown"


def evaluate(
    new_items: list,
    final_output: str,
    gt_data: dict,
    threshold: float = 0.005,
    item_index: int | None = None,
    **_kw,
) -> tuple[bool, float | None, str]:
    """
    Return (passed, plausible_fraction, detail).
    passed iff >= 1 returned box is geometrically plausible.
    """
    items = gt_data.get("items", []) if isinstance(gt_data, dict) else []
    gt_item: dict = {}
    if item_index is not None and 0 <= item_index < len(items):
        gt_item = items[item_index]
    elif items:
        gt_item = items[0]

    # First tool output that carries bounding boxes.
    box_dict: dict | None = None
    pred_boxes: list[list[float]] = []
    pred_labels: list[str] = []
    for d in scan_tool_outputs(new_items):
        b = d.get("boxes")
        if isinstance(b, list) and len(b) > 0:
            box_dict = d
            pred_boxes = [list(x) for x in b]
            lbls = d.get("labels") or []
            pred_labels = [str(x) for x in lbls]
            break

    if not pred_boxes:
        return False, 0.0, "no detections returned by segmentation"

    W, H, dim_src = _frame_dims(box_dict or {}, gt_item)
    area_lo = float(threshold)
    plausible = 0
    notes: list[str] = []
    for i, bb in enumerate(pred_boxes):
        try:
            x1, y1, x2, y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
        except (IndexError, TypeError, ValueError):
            notes.append(f"box{i}: malformed")
            continue
        if not (x2 > x1 and y2 > y1):
            notes.append(f"box{i}: non-positive area")
            continue
        if W and H:
            if (x1 < -_OOB_TOL * W or y1 < -_OOB_TOL * H
                    or x2 > W * (1 + _OOB_TOL) or y2 > H * (1 + _OOB_TOL)):
                notes.append(f"box{i}: out of bounds ({W}x{H})")
                continue
            frac = ((x2 - x1) * (y2 - y1)) / float(W * H)
            if not (area_lo <= frac <= _AREA_HI):
                notes.append(f"box{i}: area frac {frac:.3f} outside [{area_lo:.3f},{_AREA_HI}]")
                continue
            notes.append(f"box{i}: ok (frac {frac:.3f})")
        else:
            # Dimensions unknown — degrade to presence + valid-geometry only.
            notes.append(f"box{i}: ok (geometry valid; dims unknown)")
        plausible += 1

    passed = plausible >= 1
    score = plausible / len(pred_boxes)
    label_str = (" labels=" + "/".join(pred_labels[:4])) if pred_labels else ""
    detail = (
        f"{len(pred_boxes)} box(es){label_str}; dims={W}x{H}[{dim_src}]; "
        f"plausible={plausible} → {'PASS' if passed else 'FAIL'}; " + "; ".join(notes[:4])
    )
    return passed, round(score, 4), detail
