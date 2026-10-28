"""
Unit tests for tests/metrics/* modules.
Run with: pytest tests/unit/test_metrics.py -v
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

from tests.metrics.wer import compute_wer


def test_wer_identical():
    assert compute_wer("hello world", "hello world") == 0.0


def test_wer_one_substitution():
    score = compute_wer("hello world", "hello earth")
    assert abs(score - 0.5) < 1e-9  # 1/2 words wrong


def test_wer_one_deletion():
    score = compute_wer("hello world today", "hello world")
    assert abs(score - 1 / 3) < 1e-6


def test_wer_one_insertion():
    score = compute_wer("hello world", "hello cruel world")
    assert abs(score - 0.5) < 1e-9  # 1 insertion out of 2 ref words


def test_wer_empty_reference():
    assert compute_wer("", "anything") == 0.0


def test_wer_normalises_case_and_punct():
    # "Hello, World!" vs "hello world" should be 0 after normalisation
    assert compute_wer("Hello, World!", "hello world") == 0.0


# ---------------------------------------------------------------------------
# Grep match
# ---------------------------------------------------------------------------

from tests.metrics.grep_match import evaluate as eval_grep


def _make_grep_items(items):
    return {"items": items}


def _make_grep_output(match_count: int):
    """Simulate a tool output item with a results block."""

    class _FakeItem:
        output = f'{{"success": true, "results": {{"match_count": {match_count}}}}}'

    return [_FakeItem()]


def test_grep_tp_full_recall():
    gt = _make_grep_items([{"term": "elimination", "total_count": 3, "is_true_negative": False}])
    passed, score, detail = eval_grep(_make_grep_output(3), "", gt, threshold=1.0, item_index=0)
    assert passed is True
    assert abs(score - 1.0) < 1e-9


def test_grep_tp_partial_recall_fails():
    gt = _make_grep_items([{"term": "elimination", "total_count": 3, "is_true_negative": False}])
    passed, score, detail = eval_grep(_make_grep_output(2), "", gt, threshold=1.0, item_index=0)
    assert passed is False
    assert abs(score - 2 / 3) < 1e-4  # score is rounded to 4 decimal places


def test_grep_tn_zero_matches_passes():
    gt = _make_grep_items([{"term": "quantum", "total_count": 0, "is_true_negative": True}])
    passed, score, detail = eval_grep(_make_grep_output(0), "", gt, threshold=1.0, item_index=0)
    assert passed is True
    assert score == 1.0


def test_grep_tn_nonzero_matches_fails():
    gt = _make_grep_items([{"term": "quantum", "total_count": 0, "is_true_negative": True}])
    passed, score, detail = eval_grep(_make_grep_output(2), "", gt, threshold=1.0, item_index=0)
    assert passed is False
    assert score == 0.0


def test_grep_list_style_results():
    """Current Grep server also reports per-file results as a list."""
    class _ToolItem:
        output = '{"success": true, "results": [{"match_count": 2}]}'

    gt = _make_grep_items([{"term": "AI", "total_count": 2, "is_true_negative": False}])
    passed, score, _ = eval_grep([_ToolItem()], "", gt)
    assert passed is True
    assert score == 1.0


# ---------------------------------------------------------------------------
# VQA match
# ---------------------------------------------------------------------------

from tests.metrics.vqa_match import evaluate as eval_vqa


def _make_vqa_gt(required_facts, acceptable_keywords, is_tn=False, min_length=5):
    return {
        "items": [{
            "question": "Test?",
            "required_facts": required_facts,
            "acceptable_keywords": acceptable_keywords,
            "is_true_negative": is_tn,
            "min_answer_length": min_length,
        }]
    }


def test_vqa_passes_with_all_facts():
    gt = _make_vqa_gt(["lion", "gorilla"], ["lion", "gorilla"])
    passed, score, _ = eval_vqa([], "I see a lion and a gorilla in the image.", gt, threshold=0.5)
    assert passed is True
    assert score == 1.0


def test_vqa_passes_with_half_facts():
    gt = _make_vqa_gt(["lion", "gorilla"], ["lion"])
    passed, score, _ = eval_vqa([], "I see a large lion.", gt, threshold=0.5)
    assert passed is True  # 1/2 = 0.5, need ceil(0.5*2)=1


def test_vqa_fails_missing_keyword():
    gt = _make_vqa_gt(["lion"], ["lion", "cat", "feline"])
    passed, _, _ = eval_vqa([], "The image shows an animal.", gt, threshold=0.5)
    assert passed is False  # "lion"/"cat"/"feline" not in answer


def test_vqa_tn_denies_presence():
    gt = _make_vqa_gt([], [], is_tn=True, min_length=3)
    passed, score, _ = eval_vqa([], "No, there is no tiger visible.", gt)
    assert passed is True
    assert score == 1.0


def test_vqa_tn_fails_if_affirms():
    gt = _make_vqa_gt([], [], is_tn=True, min_length=3)
    passed, score, _ = eval_vqa([], "Yes, I see a tiger.", gt)
    assert passed is False
    assert score == 0.0


def test_vqa_fails_too_short():
    gt = _make_vqa_gt(["lion"], ["lion"], min_length=20)
    passed, _, _ = eval_vqa([], "Lion.", gt)
    assert passed is False


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

from tests.metrics.iou import _compute_iou, evaluate as eval_iou


def test_iou_perfect_overlap():
    box = [0, 0, 100, 100]
    assert abs(_compute_iou(box, box) - 1.0) < 1e-9


def test_iou_no_overlap():
    a = [0, 0, 10, 10]
    b = [20, 20, 30, 30]
    assert _compute_iou(a, b) == 0.0


def test_iou_half_overlap():
    a = [0, 0, 10, 10]
    b = [5, 0, 15, 10]  # 5x10 overlap, union=150
    iou = _compute_iou(a, b)
    assert abs(iou - 50 / 150) < 1e-6


def _make_iou_item(bbox, present=True, label="lion"):
    return {
        "items": [{
            "frame_path": "frames/t006s.jpg",
            "query": "lion",
            "objects": [{"label": label, "bbox": bbox, "present": present}]
        }]
    }


class _FakeItem:
    def __init__(self, boxes, labels=None):
        import json
        self.output = json.dumps({
            "success": True,
            "boxes": boxes,
            "labels": labels or (["animal"] * len(boxes)),
        })


def test_iou_eval_passes_high_iou():
    gt = _make_iou_item([0, 0, 100, 100])
    items = [_FakeItem([[0, 0, 100, 100]])]
    passed, score, detail = eval_iou(items, "", gt, threshold=0.5, item_index=0)
    assert passed is True
    assert abs(score - 1.0) < 1e-6


def test_iou_eval_fails_low_iou():
    gt = _make_iou_item([0, 0, 100, 100])
    items = [_FakeItem([[200, 200, 300, 300]])]
    passed, score, detail = eval_iou(items, "", gt, threshold=0.5, item_index=0)
    assert passed is False
    assert score == 0.0


def test_iou_tn_passes_no_boxes():
    gt = _make_iou_item(None, present=False, label="tiger")
    items = [_FakeItem([])]
    passed, score, detail = eval_iou(items, "", gt, threshold=0.5, item_index=0)
    assert passed is True


def test_iou_tn_fails_with_boxes():
    gt = _make_iou_item(None, present=False, label="tiger")
    items = [_FakeItem([[0, 0, 100, 100]])]
    passed, score, detail = eval_iou(items, "", gt, threshold=0.5, item_index=0)
    assert passed is False


# ---------------------------------------------------------------------------
# Summary check
# ---------------------------------------------------------------------------

from tests.metrics.summary_check import evaluate as eval_summary


def test_summary_passes_good_coverage():
    gt = {"key_points": ["variable elimination", "Wednesday", "efficient", "inference"]}
    # Has 3/4 key points → 75% ≥ 60%
    summary = "variable elimination is an efficient technique. No lecture on Wednesday."
    passed, score, detail = eval_summary([], summary, gt, threshold=0.6)
    assert passed is True
    assert score >= 0.6


def test_summary_fails_poor_coverage():
    gt = {"key_points": ["variable elimination", "Wednesday", "efficient", "inference"]}
    summary = "Today we discuss algorithms."
    passed, score, detail = eval_summary([], summary, gt, threshold=0.6)
    assert passed is False


# ---------------------------------------------------------------------------
# Scene check
# ---------------------------------------------------------------------------

from tests.metrics.scene_check import evaluate as eval_scene


class _SceneFakeItem:
    def __init__(self, num_scenes, scenes):
        import json
        self.output = json.dumps({
            "success": True,
            "num_scenes": num_scenes,
            "scenes": scenes,
        })


def _make_scenes(cuts: list[float]) -> list[dict]:
    scenes = [{"scene_index": 0, "start_seconds": 0.0, "end_seconds": cuts[0] if cuts else 90.0}]
    for i, cut in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else 90.0
        scenes.append({"scene_index": i + 1, "start_seconds": cut, "end_seconds": end})
    return scenes


def test_scene_passes_correct():
    gt = {"scene_count": 12, "expected_cuts": [1.0, 8.0, 16.5], "tolerance_s": 1.5}
    scenes = _make_scenes([1.0, 8.5, 16.0])  # all within 1.5s of GT cuts
    items = [_SceneFakeItem(12, scenes)]
    passed, score, detail = eval_scene(items, "", gt, threshold=1.5)
    assert passed is True


def test_scene_fails_wrong_count():
    gt = {"scene_count": 12, "expected_cuts": [1.0], "tolerance_s": 1.5}
    scenes = _make_scenes([1.0])
    items = [_SceneFakeItem(20, scenes)]  # 20 scenes vs GT 12, diff=8 > tol=3
    passed, score, detail = eval_scene(items, "", gt, threshold=1.5)
    assert passed is False


# ---------------------------------------------------------------------------
# Audio check
# ---------------------------------------------------------------------------

from tests.metrics.audio_check import evaluate as eval_audio


class _AudioFakeItem:
    def __init__(self, segments_count):
        import json
        self.output = json.dumps({"success": True, "segments_count": segments_count})


def test_audio_passes_within_tol():
    gt = {"speech_segments": [{}] * 13}  # 13 speech segs
    items = [_AudioFakeItem(14)]  # diff=1 ≤ tol=3
    passed, score, detail = eval_audio(items, "", gt, threshold=3.0)
    assert passed is True


def test_audio_fails_outside_tol():
    gt = {"speech_segments": [{}] * 13}
    items = [_AudioFakeItem(5)]  # diff=8 > tol=3
    passed, score, detail = eval_audio(items, "", gt, threshold=3.0)
    assert passed is False
