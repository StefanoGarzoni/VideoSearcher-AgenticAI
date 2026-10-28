"""
VQA match metric — fact coverage + keyword presence.

Pass iff:
  - len(answer) >= min_answer_length
  - at least ceil(threshold * len(required_facts)) facts appear (case-insensitive)
  - at least one acceptable_keyword appears
  - for true-negatives: answer denies presence ("no", "not", "none", "cannot see", etc.)
"""
from __future__ import annotations

import math
import re


_DENY_WORDS = {"no", "not", "none", "absent", "cannot", "can't", "isn't", "doesn't", "never", "zero"}

# Agent-plumbing patterns that must NOT be scored as a VQA answer. A specialized
# agent's contract line ("DONE: output_path=…"/"ERROR: …") and any echoed file
# path are transport, not content. Without stripping them, a metric can pass on a
# substring of the *input filename* (e.g. "f-35" inside "/…/f-35_t15.jpg") instead
# of a real description — a false positive. We remove them before length/keyword/
# fact matching; if nothing substantive remains, the answer fails as too short.
_CONTRACT_LINE_RE = re.compile(r"^\s*(DONE|ERROR)\b.*$", re.IGNORECASE | re.MULTILINE)
_KV_PATH_RE = re.compile(r"\b\w*(?:path|archive|dir|file)\w*\s*=\s*\S+", re.IGNORECASE)
_BARE_PATH_RE = re.compile(r"/[\w./@+\-]+")


def _strip_plumbing(text: str) -> str:
    """Remove agent contract lines and file paths so only the answer prose remains."""
    t = _CONTRACT_LINE_RE.sub(" ", text)
    t = _KV_PATH_RE.sub(" ", t)
    t = _BARE_PATH_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def _contains_any(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(t.lower() in lower for t in terms)


_DENY_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _DENY_WORDS) + r")\b"
)


def _denies_presence(text: str) -> bool:
    # Whole-word match only: bare substring matching let "no"/"not" fire on
    # "know", "another", "northern", etc. — turning affirmative answers into
    # false denials for true-negative scenarios.
    return bool(_DENY_RE.search(text.lower()))


def evaluate(
    new_items: list,
    final_output: str,
    gt_data: dict,
    threshold: float = 0.5,
    item_index: int | None = None,
    **_kw,
) -> tuple[bool, float | None, str]:
    """
    Evaluate final_output against a VQA ground-truth item.
    Returns (passed, fact_coverage_score, detail).
    """
    items = gt_data.get("items", [])
    if not items:
        return False, None, "gt has no items"
    if item_index is not None and item_index < len(items):
        gt_item = items[item_index]
    else:
        gt_item = items[0]

    required_facts: list[str] = gt_item.get("required_facts", [])
    acceptable_keywords: list[str] = gt_item.get("acceptable_keywords", [])
    min_length: int = gt_item.get("min_answer_length", 5)
    is_tn: bool = gt_item.get("is_true_negative", False)
    question: str = gt_item.get("question", "")

    # Score the agent's actual answer text — the VQA tool output when the
    # orchestrator's final message is just a DONE/path wrapper — never the
    # path/contract plumbing.
    from .utils import extract_answer_text
    answer = extract_answer_text(new_items, final_output)

    # Length check
    if len(answer) < min_length:
        return False, 0.0, f"answer too short after stripping plumbing ({len(answer)} < {min_length})"

    if is_tn:
        passed = _denies_presence(answer)
        score = 1.0 if passed else 0.0
        detail = (
            f"TN q='{question[:40]}': denies_presence={'yes' if passed else 'no'} "
            f"(answer snippet: '{answer[:60]}')"
        )
        return passed, score, detail

    # Fact coverage
    n_required = len(required_facts)
    if n_required == 0:
        fact_score = 1.0
    else:
        n_found = sum(1 for f in required_facts if f.lower() in answer.lower())
        fact_score = n_found / n_required
    min_facts_needed = math.ceil(threshold * max(n_required, 1))
    n_found_int = round(fact_score * n_required)
    facts_pass = n_found_int >= min_facts_needed

    # Keyword check (OR-logic)
    kw_pass = _contains_any(answer, acceptable_keywords) if acceptable_keywords else True

    passed = facts_pass and kw_pass
    detail = (
        f"q='{question[:40]}': "
        f"facts={n_found_int}/{n_required} (need≥{min_facts_needed}), "
        f"keyword={'found' if kw_pass else 'missing'}, "
        f"answer='{answer[:60]}'"
    )
    return passed, round(fact_score, 4), detail
