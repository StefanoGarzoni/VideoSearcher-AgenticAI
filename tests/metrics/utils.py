"""Shared utilities for metric evaluation."""
from __future__ import annotations

import ast
import json
import re
import tarfile


def scan_tool_outputs(new_items: list) -> list[dict]:
    """
    Return list of parsed dicts from all tool-call output items.

    Tries JSON parsing first, then backtick-quoted Python dict literals
    (e.g. `{'key': val, ...}`) that sub-agents include in verbose text outputs.
    Uses duck typing so it works without importing the SDK item types.
    """
    results: list[dict] = []
    for item in new_items:
        output = getattr(item, "output", None)
        if output is None:
            continue
        text = str(output)
        _before = len(results)
        # Try pure JSON first
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                results.append(data)
                continue
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        # Try triple-backtick code blocks (```json\n{...}\n```)
        for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
            raw = m.group(1)
            parsed = None
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            if parsed is None:
                try:
                    parsed = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    pass
            if isinstance(parsed, dict):
                results.append(parsed)
                continue
            # Fallback: extract 4-float box arrays and labels from malformed JSON
            if '"boxes"' in raw:
                boxes = [
                    [float(g) for g in m.groups()]
                    for m in re.finditer(
                        r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',
                        raw,
                    )
                ]
                labels_m = re.search(r'"labels"\s*:\s*(\["[^"]*"(?:\s*,\s*"[^"]*")*\])', raw)
                try:
                    labels = json.loads(labels_m.group(1)) if labels_m else []
                except (json.JSONDecodeError, ValueError):
                    labels = []
                if boxes:
                    results.append({"boxes": boxes, "labels": labels, "success": True})
        # Try single backtick-quoted dicts (handles JSON booleans + malformed JSON)
        for m in re.finditer(r"`(\{[^`]+\})`", text, re.DOTALL):
            raw = m.group(1)
            parsed = None
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            if parsed is None:
                try:
                    parsed = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    pass
            if isinstance(parsed, dict):
                results.append(parsed)
                continue
            # Malformed JSON fallback: extract 4-float box arrays + labels
            if '"boxes"' in raw:
                boxes = [
                    [float(g) for g in bm.groups()]
                    for bm in re.finditer(
                        r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',
                        raw,
                    )
                ]
                labels_m = re.search(r'"labels"\s*:\s*(\["[^"]*"(?:\s*,\s*"[^"]*")*\])', raw)
                try:
                    labels = json.loads(labels_m.group(1)) if labels_m else []
                except (json.JSONDecodeError, ValueError):
                    labels = []
                if boxes:
                    results.append({"boxes": boxes, "labels": labels, "success": True})

        # Tolerant fallback: a balanced-brace JSON object anywhere in the text.
        # Handles small-model wrappers like  DONE: output_path=`{"success": true,
        # "boxes": [...]}`  where the regexes above miss the (un)closed wrapping.
        if len(results) == _before:
            for raw in _balanced_json_objects(text):
                parsed = None
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, TypeError, ValueError):
                    try:
                        parsed = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        parsed = None
                if isinstance(parsed, dict):
                    results.append(parsed)
    return results


def _balanced_json_objects(text: str) -> list[str]:
    """Return substrings of `text` that are balanced {...} objects (string-aware)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            depth = 0
            in_str = False
            esc = False
            j = i
            while j < n:
                ch = text[j]
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            out.append(text[i:j + 1])
                            break
                j += 1
            i = j + 1
        else:
            i += 1
    return out


# Plumbing patterns — agent contract lines and file paths that are transport, not
# answer content. Kept in sync with tests/metrics/vqa_match.py + scenarios/base.py.
_PLUMBING_LINE = re.compile(r"^\s*(DONE|ERROR)\b.*$", re.IGNORECASE | re.MULTILINE)
_PLUMBING_KV = re.compile(r"\b\w*(?:path|archive|dir|file)\w*\s*=\s*\S+", re.IGNORECASE)
_PLUMBING_PATH = re.compile(r"/[\w./@+\-]+")
_MIN_SUBSTANTIVE = 5


def _strip_answer(text: str) -> str:
    t = _PLUMBING_LINE.sub(" ", text or "")
    t = _PLUMBING_KV.sub(" ", t)
    t = _PLUMBING_PATH.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_answer_text(new_items: list, final_output: str = "") -> str:
    """
    The free-text answer the *system* produced — for VQA/vision scoring.

    Prefers the orchestrator's final message when it is substantive; otherwise
    falls back to the last substantive (non-plumbing) tool output. This matters
    because small orchestrator models often end a single-step/terminal agent call
    with a bare `DONE: output_path=…` wrapper (or an empty string) while the real
    VQA answer lives in the agent's tool output. VQA is the terminal step in the
    vision scenarios, so scanning tool outputs in reverse finds its answer first.
    """
    from_final = _strip_answer(final_output or "")
    if len(from_final) >= _MIN_SUBSTANTIVE:
        return from_final
    for item in reversed(list(new_items or [])):
        o = getattr(item, "output", None)
        if o is None:
            continue
        cleaned = _strip_answer(str(o))
        if len(cleaned) >= _MIN_SUBSTANTIVE:
            return cleaned
    return from_final


def find_output_with_keys(new_items: list, *keys: str) -> dict | None:
    """Return the first tool output dict that contains any of the given keys."""
    for d in scan_tool_outputs(new_items):
        if any(k in d for k in keys):
            return d
    return None


def find_all_outputs_with_key(new_items: list, key: str) -> list[dict]:
    """Return all tool output dicts that contain a specific key."""
    return [d for d in scan_tool_outputs(new_items) if key in d]


def extract_data_paths(new_items: list, *path_fragments: str) -> list[str]:
    """
    Extract /data/... paths from text output items that contain all given fragments.
    Covers 'DONE: output_path=<path>', 'Output archive: <path>', and bare paths.
    """
    paths: list[str] = []
    for item in new_items:
        output = getattr(item, "output", None)
        if output is None:
            continue
        text = str(output)
        for m in re.finditer(r"/data/[\w./\-]+", text):
            candidate = m.group(0).rstrip(".,;:)")
            if all(frag in candidate for frag in path_fragments):
                if candidate not in paths:
                    paths.append(candidate)
    return paths


def read_json_from_path(path: str) -> dict | None:
    """Read and parse a JSON file from disk."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_json_from_archive(archive_path: str, filename: str) -> dict | None:
    """Read and parse a JSON file from inside a .tar.gz archive."""
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(filename) or filename in member.name:
                    f = tar.extractfile(member)
                    if f:
                        return json.loads(f.read().decode("utf-8", errors="replace"))
    except Exception:
        pass
    return None


def read_text_from_archive(archive_path: str, *name_patterns: str) -> str:
    """Read text from a file inside a .tar.gz archive matching any name pattern."""
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            for pattern in name_patterns:
                for member in members:
                    if pattern in member.name:
                        f = tar.extractfile(member)
                        if f:
                            return f.read().decode("utf-8", errors="replace").strip()
            for member in members:
                if member.name.endswith(".txt"):
                    f = tar.extractfile(member)
                    if f:
                        return f.read().decode("utf-8", errors="replace").strip()
    except Exception:
        pass
    return ""
