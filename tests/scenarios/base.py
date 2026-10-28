"""
Shared data structures for all test scenarios and results.
No business logic here — pure data definitions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional


# Agent-plumbing patterns that must not be mistaken for a VQA answer. A
# specialized agent's contract line ("DONE: output_path=…") and echoed file paths
# are transport, not content; without stripping them a keyword check can match a
# substring of the *input filename* (e.g. "f-35" in "/…/f-35_t15.jpg"). Keep this
# in sync with tests/metrics/vqa_match.py._strip_plumbing.
_PLUMBING_LINE = re.compile(r"^\s*(DONE|ERROR)\b.*$", re.IGNORECASE | re.MULTILINE)
_PLUMBING_KV = re.compile(r"\b\w*(?:path|archive|dir|file)\w*\s*=\s*\S+", re.IGNORECASE)
_PLUMBING_PATH = re.compile(r"/[\w./@+\-]+")


def _strip_plumbing(text: str) -> str:
    """Remove agent contract lines and file paths so only answer prose remains."""
    t = _PLUMBING_LINE.sub(" ", text or "")
    t = _PLUMBING_KV.sub(" ", t)
    t = _PLUMBING_PATH.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass(frozen=True)
class ResultCriteria:
    """
    Ground-truth–backed correctness evaluation for a scenario.

    metric: one of "wer", "grep", "vqa", "iou", "summary", "scene", "audio"
    ground_truth_ref: path relative to tests/ground_truth/
        (e.g. "videos/sample_video/transcript.json")
    threshold: metric-specific pass threshold (e.g. 0.30 for WER, 0.5 for IoU)
    item_index: optional index into a gt["items"] array (VQA, segmentation, grep)
    """

    metric: str
    ground_truth_ref: str
    threshold: float
    item_index: Optional[int] = None


@dataclass
class VisionSuccessCriteria:
    """
    Behavioral contract for vision scenarios.

    Used by the runner in addition to handoff_match when scenario.vision_criteria
    is set. All fields are optional so callers only specify what matters.

    vqa_keywords: OR-logic — at least one of these words must appear in the
        orchestrator's final output (case-insensitive). Use for phrasing variants
        where any one correct term is acceptable (e.g. "person"/"man"/"woman").

    required_keywords: AND-logic — ALL of these words must appear in the output.
        Use when the question has a specific multi-part answer that must be
        complete (e.g. both "lion" AND "gorilla" for a two-animal image).

    min_answer_length: orchestrator output must be longer than this to count as
        a valid VQA answer (guards against empty-string or single-word outputs).

    output_path_field: informational only — not used in the automated runner.
    """

    vqa_keywords: Optional[tuple[str, ...]] = None
    required_keywords: Optional[tuple[str, ...]] = None
    min_answer_length: int = 5
    output_path_field: str = "output_path"

    def answer_passes(self, text: str) -> bool:
        """Return True if text satisfies this criteria's VQA contract."""
        # Score the answer prose only — never the agent's path/contract plumbing,
        # so a filename substring can't masquerade as a real VQA answer.
        clean = _strip_plumbing(text)
        if not clean or len(clean) < self.min_answer_length:
            return False
        lower = clean.lower()
        if self.required_keywords:
            if not all(kw.lower() in lower for kw in self.required_keywords):
                return False
        if self.vqa_keywords:
            return any(kw.lower() in lower for kw in self.vqa_keywords)
        return True


@dataclass(frozen=True)
class OrchestratorScenario:
    """
    Defines a single orchestrator test scenario.

    prompt_templates: maps prompt level ("vague", "clear", "algorithmic") to a
        prompt string that may contain the placeholder {input_path}. The runner
        resolves {input_path} at runtime using the configured data directory.

    default_input_path: path relative to DATA_DIR (e.g. "fixtures/raw/sample_video.mp4"),
        or None when the scenario intentionally has no input (e.g. missing-data tests).

    expected_agents: ordered list of agent display names the orchestrator should
        invoke. Used for handoff_match evaluation.

    timeout_s: per-scenario timeout in seconds. Overrides the global TEST_TIMEOUT_S
        env var when set. Use for known-slow chains (e.g. full cross-modal).

    vision_criteria: when set, the runner evaluates a behavioral contract on the
        orchestrator output in addition to handoff_match. Only vision scenarios set
        this; audio/text scenarios leave it None.
    """

    scenario_id: str
    name: str
    difficulty: int
    prompt_templates: dict
    expected_agents: list
    default_input_path: Optional[str]
    timeout_s: Optional[int] = None
    vision_criteria: Optional[VisionSuccessCriteria] = None
    result_criteria: Optional[ResultCriteria] = None
    expected_agents_by_level: dict = field(default_factory=dict)

    def get_expected_agents(self, prompt_level: str) -> list:
        """Return level-specific expected_agents if defined, else fall back to default."""
        return self.expected_agents_by_level.get(prompt_level, self.expected_agents)

    def build_prompt(self, prompt_level: str, data_dir: str) -> str:
        key = prompt_level.lower()
        if key not in self.prompt_templates:
            raise ValueError(
                f"Scenario {self.scenario_id} has no prompt for level '{prompt_level}'. "
                f"Available: {list(self.prompt_templates.keys())}"
            )
        template = self.prompt_templates[key]
        if self.default_input_path is None:
            return template
        input_path = f"{data_dir.rstrip('/')}/{self.default_input_path}"
        return template.format(input_path=input_path)

    def resolve_input_path(self, data_dir: str) -> Optional[str]:
        if self.default_input_path is None:
            return None
        return f"{data_dir.rstrip('/')}/{self.default_input_path}"


@dataclass
class ModelConfig:
    """
    Configuration for a single LLM model under test.
    Populated from TEST_MODELS_JSON env var or LLM_ORCHESTRATOR_* env vars.
    """

    model_name: str
    provider: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    display_name: Optional[str] = None

    def slug(self) -> str:
        """URL-safe identifier for file naming."""
        return self.model_name.replace("/", "_").replace(" ", "-").replace(".", "_")


@dataclass
class ConversationItem:
    sequence: int
    item_type: str
    content: Optional[str] = None
    tool_name: Optional[str] = None
    arguments: Optional[dict] = None
    agent_role: Optional[str] = None


@dataclass
class TestResult:
    """
    Result of one scenario × prompt_level × model combination.
    handoff_match is computed automatically on construction.
    vision_pass is set by the runner when the scenario has vision_criteria.
    """

    run_id: str
    scenario_id: str
    scenario_name: str
    prompt_level: str
    model_name: str
    success: bool
    handoff_sequence: list
    expected_agents: list
    execution_time_ms: float
    orchestrator_output: str
    conversation: list
    error: Optional[str] = None
    vision_pass: Optional[bool] = None
    result_pass: Optional[bool] = None
    result_score: Optional[float] = None
    result_detail: Optional[str] = None
    first_divergence: Optional[str] = None
    # Time-to-first-token of the orchestrator LLM (model-serving latency). Measured
    # per model by run_model_comparison.py and stamped onto every record of that
    # model; None when not measured.
    ttft_ms: Optional[float] = None
    # Token usage for the whole orchestrator run of this scenario (sum across all
    # LLM turns), reported by the OpenAI Agents SDK. None on uninstrumented runs.
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    @property
    def handoff_match(self) -> bool:
        """
        True when expected_agents is an ordered subsequence of handoff_sequence.

        Subsequence (not exact-match) so redundant duplicate calls don't fail a
        test that otherwise called the right agents in the right order.
        Each entry in expected_agents may be a pipe-separated list of alternatives,
        e.g. "Reject|RequestClarification" matches either.
        Comparison is case/space/underscore-insensitive.
        """
        if not self.expected_agents:
            return True

        def _norm(s: str) -> str:
            return s.lower().replace(" ", "").replace("_", "")

        si = 0
        seq = self.handoff_sequence
        for exp in self.expected_agents:
            alternatives = {_norm(a) for a in exp.split("|")}
            matched = False
            while si < len(seq):
                if _norm(seq[si]) in alternatives:
                    si += 1
                    matched = True
                    break
                si += 1
            if not matched:
                return False
        return True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["handoff_match"] = self.handoff_match
        d["overall_pass"] = self.overall_pass
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TestResult":
        d.pop("handoff_match", None)
        d.pop("overall_pass", None)
        conv = d.get("conversation") or []
        d["conversation"] = [
            ConversationItem(**item) if isinstance(item, dict) else item
            for item in conv
        ]
        return cls(**d)

    @property
    def overall_pass(self) -> bool:
        """True when handoff_match AND vision_pass (if set) AND result_pass (if set)."""
        if not self.handoff_match:
            return False
        if self.vision_pass is not None and not self.vision_pass:
            return False
        if self.result_pass is not None and not self.result_pass:
            return False
        return True
