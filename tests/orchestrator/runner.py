"""
Orchestrator test runner.

Fully configurable — no hardcoded paths, models, or scenario lists.
All defaults come from environment variables; all defaults can be overridden
by passing explicit arguments to run_orchestrator_tests().

Environment variables (all optional with defaults):
  DATA_DIR              Root of the data directory (default: /data)
  TEST_RESULTS_DIR      Where to save JSON result files (default: tests/results)
  TEST_MODELS_JSON      JSON array of ModelConfig dicts; overrides auto-detection
  TEST_PROMPT_LEVELS    Comma-separated levels to run (default: vague,clear,algorithmic)
  TEST_SCENARIO_IDS     Comma-separated IDs to run (default: all)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from agents.items import (
    HandoffCallItem,
    HandoffOutputItem,
    MessageOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
)

from videosearcher_agents.orchestrator.main_agent import OrchestratorAgent

from tests.scenarios.base import (
    ConversationItem,
    ModelConfig,
    OrchestratorScenario,
    ResultCriteria,
    TestResult,
    VisionSuccessCriteria,
)

logger = logging.getLogger(__name__)

_ALL_PROMPT_LEVELS = ("vague", "clear", "algorithmic")

_DEFAULT_TIMEOUT_S = int(os.getenv("TEST_TIMEOUT_S", "300"))
_VISION_PREFLIGHT_TIMEOUT_S = int(os.getenv("VISION_PREFLIGHT_TIMEOUT_S", "150"))
_VQA_PREWARM_TIMEOUT_S = int(os.getenv("VQA_PREWARM_TIMEOUT_S", "7200"))


async def _wait_for_healthy(url: str, timeout_s: int = _VISION_PREFLIGHT_TIMEOUT_S) -> bool:
    """
    Poll a /health endpoint until it returns {"status": "healthy"} or times out.
    Returns True on success, False on timeout. Never raises.
    """
    health_url = url.replace("/mcp", "/health").rstrip("/")
    deadline = time.monotonic() + timeout_s
    logger.info("Pre-flight: waiting for %s to be healthy (timeout=%ds)", health_url, timeout_s)
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(health_url)
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status", "")
                    if status in ("healthy", "ok"):
                        logger.info("Pre-flight: %s is ready (%s)", health_url, status)
                        return True
        except Exception:
            pass
        await asyncio.sleep(5)
    logger.warning("Pre-flight: %s did not become healthy within %ds", health_url, timeout_s)
    return False


async def _prewarm_vqa(timeout_s: int = _VQA_PREWARM_TIMEOUT_S) -> None:
    """Load the VQA model once before any scenario timer starts.

    On this machine Phi-4 4-bit loading takes ~60 minutes (disk I/O bound).
    Running pre-warm outside the per-scenario timeout prevents test timeouts
    on the first VQA call.  Subsequent calls hit the cached _vlm global.
    """
    try:
        from videosearcher_agents.specialized.vqa_agent import _get_vlm
        logger.info(
            "Pre-warming VQA model (Phi-4 4-bit NF4, up to %d min)…",
            timeout_s // 60,
        )
        start = time.monotonic()
        loop = asyncio.get_event_loop()
        await asyncio.wait_for(
            loop.run_in_executor(None, _get_vlm),
            timeout=float(timeout_s),
        )
        elapsed = time.monotonic() - start
        logger.info("VQA model pre-warmed in %.0fs (%.1f min).", elapsed, elapsed / 60)
    except asyncio.TimeoutError:
        logger.warning(
            "VQA pre-warm timed out after %ds — VQA will attempt on-demand loading.", timeout_s
        )
    except Exception as exc:
        logger.warning("VQA pre-warm failed (%s) — VQA will attempt on-demand loading.", exc)


# ---------------------------------------------------------------------------
# Model config resolution
# ---------------------------------------------------------------------------

def load_model_configs_from_env() -> list[ModelConfig]:
    """
    Load model configs from TEST_MODELS_JSON env var (JSON array).
    Falls back to a single config built from LLM_ORCHESTRATOR_* env vars.
    """
    raw = os.getenv("TEST_MODELS_JSON")
    if raw:
        try:
            data = json.loads(raw)
            return [ModelConfig(**item) for item in data]
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("TEST_MODELS_JSON is invalid: %s — falling back to env vars", exc)

    # Auto-detect from LLM_ORCHESTRATOR_* env vars
    model_name = os.getenv("LLM_ORCHESTRATOR_MODEL") or os.getenv("LLM_MODEL", "local-model")
    return [
        ModelConfig(
            model_name=model_name,
            provider=os.getenv("LLM_ORCHESTRATOR_PROVIDER", "openai-chat"),
            base_url=os.getenv("LLM_ORCHESTRATOR_BASE_URL"),
            api_key=os.getenv("LLM_ORCHESTRATOR_API_KEY", "not-needed"),
            display_name=f"{model_name} (local)",
        )
    ]


# ---------------------------------------------------------------------------
# Handoff / conversation extraction
# ---------------------------------------------------------------------------

def _extract_handoff_sequence(new_items: list) -> list[str]:
    """
    Return ordered list of human-readable agent names invoked via handoffs.

    Agents are registered with .as_tool() so they appear as ToolCallItem with
    tool names ending in '_handoff' (e.g. 'audio_extraction_handoff').
    Converts to title-case display names: 'Audio Extraction'.
    """
    sequence = []
    for item in new_items:
        if isinstance(item, ToolCallItem):
            raw = getattr(item, "raw_item", None)
            tool_name = getattr(raw, "name", "") if raw else ""
            if tool_name.endswith("_handoff"):
                agent_name = tool_name.replace("_handoff", "").replace("_", " ").title()
                sequence.append(agent_name)
        elif isinstance(item, HandoffCallItem):
            target = (
                getattr(item, "target_agent", None)
                or getattr(item, "agent", None)
            )
            if target:
                sequence.append(str(target))
    return sequence


def _extract_conversation(new_items: list) -> list[ConversationItem]:
    """Convert SDK run items into serialisable ConversationItem objects."""
    items = []
    for i, item in enumerate(new_items):
        if isinstance(item, MessageOutputItem):
            content = ""
            if hasattr(item, "raw_item") and hasattr(item.raw_item, "content"):
                for block in item.raw_item.content or []:
                    if hasattr(block, "text"):
                        content += block.text
            items.append(ConversationItem(
                sequence=i,
                item_type="message",
                content=content,
                agent_role=getattr(item, "agent", {}).name
                    if hasattr(item, "agent") and item.agent else None,
            ))
        elif isinstance(item, ToolCallItem):
            raw = getattr(item, "raw_item", None)
            tool_name = ""
            args = {}
            if raw:
                if hasattr(raw, "name"):
                    tool_name = raw.name
                elif hasattr(raw, "function"):
                    tool_name = getattr(raw.function, "name", "")
                    try:
                        args = json.loads(getattr(raw.function, "arguments", "{}") or "{}")
                    except json.JSONDecodeError:
                        args = {}
            items.append(ConversationItem(
                sequence=i,
                item_type="tool_call",
                tool_name=tool_name,
                arguments=args,
            ))
        elif isinstance(item, ToolCallOutputItem):
            content = ""
            if hasattr(item, "output"):
                content = str(item.output)
            items.append(ConversationItem(
                sequence=i,
                item_type="tool_output",
                content=content[:2000],  # truncate very long outputs
            ))
        elif isinstance(item, (HandoffCallItem, HandoffOutputItem)):
            agent_name = str(
                getattr(item, "agent", None)
                or getattr(item, "target_agent", "unknown")
            )
            items.append(ConversationItem(
                sequence=i,
                item_type="handoff",
                tool_name=agent_name,
            ))
    return items


# ---------------------------------------------------------------------------
# Result criteria evaluation
# ---------------------------------------------------------------------------

def _compute_first_divergence(
    handoff_seq: list[str],
    expected_agents: list,
) -> Optional[str]:
    """
    Return the first expected agent that was NOT found in handoff_seq
    (longest-matched-prefix diagnostics).  None if all expected agents matched.
    """
    if not expected_agents:
        return None

    def _norm(s: str) -> str:
        return s.lower().replace(" ", "").replace("_", "")

    si = 0
    for exp in expected_agents:
        alternatives = {_norm(a) for a in exp.split("|")}
        matched = False
        while si < len(handoff_seq):
            if _norm(handoff_seq[si]) in alternatives:
                si += 1
                matched = True
                break
            si += 1
        if not matched:
            return exp
    return None


def _evaluate_result_criteria(
    rc: ResultCriteria,
    new_items: list,
    final_output: str,
    gt_dir: Path,
) -> tuple[Optional[bool], Optional[float], Optional[str]]:
    """
    Load ground truth, run the metric, return (passed, score, detail).
    Returns (None, None, error_message) on infrastructure failures.
    """
    gt_path = gt_dir / rc.ground_truth_ref
    try:
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, None, f"gt_load_error: {exc}"

    kw = dict(item_index=rc.item_index)
    try:
        if rc.metric == "wer":
            from tests.metrics.wer import evaluate
        elif rc.metric == "grep":
            from tests.metrics.grep_match import evaluate
        elif rc.metric == "vqa":
            from tests.metrics.vqa_match import evaluate
        elif rc.metric == "iou":
            from tests.metrics.iou import evaluate
        elif rc.metric == "detect":
            from tests.metrics.detection_presence import evaluate
        elif rc.metric == "summary":
            from tests.metrics.summary_check import evaluate
        elif rc.metric == "scene":
            from tests.metrics.scene_check import evaluate
        elif rc.metric == "audio":
            from tests.metrics.audio_check import evaluate
        else:
            return None, None, f"unknown metric: {rc.metric}"

        passed, score, detail = evaluate(
            new_items=new_items,
            final_output=final_output,
            gt_data=gt_data,
            threshold=rc.threshold,
            **kw,
        )
        return passed, score, detail
    except Exception as exc:
        logger.warning("ResultCriteria eval error (%s): %s", rc.metric, exc, exc_info=True)
        return None, None, f"eval_error: {exc}"


# ---------------------------------------------------------------------------
# Single test execution
# ---------------------------------------------------------------------------

async def _run_single(
    scenario: OrchestratorScenario,
    prompt_level: str,
    model_cfg: ModelConfig,
    data_dir: str,
    run_id: str,
    timeout_s: Optional[int] = None,
) -> TestResult:
    prompt = scenario.build_prompt(prompt_level, data_dir)
    effective_timeout = timeout_s or scenario.timeout_s or _DEFAULT_TIMEOUT_S
    logger.info(
        "[%s] %s / %s / %s (timeout=%ds)",
        run_id, scenario.scenario_id, prompt_level, model_cfg.model_name, effective_timeout,
    )

    orchestrator = OrchestratorAgent(
        llm_provider=model_cfg.provider,
        llm_model_name=model_cfg.model_name,
        llm_api_key=model_cfg.api_key,
        llm_base_url=model_cfg.base_url,
    )

    start = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            orchestrator.process_request_with_items(prompt),
            timeout=effective_timeout,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # process_request_with_items returns a dict {"final_output": ..., "new_items": [...]}
        final_output = result.get("final_output", "") if isinstance(result, dict) else getattr(result, "final_output", "")
        new_items = result.get("new_items", []) if isinstance(result, dict) else getattr(result, "new_items", [])
        new_items = new_items or []
        usage = result.get("usage") if isinstance(result, dict) else None
        usage = usage or {}

        handoff_seq = _extract_handoff_sequence(new_items)
        conversation = _extract_conversation(new_items)
        final_output_str = str(final_output)

        vision_pass: Optional[bool] = None
        if scenario.vision_criteria is not None:
            # Score the agent's actual answer (VQA tool output) rather than the
            # orchestrator's final wrapper line, which small models often reduce
            # to a bare DONE/path or empty string.
            from tests.metrics.utils import extract_answer_text
            answer_text = extract_answer_text(new_items, final_output_str)
            vision_pass = scenario.vision_criteria.answer_passes(answer_text)

        # Result criteria evaluation (ground-truth–backed correctness)
        result_pass: Optional[bool] = None
        result_score: Optional[float] = None
        result_detail: Optional[str] = None
        if scenario.result_criteria is not None:
            gt_dir = Path(__file__).parent.parent / "ground_truth"
            result_pass, result_score, result_detail = _evaluate_result_criteria(
                scenario.result_criteria, new_items, final_output_str, gt_dir
            )
            if result_pass is None:
                # The scenario declared a correctness contract but we could not
                # verify it (missing/broken GT, metric crash, unknown metric).
                # `overall_pass` treats result_pass=None as "no criteria → pass";
                # for a criteria-bearing scenario that would silently inflate the
                # pass rate (and the tuning objective). Count it as a failure so
                # an unverifiable result never masquerades as a passing one.
                result_pass = False
                result_detail = f"UNVERIFIED→fail: {result_detail}"
            logger.info(
                "  [result] %s metric=%s pass=%s score=%s | %s",
                scenario.scenario_id,
                scenario.result_criteria.metric,
                result_pass,
                result_score,
                result_detail,
            )

        effective_expected = scenario.get_expected_agents(prompt_level)
        first_divergence = _compute_first_divergence(handoff_seq, list(effective_expected))

        return TestResult(
            run_id=run_id,
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            prompt_level=prompt_level,
            model_name=model_cfg.model_name,
            success=True,
            handoff_sequence=handoff_seq,
            expected_agents=list(effective_expected),
            execution_time_ms=round(elapsed_ms, 1),
            orchestrator_output=final_output_str,
            conversation=[vars(c) for c in conversation],
            error=None,
            vision_pass=vision_pass,
            result_pass=result_pass,
            result_score=result_score,
            result_detail=result_detail,
            first_divergence=first_divergence,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    except asyncio.CancelledError:
        # #02 fix: never swallow cancellation. The per-scenario asyncio.wait_for
        # cancels this task on timeout; if we caught it via BaseException below
        # and returned a normal TestResult, wait_for would keep WAITING for the
        # task while the underlying stuck await never completes — the exact
        # mechanism behind a suite hanging for hours past every timeout.
        raise
    except BaseException as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "[%s] FAILED %s / %s: %s",
            run_id, scenario.scenario_id, prompt_level, exc,
        )
        return TestResult(
            run_id=run_id,
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            prompt_level=prompt_level,
            model_name=model_cfg.model_name,
            success=False,
            handoff_sequence=[],
            expected_agents=list(scenario.get_expected_agents(prompt_level)),
            execution_time_ms=round(elapsed_ms, 1),
            orchestrator_output="",
            conversation=[],
            error=str(exc),
            vision_pass=None,
            result_pass=None,
            result_score=None,
            result_detail=None,
            first_divergence=None,
        )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def run_orchestrator_tests(
    scenarios: list[OrchestratorScenario],
    model_configs: Optional[list[ModelConfig]] = None,
    prompt_levels: Optional[list[str]] = None,
    scenario_ids: Optional[list[str]] = None,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> list[TestResult]:
    """
    Run orchestrator scenarios against one or more models.

    Args:
        scenarios:      List of scenarios to run (usually ORCHESTRATOR_SCENARIOS).
        model_configs:  Models to test. Defaults to load_model_configs_from_env().
        prompt_levels:  Subset of ("vague", "clear", "algorithmic"). None = all.
        scenario_ids:   Subset of scenario IDs. None = all.
        data_dir:       Root of data directory. Defaults to DATA_DIR env var or /data.
        output_dir:     Where to save JSON results. Defaults to TEST_RESULTS_DIR env var.

    Returns:
        List of TestResult objects (one per scenario × prompt_level × model).
    """
    if model_configs is None:
        model_configs = load_model_configs_from_env()

    if prompt_levels is None:
        env_levels = os.getenv("TEST_PROMPT_LEVELS", "")
        prompt_levels = (
            [l.strip().lower() for l in env_levels.split(",") if l.strip()]
            or list(_ALL_PROMPT_LEVELS)
        )

    if scenario_ids is None:
        env_ids = os.getenv("TEST_SCENARIO_IDS", "")
        scenario_ids = (
            [s.strip() for s in env_ids.split(",") if s.strip()]
            or None
        )

    if data_dir is None:
        data_dir = os.getenv("DATA_DIR", "/data")

    if output_dir is None:
        default_output = Path(__file__).parent.parent / "results"
        output_dir = os.getenv("TEST_RESULTS_DIR", str(default_output))

    # Filter scenarios
    active = [
        s for s in scenarios
        if scenario_ids is None or s.scenario_id in scenario_ids
    ]

    if not active:
        logger.warning("No scenarios matched filter %s", scenario_ids)
        return []

    # Pre-flight: wait for vision MCP servers when any active scenario has
    # vision_criteria set (segmentation/VQA) or explicitly involves frame extraction.
    # The check is opt-in via vision_criteria so audio/text scenarios never trigger it.
    needs_frames = any(s.vision_criteria is not None for s in active)
    needs_seg = any(
        s.vision_criteria is not None and
        s.scenario_id not in ("SC-28", "SC-29")  # frame-only and single-VQA don't need seg server
        for s in active
    )
    if needs_frames:
        frames_url = os.getenv("FRAMES_MCP_SERVER_URL", "http://localhost:8006/mcp")
        await _wait_for_healthy(frames_url)
    if needs_seg:
        seg_url = os.getenv("IMAGE_SEGMENTATION_MCP_SERVER_URL", os.getenv("SEGMENTATION_MCP_SERVER_URL", "http://localhost:8008/mcp"))
        await _wait_for_healthy(seg_url)

    # Pre-warm VQA model if any scenario carries vision_criteria (implies a VQA call).
    # Phi-4 4-bit takes ~60 min to load on this machine; loading outside the scenario
    # timer means individual tests won't time out during the first VQA call.
    if os.getenv("VQA_PREWARM", "1") != "0" and needs_frames:
        await _prewarm_vqa()

    # TEST_RUN_ID lets an external driver (e.g. run_model_comparison.py) share one
    # run_id across several single-model invocations so the per-model result files
    # group into a single comparison set.
    run_id = os.getenv("TEST_RUN_ID") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    all_results: list[TestResult] = []

    for model_cfg in model_configs:
        logger.info("=== Model: %s ===", model_cfg.display_name or model_cfg.model_name)
        model_results: list[TestResult] = []
        out_path = _result_path(model_cfg, run_id, output_dir)

        for scenario in active:
            for level in prompt_levels:
                if level not in scenario.prompt_templates:
                    logger.debug(
                        "Skipping %s / %s — no template for level '%s'",
                        scenario.scenario_id, model_cfg.model_name, level,
                    )
                    continue
                result = await _run_single(scenario, level, model_cfg, data_dir, run_id)
                model_results.append(result)
                _log_result(result)
                # Save after every test so results survive interruption
                _save_results(model_results, out_path)

        all_results.extend(model_results)

    return all_results


def _log_result(result: TestResult) -> None:
    if not result.success:
        status = "ERROR"
    elif result.overall_pass:
        status = "PASS"
    else:
        status = "FAIL"

    extra = ""
    if result.vision_pass is not None:
        extra += f" vision={'PASS' if result.vision_pass else 'FAIL'}"
    if result.result_pass is not None:
        extra += f" result={'PASS' if result.result_pass else 'FAIL'}({result.result_score})"
    if result.first_divergence:
        extra += f" first_divergence={result.first_divergence!r}"

    logger.info(
        "  [%s] %s / %s — %s (%.0f ms) handoff=%s expected=%s%s",
        status,
        result.scenario_id,
        result.prompt_level,
        result.model_name,
        result.execution_time_ms,
        result.handoff_sequence,
        result.expected_agents,
        extra,
    )


def _result_path(model_cfg: ModelConfig, run_id: str, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{run_id}_{model_cfg.slug()}.json"


def _save_results(results: list[TestResult], path: Path) -> None:
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    logger.debug("Results checkpoint saved to %s (%d records)", path, len(results))
