"""
Pytest entry point for orchestrator integration tests.

Each scenario × prompt_level combination is a separate pytest test,
so you can filter with -k:

  pytest tests/orchestrator/ -v                          # all
  pytest tests/orchestrator/ -v -k "SC-01"               # one scenario, all levels
  pytest tests/orchestrator/ -v -k "SC-01 and clear"     # one specific case
  pytest tests/orchestrator/ -v -k "not algorithmic"     # skip algorithmic level

Environment variables that control the run (all optional):
  DATA_DIR              Root of the /data directory (default: /data)
  TEST_MODELS_JSON      JSON array of ModelConfig dicts
  TEST_PROMPT_LEVELS    Comma-separated levels to include (default: all)
  TEST_SCENARIO_IDS     Comma-separated scenario IDs to include (default: all)
  TEST_RESULTS_DIR      Where to write JSON results (default: tests/results/)
"""

from __future__ import annotations

import json
import os
import pytest

from tests.scenarios import ORCHESTRATOR_SCENARIOS, ModelConfig, OrchestratorScenario
from tests.orchestrator.runner import (
    _run_single,
    load_model_configs_from_env,
)


# ---------------------------------------------------------------------------
# Test matrix construction
# ---------------------------------------------------------------------------

def _active_scenario_ids() -> list[str] | None:
    env = os.getenv("TEST_SCENARIO_IDS", "")
    ids = [s.strip() for s in env.split(",") if s.strip()]
    return ids or None


def _active_prompt_levels() -> list[str]:
    env = os.getenv("TEST_PROMPT_LEVELS", "")
    levels = [l.strip().lower() for l in env.split(",") if l.strip()]
    return levels or ["vague", "clear", "algorithmic"]


def _build_test_matrix() -> list[tuple[OrchestratorScenario, str, ModelConfig]]:
    """
    Cartesian product of (active scenarios) × (active prompt levels) × (configured models).
    Each element becomes one pytest parametrize case.
    """
    model_configs = load_model_configs_from_env()
    active_ids = _active_scenario_ids()
    active_levels = _active_prompt_levels()

    matrix = []
    for scenario in ORCHESTRATOR_SCENARIOS:
        if active_ids is not None and scenario.scenario_id not in active_ids:
            continue
        for level in active_levels:
            if level not in scenario.prompt_templates:
                continue
            for model_cfg in model_configs:
                matrix.append((scenario, level, model_cfg))
    return matrix


def _test_id(val: object) -> str:
    if isinstance(val, OrchestratorScenario):
        return val.scenario_id
    if isinstance(val, ModelConfig):
        return val.slug()
    return str(val)


_MATRIX = _build_test_matrix()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario,prompt_level,model_cfg",
    _MATRIX,
    ids=[_test_id(s) + "-" + lvl + "-" + _test_id(m) for s, lvl, m in _MATRIX],
)
async def test_orchestrator_scenario(
    scenario: OrchestratorScenario,
    prompt_level: str,
    model_cfg: ModelConfig,
    data_dir: str,
    results_dir: str,
) -> None:
    """
    Runs one scenario/level/model combination and asserts:
    1. The orchestrator completed without an unhandled exception.
    2. The handoff sequence matches the expected agents (if defined).
    """
    run_id = "pytest"
    result = await _run_single(scenario, prompt_level, model_cfg, data_dir, run_id)

    # Always persist the result so it can be inspected after failure
    result_path = (
        f"{results_dir}/pytest_{scenario.scenario_id}_{prompt_level}_{model_cfg.slug()}.json"
    )
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    assert result.success, (
        f"{scenario.scenario_id}/{prompt_level}: orchestrator raised an error — {result.error}"
    )
    assert result.handoff_match, (
        f"{scenario.scenario_id}/{prompt_level}: handoff mismatch.\n"
        f"  Got:      {result.handoff_sequence}\n"
        f"  Expected: {result.expected_agents}"
    )
