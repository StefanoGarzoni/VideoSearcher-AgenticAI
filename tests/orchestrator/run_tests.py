"""
Standalone entrypoint for orchestrator tests (no pytest required).

Useful when running outside a pytest context or in CI pipelines that
prefer a simple `python run_tests.py` invocation.

Usage (inside agents-orchestrator container):
  python tests/orchestrator/run_tests.py
  python tests/orchestrator/run_tests.py --scenarios SC-01,SC-02 --levels vague,clear
  python tests/orchestrator/run_tests.py --scenarios SC-06 --levels algorithmic

All configuration can also be passed via environment variables — see runner.py.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Allow running from the project root without installing the package
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_AGENTS_ROOT = str(Path(_PROJECT_ROOT) / "agents-container")
if _AGENTS_ROOT not in sys.path:
    sys.path.insert(0, _AGENTS_ROOT)

from tests.scenarios import ALL_SCENARIOS
from tests.orchestrator.runner import run_orchestrator_tests, load_model_configs_from_env

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orchestrator integration test runner")
    p.add_argument(
        "--scenarios",
        default=os.getenv("TEST_SCENARIO_IDS", ""),
        help="Comma-separated scenario IDs (default: all). E.g. SC-01,SC-02",
    )
    p.add_argument(
        "--levels",
        default=os.getenv("TEST_PROMPT_LEVELS", "vague,clear,algorithmic"),
        help="Comma-separated prompt levels (default: vague,clear,algorithmic)",
    )
    p.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "/data"),
        help="Root of the data directory (default: /data)",
    )
    p.add_argument(
        "--output-dir",
        default=os.getenv("TEST_RESULTS_DIR", str(Path(__file__).parent.parent / "results")),
        help="Directory for JSON result files",
    )
    return p.parse_args()


async def main() -> int:
    args = _parse_args()

    # Assign a session root so every test run writes to its own isolated directory.
    import uuid
    from datetime import datetime
    if not os.environ.get("SESSION_OUTPUT_ROOT"):
        session_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + "_test_" + uuid.uuid4().hex[:4]
        os.environ["SESSION_OUTPUT_ROOT"] = f"/data/runs/{session_id}"
    logger.info("SESSION_OUTPUT_ROOT: %s", os.environ["SESSION_OUTPUT_ROOT"])

    scenario_ids = [s.strip() for s in args.scenarios.split(",") if s.strip()] or None
    prompt_levels = [l.strip().lower() for l in args.levels.split(",") if l.strip()] or None
    model_configs = load_model_configs_from_env()

    logger.info("Models: %s", [m.display_name or m.model_name for m in model_configs])
    logger.info("Scenarios: %s", scenario_ids or "all")
    logger.info("Prompt levels: %s", prompt_levels or "all")
    logger.info("Data dir: %s", args.data_dir)

    results = await run_orchestrator_tests(
        scenarios=ALL_SCENARIOS,
        model_configs=model_configs,
        prompt_levels=prompt_levels,
        scenario_ids=scenario_ids,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    total = len(results)
    passed = sum(1 for r in results if r.overall_pass)
    errors = sum(1 for r in results if not r.success)

    logger.info("=" * 60)
    logger.info("Results: %d/%d handoff match | %d errors", passed, total, errors)
    logger.info("=" * 60)

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
