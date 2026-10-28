"""
Shared pytest fixtures for all tests.

All configuration is sourced from environment variables so that the same
test suite can be run against different setups without modifying any code.

Environment variables:
  DATA_DIR          Root of the shared data volume (default: /data)
  TEST_FIXTURE_DIR  Override for raw test fixtures; defaults to DATA_DIR/raw
  TEST_RESULTS_DIR  Where JSON result files are written (default: tests/results/)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir() -> str:
    return os.getenv("DATA_DIR", "/data")


@pytest.fixture(scope="session")
def fixture_dir(data_dir: str) -> str:
    return os.getenv("TEST_FIXTURE_DIR", f"{data_dir}/fixtures/raw")


@pytest.fixture(scope="session")
def results_dir() -> str:
    default = str(Path(__file__).parent / "results")
    d = os.getenv("TEST_RESULTS_DIR", default)
    Path(d).mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def segmentation_mcp_url() -> str:
    return os.getenv(
        "IMAGE_SEGMENTATION_MCP_SERVER_URL",
        os.getenv("SEGMENTATION_MCP_SERVER_URL", "http://localhost:8008/mcp"),
    )


@pytest.fixture(scope="session")
def video_segmentation_mcp_url() -> str:
    return os.getenv("VIDEO_SEGMENTATION_MCP_SERVER_URL", "http://localhost:8009/mcp")


@pytest.fixture(scope="session")
def frames_mcp_url() -> str:
    return os.getenv("FRAMES_MCP_SERVER_URL", "http://localhost:8006/mcp")


# ---------------------------------------------------------------------------
# Derived vision fixtures — generated once per session from existing videos.
# Using ffmpeg to extract avoids committing binary files to git while keeping
# tests deterministic (same source video → same frame / clip every run).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def lionandgorilla_frame0(fixture_dir: str) -> str:
    """
    Single reference frame extracted at t=2s from lionandgorilla.mp4.
    Pre-populated in data/fixtures/raw/ to avoid write dependency.
    """
    out = Path(fixture_dir) / "lionandgorilla_frame0.jpg"
    src = Path(fixture_dir) / "lionandgorilla.mp4"
    if not out.exists():
        if not src.exists():
            pytest.skip(f"lionandgorilla.mp4 not found at {src}")
        subprocess.run(
            ["ffmpeg", "-i", str(src), "-ss", "2", "-frames:v", "1", str(out), "-y"],
            check=True,
            capture_output=True,
        )
    return str(out)


@pytest.fixture(scope="session")
def lionandgorilla_15s(fixture_dir: str) -> str:
    """
    15-second clip extracted from lionandgorilla.mp4.
    Pre-populated in data/fixtures/raw/ to avoid write dependency.
    """
    out = Path(fixture_dir) / "lionandgorilla_15s.mp4"
    if not out.exists():
        src = Path(fixture_dir) / "lionandgorilla.mp4"
        if not src.exists():
            pytest.skip(f"lionandgorilla.mp4 not found at {src}")
        subprocess.run(
            ["ffmpeg", "-ss", "0", "-t", "15", "-i", str(src), "-c", "copy", str(out), "-y"],
            check=True,
            capture_output=True,
        )
    return str(out)
