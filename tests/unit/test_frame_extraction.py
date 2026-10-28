"""
Functional tests for the frame extraction MCP server (ffmpeg-frames, port 8006).

Calls the MCP server via HTTP. Requires the ffmpeg-frames container to be running.

Run inside agents-orchestrator container:
  pytest tests/unit/test_frame_extraction.py -v
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def mcp_client_factory(frames_mcp_url: str):
    try:
        from videosearcher_agents.mcp_streamable_client import create_mcp_client
        return create_mcp_client, frames_mcp_url
    except ImportError as exc:
        pytest.skip(f"MCP client not importable: {exc}")


@pytest.fixture(scope="module")
def sample_video(fixture_dir: str) -> str:
    candidates = [
        "lionandgorilla.mp4",
        "tedtalk.mp4",
        "sample_video.mp4",
        "test_video.mp4",
    ]
    for name in candidates:
        p = Path(fixture_dir) / name
        if p.exists():
            return str(p)
    pytest.skip(f"No sample video found in {fixture_dir}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_frames_returns_frame_list(mcp_client_factory, sample_video, data_dir):
    create_mcp_client, url = mcp_client_factory
    async with create_mcp_client(url) as client:
        result = await client.call_tool(
            "extract_frames",
            {
                "video_file": sample_video,
                "t_start": 0.0,
                "t_end": 3.0,
                "fps": 1,
            },
        )
    assert isinstance(result, dict), "extract_frames must return a dict"
    assert result.get("success") is True, f"Expected success, got: {result}"
    frames = result.get("frames", [])
    assert len(frames) > 0, "At least one frame must be extracted"


@pytest.mark.asyncio
async def test_extracted_frames_exist_on_disk(mcp_client_factory, sample_video, data_dir):
    create_mcp_client, url = mcp_client_factory
    async with create_mcp_client(url) as client:
        result = await client.call_tool(
            "extract_frames",
            {
                "video_file": sample_video,
                "t_start": 0.0,
                "t_end": 2.0,
                "fps": 1,
            },
        )
    frames = result.get("frames", [])
    for frame in frames:
        path = frame.get("frame_path") or frame.get("path")
        assert path and Path(path).exists(), f"Frame file does not exist: {path}"


@pytest.mark.asyncio
async def test_frame_count_matches_fps_and_duration(mcp_client_factory, sample_video):
    create_mcp_client, url = mcp_client_factory
    t_start, t_end, fps = 0.0, 5.0, 2
    async with create_mcp_client(url) as client:
        result = await client.call_tool(
            "extract_frames",
            {"video_file": sample_video, "t_start": t_start, "t_end": t_end, "fps": fps},
        )
    frames = result.get("frames", [])
    expected = math.floor((t_end - t_start) * fps)
    # Allow ±1 tolerance for rounding
    assert abs(len(frames) - expected) <= 1, (
        f"Expected ~{expected} frames at {fps}fps over {t_end - t_start}s, got {len(frames)}"
    )


@pytest.mark.asyncio
async def test_timestamps_within_requested_range(mcp_client_factory, sample_video):
    create_mcp_client, url = mcp_client_factory
    t_start, t_end = 1.0, 4.0
    async with create_mcp_client(url) as client:
        result = await client.call_tool(
            "extract_frames",
            {"video_file": sample_video, "t_start": t_start, "t_end": t_end, "fps": 1},
        )
    frames = result.get("frames", [])
    for frame in frames:
        ts = frame.get("timestamp")
        if ts is not None:
            assert t_start - 0.5 <= float(ts) <= t_end + 0.5, (
                f"Timestamp {ts} outside requested range [{t_start}, {t_end}]"
            )
