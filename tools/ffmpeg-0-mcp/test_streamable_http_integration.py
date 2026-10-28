"""
Integration test for FFmpeg MCP Server over StreamableHTTP.

This test requires a running server (default: http://127.0.0.1:8000/mcp)
and an accessible test video file.

Usage:
    export MCP_URL="http://127.0.0.1:8000/mcp"
    export TEST_VIDEO="/workspaces/VideoSearcher-AgenticAI/data/lecture_example.mp4"
    pytest -q tools/ffmpeg-0-mcp/test_streamable_http_integration.py
"""

import asyncio
import json
import os
from typing import Optional
from pathlib import Path

import pytest

from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession
from mcp.types import TextContent


MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")
TEST_VIDEO = os.getenv(
    "TEST_VIDEO", 
    "/workspaces/VideoSearcher-AgenticAI/data/lecture_example.mp4",
)


def _server_available(url: str) -> bool:
    """Quickly check that the MCP endpoint is reachable via HTTP HEAD/GET."""
    import httpx

    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(url.replace("/mcp", "/health"), timeout=2.0)
            return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.asyncio
async def test_streamable_http_extract_audio():
    if not _server_available(MCP_URL):
        pytest.skip(f"MCP server not reachable at {MCP_URL}; skipping integration test")

    # Support container-mounted path (/data/...) by mapping to host repo path for existence check
    video_for_server = TEST_VIDEO
    host_check_path = TEST_VIDEO
    if not os.path.exists(host_check_path) and TEST_VIDEO.startswith("/data/"):
        # Map /data/<file> -> <repo_root>/data/<file>
        repo_root = Path(__file__).resolve().parents[2]
        mapped = repo_root / "data" / Path(TEST_VIDEO).name
        host_check_path = str(mapped)

    if not os.path.exists(host_check_path):
        pytest.skip(f"Test video not found at {host_check_path}")

    async with streamablehttp_client(MCP_URL) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Ensure the tool is advertised
            tools = await session.list_tools()
            tool_names = {t.name for t in tools.tools}
            assert "extract_audio" in tool_names

            # Call the tool
            result = await session.call_tool(
                "extract_audio", {"video_path": video_for_server}
            )

            # The server returns a TextContent JSON string
            assert result.content, "No content returned from tool call"
            content = result.content[0]
            assert isinstance(content, TextContent)

            # Parse JSON payload from the text
            payload = json.loads(content.text)
            assert payload.get("success") is True, payload
            assert "output_path" in payload
            assert payload.get("file_size", 0) >= 0
            assert payload.get("processing_time_ms", 0) >= 0
