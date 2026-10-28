import os
import asyncio
import argparse
from pathlib import Path
from typing import Optional

from agents import Agent, Runner, function_tool, FunctionTool
import logging

from videosearcher_agents.llm_utils import build_llm, resolve_llm_config
from videosearcher_agents.mcp_streamable_client import create_mcp_client

logger = logging.getLogger(__name__)


class SceneDetectAgent:
    """
    Scene Detect Agent using OpenAI Agents SDK with StreamableHTTP MCP Client.

    Connects to the Scene Detect MCP Server via HTTP to detect scene boundaries
    in a video file or archive using PySceneDetect.

    Architecture:
        Agent -> StreamableHTTP Client -> HTTP -> MCP Server -> PySceneDetect
    """

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_base_url: Optional[str] = None,
    ):
        cfg = resolve_llm_config(
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
            llm_provider=llm_provider,
            llm_base_url=llm_base_url,
        )
        self.model = build_llm(cfg)
        logger.info(f"Initializing SceneDetectAgent with LLM model: {cfg.model}")

        self.agent = Agent(
            name="SceneDetectAgent",
            instructions="""Detect scene boundaries in a video and report the result.

            STEP 1 — Call the tool immediately. Never ask for clarification.
            Extract the video path and optional threshold from the input and call
            detect_scenes(video_path=<path>) or detect_scenes(video_path=<path>, threshold=<value>).

            STEP 2 — Report the result in this exact format:
            If success:
              "DONE: output_path=<output_path>"
            If error:
              "ERROR: <message>"

            Do NOT add commentary, plans, or extra text. Return ONLY the one-line result above.""",
            model=self.model,
            tools=[detect_scenes],
        )


@function_tool
async def detect_scenes(video_path: str, threshold: float = 27.0) -> dict:
    """
    Detect scene boundaries in a video via the Scene Detect MCP Server (HTTP).

    Accepts a raw .mp4 OR a .tar.gz archive containing a .mp4. Uses PySceneDetect
    ContentDetector to find cut points. Writes scenes.json to the output directory.

    Args:
        video_path: Path to a .mp4 file or a .tar.gz archive containing a .mp4.
        threshold: ContentDetector sensitivity threshold (default 27.0).
    Returns:
        Dictionary with output_path, num_scenes, scenes, threshold,
        processing_time_ms, and success status.
    """
    if not Path(video_path).exists():
        return {"success": False, "error": f"File not found: {video_path}"}
    try:
        logger.info(f"Scene detect tool called for: {video_path} (threshold={threshold})")
        mcp_server_url = os.getenv("SCENE_DETECT_MCP_SERVER_URL", "http://localhost:8025/mcp")
        async with create_mcp_client(mcp_server_url) as client:
            tools = await client.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            if "detect_scenes" not in [t.name for t in tools]:
                return {"success": False, "error": "detect_scenes tool not found on MCP server"}

            session_root = os.environ.get("SESSION_OUTPUT_ROOT")
            extra = {"output_dir": f"{session_root}/scenes"} if session_root else {}
            result = await client.call_tool("detect_scenes", {"video_path": video_path, "threshold": threshold, **extra})
            logger.info(f"Tool result: {result}")

            if isinstance(result, dict) and result.get("success"):
                return {
                    "output_path": result.get("output_path"),
                    "num_scenes": result.get("num_scenes"),
                    "scenes": result.get("scenes"),
                    "threshold": result.get("threshold"),
                    "processing_time_ms": result.get("processing_time_ms"),
                    "success": True,
                }
            error = result.get("error", "Unknown error from MCP server") if isinstance(result, dict) else str(result)
            return {"success": False, "error": error}
    except Exception as e:
        logger.error(f"Scene detection failed: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Scene detection failed: {str(e)}"}


async def example_usage(video_path: str):
    agent = SceneDetectAgent().agent
    print(f"Initialized: {agent}\n")
    result = await Runner.run(agent, f"Detect scenes in the video at {video_path}")
    print(f"Scene Detect Result: {result}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Scene Detect Agent — detect scene boundaries in a video")
    parser.add_argument("video_path", nargs="?", default="/data/raw", help="Path to .mp4 or .tar.gz archive (default: %(default)s)")
    args = parser.parse_args()
    asyncio.run(example_usage(args.video_path))
