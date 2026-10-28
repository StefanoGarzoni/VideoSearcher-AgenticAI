"""
Frame Extraction Agent
Calls the FFmpeg Frames MCP server (HTTP) to extract frames from a video window.
Strict-schema friendly: no **kwargs, only explicit parameters.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

from agents import Agent, function_tool
from videosearcher_agents.llm_utils import build_llm, resolve_llm_config
from videosearcher_agents.mcp_streamable_client import create_mcp_client


@function_tool
async def extract_frames_tool(
    video_file: str,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
    fps: float = 1.0,
    image_ext: str = "jpg",
) -> Dict[str, Any]:
    """
    Extract frames from a video between [t_start, t_end] at fps frames/sec.

    Returns:
      {
        "success": bool,
        "output_folder": str,
        "frames": [{"frame_path": str, "timestamp": float}, ...],
        "n_frames": int,
        ...
      }
    """
    if not Path(video_file).exists():
        return {
            "success": False,
            "error": f"File not found: {video_file}",
            "error_code": "VALIDATION_ERROR",
        }

    mcp_url = os.getenv("FRAMES_MCP_SERVER_URL", "http://localhost:8006/mcp")

    payload: Dict[str, Any] = {
        "video_file": video_file,
        "t_start": float(t_start),
        "fps": float(fps),
        "image_ext": image_ext,
    }
    if t_end is not None:
        payload["t_end"] = float(t_end)
    session_root = os.environ.get("SESSION_OUTPUT_ROOT")
    if session_root:
        payload["output_dir"] = f"{session_root}/frames"

    async with create_mcp_client(mcp_url) as client:
        tools = await client.list_tools()
        if "extract_frames" not in [t.name for t in tools]:
            return {
                "success": False,
                "error": "extract_frames tool not found on frames MCP server",
                "error_code": "TOOL_NOT_FOUND",
            }
        result = await client.call_tool("extract_frames", payload)
        if isinstance(result, dict) and result.get("success") and result.get("frames"):
            result["first_frame_path"] = result["frames"][0].get("frame_path", "")
            # Write to a stable location so VQA can recover if the LLM mangles the hash.
            Path("/tmp/last_extracted_frame.txt").write_text(result["first_frame_path"])
        return result


class FrameExtractionAgent:
    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_base_url: Optional[str] = None,
    ):
        cfg = resolve_llm_config(
            agent_type="specialized",
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
            llm_provider=llm_provider,
            llm_base_url=llm_base_url,
        )
        model = build_llm(cfg)

        self.agent = Agent(
            name="FrameExtractionAgent",
            model=model,
            instructions=(
                "You extract frames from videos for downstream vision tasks (VQA, segmentation).\n\n"
                "Call extract_frames_tool immediately with the video path. "
                "Use these defaults when not specified: t_end = t_start + 5, fps = 1.0, image_ext = 'jpg'. "
                "Never ask for these parameters — always call the tool with defaults.\n\n"
                "Always include in your response:\n"
                "  first_frame_path: <first_frame_path from tool result>\n"
                "  output_folder: <output_folder from tool result>\n"
                "  n_frames: <n_frames from tool result>\n\n"
                "The first_frame_path is the absolute path to the first extracted frame. "
                "The orchestrator uses it directly as input to vqa_handoff or segmentation_handoff."
            ),
            tools=[extract_frames_tool],
        )