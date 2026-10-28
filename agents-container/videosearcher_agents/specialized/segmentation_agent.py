"""
Segmentation Agent

Uses two MCP servers:
- image/frame segmentation MCP server
- video segmentation MCP server

Image output modes:
- masks: internal/pipeline mode, preserves old behavior
- segmented_image: final visual output for user-facing image segmentation

Video output modes:
- video: final segmented video only
- frames: segmented frames only
"""

from typing import Optional, Dict, Any
import os

from agents import Agent, function_tool
from videosearcher_agents.llm_utils import build_llm, resolve_llm_config
from videosearcher_agents.mcp_streamable_client import create_mcp_client


@function_tool
async def segment_image_tool(
    image_path: str,
    text_query: str,
    output_mode: str = "masks",
    output_image_path: Optional[str] = None,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    segmentation_type: str = "instance",
) -> Dict[str, Any]:
    """
    Segment a single image or frame.

    output_mode:
    - "masks": saves and returns mask_paths. Use for internal pipelines, VQA, or post-processing.
    - "segmented_image": saves and returns segmented_image_path. Use for final user-facing segmentation.
    """
    if output_mode not in {"masks", "segmented_image"}:
        return {
            "success": False,
            "error": "output_mode must be one of: masks, segmented_image",
            "error_code": "VALIDATION_ERROR",
        }

    mcp_url = os.getenv(
        "IMAGE_SEGMENTATION_MCP_SERVER_URL",
        os.getenv("SEGMENTATION_MCP_SERVER_URL", "http://localhost:8008/mcp"),
    )

    async with create_mcp_client(mcp_url) as client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]

        session_root = os.environ.get("SESSION_OUTPUT_ROOT")
        common_payload = {
            "text_query": text_query,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "segmentation_type": segmentation_type,
            "output_mode": output_mode,
        }
        if session_root:
            common_payload["output_dir"] = f"{session_root}/segmentation"

        if output_image_path is not None:
            common_payload["output_image_path"] = output_image_path

        if "segment_image" in tool_names:
            payload = {
                "image_path": image_path,
                **common_payload,
            }
            return await client.call_tool("segment_image", payload)

        if "segment_frame" in tool_names:
            payload = {
                "frame_path": image_path,
                **common_payload,
            }
            return await client.call_tool("segment_frame", payload)

        return {
            "success": False,
            "error": "No image segmentation tool found. Expected segment_image or segment_frame.",
            "error_code": "TOOL_NOT_FOUND",
            "available_tools": tool_names,
        }


@function_tool
async def segment_video_tool(
    video_path: str,
    text_query: str,
    output_mode: str = "video",
    frame_mode: str = "sampled",
    fps: float = 1.0,
    max_frames: Optional[int] = None,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    segmentation_type: str = "instance",
) -> Dict[str, Any]:
    """
    Segment a video.

    output_mode:
    - "video": saves and returns only segmented_video_path.
      max_frames is forced to None to prevent truncated videos.
    - "frames": saves and returns segmented frames.
      max_frames is allowed only for explicit frame/preview requests.
    """
    if output_mode not in {"video", "frames"}:
        return {
            "success": False,
            "error": "output_mode must be one of: video, frames",
            "error_code": "VALIDATION_ERROR",
        }

    if frame_mode not in {"all", "sampled"}:
        return {
            "success": False,
            "error": "frame_mode must be one of: all, sampled",
            "error_code": "VALIDATION_ERROR",
        }

    if fps <= 0:
        return {
            "success": False,
            "error": "fps must be greater than 0",
            "error_code": "VALIDATION_ERROR",
        }

    requested_max_frames = max_frames

    if output_mode == "video":
        max_frames = None

    mcp_url = os.getenv(
        "VIDEO_SEGMENTATION_MCP_SERVER_URL",
        "http://localhost:8009/mcp",
    )

    payload = {
        "video_path": video_path,
        "text_query": text_query,
        "output_mode": output_mode,
        "frame_mode": frame_mode,
        "fps": fps,
        "max_frames": max_frames,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "segmentation_type": segmentation_type,
    }

    async with create_mcp_client(mcp_url) as client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]

        if "segment_video" not in tool_names:
            return {
                "success": False,
                "error": "segment_video tool not found on video segmentation MCP server",
                "error_code": "TOOL_NOT_FOUND",
                "available_tools": tool_names,
            }

        result = await client.call_tool("segment_video", payload)

        if isinstance(result, dict):
            result.setdefault("requested_max_frames_from_agent", requested_max_frames)
            result.setdefault("effective_max_frames_from_agent", max_frames)

        return result


class SegmentationAgent:
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
            name="SegmentationAgent",
            model=model,
            instructions=(
                "You perform text-guided segmentation on images, frames, and videos.\n\n"

                "Use segment_image_tool when the input is a single image or frame, for example "
                ".jpg, .jpeg, .png, .bmp, or .webp.\n\n"

                "Use segment_video_tool when the input is a video, for example .mp4, .mov, .avi, "
                ".mkv, or .webm.\n\n"

                "For image/frame requests:\n"
                "- If the user asks for bounding boxes, coordinates, labels, positions, detected regions, "
                "spatial locations, or object locations, OR if output_mode='masks' is specified, call "
                "segment_image_tool with output_mode='masks'. Return mask_paths, scores, labels, and boxes.\n"
                "- If the user explicitly asks for a visually annotated image, segmented image, or "
                "output_mode='segmented_image' is specified, call segment_image_tool with "
                "output_mode='segmented_image'. Return segmented_image_path.\n"
                "- When in doubt, prefer output_mode='masks'.\n"
                "- After ANY successful masks call, end your response with a single-line backtick-quoted "
                "JSON summary of the raw tool result, e.g.: "
                "`{\"success\": true, \"boxes\": [[x1, y1, x2, y2], ...], \"labels\": [\"label1\", ...]}`\n"
                "- Do not ask whether the user wants masks or a segmented image when it is clear from context.\n\n"

                "For video requests:\n"
                "- If the user asks for a segmented video, full segmented video, entire segmented video, "
                "or output video, call segment_video_tool with output_mode='video', "
                "frame_mode='sampled', fps=1.0, and max_frames=None. Return segmented_video_path.\n"
                "- Never set max_frames for output_mode='video'. The video server intentionally ignores "
                "max_frames for video output to prevent truncated videos.\n"
                "- Use frame_mode='all' for output_mode='video' only if the user explicitly asks for every "
                "original frame, original FPS, full frame-by-frame segmentation, or full-frame-rate output.\n"
                "- If the user asks for all segmented frames, every frame segmented, frame-by-frame "
                "segmentation, or segmented frames, call segment_video_tool with output_mode='frames' "
                "and frame_mode='all'. Return segmented_frames_dir and segmented_frames.\n"
                "- If the user explicitly asks for a preview, sample, reduced-cost run, or limited number "
                "of segmented frames, use output_mode='frames', frame_mode='sampled', fps=1.0, and only then "
                "set max_frames to the requested limit.\n"
                "- The video server does not support output_mode='both'. If the user asks for both video "
                "and frames, choose the output mode that best satisfies the request or explain that one "
                "mode should be run at a time.\n\n"

                "Object query normalization:\n"
                "- For very specific object names, prefer a visually segmentable phrase. "
                "For example, for 'F-35 plane', use text_query='airplane' or 'fighter jet' if needed.\n"
                "- Do not change file paths. Normalize relative data paths to absolute /data/... paths.\n\n"

                "Scores are model confidence values, not IoU. Prefer higher-confidence outputs when "
                "summarizing results.\n\n"

                "If thresholds are not specified, use defaults: box_threshold=0.3, text_threshold=0.25, "
                "segmentation_type='instance'. Do not ask the user for these defaults."
            ),
            tools=[segment_image_tool, segment_video_tool],
        )