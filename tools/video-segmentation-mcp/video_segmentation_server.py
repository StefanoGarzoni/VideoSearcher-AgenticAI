import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import TransportSecuritySettings


try:
    from codecarbon import EmissionsTracker as _EmissionsTracker
    _CODECARBON_OK = True
except Exception:
    _CODECARBON_OK = False

_EMISSIONS_DIR = os.getenv("EMISSIONS_DIR", "/app/emissions")

def _energy_tracker(project_name):
    """Start a CodeCarbon tracker, or return None if unavailable. Never raises."""
    if not _CODECARBON_OK:
        return None
    try:
        os.makedirs(_EMISSIONS_DIR, exist_ok=True)
        t = _EmissionsTracker(project_name=project_name, output_dir=_EMISSIONS_DIR, log_level="error")
        t.start()
        return t
    except Exception:
        return None


logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

PORT = int(os.getenv("PORT", "8009"))
HOST = os.getenv("HOST", "0.0.0.0")

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
SEG_DIR = DATA_DIR / "segmentation" / "videos"
SEG_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SEGMENTATION_MCP_SERVER_URL = os.getenv(
    "IMAGE_SEGMENTATION_MCP_SERVER_URL",
    "http://vlm-segmentation-mcp-server:8008/mcp",
)

app = FastAPI(title="Video Segmentation MCP Server")

mcp = FastMCP(
    "video-segmentation-http",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)
mcp.settings.streamable_http_path = "/"

_mcp_session_ctx = None


@app.on_event("startup")
async def start_mcp_session_manager():
    global _mcp_session_ctx
    _mcp_session_ctx = mcp.session_manager.run()
    await _mcp_session_ctx.__aenter__()


@app.on_event("shutdown")
async def stop_mcp_session_manager():
    global _mcp_session_ctx
    if _mcp_session_ctx is not None:
        await _mcp_session_ctx.__aexit__(None, None, None)
        _mcp_session_ctx = None


@app.get("/health")
def health():
    return JSONResponse(
        {
            "status": "ok",
            "server": "video-segmentation",
            "image_segmentation_mcp_server_url": IMAGE_SEGMENTATION_MCP_SERVER_URL,
            "ffmpeg_available": shutil.which("ffmpeg") is not None,
            "ffprobe_available": shutil.which("ffprobe") is not None,
        }
    )


def _run_command(command: List[str]) -> None:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())


def _get_video_fps(video_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.returncode != 0:
        return 25.0

    value = completed.stdout.strip()

    try:
        if "/" in value:
            numerator, denominator = value.split("/")
            denominator = float(denominator)
            if denominator == 0:
                return 25.0
            return float(numerator) / denominator

        return float(value)
    except Exception:
        return 25.0


def _has_audio(video_path: str) -> bool:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        video_path,
    ]

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return completed.returncode == 0 and bool(completed.stdout.strip())


def _extract_frames(
    video_path: str,
    frames_dir: Path,
    frame_mode: str,
    fps: Optional[float],
    max_frames: Optional[int],
) -> List[str]:
    frames_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(frames_dir / "frame_%06d.jpg")
    command = ["ffmpeg", "-y", "-i", video_path]

    if frame_mode == "sampled":
        effective_fps = fps or 1.0
        command.extend(["-vf", f"fps={effective_fps}"])
    elif frame_mode == "all":
        pass
    else:
        raise ValueError("frame_mode must be one of: all, sampled")

    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])

    command.extend(["-q:v", "2", output_pattern])
    _run_command(command)

    return sorted(str(path) for path in frames_dir.glob("frame_*.jpg"))


def _build_video_from_frames(
    segmented_frames_dir: Path,
    source_video_path: str,
    output_video_path: Path,
    fps: float,
) -> str:
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_video_path = output_video_path.with_name(
        output_video_path.stem + "_no_audio.mp4"
    )

    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(segmented_frames_dir / "frame_%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(temporary_video_path),
    ]

    _run_command(command)

    if _has_audio(source_video_path):
        merge_command = [
            "ffmpeg",
            "-y",
            "-i",
            str(temporary_video_path),
            "-i",
            source_video_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_video_path),
        ]

        try:
            _run_command(merge_command)
            temporary_video_path.unlink(missing_ok=True)
            return str(output_video_path)
        except Exception as exc:
            logger.warning(
                "Could not merge audio, returning video without audio: %s",
                exc,
            )

    temporary_video_path.replace(output_video_path)
    return str(output_video_path)


def _decode_mcp_result(result: Any) -> Dict[str, Any]:
    """
    Decode MCP call_tool results returned either by:
    - our helper client, which may already return a dict
    - raw mcp.ClientSession, which may return:
        {"result": {...}}
        {"content": [{"type": "text", "text": "{...json...}"}]}
        a CallToolResult object with .content
    """
    if isinstance(result, dict):
        nested_result = result.get("result")
        if isinstance(nested_result, dict):
            return _decode_mcp_result(nested_result)

        if "success" in result:
            return result

        for attr in ("structured_content", "structuredContent"):
            value = result.get(attr)
            if isinstance(value, dict):
                return _decode_mcp_result(value)

        content = result.get("content")
        if content:
            texts = []

            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)

                if not text:
                    continue

                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return _decode_mcp_result(parsed)
                except Exception:
                    texts.append(text)

            return {
                "success": False,
                "error": "Could not parse MCP dict content as JSON",
                "raw_texts": texts,
                "raw_result": result,
            }

        return {
            "success": False,
            "error": "MCP result dict did not contain success, result, or parseable content",
            "raw_result": result,
        }

    for attr in ("structured_content", "structuredContent"):
        value = getattr(result, attr, None)
        if isinstance(value, dict):
            return _decode_mcp_result(value)

    content = getattr(result, "content", None)
    if content:
        texts = []

        for item in content:
            text = getattr(item, "text", None)

            if text is None and isinstance(item, dict):
                text = item.get("text")

            if not text:
                continue

            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return _decode_mcp_result(parsed)
            except Exception:
                texts.append(text)

        return {
            "success": False,
            "error": "Could not parse MCP object content as JSON",
            "raw_texts": texts,
        }

    return {
        "success": False,
        "error": f"Could not decode MCP result: {result!r}",
    }


async def _select_image_segmentation_tool(session: ClientSession) -> str:
    tools_result = await session.list_tools()
    tools = getattr(tools_result, "tools", tools_result)
    tool_names = [tool.name for tool in tools]

    if "segment_image" in tool_names:
        return "segment_image"

    if "segment_frame" in tool_names:
        return "segment_frame"

    raise RuntimeError(
        "Image segmentation server has neither segment_image nor segment_frame. "
        f"Available tools: {tool_names}"
    )


async def _segment_frame_as_image(
    session: ClientSession,
    image_tool_name: str,
    frame_path: str,
    segmented_frame_path: str,
    text_query: str,
    box_threshold: float,
    text_threshold: float,
    segmentation_type: str,
) -> Dict[str, Any]:
    common_payload = {
        "text_query": text_query,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "segmentation_type": segmentation_type,
        "output_mode": "segmented_image",
        "output_image_path": segmented_frame_path,
    }

    if image_tool_name == "segment_image":
        payload = {
            "image_path": frame_path,
            **common_payload,
        }
    else:
        payload = {
            "frame_path": frame_path,
            **common_payload,
        }

    result = await session.call_tool(image_tool_name, arguments=payload)
    return _decode_mcp_result(result)


def _copy_original_frame_to_output(frame_path: str, segmented_frame_path: Path) -> None:
    segmented_frame_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(frame_path, segmented_frame_path)


def _ensure_complete_output_sequence(
    frame_paths: List[str],
    segmented_frames_dir: Path,
) -> int:
    """
    FFmpeg expects a contiguous frame_%06d.jpg sequence.
    If any output frame is missing, copy the original frame unchanged.
    """
    filled = 0

    for frame_path in frame_paths:
        expected_output_frame = segmented_frames_dir / Path(frame_path).name

        if not expected_output_frame.exists():
            _copy_original_frame_to_output(frame_path, expected_output_frame)
            filled += 1

    if filled:
        logger.warning(
            "Filled %d missing segmented frames with original frames before video build",
            filled,
        )

    return filled


@mcp.tool()
async def segment_video(
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
    Segment a video using the image/frame segmentation MCP server.

    output_mode:
    - "video": saves and returns only the final segmented .mp4.
      max_frames is intentionally ignored for this mode to avoid truncated videos.
    - "frames": saves and returns only the segmented frames.
      max_frames is respected for this mode.

    frame_mode:
    - "sampled": process frames sampled at fps. Default for video output.
    - "all": process every original frame. Use only when explicitly requested.
    """
    video = Path(video_path)

    if not video.exists():
        return {"success": False, "error": f"video_path not found: {video_path}"}

    if output_mode not in {"video", "frames"}:
        return {
            "success": False,
            "error": "output_mode must be one of: video, frames",
        }

    if frame_mode not in {"all", "sampled"}:
        return {
            "success": False,
            "error": "frame_mode must be one of: all, sampled",
        }

    if fps <= 0:
        return {
            "success": False,
            "error": "fps must be greater than 0",
        }

    requested_max_frames = max_frames

    if output_mode == "video" and max_frames is not None:
        logger.warning(
            "Ignoring max_frames=%s for output_mode='video' to avoid truncated videos",
            max_frames,
        )
        max_frames = None

    run_id = uuid.uuid4().hex[:8]
    # '%' in the stem would be interpreted by ffmpeg's image2 pattern strings
    # built under this dir (frame_%06d.jpg) and fail valid files.
    run_dir = SEG_DIR / f"{video.stem.replace('%', '_')}_{run_id}"

    tmp_source_frames_dir = run_dir / "_tmp_source_frames"

    if output_mode == "video":
        segmented_frames_dir = run_dir / "_tmp_segmented_frames"
    else:
        segmented_frames_dir = run_dir / "segmented_frames"

    segmented_video_path = run_dir / f"{video.stem}_segmented.mp4"

    _tracker = _energy_tracker("video_segmentation")

    try:
        source_fps = _get_video_fps(str(video))
        effective_fps = source_fps if frame_mode == "all" else float(fps or 1.0)

        frame_paths = _extract_frames(
            video_path=str(video),
            frames_dir=tmp_source_frames_dir,
            frame_mode=frame_mode,
            fps=effective_fps,
            max_frames=max_frames,
        )

        if not frame_paths:
            return {
                "success": False,
                "error": "No frames extracted from video",
                "video_path": str(video),
                "requested_max_frames": requested_max_frames,
                "effective_max_frames": max_frames,
            }

        segmented_frames_dir.mkdir(parents=True, exist_ok=True)

        segmented_frames: List[Dict[str, Any]] = []
        failed_frames: List[Dict[str, Any]] = []

        async with streamablehttp_client(IMAGE_SEGMENTATION_MCP_SERVER_URL) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                image_tool_name = await _select_image_segmentation_tool(session)

                for index, frame_path in enumerate(frame_paths):
                    frame_name = Path(frame_path).name
                    segmented_frame_path = segmented_frames_dir / frame_name

                    segmentation = await _segment_frame_as_image(
                        session=session,
                        image_tool_name=image_tool_name,
                        frame_path=frame_path,
                        segmented_frame_path=str(segmented_frame_path),
                        text_query=text_query,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        segmentation_type=segmentation_type,
                    )

                    scores = segmentation.get("scores", [])
                    boxes = segmentation.get("boxes", [])
                    labels = segmentation.get("labels", [])

                    if not segmentation.get("success", False):
                        failed_frames.append(
                            {
                                "frame_index": index,
                                "frame_path": frame_path,
                                "error": segmentation.get("error", "Unknown segmentation error"),
                                "raw_segmentation_response": segmentation,
                            }
                        )

                        _copy_original_frame_to_output(frame_path, segmented_frame_path)

                        segmented_frames.append(
                            {
                                "frame_index": index,
                                "segmented_frame_path": str(segmented_frame_path),
                                "scores": [],
                                "boxes": [],
                                "labels": [],
                                "detected": False,
                                "success": False,
                            }
                        )
                        continue

                    image_server_output = segmentation.get("segmented_image_path")

                    if not image_server_output:
                        _copy_original_frame_to_output(frame_path, segmented_frame_path)

                        segmented_frames.append(
                            {
                                "frame_index": index,
                                "segmented_frame_path": str(segmented_frame_path),
                                "scores": scores,
                                "boxes": boxes,
                                "labels": labels,
                                "detected": False,
                                "success": True,
                            }
                        )
                        continue

                    if not Path(image_server_output).exists():
                        failed_frames.append(
                            {
                                "frame_index": index,
                                "frame_path": frame_path,
                                "error": "Image segmentation server returned segmented_image_path but file does not exist",
                                "segmented_image_path": image_server_output,
                                "raw_segmentation_response": segmentation,
                            }
                        )

                        _copy_original_frame_to_output(frame_path, segmented_frame_path)

                        segmented_frames.append(
                            {
                                "frame_index": index,
                                "segmented_frame_path": str(segmented_frame_path),
                                "scores": scores,
                                "boxes": boxes,
                                "labels": labels,
                                "detected": False,
                                "success": False,
                            }
                        )
                        continue

                    if Path(image_server_output) != segmented_frame_path:
                        shutil.copyfile(image_server_output, segmented_frame_path)

                    segmented_frames.append(
                        {
                            "frame_index": index,
                            "segmented_frame_path": str(segmented_frame_path),
                            "scores": scores,
                            "boxes": boxes,
                            "labels": labels,
                            "detected": len(scores) > 0,
                            "success": True,
                        }
                    )

        # Failed frames are also appended to segmented_frames (as original-frame
        # fallbacks), so `not segmented_frames` alone can never fire after the
        # loop: a run where EVERY frame failed used to return success=True with
        # a re-encode of the unsegmented input. Fail loudly instead.
        if not segmented_frames or len(failed_frames) == len(frame_paths):
            return {
                "success": False,
                "error": ("No frames were processed" if not segmented_frames
                          else f"All {len(frame_paths)} frames failed to segment"),
                "video_path": str(video),
                "requested_max_frames": requested_max_frames,
                "effective_max_frames": max_frames,
                "failed_frames": failed_frames[:10],
                "failed_frames_truncated": len(failed_frames) > 10,
            }

        missing_output_frames_filled = _ensure_complete_output_sequence(
            frame_paths=frame_paths,
            segmented_frames_dir=segmented_frames_dir,
        )

        if output_mode == "video":
            final_video_path = _build_video_from_frames(
                segmented_frames_dir=segmented_frames_dir,
                source_video_path=str(video),
                output_video_path=segmented_video_path,
                fps=effective_fps,
            )

            shutil.rmtree(tmp_source_frames_dir, ignore_errors=True)
            shutil.rmtree(segmented_frames_dir, ignore_errors=True)

            return {
                "success": True,
                "video_path": str(video),
                "text_query": text_query,
                "output_mode": output_mode,
                "frame_mode": frame_mode,
                "fps": effective_fps,
                "requested_max_frames": requested_max_frames,
                "effective_max_frames": max_frames,
                "frames_processed": len(frame_paths),
                "frames_segmented": len(segmented_frames),
                "frames_failed": len(failed_frames),
                "missing_output_frames_filled": missing_output_frames_filled,
                "segmented_video_path": final_video_path,
                "run_dir": str(run_dir),
                "params": {
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold,
                    "segmentation_type": segmentation_type,
                },
                "failed_frames": failed_frames[:10],
                "failed_frames_truncated": len(failed_frames) > 10,
            }

        shutil.rmtree(tmp_source_frames_dir, ignore_errors=True)

        return {
            "success": True,
            "video_path": str(video),
            "text_query": text_query,
            "output_mode": output_mode,
            "frame_mode": frame_mode,
            "fps": effective_fps,
            "requested_max_frames": requested_max_frames,
            "effective_max_frames": max_frames,
            "frames_processed": len(frame_paths),
            "frames_segmented": len(segmented_frames),
            "frames_failed": len(failed_frames),
            "missing_output_frames_filled": missing_output_frames_filled,
            "segmented_frames_dir": str(segmented_frames_dir),
            "segmented_frames": segmented_frames[:50],
            "segmented_frames_truncated": len(segmented_frames) > 50,
            "run_dir": str(run_dir),
            "params": {
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "segmentation_type": segmentation_type,
            },
            "failed_frames": failed_frames[:10],
            "failed_frames_truncated": len(failed_frames) > 10,
        }

    except Exception as exc:
        logger.exception("Video segmentation failed")

        return {
            "success": False,
            "error": str(exc),
            "video_path": str(video),
            "text_query": text_query,
            "requested_max_frames": requested_max_frames,
            "effective_max_frames": max_frames,
        }

    finally:
        if _tracker is not None:
            try:
                _tracker.stop()
            except Exception:
                pass

        shutil.rmtree(tmp_source_frames_dir, ignore_errors=True)

        if output_mode == "video":
            shutil.rmtree(segmented_frames_dir, ignore_errors=True)


app.mount("/mcp", mcp.streamable_http_app())

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)