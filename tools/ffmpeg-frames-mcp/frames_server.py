"""
FFmpeg Frames MCP Server (StreamableHTTP)

Tool: extract_frames
Input:  video_file, t_start, t_end, fps
Output: output_folder, frames[{frame_path, timestamp}]
"""

import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

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

PORT = int(os.getenv("PORT", "8006"))
HOST = os.getenv("HOST", "0.0.0.0")

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
FRAMES_DIR = DATA_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FFmpeg Frames MCP Server")
# Disable DNS rebinding protection for Docker internal communication (same as other tools)
mcp = FastMCP(
    "ffmpeg-frames-http",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

# Align FastMCP streamable HTTP path with our mount to avoid "/mcp/mcp"
mcp.settings.streamable_http_path = "/"

# Hold session manager context so the task group stays alive
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
    return JSONResponse({"status": "ok"})

def _run(cmd: List[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {p.stderr[-2000:]}")

@mcp.tool()
def extract_frames(
    video_file: str,
    t_start: float = 0.0,
    t_end: Optional[float] = None,
    fps: float = 1.0,
    image_ext: str = "jpg",
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Extract frames from a video between [t_start, t_end] at fps frames/sec.

    Args:
        video_file: Path to video (must be accessible inside container, e.g. /data/raw/x.mp4)
        t_start: start time (seconds)
        t_end: end time (seconds). If None, extracts until end.
        fps: frames per second
        image_ext: jpg/png
        output_dir: Optional output directory override (uses DATA_DIR/frames if omitted)

    Returns:
        dict with output_folder and frames list {frame_path, timestamp}
    """
    _tracker = _energy_tracker("frame_extraction")
    try:
        if not Path(video_file).exists():
            return {"success": False, "error": f"video_file not found: {video_file}"}

        base_dir = Path(output_dir) if output_dir else FRAMES_DIR
        base_dir.mkdir(parents=True, exist_ok=True)

        run_id = uuid.uuid4().hex[:8]
        # '%' in the stem would be interpreted by ffmpeg's image2 pattern
        # (the whole out_pattern path is a pattern string) and fail valid files.
        out_dir = base_dir / f"{Path(video_file).stem.replace('%', '_')}_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Output naming: frame_000001.jpg, ...
        out_pattern = str(out_dir / f"frame_%06d.{image_ext}")

        # An inverted/empty window must be an error: silently dropping -to used
        # to extract from t_start to EOF while the response still reported the
        # requested t_end as honored.
        if t_end is not None and t_end <= t_start:
            return {"success": False,
                    "error": f"t_end ({t_end}) must be greater than t_start ({t_start})"}

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        cmd += ["-ss", str(t_start)]
        if t_end is not None:
            cmd += ["-to", str(t_end)]
        cmd += ["-i", video_file]
        cmd += ["-vf", f"fps={fps}"]
        cmd += ["-q:v", "2"] if image_ext.lower() in ("jpg", "jpeg") else []
        cmd += ["-y", out_pattern]

        try:
            _run(cmd)
        except Exception as e:
            return {"success": False, "error": str(e), "output_folder": str(out_dir)}

        # Build timestamps: for fps constant sampling, timestamp = t_start + (i-1)/fps
        frames = sorted(out_dir.glob(f"frame_*.{image_ext}"))
        result = []
        for i, fp in enumerate(frames):
            ts = t_start + (i / fps)  # i starts at 0
            result.append({"frame_path": str(fp), "timestamp": float(ts)})

        return {
            "success": True,
            "video_file": video_file,
            "t_start": float(t_start),
            "t_end": float(t_end) if t_end is not None else None,
            "fps": float(fps),
            "output_folder": str(out_dir),
            "frames": result,
            "n_frames": len(result),
        }
    finally:
        if _tracker is not None:
            try:
                _tracker.stop()
            except Exception:
                pass

app.mount("/mcp", mcp.streamable_http_app())

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)