"""
Scene Detect MCP Server (StreamableHTTP transport).

Detects scene boundaries in a video using PySceneDetect (ContentDetector).
Accepts a raw .mp4 OR a .tar.gz archive containing a .mp4. Writes scenes.json
to OUTPUT_DIRECTORY and returns JSON with scene boundaries.

Mirrors the StreamableHTTP pattern of the other VideoSearcher tool servers.
"""

import logging
import os
import json
import time
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from scene_detect_main import detect_scenes_from_video
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

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
try:
    _LEVEL = getattr(logging, _LOG_LEVEL)
except AttributeError:
    _LEVEL = logging.INFO

logging.basicConfig(
    level=_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY", "/data/scenes")
PORT = int(os.getenv("PORT", "8025"))
HOST = os.getenv("HOST", "0.0.0.0")
UVICORN_LEVEL = os.getenv("UVICORN_LEVEL", "info")
UVICORN_ACCESS_LOG = os.getenv("UVICORN_ACCESS_LOG", "true").lower() == "true"

Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Scene Detect MCP Server",
    description="StreamableHTTP MCP Server for PySceneDetect-based scene boundary detection",
    version="1.0.0",
)
mcp = FastMCP("scene-detect-http", transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))


@mcp.tool()
def detect_scenes(video_path: str, threshold: float = 27.0, output_dir: Optional[str] = None) -> str:
    """
    Detect scene boundaries in a video using PySceneDetect (ContentDetector).

    Accepts a raw .mp4 OR a .tar.gz archive containing a .mp4. If an archive
    is provided it is extracted to a temp dir (cleaned up afterwards). Writes
    scenes.json to the output directory.

    Args:
        video_path: Path to a .mp4 file or a .tar.gz archive containing a .mp4.
        threshold: ContentDetector sensitivity threshold (default 27.0).
        output_dir: Optional output directory override (uses OUTPUT_DIRECTORY env var if omitted)
    Returns:
        JSON string with output_path, num_scenes, scenes list, threshold,
        processing_time_ms, and success status.
    """
    _tracker = _energy_tracker("scene_detection")
    try:
        if not video_path:
            return json.dumps({"success": False, "error": "video_path is required"})
        if not os.path.exists(video_path):
            return json.dumps({"success": False, "error": f"Input path not found: {video_path}"})

        out = output_dir if output_dir else OUTPUT_DIRECTORY
        Path(out).mkdir(parents=True, exist_ok=True)

        run_id = uuid.uuid4().hex[:8]
        output_subdir = os.path.join(out, f"scenes_{run_id}")
        logger.info("[detect_scenes] Processing %s -> %s (threshold=%.1f)", video_path, output_subdir, threshold)

        result = detect_scenes_from_video(video_path, output_subdir, threshold=threshold)
        return json.dumps(result, ensure_ascii=False)
    finally:
        if _tracker is not None:
            try:
                _tracker.stop()
            except Exception:
                pass


@app.get("/health")
async def health_check() -> JSONResponse:
    return JSONResponse({
        "status": "healthy",
        "service": "scene-detect-mcp",
        "protocol": "streamable_http",
        "version": "1.0.0",
    })


@app.get("/metrics")
async def metrics() -> JSONResponse:
    return JSONResponse({
        "uptime": time.time(),
        "output_directory": OUTPUT_DIRECTORY,
        "output_dir_exists": os.path.exists(OUTPUT_DIRECTORY),
    })


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({
        "service": "scene-detect-mcp",
        "protocol": "StreamableHTTP",
        "description": "Scene boundary detection microservice using PySceneDetect",
        "endpoints": {"health": "/health", "metrics": "/metrics", "mcp": "/mcp"},
    })


mcp.settings.streamable_http_path = "/"
_mcp_session_ctx = None


@app.on_event("startup")
async def start_mcp_session_manager():
    logger.info("Scene Detect MCP Server starting...")
    global _mcp_session_ctx
    _mcp_session_ctx = mcp.session_manager.run()
    await _mcp_session_ctx.__aenter__()


@app.on_event("shutdown")
async def stop_mcp_session_manager():
    logger.info("Scene Detect MCP Server shutting down...")
    global _mcp_session_ctx
    if _mcp_session_ctx is not None:
        await _mcp_session_ctx.__aexit__(None, None, None)
        _mcp_session_ctx = None


app.mount("/mcp", mcp.streamable_http_app())


def run_server():
    logger.info(f"Starting Scene Detect MCP Server on {HOST}:{PORT}")
    logger.info(f"Output directory: {OUTPUT_DIRECTORY}")
    uvicorn.run(app, host=HOST, port=PORT, log_level=UVICORN_LEVEL, access_log=UVICORN_ACCESS_LOG)


if __name__ == "__main__":
    run_server()
