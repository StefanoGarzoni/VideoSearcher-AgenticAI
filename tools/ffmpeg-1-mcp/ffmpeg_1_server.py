"""
FFmpeg-1 MCP Server for video splitting with StreamableHTTP Transport

A Model Context Protocol server for splitting videos into clips based on timestamps.
Implements StreamableHTTP protocol for cloud-native microservice deployment.

Features:
- HTTP-based communication (no stdio pipes)
- Containerizable microservice
- Health checks and metrics
- Video splitting based on timestamps
- Structured logging
"""

import logging
import os
import json
import time
import sys
import subprocess
import shlex
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import TransportSecuritySettings
from codecarbon import EmissionsTracker

# Configure logging (honor LOG_LEVEL env)
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
try:
    _LEVEL = getattr(logging, _LOG_LEVEL)
except AttributeError:
    _LEVEL = logging.INFO

logging.basicConfig(
    level=_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY", "/tmp/ffmpeg_output")
PORT = int(os.getenv("PORT", "8002"))
HOST = os.getenv("HOST", "0.0.0.0")
ENABLE_SESSION_MANAGEMENT = os.getenv("ENABLE_SESSION_MANAGEMENT", "true").lower() == "true"
UVICORN_LEVEL = os.getenv("UVICORN_LEVEL", "info")
UVICORN_ACCESS_LOG = os.getenv("UVICORN_ACCESS_LOG", "true").lower() == "true"

# Ensure output directory exists
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="FFmpeg-1 MCP Server",
    description="StreamableHTTP MCP Server for video splitting",
    version="1.0.0"
)

# Initialize FastMCP server for tool definitions
# Disable DNS rebinding protection for Docker internal network communication
mcp = FastMCP("ffmpeg-1-http", transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))


def execute_command(command: str, check: bool = True) -> bool:
    """Execute shell command. Returns True on success, False on failure."""
    logger.info(f">_ {command}")
    result = subprocess.run(shlex.split(command), check=False)
    if result.returncode != 0 and check:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.returncode == 0


@mcp.tool()
def split_video_clips(archive_file: str, output_dir: Optional[str] = None) -> str:
    """
    Split video into clips based on timestamps from silence detection.

    Takes an archive containing video.mp4 and timestamps.txt, splits the video
    into clips at the specified time intervals, and saves raw clip files in a
    new output folder.

    Args:
        archive_file: Path to .tar.gz archive containing video.mp4 and timestamps.txt
        output_dir: Optional output directory override (uses OUTPUT_DIRECTORY env var if omitted)

    Returns:
        JSON string with output folder path containing clip files
    """
    tracker = EmissionsTracker(project_name="ffmpeg_video_split", output_dir="/app/emissions", log_level="error")
    tracker.start()

    start_time = time.time()

    try:
        # Validate input
        if not archive_file:
            error_msg = "archive_file is required"
            logger.error(error_msg)
            return json.dumps({
                "success": False,
                "error": error_msg
            })

        # Validate that file exists
        if not os.path.exists(archive_file):
            error_msg = f"Input archive not found: {archive_file}"
            logger.error(error_msg)
            return json.dumps({
                "success": False,
                "error": error_msg
            })

        logger.info(f"Splitting video from archive: {archive_file}")

        # Resolve output directory
        out = output_dir if output_dir else OUTPUT_DIRECTORY
        Path(out).mkdir(parents=True, exist_ok=True)

        # Generate unique output path
        folder_id = str(uuid.uuid4())[:8]
        folder_name = f"video_clips_{folder_id}"
        output_folder_path = os.path.join(out, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)
        
        # Extract archive to output folder (quote paths: shlex.split on an
        # unquoted f-string splits any path containing a space).
        execute_command(f"tar -xvzf {shlex.quote(archive_file)} -C {shlex.quote(output_folder_path)}")
        
        timestamp_path = os.path.join(output_folder_path, "timestamps.txt")
        video_path = os.path.join(output_folder_path, "video.mp4")
        
        # Read timestamps
        with open(timestamp_path) as file:
            lines = file.readlines()
        
        clips_created = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            start, end = line.split()
            clip_name = f"clip_{i}.mp4"
            clip_path = os.path.join(output_folder_path, clip_name)

            command = f"ffmpeg -ss {start} -to {end} -i {shlex.quote(video_path)} -c copy {shlex.quote(clip_path)}"
            if execute_command(command, check=False):
                clips_created.append(clip_name)
            else:
                logger.warning("Skipping clip %d (%s → %s): ffmpeg returned non-zero", i, start, end)

        # Remove original video and timestamps
        execute_command(f"rm {shlex.quote(timestamp_path)}")
        execute_command(f"rm {shlex.quote(video_path)}")

        # If every ffmpeg invocation failed we produced an empty folder; report
        # a failure instead of success with clips_count=0 (a corrupt/truncated
        # source video used to be reported as a successful split of 0 clips).
        if not clips_created:
            tracker.stop()
            return json.dumps({
                "success": False,
                "error": f"No clips could be created from {archive_file} (all ffmpeg invocations failed)"
            })
        
        # Keep raw clip files in output folder (no archiving)
        total_size = 0
        for clip in clips_created:
            clip_path = os.path.join(output_folder_path, clip)
            total_size += os.path.getsize(clip_path)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        result = {
            "success": True,
            "output_folder": output_folder_path,
            "total_clips_size": total_size,
            "clips_count": len(clips_created),
            "clips_names": clips_created,
            "processing_time_ms": processing_time_ms
        }
        
        logger.info(f"Video splitting completed: {result}")
        tracker.stop()
        return json.dumps(result)
        
    except Exception as e:
        error_msg = f"Video splitting error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        tracker.stop()
        return json.dumps({
            "success": False,
            "error": error_msg
        })


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint for load balancers and orchestrators"""
    return JSONResponse({
        "status": "healthy",
        "service": "ffmpeg-1-mcp",
        "protocol": "streamable_http",
        "version": "1.0.0"
    })


@app.get("/metrics")
async def metrics() -> JSONResponse:
    """Metrics endpoint for monitoring"""
    return JSONResponse({
        "uptime": time.time(),
        "output_directory": OUTPUT_DIRECTORY,
        "output_dir_exists": os.path.exists(OUTPUT_DIRECTORY)
    })


@app.get("/")
async def root() -> JSONResponse:
    """Root endpoint with service information"""
    return JSONResponse({
        "service": "FFmpeg-1 MCP Server",
        "protocol": "StreamableHTTP",
        "description": "Video splitting microservice",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "mcp": "/"
        }
    })


# Mount FastMCP's StreamableHTTP ASGI app
mcp.settings.streamable_http_path = "/"

_mcp_session_ctx = None

@app.on_event("startup")
async def start_mcp_session_manager():
    logger.info("FFmpeg-1 MCP Server starting...")
    global _mcp_session_ctx
    _mcp_session_ctx = mcp.session_manager.run()
    await _mcp_session_ctx.__aenter__()

@app.on_event("shutdown")
async def stop_mcp_session_manager():
    logger.info("FFmpeg-1 MCP Server shutting down...")
    global _mcp_session_ctx
    if _mcp_session_ctx is not None:
        await _mcp_session_ctx.__aexit__(None, None, None)
        _mcp_session_ctx = None

app.mount("/mcp", mcp.streamable_http_app())

def run_server():
    """Run the StreamableHTTP MCP server"""
    logger.info(f"Starting FFmpeg-1 MCP Server on {HOST}:{PORT}")
    logger.info(f"Output directory: {OUTPUT_DIRECTORY}")
    logger.info(f"Session management: {ENABLE_SESSION_MANAGEMENT}")
    logger.info(f"Log level: {logging.getLevelName(logger.level)} / uvicorn={UVICORN_LEVEL}")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=UVICORN_LEVEL,
        access_log=UVICORN_ACCESS_LOG
    )


if __name__ == "__main__":
    run_server()
