"""
FFmpeg MCP Server with StreamableHTTP Transport

A Model Context Protocol server for audio extraction from video files.
Implements StreamableHTTP protocol for cloud-native microservice deployment.

Features:
- HTTP-based communication (no stdio pipes)
- Containerizable microservice
- Health checks and metrics
- Session management
- Structured logging
"""

import logging
import os
import json
import time
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import TransportSecuritySettings
import anyio
from codecarbon import EmissionsTracker

from ffmpeg_0_main import extract_audio_to_archive

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
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
ENABLE_SESSION_MANAGEMENT = os.getenv("ENABLE_SESSION_MANAGEMENT", "true").lower() == "true"
UVICORN_LEVEL = os.getenv("UVICORN_LEVEL", "info")
UVICORN_ACCESS_LOG = os.getenv("UVICORN_ACCESS_LOG", "true").lower() == "true"

# Ensure output directory exists
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="FFmpeg MCP Server",
    description="StreamableHTTP MCP Server for audio extraction",
    version="1.0.0"
)

# Initialize FastMCP server for tool definitions
# Disable DNS rebinding protection for Docker internal network communication
mcp = FastMCP("ffmpeg-audio-http", transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))


@mcp.tool()
def extract_audio(video_path: str, output_dir: Optional[str] = None) -> str:
    """
    Extract audio from a video file and create a tar.gz archive.

    Creates an archive containing:
    - video.mp4 (copy of the input video)
    - audio.wav (extracted audio)

    Args:
        video_path: Path to the input video file
        output_dir: Optional output directory override (uses OUTPUT_DIRECTORY env var if omitted)

    Returns:
        JSON string with operation result including archive path and file size
    """
    tracker = EmissionsTracker(project_name="ffmpeg_audio_extraction", output_dir="/app/emissions", log_level="error")
    tracker.start()

    start_time = time.time()

    try:
        # Validate input
        if not video_path:
            error_msg = "video_path is required"
            logger.error(error_msg)
            return json.dumps({
                "success": False,
                "error": error_msg
            })

        # Validate that file exists
        if not os.path.exists(video_path):
            error_msg = f"Input video file not found: {video_path}"
            logger.error(error_msg)
            return json.dumps({
                "success": False,
                "error": error_msg
            })

        # Resolve output directory
        out = output_dir if output_dir else OUTPUT_DIRECTORY
        Path(out).mkdir(parents=True, exist_ok=True)

        # Generate unique output archive path
        import uuid
        archive_id = str(uuid.uuid4())[:8]
        archive_name = f"audio_extraction_{archive_id}.tar.gz"
        output_archive_path = os.path.join(out, archive_name)
        
        logger.info(f"Extracting audio from {video_path} to {output_archive_path}")
        
        # Call the core extraction utility
        success, error = extract_audio_to_archive(video_path, output_archive_path)
        
        if not success:
            return json.dumps({
                "success": False,
                "error": error
            })
        
        # Get archive info
        if not os.path.exists(output_archive_path):
            return json.dumps({
                "success": False,
                "error": f"Archive file was not created: {output_archive_path}"
            })
        
        file_size = os.path.getsize(output_archive_path)
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        result = {
            "success": True,
            "output_path": output_archive_path,
            "file_size": file_size,
            "processing_time_ms": processing_time_ms
        }
        
        logger.info(f"Audio extraction completed: {result}")
        tracker.stop()
        return json.dumps(result)
        
    except Exception as e:
        error_msg = f"Audio extraction error: {str(e)}"
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
        "service": "ffmpeg-mcp",
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
        "service": "FFmpeg MCP Server",
        "protocol": "StreamableHTTP",
        "description": "Audio extraction microservice",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "mcp": "/"
        }
    })


"""
Mount FastMCP's StreamableHTTP ASGI app at /mcp instead of handling manually.
This ensures proper stream setup (connect) and protocol handling.
"""

# Align FastMCP streamable HTTP path with our mount to avoid double "/mcp/mcp".
# We keep the FastAPI app routes (/health, /metrics, /) and mount the MCP app at /mcp.
mcp.settings.streamable_http_path = "/"

# Hold session manager context so the task group stays alive
_mcp_session_ctx = None

@app.on_event("startup")
async def start_mcp_session_manager():
    logger.info("FFmpeg MCP Server starting...")
    global _mcp_session_ctx
    _mcp_session_ctx = mcp.session_manager.run()
    await _mcp_session_ctx.__aenter__()

@app.on_event("shutdown")
async def stop_mcp_session_manager():
    logger.info("FFmpeg MCP Server shutting down...")
    global _mcp_session_ctx
    if _mcp_session_ctx is not None:
        await _mcp_session_ctx.__aexit__(None, None, None)
        _mcp_session_ctx = None

app.mount("/mcp", mcp.streamable_http_app())

def run_server():
    """Run the StreamableHTTP MCP server"""
    logger.info(f"Starting FFmpeg MCP Server on {HOST}:{PORT}")
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
