"""
Speech-To-Text MCP Server for audio transcription with StreamableHTTP Transport

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

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transcribe_main import transcribe_audio_to_text
import uvicorn

from codecarbon import EmissionsTracker

from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import TransportSecuritySettings

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
    title="Speech-To-Text MCP Server",
    description="StreamableHTTP MCP Server for audio transcription",
    version="1.0.0"
)

# Initialize FastMCP server for tool definitions
# Disable DNS rebinding protection for Docker internal network communication
mcp = FastMCP("speech-to-text-http", transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))


def _transcribe_helper(input_path: str, operation_type: str, output_dir: Optional[str] = None) -> str:
    """Internal helper for transcription operations (single archive or folder).

    Uses smart path detection to handle both input types.
    """
    tracker = EmissionsTracker(project_name="whisper_transcription", output_dir="/app/emissions", log_level="error")
    tracker.start()
    
    start_time = time.time()
    
    try:
        # Validate input
        if not input_path:
            error_msg = "input_path is required"
            logger.error(error_msg)
            return json.dumps({
                "success": False,
                "error": error_msg
            })
        
        # Validate that path exists
        if not os.path.exists(input_path):
            error_msg = f"Input path not found: {input_path}"
            logger.error(error_msg)
            return json.dumps({
                "success": False,
                "error": error_msg
            })
        
        # Resolve output directory
        out = output_dir if output_dir else OUTPUT_DIRECTORY
        Path(out).mkdir(parents=True, exist_ok=True)

        # Generate unique output path
        import uuid
        folder_id = str(uuid.uuid4())[:8]
        folder_name = f"transcription_{folder_id}"
        output_folder_path = os.path.join(out, folder_name)

        logger.info(f"[{operation_type}] Transcribing audio from {input_path} to {output_folder_path}")
        
        # Call the core utility which handles both single archives and folders via smart detection
        success, error = transcribe_audio_to_text(input_path, output_folder_path, device="auto")
        
        if not success:
            return json.dumps({
                "success": False,
                "error": error
            })
        
        # Calculate output size based on what was created
        # For single archives: archive is at output_folder_path + ".tar.gz"
        # For folders: archives are inside output_folder_path
        archive_path = output_folder_path + ".tar.gz"
        if os.path.exists(archive_path):
            # Single archive case - archive created outside folder
            folder_size = os.path.getsize(archive_path)
            actual_output_path = archive_path
        else:
            # Folder case - archives inside folder
            folder_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(output_folder_path)
                for filename in filenames
            )
            actual_output_path = output_folder_path
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        result = {
            "success": True,
            "output_path": actual_output_path,
            "output_size": folder_size,
            "processing_time_ms": processing_time_ms
        }
        
        logger.info(f"[{operation_type}] Transcription completed: {result}")
        tracker.stop()
        return json.dumps(result)
        
    except Exception as e:
        error_msg = f"Transcription error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        tracker.stop()
        return json.dumps({
            "success": False,
            "error": error_msg
        })


@mcp.tool()
def transcribe_single_archive(archive_file: str, output_dir: Optional[str] = None) -> str:
    """
    Transcribe audio from a single .tar.gz archive to text using Faster Whisper.

    Creates an output archive containing:
    - clip.mp4 (original/compressed video)
    - transcript.txt (transcription text)

    Args:
        archive_file: Path to a single .tar.gz archive containing audio
        output_dir: Optional output directory override (uses OUTPUT_DIRECTORY env var if omitted)

    Returns:
        JSON string with operation result including output archive path and size
    """
    return _transcribe_helper(archive_file, "single-archive", output_dir)


@mcp.tool()
def transcribe_archive_folder(folder_path: str, output_dir: Optional[str] = None) -> str:
    """
    Transcribe audio from all .tar.gz archives in a folder to text using Faster Whisper.

    Processes all .tar.gz files in the folder, creating an output archive for each containing:
    - clip_x.mp4 (original/compressed video)
    - transcript.txt (transcription text)

    Args:
        folder_path: Path to folder containing .tar.gz archives
        output_dir: Optional output directory override (uses OUTPUT_DIRECTORY env var if omitted)

    Returns:
        JSON string with operation result including output folder path and total size
    """
    return _transcribe_helper(folder_path, "folder", output_dir)


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint for load balancers and orchestrators"""
    return JSONResponse({
        "status": "healthy",
        "service": "speech-to-text-mcp",
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
        "service": "speech-to-text-mcp",
        "protocol": "StreamableHTTP",
        "description": "Speech-to-text microservice",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "mcp": "/"
        }
    })


"""
Mount FastMCP-2's StreamableHTTP ASGI app at /mcp instead of handling manually.
This ensures proper stream setup (connect) and protocol handling.
"""

# Align FastMCP streamable HTTP path with our mount to avoid double "/mcp/mcp".
# We keep the FastAPI app routes (/health, /metrics, /) and mount the MCP app at /mcp.
mcp.settings.streamable_http_path = "/"

# Hold session manager context so the task group stays alive
_mcp_session_ctx = None

@app.on_event("startup")
async def start_mcp_session_manager():
    logger.info("Speech-to-text MCP Server starting...")
    global _mcp_session_ctx
    _mcp_session_ctx = mcp.session_manager.run()
    await _mcp_session_ctx.__aenter__()

@app.on_event("shutdown")
async def stop_mcp_session_manager():
    logger.info("Speech-to-text MCP Server shutting down...")
    global _mcp_session_ctx
    if _mcp_session_ctx is not None:
        await _mcp_session_ctx.__aexit__(None, None, None)
        _mcp_session_ctx = None

app.mount("/mcp", mcp.streamable_http_app())

def run_server():
    """Run the StreamableHTTP MCP server"""
    logger.info(f"Starting Speech-to-text MCP Server on {HOST}:{PORT}")
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
