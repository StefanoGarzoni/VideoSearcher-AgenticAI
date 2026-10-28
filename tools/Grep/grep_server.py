"""
Grep MCP Server for pattern searching from audio transcriptions with StreamableHTTP Transport

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
from grep_main import search_transcripts
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
PORT = int(os.getenv("PORT", "8005"))
HOST = os.getenv("HOST", "0.0.0.0")
ENABLE_SESSION_MANAGEMENT = os.getenv("ENABLE_SESSION_MANAGEMENT", "true").lower() == "true"
UVICORN_LEVEL = os.getenv("UVICORN_LEVEL", "info")
UVICORN_ACCESS_LOG = os.getenv("UVICORN_ACCESS_LOG", "true").lower() == "true"

# Ensure output directory exists
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Grep MCP Server",
    description="StreamableHTTP MCP Server for pattern searching from audio transcriptions",
    version="1.0.0"
)

# Initialize FastMCP server for tool definitions
# Disable DNS rebinding protection for Docker internal network communication
mcp = FastMCP("grep-http", transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))


def _grep_helper(input_path: str, pattern: str, operation_type: str, output_dir: Optional[str] = None, case_sensitive: bool = False) -> str:
    """Internal helper for grep operations (single archive or folder).

    Uses smart path detection to handle both input types.
    """
    tracker = EmissionsTracker(project_name="grep_search", output_dir="/app/emissions", log_level="error")
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
        folder_name = f"grep_{folder_id}"
        output_folder_path = os.path.join(out, folder_name)

        logger.info(f"[{operation_type}] Matching pattern {pattern} transcripts from {input_path} to {output_folder_path}")
        
        # Call the core utility which handles both single archives and folders via smart detection
        success, error, results = search_transcripts(input_path, output_folder_path, search_pattern=pattern, case_sensitive=case_sensitive)
        
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
            "results": results,
            "output_path": actual_output_path,
            "output_size": folder_size,
            "processing_time_ms": processing_time_ms
        }
        
        logger.info(f"[{operation_type}] Pattern matching completed: {result}")
        tracker.stop()
        return json.dumps(result)
        
    except Exception as e:
        error_msg = f"Pattern matching error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        tracker.stop()
        return json.dumps({
            "success": False,
            "error": error_msg
        })

# Clarity by separating tools for single archive and folder processing
@mcp.tool()
def search_pattern_single_archive(archive_file: str, pattern: str, case_sensitive: bool = False, output_dir: Optional[str] = None) -> str:
    """
    Match a pattern in a single transcript in a .tar.gz archive using Grep.
    Search results contain positions and context of all found matching regular expressions.

    Creates an output archive containing:
    - clip.mp4 (original/compressed video)
    - search_results.json (search results)

    Args:
        archive_file: Path to a single .tar.gz archive containing transcript
        pattern: Pattern to search for (supports regex)
        case_sensitive: Whether the search should be case-sensitive (default: False)
        output_dir: Optional output directory override (uses OUTPUT_DIRECTORY env var if omitted)

    Returns:
        JSON string with operation result and output archive path and size
    """
    return _grep_helper(archive_file, pattern, "single-archive", output_dir, case_sensitive)


@mcp.tool()
def search_pattern_archive_folder(folder_path: str, pattern: str, case_sensitive: bool = False, output_dir: Optional[str] = None) -> str:
    """
    Match a pattern in all transcripts in .tar.gz archives within a folder using Grep.
    Search results contain positions and context of all found matching regular expressions.

    Processes all .tar.gz files in the folder, creating an output archive for each containing:
    - clip.mp4 (original/compressed video)
    - search_results.json (search results)

    Args:
        folder_path: Path to folder containing .tar.gz archives with transcripts
        pattern: Pattern to search for (supports regex)
        case_sensitive: Whether the search should be case-sensitive (default: False)
        output_dir: Optional output directory override (uses OUTPUT_DIRECTORY env var if omitted)

    Returns:
        JSON string with operation result and output folder path and total size
    """
    return _grep_helper(folder_path, pattern, "folder", output_dir, case_sensitive)


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint for load balancers and orchestrators"""
    return JSONResponse({
        "status": "healthy",
        "service": "grep-mcp",
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
        "service": "grep-mcp",
        "protocol": "StreamableHTTP",
        "description": "Grep microservice",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "mcp": "/"
        }
    })


"""
Mount StreamableHTTP ASGI app at /mcp instead of handling manually.
This ensures proper stream setup (connect) and protocol handling.
"""

# Align FastMCP streamable HTTP path with our mount to avoid double "/mcp/mcp".
# We keep the FastAPI app routes (/health, /metrics, /) and mount the MCP app at /mcp.
mcp.settings.streamable_http_path = "/"

# Hold session manager context so the task group stays alive
_mcp_session_ctx = None

@app.on_event("startup")
async def start_mcp_session_manager():
    logger.info("Grep MCP Server starting...")
    global _mcp_session_ctx
    _mcp_session_ctx = mcp.session_manager.run()
    await _mcp_session_ctx.__aenter__()

@app.on_event("shutdown")
async def stop_mcp_session_manager():
    logger.info("Grep MCP Server shutting down...")
    global _mcp_session_ctx
    if _mcp_session_ctx is not None:
        await _mcp_session_ctx.__aexit__(None, None, None)
        _mcp_session_ctx = None

app.mount("/mcp", mcp.streamable_http_app())

def run_server():
    """Run the StreamableHTTP MCP server"""
    logger.info(f"Starting Grep MCP Server on {HOST}:{PORT}")
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
