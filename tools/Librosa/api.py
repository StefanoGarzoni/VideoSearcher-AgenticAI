"""
Librosa MCP Server with StreamableHTTP Transport

A Model Context Protocol server for audio analysis and feature extraction.
Implements StreamableHTTP protocol for cloud-native microservice deployment.

Features:
- HTTP-based communication (no stdio pipes)
- Containerizable microservice
- Health checks and metrics
- Audio feature extraction (tempo, pitch, MFCC, spectral features)
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

# Handle codecarbon import gracefully (may fail if not installed)
try:
    from codecarbon import EmissionsTracker
    EMISSIONS_TRACKING_AVAILABLE = True
except ImportError as e:
    logger_init = logging.getLogger(__name__)
    logger_init.warning(f"Emissions tracking unavailable: {e}")
    EMISSIONS_TRACKING_AVAILABLE = False
    
    # Create a no-op tracker
    class EmissionsTracker:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def stop(self):
            pass

import librosa
import numpy as np
import subprocess
import shlex
import math
import uuid

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
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY", "/tmp/librosa_output")
PORT = int(os.getenv("PORT", "8001"))
HOST = os.getenv("HOST", "0.0.0.0")
ENABLE_SESSION_MANAGEMENT = os.getenv("ENABLE_SESSION_MANAGEMENT", "true").lower() == "true"
UVICORN_LEVEL = os.getenv("UVICORN_LEVEL", "info")
UVICORN_ACCESS_LOG = os.getenv("UVICORN_ACCESS_LOG", "true").lower() == "true"

# Ensure output directory exists
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Librosa MCP Server",
    description="StreamableHTTP MCP Server for audio analysis",
    version="1.0.0"
)

# Initialize FastMCP server for tool definitions
# Disable DNS rebinding protection for Docker internal network communication
mcp = FastMCP("librosa-audio-http", transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))


def execute_command(command: str):
    """Execute shell command safely"""
    logger.info(f">_ {command}")
    subprocess.run(shlex.split(command), check=True)


def samples_to_timestamp(sample: int, is_start: bool, sample_rate: int = 22050) -> tuple[str, float]:
    """Convert audio samples to timestamp format
    
    Args:
        sample: Sample number
        is_start: True if this is a start timestamp, False for end
        sample_rate: Audio sample rate (default 22050)
        
    Returns:
        Tuple of (formatted_time_string, time_in_seconds)
    """
    time_in_seconds = sample / sample_rate
    if is_start:
        time_in_seconds = math.floor(time_in_seconds)
    else:
        time_in_seconds = math.ceil(time_in_seconds)
    formatted_time = time.strftime('%H:%M:%S', time.gmtime(time_in_seconds))
    return formatted_time, time_in_seconds


@mcp.tool()
def detect_silence_segments(archive_file: str, output_dir: Optional[str] = None) -> str:
    """
    Detect silence segments in audio and generate timestamps for video clipping.

    Uses silence detection (librosa.effects.split) to identify natural break points
    in the audio, useful for segmenting videos into clips at sentence boundaries.
    Creates an output archive with video.mp4 and timestamps.txt.

    Args:
        archive_file: Path to .tar.gz archive containing video.mp4 and audio.wav
        output_dir: Optional output directory override (uses OUTPUT_DIRECTORY env var if omitted)

    Returns:
        JSON string with output archive path, segment count, and processing info
    """
    tracker = EmissionsTracker(project_name="librosa_audio_analysis", output_dir="/app/emissions", log_level="error")
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

        logger.info(f"Detecting silence segments in: {archive_file}")

        # Resolve output directory
        out = output_dir if output_dir else OUTPUT_DIRECTORY
        Path(out).mkdir(parents=True, exist_ok=True)

        # Generate unique output path
        folder_id = str(uuid.uuid4())[:8]
        folder_name = f"silence_detection_{folder_id}"
        output_folder_path = os.path.join(out, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)
        
        # Extract archive into a private temp dir to avoid polluting source directory
        import tempfile
        extract_dir = tempfile.mkdtemp(prefix="librosa_extract_")
        try:
            execute_command(f"tar -xvzf {shlex.quote(archive_file)} -C {shlex.quote(extract_dir)}")
            audio_path = os.path.join(extract_dir, "audio.wav")
            video_path = os.path.join(extract_dir, "video.mp4")

            # Load audio into memory (22050 Hz for consistency with original tool)
            audio, sr = librosa.load(audio_path, sr=22050, mono=True)
            duration = librosa.get_duration(y=audio, sr=sr)

            # Copy video to output folder while temp dir still exists
            output_video_path = os.path.join(output_folder_path, "video.mp4")
            execute_command(f"cp {shlex.quote(video_path)} {shlex.quote(output_video_path)}")
        finally:
            import shutil
            shutil.rmtree(extract_dir, ignore_errors=True)

        # Segment length constraints
        max_sentence_length = 30  # seconds
        min_sentence_length = 6   # seconds

        min_clips_number = duration / max_sentence_length
        max_clips_number = duration / min_sentence_length

        # Find optimal threshold for silence detection
        clips = []
        threshold_db = 24
        for threshold_db in range(24, 50):
            clips = librosa.effects.split(audio, top_db=threshold_db)
            logger.info(f"{len(clips)} clips with {threshold_db} dB as threshold")
            if len(clips) >= min_clips_number and len(clips) < max_clips_number:
                break

        # Generate timestamps file
        timestamps_path = os.path.join(output_folder_path, "timestamps.txt")
        segments_written = 0
        with open(timestamps_path, "w") as file:
            last_timestamp = "00:00:00"
            last_seconds = 0
            for i in range(len(clips)):
                c = clips[i]
                start_timestamp = last_timestamp
                start_seconds = last_seconds
                end_timestamp, end_seconds = samples_to_timestamp(c[1], False, sr)
                clip_length = end_seconds - start_seconds
                if start_seconds != end_seconds and clip_length > min_sentence_length:
                    file.write(f"{start_timestamp} {end_timestamp}\n")
                    segments_written += 1
                    last_timestamp = end_timestamp
                    last_seconds = end_seconds
        
        # Create output archive
        archive_name = f"{folder_name}.tar.gz"
        archive_path = os.path.join(out, archive_name)
        
        execute_command(f"tar -cvzf {archive_path} -C {output_folder_path} timestamps.txt video.mp4")
        
        # Cleanup temporary files
        execute_command(f"rm {timestamps_path}")
        execute_command(f"rm {output_video_path}")
        os.rmdir(output_folder_path)
        
        # Get output size
        file_size = os.path.getsize(archive_path)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        result = {
            "success": True,
            "output_path": archive_path,
            "file_size": file_size,
            # Report the segments actually WRITTEN to timestamps.txt, not the raw
            # librosa.effects.split count — sub-6s clips are merged/dropped, so
            # len(clips) overcounted vs the file a downstream clipper reads.
            "segments_count": segments_written,
            "threshold_db": threshold_db,
            "duration_seconds": float(duration),
            "processing_time_ms": processing_time_ms
        }
        
        logger.info(f"Silence detection completed: {result}")
        tracker.stop()
        return json.dumps(result)
        
    except Exception as e:
        error_msg = f"Silence detection error: {str(e)}"
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
        "service": "librosa-mcp",
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
        "service": "Librosa MCP Server",
        "protocol": "StreamableHTTP",
        "description": "Audio analysis microservice",
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
    logger.info("Librosa MCP Server starting...")
    global _mcp_session_ctx
    _mcp_session_ctx = mcp.session_manager.run()
    await _mcp_session_ctx.__aenter__()

@app.on_event("shutdown")
async def stop_mcp_session_manager():
    logger.info("Librosa MCP Server shutting down...")
    global _mcp_session_ctx
    if _mcp_session_ctx is not None:
        await _mcp_session_ctx.__aexit__(None, None, None)
        _mcp_session_ctx = None

app.mount("/mcp", mcp.streamable_http_app())

def run_server():
    """Run the StreamableHTTP MCP server"""
    logger.info(f"Starting Librosa MCP Server on {HOST}:{PORT}")
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
