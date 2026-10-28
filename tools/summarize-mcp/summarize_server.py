"""
Summarize MCP Server (StreamableHTTP transport).

Summarizes the transcript inside a .tar.gz archive using an LLM (local or
cloud, resolved by llm_client). Produces an output archive containing the
original video and a summary.txt. e.g. "summarize the 20-minute TED talk".

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
from summarize_main import summarize_archive
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

OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY", "/data/summarization")
PORT = int(os.getenv("PORT", "8016"))
HOST = os.getenv("HOST", "0.0.0.0")
UVICORN_LEVEL = os.getenv("UVICORN_LEVEL", "info")
UVICORN_ACCESS_LOG = os.getenv("UVICORN_ACCESS_LOG", "true").lower() == "true"

Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Summarize MCP Server",
    description="StreamableHTTP MCP Server for LLM transcript summarization",
    version="1.0.0",
)
mcp = FastMCP("summarize-http", transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))


def _helper(input_path: str, operation_type: str, output_dir: Optional[str] = None) -> str:
    start = time.time()
    if not input_path:
        return json.dumps({"success": False, "error": "input_path is required"})
    if not os.path.exists(input_path):
        return json.dumps({"success": False, "error": f"Input path not found: {input_path}"})

    out = output_dir if output_dir else OUTPUT_DIRECTORY
    Path(out).mkdir(parents=True, exist_ok=True)

    folder_name = f"summarization_{uuid.uuid4().hex[:8]}"
    output_folder_path = os.path.join(out, folder_name)
    logger.info("[%s] Summarizing %s -> %s", operation_type, input_path, output_folder_path)

    success, error, summary = summarize_archive(input_path, output_folder_path)
    if not success:
        return json.dumps({"success": False, "error": error})

    archive_path = output_folder_path + ".tar.gz"
    if os.path.exists(archive_path):
        actual_output_path = archive_path
        out_size = os.path.getsize(archive_path)
    else:
        actual_output_path = output_folder_path
        out_size = sum(
            os.path.getsize(os.path.join(dp, fn))
            for dp, _, fns in os.walk(output_folder_path)
            for fn in fns
        )

    return json.dumps({
        "success": True,
        "output_path": actual_output_path,
        "output_size": out_size,
        "summary": summary,
        "processing_time_ms": int((time.time() - start) * 1000),
    }, ensure_ascii=False)


@mcp.tool()
def summarize_single_archive(archive_file: str, output_dir: Optional[str] = None) -> str:
    """
    Summarize the transcript inside a single .tar.gz archive using an LLM.

    Creates an output archive containing the original video and summary.txt.

    Args:
        archive_file: Path to a .tar.gz archive containing transcript.txt.
        output_dir: Optional output directory override (uses OUTPUT_DIRECTORY env var if omitted)
    Returns:
        JSON string with output_path, summary text, output_size, and status.
    """
    _tracker = _energy_tracker("summarization")
    try:
        return _helper(archive_file, "single-archive", output_dir)
    finally:
        if _tracker is not None:
            try:
                _tracker.stop()
            except Exception:
                pass


@mcp.tool()
def summarize_archive_folder(folder_path: str, output_dir: Optional[str] = None) -> str:
    """
    Summarize transcripts in all .tar.gz archives in a folder using an LLM.

    Args:
        folder_path: Path to a folder of .tar.gz archives containing transcript.txt.
        output_dir: Optional output directory override (uses OUTPUT_DIRECTORY env var if omitted)
    Returns:
        JSON string with output folder path, summary text, total size, and status.
    """
    _tracker = _energy_tracker("summarization")
    try:
        return _helper(folder_path, "folder", output_dir)
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
        "service": "summarize-mcp",
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
        "service": "summarize-mcp",
        "protocol": "StreamableHTTP",
        "description": "Transcript summarization microservice",
        "endpoints": {"health": "/health", "metrics": "/metrics", "mcp": "/mcp"},
    })


mcp.settings.streamable_http_path = "/"
_mcp_session_ctx = None


@app.on_event("startup")
async def start_mcp_session_manager():
    logger.info("Summarize MCP Server starting...")
    global _mcp_session_ctx
    _mcp_session_ctx = mcp.session_manager.run()
    await _mcp_session_ctx.__aenter__()


@app.on_event("shutdown")
async def stop_mcp_session_manager():
    logger.info("Summarize MCP Server shutting down...")
    global _mcp_session_ctx
    if _mcp_session_ctx is not None:
        await _mcp_session_ctx.__aexit__(None, None, None)
        _mcp_session_ctx = None


app.mount("/mcp", mcp.streamable_http_app())


def run_server():
    logger.info(f"Starting Summarize MCP Server on {HOST}:{PORT}")
    logger.info(f"Output directory: {OUTPUT_DIRECTORY}")
    uvicorn.run(app, host=HOST, port=PORT, log_level=UVICORN_LEVEL, access_log=UVICORN_ACCESS_LOG)


if __name__ == "__main__":
    run_server()
