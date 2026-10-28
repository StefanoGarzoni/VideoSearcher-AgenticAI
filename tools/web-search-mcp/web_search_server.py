"""
Web Search MCP Server (StreamableHTTP transport).

Provides online search via DuckDuckGo (keyless default) plus optional
Tavily / Serper / Brave providers (auto-enabled when their API key is set).
Used by the orchestrator to fact-check or enrich transcript content, e.g.
"the video mentions the F-35 — look up its specs online and compare".

Mirrors the StreamableHTTP pattern used by the other VideoSearcher tool servers
(FastAPI for /health + /metrics, FastMCP mounted at /mcp).
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
from web_search_main import web_search
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

# Configure logging (honor LOG_LEVEL env)
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

# Configuration
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY", "/data/web_search")
PORT = int(os.getenv("PORT", "8010"))
HOST = os.getenv("HOST", "0.0.0.0")
UVICORN_LEVEL = os.getenv("UVICORN_LEVEL", "info")
UVICORN_ACCESS_LOG = os.getenv("UVICORN_ACCESS_LOG", "true").lower() == "true"

Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Web Search MCP Server",
    description="StreamableHTTP MCP Server for online web search",
    version="1.0.0",
)

# Disable DNS rebinding protection for Docker internal network communication
mcp = FastMCP("web-search-http", transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))


@mcp.tool()
def web_search_query(query: str, max_results: int = 5, provider: str = "", output_dir: Optional[str] = None) -> str:
    """
    Search the web for a query and return ranked results (title, url, snippet).

    Args:
        query: The search query (natural language or keywords).
        max_results: Maximum number of results to return (default 5).
        provider: Optional provider override: "duckduckgo" (default, keyless),
                  "tavily", "serper", or "brave". When empty, a keyed provider
                  is used if its API key is set, otherwise DuckDuckGo.
        output_dir: Optional output directory override (uses OUTPUT_DIRECTORY env var if omitted)

    Returns:
        JSON string: {success, provider, query, count,
                      results:[{title,url,snippet}], output_path, processing_time_ms}
    """
    _tracker = _energy_tracker("web_search")
    try:
        out = output_dir if output_dir else OUTPUT_DIRECTORY
        Path(out).mkdir(parents=True, exist_ok=True)
        start = time.time()
        success, error, payload = web_search(
            query, out, max_results=max_results, provider=(provider or None)
        )
        if not success:
            return json.dumps({"success": False, "error": error})
        payload["processing_time_ms"] = int((time.time() - start) * 1000)
        return json.dumps(payload, ensure_ascii=False)
    finally:
        if _tracker is not None:
            try:
                _tracker.stop()
            except Exception:
                pass


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint for load balancers and orchestrators"""
    return JSONResponse({
        "status": "healthy",
        "service": "web-search-mcp",
        "protocol": "streamable_http",
        "version": "1.0.0",
    })


@app.get("/metrics")
async def metrics() -> JSONResponse:
    """Metrics endpoint for monitoring"""
    return JSONResponse({
        "uptime": time.time(),
        "output_directory": OUTPUT_DIRECTORY,
        "output_dir_exists": os.path.exists(OUTPUT_DIRECTORY),
    })


@app.get("/")
async def root() -> JSONResponse:
    """Root endpoint with service information"""
    return JSONResponse({
        "service": "web-search-mcp",
        "protocol": "StreamableHTTP",
        "description": "Web search microservice",
        "endpoints": {"health": "/health", "metrics": "/metrics", "mcp": "/mcp"},
    })


# Align FastMCP streamable HTTP path with our mount to avoid double "/mcp/mcp".
mcp.settings.streamable_http_path = "/"

# Hold session manager context so the task group stays alive
_mcp_session_ctx = None


@app.on_event("startup")
async def start_mcp_session_manager():
    logger.info("Web Search MCP Server starting...")
    global _mcp_session_ctx
    _mcp_session_ctx = mcp.session_manager.run()
    await _mcp_session_ctx.__aenter__()


@app.on_event("shutdown")
async def stop_mcp_session_manager():
    logger.info("Web Search MCP Server shutting down...")
    global _mcp_session_ctx
    if _mcp_session_ctx is not None:
        await _mcp_session_ctx.__aexit__(None, None, None)
        _mcp_session_ctx = None


app.mount("/mcp", mcp.streamable_http_app())


def run_server():
    """Run the StreamableHTTP MCP server"""
    logger.info(f"Starting Web Search MCP Server on {HOST}:{PORT}")
    logger.info(f"Output directory: {OUTPUT_DIRECTORY}")
    uvicorn.run(app, host=HOST, port=PORT, log_level=UVICORN_LEVEL, access_log=UVICORN_ACCESS_LOG)


if __name__ == "__main__":
    run_server()
