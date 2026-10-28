"""
MCP StreamableHTTP Client Adapter

Provides a high-level client for connecting to StreamableHTTP MCP servers.
Enables agents to communicate with HTTP-based MCP servers instead of stdio.

Features:
- Automatic session management
- Retry logic with exponential backoff
- Clean async/await interface
- Type-safe tool calls
"""

import json
import logging
from typing import Optional, Any, Dict, AsyncGenerator
import asyncio
from contextlib import asynccontextmanager

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool, TextContent

logger = logging.getLogger(__name__)

# Module-level tool list cache keyed by server URL.
# Tool catalogs are static (defined at server startup), so one fetch per URL
# per process lifetime is correct. Invalidated on fetch error.
_tool_cache: Dict[str, list] = {}


def _classify_error(exc: Exception) -> str:
    """Map an exception to a structured error code string."""
    msg = str(exc).lower()
    if isinstance(exc, (ConnectionError, ConnectionRefusedError, OSError)):
        return "CONNECTION_ERROR"
    if "not found" in msg or "unknown tool" in msg or "no tool" in msg:
        return "TOOL_NOT_FOUND"
    if isinstance(exc, (ValueError, TypeError)) or "invalid" in msg or "validation" in msg:
        return "VALIDATION_ERROR"
    return "EXECUTION_ERROR"


class StreamableHTTPMCPClient:
    """
    Client for connecting to StreamableHTTP MCP servers.
    
    This replaces stdio-based connections with HTTP for cloud-native deployments.
    Supports session persistence, automatic retries, and health checks.
    """
    
    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        sse_read_timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize StreamableHTTP MCP Client
        
        Args:
            url: Server URL (e.g., http://localhost:8000/mcp)
            timeout: HTTP request timeout in seconds
            sse_read_timeout: SSE stream read timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial retry delay in seconds (exponential backoff)
        """
        self.url = url
        self.timeout = timeout
        self.sse_read_timeout = sse_read_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.session: Optional[ClientSession] = None
        self.read_stream = None
        self.write_stream = None
        self.get_session_id = None
    
    async def connect(self) -> None:
        """
        Establish connection to StreamableHTTP MCP server
        
        Raises:
            ConnectionError: If unable to connect after retries
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Connecting to {self.url} (attempt {attempt + 1}/{self.max_retries})")
                
                # Create StreamableHTTP client connection
                context_manager = streamablehttp_client(
                    url=self.url,
                    timeout=self.timeout,
                    sse_read_timeout=self.sse_read_timeout,
                )
                
                self.read_stream, self.write_stream, self.get_session_id = await context_manager.__aenter__()
                self.context_manager = context_manager
                
                # Initialize MCP session
                self.session = ClientSession(self.read_stream, self.write_stream)
                await self.session.__aenter__()
                await self.session.initialize()
                
                logger.info(f"Connected to {self.url}")
                return
                
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise ConnectionError(f"Failed to connect to {self.url} after {self.max_retries} attempts: {str(e)}")
    
    async def disconnect(self) -> None:
        """
        Close connection to MCP server
        """
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
                self.session = None
            
            if hasattr(self, 'context_manager'):
                await self.context_manager.__aexit__(None, None, None)
            
            logger.info("Disconnected from server")
        except Exception as e:
            logger.error(f"Error disconnecting: {str(e)}")
    
    async def list_tools(self) -> list:
        """
        Get list of available tools from server.

        Results are cached per URL for the lifetime of the process.
        The cache is invalidated if the server returns an error, ensuring
        stale entries don't persist after a server restart.
        """
        if not self.session:
            raise RuntimeError("Not connected to server")

        if self.url in _tool_cache:
            logger.debug("list_tools cache hit for %s", self.url)
            return _tool_cache[self.url]

        try:
            tools = await self.session.list_tools()
            tool_list = tools.tools
            _tool_cache[self.url] = tool_list
            logger.info("list_tools fetched and cached for %s: %s", self.url, [t.name for t in tool_list])
            return tool_list
        except Exception as e:
            _tool_cache.pop(self.url, None)
            logger.error("Failed to list tools from %s: %s", self.url, e)
            raise

    async def call_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool on the remote MCP server.

        Always returns a dict. On failure returns
        {"success": False, "error": "...", "error_code": "..."} so callers
        can pattern-match on error_code without catching exceptions.
        """
        if not self.session:
            return {
                "success": False,
                "error": "Not connected to MCP server",
                "error_code": "CONNECTION_ERROR",
            }

        try:
            logger.info(f"Calling tool: {tool_name} with input: {tool_input}")

            result = await self.session.call_tool(tool_name, tool_input)

            # Parse the result content
            if result.content:
                content = result.content[0]

                if isinstance(content, TextContent):
                    text_result = content.text
                    logger.debug("Tool result: %s", text_result[:500])

                    try:
                        return json.loads(text_result)
                    except (json.JSONDecodeError, TypeError):
                        return {"result": text_result}
                else:
                    return {"result": str(content)}

            return {}

        except Exception as e:
            error_code = _classify_error(e)
            logger.error("Tool call failed: %s — %s (%s)", tool_name, e, error_code, exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_code": error_code,
            }
    
    @asynccontextmanager
    async def session_context(self):
        """
        Context manager for automatic connect/disconnect
        
        Usage:
            async with client.session_context():
                await client.call_tool(...)
        """
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()


@asynccontextmanager
async def create_mcp_client(url: str, **kwargs) -> AsyncGenerator[StreamableHTTPMCPClient, None]:
    """
    Factory function for creating and managing MCP client connections
    
    Usage:
        async with create_mcp_client("http://localhost:8000/mcp") as client:
            tools = await client.list_tools()
            result = await client.call_tool("extract_audio", {"video_path": "..."})
    """
    client = StreamableHTTPMCPClient(url, **kwargs)
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()
