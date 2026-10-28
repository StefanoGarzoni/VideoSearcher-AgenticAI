"""Tiny MCP StreamableHTTP probe: connect, list tools, call one tool.

Usage (run inside a tool container, which has the `mcp` package):
    python /tmp/mcp_probe.py <url> <tool_name> '<json-args>'
e.g. python /tmp/mcp_probe.py http://localhost:8010/mcp web_search_query '{"query":"F-35","max_results":3}'
"""
import asyncio
import json
import sys

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main(url: str, tool: str, args: dict):
    async with streamablehttp_client(url) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])
            if not tool:
                return
            res = await s.call_tool(tool, args)
            txt = res.content[0].text if res.content else ""
            print("RESULT:", txt[:1200])


if __name__ == "__main__":
    url = sys.argv[1]
    tool = sys.argv[2] if len(sys.argv) > 2 else ""
    args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    asyncio.run(main(url, tool, args))
