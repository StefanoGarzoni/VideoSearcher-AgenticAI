import os
import asyncio
import argparse
from typing import Optional

from agents import Agent, Runner, function_tool, FunctionTool
import logging

from videosearcher_agents.llm_utils import build_llm, resolve_llm_config
from videosearcher_agents.mcp_streamable_client import create_mcp_client

logger = logging.getLogger(__name__)


class WebSearchAgent:
    """
    Web Search Agent using OpenAI Agents SDK with StreamableHTTP MCP Client.

    Connects to the Web Search MCP Server via HTTP and exposes a single tool to
    search the web (DuckDuckGo by default; Tavily/Serper/Brave when configured).

    Architecture:
        Agent -> StreamableHTTP Client -> HTTP -> MCP Server -> Web Search
    """

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_base_url: Optional[str] = None,
    ):
        cfg = resolve_llm_config(
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
            llm_provider=llm_provider,
            llm_base_url=llm_base_url,
        )
        self.model = build_llm(cfg)
        logger.info(f"Initializing WebSearchAgent with LLM model: {cfg.model}")

        self.agent = Agent(
            name="WebSearchAgent",
            instructions="""Search the web for information and report the results.

            STEP 1 — Call the tool immediately. Never ask for clarification first.
            Extract the search query from the input and call
            web_search_query(query=<text>, max_results=5).

            STEP 2 — Report the result in this exact format:
            If success and count > 0:
              "RESULTS: <count> results for '<query>'. Top: <title> — <url>. Output: <output_path>"
            If count == 0:
              "NO RESULTS: '0 results for <query>'."
            If error:
              "ERROR: <message>"

            Do NOT add commentary, plans, or extra text. Return ONLY the one-line result above.""",
            model=self.model,
            tools=[web_search_query],
        )


@function_tool
async def web_search_query(query: str, max_results: int = 5) -> dict:
    """
    Search the web for a query using the Web Search MCP Server (HTTP).

    Args:
        query: Search query (natural language or keywords).
        max_results: Maximum number of results to return.

    Returns:
        Dictionary with provider, results list, count, output_path, and status.
    """
    try:
        logger.info(f"Web search tool called: {query!r}")
        mcp_server_url = os.getenv("WEB_SEARCH_MCP_SERVER_URL", "http://localhost:8010/mcp")
        async with create_mcp_client(mcp_server_url) as client:
            tools = await client.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            if "web_search_query" not in [t.name for t in tools]:
                return {"success": False, "error": "web_search_query tool not found on MCP server"}

            session_root = os.environ.get("SESSION_OUTPUT_ROOT")
            extra = {"output_dir": f"{session_root}/web_search"} if session_root else {}
            result = await client.call_tool(
                "web_search_query", {"query": query, "max_results": max_results, **extra}
            )
            logger.info(f"Tool result: {result}")

            if isinstance(result, dict) and result.get("success"):
                return {
                    "provider": result.get("provider"),
                    "count": result.get("count"),
                    "results": result.get("results"),
                    "output_path": result.get("output_path"),
                    "processing_time_ms": result.get("processing_time_ms"),
                    "success": True,
                }
            error = result.get("error", "Unknown error from MCP server") if isinstance(result, dict) else str(result)
            return {"success": False, "error": error}
    except Exception as e:
        logger.error(f"Web search failed: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Web search failed: {str(e)}"}


async def example_usage(query: str):
    """Example of using the WebSearchAgent with StreamableHTTP"""
    agent = WebSearchAgent().agent
    print(f"🤖 Initialized: {agent}\n")
    result = await Runner.run(agent, f"Search the web for: {query}")
    print(f"Search Result: {result}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Web Search Agent — search the web for information")
    parser.add_argument("query", nargs="?", default="F-35 specifications", help="Search query (default: %(default)s)")
    args = parser.parse_args()
    asyncio.run(example_usage(args.query))
