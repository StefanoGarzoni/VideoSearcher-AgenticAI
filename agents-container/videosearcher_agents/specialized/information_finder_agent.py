import os
import asyncio
import json
import sys
import argparse
from typing import Optional
from pydantic import BaseModel

from agents import Agent, Runner, function_tool, FunctionTool
import logging

from videosearcher_agents.llm_utils import build_llm, resolve_llm_config
from videosearcher_agents.mcp_streamable_client import create_mcp_client

logger = logging.getLogger(__name__)

class InformationFinderResponse(BaseModel):
    """Response model for information finder"""
    success: bool
    output_path: Optional[str] = None
    output_size: Optional[int] = None
    processing_time_ms: Optional[int] = None
    results: Optional[dict] = None
    error: Optional[str] = None

class InformationFinderAgent:
    """
    Information Finder Agent using OpenAI Agents SDK with StreamableHTTP MCP Client.
    
    This agent orchestrates pattern matching and information extraction tasks by:
    1. Connecting to the Grep MCP Server via HTTP (StreamableHTTP protocol)
    2. Using the StreamableHTTP client to access grep tools
    3. Using the OpenAI Agent framework to reason about tool use
    4. Maintaining clean separation between agent logic and tool execution
    
    Architecture:
        Agent → StreamableHTTP Client → HTTP → MCP Server → Pattern Matching
    """
    
    def __init__(
        self, 
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_base_url: Optional[str] = None
    ):
        """
        Initialize the Information Finder Agent
        
        Args:
            llm_api_key: LLM API key (optional, LiteLLM auto-detects GEMINI_API_KEY/OPENAI_API_KEY)
            llm_model_name: LLM model name (uses LLM_MODEL env var if not provided)
            llm_provider: LLM provider override (uses LLM_PROVIDER env var if not provided)
            llm_base_url: LLM base URL override (uses LLM_BASE_URL env var if not provided)
        """
        cfg = resolve_llm_config(
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
            llm_provider=llm_provider,
            llm_base_url=llm_base_url
        )
        self.model = build_llm(cfg)
        logger.info(f"Initializing InformationFinderAgent with LLM model: {cfg.model}")
            
        # Initialize the OpenAI Agent with information finder tools
        self.agent = Agent(
            name="InformationFinderAgent",
            instructions="""Search for a word or phrase in a transcript archive and report the result.

            STEP 1 — Call the tool immediately. Never ask for clarification first.
            Extract from the input: (a) the archive path, (b) the search word.
            Examples of valid input formats:
              "Search /data/transcription/X.tar.gz for 'Facebook'"
              "archive_path=X, search_pattern='Facebook'"
            Call: search_pattern_single_archive(archive_file=<path>, pattern=<word>)

            STEP 2 — Report the result in this exact format:
            If match_count > 0:
              "FOUND: '<word>' at timestamp <T> seconds. Context: <context>. Output archive: <output_path>"
              To extract the timestamp T: look in the context string for a pattern like [MM:SS.xxx] or [H:MM:SS.xxx].
              Convert it to seconds (e.g. [00:26.600] → 26.6, [01:05.200] → 65.2).
              Report the timestamp of the line CONTAINING the match (the nearest [timestamp] at or before the match).
            If match_count == 0:
              "NOT FOUND: '0 matches found for <word>'."
            If error:
              "ERROR: <message>"

            Do NOT add commentary, plans, or extra text. Return ONLY the one-line result above.""",
            model=self.model,
            tools=[
                search_pattern_single_archive,
                search_pattern_archive_folder
            ],
            #output_type=InformationFinderResponse # Gemini API does not support structured outputs with function calls
        )
    
@function_tool
async def search_pattern_single_archive(archive_file: str, pattern: str) -> dict:
    """
    Search for a pattern in a single .tar.gz archive using Grep MCP Server (HTTP)
    
    Searches for a pattern in transcripts within a single archive.
    Creates output archive containing video and search results.
    
    Args:
        archive_file: Path to a single .tar.gz archive containing transcript
        pattern: Pattern to search for (supports regex)
        
    Returns:
        Dictionary with output path, search results, file size, and status
    """
    try:
        logger.info(f"Single-archive pattern search tool called for: {archive_file}, pattern: {pattern}")
        
        mcp_server_url = os.getenv("GREP_MCP_SERVER_URL", "http://localhost:8005/mcp")

        # Connect to MCP server and call tool
        async with create_mcp_client(mcp_server_url) as client:
            # List available tools
            tools = await client.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            
            if "search_pattern_single_archive" not in [t.name for t in tools]:
                error_msg = "search_pattern_single_archive tool not found on MCP server"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }

            session_root = os.environ.get("SESSION_OUTPUT_ROOT")
            extra = {"output_dir": f"{session_root}/search_results"} if session_root else {}
            result = await client.call_tool("search_pattern_single_archive", {
                "archive_file": archive_file,
                "pattern": pattern,
                "case_sensitive": False,
                **extra
            })
            
            logger.info(f"Tool result: {result}")
            
            if isinstance(result, dict) and result.get("success"):
                return {
                    "output_path": result.get("output_path"),
                    "output_size": result.get("output_size"),
                    "processing_time_ms": result.get("processing_time_ms"),
                    "results": result.get("results"),
                    "success": True
                }
            else:
                error = result.get("error", "Unknown error from MCP server") if isinstance(result, dict) else str(result)
                return {
                    "success": False,
                    "error": error
                }
        
    except Exception as e:
        logger.error(f"Pattern search failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Pattern search failed: {str(e)}"
        }


@function_tool
async def search_pattern_archive_folder(folder_path: str, pattern: str) -> dict:
    """
    Search for a pattern in all .tar.gz archives in a folder using Grep MCP Server (HTTP)
    
    Searches for a pattern in transcripts across all archives in a folder.
    Creates output archives for each containing video and search results.
    
    Args:
        folder_path: Path to folder containing .tar.gz archives with transcripts
        pattern: Pattern to search for (supports regex)
        
    Returns:
        Dictionary with output folder path, search results, total file size, and status
    """
    try:
        logger.info(f"Folder pattern search tool called for: {folder_path}, pattern: {pattern}")
        
        mcp_server_url = os.getenv("GREP_MCP_SERVER_URL", "http://localhost:8005/mcp")

        # Connect to MCP server and call tool
        async with create_mcp_client(mcp_server_url) as client:
            # List available tools
            tools = await client.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            
            if "search_pattern_archive_folder" not in [t.name for t in tools]:
                error_msg = "search_pattern_archive_folder tool not found on MCP server"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }

            # Call grep tool, routing output to session directory if available
            session_root = os.environ.get("SESSION_OUTPUT_ROOT")
            extra = {"output_dir": f"{session_root}/search_results"} if session_root else {}
            result = await client.call_tool("search_pattern_archive_folder", {
                "folder_path": folder_path,
                "pattern": pattern,
                "case_sensitive": False,
                **extra
            })
            
            logger.info(f"Tool result: {result}")
            
            if isinstance(result, dict) and result.get("success"):
                return {
                    "output_path": result.get("output_path"),
                    "output_size": result.get("output_size"),
                    "processing_time_ms": result.get("processing_time_ms"),
                    "results": result.get("results"),
                    "success": True
                }
            else:
                error = result.get("error", "Unknown error from MCP server") if isinstance(result, dict) else str(result)
                return {
                    "success": False,
                    "error": error
                }
        
    except Exception as e:
        logger.error(f"Pattern search failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Pattern search failed: {str(e)}"
        }


async def example_usage(archive_path: str, pattern: str):
    """Example of using the InformationFinderAgent with StreamableHTTP
    
    Args:
        archive_path: Path to the .tar.gz archive to search
        pattern: Pattern to search for (supports regex)
    """
    
    # Ensure API key is configured
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: No API key found!")
        print("Set GEMINI_API_KEY or OPENAI_API_KEY environment variable")
        return
    
    # Initialize agent with HTTP-based MCP server
    agent = InformationFinderAgent().agent
    
    print(f"🤖 Initialized: {agent}\n")

    for tool in agent.tools:
        if isinstance(tool, FunctionTool):
            print(tool.name)
            print(tool.description)
            print(json.dumps(tool.params_json_schema, indent=2))
            print()
    
    print("=" * 60)
    print("Search for patterns in transcripts via HTTP")
    print("=" * 60)
    
    result = await Runner.run(agent, 
                              f"Search for the pattern '{pattern}' in the folder at {archive_path}")
    
    print(f"Search Result: {result}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(
        description="Information Finder Agent - Search for patterns in video transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python information_finder_agent.py /path/to/archive.tar.gz "hello"
    python information_finder_agent.py /data/output/compression.tar.gz "pattern.*regex"
    python information_finder_agent.py  # Uses default values
        """
    )
    parser.add_argument("archive_path", nargs="?", default="/data/transcription", help="Path to folder or .tar.gz archive to search (default: %(default)s)")
    parser.add_argument("pattern", nargs="?", default="variable elimination", help="Pattern to search for (supports regex, default: %(default)s)")
    
    args = parser.parse_args()
    
    asyncio.run(example_usage(args.archive_path, args.pattern))
