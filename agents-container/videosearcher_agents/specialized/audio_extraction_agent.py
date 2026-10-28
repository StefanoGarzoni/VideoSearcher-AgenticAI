"""
Audio Extraction Agent using OpenAI Agents SDK with StreamableHTTP MCP Client
Handles audio extraction tasks from video files via HTTP-based FFmpeg MCP Server

This replaces stdio-based connections with HTTP for cloud-native deployment.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional
import logging

from agents import Agent, Runner, function_tool, FunctionTool
from videosearcher_agents.llm_utils import build_llm, resolve_llm_config
from videosearcher_agents.mcp_streamable_client import create_mcp_client

logger = logging.getLogger(__name__)


class AudioExtractionAgent:
    """
    Audio Extraction Agent using OpenAI Agents SDK with StreamableHTTP MCP Client.
    
    This agent orchestrates audio extraction tasks by:
    1. Connecting to the FFmpeg MCP Server via HTTP (StreamableHTTP protocol)
    2. Using the StreamableHTTP client to access audio extraction tools
    3. Using the OpenAI Agent framework to reason about tool use
    4. Maintaining clean separation between agent logic and tool execution
    
    Architecture:
        Agent → StreamableHTTP Client → HTTP → MCP Server → Audio Processing
    """
    
    def __init__(
        self, 
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_base_url: Optional[str] = None
    ):
        """
        Initialize the Audio Extraction Agent
        
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
        logger.info(f"Initializing AudioExtractionAgent with LLM model: {cfg.model}")
        
        # Initialize the OpenAI Agent with audio extraction tool
        self.agent = Agent(
            name="AudioExtractionAgent",
            instructions="""Extract audio from a video file.

            STEP 1 — Call the tool immediately. Never ask for clarification.
            Extract the video path from the input and call extract_audio_tool(video_path=<path>).

            STEP 2 — Report the result in this exact format:
            If success:
              "DONE: output_path=<output_path>"
            If error:
              "ERROR: <message>"

            Do NOT add commentary, plans, or extra text. Return ONLY the one-line result above.""",
            model=self.model,
            tools=[
                extract_audio_tool
            ],
            #output_type=AudioExtractionResponse # Gemini API does not support structured outputs with function calls
        )


@function_tool
async def extract_audio_tool(video_path: str) -> dict:
    """
    Extract audio from a video file using FFmpeg MCP Server (HTTP)
    
    This tool calls the FFmpeg MCP server via StreamableHTTP to extract audio 
    and create a tar.gz archive.
    
    Args:
        video_path: Path to the source video file
        
    Returns:
        Dictionary with archive path, file size, and status
    """
    if not Path(video_path).exists():
        return {"success": False, "error": f"File not found: {video_path}", "error_code": "VALIDATION_ERROR"}

    try:
        logger.info(f"Audio extraction tool called for: {video_path}")

        mcp_server_url = os.getenv("FFMPEG_0_MCP_SERVER_URL", "http://localhost:8000/mcp")
        
        # Connect to MCP server and call tool
        async with create_mcp_client(mcp_server_url) as client:
            # List available tools
            tools = await client.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            
            if "extract_audio" not in [t.name for t in tools]:
                error_msg = "Extract audio tool not found on MCP server"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "error_code": "TOOL_NOT_FOUND",
                }
            
            # Call extract_audio tool, routing output to session directory if available
            session_root = os.environ.get("SESSION_OUTPUT_ROOT")
            extra = {"output_dir": f"{session_root}/audio_extraction"} if session_root else {}
            result = await client.call_tool("extract_audio", {"video_path": video_path, **extra})
            
            logger.info(f"Tool result: {result}")
            
            if isinstance(result, dict) and result.get("success"):
                return {
                    "output_path": result.get("output_path"),
                    "output_size": result.get("file_size"),
                    "processing_time_ms": result.get("processing_time_ms"),
                    "success": True
                }
            else:
                error = result.get("error", "Unknown error from MCP server") if isinstance(result, dict) else str(result)
                return {
                    "success": False,
                    "error": error
                }
        
    except Exception as e:
        logger.error(f"Audio extraction failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Audio extraction failed: {str(e)}"
        }


async def example_usage():
    """Example of using the AudioExtractionAgent with StreamableHTTP"""
    
    # Ensure API key is configured
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: No API key found!")
        print("Set GEMINI_API_KEY or OPENAI_API_KEY environment variable")
        return
    
    # Initialize agent with HTTP-based MCP server
    agent = AudioExtractionAgent().agent
    
    print(f"🤖 Initialized: {agent}\n")

    for tool in agent.tools:
        if isinstance(tool, FunctionTool):
            print(tool.name)
            print(tool.description)
            print(json.dumps(tool.params_json_schema, indent=2))
            print()
    
    print("=" * 60)
    print("Example: Extract Audio from Video via HTTP")
    print("=" * 60)
    
    result = await Runner.run(agent, 
                              f"Extract audio from the video at /data/raw/sample_video.mp4")
    
    print(f"Audio Extraction Result: {result}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
