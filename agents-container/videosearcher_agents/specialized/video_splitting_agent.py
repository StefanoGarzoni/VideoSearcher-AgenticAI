"""
Video Splitting Agent using OpenAI Agents SDK with StreamableHTTP MCP Client
Handles video splitting tasks based on timestamps via HTTP-based FFmpeg-1 MCP Server

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


class VideoSplittingAgent:
    """
    Video Splitting Agent using OpenAI Agents SDK with StreamableHTTP MCP Client.
    
    This agent orchestrates video splitting tasks by:
    1. Connecting to the FFmpeg-1 MCP Server via HTTP (StreamableHTTP protocol)
    2. Using the StreamableHTTP client to access video splitting tools
    3. Using the OpenAI Agent framework to reason about tool use
    4. Maintaining clean separation between agent logic and tool execution
    
    Architecture:
        Agent → StreamableHTTP Client → HTTP → MCP Server → Video Splitting
    """
    
    def __init__(
        self, 
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_base_url: Optional[str] = None
    ):
        """
        Initialize the Video Splitting Agent
        
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
        logger.info(f"Initializing VideoSplittingAgent with LLM model: {cfg.model}")
           
        # Initialize the OpenAI Agent with video splitting tool
        self.agent = Agent(
            name="VideoSplittingAgent",
            instructions="""Split videos into clips based on timestamps.

            Input: Accept .tar.gz archive path (typically from /data/audio_analysis/).
            Tool: Use split_video_clips_tool to create clips from timestamps.txt.

            After successfully calling split_video_clips_tool, your response MUST end with exactly:
            DONE: output_path=<the exact output_folder value from the tool result>
            Example: DONE: output_path=/data/runs/session-id/video_clips/video_clips_abc123

            If error occurs, explain the issue clearly.""",
            model=self.model,
            tools=[
                split_video_clips_tool
            ],
            #output_type=VideoSplittingResponse # Gemini API does not support structured outputs with function calls
        )


@function_tool
async def split_video_clips_tool(archive_file: str) -> dict:
    """
    Split video into clips based on timestamps using FFmpeg-1 MCP Server (HTTP)
    
    This tool calls the FFmpeg-1 MCP server via StreamableHTTP to split videos
    at specified timestamps. Creates an output folder containing raw video clips.
    
    Args:
        archive_file: Path to .tar.gz archive containing video.mp4 and timestamps.txt
        
    Returns:
        Dictionary with output folder path, clips count, and status
    """
    if not Path(archive_file).exists():
        return {"success": False, "error": f"File not found: {archive_file}"}

    try:
        logger.info(f"Video splitting tool called for: {archive_file}")

        mcp_server_url = os.getenv("FFMPEG_1_MCP_SERVER_URL", "http://localhost:8002/mcp")

        # Connect to MCP server and call tool
        async with create_mcp_client(mcp_server_url) as client:
            # List available tools
            tools = await client.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            
            if "split_video_clips" not in [t.name for t in tools]:
                error_msg = "Split video clips tool not found on MCP server"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Call split_video_clips tool, routing output to session directory if available
            session_root = os.environ.get("SESSION_OUTPUT_ROOT")
            extra = {"output_dir": f"{session_root}/video_clips"} if session_root else {}
            result = await client.call_tool("split_video_clips", {"archive_file": archive_file, **extra})
            
            logger.info(f"Tool result: {result}")
            
            if isinstance(result, dict) and result.get("success"):
                return {
                    "output_path": result.get("output_folder"),
                    "clips_count": result.get("clips_count"),
                    "clips_names": result.get("clips_names"),
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
        logger.error(f"Video splitting failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Video splitting failed: {str(e)}"
        }


async def example_usage():
    """Example of using the VideoSplittingAgent with StreamableHTTP"""
    
    # Ensure API key is configured
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: No API key found!")
        print("Set GEMINI_API_KEY or OPENAI_API_KEY environment variable")
        return
    
    # Initialize agent with HTTP-based MCP server
    agent = VideoSplittingAgent().agent
    
    print(f"🤖 Initialized: {agent}\n")

    for tool in agent.tools:
        if isinstance(tool, FunctionTool):
            print(tool.name)
            print(tool.description)
            print(json.dumps(tool.params_json_schema, indent=2))
            print()
    
    print("=" * 60)
    print("Example: Split Video into Clips via HTTP")
    print("=" * 60)
    
    result = await Runner.run(agent, 
                              f"Split the video from the archive at /data/audio_analysis/silence_detection_0628faad.tar.gz into clips based on timestamps")
    
    print(f"\nVideo Splitting Result: {result}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
