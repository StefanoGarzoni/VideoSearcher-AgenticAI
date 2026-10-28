import os
import asyncio
import json
from pathlib import Path
from typing import Optional

from agents import Agent, Runner, function_tool, FunctionTool
import logging

from videosearcher_agents.llm_utils import build_llm, resolve_llm_config
from videosearcher_agents.mcp_streamable_client import create_mcp_client

logger = logging.getLogger(__name__)


class CompressionAgent:
    """
    Compression Agent using OpenAI Agents SDK with StreamableHTTP MCP Client.
    
    This agent orchestrates compression tasks by:
    1. Connecting to the FFmpeg-2 MCP Server via HTTP (StreamableHTTP protocol)
    2. Using the StreamableHTTP client to access compression tools
    3. Using the OpenAI Agent framework to reason about tool use
    4. Maintaining clean separation between agent logic and tool execution
    
    Architecture:
        Agent → StreamableHTTP Client → HTTP → MCP Server → Compression Processing
    """
    
    def __init__(
        self, 
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_base_url: Optional[str] = None
    ):
        """
        Initialize the Compression Agent
        
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
        logger.info(f"Initializing CompressionAgent with LLM model: {cfg.model}")
        
        # Initialize the OpenAI Agent with compression tools
        self.agent = Agent(
            name="CompressionAgent",
            instructions="""Compress video and audio files.

            Input: Accept a single .mp4 file path or a folder path containing .mp4 clips.
            Tools: Use compress_single_video for a single .mp4; use compress_video_folder for folders.

            After successfully calling the compression tool, your response MUST end with exactly:
            DONE: output_path=<the exact output_path value from the tool result>
            Example: DONE: output_path=/data/runs/session-id/compression/compression_abc123

            If error occurs, explain the issue clearly.""",
            model=self.model,
            tools=[
                compress_single_video,
                compress_video_folder
            ],
            #output_type=CompressionResponse # Gemini API does not support sructured outputs with function calls
        )
    
@function_tool
async def compress_single_video(video_file: str) -> dict:
    """
    Compress a single .mp4 video file using FFmpeg-2 MCP Server (HTTP)
    
    Extracts audio and compresses both video and audio from a single .mp4 file.
    Creates a tar.gz archive containing the compressed video and down-sampled audio.
    
    Args:
        video_file: Path to a single .mp4 file
        
    Returns:
        Dictionary with archive path, file size, and status
    """
    if not Path(video_file).exists():
        return {"success": False, "error": f"File not found: {video_file}"}

    try:
        logger.info(f"Single-file compression tool called for: {video_file}")

        mcp_server_url = os.getenv("FFMPEG_2_MCP_SERVER_URL", "http://localhost:8003/mcp")

        # Connect to MCP server and call tool
        async with create_mcp_client(mcp_server_url) as client:
            # List available tools
            tools = await client.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            
            if "compress_single_video" not in [t.name for t in tools]:
                error_msg = "compress_single_video tool not found on MCP server"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }

            # Call compress tool, routing output to session directory if available
            session_root = os.environ.get("SESSION_OUTPUT_ROOT")
            extra = {"output_dir": f"{session_root}/compression"} if session_root else {}
            result = await client.call_tool("compress_single_video", {"video_file": video_file, **extra})
            
            logger.info(f"Tool result: {result}")
            
            if isinstance(result, dict) and result.get("success"):
                return {
                    "output_path": result.get("output_path"),
                    "output_size": result.get("output_size"),
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
        logger.error(f"Compression failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Compression failed: {str(e)}"
        }


@function_tool
async def compress_video_folder(folder_path: str) -> dict:
    """
    Compress all .mp4 video files in a folder using FFmpeg-2 MCP Server (HTTP)
    
    Extracts audio and compresses both video and audio for all .mp4 files in the folder.
    Creates tar.gz archives for each video containing compressed video and down-sampled audio.
    
    Args:
        folder_path: Path to folder containing .mp4 files
        
    Returns:
        Dictionary with output folder path, total file size, and status
    """
    if not Path(folder_path).is_dir():
        return {"success": False, "error": f"Folder not found: {folder_path}"}

    try:
        logger.info(f"Folder compression tool called for: {folder_path}")

        mcp_server_url = os.getenv("FFMPEG_2_MCP_SERVER_URL", "http://localhost:8003/mcp")

        # Connect to MCP server and call tool
        async with create_mcp_client(mcp_server_url) as client:
            # List available tools
            tools = await client.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            
            if "compress_video_folder" not in [t.name for t in tools]:
                error_msg = "compress_video_folder tool not found on MCP server"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }

            # Call compress tool, routing output to session directory if available
            session_root = os.environ.get("SESSION_OUTPUT_ROOT")
            extra = {"output_dir": f"{session_root}/compression"} if session_root else {}
            result = await client.call_tool("compress_video_folder", {"folder_path": folder_path, **extra})
            
            logger.info(f"Tool result: {result}")
            
            if isinstance(result, dict) and result.get("success"):
                return {
                    "output_path": result.get("output_path"),
                    "output_size": result.get("output_size"),
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
        logger.error(f"Compression failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Compression failed: {str(e)}"
        }


async def example_usage():
    """Example of using the AudioExtractionAgent with StreamableHTTP"""
    
    # Ensure API key is configured
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: No API key found!")
        print("Set GEMINI_API_KEY or OPENAI_API_KEY environment variable")
        return
    
    # Initialize agent with HTTP-based MCP server
    agent = CompressionAgent().agent
    
    print(f"🤖 Initialized: {agent}\n")

    for tool in agent.tools:
        if isinstance(tool, FunctionTool):
            print(tool.name)
            print(tool.description)
            print(json.dumps(tool.params_json_schema, indent=2))
            print()
    
    print("=" * 60)
    print("Compress videos and audio via HTTP")
    print("=" * 60)
    
    result = await Runner.run(agent, 
                              f"Compress video and audio files in folder /data/audio_extraction")
    
    print(f"Compression Result: {result}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
