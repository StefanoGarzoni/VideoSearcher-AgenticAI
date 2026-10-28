"""
Audio Analysis Agent using OpenAI Agents SDK with StreamableHTTP MCP Client
Handles audio analysis tasks from audio files via HTTP-based Librosa MCP Server

This replaces stdio-based connections with HTTP for cloud-native deployment.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional
import logging

from agents import Agent, Runner, function_tool, FunctionTool
from videosearcher_agents.mcp_streamable_client import create_mcp_client
from videosearcher_agents.llm_utils import build_llm, resolve_llm_config

logger = logging.getLogger(__name__)


class AudioAnalysisAgent:
    """
    Audio Analysis Agent using OpenAI Agents SDK with StreamableHTTP MCP Client.
    
    This agent orchestrates audio analysis tasks by:
    1. Connecting to the Librosa MCP Server via HTTP (StreamableHTTP protocol)
    2. Using the StreamableHTTP client to access audio analysis tools
    3. Using the OpenAI Agent framework to reason about tool use
    4. Maintaining clean separation between agent logic and tool execution
    
    Architecture:
        Agent → StreamableHTTP Client → HTTP → MCP Server → Audio Analysis
    """
    
    def __init__(
        self, 
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_base_url: Optional[str] = None
    ):
        """
        Initialize the Audio Analysis Agent
        
        Args:
            llm_api_key: LLM API key
            llm_model_name: LLM model name
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
        logger.info(f"Initializing AudioAnalysisAgent with LLM model: {cfg.model}")
        
        
        # Initialize the OpenAI Agent with silence detection tool
        self.agent = Agent(
            name="AudioAnalysisAgent",
            instructions="""Detect silence segments in audio and generate timestamps for video clipping.

            Input: Accept .tar.gz archive path (typically from /data/audio_extraction/).
            Tool: Use detect_silence_segments_tool to identify natural break points.

            After successfully calling detect_silence_segments_tool, your response MUST end with exactly:
            DONE: output_path=<the exact output_path value from the tool result>
            Example: DONE: output_path=/data/runs/session-id/audio_analysis/silence_detection_abc.tar.gz

            If error occurs, explain the issue clearly.""",
            model=self.model,
            tools=[
                detect_silence_segments_tool
            ],
            #output_type=AudioAnalysisResponse # Gemini API does not support structured outputs with function calls
        )


@function_tool
async def detect_silence_segments_tool(archive_file: str) -> dict:
    """
    Detect silence segments in audio for video clipping using Librosa MCP Server (HTTP)
    
    This tool performs silence detection to identify natural break points in audio,
    generating timestamps suitable for segmenting videos into clips. Creates an archive
    with video.mp4 and timestamps.txt file.
    
    Args:
        archive_file: Path to .tar.gz archive containing video.mp4 and audio.wav
        
    Returns:
        Dictionary with output archive path, segment count, and status
    """
    if not Path(archive_file).exists():
        return {"success": False, "error": f"File not found: {archive_file}"}

    try:
        logger.info(f"Silence detection tool called for: {archive_file}")

        mcp_server_url = os.getenv("LIBROSA_MCP_SERVER_URL", "http://localhost:8001/mcp")
        
        # Connect to MCP server and call tool
        async with create_mcp_client(mcp_server_url) as client:
            # List available tools
            tools = await client.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            
            if "detect_silence_segments" not in [t.name for t in tools]:
                error_msg = "Detect silence segments tool not found on MCP server"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }
            
            # Call detect_silence_segments tool, routing output to session directory if available
            session_root = os.environ.get("SESSION_OUTPUT_ROOT")
            extra = {"output_dir": f"{session_root}/audio_analysis"} if session_root else {}
            result = await client.call_tool("detect_silence_segments", {"archive_file": archive_file, **extra})
            
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
        logger.error(f"Silence detection failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Silence detection failed: {str(e)}"
        }


async def example_usage():
    """Example of using the AudioAnalysisAgent with StreamableHTTP"""
    
    # Ensure API key is configured
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: No API key found!")
        print("Set GEMINI_API_KEY or OPENAI_API_KEY environment variable")
        return
    
    # Initialize agent with HTTP-based MCP server
    agent = AudioAnalysisAgent().agent
    
    print(f"🤖 Initialized: {agent}\n")

    for tool in agent.tools:
        if isinstance(tool, FunctionTool):
            print(tool.name)
            print(tool.description)
            print(json.dumps(tool.params_json_schema, indent=2))
            print()
    
    print("=" * 60)
    print("Example: Detect Silence Segments for Video Clipping")
    print("=" * 60)
    
    result = await Runner.run(agent, 
                              f"Detect silence segments in the archive at /data/audio_extraction/audio_extraction_8c9b84f4.tar.gz and create timestamps for video clipping")
    
    # Extract the final message from the result
    if hasattr(result, 'final_message') and result.final_message:
        print(f"\n✅ Silence Detection Result:\n{result.final_message}")
    elif hasattr(result, 'messages') and result.messages:
        # Get the last message
        last_message = result.messages[-1]
        if hasattr(last_message, 'content'):
            print(f"\n✅ Silence Detection Result:\n{last_message.content}")
        else:
            print(f"\n✅ Silence Detection Result:\n{last_message}")
    else:
        print(f"\n✅ Silence detection completed. Result: {result}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(example_usage())
