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


class TranscriptionAgent:
    """
    Transcription Agent using OpenAI Agents SDK with StreamableHTTP MCP Client.
    
    This agent orchestrates transcription tasks by:
    1. Connecting to the Speech-To-Text MCP Server via HTTP (StreamableHTTP protocol)
    2. Using the StreamableHTTP client to access transcription tools
    3. Using the OpenAI Agent framework to reason about tool use
    4. Maintaining clean separation between agent logic and tool execution
    
    Architecture:
        Agent → StreamableHTTP Client → HTTP → MCP Server → Transcription Processing
    """
    
    def __init__(
        self, 
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_base_url: Optional[str] = None
    ):
        """
        Initialize the Transcription Agent
        
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
        logger.info(f"Initializing TranscriptionAgent with LLM model: {cfg.model}")
            
        # Initialize the OpenAI Agent with transcription tools
        self.agent = Agent(
            name="TranscriptionAgent",
            instructions="""Transcribe audio to text using Faster Whisper.

            STEP 1 — Call the tool immediately. Never ask for clarification.
            Extract the archive path from the input and call transcribe_single_archive(archive_file=<path>).

            STEP 2 — Report the result in this exact format:
            If success:
              "DONE: output_path=<output_path>"
            If error:
              "ERROR: <message>"

            Do NOT add commentary, plans, or extra text. Return ONLY the one-line result above.""",
            model=self.model,
            tools=[
                transcribe_single_archive,
                transcribe_archive_folder
            ],
            #output_type=CompressionResponse # Gemini API does not support sructured outputs with function calls
        )
    
@function_tool
async def transcribe_single_archive(archive_file: str) -> dict:
    """
    Transcribe audio from a single .tar.gz archive using Speech-To-Text MCP Server (HTTP)
    
    Transcribes audio in a single archive to text using Faster Whisper.
    Creates output archive containing video and transcript.
    
    Args:
        archive_file: Path to a single .tar.gz archive containing audio
        
    Returns:
        Dictionary with archive path, file size, and status
    """
    if not Path(archive_file).exists():
        return {"success": False, "error": f"File not found: {archive_file}"}

    try:
        logger.info(f"Single-archive transcription tool called for: {archive_file}")

        mcp_server_url = os.getenv("SPEECH_TO_TEXT_MCP_SERVER_URL", "http://localhost:8004/mcp")

        # Connect to MCP server and call tool
        async with create_mcp_client(mcp_server_url) as client:
            # List available tools
            tools = await client.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            
            if "transcribe_single_archive" not in [t.name for t in tools]:
                error_msg = "transcribe_single_archive tool not found on MCP server"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }

            # Call transcription tool, routing output to session directory if available
            session_root = os.environ.get("SESSION_OUTPUT_ROOT")
            extra = {"output_dir": f"{session_root}/transcription"} if session_root else {}
            result = await client.call_tool("transcribe_single_archive", {"archive_file": archive_file, **extra})
            
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
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Transcription failed: {str(e)}"
        }


@function_tool
async def transcribe_archive_folder(folder_path: str) -> dict:
    """
    Transcribe audio from all .tar.gz archives in a folder using Speech-To-Text MCP Server (HTTP)
    
    Transcribes audio in all archives to text using Faster Whisper.
    Creates output archives for each containing video and transcript.
    
    Args:
        folder_path: Path to folder containing .tar.gz archives
        
    Returns:
        Dictionary with output folder path, total file size, and status
    """
    if not Path(folder_path).is_dir():
        return {"success": False, "error": f"Folder not found: {folder_path}"}

    try:
        logger.info(f"Folder transcription tool called for: {folder_path}")

        mcp_server_url = os.getenv("SPEECH_TO_TEXT_MCP_SERVER_URL", "http://localhost:8004/mcp")

        # Connect to MCP server and call tool
        async with create_mcp_client(mcp_server_url) as client:
            # List available tools
            tools = await client.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            
            if "transcribe_archive_folder" not in [t.name for t in tools]:
                error_msg = "transcribe_archive_folder tool not found on MCP server"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg
                }

            # Call transcription tool, routing output to session directory if available
            session_root = os.environ.get("SESSION_OUTPUT_ROOT")
            extra = {"output_dir": f"{session_root}/transcription"} if session_root else {}
            result = await client.call_tool("transcribe_archive_folder", {"folder_path": folder_path, **extra})
            
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
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Transcription failed: {str(e)}"
        }


async def example_usage():
    """Example of using the TranscriptionAgent with StreamableHTTP"""
    
    # Ensure API key is configured
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: No API key found!")
        print("Set GEMINI_API_KEY or OPENAI_API_KEY environment variable")
        return
    
    # Initialize agent with HTTP-based MCP server
    agent = TranscriptionAgent().agent
    
    print(f"🤖 Initialized: {agent}\n")

    for tool in agent.tools:
        if isinstance(tool, FunctionTool):
            print(tool.name)
            print(tool.description)
            print(json.dumps(tool.params_json_schema, indent=2))
            print()
    
    print("=" * 60)
    print("Transcribe audio files via HTTP")
    print("=" * 60)
    
    # Use the latest file from audio_extraction or compression output
    result = await Runner.run(agent, 
                              f"Transcribe audio from archives in the folder /data/audio_extraction")
    
    print(f"Transcription Result: {result}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
