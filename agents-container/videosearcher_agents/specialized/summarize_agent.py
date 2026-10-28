import os
import asyncio
import argparse
from pathlib import Path
from typing import Optional

from agents import Agent, Runner, function_tool, FunctionTool
import logging

from videosearcher_agents.llm_utils import build_llm, resolve_llm_config
from videosearcher_agents.mcp_streamable_client import create_mcp_client

logger = logging.getLogger(__name__)


class SummarizationAgent:
    """
    Summarization Agent using OpenAI Agents SDK with StreamableHTTP MCP Client.

    Connects to the Summarize MCP Server via HTTP to summarize the transcript
    inside an archive (produced by the transcription tool).

    Architecture:
        Agent -> StreamableHTTP Client -> HTTP -> MCP Server -> LLM Summarization
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
        logger.info(f"Initializing SummarizationAgent with LLM model: {cfg.model}")

        self.agent = Agent(
            name="SummarizationAgent",
            instructions="""Summarize the transcript inside an archive and report the result.

            STEP 1 — Call the tool immediately. Never ask for clarification.
            Extract the archive path from the input and call
            summarize_single_archive(archive_file=<path>).

            STEP 2 — Report the result in this exact format:
            If success:
              "DONE: output_path=<output_path>"
            If error:
              "ERROR: <message>"

            Do NOT add commentary, plans, or extra text. Return ONLY the one-line result above.""",
            model=self.model,
            tools=[summarize_single_archive, summarize_archive_folder],
        )


@function_tool
async def summarize_single_archive(archive_file: str) -> dict:
    """
    Summarize the transcript in a single .tar.gz archive via the Summarize MCP Server (HTTP).

    Args:
        archive_file: Path to a .tar.gz archive containing transcript.txt.
    Returns:
        Dictionary with output_path, summary, output_size, and status.
    """
    if not Path(archive_file).exists():
        return {"success": False, "error": f"File not found: {archive_file}"}
    try:
        logger.info(f"Single-archive summarization tool called for: {archive_file}")
        mcp_server_url = os.getenv("SUMMARIZE_MCP_SERVER_URL", "http://localhost:8016/mcp")
        async with create_mcp_client(mcp_server_url) as client:
            tools = await client.list_tools()
            logger.info(f"Available tools: {[t.name for t in tools]}")
            if "summarize_single_archive" not in [t.name for t in tools]:
                return {"success": False, "error": "summarize_single_archive tool not found on MCP server"}

            session_root = os.environ.get("SESSION_OUTPUT_ROOT")
            extra = {"output_dir": f"{session_root}/summarization"} if session_root else {}
            result = await client.call_tool("summarize_single_archive", {"archive_file": archive_file, **extra})
            logger.info(f"Tool result: {result}")

            if isinstance(result, dict) and result.get("success"):
                return {
                    "output_path": result.get("output_path"),
                    "output_size": result.get("output_size"),
                    "summary": result.get("summary"),
                    "processing_time_ms": result.get("processing_time_ms"),
                    "success": True,
                }
            error = result.get("error", "Unknown error from MCP server") if isinstance(result, dict) else str(result)
            return {"success": False, "error": error}
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Summarization failed: {str(e)}"}


@function_tool
async def summarize_archive_folder(folder_path: str) -> dict:
    """
    Summarize transcripts in all .tar.gz archives in a folder via the Summarize MCP Server (HTTP).

    Args:
        folder_path: Path to a folder of .tar.gz archives containing transcript.txt.
    Returns:
        Dictionary with output_path, summary, output_size, and status.
    """
    if not Path(folder_path).is_dir():
        return {"success": False, "error": f"Folder not found: {folder_path}"}
    try:
        logger.info(f"Folder summarization tool called for: {folder_path}")
        mcp_server_url = os.getenv("SUMMARIZE_MCP_SERVER_URL", "http://localhost:8016/mcp")
        async with create_mcp_client(mcp_server_url) as client:
            tools = await client.list_tools()
            if "summarize_archive_folder" not in [t.name for t in tools]:
                return {"success": False, "error": "summarize_archive_folder tool not found on MCP server"}

            session_root = os.environ.get("SESSION_OUTPUT_ROOT")
            extra = {"output_dir": f"{session_root}/summarization"} if session_root else {}
            result = await client.call_tool("summarize_archive_folder", {"folder_path": folder_path, **extra})
            if isinstance(result, dict) and result.get("success"):
                return {
                    "output_path": result.get("output_path"),
                    "output_size": result.get("output_size"),
                    "summary": result.get("summary"),
                    "processing_time_ms": result.get("processing_time_ms"),
                    "success": True,
                }
            error = result.get("error", "Unknown error from MCP server") if isinstance(result, dict) else str(result)
            return {"success": False, "error": error}
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}", exc_info=True)
        return {"success": False, "error": f"Summarization failed: {str(e)}"}


async def example_usage(archive_path: str):
    agent = SummarizationAgent().agent
    print(f"🤖 Initialized: {agent}\n")
    result = await Runner.run(agent, f"Summarize the transcript in the archive at {archive_path}")
    print(f"Summarization Result: {result}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Summarization Agent — summarize a transcript archive")
    parser.add_argument("archive_path", nargs="?", default="/data/transcription", help="Path to archive or folder (default: %(default)s)")
    args = parser.parse_args()
    asyncio.run(example_usage(args.archive_path))
