"""
Orchestrator run logger — saves plan, tool calls and reasoning to a JSON file.

Usage:
    from videosearcher_agents.run_logger import log_run
    result = await Runner.run(agent, request)
    log_run(result, request)

Logs are written to LOG_DIR (default: /data/logs/orchestrator/).
Set ORCHESTRATOR_LOG_DIR env var to override.
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_DIR = os.getenv("ORCHESTRATOR_LOG_DIR", "/data/logs/orchestrator")


def _serialise_item(item, step: int) -> dict:
    """Convert a single RunItem to a readable JSON-safe dict."""
    from agents.items import (
        MessageOutputItem,
        ToolCallItem,
        ToolCallOutputItem,
        HandoffCallItem,
        HandoffOutputItem,
        ReasoningItem,
    )

    agent_name = getattr(item.agent, "name", str(item.agent))
    entry: dict = {"step": step}

    if isinstance(item, MessageOutputItem):
        texts = []
        for part in getattr(item.raw_item, "content", []):
            if hasattr(part, "text"):
                texts.append(part.text)
        entry["actor"] = agent_name
        entry["action"] = "responds to user"
        entry["message"] = "\n".join(texts) if texts else str(item.raw_item)

    elif isinstance(item, ToolCallItem):
        raw = item.raw_item
        tool_name = getattr(raw, "name", None) or getattr(raw, "type", "unknown")
        entry["actor"] = agent_name
        entry["action"] = f"calls {tool_name}"
        # Try to parse arguments as JSON for readability
        raw_args = getattr(raw, "arguments", None)
        try:
            entry["arguments"] = json.loads(raw_args) if raw_args else None
        except (json.JSONDecodeError, TypeError):
            entry["arguments"] = raw_args

    elif isinstance(item, ToolCallOutputItem):
        output = item.output
        text = str(output)
        entry["actor"] = agent_name
        entry["action"] = "returns result"
        # Try to parse output as JSON for readability
        try:
            entry["output"] = json.loads(text) if text else None
        except (json.JSONDecodeError, TypeError):
            entry["output"] = text[:2000] + "…" if len(text) > 2000 else text

    elif isinstance(item, HandoffCallItem):
        raw = item.raw_item
        entry["actor"] = agent_name
        entry["action"] = f"hands off to {getattr(raw, 'name', 'agent')}"
        raw_args = getattr(raw, "arguments", None)
        try:
            entry["arguments"] = json.loads(raw_args) if raw_args else None
        except (json.JSONDecodeError, TypeError):
            entry["arguments"] = raw_args

    elif isinstance(item, HandoffOutputItem):
        entry["actor"] = getattr(item.source_agent, "name", str(item.source_agent))
        entry["action"] = f"delegated to {getattr(item.target_agent, 'name', str(item.target_agent))}"

    elif isinstance(item, ReasoningItem):
        raw = item.raw_item
        summary = getattr(raw, "summary", None)
        entry["actor"] = agent_name
        entry["action"] = "reasoning"
        if summary and isinstance(summary, list):
            entry["thinking"] = [getattr(s, "text", str(s)) for s in summary]
        else:
            entry["thinking"] = str(raw)

    else:
        entry["actor"] = agent_name
        entry["action"] = getattr(item, "type", item.__class__.__name__)
        entry["raw"] = str(item.raw_item)[:1000]

    return entry


def _extract_raw_conversation(raw_responses) -> list[dict]:
    """
    Extract the full model conversation from raw_responses.
    
    This captures intermediate text the orchestrator produces alongside tool calls
    (e.g. "I'll extract the audio first…") that new_items alone misses.
    """
    from openai.types.responses import (
        ResponseOutputMessage,
        ResponseFunctionToolCall,
        ResponseReasoningItem,
    )

    conversation = []
    for resp_idx, resp in enumerate(raw_responses):
        for item in resp.output:
            if isinstance(item, ResponseOutputMessage):
                texts = []
                for part in item.content:
                    if hasattr(part, "text"):
                        texts.append(part.text)
                if texts:
                    conversation.append({
                        "role": "assistant",
                        "content": "\n".join(texts),
                    })

            elif isinstance(item, ResponseFunctionToolCall):
                entry = {
                    "role": "tool_call",
                    "tool": item.name,
                }
                try:
                    entry["arguments"] = json.loads(item.arguments) if item.arguments else None
                except (json.JSONDecodeError, TypeError):
                    entry["arguments"] = item.arguments
                conversation.append(entry)

            elif isinstance(item, ResponseReasoningItem):
                summary = getattr(item, "summary", None)
                if summary and isinstance(summary, list):
                    texts = [getattr(s, "text", str(s)) for s in summary]
                else:
                    texts = [str(item)]
                conversation.append({
                    "role": "reasoning",
                    "content": "\n".join(texts),
                })

    return conversation


def log_run(result, request: str, *, log_dir: str | None = None) -> str | None:
    """
    Write a JSON log of an orchestrator run.

    Args:
        result: RunResult returned by Runner.run()
        request: The original user request string
        log_dir: Override log directory (default: LOG_DIR)

    Returns:
        Path to the log file, or None on failure.
    """
    target_dir = Path(log_dir or LOG_DIR)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Cannot create log dir %s: %s", target_dir, exc)
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    filename = target_dir / f"run_{timestamp}.json"

    payload = {
        "timestamp": timestamp,
        "request": request,
        "conversation": _extract_raw_conversation(result.raw_responses),
        "steps": [_serialise_item(item, i + 1) for i, item in enumerate(result.new_items)],
        "final_output": result.final_output,
    }

    try:
        with open(filename, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Run log saved to %s", filename)
        return str(filename)
    except OSError as exc:
        logger.warning("Failed to write run log: %s", exc)
        return None
