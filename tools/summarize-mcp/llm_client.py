"""Minimal OpenAI-compatible LLM client for MCP tool servers.

Resolves an LLM backend from environment variables with a local-first,
cloud-fallback strategy so the same tool works against a local llama.cpp /
vLLM server OR a cloud API (OpenAI, Gemini) without code changes.

Resolution order (first match wins), with an optional TOOL-specific prefix
(e.g. prefix="SUMMARIZE" checks SUMMARIZE_LLM_BASE_URL first):
  1) <PREFIX>_LLM_BASE_URL / LLM_BASE_URL / LLM_ORCHESTRATOR_BASE_URL
        -> local or custom OpenAI-compatible endpoint
  2) OPENAI_API_KEY   -> OpenAI cloud
  3) GEMINI_API_KEY   -> Gemini cloud (OpenAI-compatible endpoint)

This module is intentionally self-contained (no project imports) so it can be
copied verbatim into any tool container that needs LLM access.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _env(prefix: str, key: str) -> Optional[str]:
    if prefix:
        v = os.getenv(f"{prefix}_{key}")
        if v:
            return v
    return None


def resolve_llm(prefix: str = ""):
    """Return (client, model_name). Raises RuntimeError if nothing is configured."""
    from openai import OpenAI

    base_url = (
        _env(prefix, "LLM_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or os.getenv("LLM_ORCHESTRATOR_BASE_URL")
    )
    model = (
        _env(prefix, "LLM_MODEL")
        or os.getenv("LLM_MODEL")
        or os.getenv("LLM_ORCHESTRATOR_MODEL")
    )
    api_key = _env(prefix, "LLM_API_KEY") or os.getenv("LLM_API_KEY")

    # 1) Explicit local / custom OpenAI-compatible endpoint
    if base_url:
        client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        return client, (model or "local-model")

    # 2) OpenAI cloud
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return client, (model or "gpt-4o-mini")

    # 3) Gemini cloud (OpenAI-compatible)
    if os.getenv("GEMINI_API_KEY"):
        client = OpenAI(base_url=_GEMINI_BASE, api_key=os.getenv("GEMINI_API_KEY"))
        return client, (model or "gemini-2.0-flash")

    raise RuntimeError(
        "No LLM backend configured. Set LLM_BASE_URL (local), OPENAI_API_KEY, or GEMINI_API_KEY."
    )


def chat(messages: list, prefix: str = "", temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """Run a chat completion and return the text content."""
    client, model = resolve_llm(prefix)
    logger.info("LLM chat via model=%s", model)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()
