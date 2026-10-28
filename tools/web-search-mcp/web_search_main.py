"""Core web search logic for the Web Search MCP server.

Supports multiple providers, defaulting to DuckDuckGo (no API key required).
Optional providers (Tavily, Serper, Brave) activate automatically when their
API key env var is set, or when explicitly requested via the `provider` arg.
"""

import os
import json
import logging
import uuid
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


def _search_duckduckgo(query: str, max_results: int) -> list:
    """Keyless search via the `ddgs` package (DuckDuckGo)."""
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS  # type: ignore

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", r.get("url", "")),
                "snippet": r.get("body", r.get("snippet", "")),
            })
    return results


def _search_tavily(query: str, max_results: int) -> list:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")
    resp = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": max_results},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ]


def _search_serper(query: str, max_results: int) -> list:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY not set")
    resp = httpx.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": max_results},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
        for r in data.get("organic", [])[:max_results]
    ]


def _search_brave(query: str, max_results: int) -> list:
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise RuntimeError("BRAVE_API_KEY not set")
    resp = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        params={"q": query, "count": max_results},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    web = data.get("web", {}).get("results", [])
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
        for r in web[:max_results]
    ]


_PROVIDERS = {
    "duckduckgo": _search_duckduckgo,
    "tavily": _search_tavily,
    "serper": _search_serper,
    "brave": _search_brave,
}


def _resolve_provider(provider: Optional[str]) -> str:
    if provider:
        return provider.lower()
    env = os.getenv("WEB_SEARCH_PROVIDER")
    if env:
        return env.lower()
    # Auto-pick a keyed provider if available, else DuckDuckGo (keyless).
    if os.getenv("TAVILY_API_KEY"):
        return "tavily"
    if os.getenv("SERPER_API_KEY"):
        return "serper"
    if os.getenv("BRAVE_API_KEY"):
        return "brave"
    return "duckduckgo"


def web_search(query: str, output_dir: str, max_results: int = 5,
               provider: Optional[str] = None) -> tuple[bool, str, dict]:
    """Run a web search and persist the results as JSON.

    Returns (success, error, results_dict).
    """
    if not query or not query.strip():
        return False, "query is required", {}

    name = _resolve_provider(provider)
    fn = _PROVIDERS.get(name)
    if fn is None:
        return False, f"Unknown provider '{name}'. Options: {list(_PROVIDERS)}", {}

    try:
        logger.info("Web search via %s: %r (max_results=%d)", name, query, max_results)
        results = fn(query, max_results)
    except Exception as e:
        logger.error("Web search failed (%s): %s", name, e, exc_info=True)
        # When a keyed provider was AUTO-selected (not explicitly requested) and
        # fails — e.g. an expired/rate-limited key — fall back to the keyless
        # DuckDuckGo path instead of losing web search entirely. An explicit
        # provider request still fails honestly.
        auto_selected = provider is None and not os.getenv("WEB_SEARCH_PROVIDER")
        if name != "duckduckgo" and auto_selected:
            try:
                logger.info("Falling back to duckduckgo after %s failure", name)
                results = _search_duckduckgo(query, max_results)
                name = "duckduckgo"
            except Exception as e2:
                return False, f"Web search failed ({name}: {e}; duckduckgo fallback: {e2})", {}
        else:
            return False, f"Web search failed ({name}): {e}", {}

    payload = {
        "success": True,
        "provider": name,
        "query": query,
        "count": len(results),
        "results": results,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"web_search_{uuid.uuid4().hex[:8]}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    payload["output_path"] = out_path
    return True, "", payload
