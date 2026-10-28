# Extension Tools — VideoSearcher AgenticAI

This document covers the three extension agents and their MCP servers added on top of the core audio/vision pipeline. These agents are registered with the orchestrator and can be invoked via natural-language requests.

---

## Overview

| Handoff tool | Agent class | MCP server | Port | Purpose |
|---|---|---|---:|---|
| `summarize_handoff` | `SummarizationAgent` | `summarize-mcp` | 8016 | Summarize a video transcript using an LLM |
| `web_search_handoff` | `WebSearchAgent` | `web-search-mcp` | 8010 | Search the web for information |
| `scene_detect_handoff` | `SceneDetectAgent` | `scene-detect-mcp` | 8025 | Detect shot/scene boundaries in a video |

These are extension tools — the core pipeline (audio extraction → transcription → search → frame extraction → VQA/segmentation) works without them. They add capabilities for summarization, knowledge enrichment, and visual cut detection.

All three extension servers report energy via CodeCarbon (Layer 2): each wraps its tool with the `_energy_tracker()` helper, writing to `/app/emissions` under project names `summarization` (8016), `web_search` (8010), and `scene_detection` (8025). See [`CODECARBON.md`](CODECARBON.md).

---

## 1. Summarization — `summarize_handoff`

### What it does

Takes a transcription archive (output from `transcription_handoff`) and produces a concise text summary of the transcript content using an LLM.

### When the orchestrator uses it

When the user asks to summarize, condense, or get a brief overview of what is said in a video.

Example requests:
```text
> Summarize what is discussed in /data/raw/tedtalk.mp4
> Give me a brief summary of the lecture at /data/raw/lecture.mp4
```

### Pipeline chain

**Chain I (Transcript Summarization):**
```
audio_extraction_handoff → transcription_handoff → summarize_handoff
```

### MCP server

**`tools/summarize-mcp/`** — FastMCP server. It reads the `transcript.txt` from the archive, calls the configured LLM (same as the orchestrator), and writes a `summary.txt` into an output archive.

**Input:** path to a `.tar.gz` transcription archive (contains `transcript.txt`)  
**Output:** archive with `summary.txt` + the summary text in the tool result

---

## 2. Web Search — `web_search_handoff`

### What it does

Searches the web for a query string and returns ranked results (title, URL, snippet). Uses DuckDuckGo by default (no API key required); supports Tavily, Serper, or Brave with the appropriate API key configured.

### When the orchestrator uses it

When the user wants to fact-check something mentioned in a video, look up additional context, or enrich the analysis with external information.

Example requests:
```text
> What is the maximum altitude of an F-35? Search the web.
> Find information about the TED speaker who talks about "scrolling Facebook"
```

### Pipeline chain

**Chain II (Web Fact-Check):**
```
[transcription_handoff →] web_search_handoff(query string)
```

The orchestrator can extract a topic or claim from the transcript and then use `web_search_handoff` to look it up. Web search can also be used standalone with any query string.

### MCP server

**`tools/web-search-mcp/`** — FastMCP server. Wraps DuckDuckGo (via `duckduckgo-search`) for zero-configuration web search. Optional API key configuration:

| Provider | Env var | Notes |
|---|---|---|
| DuckDuckGo | — | Default, no key needed |
| Tavily | `TAVILY_API_KEY` | Higher quality results, requires account |
| Serper | `SERPER_API_KEY` | Google SERP results |
| Brave | `BRAVE_API_KEY` | Privacy-focused search |

**Input:** a search query string  
**Output:** ranked list of results with title, URL, and snippet

### Test fixture

`tests/fixtures/web_search/` contains pre-recorded mock results for deterministic web search testing.

---

## 3. Scene Detection — `scene_detect_handoff`

### What it does

Detects visual scene/shot boundaries in a video — the moments where the camera cuts from one scene to another. Uses PySceneDetect with the `ContentDetector` algorithm, which looks for frame-to-frame color and luminance changes.

This is different from silence-based splitting (`audio_analysis_handoff` + `video_splitting_handoff`):

| Method | What it detects | Input |
|---|---|---|
| Silence-based splitting | Pauses in speech (audio silence) | Audio waveform |
| Scene detection | Visual cuts (camera changes) | Video frame differences |

### When the orchestrator uses it

When the user wants to find the visual structure of a video — where scenes change, not where speech pauses.

Example requests:
```text
> Find all scene boundaries in /data/raw/f-35.mp4
> Detect shot changes in /data/raw/demo.mp4
```

### MCP server

**`tools/scene-detect-mcp/`** — FastMCP server. Wraps PySceneDetect with the `ContentDetector` algorithm.

**Input:** path to a `.mp4` video file, or an archive containing one  
**Output:** list of scene boundary timestamps (seconds), split by scene — not by silence

---

## Integration in the Orchestrator System Prompt

The orchestrator's system prompt includes an "EXTENSION TOOLS" section that lists these three agents and their chains:

```
Transcript-based (input = transcription archive):
- summarize_handoff       — summarize the transcript

Video/image-based:
- scene_detect_handoff    — shot/scene boundaries (visual cuts, not silence)

Web/knowledge:
- web_search_handoff      — search the web (query string)

EXTENSION CHAINS:
  I.  Summarize transcript: audio_extraction → transcription → summarize
  II. Web fact-check:       transcription → web_search (query string)
```

---

## Docker and Port Configuration

Extension tools are defined in `docker-compose.yml` alongside the core MCP servers. Their URLs are set as environment variables in `docker-compose.agents.yml`:

| Variable | Value |
|---|---|
| `SUMMARIZE_MCP_SERVER_URL` | `http://summarize-mcp-server:8016/mcp` |
| `WEB_SEARCH_MCP_SERVER_URL` | `http://web-search-mcp-server:8010/mcp` |
| `SCENE_DETECT_MCP_SERVER_URL` | `http://scene-detect-mcp-server:8025/mcp` |

---

## Testing Extension Tools

The test suite in `tests/scenarios/` includes extension tool scenarios. Run them with:

```bash
# Inside the agents-orchestrator container
python tests/orchestrator/run_tests.py --scenarios SC-40,SC-42,SC-47
```

SC-42 (web search) has both a live integration test and a deterministic mocked test — the mock uses pre-recorded results from `tests/fixtures/web_search/`.
