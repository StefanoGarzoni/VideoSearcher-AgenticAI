# StreamableHTTP Architecture & Topology

## Topology

```text
Agent container
  -> StreamableHTTP client
     -> MCP servers (/mcp, /health)
        -> tool execution
           -> outputs in /data
```

The local LLM is separate from MCP:

```text
Agent container
  -> http://host.docker.internal:9000/v1
     -> local-ai-service/run_agentic_system.sh
        -> llama-server
           -> local GGUF model (registry, --orchestrator-model)
```

---

## Request Flow

```text
Specialized agent
  -> MCP StreamableHTTP client
     -> POST /mcp
        -> JSON-RPC tool call
           -> MCP tool handler executes
              -> result streamed back
```

---

## Container Network

MCP servers run on the Docker network:

```text
videosearcher_agenticai_videosearcher
```

The agent container receives MCP URLs from `docker-compose.agents.yml`.

All containers share:

```text
./data -> /data
```

---

## MCP Servers

| Server | Port | Endpoint | Purpose |
|---|---:|---|---|
| `ffmpeg-0-mcp-server` | 8000 | `/mcp` | Audio extraction |
| `librosa-mcp-server` | 8001 | `/mcp` | Audio analysis |
| `ffmpeg-1-mcp-server` | 8002 | `/mcp` | Video splitting |
| `ffmpeg-2-mcp-server` | 8003 | `/mcp` | Compression |
| `speech-to-text-mcp-server` | 8004 | `/mcp` | Transcription |
| `grep-mcp-server` | 8005 | `/mcp` | Transcript search |
| `ffmpeg-frames-mcp-server` | 8006 | `/mcp` | Frame extraction |
| `vlm-segmentation-mcp-server` | 8008 | `/mcp` | Image/frame segmentation |
| `video-segmentation-mcp-server` | 8009 | `/mcp` | Video segmentation |
| `web-search-mcp-server` | 8010 | `/mcp` | Web search |
| `summarize-mcp-server` | 8016 | `/mcp` | Transcript summarization |
| `scene-detect-mcp-server` | 8025 | `/mcp` | Scene/shot detection |

The orchestrator LLM that calls these servers is selectable from a 5-model registry via
`run_agentic_system.sh --orchestrator-model KEY` (default `gemma`).

---

## Health Endpoints

Every MCP server exposes:

```text
GET /health
```

Responses always include at least:

```json
{
  "status": "ok"
}
```

Some services expose additional fields, for example:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

or:

```json
{
  "status": "ok",
  "server": "video-segmentation",
  "ffmpeg_available": true,
  "ffprobe_available": true
}
```

---

## Segmentation Flow

### Image/frame segmentation

```text
SegmentationAgent
  -> vlm-segmentation-mcp-server:8008/mcp
     -> segment_frame or segment_image
        -> output_mode="masks" or "segmented_image"
```

### Video segmentation

```text
SegmentationAgent
  -> video-segmentation-mcp-server:8009/mcp
     -> segment_video
        -> extracts frames with FFmpeg
        -> calls vlm-segmentation-mcp-server for each frame
        -> returns segmented video OR segmented frames
```

The video segmentation server does not load a second VLM model.

---

## Local LLM Flow

`local-ai-service/run_agentic_system.sh` starts (among the full stack):

```text
llama-server --host 0.0.0.0 --port 9000
```

The host can access:

```text
http://localhost:9000
```

The Docker agent container accesses:

```text
http://host.docker.internal:9000/v1
```

Recommended agent variables:

```bash
LLM_PROVIDER=openai-chat
LLM_MODEL=gemma-4-E2B-it-Q8_0
LLM_ORCHESTRATOR_MODEL=gemma-4-E2B-it-Q8_0
LLM_ORCHESTRATOR_BASE_URL=http://host.docker.internal:9000/v1
LLM_BASE_URL=http://host.docker.internal:9000/v1
OPENAI_API_KEY=local
LLM_API_KEY=local
```