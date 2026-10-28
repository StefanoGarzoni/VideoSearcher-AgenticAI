# Design Decisions — VideoSearcher AgenticAI

This document records the key architectural choices made in this project, the alternatives that were considered, and the reasons for each decision. Reading this is the fastest way to understand *why* the system is built the way it is — the kind of questions a professor or technical reviewer will ask.

---

## 1. MCP (Model Context Protocol) for Agent–Tool Communication

**Decision:** All agents communicate with tool servers exclusively through MCP over HTTP (`/mcp` endpoint). Agents never import tool code directly.

**Why:**
- **Dependency isolation.** Tools like FFmpeg, Librosa, Whisper, GroundingDINO+SAM2, PySceneDetect, and DuckDuckGo each have conflicting Python or CUDA requirements. Putting them in separate containers prevents dependency hell.
- **Replaceability.** A tool server can be rewritten in any language or replaced with a better implementation without touching any agent code. The contract is the MCP schema, not a function signature.
- **Protocol standardization.** MCP is a standard (similar in spirit to OpenAPI for tools) that is explicitly supported by the OpenAI Agents SDK used in this project, making the integration clean and forward-compatible.

**Alternative considered:** Direct Python imports (no Docker, no HTTP). Rejected because all components would share a single Python environment — impossible when Librosa, Whisper, and SAM2 each pin conflicting versions.

---

## 2. Handoff Pattern (Orchestrator + Specialized Agents)

**Decision:** One orchestrator agent coordinates multiple specialized sub-agents via the Handoff pattern from the OpenAI Agents SDK. Each specialized agent handles exactly one domain (audio extraction, transcription, segmentation, etc.).

**Why:**
- **Single responsibility.** Each agent has a narrow system prompt and a small tool set. A specialized agent is easier to test in isolation and easier to reason about than one mega-agent with 15 tools.
- **Parallel reasoning.** The orchestrator reasons about which agents to invoke and in what order, while each specialized agent reasons about how to use its specific MCP tools. These are genuinely different reasoning tasks and benefit from separate LLM calls.
- **Observability.** The `handoff_sequence` field in every test result shows exactly which agents were called in which order. This makes debugging LLM routing failures straightforward.
- **Reusability.** Specialized agents can be called individually (e.g., directly calling `FrameExtractionAgent` in unit tests) without going through the orchestrator.

**Alternative considered:** Single agent with all tools. Rejected because a flat list of 15+ tools causes LLMs to make more routing errors and produces unreadable system prompts.

---

## 3. Local LLM for the Orchestrator (llama.cpp + GGUF)

**Decision:** The orchestrator LLM is a local GGUF model served by `llama-server` from `llama.cpp`, running on the host GPU. `gemma-4-E2B-it-Q8_0.gguf` is the **default**, but `run_agentic_system.sh` ships a **multi-model registry** (`gemma`, `gemma4-12b`, `gpt-oss-120b`, `qwen36-27b`, `qwen36-a3b`, plus legacy keys `phi4-mini`, `qwen25-7b`, …) selectable via `--orchestrator-model KEY`. Multiple models are provided deliberately so capability / latency / energy trade-offs can be compared (see `scripts/run_model_comparison.py`, which benchmarks every model over the full suite and records **TTFT** and per-model **energy**). The VQA model is held constant so the orchestrator LLM is the only variable.

**Why:**
- **No API cost.** Running the suite (22 scenarios × 3 levels = 66 runs per model, up to 330 across all 5) against a cloud API like Gemini or GPT-4 would be expensive during development iteration.
- **Reproducibility.** A fixed local GGUF produces the same outputs for the same inputs without rate limits or model deprecation surprises.
- **Offline operation.** The system can run in an air-gapped lab environment.
- **GPU utilization.** The GB10 GPU (128 GB unified memory) can host a large local model alongside the segmentation and VQA models simultaneously.

**Tradeoff:** A 2B quantized model (Gemma-4-E2B Q8) is significantly less capable than GPT-4o or Gemini Pro at multi-step pipeline routing. This is why the orchestrator system prompt is so explicit (with named pipeline chains A–H) — it compensates for model capability with structured instructions.

**LiteLLM compatibility.** The system also supports cloud providers (Gemini, OpenAI) by changing two env vars (`LLM_MODEL`, `LLM_ORCHESTRATOR_MODEL`). This was used during early development before the local model was set up.

---

## 4. VQA Runs In-Process (Not as an MCP Server)

**Decision:** The VQA agent (`VqaAgent`) loads and runs Phi-4-reasoning-vision-15B directly inside the `agents-orchestrator` container via HuggingFace Transformers. There is no separate MCP server for VQA.

**Why:**
- **Model size.** Phi-4 is a 15B parameter model that weighs ~30 GB in bfloat16. Serializing its outputs over HTTP is fine, but running it inside a separate Docker container on a separate GPU would require 30 GB of separate VRAM — impossible if only one GPU is available.
- **No intermediate serialization.** Frame images are accessed as file paths on the shared `/data` volume. Both the agent container and the VQA model see the same file system, so no image encoding/decoding over HTTP is needed.
- **Startup latency.** Loading the model takes 2–5 minutes on a warm cache. Running it in-process means this cost is paid once per orchestrator startup, not per request. The test runner pre-warms the model before the first scenario timer starts.

**Tradeoff:** The orchestrator container now needs `runtime: nvidia` and ≥30 GB GPU VRAM. This is a significant infrastructure requirement.

---

## 5. Segmentation as an MCP Server (Not In-Process)

**Decision:** GroundingDINO + SAM2 run inside `vlm-segmentation-mcp-server` as a separate container. This is the opposite decision from VQA.

**Why:**
- **Smaller footprint.** GroundingDINO + SAM2 together use approximately 3 GB VRAM. This is small enough to run in a separate container without memory pressure.
- **Better isolation.** The segmentation models have complex CUDA and Python dependencies (torchvision at a specific version, custom SAM2 weights). Keeping them in their own container prevents dependency conflicts with the orchestrator.
- **HTTP calls are fine.** Segmentation inputs and outputs are file paths (the images live on the shared volume). The MCP call only carries the path string and the text query, not raw image bytes.

---

## 6. Docker for All MCP Tool Servers

**Decision:** Every MCP server runs in its own Docker container defined in `docker-compose.yml`.

**Why:**
- **Dependency isolation.** FFmpeg (binary), Librosa (Python + audio codec bindings), Whisper (OpenAI library), GroundingDINO+SAM2 (custom CUDA code), PySceneDetect (OpenCV), DuckDuckGo search (requests) — each has different system requirements. Docker makes each container self-contained.
- **Reproducibility.** Any developer with Docker + NVIDIA Container Toolkit can run the full stack. No manual system-level installations needed beyond Docker.
- **Horizontal scaling.** Individual MCP servers can be scaled or replaced without restarting the whole stack. A faster transcription server can be swapped in by rebuilding one image.

**Tradeoff:** Docker adds startup overhead (pulling images, initializing containers). The vlm-segmentation container takes ~60 seconds to initialize because GroundingDINO and SAM2 load their models at startup.

---

## 7. StreamableHTTP Transport for MCP

**Decision:** The MCP client (`mcp_streamable_client.py`) uses the StreamableHTTP transport (JSON-RPC over HTTP POST to `/mcp`), not the stdio transport.

**Why:**
- **Container network.** Agents and tools run in separate Docker containers. The only natural communication channel between them is the Docker network (HTTP). Stdio transport only works within the same process.
- **Health checking.** Each MCP server exposes a `GET /health` endpoint alongside the `POST /mcp` endpoint. The test runner polls `/health` before starting test timers, so infrastructure startup failures are distinguished from actual test failures.
- **Tool schema caching.** The client caches `list_tools()` results (keyed by server URL) so each tool's JSON schema is fetched once per process lifetime rather than once per call. This significantly reduces round-trips during long test runs.

---

## 8. Shared `/data` Volume for All Containers

**Decision:** All containers (agent container + all MCP tool servers) mount `./data` at `/data`. Intermediate artifacts (audio files, transcripts, frames, masks) are written here and referenced by path.

**Why:**
- **No serialization overhead.** A video frame or audio file can be hundreds of megabytes. Passing it through MCP as base64 would be orders of magnitude slower and would break the HTTP buffer limits.
- **Debugging.** Every intermediate artifact is visible on the host filesystem. If a pipeline step fails, you can inspect the exact file that was passed to the next step.
- **Session isolation.** Each orchestrator run creates a unique session directory under `/data/runs/{TIMESTAMP}_{HEX}` so parallel runs don't overwrite each other.

---

## 9. Three-Layer Transcript Search (Grep Tool)

**Decision:** The grep tool matches search queries against transcripts in three layers: (1) normalized literal, (2) word-level flexible regex, (3) RapidFuzz fuzzy fallback.

**Why:**
- **Whisper output is imperfect.** Whisper may add punctuation, change capitalization, or slightly paraphrase what was said. A plain `str.find()` would miss "neural network." vs "neural network" vs "Neural Networks".
- **Graceful degradation.** Each layer only runs if the previous returned zero matches. The response always indicates which layer matched (`match_type: "literal" | "fuzzy"`), so callers can assess match quality.
- **Timestamp preservation.** The `[MM:SS.mmm]` prefix in each transcript line survives all three layers unchanged, enabling cross-modal chaining from audio search → frame extraction.

---

## 10. Behavioral Contract Testing (Not Accuracy Metrics)

**Decision:** The test suite checks *routing behavior* (did the orchestrator call the right agents in the right order?) rather than output accuracy (is the transcript word-for-word correct?).

**Why:**
- **The orchestrator's job is routing.** The core risk in an agentic system is the LLM picking the wrong agent, calling agents in the wrong order, or stopping mid-pipeline. These bugs are not caught by checking if the transcript has 99% WER.
- **Labeled datasets are expensive.** Building ground-truth labels for a multi-modal video system (correct transcripts, correct masks, correct VQA answers) requires significant manual effort. Behavioral contracts require only the list of expected agents.
- **The tools themselves handle correctness.** Whisper handles transcription quality; GroundingDINO+SAM2 handle segmentation quality. The tests validate that these tools are called correctly, not that they produce perfect results.

**Where this falls short:** SC-01 passes if AudioExtraction is called — even if the extracted audio is silent. `tests/unit/` contains structural assertions on individual MCP servers that partially fill this gap.

---

## 11. `--reasoning-budget 0` for Gemma-4 (and Why NOT for Other Models)

**Decision:** The local LLM server is started with `--reasoning-budget 0`, which suppresses the internal "thinking" token generation in Gemma-4-E2B before each tool call.

**Why:** Gemma-4 is a "thinking" model that by default generates 640+ internal reasoning tokens (~183 seconds) before each tool call. A 2-step pipeline would spend 6+ minutes thinking before any agent was invoked. With the 300-second test timeout, most tests timed out before the first tool call.

**Important:** `--reasoning-budget 0` is specific to DeepSeek-R1 / "thinking" models. On standard instruct models (Qwen2.5, Llama, Phi-4-mini), it suppresses end-of-sequence tokens, causing runaway generation. The script uses `--n-predict 2048` for non-thinking models instead.

---

## 12. Three-Layer CodeCarbon Energy Tracking (Internal, Not Wrapped)

**Decision:** Energy is tracked *inside* each component rather than by wrapping the launcher. Three layers: **L2** — every one of the 12 MCP tool servers starts its own `EmissionsTracker` per call (guarded `_energy_tracker()` helper); **L1** — the orchestrator LLM and the in-process VQA model (`orchestrator_request` gpu energy + `vqa_phi4_inference`); **L3** — orchestration overhead (`orchestrator_request` cpu+ram). All write to the shared `logs/emissions/` volume and are summed by `scripts/aggregate_emissions.py`.

**Why:** The orchestrator LLM (`llama-server`) is a separate host process and the tools are separate containers, so no single wrapper can see all the work. Per-component trackers attribute energy to the right place, and CodeCarbon's cpu/gpu/ram split lets one `orchestrator_request` tracker serve both L1 (gpu ≈ LLM) and L3 (cpu+ram ≈ overhead).

**Tradeoff:** A single shared GPU means `gpu_energy` is whole-GPU during a request window, and a VQA call is double-counted in both `orchestrator_request.gpu` and `vqa_phi4_inference` — these are documented caveats, not summed blindly. See [`CODECARBON.md`](CODECARBON.md).
