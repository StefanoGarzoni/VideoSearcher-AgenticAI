# Tests

Automated test suite for the VideoSearcher AgenticAI orchestrator. It covers 22 integration scenarios (9 audio/text + 10 vision + 3 extension) that run the full agent stack against real MCP tool servers, plus 4 unit-test suites that test each vision MCP server in isolation. Every test asserts a **behavioral contract**: did the orchestrator call the right agents in the right order, and does the output contain what it should? Where curated ground truth exists (`tests/ground_truth/`), a scenario additionally gets a metric-based **result check** (`ResultCriteria`: WER, grep match, VQA fact match, segmentation IoU/detect, …) computed by `tests/metrics/`.

---

## Table of Contents

1. [Test Philosophy — Why Behavioral Contracts](#1-test-philosophy--why-behavioral-contracts)
2. [Architecture Overview](#2-architecture-overview)
3. [Scenario Catalog](#3-scenario-catalog)
4. [How `handoff_match` Works](#4-how-handoff_match-works)
5. [Vision Success Criteria](#5-vision-success-criteria)
6. [Infrastructure Decisions](#6-infrastructure-decisions)
7. [Test Matrix & Current Results](#7-test-matrix--current-results)
8. [Running Tests](#8-running-tests)
9. [Configuration](#9-configuration)
10. [Adding New Scenarios](#10-adding-new-scenarios)
11. [Adding VQA Scenarios](#11-adding-vqa-scenarios)
12. [Result Format](#12-result-format)

---

## 1. Test Philosophy — Why Behavioral Contracts

### What we test

For every scenario we ask two questions:

1. **Did the orchestrator invoke the right agents, in the right order?**
   This is the `handoff_match` check. A scenario that should run `extract_frames → VQA` fails if the model runs `extract_frames → transcription` instead — even if the final text looks plausible.

2. **For vision scenarios: does the final output contain the expected content?**
   This is the `vision_pass` check. After VQA runs on an F-35 video frame, the orchestrator's final response must contain at least one word from a pre-defined keyword set (`f-35`, `fighter jet`, `aircraft`, ...). Scenarios with a `ResultCriteria` additionally score the answer against the curated ground truth in `tests/ground_truth/videos/f-35/` (`result_pass`).

### Why not accuracy metrics

Building a labeled evaluation dataset for a multi-modal video search system is expensive and fragile. The system's correctness is almost entirely captured by its *routing behavior*:

- Calling the transcription agent on an image is always a bug.
- Using `output_mode="segmented_image"` instead of `"masks"` when the next step needs masks is always a bug.
- Producing a VQA answer that says "fighter jet" instead of "F-35" is *not* a bug.

Behavioral contracts catch the bugs that matter without requiring ground-truth labels.

### Why keyword matching is sufficient for VQA

The vision tests use a wide keyword set (`f-35`, `fighter jet`, `stealth fighter`, `aircraft`, `jet`, `warplane`, `plane`, … — see `_AIRCRAFT_KEYWORDS`/`_PILOT_KEYWORDS` in `vision_scenarios.py`). Any of these words in the orchestrator's final output indicates the VLM ran and produced a plausible description of the frame. The keyword check does not care whether the answer is precise — it cares that the pipeline executed and returned something coherent. A model that returns an empty string, a file path, or an error message fails the `min_answer_length` guard. Precision is handled separately: where a `ResultCriteria` is set, the answer is scored against ground-truth facts (`vqa` metric) or masks (`iou`/`detect` metrics).

---

## 2. Architecture Overview

### Execution layers

```
┌────────────────────────────────────┐
│  test CLI (run_tests.py)           │  standalone, no pytest
│  pytest (test_orchestrator.py)     │  parametrized, supports -k filtering
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│  runner.py                         │  async batch runner
│  • pre-flight MCP health checks    │  polls /health before timers start
│  • VQA model pre-warming           │  loads VLM before scenario timers
│  • _run_single()                   │  one scenario × level × model
│  • incremental JSON saving         │  after every test
└──────────────┬─────────────────────┘
               │  asyncio timeout wrapper
┌──────────────▼─────────────────────┐
│  OrchestratorAgent                 │  the system under test
│  (videosearcher_agents/orchestrator│  receives prompt, dispatches to
│   /main_agent.py)                  │  specialized agents via handoffs
└──────────────┬─────────────────────┘
               │  HTTP / MCP protocol
┌──────────────▼─────────────────────┐
│  MCP Tool Servers  (real services) │  ffmpeg-frames :8006
│                                    │  vlm-segmentation :8008
│                                    │  speech-to-text :8004
│                                    │  (etc.)
└────────────────────────────────────┘
```

Tests hit real MCP servers — not mocks. This means a failing MCP server causes test failures, which is intentional: the goal is to validate end-to-end behavior.

### Data class hierarchy

All data classes live in `tests/scenarios/base.py`.

```
OrchestratorScenario   defines one test case
  ├─ scenario_id       "F-30"
  ├─ prompt_templates  {"vague": "...", "clear": "...", "algorithmic": "..."}
  ├─ expected_agents   ["ExtractFrames", "Vqa"]
  ├─ timeout_s         per-scenario override (None = global default)
  ├─ vision_criteria   VisionSuccessCriteria | None
  └─ result_criteria   ResultCriteria | None

VisionSuccessCriteria  behavioral contract for the orchestrator's output text
  ├─ vqa_keywords      tuple of acceptable keywords — OR-logic (any one must appear)
  ├─ required_keywords tuple of mandatory keywords — AND-logic (all must appear)
  └─ min_answer_length minimum output length (guards against empty strings)

ResultCriteria         ground-truth–backed correctness check (tests/metrics/)
  ├─ metric            "wer" | "grep" | "vqa" | "iou" | "detect" | "summary" | "scene" | "audio"
  ├─ ground_truth_ref  path relative to tests/ground_truth/ (e.g. "videos/f-35/vqa.json")
  ├─ threshold         metric-specific pass threshold (e.g. 0.30 WER, 0.5 VQA, 0.3 IoU)
  └─ item_index        optional index into the GT file's "items" array

ModelConfig            which LLM to use for a run
  ├─ model_name        "gemma-4-E2B-it-Q8_0.gguf"
  ├─ provider          "openai-chat"
  └─ base_url          "http://host.docker.internal:9000/v1"

TestResult             outcome of one scenario × level × model
  ├─ success           orchestrator completed without exception
  ├─ handoff_sequence  ["Extract Frames", "Vqa"]  (actual)
  ├─ expected_agents   ["ExtractFrames", "Vqa"]   (from scenario)
  ├─ handoff_match     computed property (see §4)
  ├─ vision_pass       bool | None  (set when vision_criteria exists)
  ├─ overall_pass      handoff_match AND vision_pass
  ├─ execution_time_ms elapsed wall-clock time
  └─ conversation      list[ConversationItem]  (full SDK item trace)

ConversationItem       one exchange in the trace
  ├─ item_type         "message" | "tool_call" | "tool_output" | "handoff"
  ├─ tool_name         agent or tool name
  └─ arguments         parsed JSON arguments (for tool_call items)
```

### File layout

```
tests/
├── README.md
├── conftest.py                      pytest fixtures (paths, URLs, derived video fixtures)
├── scenarios/
│   ├── base.py                      data classes and matching logic
│   ├── orchestrator_scenarios.py    SC-01..SC-09 (audio/text pipeline)
│   ├── vision_scenarios.py          F-28..F-37 (vision pipeline, f-35 fixtures;
│   │                                legacy SC-28..SC-37 preserved commented-out)
│   └── extension_scenarios.py       SC-40/42/47 (summarize, web search, scene detect)
├── metrics/                         result-check metrics (wer, grep, vqa, iou, …)
├── ground_truth/                    curated GT (videos/f-35/, videos/sample_video/, …)
├── orchestrator/
│   ├── runner.py                    core async runner
│   ├── run_tests.py                 standalone CLI entrypoint
│   └── test_orchestrator.py         pytest parametrized integration tests
├── unit/
│   ├── test_vqa.py                  Phi-4 VQA functional tests
│   ├── test_segmentation.py         GroundingDINO + SAM2 direct tests
│   ├── test_frame_extraction.py     ffmpeg-frames MCP server tests
│   ├── test_video_segmentation.py   video segmentation MCP server tests
│   └── test_metrics.py              unit tests for the metric implementations
└── results/                         JSON result files (gitignored)
```

---

## 3. Scenario Catalog

### 3a. Audio / Text Scenarios (SC-01 – SC-09)

These test the audio-analysis pipeline: audio extraction → silence detection → video splitting → compression → transcription → keyword search.

| ID | Name | Difficulty | Expected Agent Chain |
|----|------|-----------|----------------------|
| SC-01 | Single Tool: Audio Extraction | 1 | AudioExtraction |
| SC-02 | Single Tool: Transcription | 2 | Transcription |
| SC-03 | Single Tool: Grep Search | 3 | InformationFinder |
| SC-04 | Early Pipeline | 4 | AudioExtraction → AudioAnalysis |
| SC-05 | Late Pipeline | 5 | Compression → Transcription → InformationFinder |
| SC-06 | Full Pipeline | 6 | AudioExtraction → AudioAnalysis → VideoSplitting → Compression → Transcription → InformationFinder |
| SC-07 | Error: Wrong File Format | 7 | Reject |
| SC-08 | Error: Missing Input | 8 | RequestClarification |
| SC-09 | Null Results Handling | 10 | AudioExtraction → … → InformationFinder *(empty results)* |

SC-07 gives a `.png` image where a video is expected. SC-08 omits the file path entirely. SC-09 runs the full pipeline but searches for the word "Aliens" — a keyword that will not appear in the transcript. The test verifies that the pipeline completes gracefully and returns empty results rather than crashing.

### 3b. Vision Scenarios (F-28 – F-37)

These test the visual pipeline: frame extraction → VQA / segmentation. All vision scenarios use the **F-35 fixture set** (a ~88 s pilot + F-35 fighter-jet clip with speech) and, where a `ResultCriteria` is set, are scored against the curated ground truth in `tests/ground_truth/videos/f-35/`. They are the f-35 equivalents of the original lionandgorilla suite (SC-28..SC-37) — same behavioral objectives; the originals are preserved commented-out at the bottom of `vision_scenarios.py` (fixtures lost).

**Fixture strategy.** All fixtures are pre-populated in `data/fixtures/raw/`. The three reference frames are copied verbatim from `tests/ground_truth/videos/f-35/frames/` so they align exactly with the GT; the clip/wav are derived from `f-35.mp4` with ffmpeg:

- `f-35.mp4` — primary vision fixture (pilot + F-35 jet, ~88 s, 1280×720, has speech)
- `f-35_t05.jpg` — reference frame at t=5 s (pilot holding helmet; NO aircraft)
- `f-35_t15.jpg` — reference frame at t=15 s (F-35 in flight)
- `f-35_t65.jpg` — reference frame at t=65 s (4 jets in formation)
- `f-35_clip.mp4` — ~15 s clip (≈12–27 s) covering the in-flight jet, for video segmentation
- `f-35.wav` — audio track (first 10 s), used in the VQA-on-audio error scenario

Using pre-extracted reference frames breaks the dependency between VQA/segmentation tests and the frame-extraction MCP server, and keeps answers directly comparable to the GT.

| ID | Name | Expected Agents | Input | `vision_criteria` | `result_criteria` | Timeout |
|----|------|-----------------|-------|-------------------|-------------------|---------|
| F-28 | Single Agent: Frame Extraction | ExtractFrames | `f-35.mp4` | none | none | 300 s |
| F-29 | Single Agent: VQA on Image | Vqa | `f-35_t15.jpg` | keywords | `vqa` @ 0.5 | 900 s |
| F-30 | Frame → VQA | ExtractFrames → Vqa | `f-35.mp4` | keywords | `vqa` @ 0.5 | 900 s |
| F-31 | Image Segmentation | Segmentation | `f-35_t15.jpg` | none | `iou` @ 0.3 | 300 s |
| F-32 | Frame → Segmentation | ExtractFrames → Segmentation | `f-35.mp4` | none | `detect` @ 0.005 | 600 s |
| F-33 | Full Cross-Modal Chain | AudioExtraction → Transcription → InformationFinder → ExtractFrames → Vqa | `f-35.mp4` | keywords + min\_length=20 | none | 600 s |
| F-34 | Video Segmentation | Segmentation | `f-35_clip.mp4` | none | none | 600 s |
| F-35 | Error: VQA on Audio File | Reject\|RequestClarification | `f-35.wav` | none | none | 300 s |
| F-36 | Error: Segmentation, No Path | RequestClarification | *(none)* | none | none | 300 s |
| F-37 | Error: Search Word Not in Transcript | Reject\|RequestClarification\|InformationFinder | `f-35.mp4` | none | none | 600 s |

The `vision_criteria` keywords for F-29/F-30 are (OR-logic — any one is sufficient):
```python
_AIRCRAFT_KEYWORDS = ("f-35", "f35", "fighter jet", "stealth fighter", "aircraft",
                      "jet", "fighter", "military aircraft", "warplane", "plane")
```

F-33 uses a different keyword set targeting the pilot visible at the matched timestamp
(the word `"helmet"` spoken in `f-35.mp4`):
```python
_PILOT_KEYWORDS = ("pilot", "person", "man", "helmet", "flight suit",
                   "g-suit", "uniform", "people", "aviator")
min_answer_length=20
```

**Scenario groups:**

- **F-28/29** — single-agent baselines. F-28 tests frame extraction alone (does the tool call happen and return files?). F-29 tests VQA on a pre-extracted reference frame — the simplest possible vision test, GT-scored against `videos/f-35/vqa.json`.
- **F-30/32** — two-step chains. These are the core multi-step chaining tests: extract a frame from a raw video, then pass that frame to VQA or segmentation. These were the primary failure mode before the `first_frame_path` fix (see §6f).
- **F-31/34** — single segmentation. F-31 segments a pre-extracted frame (IoU vs GT masks); F-34 segments the ~15-second clip.
- **F-33** — full cross-modal. The hardest scenario: extract audio, transcribe it, find where the word "helmet" is spoken, jump to that timestamp, extract a frame, and answer a visual question about it. Five agents in sequence on `f-35.mp4`.
- **F-35/36/37** — error handling. F-35 asks for VQA on a `.wav` audio file — an impossible request that the orchestrator should reject. F-36 asks for segmentation without providing a file path, triggering a clarification request. F-37 runs the full audio→transcript→search pipeline but searches for a word guaranteed not to appear. The orchestrator must stop at InformationFinder and report "0 matches found" without fabricating a timestamp.

Five extra f-35 scenarios (F-38..F-42: true-negatives, counting, attribute, multi-instance) exist commented-out in `vision_scenarios.py` — disabled to keep the active set at the original 10.

---

## 4. How `handoff_match` Works

`handoff_match` is a property on `TestResult` (in `tests/scenarios/base.py`) that returns `True` when the test's behavioral contract on agent routing is satisfied.

### What it checks

`expected_agents` must be an **ordered subsequence** of `handoff_sequence`. This means:

- Every expected agent must appear in the actual sequence.
- They must appear in the same relative order.
- Extra or repeated calls between expected agents are allowed.

### Why subsequence, not exact match

LLMs under load sometimes retry a slow tool call, producing a handoff sequence like:
```
["Extract Frames", "Extract Frames", "Segmentation"]
```
instead of the ideal:
```
["Extract Frames", "Segmentation"]
```

Both sequences satisfy the contract `["ExtractFrames", "Segmentation"]` under subsequence matching. Exact-length matching would incorrectly fail the first sequence.

### Pipe-separated alternatives

Each entry in `expected_agents` may contain a `|`-separated list of acceptable agent names:
```python
expected_agents=["Reject|RequestClarification"]
```
This matches either `Reject` or `RequestClarification`. Both are valid refusals for an impossible request (VQA on audio), and the model's choice depends on how it interprets the prompt's tone. Tying the test to a specific phrasing would make it brittle.

### Name normalization

Agent names are normalized before comparison:
```
"Audio Extraction"  →  "audioextraction"
"audio_extraction"  →  "audioextraction"
"AudioExtraction"   →  "audioextraction"
```
Normalization: lowercase, remove spaces, remove underscores. This means scenario definitions and tool names do not need to match character-for-character.

### Algorithm (pseudocode)

```
function handoff_match(expected_agents, handoff_sequence):
    si = 0                              # pointer into handoff_sequence
    for each exp in expected_agents:
        alternatives = exp.split("|")   # pipe-separated options
        matched = False
        while si < len(handoff_sequence):
            if norm(handoff_sequence[si]) in {norm(a) for a in alternatives}:
                si += 1
                matched = True
                break
            si += 1
        if not matched:
            return False
    return True
```

---

## 5. Vision Success Criteria

`VisionSuccessCriteria` (in `tests/scenarios/base.py`) defines the behavioral contract for the *content* of the orchestrator's final text output. It is only set on scenarios where VQA runs and produces a text answer.

### Fields

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `vqa_keywords` | `tuple[str, ...]` \| `None` | `None` | OR-logic — at least one must appear (case-insensitive) |
| `required_keywords` | `tuple[str, ...]` \| `None` | `None` | AND-logic — all must appear (case-insensitive) |
| `min_answer_length` | `int` | `5` | Minimum character length for a valid answer |

Both keyword fields are optional. You can set either, both, or neither.

### When to use each keyword field

**`vqa_keywords` (OR-logic)** — use when any synonym for the subject is acceptable. Phi-4 may describe the F-35 as "a fighter jet", "a stealth fighter", or "a military aircraft". Any of these in the output is equally correct. This is the right choice when the vocabulary varies but the meaning is equivalent.

**`required_keywords` (AND-logic)** — use when the answer must explicitly identify multiple distinct subjects. For a frame showing both a pilot and a jet, `required_keywords=("pilot", "jet")` ensures the VLM described *both*, not just one.

You may combine both fields: `required_keywords` for mandatory identifications that must all be present, plus `vqa_keywords` for acceptable synonyms. The criteria pass only when both constraints are satisfied.

### How `answer_passes` evaluates the output

```python
def answer_passes(self, text: str) -> bool:
    if not text or len(text.strip()) < self.min_answer_length:
        return False
    lower = text.lower()
    if self.required_keywords:
        if not all(kw.lower() in lower for kw in self.required_keywords):
            return False          # AND-logic: all required keywords must be present
    if self.vqa_keywords:
        return any(kw.lower() in lower for kw in self.vqa_keywords)  # OR-logic
    return True                   # no keyword constraints — length check was enough
```

### How overall_pass is computed

```python
@property
def overall_pass(self) -> bool:
    if not self.handoff_match:
        return False          # wrong tool chain — always fail
    if self.vision_pass is not None:
        return self.vision_pass   # vision_pass evaluated → must be True
    return True               # no vision contract — handoff_match is enough
```

A test can fail `handoff_match` even if the VQA output is correct, because calling the wrong agents is a routing bug regardless of the output.

---

## 6. Infrastructure Decisions

This section documents the non-obvious engineering decisions made during development. Each follows the pattern: **Problem → Decision → Why**.

### 6a. LLM Reasoning Budget: Disabled

**Problem.** Gemma-4-E2B-it-Q8_0.gguf is a "thinking" model. By default, it generates an internal reasoning trace (640+ tokens, ~183 seconds of wall-clock time) *before* producing each tool call. A scenario with two tool calls therefore spent 6+ minutes in thinking before doing any actual work. With the default 300 s timeout, most tests timed out before the first agent was ever invoked.

**Decision.** Thinking is disabled at the `llama-server` level. Historically this used `--reasoning-budget 0`; the current launcher is `local-ai-service/run_agentic_system.sh`, and thinking-mode models (e.g. Qwen3.6) are disabled via `enable_thinking=false` in their chat-template kwargs. With thinking off, each tool call takes approximately 8 seconds.

**Why safe for this use case.** The orchestrator's task is tool routing — mapping a natural-language request to the correct sequence of specialized agents. This is a pattern-matching problem that does not require deep logical reasoning. Thinking mode adds latency without improving routing accuracy in practice.

**To re-enable.** Pass `--reasoning-budget -1` (or a token budget) via `LLAMA_EXTRA_ARGS` to `run_agentic_system.sh`, or remove the model's `enable_thinking=false` kwarg.

### 6b. Subsequence Handoff Matching

**Problem.** During initial vision test runs, the orchestrator occasionally called `extract_frames_handoff` twice before proceeding to `vqa_handoff` (a retry triggered by a slow first response from the frames MCP server). Exact-length handoff matching (`len(actual) == len(expected)`) failed these tests even though the correct pipeline eventually ran.

**Decision.** `handoff_match` was changed from exact-length equality to an ordered-subsequence check (see §4 for the full algorithm). The check answers "did the model use these agents in this order?" rather than "did the model use exactly these agents and no others?".

### 6c. Pre-flight Health Checks

**Problem.** The `vlm-segmentation-mcp-server` loads GroundingDINO and SAM2 at startup, which takes approximately 60 seconds. When the test runner started immediately after `docker compose up`, the first vision scenarios failed with HTTP connection errors before any agent logic ran. These were not test failures — they were infrastructure startup failures.

**Decision.** `runner.py` calls `_wait_for_healthy()` before the first scenario timer starts. This function polls the `/health` endpoint of each required MCP server (frames server, segmentation server) every 5 seconds until it receives `{"status": "ok"}` or until `VISION_PREFLIGHT_TIMEOUT_S` (default 150 s) elapses. Pre-flight only runs when at least one active scenario has `vision_criteria` set — audio-only runs skip it entirely.

### 6d. VQA Model Pre-warming

**Problem.** The first call to `_get_vlm()` triggers model loading from the HuggingFace local cache and GPU initialization. On a warm cache (model already downloaded) Phi-4 takes ~2–5 minutes to load. On a completely cold cache (first run ever, ~8 GB to download) it can take ~15 minutes. The scenario timers started after pre-flight but before the VQA model was loaded, causing the first VQA scenario to always time out on a cold environment.

**Decision.** `_prewarm_vqa()` is called once in `run_orchestrator_tests()`, after pre-flight checks and before any scenario timer starts. It runs `_get_vlm()` in a thread pool executor with a generous timeout (`VQA_PREWARM_TIMEOUT_S`, default 7200 s). On failure or timeout, it logs a warning and continues — the VQA model will attempt on-demand loading, which may fail individual scenarios but will not block the run.

### 6e. Incremental Result Saving

**Problem.** A complete test run across 22 scenarios × 3 prompt levels × 1 model takes 1–3 hours (model-dependent). A container restart, OOM kill, or any other interruption mid-run would lose all results.

**Decision.** `_save_results()` is called inside the innermost loop, after every single test completes. Results are serialized to a JSON file whose name includes the run timestamp and model slug. On each call the file is overwritten with the full accumulated list for that model, so the file always contains a consistent checkpoint.

### 6f. `first_frame_path` Injection

**Problem.** After `extract_frames_handoff` returned, the orchestrator LLM received tool output containing frame timestamps (`"Timestamp: 1.5, Timestamp: 1.5333..."`). The LLM then tried to *reconstruct* the frame file path from the hash-named output folder and the timestamp. This introduced transcription errors: for example, the real folder was `lionandgorilla_95538ea9` but the LLM wrote `lionandgorilla_955338ea9` (one extra digit). Subsequent `vqa_handoff` and `segmentation_handoff` calls received a non-existent path and returned "frame not found" errors.

**Decision.** Two complementary fixes:

1. `frame_extraction_agent.py` (`extract_frames_tool`) injects `first_frame_path` — the absolute path to the first extracted frame — directly into the tool result dict before returning it. The orchestrator's system prompt explicitly instructs the model to pass `first_frame_path` verbatim to the next agent.

2. `vqa_agent.py` (`answer_visual_question_tool`) reads `/tmp/last_extracted_frame.txt` as a fallback when the given path does not exist. The frame extraction tool writes this file on every successful extraction. This handles cases where the LLM still mangles the path despite the explicit instruction.

---

## 7. Test Matrix & Current Results

### Matrix size

```
22 scenarios  ×  3 prompt levels  =  66 runs per model
(9 audio/text + 10 vision + 3 extension)   (vague / clear / algorithmic)
```

### Full 5-model run `2026-07-01T21-35-33` (per-scenario, `overall_pass`)

Each cell shows vague/clear/algorithmic. Columns: **E2B** = gemma-4-E2B, **12b** =
gemma-4-12b, **q27** = Qwen3.6-27B, **qA3B** = Qwen3.6-35B-A3B, **oss** = gpt-oss-120b.

| ID | E2B | 12b | q27 | qA3B | oss |
|----|-----|-----|-----|------|-----|
| SC-01 | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| SC-02 | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| SC-03 | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| SC-04 | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| SC-05 | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| SC-06 | ✗✓✓ | ✗✓✓ | ✗✓✓ | ✗✓✓ | ✗✓✓ |
| SC-07 | ✓✓✓ | ✓✓✓ | ✓✓✗ | ✓✓✓ | ✓✓✓ |
| SC-08 | ✓✓✗ | ✓✓✓ | ✓✗✓ | ✓✓✓ | ✓✓✗ |
| SC-09 | ✗✗✓ | ✓✗✓ | ✗✗✓ | ✓✗✓ | ✗✗✓ |
| F-28 | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| F-29 | ✓✗✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| F-30 | ✗✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| F-31 | ✓✓✓ | ✗✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| F-32 | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| F-33 | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✗✗✗ |
| F-34 | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| F-35 | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| F-36 | ✓✓✓ | ✓✓✓ | ✓✗✗ | ✗✓✓ | ✓✓✓ |
| F-37 | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| SC-40 | ✓✓✓ | ✓✗✓ | ✗✓✓ | ✓✓✓ | ✓✓✓ |
| SC-42 | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| SC-47 | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✗✓✓ |

**Totals:** Qwen3.6-35B-A3B **63/66 (95.5%)** · gemma-4-12b 62/66 (93.9%) ·
gemma-4-E2B 60/66 (90.9%) · Qwen3.6-27B 58/66 (87.9%) · gpt-oss-120b 58/66 (87.9%).
The full 14-metric comparison (latency, TTFT, throughput, energy, passes/kWh, …) is in
the top-level README and [`docs/MODEL_COMPARISON.md`](../docs/MODEL_COMPARISON.md).

### What still fails and why

**SC-06/vague — fails all five models.** "Do a full analysis of this video" is too
under-specified to reliably elicit the full 6-step chain; every model stops early or
picks a shorter chain. Clear and algorithmic prompts pass for all models — the failure
is prompt ambiguity, not capability.

**SC-09 vague/clear — fails most models.** The null-results scenario (search for a word
absent from the transcript) tempts models into re-searching or answering from priors
instead of reporting "0 matches". The algorithmic prompt passes everywhere.

**F-33 — fails only gpt-oss-120b (all three levels).** The 5-step cross-modal chain
breaks down for gpt-oss at the transcript→timestamp handoff (path errors mid-chain);
all other models pass it at every level.

**Error-handling spread (F-36, SC-07/08).** Qwen3.6-27B is the weakest at
reject/clarify behavior (11/15 across the 5 error scenarios × 3 levels);
gemma-12b and gpt-oss are perfect (15/15).

---

## 8. Running Tests

### Prerequisites

1. **LLM server must be running with `--reasoning-budget 0`:**
   ```bash
   # Starts llama-server + the full stack; blocks until ALL SYSTEMS READY.
   ./local-ai-service/run_agentic_system.sh
   ```
   Without this, Gemma-4 spends ~183 s per tool call on internal reasoning and most tests time out.

2. **All commands run inside the container:**
   ```bash
   docker exec -it agents-orchestrator bash
   ```

3. **MCP servers must be running** (for vision tests, the segmentation server takes ~60 s to start):
   ```bash
   docker compose -f docker-compose.yml up -d
   ```

### Orchestrator integration tests

```bash
# All 22 scenarios, all prompt levels
python tests/orchestrator/run_tests.py

# Specific scenarios
python tests/orchestrator/run_tests.py --scenarios F-30,F-32,F-35

# One prompt level only
python tests/orchestrator/run_tests.py --levels clear

# Skip VQA pre-warming (faster for audio-only runs)
VQA_PREWARM=0 python tests/orchestrator/run_tests.py --scenarios SC-01,SC-04,SC-06

# Watch live output
docker exec agents-orchestrator python tests/orchestrator/run_tests.py \
  --scenarios F-30,F-32 > /tmp/run.log 2>&1 &
docker exec agents-orchestrator tail -f /tmp/run.log
```

### Via pytest (supports `-k` filtering)

```bash
# All integration tests
pytest tests/orchestrator/ -v

# Filter by scenario ID or prompt level
pytest tests/orchestrator/ -v -k "F-30"
pytest tests/orchestrator/ -v -k "F-30 and clear"
pytest tests/orchestrator/ -v -k "not algorithmic"
```

### Unit tests (require MCP servers + GPU)

```bash
pytest tests/unit/ -v
pytest tests/unit/test_vqa.py -v
pytest tests/unit/test_segmentation.py -v
pytest tests/unit/test_frame_extraction.py -v
pytest tests/unit/test_video_segmentation.py -v
```

---

## 9. Configuration

All settings are environment variables. No code changes are needed to change the model, data directory, or which scenarios run.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `/data` | Root of the shared data volume |
| `TEST_FIXTURE_DIR` | `$DATA_DIR/fixtures/raw` | Where raw test video files live |
| `TEST_RESULTS_DIR` | `tests/results/` | Where JSON result files are written |
| `TEST_MODELS_JSON` | *(auto-detect)* | JSON array of ModelConfig dicts; overrides auto-detection |
| `TEST_SCENARIO_IDS` | *(all)* | Comma-separated IDs, e.g. `SC-01,SC-02` |
| `TEST_PROMPT_LEVELS` | `vague,clear,algorithmic` | Prompt levels to run |
| `TEST_TIMEOUT_S` | `300` | Default per-scenario timeout in seconds |
| `LLM_ORCHESTRATOR_MODEL` | *(from .env)* | Model name for auto-detected config |
| `LLM_ORCHESTRATOR_BASE_URL` | *(from .env)* | Local model server URL |
| `LLM_ORCHESTRATOR_API_KEY` | `not-needed` | API key (ignored for local models) |
| `VQA_PREWARM` | `1` | Set to `0` to skip VQA model pre-warming |
| `VQA_PREWARM_TIMEOUT_S` | `7200` | Max seconds for VQA pre-warm (2 h for Phi-4 on cold disk) |
| `VISION_PREFLIGHT_TIMEOUT_S` | `150` | Max seconds to wait for MCP servers to become healthy |
| `FRAMES_MCP_SERVER_URL` | `http://ffmpeg-frames-mcp-server:8006/mcp` | Frame extraction server URL |
| `IMAGE_SEGMENTATION_MCP_SERVER_URL` | `http://vlm-segmentation-mcp-server:8008/mcp` | Segmentation server URL |

### Testing multiple models

Set `TEST_MODELS_JSON` to a JSON array:

```bash
export TEST_MODELS_JSON='[
  {
    "model_name": "gemma-4-E2B-it-Q8_0.gguf",
    "provider": "openai-chat",
    "base_url": "http://host.docker.internal:9000/v1",
    "api_key": "not-needed",
    "display_name": "Gemma-4 8-bit (local)"
  }
]'
python tests/orchestrator/run_tests.py
```

Results land in separate JSON files: `{run_id}_{model_slug}.json`.

---

## 10. Adding New Scenarios

### Audio / text scenario

1. Open `tests/scenarios/orchestrator_scenarios.py`.
2. Add an `OrchestratorScenario(...)` entry to `ORCHESTRATOR_SCENARIOS`.
3. Done — both the standalone runner and pytest pick it up automatically.

Minimal example:
```python
OrchestratorScenario(
    scenario_id="SC-10",
    name="Custom: Audio + Transcription",
    difficulty=3,
    prompt_templates={
        "vague": "Transcribe the speech in {input_path}.",
        "clear": "Extract audio from {input_path} and transcribe it.",
        "algorithmic": "audio_extraction_handoff({input_path}) → transcription_handoff",
    },
    expected_agents=["AudioExtraction", "Transcription"],
    default_input_path="fixtures/raw/sample_video.mp4",
),
```

### Vision scenario

For vision scenarios, also add to `tests/scenarios/vision_scenarios.py` and consider:

- **`timeout_s`**: VQA scenarios need at least 900 s (model inference is slow). Segmentation chains need at least 600 s.
- **`vision_criteria`**: set when the scenario includes VQA and you want to assert something about the answer's content. Leave `None` for segmentation-only scenarios where the output is a file path, not text.

```python
OrchestratorScenario(
    scenario_id="SC-XX",
    name="Frame → VQA on Custom Video",
    difficulty=3,
    prompt_templates={
        "vague": "What do you see in a frame from {input_path}?",
        "clear": "Extract a frame from {input_path} and describe what is visible.",
        "algorithmic": "extract_frames_handoff({input_path}) → vqa_handoff",
    },
    expected_agents=["ExtractFrames", "Vqa"],
    default_input_path="fixtures/raw/my_video.mp4",
    timeout_s=900,
    vision_criteria=VisionSuccessCriteria(
        vqa_keywords=("dog", "cat", "animal", "mammal"),
        min_answer_length=10,
    ),
),
```

---

## 11. Adding VQA Scenarios

This section covers VQA-specific concerns. For the general `OrchestratorScenario` fields, see §10.

### Where to add

Open `tests/scenarios/vision_scenarios.py` and append to the `VISION_SCENARIOS` list.
The runner and pytest discover it automatically — no other files need editing.

### Choosing `vision_criteria`

`VisionSuccessCriteria` has two keyword fields with different semantics:

| Field | Logic | When to use |
|-------|-------|-------------|
| `vqa_keywords` | OR — at least one must appear | Any synonym for the subject is acceptable. A dog may be called "dog", "puppy", "canine", or "animal" — any one is correct. |
| `required_keywords` | AND — all must appear | The answer must explicitly identify multiple distinct subjects. A pilot-and-jet frame: both must be named. |

You may combine both fields — `required_keywords` for mandatory identifications plus `vqa_keywords` for synonyms. Leave `vision_criteria=None` for segmentation-only scenarios whose output is a file path, not text.

Set `min_answer_length` higher when the output should be a full description rather than a single word. Use `min_answer_length=20` for 5-step cross-modal chains (F-33 style) where the final answer is expected to be a complete sentence.

### Choosing `timeout_s`

| Chain type | Recommended timeout |
|---|---|
| Single VQA on pre-extracted image | 900 s |
| ExtractFrames → VQA | 900 s |
| Full cross-modal (5 steps, F-33 style) | 600 s |
| Segmentation-only | 600 s |

The bottleneck for VQA scenarios is Phi-4 inference inside the agents container (~8 s per sub-agent call on warm GPU).

### Choosing `default_input_path`

| Fixture | Use for |
|---|---|
| `fixtures/raw/f-35_t15.jpg` | Single-image VQA/segmentation — avoids dependency on frame extraction server, aligns with GT |
| `fixtures/raw/f-35.mp4` | Frame extraction → VQA/segmentation chains, and speech-based cross-modal chains (audio → transcript → keyword → frame → VQA) |
| `fixtures/raw/f-35_clip.mp4` | Video segmentation (shorter clip keeps GPU time manageable) |
| `fixtures/raw/f-35.wav` | Error scenarios: impossible VQA on audio |
| `fixtures/raw/sample_video.mp4` | Audio/text pipeline scenarios |

Place any new fixture in `data/fixtures/raw/` (mounted read-only at `/data/fixtures/raw/`) before running the test.

### Validating keywords before committing

Run VQA manually on the fixture to see what Phi-4 actually produces, then pick keywords from that output:

```bash
docker exec agents-orchestrator python -c "
from videosearcher_agents.specialized.vqa_agent import VqaAgent
import asyncio
agent = VqaAgent()
result = asyncio.run(agent.agent.run(
    'Describe the image at /data/fixtures/raw/f-35_t15.jpg'
))
print(result.final_output)
"
```

If the output says "an F-35 fighter jet in flight against a blue sky", then
`vqa_keywords=("f-35", "fighter jet", "aircraft", "jet")` is safe. When the answer
must name multiple distinct subjects (e.g. a formation frame), use
`required_keywords` instead.

### Full annotated example

```python
from tests.scenarios.base import OrchestratorScenario, VisionSuccessCriteria

OrchestratorScenario(
    scenario_id="SC-XX",           # replace XX with the next available ID
    name="Frame → VQA on Custom Video",
    difficulty=3,
    prompt_templates={
        "vague": "What do you see in a frame from {input_path}?",
        "clear": (
            "Extract a frame from {input_path} around 2 seconds "
            "and describe what animals are visible."
        ),
        # Algorithmic: use natural language, not key=value syntax
        # (the orchestrator receives a single input string via as_tool())
        "algorithmic": (
            "Use extract_frames_handoff on {input_path} with t_start=1, t_end=3, fps=1. "
            "Then call vqa_handoff with question 'What animals are visible in this image?'"
        ),
    },
    # ExtractFrames must appear before Vqa in the handoff sequence (ordered subsequence)
    expected_agents=["ExtractFrames", "Vqa"],
    default_input_path="fixtures/raw/my_video.mp4",
    timeout_s=900,
    vision_criteria=VisionSuccessCriteria(
        # OR-logic: any one of these in the output is sufficient
        vqa_keywords=("dog", "puppy", "canine", "animal", "mammal"),
        # Minimum output length — rejects empty strings and single-word error messages
        min_answer_length=10,
    ),
)
```

To run just the new scenario across all prompt levels:
```bash
docker exec agents-orchestrator python tests/orchestrator/run_tests.py \
    --scenarios SC-XX --levels vague,clear,algorithmic
```

---

## 12. Result Format

Results are saved as `tests/results/<ISO_TIMESTAMP>_<model_slug>.json` — one file per run per model, checkpointed after every test.

Each file is a JSON array:

```json
[
  {
    "run_id": "2026-07-01T21-35-33",
    "scenario_id": "F-30",
    "scenario_name": "2-Step Pipeline: Frame Extraction → VQA",
    "prompt_level": "clear",
    "model_name": "Qwen3.6-35B-A3B-Q4_K_M.gguf",
    "success": true,
    "handoff_sequence": ["Extract Frames", "Vqa"],
    "expected_agents": ["ExtractFrames", "Vqa"],
    "handoff_match": true,
    "execution_time_ms": 94677.3,
    "orchestrator_output": "The frame shows an F-35 fighter jet in flight...",
    "vision_pass": true,
    "error": null,
    "conversation": [
      {
        "sequence": 0,
        "item_type": "tool_call",
        "tool_name": "extract_frames_handoff",
        "arguments": {"video_file": "/data/fixtures/raw/f-35.mp4", "t_start": 15.0},
        "content": null,
        "agent_role": null
      },
      {
        "sequence": 1,
        "item_type": "tool_output",
        "tool_name": null,
        "arguments": null,
        "content": "{\"success\": true, \"first_frame_path\": \"/data/runs/<session>/frames/f-35_95538ea9/frame_000001.jpg\", ...}",
        "agent_role": null
      },
      {
        "sequence": 2,
        "item_type": "tool_call",
        "tool_name": "vqa_handoff",
        "arguments": {"frame_path": "/data/runs/<session>/frames/f-35_95538ea9/frame_000001.jpg", "question": "What aircraft is visible?"},
        "content": null,
        "agent_role": null
      },
      {
        "sequence": 3,
        "item_type": "message",
        "tool_name": null,
        "arguments": null,
        "content": "The frame shows an F-35 fighter jet in flight.",
        "agent_role": "VideoSearchOrchestrator"
      }
    ]
  }
]
```

**Key fields:**

| Field | Type | Meaning |
|-------|------|---------|
| `success` | bool | Orchestrator completed without exception |
| `handoff_sequence` | list[str] | Agents actually invoked, in order |
| `expected_agents` | list[str] | Agents the scenario requires |
| `handoff_match` | bool | `expected_agents` is an ordered subsequence of `handoff_sequence` |
| `vision_pass` | bool \| null | Output satisfies `VisionSuccessCriteria`; null if not evaluated |
| `error` | str \| null | Exception message when `success=false` |
| `conversation` | list | Full SDK item trace (tool calls, outputs, messages, handoffs) |

A test is considered **passed** when `success=true AND handoff_match=true AND (vision_pass=true OR vision_pass=null)`.
