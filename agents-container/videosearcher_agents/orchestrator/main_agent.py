"""
Main Orchestrator Agent using OpenAI Agents SDK
Coordinates specialized agents to process video search requests using the Handoff pattern

This orchestrator implements the multi-agent handoff pattern from OpenAI Agents SDK:
https://openai.github.io/openai-agents-python/agents/#multi-agent-system-design-patterns
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")

import os
import uuid
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from agents import Agent, Runner, function_tool, ModelSettings
from videosearcher_agents.llm_utils import build_llm, resolve_llm_config
from videosearcher_agents.run_logger import log_run

# CodeCarbon energy tracking. One tracker wraps each orchestrator request:
#   - its GPU energy ≈ Layer 1 (orchestrator LLM, served by llama-server on the
#     same GPU; whole-GPU draw during the call window),
#   - its CPU + RAM energy ≈ Layer 3 (client-side orchestration overhead).
# Written to the shared emissions dir under project "orchestrator_request".
_EMISSIONS_DIR = os.getenv("EMISSIONS_DIR", "/app/emissions")
try:
    from codecarbon import EmissionsTracker as _EmissionsTracker
    _CODECARBON_OK = True
except Exception:  # pragma: no cover - codecarbon optional
    _CODECARBON_OK = False


def _start_energy_tracker(project_name: str):
    """Return a started EmissionsTracker, or None if unavailable. Never raises."""
    if not _CODECARBON_OK:
        return None
    try:
        os.makedirs(_EMISSIONS_DIR, exist_ok=True)
        t = _EmissionsTracker(
            project_name=project_name,
            output_dir=_EMISSIONS_DIR,
            log_level="error",
            save_to_file=True,
        )
        t.start()
        return t
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Energy tracker (%s) failed to start: %s", project_name, exc
        )
        return None
from videosearcher_agents.specialized.audio_extraction_agent import AudioExtractionAgent
from videosearcher_agents.specialized.audio_analysis_agent import AudioAnalysisAgent
from videosearcher_agents.specialized.video_splitting_agent import VideoSplittingAgent
from videosearcher_agents.specialized.compression_agent import CompressionAgent
from videosearcher_agents.specialized.transcription_agent import TranscriptionAgent
from videosearcher_agents.specialized.information_finder_agent import InformationFinderAgent

# For segmentation and VQA
from videosearcher_agents.specialized.frame_extraction_agent import FrameExtractionAgent
from videosearcher_agents.specialized.segmentation_agent import SegmentationAgent
from videosearcher_agents.specialized.vqa_agent import VqaAgent
# Extension tools (added on top of the original pipeline)
from videosearcher_agents.specialized.summarize_agent import SummarizationAgent
from videosearcher_agents.specialized.web_search_agent import WebSearchAgent
from videosearcher_agents.specialized.scene_detect_agent import SceneDetectAgent

logger = logging.getLogger(__name__)


@function_tool
def reject_handoff(reason: str) -> str:
    """
    Call this as the VERY FIRST action (before any other agent) when the user's
    initial request is impossible: unsupported file format (e.g. image given for
    audio extraction), or operation that no agent can perform.

    Do NOT call this mid-pipeline or after any other agent has already been called.
    If a pipeline step fails, report the error in text — do not call reject_handoff.

    Args:
        reason: Human-readable explanation of why the request is rejected.

    Returns:
        Rejection message to surface to the user.
    """
    return f"Request rejected: {reason}"


@function_tool
def request_clarification_handoff(question: str) -> str:
    """
    Call this ONLY when the user's goal is genuinely ambiguous — you cannot determine
    what operation they want to perform, even approximately.

    Do NOT call this when:
    - The user's goal is clear but input is missing or incomplete. In that case, proceed
      with the best available action; let the tool report its own error.
    - The user mentions a tool or operation by name (even approximately or incorrectly).
      In that case, identify the closest available tool that serves the described purpose
      and call it — the user has stated their intent, even if imprecisely.

    Ask for clarification only when intent is irrecoverably unclear (e.g. "do something
    useful" with no context). Never substitute clarification for a failed execution attempt.

    Args:
        question: The specific question to ask the user to unblock the request.

    Returns:
        Clarification request to surface to the user.
    """
    return f"Clarification needed: {question}"


def _make_vqa_handoff_tool(vqa_agent):
    """Create a vqa_handoff tool that accepts both `input` string and named frame_path/question params."""
    @function_tool
    async def vqa_handoff(
        input: str = "",
        frame_path: str = "",
        question: str = "",
    ) -> str:
        """Answer a visual question about a video frame using a local VLM.
        INPUT: either a single 'input' string "frame_path=<abs_path> question=<text>"
        OR separate frame_path and question parameters.
        OUTPUT: answer (text).
        """
        effective_input = input
        if not effective_input:
            parts = []
            if frame_path:
                parts.append(f"frame_path={frame_path}")
            if question:
                parts.append(f"question={question}")
            effective_input = " ".join(parts)
        result = await Runner.run(vqa_agent, effective_input, max_turns=15)
        return getattr(result, "final_output", str(result))
    return vqa_handoff


class OrchestratorAgent:
    """
    Main Orchestrator Agent coordinating specialized video processing agents.

    This orchestrator implements the Handoff design pattern:
    1. Receives high-level user requests (natural language)
    2. Reasons about which specialized agent(s) to invoke
    3. Delegates tasks to specialized agents via handoff tools
    4. Aggregates results and provides unified responses
    
    Multi-Agent Pattern:
        User → Orchestrator Agent → Handoff Tools → Specialized Agents → MCP Servers
        
    The orchestrator can optionally use a larger/different LLM model than specialized agents
    (e.g., local LLM via llama.cpp on port 9000, or GPT-4 for better reasoning).
    
    Supported Pipelines:
        - raw → audio_extraction → transcription → search
        - raw → audio_extraction → audio_analysis → video_clips
        - raw → audio_extraction → compression
        - And any other combination of stages
    """
    
    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        use_local_llm: bool = False
    ):
        """
        Initialize the Orchestrator Agent
        
        Args:
            llm_api_key: LLM API key for orchestrator (optional, LiteLLM auto-detects GEMINI_API_KEY/OPENAI_API_KEY)
            llm_model_name: LLM model for orchestrator (uses LLM_ORCHESTRATOR_MODEL or LLM_MODEL env var)
            llm_provider: LLM provider override (uses LLM_ORCHESTRATOR_PROVIDER env var if not set)
            llm_base_url: LLM base URL override (uses LLM_ORCHESTRATOR_BASE_URL env var if not set)
            use_local_llm: If True, use local llama.cpp server at http://localhost:9000/v1
                          (overrides other LLM settings)
        """
        shared_cfg = resolve_llm_config(
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
            llm_provider=llm_provider,
            llm_base_url=llm_base_url,
            use_local_llm=use_local_llm,
            local_base_url_env="LLM_ORCHESTRATOR_BASE_URL"
        )

        # Initialize specialized agents (they use the same overrides by default)
        self.audio_extraction_agent = AudioExtractionAgent(
            llm_api_key=shared_cfg.api_key,
            llm_model_name=shared_cfg.model,
            llm_provider=shared_cfg.provider,
            llm_base_url=shared_cfg.base_url
        )
        self.audio_analysis_agent = AudioAnalysisAgent(
            llm_api_key=shared_cfg.api_key,
            llm_model_name=shared_cfg.model,
            llm_provider=shared_cfg.provider,
            llm_base_url=shared_cfg.base_url
        )
        self.video_splitting_agent = VideoSplittingAgent(
            llm_api_key=shared_cfg.api_key,
            llm_model_name=shared_cfg.model,
            llm_provider=shared_cfg.provider,
            llm_base_url=shared_cfg.base_url
        )
        self.compression_agent = CompressionAgent(
            llm_api_key=shared_cfg.api_key,
            llm_model_name=shared_cfg.model,
            llm_provider=shared_cfg.provider,
            llm_base_url=shared_cfg.base_url
        )
        self.transcription_agent = TranscriptionAgent(
            llm_api_key=shared_cfg.api_key,
            llm_model_name=shared_cfg.model,
            llm_provider=shared_cfg.provider,
            llm_base_url=shared_cfg.base_url
        )
        self.information_finder_agent = InformationFinderAgent(
            llm_api_key=shared_cfg.api_key,
            llm_model_name=shared_cfg.model,
            llm_provider=shared_cfg.provider,
            llm_base_url=shared_cfg.base_url
        )

        # our agents for segmentation
        self.frame_extraction_agent = FrameExtractionAgent(
            llm_api_key=shared_cfg.api_key,
            llm_model_name=shared_cfg.model,
            llm_provider=shared_cfg.provider,
            llm_base_url=shared_cfg.base_url
        )
        self.segmentation_agent = SegmentationAgent(
            llm_api_key=shared_cfg.api_key,
            llm_model_name=shared_cfg.model,
            llm_provider=shared_cfg.provider,
            llm_base_url=shared_cfg.base_url
        )
        self.vqa_agent = VqaAgent(
            llm_api_key=shared_cfg.api_key,
            llm_model_name=shared_cfg.model,
            llm_provider=shared_cfg.provider,
            llm_base_url=shared_cfg.base_url
        )

        # Extension tool agents
        self.summarize_agent = SummarizationAgent(
            llm_api_key=shared_cfg.api_key,
            llm_model_name=shared_cfg.model,
            llm_provider=shared_cfg.provider,
            llm_base_url=shared_cfg.base_url
        )
        self.web_search_agent = WebSearchAgent(
            llm_api_key=shared_cfg.api_key,
            llm_model_name=shared_cfg.model,
            llm_provider=shared_cfg.provider,
            llm_base_url=shared_cfg.base_url
        )
        self.scene_detect_agent = SceneDetectAgent(
            llm_api_key=shared_cfg.api_key,
            llm_model_name=shared_cfg.model,
            llm_provider=shared_cfg.provider,
            llm_base_url=shared_cfg.base_url
        )

        logger.info("✅ Initialized all specialized agents")

        # Configure orchestrator LLM (can be different from specialized agents)
        orchestrator_cfg = resolve_llm_config(
            agent_type="orchestrator",
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
            llm_provider=llm_provider,
            llm_base_url=llm_base_url,
            use_local_llm=use_local_llm,
            local_base_url_env="LLM_ORCHESTRATOR_BASE_URL"
        )

        self.model = build_llm(orchestrator_cfg)

        # Some local GGUFs need a per-request `stop` to halt cleanly after a tool
        # call (e.g. Command-R in Hermes tool-format loops without "</tool_call>").
        # The model-comparison driver sets LLM_EXTRA_STOP (JSON list) per model.
        self._model_settings = None
        _extra_stop = os.getenv("LLM_EXTRA_STOP")
        if _extra_stop:
            try:
                stops = json.loads(_extra_stop)
                if isinstance(stops, str):
                    stops = [stops]
                self._model_settings = ModelSettings(extra_args={"stop": stops})
                logger.info(f"🛑 Orchestrator stop sequences: {stops}")
            except Exception as exc:
                logger.warning(f"Ignoring invalid LLM_EXTRA_STOP ({_extra_stop!r}): {exc}")

        logger.info(
            f"🤖 Orchestrator using LLM: {orchestrator_cfg.model} "
            f"(provider: {orchestrator_cfg.provider})"
        )
        if use_local_llm and orchestrator_cfg.base_url:
            logger.info(f"🖥️  Using local LLM server at {orchestrator_cfg.base_url}")
        
        # Initialize the main orchestrator agent with handoff tools
        self.agent = Agent(
            name="VideoSearchOrchestrator",
            instructions="""You are the VideoSearcher Orchestrator Agent.
Your role is to coordinate specialized video processing agents to fulfill user requests.

═══════════════════════════════════════════════════════
CRITICAL: PATH PASS-THROUGH RULE
═══════════════════════════════════════════════════════

Every agent output contains "DONE: output_path=<absolute_path>".
Extract <absolute_path> by reading AFTER "output_path=" — this is the ONLY valid path to pass forward.
Each pipeline step produces a NEW path with a NEW unique hash. NEVER derive or guess a path suffix from a previous step.

NEVER construct, abbreviate, or modify the path. Copy it verbatim from the tool result.

CRITICAL: "DONE: output_path=..." in a sub-agent's output means THAT ONE STEP completed.
It does NOT mean the overall task is finished. After extracting the path, IMMEDIATELY call the
next required step. Only stop when ALL pipeline steps have been executed.
If output_path is None, the step ran but produced no file — use the previous step's path to continue.

Example: Transcription returns "DONE: output_path=/data/transcription/transcription_7bd09c35.tar.gz"
→ Pass EXACTLY "/data/transcription/transcription_7bd09c35.tar.gz" to the next step.
   Do NOT use the hash from the audio_extraction archive ← WRONG.
   Do NOT guess "/data/transcription/transcription_abc123.tar.gz" ← WRONG.

═══════════════════════════════════════════════════════
PIPELINE EXECUTION RULES
═══════════════════════════════════════════════════════

1. EXECUTE: Call agents one by one WITHOUT writing planning text first. Call the first tool immediately.
2. SEQUENTIAL ONLY: Call EXACTLY ONE tool per turn. Never issue multiple tool calls at the same time.
   Wait for the result before calling the next tool.
3. After each tool returns, immediately call the next using the output_path from the result.
4. STOP: Only when ALL required steps are complete, or an agent returns an error.
5. NO RETRIES: If an agent fails, report the error and stop. Do NOT re-call the same agent.
6. NO LOOPS: Never call the same agent twice for the same input.
7. DEPENDENCY PRINCIPLE: each tool's input requirement determines its prerequisite.
   - transcription_handoff requires audio.wav → input must come from audio_extraction_handoff
     or compression_handoff. It CANNOT accept audio_analysis output (no audio) or raw clip folders.
   - video_splitting output (folder of clips) → run compression_handoff before transcription.
   - When unsure which tool to call next: ask "what data format does the next tool need?" and
     trace backward to find which prior tool produces that format.

═══════════════════════════════════════════════════════
PIPELINE CHAINS
═══════════════════════════════════════════════════════

A. Find sentence breaks / generate timestamps only (NO splitting needed):
   audio_extraction_handoff → audio_analysis_handoff
   STOP after audio_analysis_handoff. Pass audio_extraction output_path to audio_analysis.

B. Split video into clips at silence points:
   audio_extraction_handoff → audio_analysis_handoff → video_splitting_handoff
   Pass each output_path to the next step.

C. Search for keyword in compressed clips (input: folder of .mp4 clips or single .mp4):
   compression_handoff → transcription_handoff → information_finder_handoff(query=<keyword>)
   Pass each output_path to the next step.
   Example: compression returns output_path="/data/compression/abc.tar.gz"
            → pass "/data/compression/abc.tar.gz" to transcription_handoff
            → transcription returns output_path="/data/transcription/abc.tar.gz"
            → pass "/data/transcription/abc.tar.gz" + query to information_finder_handoff

D. Full pipeline: split + compress + search (input: raw .mp4 video):
   audio_extraction_handoff → audio_analysis_handoff → video_splitting_handoff
   → compression_handoff → transcription_handoff → information_finder_handoff(query=<keyword>)
   Pass each output_path to the next step.

E. Quick search without splitting (input: raw .mp4 video):
   audio_extraction_handoff → transcription_handoff → information_finder_handoff(query=<keyword>)

F. Frame visual question answering OR image segmentation on a video:
   extract_frames_handoff → vqa_handoff(frame_path=<first_frame_path>)
   extract_frames_handoff → segmentation_handoff(image_path=<first_frame_path>, output_mode="masks")   [use when user wants bounding boxes or labels]
   extract_frames_handoff → segmentation_handoff(image_path=<first_frame_path>, output_mode="segmented_image")   [use when user wants a visual annotated image]
   CRITICAL: You MUST call extract_frames_handoff FIRST if the input is a .mp4 video.
   Only call vqa_handoff or segmentation_handoff AFTER you have first_frame_path from extract_frames.
   CRITICAL: extract_frames_handoff returns first_frame_path (absolute path to the first frame).
   Pass first_frame_path verbatim as frame_path/image_path. Do NOT construct paths from timestamps.

G. Cross-modal: find what something looks like when a phrase is said:
   audio_extraction_handoff(video_path=<mp4>) → transcription_handoff(archive=<audio_archive>)
   → information_finder_handoff(archive=<transcript_archive>, query=<search_word>)
   → extract_frames_handoff(video_path=<mp4>, t_start=<timestamp_seconds>) → vqa_handoff(frame_path=<first_frame_path>)
   Use first_frame_path from extract_frames output as frame_path for vqa_handoff.
   CRITICAL: When calling information_finder_handoff you MUST include BOTH the transcript archive path
   AND the search word/phrase from the user request. Example: "search /data/transcription/X.tar.gz for 'radar'".
   CRITICAL: Do NOT call audio_analysis_handoff in this chain. Pass audio_extraction output DIRECTLY
   to transcription_handoff. audio_analysis is ONLY for Chain E (video splitting — never for search).
   If information_finder returns 0 matches, report "no match found" — do NOT fabricate a timestamp.

H. Image/frame segmentation for further analysis → VQA:
   segmentation_handoff(output_mode="masks") → vqa_handoff

═══════════════════════════════════════════════════════
AVAILABLE AGENTS
═══════════════════════════════════════════════════════

1. audio_extraction_handoff       — Input: .mp4 path → Output: archive (video.mp4 + audio.wav)
2. audio_analysis_handoff         — Input: archive path from audio_extraction → Output: archive (video.mp4 + timestamps.txt)
3. video_splitting_handoff        — Input: archive path from audio_analysis → Output: folder of .mp4 clips
4. compression_handoff            — Input: .mp4 path OR folder of clips → Output: archive or folder (.mp4 + .wav)
5. transcription_handoff          — Input: archive path (with audio) OR folder of archives → Output: archive (video.mp4 + transcript.txt)
6. information_finder_handoff     — Input: archive or folder path (with transcript) + search query → Output: search_results.json
7. extract_frames_handoff         — Input: video path + time window → Output: frame image files
8. segmentation_handoff           — Input: image/frame/video path + text query → Output: masks, segmented image, or video
9. vqa_handoff                    — Input: frame path (absolute) + question → Output: text answer
10. reject_handoff                — Call ONLY as the VERY FIRST action when the initial request is impossible (wrong file format). NEVER call mid-pipeline.
11. request_clarification_handoff — Call ONLY as the VERY FIRST action when NO file path is in the user message. NEVER call after any agent has already run.

═══════════════════════════════════════════════════════
EXTENSION TOOLS (additional capabilities)
═══════════════════════════════════════════════════════
Transcript-based (input = transcription archive):
- summarize_handoff       — summarize the transcript
Video/image-based:
- scene_detect_handoff    — shot/scene boundaries (visual cuts, not silence)
Web/knowledge:
- web_search_handoff      — search the web (query string)

EXTENSION CHAINS (pass each output_path forward verbatim, same rule as the core pipeline):
  I.  Summarize transcript:  audio_extraction_handoff → transcription_handoff → summarize_handoff
  II. Web fact-check:        transcription_handoff → web_search_handoff (query string)
  summarize_handoff comes AFTER transcription_handoff.

═══════════════════════════════════════════════════════
ROUTING HINTS
═══════════════════════════════════════════════════════

INPUT VALIDATION (before calling anything):
- User message has NO file path and task needs one → request_clarification_handoff
- File format is incompatible (e.g., .png for audio extraction) → reject_handoff with reason
- Always normalize: "data/raw/video.mp4" → "/data/raw/video.mp4"

CHAIN SELECTION KEYWORDS:
- "find timestamps", "sentence breaks", "detect pauses" → Chain A (2 steps, stop after audio_analysis)
- "split into clips", "clip at silence" → Chain B (3 steps)
- "find/search keyword in clips/folder", OR input path is a folder/directory → Chain C (3 steps:
  compression is entry point; do NOT prepend audio_extraction)
- "full pipeline", "run all steps", "split by sentence then search", "full analysis" → Chain D (6 steps).
  Execute ALL 6 steps without stopping after video_splitting.
- User wants to locate WHEN something is discussed in a raw video, no visual frame needed afterward → Chain D.
  Splitting produces accurate temporal segments; Chain D is correct when the goal is temporal location.
- "quick search", "just transcribe and find", OR user explicitly names 3 steps
  (audio_extraction → transcription → search) → Chain E (4 steps, no splitting)
- User wants to find a phrase AND then see a visual frame at that moment → Chain G
  (audio_extraction → transcription → information_finder → extract_frames → vqa; no splitting).
- "describe/answer question about frame/image" → Chain F or G
- "detect/find bounding boxes", "return labels", "where are", "locate objects", "spatial location", "segment and return boxes" → segmentation_handoff(output_mode="masks")
- "segment/highlight/mark/outline objects" (user wants a visual annotated image) → segmentation_handoff(output_mode="segmented_image")
- "segment image/frame" (for analysis, pipeline intermediate step) → segmentation_handoff(output_mode="masks")
- "segment video" → segmentation_handoff(output_mode="video")
- "segment all frames of video" → segmentation_handoff(output_mode="frames")

VQA: If user provides image path (.jpg/.jpeg/.png) + visual question → vqa_handoff DIRECTLY.
VQA on audio (.wav/.mp3/.ogg): IMPOSSIBLE — audio files have no visual content. Use reject_handoff immediately.

SEGMENTATION: Do NOT use output_mode="both" (not supported).

TIMESTAMP CROSS-MODAL: Parse [MM:SS.mmm] from information_finder context → convert to seconds → extract_frames_handoff.

═══════════════════════════════════════════════════════
CONSTRAINTS
═══════════════════════════════════════════════════════
- Always use the EXACT output_path returned by each agent as input for the next agent.
- Execute ALL steps of the chosen chain. Never stop mid-chain unless an agent errors.
- Never ask the user for clarification mid-pipeline.
- Include exact output paths in your final response to the user.
""",
            model=self.model,
            model_settings=self._model_settings or ModelSettings(),
            tools=[
                self.audio_extraction_agent.agent.as_tool(
                    tool_name="audio_extraction_handoff",
                    tool_description="""Extract audio from video files
                        INPUT: path to .mp4 file
                        OUTPUT: path to archive with .mp4 and .wav file
                    """
                ),
                self.audio_analysis_agent.agent.as_tool(
                    tool_name="audio_analysis_handoff",
                    tool_description="""Detect silence segments for video SPLITTING only (Chain B/C/D/E).
                        DO NOT use this for transcription or keyword search — its output has no audio and cannot be transcribed.
                        INPUT: path to archive from audio_extraction_handoff
                        OUTPUT: archive with timestamps.txt (silence segments for splitting)
                    """
                ),
                self.video_splitting_agent.agent.as_tool(
                    tool_name="video_splitting_handoff",
                    tool_description="""Split video into clips at silence boundary timestamps.
                        INPUT: archive from audio_analysis_handoff (must contain video.mp4 + timestamps.txt).
                        PREREQUISITE: audio_analysis_handoff must run first to produce timestamps.txt.
                        OUTPUT: folder of .mp4 clip files.
                        IMPORTANT: clips are raw .mp4 with NO audio archive — they CANNOT be passed
                        directly to transcription_handoff (which requires audio.wav).
                        Run compression_handoff next to produce transcription-ready archives.
                    """
                ),
                self.compression_agent.agent.as_tool(
                    tool_name="compression_handoff",
                    tool_description="""Compress video or audio files to reduce size.
                        INPUT: path to a single .mp4 file, or a folder of .mp4 clip files.
                        ENTRY POINT: use this as the FIRST step when the input is already a folder
                        of .mp4 clips (e.g. from video_splitting_handoff or a pre-existing clips dir).
                        Do NOT call audio_extraction_handoff before this when input is a clips folder —
                        the clips already contain video; compression extracts and packages audio for you.
                        OUTPUT: archive(s) with compressed .mp4 + .wav — ready for transcription_handoff.
                    """
                ),
                self.transcription_agent.agent.as_tool(
                    tool_name="transcription_handoff",
                    tool_description="""Transcribe audio files to text.
                        INPUT: path to .tar.gz archive from audio_extraction_handoff (contains video.mp4 + audio.wav), or folder of such archives.
                        OUTPUT: path to .tar.gz archive with transcript.txt, or folder of such archives.
                        NOTE: Never pass a raw .mp4 file or an audio_analysis archive — input must come from audio_extraction_handoff.
                    """
                ),
                self.information_finder_agent.agent.as_tool(
                    tool_name="information_finder_handoff",
                    tool_description="""Search a transcript archive for a specific word or phrase.
                        INPUT: MUST include BOTH (1) path to .tar.gz archive from transcription_handoff AND (2) the search word/phrase.
                        Example input: "Search /data/transcription/transcription_abc.tar.gz for 'radar'"
                        OUTPUT: search results with timestamps, match count, output archive path
                    """
                ),
                self.frame_extraction_agent.agent.as_tool(
                    tool_name="extract_frames_handoff",
                    tool_description="""Extract frames from a video in a time window.
                        INPUT: video_file path + t_start/t_end (seconds) + fps
                        OUTPUT: first_frame_path (absolute path to first extracted frame —
                                pass this directly to vqa_handoff or segmentation_handoff),
                                output_folder, n_frames
                    """
                ),
                self.segmentation_agent.agent.as_tool(
                    tool_name="segmentation_handoff",
                    tool_description="""Run text-guided segmentation on an image, frame, or video.
                        INPUT:
                        - image/frame segmentation: image_path or frame_path + text_query
                        - video segmentation: video_path + text_query

                        OUTPUT MODES:
                        - To detect objects and get bounding boxes + labels (coordinates): output_mode="masks".
                          Use this when user asks for boxes, labels, positions, or object locations.
                          Output: mask_paths + scores + labels + boxes.
                        - To produce a visual annotated image: output_mode="segmented_image".
                          Use this when user wants to see the image with annotations overlaid.
                          Output: segmented_image_path + scores + labels + boxes.
                        - For final video segmentation, ask for output_mode="video".
                        Output: segmented_video_path.
                        - For segmented video frames, ask for output_mode="frames".
                        Output: segmented_frames_dir + segmented_frames.

                        The video segmentation server does not support output_mode="both".
                    """
                ),
                _make_vqa_handoff_tool(self.vqa_agent.agent),
                self.summarize_agent.agent.as_tool(
                    tool_name="summarize_handoff",
                    tool_description="""Summarize the transcript inside an archive using an LLM.
                        INPUT: path to a transcription .tar.gz archive (contains transcript.txt).
                        OUTPUT: archive with summary.txt + the summary text.
                    """
                ),
                self.web_search_agent.agent.as_tool(
                    tool_name="web_search_handoff",
                    tool_description="""Search the web (DuckDuckGo by default; Tavily/Serper/Brave if configured).
                        INPUT: a search query string. OUTPUT: ranked results (title, url, snippet).
                        Use to fact-check or enrich something mentioned in the video.
                    """
                ),
                self.scene_detect_agent.agent.as_tool(
                    tool_name="scene_detect_handoff",
                    tool_description="""Detect shot/scene boundaries via visual cuts (PySceneDetect).
                        INPUT: a .mp4 video or an archive containing one. OUTPUT: scene boundary timestamps (split by scene, not silence).
                    """
                ),
                reject_handoff,
                request_clarification_handoff,
            ]
        )
        
        logger.info(f"🎭 Orchestrator Agent initialized: {self.agent.name}")
        self._init_session()

    def _init_session(self) -> str:
        """Create or adopt a session-scoped output root under /data/runs/{SESSION_ID}."""
        existing = os.environ.get("SESSION_OUTPUT_ROOT")
        if existing:
            session_root = existing
            session_id = Path(existing).name
        else:
            session_id = (
                datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                + "_"
                + uuid.uuid4().hex[:6]
            )
            session_root = f"/data/runs/{session_id}"
            os.environ["SESSION_OUTPUT_ROOT"] = session_root

        stages = [
            "audio_extraction", "audio_analysis", "video_clips", "compression",
            "transcription", "search_results", "frames", "segmentation",
            "scenes", "summarization", "web_search", "logs/orchestrator",
        ]
        for stage in stages:
            Path(f"{session_root}/{stage}").mkdir(parents=True, exist_ok=True)

        logger.info(f"📂 Session: {session_id}  root: {session_root}")
        self.session_id = session_id
        return session_id

    async def process_request(self, request: str) -> str:
        """
        Process user request by coordinating specialized agents
        
        Args:
            request: Natural language description of the task
            
        Returns:
            String with processing results and output paths
        """
        logger.info(f"📥 Processing request: {request}")
        
        result = await Runner.run(self.agent, request, max_turns=15)

        session_log_dir = f"/data/runs/{self.session_id}/logs/orchestrator"
        log_run(result, request, log_dir=session_log_dir)
        logger.info("✅ Request completed")
        return result.final_output

    async def process_request_with_items(self, request: str) -> dict:
        """
        Process user request and return full result with new_items for tracking

        Args:
            request: Natural language description of the task

        Returns:
            Dict with 'final_output' and 'new_items' for handoff/tool tracking
        """
        logger.info(f"📥 Processing request: {request}")

        # Layer 1 (orchestrator LLM, GPU) + Layer 3 (orchestration overhead, CPU/RAM)
        _tracker = _start_energy_tracker("orchestrator_request")
        try:
            result = await Runner.run(self.agent, request, max_turns=15)
        finally:
            if _tracker is not None:
                try:
                    _tracker.stop()
                except Exception:
                    pass

        session_log_dir = f"/data/runs/{self.session_id}/logs/orchestrator"
        log_run(result, request, log_dir=session_log_dir)
        logger.info("✅ Request completed")

        # Token usage across all LLM turns of this request (OpenAI Agents SDK).
        # Defensive: the SDK/usage shape can vary, so degrade to None on any miss.
        usage = None
        try:
            u = result.context_wrapper.usage
            usage = {
                "input_tokens": getattr(u, "input_tokens", None),
                "output_tokens": getattr(u, "output_tokens", None),
                "total_tokens": getattr(u, "total_tokens", None),
            }
        except Exception:
            usage = None

        return {
            'final_output': result.final_output,
            'new_items': result.new_items,
            'usage': usage,
        }
    
    def __repr__(self) -> str:
        return (
            f"OrchestratorAgent(\n"
            f"  model={self.model},\n"
            f"  specialized_agents=[\n"
            f"    {self.audio_extraction_agent.agent.name},\n"
            f"    {self.audio_analysis_agent.agent.name},\n"
            f"    {self.video_splitting_agent.agent.name},\n"
            f"    {self.compression_agent.agent.name},\n"
            f"    {self.transcription_agent.agent.name},\n"
            f"    {self.information_finder_agent.agent.name},\n"
            f"    {self.frame_extraction_agent.agent.name},\n"
            f"    {self.segmentation_agent.agent.name},\n"
            f"    {self.vqa_agent.agent.name}\n"
            f"  ]\n"
            f")"
        )

async def example_usage():
    """Example of using the OrchestratorAgent"""
    
    # Ensure API keys are configured
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: No API key found!")
        print("Set one of these environment variables:")
        print("  export GEMINI_API_KEY='your-gemini-key'  # Recommended")
        print("  export OPENAI_API_KEY='your-openai-key'")
        print("\nGet keys from:")
        print("  Gemini: https://makersuite.google.com/app/apikey")
        print("  OpenAI: https://platform.openai.com/api-keys")
        return
    
    # Initialize orchestrator
    # Option 1: Use default LLM config (from env vars)
    orchestrator = OrchestratorAgent()
    
    # Option 2: Use local LLM (requires local-ai-service running on port 9000)
    # orchestrator = OrchestratorAgent(use_local_llm=True)
    
    # Option 3: Use specific model for orchestrator
    # orchestrator = OrchestratorAgent(
    #     llm_model_name="gpt-4o"  # Larger model for better reasoning
    # )
    
    print(f"🎭 {orchestrator}\n")
    
    print("=" * 70)
    print("Example 1: Extract Audio from Video")
    print("=" * 70)
    
    result = await orchestrator.process_request(
        "Extract audio from the video at /data/raw/sample_video.mp4"
    )
    
    print(f"\n📤 Result:\n{result}\n")
    
    print("=" * 70)
    print("Example 2: Complete Pipeline - Extract, Transcribe, Search")
    print("=" * 70)
    
    result = await orchestrator.process_request(
        "I want to find all mentions of 'artificial intelligence' in /data/raw/sample_video.mp4. "
        "First extract the audio, then transcribe it, then search for the pattern."
    )
    
    print(f"\n📤 Result:\n{result}\n")
    
    print("=" * 70)
    print("Example 3: Video Segmentation Pipeline")
    print("=" * 70)
    
    result = await orchestrator.process_request(
        "Split /data/raw/sample_video.mp4 into clips at natural break points. "
        "Extract audio first, analyze for silence, then split the video."
    )
    
    print(f"\n📤 Result:\n{result}\n")

async def ask_orchestrator(question: str) -> str:
    """Helper function to ask the orchestrator a question"""
    orchestrator = OrchestratorAgent()
    response = await orchestrator.process_request(question)
    print(f"\n📤 Response:\n{response}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(ask_orchestrator("Find the sentence where the word \"uncertainty\" is mentioned from the video /data/raw/sample_video.mp4"))
