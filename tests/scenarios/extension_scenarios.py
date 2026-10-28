"""
Extension tool test scenarios (SC-40, SC-42, SC-47).

Scenarios for the 3 extension MCP tools kept on top of the original pipeline
(summarize, web_search, scene_detect). Mirrors the structure of
orchestrator_scenarios.py: each scenario has vague / clear / algorithmic prompt
levels and an expected_agents handoff sequence.

Expected agent names normalize (lower + strip spaces/underscores) to the names
produced by _extract_handoff_sequence() in runner.py, e.g.
  web_search_handoff -> "Web Search" -> "websearch"
"""

from .base import OrchestratorScenario, ResultCriteria

# Extension-tool agents
_WEB_SEARCH = "WebSearch"
_SUMMARIZE = "Summarize"
_SCENE_DETECT = "SceneDetect"

# Stable fixtures (created by tests/conftest fixture setup when missing).
# Use a TIMESTAMPED transcript ("[MM:SS.mmm] text") for per-segment timing;
# sample_transcript.tar.gz is older plain text.
_TRANSCRIPT_ARCHIVE = "fixtures/transcription/sample_timestamped_transcript.tar.gz"


EXTENSION_SCENARIOS: list[OrchestratorScenario] = [
    # ------------------------------------------------------------------ SC-40
    OrchestratorScenario(
        scenario_id="SC-40",
        name="Single Tool: Summarize",
        difficulty=3,
        prompt_templates={
            "vague": "Give me a short summary of this. File: {input_path}",
            "clear": "Summarize the transcript in the archive at {input_path}.",
            "algorithmic": "Run summarize_handoff with the input archive: {input_path}.",
        },
        expected_agents=[_SUMMARIZE],
        default_input_path=_TRANSCRIPT_ARCHIVE,
        result_criteria=ResultCriteria(
            metric="summary",
            ground_truth_ref="videos/sample_video/summary.json",
            threshold=0.6,  # ≥60% of 4 key points must appear in summary
        ),
    ),
    # ------------------------------------------------------------------ SC-42
    OrchestratorScenario(
        scenario_id="SC-42",
        name="Single Tool: Web Search",
        difficulty=2,
        prompt_templates={
            "vague": "Look up the specifications of the F-35 online.",
            "clear": "Use the web search agent to search the web for 'F-35 specifications'.",
            "algorithmic": "web_search_handoff. query='F-35 specifications'. max_results=5.",
        },
        expected_agents=[_WEB_SEARCH],
        default_input_path=None,
    ),
    # ------------------------------------------------------------------ SC-47
    OrchestratorScenario(
        scenario_id="SC-47",
        name="Single Tool: Scene Detection",
        difficulty=3,
        prompt_templates={
            "vague": "Split this video by scene, not just by silence. File: {input_path}",
            "clear": "Detect the scene boundaries in the video {input_path}.",
            "algorithmic": "scene_detect_handoff. video_path={input_path}. threshold=27.0.",
        },
        expected_agents=[_SCENE_DETECT],
        default_input_path="fixtures/raw/f-35.mp4",
        timeout_s=600,
        result_criteria=ResultCriteria(
            metric="scene",
            ground_truth_ref="videos/f-35/scene_detect.json",
            threshold=1.5,  # cut timestamps must be within ±1.5s of GT cuts
        ),
    ),
]

# Convenience lookup by scenario_id
EXTENSION_SCENARIOS_BY_ID = {s.scenario_id: s for s in EXTENSION_SCENARIOS}
