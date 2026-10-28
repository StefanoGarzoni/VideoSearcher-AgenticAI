"""
All orchestrator test scenarios as pure Python — no Excel dependency at runtime.

Source: migrated from orchestrator_test/Testing Scenarios.xlsx (Test Cases sheet).
These are the authoritative definitions. Edit this file to change scenario prompts
or add new scenarios. The Excel file is kept for historical reference only.

Input paths use DATA_DIR-relative paths (e.g. "fixtures/raw/sample_video.mp4").
Prompts use {input_path} where the resolved absolute path should be inserted.
"""

from .base import OrchestratorScenario, ResultCriteria

# ---------------------------------------------------------------------------
# Canonical agent names used in expected_agents lists.
# These are matched case-insensitively against extracted handoff sequences.
# ---------------------------------------------------------------------------
_AUDIO_EXTRACTION = "AudioExtraction"
_AUDIO_ANALYSIS = "AudioAnalysis"
_VIDEO_SPLITTING = "VideoSplitting"
_COMPRESSION = "Compression"
_TRANSCRIPTION = "Transcription"
_INFORMATION_FINDER = "InformationFinder"
_FRAME_EXTRACTION = "FrameExtraction"
_SEGMENTATION = "Segmentation"
_VQA = "VQA"

# Special expected values for error/edge-case scenarios.
_REJECT = "Reject"
_REQUEST_CLARIFICATION = "RequestClarification"


ORCHESTRATOR_SCENARIOS: list[OrchestratorScenario] = [
    # ------------------------------------------------------------------
    # SC-01: Single tool — audio extraction
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="SC-01",
        name="Single Tool: Audio Extraction",
        difficulty=1,
        prompt_templates={
            "vague": (
                "Extract the sound from this video. "
                "File: {input_path}"
            ),
            "clear": (
                "Use Audio Extraction Agent to extract audio.wav from the video file "
                "at {input_path}."
            ),
            "algorithmic": (
                "Run audio_extraction_handoff. Input: {input_path}."
            ),
        },
        expected_agents=[_AUDIO_EXTRACTION],
        default_input_path="fixtures/raw/sample_video.mp4",
    ),

    # ------------------------------------------------------------------
    # SC-02: Single tool — transcription
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="SC-02",
        name="Single Tool: Transcription",
        difficulty=2,
        prompt_templates={
            "vague": (
                "Turn this recording into text. "
                "File: {input_path}"
            ),
            "clear": (
                "Run the transcription agent on the archive at {input_path}."
            ),
            "algorithmic": (
                "Run the transcription agent with the provided input archive: {input_path}."
            ),
        },
        expected_agents=[_TRANSCRIPTION],
        default_input_path="fixtures/audio_extraction/sample_archive.tar.gz",
        result_criteria=ResultCriteria(
            metric="wer",
            ground_truth_ref="videos/sample_video/transcript.json",
            threshold=0.30,  # ≤30% WER vs large-v3 silver (lecture content)
        ),
    ),

    # ------------------------------------------------------------------
    # SC-03: Single tool — grep search
    # Search term updated from "Error" (not in transcript) to "elimination"
    # (verified present 3× in sample_transcript.tar.gz).
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="SC-03",
        name="Single Tool: Grep Search",
        difficulty=3,
        prompt_templates={
            "vague": (
                "Does this file mention 'elimination'? "
                "File: {input_path}"
            ),
            "clear": (
                "Search the transcript.txt in this archive for the word 'elimination'. "
                "Archive: {input_path}"
            ),
            "algorithmic": (
                "Call information finder agent with (search_pattern='elimination', "
                "archive_path={input_path})."
            ),
        },
        expected_agents=[_INFORMATION_FINDER],
        default_input_path="fixtures/transcription/sample_transcript.tar.gz",
        result_criteria=ResultCriteria(
            metric="grep",
            ground_truth_ref="videos/sample_video/grep.json",
            threshold=1.0,  # recall=1.0 (all 3 expected matches must be found)
            item_index=0,   # "elimination" item (total_count=3)
        ),
    ),

    # ------------------------------------------------------------------
    # SC-04: Early pipeline — audio extraction → silence detection
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="SC-04",
        name="Early Pipeline (Initial Processing)",
        difficulty=4,
        prompt_templates={
            "vague": (
                "Find the sentence breaks in this video. "
                "File: {input_path}"
            ),
            "clear": (
                "Extract the audio and then find timestamps from {input_path}."
            ),
            "algorithmic": (
                "1. Audio Extraction Agent on {input_path} "
                "-> 2. Audio Analysis Agent. Output timestamps.txt in archive."
            ),
        },
        expected_agents=[_AUDIO_EXTRACTION, _AUDIO_ANALYSIS],
        default_input_path="fixtures/raw/sample_video.mp4",
        result_criteria=ResultCriteria(
            metric="audio",
            ground_truth_ref="videos/sample_video/audio.json",
            threshold=3.0,  # |returned_segment_count - gt_count| ≤ 3
        ),
    ),

    # ------------------------------------------------------------------
    # SC-05: Late pipeline — compression → transcription → grep
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="SC-05",
        name="Late Pipeline (Text Analysis)",
        difficulty=5,
        prompt_templates={
            "vague": (
                "Find the word 'last week' in these video clips. "
                "Folder: {input_path}"
            ),
            "clear": (
                "Compress the clips in {input_path}, transcribe them, "
                "and search for 'last week'."
            ),
            "algorithmic": (
                "Use following pipeline: Compression -> Transcription -> "
                "Information Finder with (query='last week') for folder {input_path}."
            ),
        },
        expected_agents=[_COMPRESSION, _TRANSCRIPTION, _INFORMATION_FINDER],
        default_input_path="fixtures/video_clips/sample_clips",
    ),

    # ------------------------------------------------------------------
    # SC-06: Full pipeline
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="SC-06",
        name="Full Pipeline Execution",
        difficulty=6,
        prompt_templates={
            "vague": (
                "Tell me when they talk about 'Last week' in this video. "
                "File: {input_path}"
            ),
            "clear": (
                "Run the full analysis on {input_path}: extract audio, split by "
                "sentence, and search for 'last week'."
            ),
            "algorithmic": (
                "Use full pipeline on {input_path}: Audio Extraction, Audio Analysis, "
                "Video Splitting, Compression, Transcription, Information Finder "
                "with (query='last week')."
            ),
        },
        expected_agents=[
            _AUDIO_EXTRACTION,
            _AUDIO_ANALYSIS,
            _VIDEO_SPLITTING,
            _COMPRESSION,
            _TRANSCRIPTION,
            _INFORMATION_FINDER,
        ],
        default_input_path="fixtures/raw/sample_video.mp4",
    ),

    # ------------------------------------------------------------------
    # SC-07: Error handling — wrong file format
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="SC-07",
        name="Error Handling: Format",
        difficulty=7,
        prompt_templates={
            "vague": (
                "Transcribe this photo. File: {input_path}"
            ),
            "clear": (
                "Process this .png file at {input_path} through the audio extraction agent."
            ),
            "algorithmic": (
                "Attempt Audio Extraction on {input_path}. Expected: InputValidationError."
            ),
        },
        expected_agents=[_REJECT],
        default_input_path="fixtures/raw/lionandgorilla_frame0.jpg",
    ),

    # ------------------------------------------------------------------
    # SC-08: Error handling — missing input
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="SC-08",
        name="Error Handling: Missing Data",
        difficulty=8,
        prompt_templates={
            "vague": "Search for keywords. (No file provided)",
            "clear": "Perform a pattern matching on the video I haven't uploaded yet.",
            "algorithmic": (
                "Execute Information Finder with (query='x') where Input=None. "
                "Expected: NoInputError."
            ),
        },
        expected_agents=[_REQUEST_CLARIFICATION],
        expected_agents_by_level={"algorithmic": [_INFORMATION_FINDER]},
        default_input_path=None,
    ),

    # ------------------------------------------------------------------
    # SC-09: Null results — keyword not present in video
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="SC-09",
        name="Null Results Handling",
        difficulty=10,
        prompt_templates={
            "vague": (
                "Find where they discuss 'Aliens' in this video. "
                "File: {input_path}"
            ),
            "clear": (
                "Run the search for keyword 'Aliens' even though it's unlikely "
                "to match. File: {input_path}"
            ),
            "algorithmic": (
                "Full Pipeline on {input_path}. Target='Aliens'. "
                "Expected: {{matches: []}}."
            ),
        },
        expected_agents=[
            _AUDIO_EXTRACTION,
            _AUDIO_ANALYSIS,
            _VIDEO_SPLITTING,
            _COMPRESSION,
            _TRANSCRIPTION,
            _INFORMATION_FINDER,
        ],
        default_input_path="fixtures/raw/sample_video.mp4",
    ),
]

# Convenience lookup by scenario_id
SCENARIOS_BY_ID: dict[str, OrchestratorScenario] = {
    s.scenario_id: s for s in ORCHESTRATOR_SCENARIOS
}
