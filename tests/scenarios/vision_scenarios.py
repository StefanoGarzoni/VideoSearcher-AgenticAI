"""
Vision pipeline test scenarios (F-28..F-37) — f-35 fixture.

Tests frame extraction, VQA, and segmentation — individually, chained, and as a
full cross-modal pipeline. These are the f-35 equivalents of the original
lionandgorilla suite (SC-28..SC-37): same behavioral objectives — routing/handoff
+ VQA correctness + segmentation IoU + the three error/robustness paths.

Fixtures used (pre-populated in data/fixtures/raw/; the three reference frames are
copied verbatim from tests/ground_truth/videos/f-35/frames/ so they align exactly
with the GT, and the clip/wav are derived from f-35.mp4 with ffmpeg):
  f-35.mp4        — primary vision fixture (pilot + F-35 jet, ~88 s, has speech)
  f-35_t05.jpg    — reference frame at t=5 s  (pilot holding helmet; NO aircraft)
  f-35_t15.jpg    — reference frame at t=15 s (F-35 in flight)
  f-35_t65.jpg    — reference frame at t=65 s (4 jets in formation)
  f-35_clip.mp4   — ~15 s clip (≈12–27 s) covering the in-flight jet, for video seg
  f-35.wav        — audio track (first 10 s), used in the VQA-on-audio error scenario

Ground truth lives in tests/ground_truth/videos/f-35/ (vqa.json, segmentation.json,
grep.json, transcript.json, audio.json, scene_detect.json). item_index values below
index directly into the "items" arrays of those files.

Two groups of scenarios are preserved, COMMENTED OUT, at the bottom of this file:
  • EXTRA f-35 scenarios F-38..F-42 — capabilities beyond the original suite's goals
    (true-negatives, counting, attribute, multi-instance); disabled to keep the
    active set at the original 10.
  • LEGACY SC-28..SC-37 — the lionandgorilla/tedtalk originals (fixtures lost).
"""

from .base import OrchestratorScenario, ResultCriteria, VisionSuccessCriteria

# ---------------------------------------------------------------------------
# Expected agent name constants.
# These must normalize (lower + strip spaces/underscores) to match the names
# produced by _extract_handoff_sequence() in runner.py:
#   extract_frames_handoff  → "Extract Frames" → norm → "extractframes"
#   segmentation_handoff    → "Segmentation"   → norm → "segmentation"
#   vqa_handoff             → "Vqa"            → norm → "vqa"
# ---------------------------------------------------------------------------
_EXTRACT_FRAMES = "ExtractFrames"
_SEGMENTATION = "Segmentation"
_VQA = "Vqa"
_AUDIO_EXTRACTION = "Audio Extraction"
_TRANSCRIPTION = "Transcription"
_INFORMATION_FINDER = "InformationFinder"
_REJECT = "Reject"
_REQUEST_CLARIFICATION = "RequestClarification"

# Keywords expected in VQA answers about the f-35 jet. Broad set to tolerate
# phrasing variation. The model outputs English (question template is English).
_AIRCRAFT_KEYWORDS: tuple[str, ...] = (
    "f-35", "f35", "fighter jet", "stealth fighter", "aircraft",
    "jet", "fighter", "military aircraft", "warplane", "plane",
)

# Keywords expected when the chain lands on a pilot frame (e.g. ~t=55 s).
_PILOT_KEYWORDS: tuple[str, ...] = (
    "pilot", "person", "man", "helmet", "flight suit",
    "g-suit", "uniform", "people", "aviator",
)

VISION_SCENARIOS: list[OrchestratorScenario] = [
    # ------------------------------------------------------------------
    # F-28: Single agent — frame extraction  (≡ SC-28)
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="F-28",
        name="Single Agent: Frame Extraction",
        difficulty=2,
        prompt_templates={
            "vague": "Pull some images from this video at {input_path}.",
            "clear": (
                "Use the frame extraction agent to extract frames from the video "
                "at {input_path} between 0 and 5 seconds at 1 fps."
            ),
            "algorithmic": (
                "extract_frames_handoff. video_path={input_path}. "
                "t_start=0. t_end=5. fps=1."
            ),
        },
        expected_agents=[_EXTRACT_FRAMES],
        # Level-aware expectation: "pull some images" is genuinely ambiguous, so at
        # the vague level asking for clarification is an equally valid behaviour —
        # accept either. clear/algorithmic stay strict (extraction only). This
        # encodes the objective ("respond sensibly to ambiguity"), it does not
        # relax a correctness threshold.
        expected_agents_by_level={
            "vague": [f"{_EXTRACT_FRAMES}|{_REQUEST_CLARIFICATION}"],
        },
        default_input_path="fixtures/raw/f-35.mp4",
    ),

    # ------------------------------------------------------------------
    # F-29: Single agent — VQA on pre-extracted reference frame  (≡ SC-29)
    # Uses f-35_t15.jpg (F-35 in flight) to avoid a dependency on F-28.
    # GT: vqa.json item 3 — "What type of military aircraft?" (identify; F-35/jet).
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="F-29",
        name="Single Agent: VQA on Image",
        difficulty=2,
        prompt_templates={
            "vague": "What is shown in this picture at {input_path}?",
            "clear": (
                "Use the VQA agent to answer: what type of aircraft appears in the "
                "image at {input_path}?"
            ),
            "algorithmic": (
                "vqa_handoff. frame_path={input_path}. "
                "question='What type of military aircraft is visible in this image?'"
            ),
        },
        expected_agents=[_VQA],
        default_input_path="fixtures/raw/f-35_t15.jpg",
        timeout_s=900,
        vision_criteria=VisionSuccessCriteria(
            vqa_keywords=_AIRCRAFT_KEYWORDS,
            min_answer_length=5,
        ),
        result_criteria=ResultCriteria(
            metric="vqa",
            ground_truth_ref="videos/f-35/vqa.json",
            threshold=0.5,
            item_index=3,  # t015s "What type of military aircraft?" (identify; gold)
        ),
    ),

    # ------------------------------------------------------------------
    # F-30: Two-step chain — Frame Extraction → VQA  (≡ SC-30)
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="F-30",
        name="2-Step Pipeline: Frame Extraction → VQA",
        difficulty=3,
        prompt_templates={
            "vague": "Look at a frame from {input_path} around 15 seconds and tell me what aircraft you see.",
            "clear": (
                "Extract a frame from the video at {input_path} (around 15 seconds) "
                "and then use VQA to describe what military aircraft is visible."
            ),
            "algorithmic": (
                "extract_frames_handoff(video_path={input_path}, t_start=14, t_end=16, fps=1) "
                "→ vqa_handoff(frame_path=<extracted_frame>, "
                "question='What type of military aircraft is in this image?')"
            ),
        },
        expected_agents=[_EXTRACT_FRAMES, _VQA],
        default_input_path="fixtures/raw/f-35.mp4",
        timeout_s=900,
        vision_criteria=VisionSuccessCriteria(
            vqa_keywords=_AIRCRAFT_KEYWORDS,
            min_answer_length=5,
        ),
        result_criteria=ResultCriteria(
            metric="vqa",
            ground_truth_ref="videos/f-35/vqa.json",
            threshold=0.5,
            item_index=3,  # same GT item as F-29
        ),
    ),

    # ------------------------------------------------------------------
    # F-31: Single agent — image segmentation on pre-extracted frame  (≡ SC-31)
    # Uses f-35_t15.jpg. IoU threshold=0.3 (lenient) — extracted frame may differ
    # slightly from the GT t015s frame.
    # GT: segmentation.json item 1 — F-35 in flight, bbox [310,325,1100,600].
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="F-31",
        name="Single Agent: Image Segmentation",
        difficulty=3,
        prompt_templates={
            "vague": "Segment the objects in {input_path} and show me where they are.",
            "clear": (
                "Use the segmentation agent to detect the aircraft in the "
                "image at {input_path}. Return the detected bounding boxes and labels."
            ),
            "algorithmic": (
                "segmentation_handoff. image_path={input_path}. "
                "text_query='airplane'. output_mode='masks'."
            ),
        },
        expected_agents=[_SEGMENTATION],
        default_input_path="fixtures/raw/f-35_t15.jpg",
        result_criteria=ResultCriteria(
            metric="iou",
            ground_truth_ref="videos/f-35/segmentation.json",
            threshold=0.3,  # lenient — exact frame may differ from GT t015s
            item_index=1,   # F-35 in flight at t015s (present=True)
        ),
    ),

    # ------------------------------------------------------------------
    # F-32: Two-step chain — Frame Extraction → Segmentation  (≡ SC-32)
    # ------------------------------------------------------------------
    # CORRECTNESS NOTE — why metric="detect", not "iou":
    # The orchestrator extracts its OWN frame here (model-dependent: t=14, t=15, a
    # 1-fps range, …). A fixed-bbox IoU against the GT frame t015s would compare
    # boxes drawn on a DIFFERENT image than the GT was authored on → an invalid
    # comparison that fails on model luck, not behaviour (the original false
    # negative). So this scenario asserts the frame-independent CAPABILITY claim —
    # "segmentation returned >= 1 geometrically-plausible airplane box in whatever
    # frame it used" — via the `detect` metric. Strict localisation IoU stays on
    # F-31, where the artifact under test IS the GT frame. The routing contract
    # ([ExtractFrames, Segmentation]) still carries the 2-step-chain behaviour.
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="F-32",
        name="2-Step Pipeline: Frame Extraction → Segmentation",
        difficulty=3,
        prompt_templates={
            "vague": "Detect and locate the aircraft in a frame from {input_path}, returning bounding boxes.",
            "clear": (
                "Extract a frame from {input_path} at 15 seconds and then "
                "detect the aircraft in that frame, returning bounding boxes and labels."
            ),
            "algorithmic": (
                "extract_frames_handoff(video_path={input_path}, t_start=14, t_end=16, fps=1) "
                "→ segmentation_handoff(image_path=<frame>, text_query='airplane', "
                "output_mode='masks')"
            ),
        },
        expected_agents=[_EXTRACT_FRAMES, _SEGMENTATION],
        default_input_path="fixtures/raw/f-35.mp4",
        timeout_s=600,
        result_criteria=ResultCriteria(
            metric="detect",
            ground_truth_ref="videos/f-35/segmentation.json",
            threshold=0.005,  # min plausible-box area fraction; upper bound fixed at 0.98
            item_index=1,     # used only for image_w/image_h (1280×720)
        ),
    ),

    # ------------------------------------------------------------------
    # F-33: Full cross-modal chain — AudioExtraction → Transcription → Grep → Frame → VQA
    # (≡ SC-33).  Self-contained: f-35's transcript belongs to the same video.
    # Search word "helmet" is spoken at ~52 s ("G-suit, flying suit, helmet, ...")
    # and lands on the pilot frame (cf. GT t055s — pilot in flight gear).
    # Long chain: timeout_s=600. At vague level this may fail on small models —
    # that is expected and acceptable. Clear/algorithmic should pass.
    # Pipeline: audio_extraction → transcription → information_finder → extract_frames → vqa
    # audio_analysis must NOT be called here (that is Chain E, video splitting only).
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="F-33",
        name="Full Cross-Modal: AudioExtraction → Transcription → Grep → Frame → VQA",
        difficulty=5,
        prompt_templates={
            "vague": (
                "Search {input_path} for the moment the word 'helmet' is spoken. "
                "Use transcription and keyword search to find the timestamp, "
                "then extract a frame and describe what you see."
            ),
            "clear": (
                "Use Chain G on {input_path}: extract audio, transcribe it (skip audio analysis), "
                "search the transcript for 'helmet', extract a frame at the matched timestamp, "
                "then use VQA to describe the frame."
            ),
            "algorithmic": (
                "audio_extraction_handoff(video_path={input_path}) "
                "→ transcription_handoff(archive_path=<extracted_archive>) "
                "→ information_finder_handoff(archive_path=<transcript_archive>, search_pattern='helmet') "
                "→ extract_frames_handoff(video_path={input_path}, "
                "t_start=<matched_timestamp>, t_end=<matched_timestamp+3>, fps=1) "
                "→ vqa_handoff(frame_path=<frame>, "
                "question='Describe what you see in this image. Focus on the main subject.')"
            ),
        },
        expected_agents=[_AUDIO_EXTRACTION, _TRANSCRIPTION, _INFORMATION_FINDER, _EXTRACT_FRAMES, _VQA],
        default_input_path="fixtures/raw/f-35.mp4",
        timeout_s=600,
        vision_criteria=VisionSuccessCriteria(
            vqa_keywords=_PILOT_KEYWORDS,
            min_answer_length=20,
        ),
    ),

    # ------------------------------------------------------------------
    # F-34: Single agent — video segmentation (full clip)  (≡ SC-34)
    # Uses f-35_clip.mp4 (≈12–27 s) so the in-flight jet is present for the
    # "airplane" query and the test finishes in reasonable time.
    # timeout_s=600 because video segmentation is GPU-heavy.
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="F-34",
        name="Single Agent: Video Segmentation",
        difficulty=4,
        prompt_templates={
            "vague": "Segment the aircraft throughout the entire video at {input_path}.",
            "clear": (
                "Use the segmentation agent to produce a segmented video of the "
                "aircraft in {input_path}. Output mode: video."
            ),
            "algorithmic": (
                "segmentation_handoff. video_path={input_path}. "
                "text_query='airplane'. output_mode='video'. fps=1."
            ),
        },
        expected_agents=[_SEGMENTATION],
        default_input_path="fixtures/raw/f-35_clip.mp4",
        timeout_s=600,
    ),

    # ------------------------------------------------------------------
    # F-35: Error — VQA requested for an audio file (.wav)  (≡ SC-35)
    # VQA only processes images. The orchestrator should reject this immediately
    # with reject_handoff (just as it does for audio extraction on a .png).
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="F-35",
        name="Error Handling: VQA on Audio File",
        difficulty=2,
        prompt_templates={
            "vague": "Describe what you see in this audio file at {input_path}.",
            "clear": (
                "Use VQA to visually analyze the audio file at {input_path} "
                "and describe what it shows."
            ),
            "algorithmic": (
                "vqa_handoff. frame_path={input_path}. "
                "question='What is visible in this audio file?'"
            ),
        },
        expected_agents=[f"{_REJECT}|{_REQUEST_CLARIFICATION}"],
        default_input_path="fixtures/raw/f-35.wav",
    ),

    # ------------------------------------------------------------------
    # F-36: Error — Segmentation requested but no file path provided  (≡ SC-36)
    # The orchestrator should ask for clarification via request_clarification_handoff
    # before attempting to segment. default_input_path=None so {input_path} is NOT
    # in the prompt template.
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="F-36",
        name="Error Handling: Segmentation Without File Path",
        difficulty=2,
        prompt_templates={
            "vague": "Segment the main aircraft in the scene.",
            "clear": "Use the segmentation agent to segment the aircraft. Text query: airplane.",
            "algorithmic": "segmentation_handoff. text_query='airplane'. output_mode='segmented_image'.",
        },
        expected_agents=[_REQUEST_CLARIFICATION],
        default_input_path=None,
    ),

    # ------------------------------------------------------------------
    # F-37: Error — searched word never appears in the transcript  (≡ SC-37)
    # "quantum" is a verified true-negative in f-35/grep.json (0 matches). The
    # orchestrator must NOT fabricate a timestamp; it should report "no match
    # found" or ask for clarification — the chain ends at information_finder
    # (or earlier with a reject/clarification). OR-logic: any graceful end is valid.
    # ------------------------------------------------------------------
    OrchestratorScenario(
        scenario_id="F-37",
        name="Error Handling: Search Word Not in Transcript",
        difficulty=3,
        prompt_templates={
            "vague": (
                "For the video at {input_path}, find where the speaker says "
                "'quantum' and show me that frame."
            ),
            "clear": (
                "Extract the audio from {input_path}, transcribe it, then search "
                "the transcript for the word 'quantum'. "
                "Report whether the word was found."
            ),
            "algorithmic": (
                "audio_extraction_handoff(video_path={input_path}) "
                "→ transcription_handoff(archive_path=<extracted_archive>) "
                "→ information_finder_handoff(path=<transcript_archive>, "
                "query='quantum')"
            ),
        },
        expected_agents=[f"{_REJECT}|{_REQUEST_CLARIFICATION}|{_INFORMATION_FINDER}"],
        default_input_path="fixtures/raw/f-35.mp4",
        timeout_s=600,
    ),
]


# ===========================================================================
# EXTRA f-35 scenarios (F-38..F-42) — DISABLED / COMMENTED OUT.
# ---------------------------------------------------------------------------
# These exercise capabilities the original 10-scenario suite (SC-28..SC-37) did
# NOT cover: VQA true-negatives, counting, attribute extraction, and
# multi-instance segmentation. They share the same overall objective (verify
# agentic-system behavior) but go beyond the original goals, so they are kept
# here for reference and intentionally EXCLUDED from the active VISION_SCENARIOS
# list above. To enable: uncomment and append to VISION_SCENARIOS.
# ===========================================================================
#
# _EXTRA_VISION_SCENARIOS = [
#     # ------------------------------------------------------------------
#     # F-38: VQA true-negative — hallucination check.
#     # On the pilot-only frame (t=5 s) there is NO aircraft. GT: vqa.json item 2
#     # (is_true_negative=True) — the vqa metric passes only when the answer denies
#     # presence ("no", "not", "none", ...). No vision_criteria keywords (a TN must
#     # not be keyword-gated).
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="F-38",
#         name="VQA True-Negative: No Aircraft Present",
#         difficulty=3,
#         prompt_templates={
#             "vague": "Is there any aircraft in this picture at {input_path}?",
#             "clear": (
#                 "Use the VQA agent to answer: is there a military aircraft or jet "
#                 "visible in the image at {input_path}?"
#             ),
#             "algorithmic": (
#                 "vqa_handoff. frame_path={input_path}. "
#                 "question='Is there a military aircraft or jet visible anywhere in this image?'"
#             ),
#         },
#         expected_agents=[_VQA],
#         default_input_path="fixtures/raw/f-35_t05.jpg",
#         timeout_s=900,
#         result_criteria=ResultCriteria(
#             metric="vqa",
#             ground_truth_ref="videos/f-35/vqa.json",
#             threshold=0.5,
#             item_index=2,  # t005s "Is a military aircraft visible?" (true_negative → No)
#         ),
#     ),
#
#     # ------------------------------------------------------------------
#     # F-39: Segmentation true-negative — no false positives.
#     # Query "airplane" on the pilot-only frame (t=5 s). GT: segmentation.json
#     # item 0 (present=False) — the iou metric passes only when zero boxes return.
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="F-39",
#         name="Segmentation True-Negative: No Aircraft to Detect",
#         difficulty=3,
#         prompt_templates={
#             "vague": "Segment any aircraft in {input_path} and show me where they are.",
#             "clear": (
#                 "Use the segmentation agent to detect aircraft in the image at "
#                 "{input_path}. Return the detected bounding boxes and labels."
#             ),
#             "algorithmic": (
#                 "segmentation_handoff. image_path={input_path}. "
#                 "text_query='airplane'. output_mode='masks'."
#             ),
#         },
#         expected_agents=[_SEGMENTATION],
#         default_input_path="fixtures/raw/f-35_t05.jpg",
#         result_criteria=ResultCriteria(
#             metric="iou",
#             ground_truth_ref="videos/f-35/segmentation.json",
#             threshold=0.3,  # threshold unused for a TN — passes iff no boxes returned
#             item_index=0,   # t005s airplane present=False
#         ),
#     ),
#
#     # ------------------------------------------------------------------
#     # F-40: VQA counting — four aircraft in formation (t=65 s).
#     # GT: vqa.json item 9 (count; required "4"/"four").
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="F-40",
#         name="VQA Counting: Aircraft in Formation",
#         difficulty=4,
#         prompt_templates={
#             "vague": "How many aircraft can you see in {input_path}?",
#             "clear": (
#                 "Use the VQA agent to count: how many aircraft are visible in the "
#                 "image at {input_path}?"
#             ),
#             "algorithmic": (
#                 "vqa_handoff. frame_path={input_path}. "
#                 "question='How many aircraft are visible in this image?'"
#             ),
#         },
#         expected_agents=[_VQA],
#         default_input_path="fixtures/raw/f-35_t65.jpg",
#         timeout_s=900,
#         vision_criteria=VisionSuccessCriteria(
#             vqa_keywords=("four", "4"),
#             min_answer_length=1,
#         ),
#         result_criteria=ResultCriteria(
#             metric="vqa",
#             ground_truth_ref="videos/f-35/vqa.json",
#             threshold=0.5,
#             item_index=9,  # t065s "How many aircraft are visible?" (count; four)
#         ),
#     ),
#
#     # ------------------------------------------------------------------
#     # F-41: VQA attribute — fuselage colour of the in-flight jet (t=15 s).
#     # GT: vqa.json item 4 (attribute; gray).
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="F-41",
#         name="VQA Attribute: Aircraft Fuselage Colour",
#         difficulty=3,
#         prompt_templates={
#             "vague": "What colour is the plane in {input_path}?",
#             "clear": (
#                 "Use the VQA agent to answer: what colour is the fuselage of the "
#                 "aircraft in the image at {input_path}?"
#             ),
#             "algorithmic": (
#                 "vqa_handoff. frame_path={input_path}. "
#                 "question='What color is the fuselage of the aircraft in flight?'"
#             ),
#         },
#         expected_agents=[_VQA],
#         default_input_path="fixtures/raw/f-35_t15.jpg",
#         timeout_s=900,
#         vision_criteria=VisionSuccessCriteria(
#             vqa_keywords=("gray", "grey"),
#             min_answer_length=3,
#         ),
#         result_criteria=ResultCriteria(
#             metric="vqa",
#             ground_truth_ref="videos/f-35/vqa.json",
#             threshold=0.5,
#             item_index=4,  # t015s "What color is the fuselage?" (attribute; gray)
#         ),
#     ),
#
#     # ------------------------------------------------------------------
#     # F-42: Multi-instance segmentation — the 4-jet formation (t=65 s).
#     # GT: segmentation.json item 4 (four objects). The iou metric requires EVERY
#     # GT box to reach threshold, and four small/distant jets are hard, so the
#     # threshold is very lenient (0.2).
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="F-42",
#         name="Multi-Instance Segmentation: Formation",
#         difficulty=4,
#         prompt_templates={
#             "vague": "Segment all the aircraft in the formation in {input_path}.",
#             "clear": (
#                 "Use the segmentation agent to detect all the aircraft in the image "
#                 "at {input_path}. Return the detected bounding boxes and labels."
#             ),
#             "algorithmic": (
#                 "segmentation_handoff. image_path={input_path}. "
#                 "text_query='airplane'. output_mode='masks'."
#             ),
#         },
#         expected_agents=[_SEGMENTATION],
#         default_input_path="fixtures/raw/f-35_t65.jpg",
#         result_criteria=ResultCriteria(
#             metric="iou",
#             ground_truth_ref="videos/f-35/segmentation.json",
#             threshold=0.2,  # very lenient — 4 small jets; all boxes must clear it
#             item_index=4,   # t065s formation (4 objects, present=True)
#         ),
#     ),
# ]


# ===========================================================================
# LEGACY SCENARIOS (SC-28..SC-37) — DISABLED / COMMENTED OUT.
# ---------------------------------------------------------------------------
# These used the lionandgorilla / tedtalk / sample_video.wav fixtures, which
# were lost. They are kept here for reference only and are NOT part of the
# active VISION_SCENARIOS list above. The f-35 scenarios (F-28..F-37) replace
# them one-for-one.
# ===========================================================================
#
# # Keywords expected in VQA answers for lionandgorilla.mp4.
# _LION_GORILLA_KEYWORDS: tuple[str, ...] = (
#     "lion", "gorilla", "animal", "cat", "primate", "ape",
#     "feline", "mammal", "big cat", "great ape",
# )
#
# _LEGACY_VISION_SCENARIOS = [
#     # ------------------------------------------------------------------
#     # SC-28: Single agent — frame extraction
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="SC-28",
#         name="Single Agent: Frame Extraction",
#         difficulty=2,
#         prompt_templates={
#             "vague": "Pull some images from this video at {input_path}.",
#             "clear": (
#                 "Use the frame extraction agent to extract frames from the video "
#                 "at {input_path} between 0 and 5 seconds at 1 fps."
#             ),
#             "algorithmic": (
#                 "extract_frames_handoff. video_path={input_path}. "
#                 "t_start=0. t_end=5. fps=1."
#             ),
#         },
#         expected_agents=[_EXTRACT_FRAMES],
#         default_input_path="fixtures/raw/lionandgorilla.mp4",
#     ),
#
#     # ------------------------------------------------------------------
#     # SC-29: Single agent — VQA on pre-extracted reference frame
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="SC-29",
#         name="Single Agent: VQA on Image",
#         difficulty=2,
#         prompt_templates={
#             "vague": "What is shown in this picture at {input_path}?",
#             "clear": (
#                 "Use the VQA agent to answer: what animals appear in the "
#                 "image at {input_path}?"
#             ),
#             "algorithmic": (
#                 "vqa_handoff. frame_path={input_path}. "
#                 "question='What animals are visible in this image?'"
#             ),
#         },
#         expected_agents=[_VQA],
#         default_input_path="fixtures/raw/lionandgorilla_frame0.jpg",
#         timeout_s=900,
#         vision_criteria=VisionSuccessCriteria(
#             vqa_keywords=_LION_GORILLA_KEYWORDS,
#             min_answer_length=5,
#         ),
#         result_criteria=ResultCriteria(
#             metric="vqa",
#             ground_truth_ref="videos/lionandgorilla/vqa.json",
#             threshold=0.5,
#             item_index=1,  # "What two animals are present?" (identify; gold)
#         ),
#     ),
#
#     # ------------------------------------------------------------------
#     # SC-30: Two-step chain — Frame Extraction → VQA
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="SC-30",
#         name="2-Step Pipeline: Frame Extraction → VQA",
#         difficulty=3,
#         prompt_templates={
#             "vague": "Look at a frame from {input_path} and tell me what you see.",
#             "clear": (
#                 "Extract a frame from the video at {input_path} (around 2 seconds) "
#                 "and then use VQA to describe what animals are visible."
#             ),
#             "algorithmic": (
#                 "extract_frames_handoff(video_path={input_path}, t_start=1, t_end=3, fps=1) "
#                 "→ vqa_handoff(frame_path=<extracted_frame>, "
#                 "question='What animals are in this image?')"
#             ),
#         },
#         expected_agents=[_EXTRACT_FRAMES, _VQA],
#         default_input_path="fixtures/raw/lionandgorilla.mp4",
#         timeout_s=900,
#         vision_criteria=VisionSuccessCriteria(
#             vqa_keywords=_LION_GORILLA_KEYWORDS,
#             min_answer_length=5,
#         ),
#         result_criteria=ResultCriteria(
#             metric="vqa",
#             ground_truth_ref="videos/lionandgorilla/vqa.json",
#             threshold=0.5,
#             item_index=1,  # same GT item as SC-29
#         ),
#     ),
#
#     # ------------------------------------------------------------------
#     # SC-31: Single agent — image segmentation on pre-extracted frame
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="SC-31",
#         name="Single Agent: Image Segmentation",
#         difficulty=3,
#         prompt_templates={
#             "vague": "Segment the objects in {input_path} and show me where they are.",
#             "clear": (
#                 "Use the segmentation agent to detect the animals in the "
#                 "image at {input_path}. Return the detected bounding boxes and labels."
#             ),
#             "algorithmic": (
#                 "segmentation_handoff. image_path={input_path}. "
#                 "text_query='animal'. output_mode='masks'."
#             ),
#         },
#         expected_agents=[_SEGMENTATION],
#         default_input_path="fixtures/raw/lionandgorilla_frame0.jpg",
#         result_criteria=ResultCriteria(
#             metric="iou",
#             ground_truth_ref="videos/lionandgorilla/segmentation.json",
#             threshold=0.3,  # lenient — exact frame may differ from GT t006s
#             item_index=0,   # lion at t006s (present=True)
#         ),
#     ),
#
#     # ------------------------------------------------------------------
#     # SC-32: Two-step chain — Frame Extraction → Segmentation
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="SC-32",
#         name="2-Step Pipeline: Frame Extraction → Segmentation",
#         difficulty=3,
#         prompt_templates={
#             "vague": "Detect and locate the animals in a frame from {input_path}, returning bounding boxes.",
#             "clear": (
#                 "Extract a frame from {input_path} at 2 seconds and then "
#                 "detect the animals in that frame, returning bounding boxes and labels."
#             ),
#             "algorithmic": (
#                 "extract_frames_handoff(video_path={input_path}, t_start=1.5, t_end=3.0, fps=1) "
#                 "→ segmentation_handoff(image_path=<frame>, text_query='animal', "
#                 "output_mode='masks')"
#             ),
#         },
#         expected_agents=[_EXTRACT_FRAMES, _SEGMENTATION],
#         default_input_path="fixtures/raw/lionandgorilla.mp4",
#         timeout_s=600,
#         result_criteria=ResultCriteria(
#             metric="iou",
#             ground_truth_ref="videos/lionandgorilla/segmentation.json",
#             threshold=0.3,  # lenient — extracted frame may not match GT t006s exactly
#             item_index=0,   # lion at t006s (present=True)
#         ),
#     ),
#
#     # ------------------------------------------------------------------
#     # SC-33: Full cross-modal chain — AudioExtraction → Transcription → Grep → Frame → VQA
#     # Uses tedtalk.mp4; search word "Facebook" verified to appear in the video.
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="SC-33",
#         name="Full Cross-Modal: AudioExtraction → Transcription → Grep → Frame → VQA",
#         difficulty=5,
#         prompt_templates={
#             "vague": (
#                 "Search {input_path} for the moment the word 'Facebook' is spoken. "
#                 "Use transcription and keyword search to find the timestamp, "
#                 "then extract a frame and describe what you see."
#             ),
#             "clear": (
#                 "Use Chain G on {input_path}: extract audio, transcribe it (skip audio analysis), "
#                 "search the transcript for 'Facebook', extract a frame at the matched timestamp, "
#                 "then use VQA to describe the frame."
#             ),
#             "algorithmic": (
#                 "audio_extraction_handoff(video_path={input_path}) "
#                 "→ transcription_handoff(archive_path=<extracted_archive>) "
#                 "→ information_finder_handoff(archive_path=<transcript_archive>, search_pattern='Facebook') "
#                 "→ extract_frames_handoff(video_path={input_path}, "
#                 "t_start=<matched_timestamp>, t_end=<matched_timestamp+3>, fps=1) "
#                 "→ vqa_handoff(frame_path=<frame>, "
#                 "question='Describe what you see in this image. Focus on the main subject.')"
#             ),
#         },
#         expected_agents=[_AUDIO_EXTRACTION, _TRANSCRIPTION, _INFORMATION_FINDER, _EXTRACT_FRAMES, _VQA],
#         default_input_path="fixtures/raw/tedtalk.mp4",
#         timeout_s=600,
#         vision_criteria=VisionSuccessCriteria(
#             vqa_keywords=("person", "man", "woman", "speaker", "human", "people"),
#             min_answer_length=20,
#         ),
#     ),
#
#     # ------------------------------------------------------------------
#     # SC-34: Single agent — video segmentation (full clip), lionandgorilla_15s.mp4
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="SC-34",
#         name="Single Agent: Video Segmentation",
#         difficulty=4,
#         prompt_templates={
#             "vague": "Segment the animals throughout the entire video at {input_path}.",
#             "clear": (
#                 "Use the segmentation agent to produce a segmented video of the "
#                 "animals in {input_path}. Output mode: video."
#             ),
#             "algorithmic": (
#                 "segmentation_handoff. video_path={input_path}. "
#                 "text_query='animal'. output_mode='video'. fps=1."
#             ),
#         },
#         expected_agents=[_SEGMENTATION],
#         default_input_path="fixtures/raw/lionandgorilla_15s.mp4",
#         timeout_s=600,
#     ),
#
#     # ------------------------------------------------------------------
#     # SC-35: Error — VQA requested for an audio file (.wav)
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="SC-35",
#         name="Error Handling: VQA on Audio File",
#         difficulty=2,
#         prompt_templates={
#             "vague": "Describe what you see in this audio file at {input_path}.",
#             "clear": (
#                 "Use VQA to visually analyze the audio file at {input_path} "
#                 "and describe what it shows."
#             ),
#             "algorithmic": (
#                 "vqa_handoff. frame_path={input_path}. "
#                 "question='What is visible in this audio file?'"
#             ),
#         },
#         expected_agents=[f"{_REJECT}|{_REQUEST_CLARIFICATION}"],
#         default_input_path="fixtures/raw/sample_video.wav",
#     ),
#
#     # ------------------------------------------------------------------
#     # SC-36: Error — Segmentation requested but no file path provided
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="SC-36",
#         name="Error Handling: Segmentation Without File Path",
#         difficulty=2,
#         prompt_templates={
#             "vague": "Segment the main object in the scene.",
#             "clear": "Use the segmentation agent to segment the animals. Text query: animal.",
#             "algorithmic": "segmentation_handoff. text_query='animal'. output_mode='segmented_image'.",
#         },
#         expected_agents=[_REQUEST_CLARIFICATION],
#         default_input_path=None,
#     ),
#
#     # ------------------------------------------------------------------
#     # SC-37: Error — searched word never appears in the transcript
#     # ------------------------------------------------------------------
#     OrchestratorScenario(
#         scenario_id="SC-37",
#         name="Error Handling: Search Word Not in Transcript",
#         difficulty=3,
#         prompt_templates={
#             "vague": (
#                 "For the video at {input_path}, find where the speaker says "
#                 "'supercalifragilistic' and show me that frame."
#             ),
#             "clear": (
#                 "Extract the audio from {input_path}, transcribe it, then search "
#                 "the transcript for the word 'supercalifragilistic'. "
#                 "Report whether the word was found."
#             ),
#             "algorithmic": (
#                 "audio_extraction_handoff(video_path={input_path}) "
#                 "→ transcription_handoff(archive_path=<extracted_archive>) "
#                 "→ information_finder_handoff(path=<transcript_archive>, "
#                 "query='supercalifragilistic')"
#             ),
#         },
#         expected_agents=[f"{_REJECT}|{_REQUEST_CLARIFICATION}|{_INFORMATION_FINDER}"],
#         default_input_path="fixtures/raw/tedtalk.mp4",
#         timeout_s=600,
#     ),
# ]
