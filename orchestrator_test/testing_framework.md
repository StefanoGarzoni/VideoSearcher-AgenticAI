This Markdown document provides a comprehensive structure for your testing framework, specifically designed for a multi-agent video processing pipeline. It includes the logic for both the master test case repository and the live results tracking sheet.

This final version of the Markdown document incorporates the single-tool unit tests, re-orders scenarios by difficulty, and includes file and folder paths.

# AI Orchestrator Testing Framework

## 1. Test Case Sheet (Input & Design)

This sheet defines the ground truth for your testing suite. It progresses from simple unit tests (single tool) to complex error handling and full-pipeline integration.

| Scenario ID | Difficulty | Scenario Name | Prompt Level | Input Prompt | Expected Sequence | Input Path |
| --- | --- | --- | --- | --- | --- | --- |
| **SC-01** | 1 | **Unit: Audio Extraction** | Vague | "Extract the sound from this video." | Ffmpeg-0 | `[PATH_TO_VIDEO_MP4]` |
| **SC-01** | 1 | **Unit: Audio Extraction** | Clear | "Use Ffmpeg-0 to extract audio.wav from the video file." | Ffmpeg-0 | `[PATH_TO_VIDEO_MP4]` |
| **SC-01** | 1 | **Unit: Audio Extraction** | Algorithmic | "Run Ffmpeg-0: Input=[PATH], Output=[PATH]/audio.wav." | Ffmpeg-0 | `[PATH_TO_VIDEO_MP4]` |
| **SC-02** | 2 | **Unit: Transcription** | Vague | "Turn this recording into text." | Transcribe | `[PATH_TO_ARCHIVE_TAR_GZ]` |
| **SC-02** | 2 | **Unit: Transcription** | Clear | "Run the transcription agent on this archive to get transcript.txt." | Transcribe | `[PATH_TO_ARCHIVE_TAR_GZ]` |
| **SC-02** | 2 | **Unit: Transcription** | Algorithmic | "Execute Faster-Whisper transcription on the provided .tar.gz." | Transcribe | `[PATH_TO_ARCHIVE_TAR_GZ]` |
| **SC-03** | 3 | **Unit: Grep Search** | Vague | "Does this file mention 'Error'?" | Grep | `[PATH_TO_TRANSCRIPT_ARCHIVE]` |
| **SC-03** | 3 | **Unit: Grep Search** | Clear | "Search the transcript.txt in this folder for the word 'Error'." | Grep | `[PATH_TO_TRANSCRIPT_ARCHIVE]` |
| **SC-03** | 3 | **Unit: Grep Search** | Algorithmic | "Call Grep(query='Error') on the transcript.txt within the archive." | Grep | `[PATH_TO_TRANSCRIPT_ARCHIVE]` |
| **SC-04** | 4 | **Sequence: Early Pipeline** | Vague | "Find the sentence breaks in this video." | Ffmpeg-0 -> Librosa | `[PATH_TO_VIDEO_MP4]` |
| **SC-04** | 4 | **Sequence: Early Pipeline** | Clear | "Extract the audio and then use Librosa to find timestamps." | Ffmpeg-0 -> Librosa | `[PATH_TO_VIDEO_MP4]` |
| **SC-04** | 4 | **Sequence: Early Pipeline** | Algorithmic | "1. Ffmpeg-0 -> 2. Librosa. Output timestamps.txt in archive." | Ffmpeg-0 -> Librosa | `[PATH_TO_VIDEO_MP4]` |
| **SC-05** | 5 | **Sequence: Late Pipeline** | Vague | "Find the word 'Apple' in these video clips." | Ffmpeg-2 -> Transcribe -> Grep | `[FOLDER_PATH_OF_MP4_CLIPS]` |
| **SC-05** | 5 | **Sequence: Late Pipeline** | Clear | "Compress these clips, transcribe them, and search for 'Apple'." | Ffmpeg-2 -> Transcribe -> Grep | `[FOLDER_PATH_OF_MP4_CLIPS]` |
| **SC-05** | 5 | **Sequence: Late Pipeline** | Algorithmic | "Loop: Ffmpeg-2 -> Transcribe -> Grep(query='Apple') for folder." | Ffmpeg-2 -> Transcribe -> Grep | `[FOLDER_PATH_OF_MP4_CLIPS]` |
| **SC-06** | 6 | **Integration: Full Pipeline** | Vague | "Tell me when they talk about 'Revenue' in this movie." | Ffmpeg-0 -> Librosa -> ... -> Grep | `[PATH_TO_VIDEO_MP4]` |
| **SC-06** | 6 | **Integration: Full Pipeline** | Clear | "Extract, split by sentence, transcribe, and search for 'Revenue'." | Ffmpeg-0 -> Librosa -> ... -> Grep | `[PATH_TO_VIDEO_MP4]` |
| **SC-06** | 6 | **Integration: Full Pipeline** | Algorithmic | "Full Chain: Ffmpeg-0, Librosa, Ffmpeg-1, Ffmpeg-2, Transcribe, Grep." | Ffmpeg-0 -> ... -> Grep | `[PATH_TO_VIDEO_MP4]` |
| **SC-07** | 7 | **Error: Invalid Format** | Vague | "Transcribe this photo." | Reject / Error | `[PATH_TO_IMAGE_PNG]` |
| **SC-07** | 7 | **Error: Invalid Format** | Clear | "Process this .png file through the audio extraction agent." | Reject / Error | `[PATH_TO_IMAGE_PNG]` |
| **SC-07** | 7 | **Error: Invalid Format** | Algorithmic | "Attempt Ffmpeg-0 on 'file.png'. Expected: InputValidationError." | Reject / Error | `[PATH_TO_IMAGE_PNG]` |
| **SC-08** | 8 | **Error: Missing Data** | Vague | "Search for keywords." (No file provided) | Request Missing File | `[NULL]` |
| **SC-08** | 8 | **Error: Missing Data** | Clear | "Perform a grep search on the video I haven't uploaded yet." | Request Missing File | `[NULL]` |
| **SC-08** | 8 | **Error: Missing Data** | Algorithmic | "Execute Grep(query='x') where Input=None. Expected: NoInputError." | Request Missing File | `[NULL]` |
| **SC-09** | 9 | **Logic: State Recovery** | Vague | "Check the sound again." | Clarify or Re-run Ffmpeg-0 | `[PATH_TO_VIDEO_MP4]` |
| **SC-09** | 9 | **Logic: State Recovery** | Clear | "The audio was bad, try extracting it again with the agent." | Ffmpeg-0 (Override) | `[PATH_TO_VIDEO_MP4]` |
| **SC-09** | 9 | **Logic: State Recovery** | Algorithmic | "Force re-execution of Ffmpeg-0 on existing video.mp4." | Ffmpeg-0 (Override) | `[PATH_TO_VIDEO_MP4]` |
| **SC-10** | 10 | **Logic: Null Handling** | Vague | "Find 'Aliens' in this cooking show." | Full Pipeline (Empty JSON) | `[PATH_TO_COOKING_VIDEO_MP4]` |
| **SC-10** | 10 | **Logic: Null Handling** | Clear | "Run search for 'Aliens' even though it's likely to fail." | Full Pipeline (Empty JSON) | `[PATH_TO_COOKING_VIDEO_MP4]` |
| **SC-10** | 10 | **Logic: Null Handling** | Algorithmic | "Full Pipeline. Target='Aliens'. Expected: {matches: []}." | Full Pipeline (Empty JSON) | `[PATH_TO_COOKING_VIDEO_MP4]` |

---

## 2. Results Tracking Sheet (Headers)

Paste these headers into your second Excel tab to record live performance data.

| Column | Description |
| --- | --- |
| **Run ID** | Unique test run identifier (e.g., `2026-01-28_V1`). |
| **Scenario ID** | References `SC-01` through `SC-10`. |
| **Model Tested** | Name of the LLM (e.g., `Gemini 1.5 Pro`). |
| **Actual Sequence** | The tools the AI actually invoked. |
| **Pass/Fail (1/0)** | Binary indicator of success. |
| **Failed At Step** | Which tool or data transition caused the error? |
| **Reasoning Score** | Score (1-5) on how logical the tool choice was. |
| **Latency (s)** | Time taken for the Orchestrator to respond. |
| **Observations** | Qualitative notes on hallucinations or edge cases. |

---

## 3. Performance Metrics

Track these to measure the reliability of your Orchestrator over time.

* **Success Rate ():** Total Passes / Total Tests run.
* **Tool-Call Precision:** The ratio of correct tool selections vs. unnecessary "extra" calls.
* **Prompt Sensitivity:** The difference in Success Rate between Vague and Algorithmic prompts.
* **Action Success Rate:** Percentage of attempted tool calls that technically completed (no file format errors).

Would you like me to generate a Python script that reads a CSV version of this Master Sheet and automatically sends the prompts to your AI for testing?
