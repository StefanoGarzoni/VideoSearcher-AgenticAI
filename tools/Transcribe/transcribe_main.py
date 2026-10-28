import sys
import subprocess
import os
import shlex
import uuid
from pathlib import Path

import logging
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

def execute_command(command):
    logger.info(">_ " + command)
    subprocess.run(shlex.split(command), check=True)


def _transcribe_single_archive(orig_input: str, orig_output: str, device: str = "auto") -> tuple[bool, str]:
    """Internal helper: Transcribe a single .tar.gz archive containing audio.
    
    Args:
        orig_input (str): Path to the input .tar.gz archive containing audio files.
        orig_output (str): Path to the output .tar.gz archive to save results.
        device (str): Device to use - "auto" (default), "cuda", or "cpu". 
                     "auto" will try CUDA and fall back to CPU if unavailable.
    Returns:
        tuple[bool, str]: A tuple containing a success flag and an error message (if any).
    """

    logger.info("Faster_Whisper: Input at '%s', saving output in '%s'", orig_input, orig_output)

    import tempfile
    import shutil

    output_dir = os.path.dirname(orig_output)
    output_name = os.path.basename(orig_output)
    # Remove any file type extensions for naming
    output_name = os.path.splitext(output_name)[0]

    archive_name = output_name + ".tar.gz"
    video_name = output_name + ".mp4"
    video_path = os.path.join(output_dir, video_name)

    # Extract into a private temp dir so we never pollute the source directory.
    # The finally block below guarantees cleanup even if transcription fails.
    extract_dir = tempfile.mkdtemp(prefix="transcribe_extract_")
    try:
        execute_command("tar -xvzf %s -C %s" % (shlex.quote(orig_input), shlex.quote(extract_dir)))

        # ffmpeg-0 archives always extract as audio.wav / video.mp4
        audio_path = os.path.join(extract_dir, "audio.wav")
        if not os.path.exists(audio_path):
            wav_files = list(Path(extract_dir).rglob("*.wav"))
            if not wav_files:
                error_msg = f"No .wav file found in {extract_dir} after extracting {orig_input}"
                logger.error(error_msg)
                return False, error_msg
            audio_path = str(wav_files[0])
            logger.warning("audio.wav not at expected path; using %s", audio_path)

        extracted_video_path = os.path.join(extract_dir, "video.mp4")
        if not os.path.exists(extracted_video_path):
            mp4_files = list(Path(extract_dir).rglob("*.mp4"))
            if not mp4_files:
                error_msg = f"No .mp4 file found in {extract_dir} after extracting {orig_input}"
                logger.error(error_msg)
                return False, error_msg
            extracted_video_path = str(mp4_files[0])
            logger.warning("video.mp4 not at expected path; using %s", extracted_video_path)

        # Faster-Whisper transcription here:
        # "small" is multilingual (99 languages, ~244M params, real-time on CPU).
        # Use "small.en" for English-only audio if you want a small accuracy boost.
        model_size = "small"  # multilingual model

        try:
            # Try to initialize model with requested device
            model = None
            device_to_use = device

            if device == "auto":
                try:
                    logger.info("Attempting to use CUDA device...")
                    try:
                        model = WhisperModel(model_size, device="cuda", compute_type="float16", local_files_only=True)
                    except (OSError, ValueError):
                        model = WhisperModel(model_size, device="cuda", compute_type="float16")
                    device_to_use = "cuda"
                    logger.info("✓ CUDA device initialized successfully")
                except (RuntimeError, ValueError, Exception) as e:
                    logger.warning("CUDA initialization failed: %s", str(e))
                    logger.info("Falling back to CPU device...")
                    try:
                        model = WhisperModel(model_size, device="cpu", compute_type="int8", local_files_only=True)
                    except (OSError, ValueError):
                        model = WhisperModel(model_size, device="cpu", compute_type="int8")
                    device_to_use = "cpu"
                    logger.info("✓ CPU device initialized successfully")
            else:
                compute_type = "float16" if device == "cuda" else "int8"
                try:
                    model = WhisperModel(model_size, device=device, compute_type=compute_type, local_files_only=True)
                except (OSError, ValueError):
                    model = WhisperModel(model_size, device=device, compute_type=compute_type)
                logger.info("✓ Initialized model with device: %s", device)

            logger.info("Model size: %s, Device: %s", model_size, device_to_use)
            # language=None → auto-detect. The model was chosen BECAUSE it is
            # multilingual; hardcoding "en" forced English decoding of e.g.
            # Italian audio into garbled text returned as success.
            segments, info = model.transcribe(audio_path, beam_size=5, language=None)
            segments = list(segments)

            def _fmt_ts(seconds: float) -> str:
                m = int(seconds // 60)
                s = seconds - m * 60
                return f"{m:02d}:{s:06.3f}"

            transcript_lines = [
                f"[{_fmt_ts(seg.start)}] {seg.text.strip()}" for seg in segments
            ]
            transcript = "\n".join(transcript_lines)
            logger.info(transcript)

            # Stage transcript.txt and the video inside the PRIVATE extract_dir,
            # not the shared output_dir: with a fixed name in the shared dir two
            # concurrent calls overwrote each other's transcript.txt, so one
            # archive silently packed the other call's transcript. extract_dir
            # is removed by the finally block, so no explicit rm is needed.
            staged_transcript = os.path.join(extract_dir, "transcript.txt")
            staged_video = os.path.join(extract_dir, video_name)
            with open(staged_transcript, "w") as file:
                file.write(transcript + "\n")

            execute_command("cp %s %s" % (
                shlex.quote(extracted_video_path), shlex.quote(staged_video)))
            archive_full_path = os.path.join(output_dir, archive_name)
            execute_command("tar -czvf %s -C %s %s %s" % (
                shlex.quote(archive_full_path), shlex.quote(extract_dir),
                shlex.quote(video_name), "transcript.txt"))
            return True, ""

        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            if "model" in str(e).lower() or "huggingface" in str(e).lower() or "403" in str(e):
                error_msg = (
                    "Failed to load Whisper model. This is likely due to missing internet access. "
                    "The Faster Whisper model needs to be downloaded from Hugging Face Hub on first use. "
                    "Please ensure the container has internet access or pre-download the model. "
                    f"Original error: {str(e)}"
                )
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def transcribe_audio_to_text(input_path: str, output_path: str, device: str = "auto") -> tuple[bool, str]:
    """Transcribe audio to text - accepts both single .tar.gz archives and folders containing archives.
    
    Intelligently detects input type:
    - If input_path is a single .tar.gz file: processes that archive
    - If input_path is a folder: processes all .tar.gz archives in the folder
    
    Args:
        input_path (str): Path to either a single .tar.gz archive OR a folder containing .tar.gz archives
        output_path (str): Output base path. For folders, this is the output directory.
        device (str): Device to use - "auto" (default), "cuda", or "cpu".
                     "auto" will try CUDA and fall back to CPU if unavailable.
    Returns:
        tuple[bool, str]: A tuple containing a success flag and an error message (if any).
    """
    if os.path.isfile(input_path) and input_path.lower().endswith(".tar.gz"):
        # Single archive mode
        logger.info(f"Processing single archive: {input_path}")
        return _transcribe_single_archive(input_path, output_path, device)
    elif os.path.isdir(input_path):
        # Folder mode - process all .tar.gz archives
        logger.info(f"Processing folder with multiple archives: {input_path}")
        os.makedirs(output_path, exist_ok=True)
        for input_archive in os.listdir(input_path):
            if not input_archive.lower().endswith(".tar.gz"):
                logger.warning(f"Skipping non-tar.gz file: {input_archive}")
                continue
            input_archive_path = os.path.join(input_path, input_archive)
            archive_output_path = os.path.join(output_path, input_archive)
            archive_output_path = os.path.splitext(archive_output_path)[0]
            success, error = _transcribe_single_archive(input_archive_path, archive_output_path, device)
            if not success:
                return False, error
        return True, ""
    else:
        error_msg = f"Invalid input: must be a single .tar.gz archive or a folder containing .tar.gz archives. Got: {input_path}"
        logger.error(error_msg)
        return False, error_msg

