"""Core summarization logic for the Summarize MCP server.

Reads transcript.txt from a .tar.gz archive (produced by the transcription
tool), summarizes it with an LLM (local or cloud, via llm_client), and writes
an output archive containing the original video and a summary.txt.

Follows the same archive-in / archive-out convention as the transcription and
grep tools: smart detection of single-archive vs folder input, and cleanup of
intermediate files.
"""

import os
import shlex
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path

from llm_client import chat

logger = logging.getLogger(__name__)


def execute_command(command: str):
    logger.info(">_ " + command)
    subprocess.run(shlex.split(command), check=True)


def _read_transcript(extract_dir: str) -> str:
    path = os.path.join(extract_dir, "transcript.txt")
    if not os.path.exists(path):
        txts = list(Path(extract_dir).rglob("*.txt"))
        if not txts:
            return ""
        path = str(txts[0])
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def summarize_text(transcript: str) -> str:
    """Summarize a transcript via the configured LLM (local or cloud)."""
    transcript = transcript.strip()
    if not transcript:
        return ""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise assistant that summarizes video transcripts. "
                "Produce a clear summary capturing the main points as short bullet "
                "points. Reply in the same language as the transcript."
            ),
        },
        {"role": "user", "content": f"Summarize the following transcript:\n\n{transcript[:24000]}"},
    ]
    return chat(messages, prefix="SUMMARIZE", temperature=0.2, max_tokens=800)


def _summarize_single_archive(orig_input: str, orig_output: str) -> tuple[bool, str, str]:
    output_dir = os.path.dirname(orig_output)
    output_name = os.path.splitext(os.path.basename(orig_output))[0]
    archive_name = output_name + ".tar.gz"

    extract_dir = tempfile.mkdtemp(prefix="summarize_extract_")
    try:
        execute_command("tar -xvzf %s -C %s" % (shlex.quote(orig_input), shlex.quote(extract_dir)))

        transcript = _read_transcript(extract_dir)
        if not transcript.strip():
            return False, f"No transcript.txt found in archive {orig_input}", ""

        summary = summarize_text(transcript)
        if not summary.strip():
            return False, "LLM returned an empty summary", ""

        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary + "\n")

        # Carry the original video forward if present.
        files_to_archive = ["summary.txt"]
        mp4s = list(Path(extract_dir).rglob("*.mp4"))
        if mp4s:
            video_name = os.path.basename(str(mp4s[0]))
            shutil.copy(str(mp4s[0]), os.path.join(output_dir, video_name))
            files_to_archive.append(video_name)

        archive_full_path = os.path.join(output_dir, archive_name)
        execute_command(
            "tar -czvf %s -C %s %s" % (
                shlex.quote(archive_full_path), shlex.quote(output_dir),
                " ".join(shlex.quote(f) for f in files_to_archive))
        )
        for f in files_to_archive:
            fp = os.path.join(output_dir, f)
            if os.path.exists(fp):
                os.remove(fp)
        return True, "", summary

    except subprocess.CalledProcessError as e:
        return False, f"Archive operation failed: {e}", ""
    except Exception as e:
        logger.error("Summarization failed: %s", e, exc_info=True)
        return False, f"Summarization failed: {e}", ""
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def summarize_archive(input_path: str, output_path: str) -> tuple[bool, str, str]:
    """Summarize a single .tar.gz archive or all archives in a folder.

    Returns (success, error, summary_text).
    """
    if os.path.isfile(input_path) and input_path.lower().endswith(".tar.gz"):
        logger.info("Processing single archive: %s", input_path)
        return _summarize_single_archive(input_path, output_path)
    elif os.path.isdir(input_path):
        logger.info("Processing folder with multiple archives: %s", input_path)
        os.makedirs(output_path, exist_ok=True)
        # Accumulate EVERY archive's summary: the old code kept only the last
        # one (scalar overwritten each loop), so a folder of N clips returned a
        # single arbitrary summary. Skip-and-continue on failures (like the grep
        # tool) instead of aborting and discarding already-completed work.
        summaries = []
        for name in os.listdir(input_path):
            if not name.lower().endswith(".tar.gz"):
                logger.warning("Skipping non-tar.gz file: %s", name)
                continue
            in_path = os.path.join(input_path, name)
            out_path = os.path.splitext(os.path.join(output_path, name))[0]
            ok, err, summary = _summarize_single_archive(in_path, out_path)
            if not ok:
                logger.error("Failed to process %s: %s", name, err)
                continue
            summaries.append(f"[{name}]\n{summary}")
        if not summaries:
            return False, "No archives were successfully processed", ""
        return True, "", "\n\n".join(summaries)
    return False, f"Invalid input: must be a .tar.gz archive or a folder. Got: {input_path}", ""
