"""
FFmpeg audio extraction utilities

Core implementation copied from videosearcher/src/ffmpeg_0
Handles audio extraction and archive creation
"""

import subprocess
import os
import shlex
import shutil
import tempfile
import logging

logger = logging.getLogger(__name__)


def execute_command(command: str) -> bool:
    logger.info(">_ " + command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        success = result.returncode == 0
        if not success:
            logger.error(f"Command failed (code {result.returncode}): {command}\n{result.stderr.decode(errors='ignore')}")
        return success
    except Exception as e:
        logger.error(f"Command execution exception: {e}")
        return False


def extract_audio_to_archive(orig_input: str, orig_output: str) -> tuple[bool, str]:
    """
    Extract audio from video and create tar.gz archive.
    
    Creates an archive at orig_output containing:
    - video.mp4 (copy of input video)
    - audio.wav (extracted audio)
    
    Args:
        orig_input: Path to input video file
        orig_output: Path where archive should be created (e.g., /path/to/output.tar.gz)
        
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        # Validate input
        if not os.path.exists(orig_input):
            error_msg = f"Input video file not found: {orig_input}"
            logger.error(error_msg)
            return False, error_msg
        
        # Get output directory and archive name
        output_dir = os.path.dirname(orig_output)
        output_name = os.path.basename(orig_output)
        
        logger.info(f"Extracting audio from '{orig_input}' to '{orig_output}'")
        
        # Paths for intermediate files (in output directory)
        # Create archive with the exact requested name (no double extension)
        archive_name = output_name
        # Intermediates go in a private per-call dir with a '-y' ffmpeg, not the
        # shared output_dir with fixed names: a mid-pipeline failure used to
        # leave video.mp4/audio.wav behind, and ffmpeg's overwrite prompt (no
        # tty → EOF) then made EVERY later call fail forever; two overlapping
        # calls also cross-packed each other's video.
        work_dir = tempfile.mkdtemp(prefix=".extract_", dir=output_dir)
        video_path = os.path.join(work_dir, "video.mp4")
        audio_path = os.path.join(work_dir, "audio.wav")
        # shlex.quote every caller-controlled path: execute_command uses
        # shell=True, so unquoted paths break on spaces AND allow injection.
        qi, qv, qa, qo, qw = (shlex.quote(p) for p in (
            orig_input, video_path, audio_path, orig_output, work_dir))

        if not execute_command("ffmpeg -y -i %s -map 0:a %s" % (qi, qa)):
            shutil.rmtree(work_dir, ignore_errors=True)
            return False, "Audio extraction failed"

        if not execute_command("cp %s %s" % (qi, qv)):
            shutil.rmtree(work_dir, ignore_errors=True)
            return False, "Failed to copy video file"

        if not execute_command("tar -czvf %s -C %s %s %s" % (qo, qw, "video.mp4", "audio.wav")):
            shutil.rmtree(work_dir, ignore_errors=True)
            return False, "Archive creation failed"

        shutil.rmtree(work_dir, ignore_errors=True)
        
        # Verify archive was created
        if not os.path.exists(orig_output):
            return False, f"Archive file was not created: {orig_output}"
        
        file_size = os.path.getsize(orig_output)
        logger.info(f"Archive created successfully: {orig_output} ({file_size} bytes)")
        
        return True, ""
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
