import sys
import subprocess
import shlex
import os
import uuid

import logging

logger = logging.getLogger(__name__)

def execute_command(command: str) -> bool:
    logger.info(">_ " + command)
    try:
        result = subprocess.run(
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Command failed (code %d): %s\n%s", e.returncode, command, e.stderr.decode(errors='ignore'))
        return False
    except Exception as e:
        logger.error("Command execution exception: %s", e)
        return False

def _compress_single_video_file(orig_input: str, orig_output: str) -> tuple[bool, str]:
    """Internal helper: Compress a single .mp4 video file to archive.
    
    Args:
        orig_input: Path to single .mp4 file
        orig_output: Output base path (without extension)
        
    Returns:
        tuple[bool, str]: Success flag and error message (if any)
    """
    logger.info("SCRIPT: Input at '{}', saving output in '{}'".format(orig_input, orig_output))

    output_dir = os.path.dirname(orig_output)
    output_name = os.path.basename(orig_output)

    temp_audio_path = os.path.join(output_dir, "temp_" + output_name + ".wav")
    output_audio_path = orig_output + ".wav"
    output_clip_path = orig_output + ".mp4"
    output_zip_path = orig_output + ".tar.gz"

    output_audio_name = output_name + ".wav"
    output_clip_name = output_name + ".mp4"
    output_zip_name = output_name + ".tar.gz"
    
    # extract audio track — check every step: silently returning success on a
    # failed ffmpeg/tar produced a phantom output_path (e.g. an mp4 with no audio
    # track) that the orchestrator then fed downstream. shlex.quote so paths with
    # spaces/quotes survive execute_command's shlex.split.
    qi, qta, qoa, qoc, qz, qd = (shlex.quote(p) for p in (
        orig_input, temp_audio_path, output_audio_path,
        output_clip_path, output_zip_path, output_dir))
    if not execute_command("ffmpeg -i %s -map 0:a %s" % (qi, qta)):
        return False, "Audio extraction failed for %s" % orig_input

    # down-sample audio track
    if not execute_command("ffmpeg -i %s -vn -ar 16000 -ac 1 %s" % (qta, qoa)):
        return False, "Audio down-sampling failed for %s" % orig_input

    # compress video file
    if not execute_command("ffmpeg -i %s -vcodec libx264 -crf 30 %s" % (qi, qoc)):
        return False, "Video compression failed for %s" % orig_input

    # repack everything, remove unzipped outputs (incl. the temp .wav, which was
    # leaking full-size into the returned folder and inflating output_size)
    if not execute_command("tar -czvf %s -C %s %s %s" % (
            qz, qd, shlex.quote(output_audio_name), shlex.quote(output_clip_name))):
        return False, "Archive creation failed for %s" % orig_input
    execute_command("rm " + qoa)
    execute_command("rm " + qoc)
    execute_command("rm " + qta)

    return True, ""


def compress_video_and_audio_to_archive(input_path: str, orig_output: str) -> tuple[bool, str]:
    """Compress video and audio - accepts both single .mp4 files and folders containing .mp4 files.
    
    Intelligently detects input type:
    - If input_path is a single .mp4 file: processes that file
    - If input_path is a folder: processes all .mp4 files in the folder
    
    Args:
        input_path: Path to either a single .mp4 file or a folder containing .mp4 files
        orig_output: Output base path (without extension). For folders, this is the output directory.
        
    Returns:
        tuple[bool, str]: Success flag and error message (if any)
    """
    if os.path.isfile(input_path) and input_path.lower().endswith(".mp4"):
        # Single file mode
        logger.info(f"Processing single file: {input_path}")
        return _compress_single_video_file(input_path, orig_output)
    elif os.path.isdir(input_path):
        # Folder mode - process all .mp4 files
        logger.info(f"Processing folder with multiple files: {input_path}")
        os.makedirs(orig_output, exist_ok=True)
        for input_video in os.listdir(input_path):
            if not input_video.lower().endswith(".mp4"):
                logger.warning(f"Skipping non-mp4 file: {input_video}")
                continue
            input_video_path = os.path.join(input_path, input_video)
            output_path = os.path.join(orig_output, input_video)
            output_path = os.path.splitext(output_path)[0]
            success, error = _compress_single_video_file(input_video_path, output_path)
            if not success:
                return False, error
        return True, ""
    else:
        error_msg = f"Invalid input: must be a single .mp4 file or a folder containing .mp4 files. Got: {input_path}"
        logger.error(error_msg)
        return False, error_msg
