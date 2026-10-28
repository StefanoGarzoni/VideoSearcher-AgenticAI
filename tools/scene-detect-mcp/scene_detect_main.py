"""Core scene detection logic for the Scene Detect MCP server.

Accepts a raw .mp4 OR a .tar.gz archive containing a .mp4. Uses PySceneDetect
(ContentDetector) to find cut boundaries, then writes scenes.json and returns
a structured result dict. Answers "split by scene, not just by silence".

Follows the same archive-in / JSON-out convention as the other VideoSearcher
tool servers: smart detection of single archive vs plain .mp4, temp dir
extracted and cleaned up in a finally block.
"""

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

from scenedetect import detect, ContentDetector

logger = logging.getLogger(__name__)


def _find_mp4(directory: str) -> str | None:
    """Return the first .mp4 file found in directory, or None."""
    mp4s = list(Path(directory).rglob("*.mp4"))
    if mp4s:
        return str(mp4s[0])
    return None


def detect_scenes_from_video(
    video_path: str,
    output_directory: str,
    threshold: float = 27.0,
) -> dict:
    """Detect scenes in a video file or archive.

    Args:
        video_path: Path to a raw .mp4 OR a .tar.gz archive containing a .mp4.
        output_directory: Directory where scenes.json will be written.
        threshold: ContentDetector threshold (default 27.0).

    Returns:
        dict with keys: success, output_path, num_scenes, scenes,
        threshold, processing_time_ms  (or success=False + error).
    """
    start = time.time()

    extract_dir = None
    mp4_path = None

    try:
        # ── Resolve the .mp4 path ────────────────────────────────────────────
        if video_path.lower().endswith(".tar.gz"):
            extract_dir = tempfile.mkdtemp(prefix="scene_detect_extract_")
            import subprocess
            # List args, not shlex on an f-string: a path with spaces/quotes
            # would be split into multiple argv entries and fail on valid files.
            subprocess.run(
                ["tar", "-xzf", video_path, "-C", extract_dir],
                check=True,
            )
            mp4_path = _find_mp4(extract_dir)
            if not mp4_path:
                return {
                    "success": False,
                    "error": f"No .mp4 found inside archive: {video_path}",
                }
            logger.info("Extracted .mp4 from archive: %s", mp4_path)
        elif video_path.lower().endswith(".mp4"):
            mp4_path = video_path
        else:
            # Try treating it as a .mp4 anyway (some paths lack extension)
            mp4_path = video_path

        if not os.path.exists(mp4_path):
            return {"success": False, "error": f"Video file not found: {mp4_path}"}

        # ── Run PySceneDetect ────────────────────────────────────────────────
        logger.info("Running ContentDetector (threshold=%.1f) on %s", threshold, mp4_path)
        scene_list = detect(mp4_path, ContentDetector(threshold=threshold))

        scenes = [
            {
                "scene_index": i,
                "start_seconds": start_tc.get_seconds(),
                "end_seconds": end_tc.get_seconds(),
            }
            for i, (start_tc, end_tc) in enumerate(scene_list)
        ]

        # ── Write scenes.json ────────────────────────────────────────────────
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, "scenes.json")
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "num_scenes": len(scenes),
                    "threshold": threshold,
                    "scenes": scenes,
                },
                fh,
                indent=2,
            )

        processing_time_ms = int((time.time() - start) * 1000)
        logger.info(
            "Detected %d scenes in %d ms (threshold=%.1f)",
            len(scenes),
            processing_time_ms,
            threshold,
        )

        return {
            "success": True,
            "output_path": output_path,
            "num_scenes": len(scenes),
            "scenes": scenes,
            "threshold": threshold,
            "processing_time_ms": processing_time_ms,
        }

    except Exception as exc:
        logger.error("Scene detection failed: %s", exc, exc_info=True)
        return {"success": False, "error": f"Scene detection failed: {exc}"}

    finally:
        if extract_dir is not None:
            shutil.rmtree(extract_dir, ignore_errors=True)
