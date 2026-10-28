
from pathlib import Path
import subprocess
import sys
import os
import argparse

# Dynamically compute project root and paths so users don't need to edit code
ROOT = Path(__file__).resolve().parents[2]
VS_DIR = ROOT / "videosearcher" / "src"
OUTPUT_DIR = ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_tool_script(tool_name: str) -> Path:
    """Return the main.py path for a tool inside `videosearcher/src/<tool_name>/main.py`.
    Raises FileNotFoundError if not present."""
    p = VS_DIR / tool_name / "main.py"
    if p.exists():
        return p
    raise FileNotFoundError(f"Tool script not found: {p}")


def execute_command(cmd_args):
    """Run a command (list form) with the current Python executable by default."""
    print(">_ " + " ".join(map(str, cmd_args)))
    res = subprocess.run(list(map(str, cmd_args)))
    res.check_returncode()
    return res


# Audio extraction from video
def run_ffmpeg_0(input_video_path):
    script = find_tool_script("ffmpeg-0")
    output_path = OUTPUT_DIR / "ffmpeg-0"
    output_path_str = str(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), "-i", str(input_video_path), "-o", output_path_str]
    execute_command(cmd)
    tar = Path(str(output_path) + ".tar.gz")
    return str(tar) if tar.exists() else output_path_str


# Sentence recognition from audio
def run_librosa(input_archive):
    script = find_tool_script("librosa")
    output_path = OUTPUT_DIR / "librosa"
    output_path_str = str(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), "-i", str(input_archive), "-o", output_path_str]
    execute_command(cmd)
    tar = Path(str(output_path) + ".tar.gz")
    return str(tar) if tar.exists() else output_path_str


# Clip extraction from video
def run_ffmpeg_1(input_archive):
    script = find_tool_script("ffmpeg-1")
    output_folder = OUTPUT_DIR / "ffmpeg-1"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / "clip"
    cmd = [sys.executable, str(script), "-i", str(input_archive), "-o", str(output_path)]
    execute_command(cmd)
    return str(output_folder)


# Audio and video compression
def run_ffmpeg_2(input_folder):
    script = find_tool_script("ffmpeg-2")
    output_folder = OUTPUT_DIR / "ffmpeg-2"
    output_folder.mkdir(parents=True, exist_ok=True)
    for input_video in os.listdir(input_folder):
        input_video_path = Path(input_folder) / input_video
        output_path = output_folder / Path(input_video)
        output_path = output_path.with_suffix("")
        cmd = [sys.executable, str(script), "-i", str(input_video_path), "-o", str(output_path)]
        execute_command(cmd)
    return str(output_folder)


# Transcript generation from audio
def run_deepspeech(input_folder):
    script = find_tool_script("deepspeech")
    output_folder = OUTPUT_DIR / "deepspeech"
    output_folder.mkdir(parents=True, exist_ok=True)
    for input_archive in os.listdir(input_folder):
        input_archive_path = Path(input_folder) / input_archive
        output_path = output_folder / input_archive
        output_path = Path(str(output_path).rsplit(".", 2)[0])
        cmd = [sys.executable, str(script), "-i", str(input_archive_path), "-o", str(output_path)]
        execute_command(cmd)
    return str(output_folder)


def main():
    parser = argparse.ArgumentParser(description="Run the VideoSearcher pipeline end-to-end")
    parser.add_argument("-i", "--input", type=str, default=str((ROOT / "data" / "input" / "sample_video.mp4")), help="Input video path")
    parser.add_argument("--skip-missing", action="store_true", help="If set, skip steps whose tool scripts are missing instead of erroring")
    args = parser.parse_args()

    input_video = args.input

    steps = [
        ("ffmpeg-0", run_ffmpeg_0),
        ("librosa", run_librosa),
        ("ffmpeg-1", run_ffmpeg_1),
        ("ffmpeg-2", run_ffmpeg_2),
        #("deepspeech", run_deepspeech),
    ]

    res = input_video
    for name, func in steps:
        try:
            print(f"Running {name}...")
            res = func(res)
            print(f"{name} finished -> {res}")
        except FileNotFoundError as e:
            if args.skip_missing:
                print(f"Skipping {name}: {e}")
                continue
            raise


if __name__ == '__main__':
    main()


# ============================================================================
# HOW TO USE THIS PIPELINE
# ============================================================================
#
# 1. Place your input video file at: /data/input/sample_video.mp4
#    - Supported formats: .mp4, .avi, .mov, .webm
#
# 2. Run with default input:
#    python videosearcher/src/pipeline.py
#
# 3. Run with custom video:
#    python videosearcher/src/pipeline.py -i /path/to/your/video.mp4
#
# 4. Skip missing tool steps:
#    python videosearcher/src/pipeline.py --skip-missing
#
# Output will be saved to: /data/output/<tool-name>/
# ============================================================================
