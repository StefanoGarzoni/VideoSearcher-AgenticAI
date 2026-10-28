import subprocess
import os
import shlex
import re
import json
import tempfile
import shutil

import logging

logger = logging.getLogger(__name__)

def execute_command(command):
    logger.info(">_ " + command)
    subprocess.run(shlex.split(command), check=True)


def _search_single_archive(orig_input: str, orig_output: str, search_pattern: str, case_sensitive: bool = False) -> tuple[bool, str, dict]:
    """Internal helper: Search for pattern in a single .tar.gz archive containing transcript.txt."""

    logger.info("Grep: Input at '%s', saving output in '%s', pattern: '%s'", orig_input, orig_output, search_pattern)

    output_dir = os.path.dirname(orig_output)
    output_name = os.path.splitext(os.path.basename(orig_output))[0]
    archive_name = output_name + ".tar.gz"

    def _normalize(s: str) -> str:
        return re.sub(r"\W+", " ", s.lower()).strip()

    # Extract to a temp dir — the input archive may be on a read-only mount.
    extract_tmp = tempfile.mkdtemp(prefix="grep_extract_")
    search_results: dict = {}
    try:
        execute_command("tar -xf %s -C %s" % (shlex.quote(orig_input), shlex.quote(extract_tmp)))
        transcript_path = os.path.join(extract_tmp, "transcript.txt")

        if not os.path.exists(transcript_path):
            error_msg = f"No transcript.txt found in archive {orig_input}"
            logger.error(error_msg)
            return False, error_msg, {}

        search_results = {
            "matches": [],
            "match_count": 0,
            "pattern": search_pattern,
            "case_sensitive": case_sensitive,
            "input_archive": orig_input,
        }

        with open(transcript_path, "r", encoding="utf-8") as fh:
            content = fh.read()

        normalized_query = _normalize(search_pattern)
        # For case-sensitive search the regex must be built from the ORIGINAL
        # casing — _normalize lowercases, so with flags=0 an uppercase pattern
        # ("NASA") could never match.
        regex_source = (re.sub(r"\W+", " ", search_pattern).strip()
                        if case_sensitive else normalized_query)
        query_words = regex_source.split() if regex_source else []
        if not query_words:
            error_msg = f"Empty search pattern after normalization: '{search_pattern}'"
            logger.error(error_msg)
            return False, error_msg, {}

        flexible_regex = r"\W+".join(re.escape(w) for w in query_words)
        flags = 0 if case_sensitive else re.IGNORECASE
        # The tool contract says "supports regex": if the caller passed real
        # metacharacters, honor them; otherwise fall back to the flexible,
        # whitespace-tolerant word match. Bare "engine|thrust" used to be
        # escaped into the literal phrase "engine\W+thrust" (matching neither).
        pattern = None
        if any(c in r".^$*+?{}[]|()\\" for c in search_pattern):
            try:
                pattern = re.compile(search_pattern, flags)
                flexible_regex = search_pattern
            except re.error:
                pattern = None
        if pattern is None:
            try:
                pattern = re.compile(flexible_regex, flags)
            except re.error as e:
                error_msg = f"Could not build regex from '{search_pattern}': {str(e)}"
                logger.error(error_msg)
                return False, error_msg, {}

        matches = []
        for m in pattern.finditer(content):
            start_pos = max(0, m.start() - 50)
            end_pos = min(len(content), m.end() + 50)
            matches.append({
                "match_text": m.group(0),
                "context": content[start_pos:end_pos].strip(),
                "position": m.start(),
                "match_type": "literal",
            })

        # Fuzzy fallback when literal search finds nothing.
        if not matches:
            try:
                from rapidfuzz import fuzz
                FUZZY_THRESHOLD = 80
                cursor = 0
                for line in content.splitlines(keepends=True):
                    stripped = line.rstrip("\n")
                    if stripped.strip():
                        score = fuzz.WRatio(_normalize(stripped), normalized_query)
                        if score >= FUZZY_THRESHOLD:
                            matches.append({
                                "match_text": stripped.strip(),
                                "context": stripped.strip(),
                                "position": cursor,
                                "match_type": "fuzzy",
                                "fuzzy_score": int(score),
                            })
                    cursor += len(line)
                if matches:
                    logger.info("Fuzzy fallback matched %d line(s)", len(matches))
            except ImportError:
                logger.warning("rapidfuzz not installed — fuzzy fallback unavailable")

        search_results["matches"] = matches
        search_results["match_count"] = len(matches)
        search_results["normalized_query"] = normalized_query
        search_results["flexible_regex"] = flexible_regex
        logger.info("Found %d matches for pattern '%s'", len(matches), search_pattern)

    except Exception as e:
        error_msg = f"Error processing archive {orig_input}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, {}
    finally:
        shutil.rmtree(extract_tmp, ignore_errors=True)

    # Write results JSON
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "search_results.json")
    with open(results_file, "w", encoding="utf-8") as fh:
        json.dump(search_results, fh, indent=2)

    # Optionally include the .mp4 sibling if it exists alongside the archive (writable path only).
    video_path = orig_input.replace(".tar.gz", ".mp4")
    files_to_archive = ["search_results.json"]
    if os.path.exists(video_path):
        output_video = os.path.join(output_dir, os.path.basename(video_path))
        execute_command("cp %s %s" % (shlex.quote(video_path), shlex.quote(output_video)))
        files_to_archive.append(os.path.basename(video_path))

    # Create output archive with `tar -C` instead of a process-wide os.chdir:
    # the chdir raced concurrent calls (one request's archive was built inside
    # another's directory). Quote every path for whitespace safety.
    execute_command("tar -czvf %s -C %s %s" % (
        shlex.quote(os.path.join(output_dir, archive_name)),
        shlex.quote(output_dir),
        " ".join(shlex.quote(f) for f in files_to_archive)))

    # Clean up intermediate files in output_dir
    for fname in files_to_archive:
        fp = os.path.join(output_dir, fname)
        if os.path.exists(fp):
            execute_command("rm " + shlex.quote(fp))

    return True, "", search_results


def search_transcripts(input_path: str, output_path: str, search_pattern: str, case_sensitive: bool = False) -> tuple[bool, str, list]:
    """Search for patterns in transcript files - accepts both single .tar.gz archives and folders containing archives.
    
    Intelligently detects input type:
    - If input_path is a single .tar.gz file: processes that archive
    - If input_path is a folder: processes all .tar.gz archives in the folder
    
    Args:
        input_path (str): Path to either a single .tar.gz archive OR a folder containing .tar.gz archives
        output_path (str): Output base path. For folders, this is the output directory.
        search_pattern (str): Pattern to search for (supports regex).
        case_sensitive (bool): Whether the search should be case-sensitive (default: False).
    
    Returns:
        tuple[bool, str, list]: A tuple containing (success flag, error message, list of search results).
    """
    all_results = []
    
    if os.path.isfile(input_path) and input_path.lower().endswith(".tar.gz"):
        # Single archive mode
        logger.info(f"Processing single archive: {input_path}")
        success, error, results = _search_single_archive(input_path, output_path, search_pattern, case_sensitive)
        if success:
            all_results.append(results)
        return success, error, all_results
        
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
            
            success, error, results = _search_single_archive(input_archive_path, archive_output_path, search_pattern, case_sensitive)
            if not success:
                logger.error(f"Failed to process {input_archive}: {error}")
                continue
            
            all_results.append(results)
        
        if not all_results:
            error_msg = "No archives were successfully processed"
            logger.error(error_msg)
            return False, error_msg, []
            
        return True, "", all_results
        
    else:
        error_msg = f"Invalid input: must be a single .tar.gz archive or a folder containing .tar.gz archives. Got: {input_path}"
        logger.error(error_msg)
        return False, error_msg, []
