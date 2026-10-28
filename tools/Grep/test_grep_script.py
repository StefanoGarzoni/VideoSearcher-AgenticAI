#!/usr/bin/env python
"""
Simple script to test grep_server functionalities with a real file.
Uses example archives from the data directory.

Usage:
    python test_grep_script.py       
"""

import os
import sys
import argparse
import tempfile
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grep_main import search_transcripts

# Example archive directory
EXAMPLE_ARCHIVE_DIR = "/workspaces/VideoSearcher-AgenticAI/data/output/deepspeech/ffmpeg_output/transcription_5650085f"
TEST_PATTERN = "variable elimination"
CASE_SENSITIVE = False

def main():
    parser = argparse.ArgumentParser(description='Test grep_server functionalities with real files')
    args = parser.parse_args()
    
    # Check if example archive directory exists
    if not os.path.exists(EXAMPLE_ARCHIVE_DIR):
        print(f"❌ Example archive directory not found: {EXAMPLE_ARCHIVE_DIR}")
        sys.exit(1)
    
    # Find first available archive
    archives = [f for f in os.listdir(EXAMPLE_ARCHIVE_DIR) if f.endswith('.tar.gz')]
    if not archives:
        print("❌ No .tar.gz archives found in example directory")
        sys.exit(1)
    
    input_archive = os.path.join(EXAMPLE_ARCHIVE_DIR, archives[0])
    #input_archive = os.path.join(EXAMPLE_ARCHIVE_DIR, "clip_0.tar.gz")
    print(f"📦 Using archive: {input_archive}")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # NOTE: output_archive should not have .tar.gz extension since the function adds it
        output_archive = os.path.join(temp_dir, "output")
        
        print(f"📂 Output directory: {temp_dir}")
        print("\n🚀 Starting grep search...\n")
        
        try:
            # Call the grep search function
            success, error, results = search_transcripts(input_archive, output_archive, search_pattern=TEST_PATTERN, case_sensitive=CASE_SENSITIVE)
            
            if success:
                print("\n✅ Transcription completed successfully!")
                
                # Check if output archive was created (it adds .tar.gz extension)
                output_archive_path = output_archive + ".tar.gz"
                if os.path.exists(output_archive_path):
                    size_mb = os.path.getsize(output_archive_path) / (1024 * 1024)
                    print(f"📊 Output archive size: {size_mb:.2f} MB")
                    
                    # List contents
                    import tarfile
                    print("\n📋 Contents of output archive:")
                    with tarfile.open(output_archive_path, 'r:gz') as tar:
                        for member in tar.getmembers():
                            print(f"  - {member.name} ({member.size} bytes)")
                    
                    # Extract and display transcript
                    with tempfile.TemporaryDirectory() as extract_dir:
                        with tarfile.open(output_archive_path, 'r:gz') as tar:
                            tar.extractall(extract_dir)
                        
                        # Check for search results JSON
                        results_path = os.path.join(extract_dir, "search_results.json")
                        if os.path.exists(results_path):
                            print("\n📊 Search results:")
                            with open(results_path, 'r') as f:
                                search_results = json.load(f)
                                matches = search_results.get('matches', [])
                                if not matches:
                                    print("⚠️  No matches found.")
                                else:
                                    print(f"  Total matches: {len(matches)}")
                                    for i, result in enumerate(matches[:2], 1):
                                        print(f"  [{i}] {result}")
                                    if len(matches) > 2:
                                        print(f"  ... and {len(matches) - 2} more matches")
                        else:
                            print(f"⚠️  search_results.json not found in archive")
                else:
                    print(f"⚠️  Output archive not found at {output_archive_path}")
                    print(f"📂 Files in output directory: {os.listdir(temp_dir)}")
            else:
                print(f"\n❌ Transcription failed: {error}")
                sys.exit(1)
                
        except Exception as e:
            print(f"\n❌ Error during transcription: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()
