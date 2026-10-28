#!/usr/bin/env python
"""
Simple script to test transcribe_audio_to_text with a real file.
Uses example archives from the data directory.

Usage:
    python test_transcribe_script.py              # Uses auto device (tries CUDA, falls back to CPU)
    python test_transcribe_script.py --device cpu # Forces CPU
    python test_transcribe_script.py --device cuda # Forces CUDA (may fail if not available)
"""

import os
import sys
import argparse
import tempfile
import shutil

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transcribe_main import transcribe_audio_to_text

# Example archive directory
EXAMPLE_ARCHIVE_DIR = "/workspaces/VideoSearcher-AgenticAI/data/output/ffmpeg-2/ffmpeg_output/compression_b6c8973b"

def main():
    parser = argparse.ArgumentParser(description='Test transcribe_audio_to_text with real files')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use: auto (default), cpu, or cuda')
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
    print(f"📦 Using archive: {input_archive}")
    print(f"🖥️  Device mode: {args.device}")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # NOTE: output_archive should not have .tar.gz extension since the function adds it
        output_archive = os.path.join(temp_dir, "output")
        
        print(f"📂 Output directory: {temp_dir}")
        print("\n🚀 Starting transcription...\n")
        
        try:
            # Call the transcription function with specified device
            success, error = transcribe_audio_to_text(input_archive, output_archive, device=args.device)
            
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
                        
                        transcript_path = os.path.join(extract_dir, "transcript.txt")
                        if os.path.exists(transcript_path):
                            print("\n📝 Transcript preview:")
                            with open(transcript_path, 'r') as f:
                                transcript = f.read().strip()
                                # Show first 500 characters
                                preview = transcript[:500] + ("..." if len(transcript) > 500 else "")
                                print(f"  {preview}")
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
