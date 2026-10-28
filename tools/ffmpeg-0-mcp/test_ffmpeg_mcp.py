"""
Unit tests for FFmpeg Audio Extraction MCP Server
"""

import json
import os
import tempfile

import pytest

from pathlib import Path
from unittest.mock import patch, MagicMock

from ffmpeg_0_main import execute_command, extract_audio_to_archive
from ffmpeg_0_server import extract_audio


class TestFFmpegUtils:
    """Tests for the core ffmpeg_0_main module"""
    
    def test_execute_command_success(self):
        """Test successful command execution"""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            
            success = execute_command("ffmpeg -version")
            
            assert success is True
    
    def test_execute_command_failure(self):
        """Test failed command execution"""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error")
            
            success = execute_command("ffmpeg -invalid")
            
            assert success is False
    
    def test_execute_command_exception(self):
        """Test command execution with exception"""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Test error")
            
            success = execute_command("ffmpeg -version")
            
            assert success is False
    
    def test_extract_audio_to_archive_missing_file(self):
        """Test extraction with missing input file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.tar.gz")
            
            success, error = extract_audio_to_archive("/nonexistent/video.mp4", output_path)
            
            assert success is False
            assert "not found" in error.lower()
    
    def test_extract_audio_to_archive_success(self):
        """Test successful audio extraction"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock video file
            input_video = os.path.join(tmpdir, "test.mp4")
            Path(input_video).write_bytes(b"fake video")
            
            output_path = os.path.join(tmpdir, "output.tar.gz")
            
            # Create the archive file to simulate successful execution
            Path(output_path).write_bytes(b"fake archive")
            
            with patch("ffmpeg_0_main.execute_command") as mock_execute:
                # Mock all commands to succeed
                mock_execute.return_value = True
                
                success, error = extract_audio_to_archive(input_video, output_path)
                
                assert success is True
                assert error == ""
                # Verify the function called execute_command multiple times
                assert mock_execute.call_count >= 3  # extract, copy, tar


class TestFFmpegMCPServer:
    """Tests for the MCP server wrapper"""
    
    def test_extract_audio_missing_path(self):
        """Test extract_audio with empty path"""
        result_json = extract_audio("")
        result = json.loads(result_json)
        
        assert result["success"] is False
        assert "required" in result["error"].lower()
    
    def test_extract_audio_missing_file(self):
        """Test extract_audio with non-existent file"""
        result_json = extract_audio("/nonexistent/video.mp4")
        result = json.loads(result_json)
        
        assert result["success"] is False
        assert "not found" in result["error"].lower()
    
    def test_extract_audio_success(self):
        """Test successful audio extraction via MCP"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock video file
            input_video = os.path.join(tmpdir, "test.mp4")
            Path(input_video).write_bytes(b"fake video")
            
            with (
                patch("ffmpeg_0_main.execute_command") as mock_execute,
                patch("ffmpeg_0_main.os.path.exists") as mock_exists_main,
                patch("ffmpeg_0_server.os.path.exists") as mock_exists_server,
                patch("ffmpeg_0_server.os.path.getsize") as mock_size,
            ):
                mock_execute.return_value = True
                mock_exists_main.return_value = True  # extraction util sees archive exists
                mock_exists_server.return_value = True  # server confirmation
                mock_size.return_value = 5242880

                result_json = extract_audio(input_video)
                result = json.loads(result_json)

                assert result["success"] is True
                assert "output_path" in result
                assert result["file_size"] == 5242880
                assert "processing_time_ms" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
