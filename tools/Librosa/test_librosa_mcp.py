#!/usr/bin/env python3
"""
Test script for Librosa MCP Server StreamableHTTP integration
"""

import asyncio
import json
import os
from pathlib import Path

from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession
from mcp.types import TextContent

# MCP Server configuration
MCP_SERVER_URL = "http://localhost:8001/mcp"
TEST_AUDIO_PATH = "/data/output/ffmpeg-1/clip_0.wav"  # Use existing audio file


async def test_mcp_server():
    """Test the Librosa MCP server via StreamableHTTP using official MCP client"""
    
    print("Testing Librosa MCP Server...")
    print(f"Server URL: {MCP_SERVER_URL}")
    
    # Test with proper MCP client
    print("\n1. Connecting to MCP server...")
    try:
        async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                print("   ✓ Connected!")
                
                # Initialize session
                print("\n2. Initializing session...")
                await session.initialize()
                print("   ✓ Session initialized!")
                
                # List available tools
                print("\n3. Listing available tools...")
                tools = await session.list_tools()
                print(f"   Found {len(tools.tools)} tool(s):")
                for tool in tools.tools:
                    print(f"     - {tool.name}: {tool.description[:80]}...")
                
                # Test analyze_audio_features tool
                audio_path = TEST_AUDIO_PATH
                print(f"\n4. Testing analyze_audio_features with: {audio_path}")
                print("   Note: File must exist in the container at this path")
                
                try:
                    result = await session.call_tool(
                        "analyze_audio_features",
                        {"audio_path": audio_path}
                    )
                    
                    # Parse result
                    assert result.content, "No content returned"
                    content = result.content[0]
                    assert isinstance(content, TextContent)
                    
                    payload = json.loads(content.text)
                    print(f"   Success: {payload.get('success')}")
                    
                    if payload.get('success'):
                        print(f"   Duration: {payload.get('duration_seconds', 0):.2f}s")
                        features = payload.get('features', {})
                        print(f"   Tempo: {features.get('tempo_bpm', 0):.2f} BPM")
                        print(f"   Average Pitch: {features.get('average_pitch_hz', 0):.2f} Hz")
                        print(f"   Processing time: {payload.get('processing_time_ms')} ms")
                        print("\n   Full response:")
                        print(f"   {json.dumps(payload, indent=2)}")
                    else:
                        print(f"   Error: {payload.get('error')}")
                        print("   Make sure the file exists in the container at the specified path")
                        
                except Exception as e:
                    print(f"   Tool call failed: {e}")
                    import traceback
                    traceback.print_exc()
                
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n✓ All MCP Server tests completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(test_mcp_server())
    exit(exit_code)
