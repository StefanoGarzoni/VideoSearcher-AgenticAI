# Legacy Code

This folder contains deprecated code and files that are no longer part of the active system but are preserved for reference.

## Contents

### `videosearcher/`
The original video processing pipeline implementation before the transition to the agentic AI architecture with MCP servers.

**Status**: Deprecated  
**Reason**: Replaced by MCP-based agent architecture

### `public/`
Miscellaneous files (PDFs, presentations, etc.) from earlier project phases.

**Status**: Archived for reference

## Note

The current system uses:
- **Agents**: Multi-agent orchestration with OpenAI Agents SDK
- **MCP Servers**: Containerized tool servers (FFmpeg, Librosa, Transcribe, Grep)
- **StreamableHTTP**: HTTP-based communication protocol

See the main [README.md](../README.md) for current architecture.
