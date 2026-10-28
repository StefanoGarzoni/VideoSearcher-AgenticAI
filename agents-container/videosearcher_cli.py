#!/usr/bin/env python3
r"""
VideoSearcher AgenticAI - CLI Entry Point
==========================================

This is the main entry point for interacting with the VideoSearcher AgenticAI system.
The system uses a multi-agent orchestrator to coordinate video processing tasks through
specialized agents and MCP (Model Context Protocol) servers.

================================================================================
COMPLETE SETUP GUIDE
================================================================================

STEP 1: SYSTEM REQUIREMENTS
----------------------------
- Docker & Docker Compose (verified with: docker --version)

If you run the system fully in Docker (recommended), you do NOT need local
FFmpeg, ai-sprint, or Python dependencies. The containers include everything.

If you run the agents locally (no Docker), then you need:
- Python 3.10+ (verified with: python --version)
- FFmpeg CLI tool (verified with: ffmpeg -version)

STEP 2: LIBRARY INSTALLATION (LOCAL RUN ONLY)
----------------------------------------------
Skip this section if you run the agents in Docker.

1. Clone the repository:
    $ git clone https://github.com/SanteriPaljakka/videosearcher_agenticai.git
    $ cd videosearcher_agenticai

2. Set up Python environment:
    $ python -m venv .venv
    $ source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install ai-sprint (required before requirements.txt):
    $ python -m pip install cookiecutter==2.1.1
    $ git clone https://gitlab.polimi.it/ai-sprint/ai-sprint-design.git
    $ cd ai-sprint-design
    $ python -m pip install .
    $ cd ..

4. Install remaining Python dependencies:
    $ pip install -r requirements.txt

5. Set up data directories:
    $ mkdir -p data/{raw,audio_extraction,compression,transcription,audio_analysis,video_clips,search_results}

STEP 3: START MCP SERVERS (from STREAMABLE_HTTP_QUICK_START.md)
----------------------------------------------------------------
All specialized agents communicate with MCP servers via HTTP. You MUST start these servers first:

1. Start all MCP servers:
   $ docker-compose -f docker-compose.yml up

2. Verify servers are running (in another terminal):
   $ curl http://localhost:8000/health  # FFmpeg-0 (audio extraction)
   $ curl http://localhost:8001/health  # Librosa (audio analysis)
   $ curl http://localhost:8002/health  # FFmpeg-1 (video splitting)
   $ curl http://localhost:8003/health  # FFmpeg-2 (compression)
   $ curl http://localhost:8004/health  # Transcribe (speech-to-text)
   $ curl http://localhost:8005/health  # Grep (pattern search)

   Expected response for each: {"status": "healthy", "service": "..."}

STEP 4: API KEY CONFIGURATION (CHOOSE ONE OPTION)
--------------------------------------------------

OPTION A: Use Gemini (Recommended - Free Tier Available)
---------------------------------------------------------
1. Get API key: https://aistudio.google.com/app/apikey

2. Set environment variables:
   $ export LLM_API_KEY="your-gemini-api-key"
   $ export LLM_MODEL="gemini/gemini-2.5-flash"

3. (Optional) Use different model for orchestrator:
   $ export LLM_ORCHESTRATOR_MODEL="gemini/gemini-2.0-flash-exp"

OPTION B: Use OpenAI
--------------------
1. Get API key: https://platform.openai.com/api-keys

2. Set environment variables:
   $ export LLM_API_KEY="your-openai-api-key"
   $ export LLM_MODEL="gpt-4o-mini"

3. (Optional) Use GPT-4 for orchestrator (better reasoning):
   $ export LLM_ORCHESTRATOR_MODEL="gpt-4o"

OPTION C: Use a Local LLM Endpoint (Orchestrator Only)
------------------------------------------------------
Use this when you already have a local OpenAI-compatible server running.
Specialized agents still need a cloud API key for tool calling.

1. Set environment variables:
    # Orchestrator uses the local endpoint
    $ export LLM_ORCHESTRATOR_PROVIDER="openai-chat"
    $ export LLM_ORCHESTRATOR_BASE_URL="http://localhost:9000/v1"
    $ export LLM_ORCHESTRATOR_MODEL="local-model"
   
    # Specialized agents use cloud API (required for tool calling)
    $ export LLM_API_KEY="your-gemini-or-openai-key"
    $ export LLM_MODEL="gemini/gemini-2.5-flash"

2. Verify the endpoint is running:
    $ curl http://localhost:9000/health

STEP 5: PLACE VIDEO FILES
--------------------------
Place your video files in the data/raw/ directory:
   $ cp your_video.mp4 data/raw/sample_video.mp4

STEP 6: RUN THIS CLI SCRIPT
----------------------------
   $ python videosearcher_cli.py

   Or make it executable:
   $ chmod +x videosearcher_cli.py
   $ ./videosearcher_cli.py

================================================================================
USAGE EXAMPLES
================================================================================

Once in the CLI, you can enter natural language requests:

Example 1: Extract audio
   > Extract audio from /data/raw/sample_video.mp4

Example 2: Transcribe video
   > Transcribe the video at /data/raw/lecture.mp4

Example 3: Split video into clips
   > Split /data/raw/presentation.mp4 into clips at natural break points

Example 4: Search for content
   > Find all mentions of "machine learning" in /data/raw/tutorial.mp4

Example 5: Complete pipeline
   > I want to find "artificial intelligence" in /data/raw/video.mp4. 
     Extract audio, transcribe it, and search for the pattern.

================================================================================
TROUBLESHOOTING
================================================================================

❌ "Error: LLM_API_KEY environment variable not set"
   → Set API key: export LLM_API_KEY="your-key"

❌ "Connection refused" or "MCP server not responding"
   → Start MCP servers: docker-compose up
   → Check servers: curl http://localhost:8000/health

❌ "ModuleNotFoundError: No module named 'agents'"
   → Install dependencies: pip install -r requirements.txt

❌ "File not found: /data/raw/video.mp4"
   → Place video files in data/raw/ directory
   → Create directories: mkdir -p data/raw

❌ Local LLM not working
   → Verify server: curl http://localhost:9000/health
   → Check model exists in local-ai-service/models/
   → Restart server: cd local-ai-service && ./run_server.sh

================================================================================
SYSTEM ARCHITECTURE
================================================================================

User (this CLI) → Orchestrator Agent → Handoff Tools → Specialized Agents → MCP Servers

Specialized Agents:
1. AudioExtractionAgent - Extract audio from videos
2. AudioAnalysisAgent - Detect silence, generate timestamps
3. VideoSplittingAgent - Split videos into clips
4. CompressionAgent - Compress video/audio
5. TranscriptionAgent - Speech-to-text transcription
6. InformationFinderAgent - Search patterns in transcripts

Data Flow:
raw/ → audio_extraction/ → audio_analysis/ → video_clips/
                         → compression/ → transcription/ → search_results/

================================================================================
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from urllib.parse import urlparse, urlunparse

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from videosearcher_agents.orchestrator.main_agent import OrchestratorAgent


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_banner():
    """Print welcome banner"""
    banner = f"""
{Colors.OKBLUE}╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║           🎬 VideoSearcher AgenticAI System 🤖                       ║
║                                                                      ║
║        Multi-Agent Video Processing with MCP Servers                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
    print(banner)


def check_environment():
    """Check if required environment variables and services are configured"""
    issues = []
    warnings = []

    data_root = get_data_root()
    
    # Check LLM model first to determine which API key to check
    llm_model = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash")
    
    # Check LLM API key - support provider-specific keys based on model prefix
    # LiteLLM auto-detects: GEMINI_API_KEY for gemini/*, OPENAI_API_KEY for gpt-*
    llm_api_key = os.getenv("LLM_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Check if using local LLM for orchestrator
    orchestrator_base_url = os.getenv("LLM_ORCHESTRATOR_BASE_URL")
    use_local_orchestrator = bool(orchestrator_base_url)

    api_key_found = False
    api_key_source = None
    
    if llm_api_key:
        api_key_found = True
        api_key_source = "LLM_API_KEY"
    elif llm_model.startswith("gemini/") and gemini_api_key:
        api_key_found = True
        api_key_source = "GEMINI_API_KEY"
    elif (llm_model.startswith("gpt-") or llm_model.startswith("openai/")) and openai_api_key:
        api_key_found = True
        api_key_source = "OPENAI_API_KEY"
    
    if not api_key_found and not use_local_orchestrator:
        if llm_model.startswith("gemini/"):
            issues.append("GEMINI_API_KEY (or LLM_API_KEY) environment variable not set")
        elif llm_model.startswith("gpt-") or llm_model.startswith("openai/"):
            issues.append("OPENAI_API_KEY (or LLM_API_KEY) environment variable not set")
        else:
            issues.append("LLM_API_KEY environment variable not set")
    
    # Check data directories
    data_dirs = ["raw", "audio_extraction", "compression", "transcription",
                 "audio_analysis", "video_clips", "search_results"]
    
    missing_dirs = []
    for dir_name in data_dirs:
        dir_path = Path(data_root) / dir_name
        if not dir_path.exists():
            missing_dirs.append(f"{data_root}/{dir_name}")
    
    if missing_dirs:
        warnings.append(f"Missing data directories: {', '.join(missing_dirs)}")
        warnings.append("Create them with: mkdir -p " + " ".join(missing_dirs))
    
    # Print results
    if use_local_orchestrator:
        print(f"{Colors.OKGREEN}✅ Configuration: Local LLM for orchestrator + Cloud API for agents{Colors.ENDC}")
        print(f"   Orchestrator: {orchestrator_base_url}")
        print(f"   Agents: {llm_model}")
        if api_key_source:
            print(f"   API Key: {api_key_source} ✓")
    else:
        print(f"{Colors.OKGREEN}✅ Configuration: Cloud LLM{Colors.ENDC}")
        print(f"   Model: {llm_model}")
        if api_key_source:
            print(f"   API Key: {api_key_source} ✓")
        if os.getenv("LLM_ORCHESTRATOR_MODEL"):
            print(f"   Orchestrator Model: {os.getenv('LLM_ORCHESTRATOR_MODEL')}")
    
    print()
    
    if issues:
        print(f"{Colors.FAIL}❌ Configuration Issues:{Colors.ENDC}")
        for issue in issues:
            print(f"   • {issue}")
        print()
        print(f"{Colors.WARNING}📖 See setup instructions at the top of this file{Colors.ENDC}")
        return False
    
    if warnings:
        print(f"{Colors.WARNING}⚠️  Warnings:{Colors.ENDC}")
        for warning in warnings:
            print(f"   • {warning}")
        print()
    
    return True


def print_help():
    """Print help information"""
    help_text = f"""
{Colors.BOLD}Available Commands:{Colors.ENDC}

{Colors.OKCYAN}help{Colors.ENDC}     - Show this help message
{Colors.OKCYAN}examples{Colors.ENDC} - Show example requests
{Colors.OKCYAN}status{Colors.ENDC}   - Check system status
{Colors.OKCYAN}config{Colors.ENDC}   - Show current configuration
{Colors.OKCYAN}quit{Colors.ENDC}     - Exit the CLI
{Colors.OKCYAN}exit{Colors.ENDC}     - Exit the CLI

{Colors.BOLD}Natural Language Requests:{Colors.ENDC}

Just type what you want to do! The orchestrator will figure out which agents to use.

{Colors.OKGREEN}Example:{Colors.ENDC} Extract audio from /data/raw/sample_video.mp4
{Colors.OKGREEN}Example:{Colors.ENDC} Transcribe the video at /data/raw/lecture.mp4
{Colors.OKGREEN}Example:{Colors.ENDC} Split /data/raw/video.mp4 into clips
{Colors.OKGREEN}Example:{Colors.ENDC} Find "AI" in /data/raw/tutorial.mp4
"""
    print(help_text)


def print_examples():
    """Print example requests"""
    examples = f"""
{Colors.BOLD}Example Requests:{Colors.ENDC}

{Colors.OKCYAN}1. Audio Extraction:{Colors.ENDC}
   Extract audio from /data/raw/sample_video.mp4

{Colors.OKCYAN}2. Transcription:{Colors.ENDC}
   Transcribe the video at /data/raw/lecture.mp4
   (Automatically chains: audio extraction → transcription)

{Colors.OKCYAN}3. Video Segmentation:{Colors.ENDC}
   Split /data/raw/presentation.mp4 into clips at natural break points
   (Chains: audio extraction → silence detection → video splitting)

{Colors.OKCYAN}4. Content Search:{Colors.ENDC}
   Find all mentions of "machine learning" in /data/raw/tutorial.mp4
   (Chains: audio extraction → transcription → pattern search)

{Colors.OKCYAN}5. Compression:{Colors.ENDC}
   Compress the video at /data/raw/podcast.mp4

{Colors.OKCYAN}6. Multi-Step Pipeline:{Colors.ENDC}
   I want to find "artificial intelligence" in /data/raw/video.mp4. 
   Extract audio, transcribe it, and search for the pattern.

{Colors.OKCYAN}7. Custom Pipeline:{Colors.ENDC}
   Extract audio from /data/raw/video.mp4, compress it, then transcribe
"""
    print(examples)


def show_config():
    """Show current configuration"""
    data_root = get_data_root()
    config = f"""
{Colors.BOLD}Current Configuration:{Colors.ENDC}

{Colors.OKCYAN}LLM Configuration:{Colors.ENDC}
  API Key: {Colors.OKGREEN}{'✓ Set' if os.getenv('LLM_API_KEY') else Colors.FAIL + '✗ Not set'}{Colors.ENDC}
  Model: {os.getenv('LLM_MODEL', 'gemini/gemini-2.5-flash')}
  Provider: {os.getenv('LLM_PROVIDER', 'litellm')}
  
{Colors.OKCYAN}Orchestrator LLM:{Colors.ENDC}
  Model: {os.getenv('LLM_ORCHESTRATOR_MODEL', 'Same as agents')}
  Provider: {os.getenv('LLM_ORCHESTRATOR_PROVIDER', 'Same as agents')}
  Base URL: {os.getenv('LLM_ORCHESTRATOR_BASE_URL', 'Default')}

{Colors.OKCYAN}MCP Server URLs:{Colors.ENDC}
  FFmpeg-0 (Audio Extraction): {os.getenv('FFMPEG_0_MCP_SERVER_URL', 'http://localhost:8000/mcp')}
  FFmpeg-1 (Video Splitting): {os.getenv('FFMPEG_1_MCP_SERVER_URL', 'http://localhost:8002/mcp')}
  FFmpeg-2 (Compression): {os.getenv('FFMPEG_2_MCP_SERVER_URL', 'http://localhost:8003/mcp')}
  Librosa (Audio Analysis): {os.getenv('LIBROSA_MCP_SERVER_URL', 'http://localhost:8001/mcp')}
  Transcribe (Speech-to-Text): {os.getenv('SPEECH_TO_TEXT_MCP_SERVER_URL', 'http://localhost:8004/mcp')}
  Grep (Pattern Search): {os.getenv('GREP_MCP_SERVER_URL', 'http://localhost:8005/mcp')}

{Colors.OKCYAN}Data Directories:{Colors.ENDC}
  Input: {data_root}/raw/
  Output Stages: audio_extraction/, compression/, transcription/, 
                 audio_analysis/, video_clips/, search_results/
"""
    print(config)


def get_data_root() -> str:
    """Resolve the data root path for host or container usage."""
    env_root = os.getenv("DATA_DIR")
    if env_root:
        return env_root
    if Path("/data").exists():
        return "/data"
    return "data"


async def check_mcp_servers():
    """Check if MCP servers are responding"""
    import httpx

    servers = {
        "FFmpeg-0 (Audio)": os.getenv("FFMPEG_0_MCP_SERVER_URL", "http://localhost:8000/mcp"),
        "Librosa (Analysis)": os.getenv("LIBROSA_MCP_SERVER_URL", "http://localhost:8001/mcp"),
        "FFmpeg-1 (Splitting)": os.getenv("FFMPEG_1_MCP_SERVER_URL", "http://localhost:8002/mcp"),
        "FFmpeg-2 (Compression)": os.getenv("FFMPEG_2_MCP_SERVER_URL", "http://localhost:8003/mcp"),
        "Transcribe (Speech)": os.getenv("SPEECH_TO_TEXT_MCP_SERVER_URL", "http://localhost:8004/mcp"),
        "Grep (Search)": os.getenv("GREP_MCP_SERVER_URL", "http://localhost:8005/mcp"),
        "FFmpeg-Frames (Frames)": os.getenv("FRAMES_MCP_SERVER_URL", "http://localhost:8006/mcp"),
        "Segmentation (HTTP Proxy)": os.getenv("SEGMENTATION_MCP_SERVER_URL", "http://localhost:8007/mcp"),
    }
    
    print(f"{Colors.BOLD}Checking MCP Servers:{Colors.ENDC}\n")
    
    all_healthy = True
    async with httpx.AsyncClient(timeout=2.0) as client:
        for name, url in servers.items():
            try:
                health_url = to_health_url(url)
                response = await client.get(health_url)
                if response.status_code == 200:
                    print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {name}: {Colors.OKGREEN}Healthy{Colors.ENDC}")
                else:
                    print(f"  {Colors.FAIL}✗{Colors.ENDC} {name}: {Colors.FAIL}Unhealthy (status {response.status_code}){Colors.ENDC}")
                    all_healthy = False
            except Exception as e:
                print(f"  {Colors.FAIL}✗{Colors.ENDC} {name}: {Colors.FAIL}Not responding{Colors.ENDC}")
                all_healthy = False
    
    print()
    
    if not all_healthy:
        print(f"{Colors.WARNING}⚠️  Some MCP servers are not responding{Colors.ENDC}")
        print(f"   Start them with: {Colors.BOLD}docker-compose up{Colors.ENDC}")
        print()
    
    return all_healthy


def to_health_url(mcp_url: str) -> str:
    """Convert an MCP base URL to its /health endpoint."""
    parsed = urlparse(mcp_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/mcp"):
        path = path[: -len("/mcp")]
    path = f"{path}/health" if path else "/health"
    return urlunparse(parsed._replace(path=path))


async def initialize_orchestrator():
    """Initialize the orchestrator agent"""
    try:
        # Check if using local LLM for orchestrator
        use_local_llm = bool(os.getenv("LLM_ORCHESTRATOR_BASE_URL"))
        
        if use_local_llm:
            print(f"{Colors.OKCYAN}🖥️  Initializing orchestrator with local LLM...{Colors.ENDC}")
            orchestrator = OrchestratorAgent(
                llm_model_name=os.getenv("LLM_ORCHESTRATOR_MODEL"),
                llm_provider=os.getenv("LLM_ORCHESTRATOR_PROVIDER"),
                llm_base_url=os.getenv("LLM_ORCHESTRATOR_BASE_URL"),
                use_local_llm=True
            )
        else:
            print(f"{Colors.OKCYAN}🤖 Initializing orchestrator with cloud LLM...{Colors.ENDC}")
            orchestrator = OrchestratorAgent()
        
        print(f"{Colors.OKGREEN}✅ Orchestrator initialized successfully{Colors.ENDC}\n")
        return orchestrator
        
    except Exception as e:
        print(f"{Colors.FAIL}❌ Failed to initialize orchestrator: {e}{Colors.ENDC}")
        return None


async def process_user_request(orchestrator, user_input: str):
    """Process user request through the orchestrator"""
    print(f"\n{Colors.OKCYAN}🔄 Processing request...{Colors.ENDC}\n")
    
    try:
        result = await orchestrator.process_request(user_input)
        
        print(f"\n{Colors.OKGREEN}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}📤 Result:{Colors.ENDC}\n")
        print(result)
        print(f"{Colors.OKGREEN}{'='*70}{Colors.ENDC}\n")
        
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Error processing request: {e}{Colors.ENDC}\n")
        logging.error(f"Request processing failed", exc_info=True)


async def main():
    """Main CLI loop"""
    # Set up logging
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors in CLI
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Print banner
    print_banner()
    
    # Check environment
    print(f"{Colors.BOLD}Checking environment...{Colors.ENDC}\n")
    if not check_environment():
        print(f"\n{Colors.FAIL}Please configure the system before proceeding.{Colors.ENDC}")
        print(f"{Colors.WARNING}See setup instructions at the top of this file (videosearcher_cli.py){Colors.ENDC}\n")
        return
    
    # Check MCP servers
    servers_healthy = await check_mcp_servers()
    
    if not servers_healthy:
        response = input(f"{Colors.WARNING}Continue anyway? (y/n): {Colors.ENDC}").strip().lower()
        if response != 'y':
            print(f"\n{Colors.FAIL}Exiting. Please start MCP servers first.{Colors.ENDC}\n")
            return
        print()
    
    # Initialize orchestrator
    orchestrator = await initialize_orchestrator()
    if orchestrator is None:
        return
    
    # Print help
    print_help()
    
    # Main interaction loop
    print(f"{Colors.BOLD}Ready to process requests!{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Type 'help' for commands, 'examples' for example requests{Colors.ENDC}\n")
    
    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.BOLD}> {Colors.ENDC}").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit']:
                print(f"\n{Colors.OKGREEN}👋 Goodbye!{Colors.ENDC}\n")
                break
            
            elif user_input.lower() == 'help':
                print_help()
                continue
            
            elif user_input.lower() == 'examples':
                print_examples()
                continue
            
            elif user_input.lower() == 'config':
                show_config()
                continue
            
            elif user_input.lower() == 'status':
                await check_mcp_servers()
                continue
            
            # Process natural language request
            await process_user_request(orchestrator, user_input)
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.OKGREEN}👋 Goodbye!{Colors.ENDC}\n")
            break
        
        except Exception as e:
            print(f"\n{Colors.FAIL}❌ Unexpected error: {e}{Colors.ENDC}\n")
            logging.error("Unexpected error in main loop", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
