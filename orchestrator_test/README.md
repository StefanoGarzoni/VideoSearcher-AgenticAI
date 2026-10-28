# Orchestrator Testing Framework

Simple test harness for evaluating orchestrator agent performance across different LLMs.

## Quick Start

Tests run inside the `agents-orchestrator` container (same container as the CLI).
The `orchestrator_test/` folder is mounted automatically at `/app/orchestrator_test/`.

**Prerequisites:** MCP servers and agent container must be running (see main README).

```bash
# Run all configured tests
docker exec -e PYTHONPATH=/app agents-orchestrator python /app/orchestrator_test/main.py
```

Results are saved to `orchestrator_test/results/` (visible on host via volume mount).

## How It Works

1. Loads test scenarios from `Testing Scenarios.xlsx` (27 test cases: 9 scenarios × 3 prompt levels)
2. Runs each scenario with configured LLM models (Gemini, GPT-4, Claude, etc.)
3. Records orchestrator output plus tool-call conversation data to Excel
4. Manual evaluation of results based on execution logs and conversation flow

## Test Scenarios

Input: `Testing Scenarios.xlsx` sheet "Test Cases"

Required columns:
- **Scenario ID** - Unique identifier (SC-01, SC-02, etc.)
- **Difficulty** - Complexity level 1-10
- **Scenario Name** - Descriptive name
- **Prompt Level** - Vague, Clear, or Algorithmic
- **Input Prompt** - The prompt to send to orchestrator
- **Expected Sequence** - Tool chain (e.g., "Ffmpeg-0 -> Transcribe")
- **Input Path** - File/folder path (use placeholders like `[PATH_TO_VIDEO_MP4]` or actual paths)

The framework automatically includes the Input Path in the prompt sent to the orchestrator.

## Output

Results saved to: `results/results_YYYY-MM-DD_HH-MM-SS.xlsx`

Columns:
- test_run_id - Test batch identifier
- scenario_id - Which scenario was tested
- prompt_level - Vague, Clear, or Algorithmic
- model_name - Which LLM was used
- execution_time_ms - How long it took
- execution_log - Raw orchestrator response (for manual evaluation)
- handoff_sequence - Agent call chain derived from tool calls
- conversation - Full run items (messages, tool calls, tool outputs) serialized as JSON

## Configuration

### Model Names

The framework passes model names to the orchestrator, which handles LLM configuration internally via `llm_utils.py`.

Edit `main.py` `TEST_MODELS` list to test different models:

```python
TEST_MODELS = [
    TestModelConfig(model_name="gemini/gemini-2.0-flash-exp", display_name="Gemini Flash 2.5"),
    TestModelConfig(model_name="gpt-4o", display_name="GPT-4o"),
    TestModelConfig(model_name="claude-3-5-sonnet-20241022", display_name="Claude 3.5"),
]
```

The orchestrator receives only the `model_name` and configures LLM settings internally using environment variables from `agents-container/.env` and `llm_utils.get_llm_config()`.

### Filter Tests

In `main.py` `main()` function:

```python
results = await tester.run_all_tests(
    scenarios=scenarios,
    models=TEST_MODELS,
    filter_scenario_ids=["SC-01"],        # Optional
    filter_prompt_levels=["Clear"]        # Optional
)
```

## Conversation Capture (New)

The test harness now captures the full run “conversation” using OpenAI Agents SDK `RunResult.new_items`.
This includes:

- MessageOutputItem → agent messages (stored as `conversation` entries with `item_type="message"`)
- ToolCallItem → tool invocations (stored with `item_type="tool_call"`, tool name + arguments)
- ToolCallOutputItem → tool outputs (stored with `item_type="tool_output"`)

This makes it easy to audit exactly which specialized tools were called and what they returned.

## Manual Evaluation Workflow

1. Run tests: `docker exec -e PYTHONPATH=/app agents-orchestrator python /app/orchestrator_test/main.py`
2. Open results Excel file in `orchestrator_test/results/`
3. Review `execution_log` and `conversation` columns for each test
4. Compare orchestrator output against scenario expectations
5. Document success/failure and quality assessment
6. Use patterns to refine orchestrator instructions

## Dependencies

All dependencies are installed in the `agents-orchestrator` container.
No local install needed — the test runs via `docker exec`.
