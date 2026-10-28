"""
Orchestrator Testing Framework

Tests the orchestrator agent's ability to convert natural language prompts
into valid tool execution algorithms across different LLMs.

Records raw orchestrator output for manual evaluation.
"""

import os
import asyncio
import logging
import json
from datetime import datetime
from typing import List, Optional, Any, Dict, Tuple
import pandas as pd
from pydantic import BaseModel

from agents.items import HandoffCallItem, HandoffOutputItem, ToolCallItem, ToolCallOutputItem, MessageOutputItem

from videosearcher_agents.orchestrator.main_agent import OrchestratorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class TestScenario(BaseModel):
    """Test scenario from Testing Scenarios.xlsx"""
    scenario_id: str
    scenario_name: str
    prompt_level: str
    input_prompt: str
    expected_sequence: List[str]
    input_path: Optional[str] = None
    difficulty: Optional[str] = None


class HandoffRecord(BaseModel):
    """Record of a handoff call between agents"""
    sequence_number: int
    from_agent: str
    to_agent: str
    timestamp: str
    tool_name: Optional[str] = None
    arguments_raw: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


class ConversationItem(BaseModel):
    """Single item in the agent conversation flow"""
    sequence: int
    item_type: str  # "message", "tool_call", "tool_output"
    content: Optional[str] = None  # Message text or tool output
    tool_name: Optional[str] = None  # Name of tool called (for tool_call/tool_output)
    arguments: Optional[Dict[str, Any]] = None  # Tool arguments (for tool_call)
    agent_role: Optional[str] = None  # e.g., "Orchestrator", "Audio Extraction"
    

class TestResult(BaseModel):
    """Result of a single test execution"""
    test_run_id: str
    scenario_id: str
    prompt_level: str
    model_name: str
    execution_time_ms: Optional[float] = None
    execution_log: Optional[str] = None
    handoff_sequence: Optional[str] = None  # "Agent1 → Agent2 → Agent3"
    conversation: Optional[List[ConversationItem]] = None  # Full conversation transcript


class TestModelConfig(BaseModel):
    """Configuration for testing different models"""
    model_name: str  # Model name to pass to orchestrator (e.g., "gpt-4o", "gemini/gemini-2.0-flash-exp")
    api_key: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    display_name: Optional[str] = None  # Optional display name for results
    use_local_model: bool = False  # When True, orchestrator uses local LLM defaults


# ============================================================================
# Configuration
# ============================================================================

TEST_SCENARIOS_FILE = os.path.join(os.path.dirname(__file__), "Testing Scenarios.xlsx")
RESULTS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

# Test models - orchestrator configures LLM internally based on model_name
# Orchestrator will use llm_utils to get config for each model
TEST_MODELS = [
    #TestModelConfig(model_name="gemini/gemini-2.0-flash-exp", display_name="Gemini Flash 2.5"),
    #TestModelConfig(model_name="gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"), display_name="GPT-5-mini"),
    # Local model (no parameters passed to OrchestratorAgent)
    TestModelConfig(model_name="GPT-OSS-120B", provider = "openai-chat", base_url=os.getenv("LLM_ORCHESTRATOR_BASE_URL"), display_name="Local Model", use_local_model=True),
    # TestModelConfig(model_name="claude-3-5-sonnet-20241022", display_name="Claude 3.5 Sonnet"),
]

# ============================================================================
# Excel I/O
# ============================================================================

def load_test_scenarios(excel_path: str) -> List[TestScenario]:
    """Load test scenarios from Excel file"""
    logger.info(f"Loading test scenarios from {excel_path}")
    
    df = pd.read_excel(excel_path, sheet_name="Test Cases")
    
    scenarios = []
    for _, row in df.iterrows():
        # Parse expected sequence (handle both '->' and comma-separated formats)
        expected_seq_str = str(row['Expected Sequence'])
        if '->' in expected_seq_str:
            expected_sequence = [t.strip() for t in expected_seq_str.split('->')]
        else:
            expected_sequence = [t.strip() for t in expected_seq_str.split(',')]
        
        scenario = TestScenario(
            scenario_id=str(row['Scenario ID']),
            scenario_name=str(row['Scenario Name']),
            prompt_level=str(row['Prompt Level']),
            input_prompt=str(row['Input Prompt']),
            expected_sequence=expected_sequence,
            input_path=row.get('Input Path'),
            difficulty=row.get('Difficulty')
        )
        scenarios.append(scenario)
    
    logger.info(f"Loaded {len(scenarios)} test scenarios")
    return scenarios


def save_results_to_excel(results: List[TestResult], output_path: str):
    """Save test results to Excel file"""
    logger.info(f"Saving {len(results)} test results to {output_path}")
    
    results_data = [result.model_dump() for result in results]
    df = pd.DataFrame(results_data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Test Results', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Test Results']
        for column in worksheet.columns:
            max_length = 0
            column_cells = [cell for cell in column]
            for cell in column_cells:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 80)
            worksheet.column_dimensions[column_cells[0].column_letter].width = adjusted_width
    
    logger.info(f"Results saved to {output_path}")


# ============================================================================
# Test Execution
# ============================================================================

class OrchestratorTester:
    """Executes test scenarios and records results"""
    
    def __init__(self, test_run_id: Optional[str] = None):
        self.test_run_id = test_run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.results: List[TestResult] = []
        logger.info(f"Test run ID: {self.test_run_id}")
    
    def _extract_handoffs(self, new_items: List[Any]) -> Tuple[List[HandoffRecord], str]:
        """Extract handoff calls from RunResult.new_items"""
        handoffs: List[HandoffRecord] = []
        sequence_number = 0
        agent_sequence = []
        
        for idx, item in enumerate(new_items):
            # Check for ToolCallItem - agents are called as tools via .as_tool()
            if isinstance(item, ToolCallItem):
                # Extract tool name + arguments from raw_item
                tool_name = None
                arguments_raw = None
                arguments_parsed = None
                if hasattr(item, 'raw_item'):
                    raw = item.raw_item
                    if hasattr(raw, 'name'):
                        tool_name = raw.name
                    if hasattr(raw, 'arguments'):
                        arguments_raw = raw.arguments
                        if isinstance(arguments_raw, str):
                            try:
                                arguments_parsed = json.loads(arguments_raw)
                            except json.JSONDecodeError:
                                pass
                
                # Check if this is a handoff tool (ends with _handoff)
                if tool_name and tool_name.endswith('_handoff'):
                    sequence_number += 1
                    
                    # Extract agent name from tool name
                    # e.g., "audio_extraction_handoff" -> "Audio Extraction"
                    agent_name = tool_name.replace('_handoff', '').replace('_', ' ').title()
                    
                    source_agent = "Orchestrator"
                    
                    handoff = HandoffRecord(
                        sequence_number=sequence_number,
                        from_agent=source_agent,
                        to_agent=agent_name,
                        timestamp=datetime.now().isoformat(),
                        tool_name=tool_name,
                        arguments_raw=arguments_raw,
                        arguments=arguments_parsed
                    )
                    handoffs.append(handoff)
                    agent_sequence.append(agent_name)
            
            # Check for HandoffCallItem (in case real handoffs are used)
            elif isinstance(item, HandoffCallItem):
                sequence_number += 1
                
                # Extract target agent name from the handoff call
                target_agent = self._extract_agent_name_from_handoff(item)
                
                source_agent = "Orchestrator"
                
                handoff = HandoffRecord(
                    sequence_number=sequence_number,
                    from_agent=source_agent,
                    to_agent=target_agent,
                    timestamp=datetime.now().isoformat()
                )
                handoffs.append(handoff)
                agent_sequence.append(target_agent)
        
        handoff_sequence = " → ".join(agent_sequence) if agent_sequence else "None"
        return handoffs, handoff_sequence
    
    def _extract_agent_name_from_handoff(self, handoff_item: HandoffCallItem) -> str:
        """Extract agent name from HandoffCallItem"""
        # The target_agent property should give us the agent being handed off to
        if hasattr(handoff_item, 'target_agent'):
            target = handoff_item.target_agent
            
            if hasattr(target, 'name'):
                agent_name = target.name
            else:
                agent_name = str(target)
        else:
            # Fallback: try to extract from raw item
            raw = handoff_item.raw_item
            agent_name = getattr(raw, 'name', 'Unknown')
        
        # Clean up agent name for display
        if isinstance(agent_name, str):
            cleaned = agent_name.replace("_", " ").title()
            return cleaned
        
        return str(agent_name)

    
    def _extract_conversation(self, new_items: List[Any]) -> List[ConversationItem]:
        """Extract full conversation flow from all items"""
        conversation = []
        sequence = 0

        for idx, item in enumerate(new_items):
            sequence += 1
            
            # MessageOutputItem - LLM messages
            if isinstance(item, MessageOutputItem):
                message_text = None
                
                if hasattr(item, 'raw_item'):
                    raw = item.raw_item
                    
                    # Extract message content
                    if hasattr(raw, 'content'):
                        content = raw.content
                        if isinstance(content, list):
                            # Usually a list of content blocks
                            message_parts = []
                            for block in content:
                                if hasattr(block, 'text'):
                                    message_parts.append(block.text)
                            message_text = " ".join(message_parts)
                        else:
                            message_text = str(content)
                
                if message_text:
                    conv_item = ConversationItem(
                        sequence=sequence,
                        item_type="message",
                        content=message_text[:500],  # Truncate for Excel
                        agent_role="Orchestrator"
                    )
                    conversation.append(conv_item)
            
            # ToolCallItem - Tool invocations
            elif isinstance(item, ToolCallItem):
                tool_name = None
                arguments_parsed = None
                
                if hasattr(item, 'raw_item'):
                    raw = item.raw_item
                    
                    if hasattr(raw, 'name'):
                        tool_name = raw.name
                    
                    if hasattr(raw, 'arguments'):
                        arguments_raw = raw.arguments
                        if isinstance(arguments_raw, str):
                            try:
                                arguments_parsed = json.loads(arguments_raw)
                            except json.JSONDecodeError:
                                pass
                
                # Extract agent name from tool name if it's a handoff
                agent_role = None
                if tool_name and tool_name.endswith('_handoff'):
                    agent_role = tool_name.replace('_handoff', '').replace('_', ' ').title()
                
                conv_item = ConversationItem(
                    sequence=sequence,
                    item_type="tool_call",
                    tool_name=tool_name,
                    arguments=arguments_parsed,
                    agent_role=agent_role
                )
                conversation.append(conv_item)
            
            # ToolCallOutputItem - Tool results
            elif isinstance(item, ToolCallOutputItem):
                output_text = None
                
                if hasattr(item, 'output'):
                    output = item.output
                    output_text = str(output)
                
                conv_item = ConversationItem(
                    sequence=sequence,
                    item_type="tool_output",
                    content=output_text[:500] if output_text else None,
                    agent_role="Tool Result"
                )
                conversation.append(conv_item)

        return conversation
    
    async def run_single_test(
        self, 
        scenario: TestScenario, 
        model_config: TestModelConfig
    ) -> TestResult:
        """Run a single test scenario with a specific LLM model"""
        display_name = model_config.display_name or model_config.model_name
        logger.info(f"Testing {scenario.scenario_id} ({scenario.prompt_level}) with {display_name}")
        
        start_time = datetime.now()
        
        try:
            # Build the full prompt including input path
            full_prompt = scenario.input_prompt
            if scenario.input_path and '[PATH' not in scenario.input_path.upper():
                # Add actual input path to the prompt
                full_prompt = f"{scenario.input_prompt} Input file: \"{scenario.input_path}\""
            
            if model_config.use_local_model:
                # Local model config is handled inside OrchestratorAgent
                orchestrator = OrchestratorAgent(
                    llm_model_name=model_config.model_name,
                    llm_provider=model_config.provider,
                    llm_base_url=model_config.base_url,
                    use_local_llm=True
                )
            else:
                orchestrator = OrchestratorAgent(
                    llm_api_key=model_config.api_key,
                    llm_model_name=model_config.model_name,
                    llm_provider=model_config.provider,
                    llm_base_url=model_config.base_url
                )
            # Get full result object with new_items
            result_obj = await orchestrator.process_request_with_items(full_prompt)
            result_text = result_obj['final_output']
            new_items = result_obj['new_items']

            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Extract handoffs and tool calls from new_items
            handoff_records, handoff_sequence = self._extract_handoffs(new_items)
            conversation = self._extract_conversation(new_items)
            
            test_result = TestResult(
                test_run_id=self.test_run_id,
                scenario_id=scenario.scenario_id,
                prompt_level=scenario.prompt_level,
                model_name=display_name,
                execution_time_ms=execution_time,
                execution_log=str(result_text),
                handoff_sequence=handoff_sequence if handoff_records else None,
                conversation=conversation if conversation else None
            )
            
            logger.info(f"Orchestrator output:\n{result_text}\n")
            logger.info(f"Test completed: {scenario.scenario_id} | Handoffs: {handoff_sequence}")
            return test_result
            
        except Exception as e:
            logger.error(f"Error during test: {e}", exc_info=True)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return TestResult(
                test_run_id=self.test_run_id,
                scenario_id=scenario.scenario_id,
                prompt_level=scenario.prompt_level,
                model_name=display_name,
                execution_time_ms=execution_time,
                execution_log=f"ERROR: {str(e)}"
            )
    
    
    async def run_all_tests(
        self,
        scenarios: List[TestScenario],
        models: List[TestModelConfig],
        filter_scenario_ids: Optional[List[str]] = None,
        filter_prompt_levels: Optional[List[str]] = None
    ) -> List[TestResult]:
        """Run all test scenarios across all models"""
        
        filtered_scenarios = scenarios
        if filter_scenario_ids:
            filtered_scenarios = [s for s in filtered_scenarios if s.scenario_id in filter_scenario_ids]
        if filter_prompt_levels:
            filtered_scenarios = [s for s in filtered_scenarios if s.prompt_level in filter_prompt_levels]
        
        total_tests = len(filtered_scenarios) * len(models)
        logger.info(f"Running {total_tests} tests ({len(filtered_scenarios)} scenarios × {len(models)} models)")
        
        for scenario in filtered_scenarios:
            for model in models:
                result = await self.run_single_test(scenario, model)
                self.results.append(result)
                
                await asyncio.sleep(0.5)
        
        logger.info(f"Completed {len(self.results)} tests")
        return self.results


# ============================================================================
# Main
# ============================================================================

async def main():
    """Main test execution"""
    logger.info("=" * 80)
    logger.info("Orchestrator Testing Framework")
    logger.info("=" * 80)
    
    # Load scenarios
    scenarios = load_test_scenarios(TEST_SCENARIOS_FILE)
    
    # Run tests
    tester = OrchestratorTester()
    results = await tester.run_all_tests(
        scenarios=scenarios,
        models=TEST_MODELS,
        filter_scenario_ids=["SC-02"],  # e.g., ["SC-02", "SC-05"]
        filter_prompt_levels=None  # ["Vague", "Clear", "Algorithmic"]
    )
    
    # Save results
    output_path = os.path.join(
        RESULTS_OUTPUT_DIR,
        f"results_{tester.test_run_id}.xlsx"
    )
    save_results_to_excel(results, output_path)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Test Run: {tester.test_run_id}")
    logger.info(f"Total Tests: {len(results)}")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
