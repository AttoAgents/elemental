"""
Example of a context-aware agent that uses a context manager to handle project
files and directories.
"""

import json

from rich.console import Console

from elemental_agents.core.agent.generic_agent_with_context import (
    ContextAwareGenericAgent,
)
from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.generic_agent import GenericAgentLogic
from elemental_agents.core.context.context_manager import (
    ContextConfig,
    LLMContextManager,
)
from elemental_agents.core.prompt_strategy.prompt_template import FileTemplate
from elemental_agents.core.prompt_strategy.react_prompt import ReactPrompt
from elemental_agents.core.toolbox.toolbox import ToolBox
from elemental_agents.llm.llm_factory import LLMFactory

console = Console()

# Create context manager with configuration
context_config = ContextConfig(
    max_files=100,
    max_file_size=512 * 1024,
    include_line_numbers=True,
    max_content_length=20000,
    include_extensions=[".py", ".md", ".txt", ".json", ".yaml"],
)

context_manager = LLMContextManager(
    config=context_config,
    context_refresh_mode="always",
    include_file_content_default=True,
    context_section_name="## Project Context",
)

# Add auto-context sources
context_manager.add_auto_context_directory("./src")
context_manager.add_auto_context_directory("./config")
context_manager.add_auto_context_file("./README.md")
context_manager.add_auto_context_file("./requirements.txt")

# LLM and toolbox setup
llm_factory = LLMFactory()
llm = llm_factory.create()

box = ToolBox()
box.register_tool_by_name("Calculator")
box.register_tool_by_name("ListFiles")

# Agent setup
context = AgentContext(
    agent_name="ContextAwareAgent",
    agent_persona="Expert developer with full project context",
)

template = FileTemplate(context.model_dump(), "ReAct.template")
strategy = ReactPrompt(system_template=template, tool_dictionary=box.describe())

test_agent_logic = GenericAgentLogic(
    context=context,
    model=llm,
    prompt_strategy=strategy,
    stop_word="<PAUSE>",
)

# Create agent with context manager
executor = ContextAwareGenericAgent(
    agent_logic=test_agent_logic,
    short_memory_capacity=-1,
    toolbox=box,
    termination_sequence="<result>",
    context_manager=context_manager,
    enable_auto_context=True,
)

# Test the agent
console.print("[bold green]Testing Context-Aware Agent[/bold green]")

# Show context status
status = executor.get_context_status()
console.print(f"[blue]Context Status:[/blue]")
console.print(json.dumps(status, indent=2))

# Test with context
result = executor.run(
    "Analyze the project structure and main functionality", "TestSession"
)
console.print(f"[green]Result:[/green] {result}")

# Show cache statistics
cache_stats = context_manager.get_cache_statistics()
console.print(f"[yellow]Cache Statistics:[/yellow]")
console.print(json.dumps(cache_stats, indent=2))
