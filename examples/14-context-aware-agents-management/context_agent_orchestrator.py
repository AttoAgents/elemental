"""
Orchestration of context-aware agents for specialized tasks.
"""

from pathlib import Path
from typing import Dict, Union

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


class ContextAwareAgentManager:
    """
    Utility class for managing multiple context-aware agents.
    """

    def __init__(self) -> None:
        """
        Initialize the context manager with an empty agent registry.
        """
        self.agents: Dict[str, ContextAwareGenericAgent] = {}

    def register_agent(self, name: str, agent: ContextAwareGenericAgent) -> None:
        """Register a context-aware agent."""
        self.agents[name] = agent

    def add_global_context_directory(self, directory: Union[str, Path]) -> None:
        """Add a directory to all registered agents."""
        for agent in self.agents.values():
            agent.add_context_directory(directory)

    def add_global_context_file(self, file_path: Union[str, Path]) -> None:
        """Add a file to all registered agents."""
        for agent in self.agents.values():
            agent.add_context_file(file_path)

    def set_global_refresh_mode(self, mode: str) -> None:
        """Set refresh mode for all agents."""
        for agent in self.agents.values():
            agent.set_context_refresh_mode(mode)

    def clear_all_caches(self) -> None:
        """Clear context caches for all agents."""
        for agent in self.agents.values():
            agent.clear_context_cache()

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status for all agents."""
        return {name: agent.get_context_status() for name, agent in self.agents.items()}


def create_code_review_agent(llm_model_name: str) -> ContextAwareGenericAgent:
    """
    Create an agent specialized for code review with relevant context.

    This agent will focus on analyzing source code, identifying potential issues,
    and suggesting improvements based on best practices.

    :param llm_model_name: Name of the LLM model to use for code review tasks.
    :return: ContextAwareGenericAgent instance configured for code review tasks.
    """

    llm_factory = LLMFactory()
    llm = llm_factory.create(engine_name=llm_model_name)

    context_config = ContextConfig(
        max_files=200,
        max_file_size=1024 * 1024,  # 1MB
        include_line_numbers=True,
        max_content_length=50000,
        include_extensions=[
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
        ],
        exclude_directories=[
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "dist",
            "build",
            ".pytest_cache",
            ".mypy_cache",
            "coverage",
            ".coverage",
        ],
    )

    agent_context = AgentContext(
        agent_name="CodeReviewAgent",
        agent_persona="""
        Expert code reviewer with deep knowledge of software engineering best
        practices. You have access to the complete codebase and can analyze code
        quality, security issues, performance problems, and suggest
        improvements.
        """,
    )

    box = ToolBox()
    template = FileTemplate(agent_context.model_dump(), "ReAct.template")
    strategy = ReactPrompt(system_template=template, tool_dictionary=box.describe())

    agent_logic = GenericAgentLogic(
        context=agent_context, model=llm, prompt_strategy=strategy, stop_word="<result>"
    )

    context_manager = LLMContextManager(
        config=context_config,
        context_refresh_mode="always",
        include_file_content_default=True,
        context_section_name="## Codebase Context",
    )

    context_manager.add_auto_context_directory("./src")
    context_manager.add_auto_context_directory("./lib")
    context_manager.add_auto_context_directory("./docs")
    context_manager.add_auto_context_directory("./tests")

    context_manager.add_auto_context_file("./README.md")
    context_manager.add_auto_context_file("./CHANGELOG.md")
    context_manager.add_auto_context_file("./requirements.txt")

    return ContextAwareGenericAgent(
        agent_logic=agent_logic,
        short_memory_capacity=-1,
        toolbox=box,
        termination_sequence="<result>",
        context_manager=context_manager,
        enable_auto_context=True,
    )


def create_documentation_agent(llm_model_name: str) -> ContextAwareGenericAgent:
    """
    Create an agent specialized for documentation with selective context.

    This agent will focus on generating and maintaining documentation,
    including API references, user guides, and tutorials based on the codebase
    and existing documentation.
    :return: ContextAwareGenericAgent instance configured for documentation tasks."""

    llm_factory = LLMFactory()
    llm = llm_factory.create(engine_name=llm_model_name)

    context_config = ContextConfig(
        max_files=150,
        include_line_numbers=False,  # Not needed for docs
        max_content_length=30000,
        include_extensions=[".md", ".rst", ".txt", ".py"],  # Code for API docs
        exclude_directories=[
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "dist",
            "build",
            "coverage",
        ],
    )

    agent_context = AgentContext(
        agent_name="DocumentationAgent",
        agent_persona="""Technical writer and documentation specialist with access to source code.
        You can create comprehensive documentation, API references, tutorials, and user guides
        based on the codebase and existing documentation.""",
    )

    box = ToolBox()
    template = FileTemplate(agent_context.model_dump(), "ReAct.template")
    strategy = ReactPrompt(system_template=template, tool_dictionary=box.describe())

    agent_logic = GenericAgentLogic(
        context=agent_context, model=llm, prompt_strategy=strategy, stop_word="<result>"
    )

    context_manager = LLMContextManager(
        config=context_config,
        context_refresh_mode="always",
        include_file_content_default=True,
        context_section_name="## Documentation Context",
    )

    context_manager.add_auto_context_directory("./src")
    context_manager.add_auto_context_directory("./tests")

    context_manager.add_auto_context_file("./README.md")
    context_manager.add_auto_context_file("./CHANGELOG.md")
    context_manager.add_auto_context_file("./requirements.txt")

    return ContextAwareGenericAgent(
        agent_logic=agent_logic,
        short_memory_capacity=15,
        toolbox=box,
        termination_sequence="<result>",
        context_manager=context_manager,
        enable_auto_context=True,
    )


def create_project_management_agent(llm_model_name: str) -> ContextAwareGenericAgent:
    """
    Create an agent for project management with configuration focus.
    This agent will analyze project structure, dependencies, and configuration files,
    providing insights on project management and development workflow.

    :return: ContextAwareGenericAgent instance configured for project management tasks.
    """

    llm_factory = LLMFactory()
    llm = llm_factory.create(engine_name=llm_model_name)

    context_config = ContextConfig(
        max_files=50,
        include_line_numbers=False,
        max_content_length=15000,
        include_extensions=[
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".cfg",
            ".ini",
            ".py",
        ],
        exclude_directories=[
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "dist",
            "build",
            "coverage",
            "src",
            "lib",  # Exclude source code
        ],
    )

    agent_context = AgentContext(
        agent_name="ProjectManagerAgent",
        agent_persona="""Project manager and DevOps specialist with access to project configuration
        and documentation. You can analyze project structure, dependencies, deployment configuration,
        and provide insights on project management and development workflow.""",
    )

    box = ToolBox()
    template = FileTemplate(agent_context.model_dump(), "ReAct.template")
    strategy = ReactPrompt(system_template=template, tool_dictionary=box.describe())

    agent_logic = GenericAgentLogic(
        context=agent_context, model=llm, prompt_strategy=strategy, stop_word="<result>"
    )

    context_manager = LLMContextManager(
        config=context_config,
        context_refresh_mode="always",
        include_file_content_default=True,
        context_section_name="## Project Configuration Context",
    )

    context_manager.add_auto_context_directory("./docs")
    context_manager.add_auto_context_directory("./config")
    context_manager.add_auto_context_directory("./scripts")

    context_manager.add_auto_context_file("./README.md")
    context_manager.add_auto_context_file("./CHANGELOG.md")
    context_manager.add_auto_context_file("./requirements.txt")
    context_manager.add_auto_context_file("./TODO.txt")
    context_manager.add_auto_context_file("./setup.py")
    context_manager.add_auto_context_file("./pyproject.toml")
    context_manager.add_auto_context_file("./package.json")
    context_manager.add_auto_context_file("./Dockerfile")
    context_manager.add_auto_context_file("./docker-compose.yml")

    return ContextAwareGenericAgent(
        agent_logic=agent_logic,
        short_memory_capacity=20,
        toolbox=box,
        termination_sequence="<result>",
        context_manager=context_manager,
        enable_auto_context=True,
    )


# Usage example
def main() -> Dict[str, str]:
    """
    Main function to demonstrate the orchestration of context-aware agents.
    This function creates specialized agents for code review, documentation,
    and project management, registers them with a context manager, and runs tasks.

    :return: Results from each agent's task execution.
    """
    manager = ContextAwareAgentManager()

    # Create specialized agents
    code_reviewer = create_code_review_agent("openai|gpt-4.1-mini")
    doc_writer = create_documentation_agent("openai|gpt-4.1-mini")
    project_manager = create_project_management_agent("openai|gpt-4.1-mini")

    # Register agents
    manager.register_agent("code_review", code_reviewer)
    manager.register_agent("documentation", doc_writer)
    manager.register_agent("project_mgmt", project_manager)

    # Add global context
    manager.add_global_context_file("./ARCHITECTURE.md")

    # Set all to always refresh
    manager.set_global_refresh_mode("always")

    # Use agents for different tasks
    tasks = {
        "code_review": "Review the main application code for potential issues",
        "documentation": "Create API documentation for the main modules",
        "project_mgmt": "Analyze the project dependencies and suggest updates",
    }

    results = {}
    for agent_name, task in tasks.items():
        agent = manager.agents[agent_name]
        results[agent_name] = agent.run(task, f"Session_{agent_name}")

    return results


if __name__ == "__main__":
    results = main()
    for agent_name, result in results.items():
        print(f"\n=== {agent_name.upper()} RESULT ===")
        print(result)
