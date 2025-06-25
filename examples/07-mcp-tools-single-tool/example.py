"""
Example of PlanReAct agent usage with a simple task. Single
MCP tool from a server is added to the toolbox. This example
requires definition of the MCP server in `.env` file, i.e.

mcpServers='{"github": {"command": "npx", "args": ["-y","@modelcontextprotocol/server-github"], "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "<YOUR_TOKEN>"}}}'

"""

from loguru import logger

from elemental_agents.core.agent.agent_factory import AgentFactory

TASK = "Search Github repositories for REST API creation in Python."
SESSION = "Test Session"

assistant = AgentFactory.create(
    agent_name="AssistantAgent",
    agent_persona="You are a helpful assistant.",
    agent_type="ReAct",
    llm_model="openai|gpt-4.1-mini",
    tools=[
        "Calculator",
        "CurrentTime",
        "NoAction",
        "MCP|Github|search_repositories",
    ],
)
result = assistant.run(task=TASK, input_session=SESSION)
logger.info(f"Result: {result}")
