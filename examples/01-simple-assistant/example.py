"""
This example demonstrates how to create a simple assistant agent using the Elemental Agents library.
It shows how to create an agent, select language model run the task.
"""

from loguru import logger

from elemental_agents.core.agent.agent_factory import AgentFactory

TASK = "Why is the sky blue?"
SESSION = "TestSession"

assistant = AgentFactory.create(
    agent_name="AssistantAgent",
    agent_persona="Simple always helpful assistant",
    agent_type="simple",
    llm_model="ollama|gemma3",
)

result = assistant.run(task=TASK, input_session=SESSION)
logger.info(f"Result: {result}")
