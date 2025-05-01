"""
Example of a simple agent team with two agents.
"""

from loguru import logger

from elemental_agents.core.agent.agent_factory import AgentFactory
from elemental_agents.core.agent_team.generic_agent_team import GenericAgentTeam
from elemental_agents.core.selector.agent_selector_factory import AgentSelectorFactory

if __name__ == "__main__":

    factory = AgentFactory()
    agent1 = factory.create(
        agent_name="AssistantAgent",
        agent_persona="You are a helpful assistant.",
        agent_type="ConvPlanReAct",
        llm_model="openai|gpt-4.1-mini",
        tools=["Calculator", "CurrentTime", "NoAction"],
    )
    agent2 = factory.create(
        agent_name="ProgrammerAgent",
        agent_persona="You are a helpful programmer.",
        agent_type="ConvPlanReAct",
        llm_model="openai|gpt-4.1-mini",
        tools=["Calculator", "CurrentTime", "NoAction"],
    )

    selector_factory = AgentSelectorFactory()
    agent_selector = selector_factory.create(
        selector_name="conversational", lead_agent="AssistantAgent"
    )
    agent_team = GenericAgentTeam(selector=agent_selector)
    agent_team.register_agent("AssistantAgent", agent1, "ConvPlanReAct")
    agent_team.register_agent("ProgrammerAgent", agent2, "ConvPlanReAct")

    result = agent_team.run(
        task="What is the color of sky on Mars?", input_session="Example Session"
    )
    logger.info(f"Result: {result}")
