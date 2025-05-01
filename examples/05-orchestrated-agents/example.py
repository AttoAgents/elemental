"""
Example of a two agents with distinct roles orchestrated to run together.
"""

from loguru import logger

from elemental_agents.core.agent.agent_factory import AgentFactory
from elemental_agents.core.orchestration.dynamic_agent_orchestrator import (
    DynamicAgentOrchestrator,
)

if __name__ == "__main__":

    factory = AgentFactory()

    planner_agent = factory.create(
        agent_name="PlannerAgent",
        agent_persona="",
        agent_type="planner",
        llm_model="openai|gpt-4.1-mini",
    )
    executor_agent = factory.create(
        agent_name="ExecutorAgent",
        agent_persona="You are an expert software engineer.",
        agent_type="ReAct",
        llm_model="openai|gpt-4.1-mini",
        tools=[
            "Calculator",
            "CurrentTime",
            "NoAction",
            "ReadFiles",
            "WriteFile",
            "ListFiles",
        ],
    )

    orchestrator = DynamicAgentOrchestrator(
        planner=planner_agent, executor=executor_agent
    )

    result = orchestrator.run(
        instruction="Create FastAPI backend for a TODO application.",
        input_session="Example Session",
    )
    logger.info(f"Result: {result}")
