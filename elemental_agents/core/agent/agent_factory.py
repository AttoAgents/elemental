"""
AgentFactory class is responsible for creating agents based on the configuration provided.
"""

from typing import List, Optional

from loguru import logger

from elemental_agents.core.agent.agent import Agent
from elemental_agents.core.agent.generic_agent import GenericAgent
from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.planner_agent import PlannerAgentLogic
from elemental_agents.core.agent_logic.simple_agent import SimpleAgentLogic
from elemental_agents.core.agent_logic.verifier_agent import VerifierAgentLogic
from elemental_agents.core.toolbox.toolbox import ToolBox
from elemental_agents.llm.llm_factory import LLMFactory
from elemental_agents.utils.config import ConfigModel
from elemental_agents.utils.exceptions import AgentTypeException


class AgentFactory:
    """
    Agent Factory class to create agent instances based on the agent type.
    """

    def __init__(self) -> None:
        self._config = ConfigModel()

    def create(
        self,
        agent_name: str = None,
        agent_type: str = None,
        llm_model: str = None,
        memory_capacity: Optional[int] = None,
        tools: Optional[List[str]] = None,
        termination: Optional[str] = None,
        agent_persona: Optional[str] = None,
        template: Optional[str] = None,
    ) -> Agent:
        """
        Create an agent instance based on the agent type.
        If the agent type is not provided, the default agent
        is used that is specified in the configuration file.

        :param agent_name: The name of the agent to create.
        :param agent_type: The type of the agent to create.
        :param llm_model: The LLM model to use for the agent.
        :param memory_capacity: The memory capacity of the agent.
        :param tools: The tools available to the agent.
        :param termination: The termination condition for the agent.
        :param agent_persona: The persona of the agent.
        :param template: The template to use for the agent.
        :return: An instance of the Agent class.
        """
        local_agent_type = agent_type or self._config.agent_default_type
        short_memory_capacity = memory_capacity or self._config.short_memory_items

        # LLM setup
        llm_factory = LLMFactory()
        llm = llm_factory.create(engine_name=llm_model)

        # Prompt variables
        agent_context = AgentContext(
            agent_name=agent_name,
            agent_persona=agent_persona,
        )

        # Toolbox
        if tools is not None and len(tools) > 0:
            toolbox = ToolBox()
            for tool in tools:
                toolbox.register_tool_by_name(tool)

        agent_logic: SimpleAgentLogic | PlannerAgentLogic | VerifierAgentLogic = None

        if local_agent_type == "simple":
            agent_logic = SimpleAgentLogic(
                model=llm, context=agent_context, template=template
            )

        if local_agent_type == "planner":
            agent_logic = PlannerAgentLogic(
                model=llm, context=agent_context, template=template
            )

        if local_agent_type == "verifier":
            agent_logic = VerifierAgentLogic(
                model=llm, context=agent_context, template=template
            )

        if agent_logic is None:
            logger.error(
                f"Agent type {local_agent_type} is not supported. "
                f"Supported agent types are: simple, planner, verifier."
            )
            raise AgentTypeException(
                f"Agent type {local_agent_type} is not supported. "
                f"Supported agent types are: simple, planner, verifier."
            )

        agent = GenericAgent(
            agent_logic=agent_logic,
            short_memory_capacity=short_memory_capacity,
            toolbox=toolbox,
            termination_sequence=termination,
        )

        return agent
