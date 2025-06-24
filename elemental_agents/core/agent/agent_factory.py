"""
AgentFactory class is responsible for creating agents based on the configuration provided.
"""

from typing import List, Optional

from loguru import logger

from elemental_agents.core.agent.agent import Agent
from elemental_agents.core.agent.generic_agent import GenericAgent
from elemental_agents.core.agent.simple_agent import SimpleAgent
from elemental_agents.core.agent_logic.agent_logic import AgentLogic
from elemental_agents.core.agent_logic.agent_logic_factory import AgentLogicFactory
from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.toolbox.toolbox import ToolBox
from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm_factory import LLMFactory
from elemental_agents.utils.config import ConfigModel
from elemental_agents.utils.exceptions import AgentTypeException


class AgentFactory:
    """
    Agent Factory class to create agent instances based on the agent type.
    """

    @staticmethod
    def create(
        agent_name: str = None,
        agent_type: str = None,
        llm_model: str = None,
        agent_persona: str = None,
        memory_capacity: Optional[int] = None,
        tools: Optional[List[str]] = None,
        termination: Optional[str] = "<result>",
        template: Optional[str] = None,
        model_parameters: Optional[ModelParameters] = None,
    ) -> Agent:
        """
        Create an agent instance based on the agent type. If the agent type is
        not provided, the default agent is used that is specified in the
        configuration file.

        :param agent_name: The name of the agent to create.
        :param agent_type: The type of the agent to create.
        :param llm_model: The LLM model to use for the agent.
        :param memory_capacity: The memory capacity of the agent.
        :param tools: The tools available to the agent.
        :param termination: The termination condition for the agent.
        :param agent_persona: The persona of the agent.
        :param template: The template string to use for the agent. If None, the
            default file template will be used.
        :param model_parameters: The language model parameters to use for the agent.
        :return: An instance of the Agent class.
        """

        supported_agent_types = [
            "simple",
            "planner",
            "planverifier",
            "replanner",
            "composer",
            "verifier",
            "react",
            "planreact",
            "convplanreact",
        ]

        config = ConfigModel()
        local_agent_type = agent_type or config.agent_default_type
        short_memory_capacity = memory_capacity or config.short_memory_items

        # LLM setup
        llm_factory = LLMFactory()
        if model_parameters is None:
            llm_parameters = ModelParameters()
        else:
            llm_parameters = model_parameters

        llm = llm_factory.create(engine_name=llm_model, model_parameters=llm_parameters)

        # Prompt variables
        agent_context = AgentContext(
            agent_name=agent_name,
            agent_persona=agent_persona,
        )

        # Toolbox
        toolbox: ToolBox = None
        if tools is not None and len(tools) > 0:
            toolbox = ToolBox()
            for tool in tools:
                toolbox.register_tool_by_name(tool)

        agent_logic: AgentLogic = AgentLogicFactory.create_agent_logic(
            logic_type=local_agent_type,
            llm=llm,
            context=agent_context,
            template=template,
            toolbox=toolbox,
        )
        agent: Agent = None

        match local_agent_type:

            # Non-iterative agents
            case (
                "simple"
                | "planner"
                | "planverifier"
                | "replanner"
                | "composer"
                | "verifier"
            ):

                agent = SimpleAgent(
                    agent=agent_logic, short_memory_capacity=short_memory_capacity
                )
                return agent

            # Iterative agents
            case "react" | "planreact" | "convplanreact":

                agent = GenericAgent(
                    agent_logic=agent_logic,
                    short_memory_capacity=short_memory_capacity,
                    toolbox=toolbox,
                    termination_sequence=termination,
                )
                return agent

            case _:
                logger.error(
                    f"Agent type {local_agent_type} is not supported. "
                    f"Supported agent types are: {supported_agent_types}."
                )
                raise AgentTypeException(
                    f"Agent type {local_agent_type} is not supported. "
                    f"Supported agent types are: {supported_agent_types}."
                )
