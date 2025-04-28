"""
Definition of the agent unit as basis for multi-agent orchestrator.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from loguru import logger

from elemental_agents.core.agent.agent import Agent
from elemental_agents.core.selector.agent_selector import AgentSelector


class AgentTeam(ABC):
    """
    Abstract class for the unit of multiple agents (represented by their
    respective executors).
    """

    def __init__(self, selector: AgentSelector = None):
        """
        Initialize the agent unit with the specified agent selector.
        This unit includes a dictionary of agent executors. This dictionary
        is expected to be populated by the user by registering the agents
        with their respective executors.

        :param selector: The agent selector to use for selecting agents.
        """

        self._agents: Dict[str, Agent] = {}
        self._agent_types: Dict[str, str] = {}
        self._selector = selector

    def get_selector(self) -> AgentSelector:
        """
        Get the agent selector used by the unit.

        :return: The agent selector.
        """

        return self._selector

    def get_agents(self) -> dict:
        """
        Get the dictionary of agents with their respective executors.

        :return: The dictionary of agents.
        """

        return self._agents

    def register_agent(self, agent_name: str, executor: Agent, agent_type: str) -> None:
        """
        Register an agent with its executor in the unit. This method is used
        to populate the dictionary of agents with their respective executors.

        :param agent_name: The name of the agent to register.
        :param executor: The executor object for the agent.
        :param agent_type: The type of the agent.
        """

        self._agents[agent_name] = executor
        self._agent_types[agent_name] = agent_type

        logger.info(
            f"Agent {agent_name} registered with agent unit. Agent type: {agent_type}"
        )

    def describe(self) -> Dict[str, str]:
        """
        Describe the agent unit by creating a dictionary of all agents
        registered in the unit and their descriptions (agent personas).

        :return: Dictionary of agent names and their descriptions.
        """

        unit_description = {}
        for _, agent in self._agents.items():

            agent_name = agent.get_agent_name()
            agent_persona = agent.get_agent_persona()
            agent_tools = agent.describe_toolbox()

            unit_description[agent_name] = (
                f"Persona: {agent_persona}, Abilities: {agent_tools}"
            )

        return unit_description

    def update_conversational_strategies(self) -> None:
        """
        Update the conversational strategies of all agents in the unit.
        """

        agent_description = self.describe()

        for agent_name, agent in self._agents.items():
            agent_type = self._agent_types[agent_name]
            if agent_type == "ConvPlanReAct":
                agent.update_prompt_strategy(agent_description)

    @abstractmethod
    def run(self, task: str | List[str], input_session: str) -> str:
        """
        Run the unit executor with the specified task. This method is an equivalent
        to run method of the single agent executor.

        :param task: The task to run the agents on.
        :param input_session: The input session to run the agents with.
        :return: The final result from the agents.
        """
