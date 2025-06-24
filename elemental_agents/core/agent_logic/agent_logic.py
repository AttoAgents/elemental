"""
Agent class that represents a basic agent in the framework.
"""

from abc import ABC, abstractmethod

from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.memory.short_memory import ShortMemory
from elemental_agents.core.prompt_strategy.prompt import PromptStrategy
from elemental_agents.llm.data_model import Message
from elemental_agents.llm.llm import LLM


class AgentLogic(ABC):
    """
    Agent class that represents a basic agent in the framework.
    """

    def __init__(
        self, context: AgentContext, llm: LLM, prompt_strategy: PromptStrategy
    ) -> None:
        """
        Initialize the Agent object.

        :param name: The name of the agent.
        :param persona: The persona of the agent.
        :param llm: The large language model to use.
        :param prompt_strategy: The prompt strategy to use.
        """

        self._name = context.agent_name
        self._persona = context.agent_persona
        self._llm = llm
        self._prompt_strategy = prompt_strategy

    def get_name(self) -> str:
        """
        Get the name of the agent.

        :return: The name of the agent.
        """

        return self._name

    def get_persona(self) -> str:
        """
        Get the persona of the agent.

        :return: The persona of the agent.
        """

        return self._persona

    @abstractmethod
    def run(self, instruction: str | Message, short_memory: ShortMemory) -> str:
        """
        Run the agent. This is the main method that should be used to run the
        agent.

        :param instruction: The instruction to run the agent with - user message
            or a task description in the first iteration, observations in the
            following iterations.
        :param short_memory: The short memory object to use for the agent.
        :return: The response from the agent.
        """
