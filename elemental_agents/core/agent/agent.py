"""
Interface for the agent executor class.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from elemental_agents.core.agent_logic.agent_logic import AgentLogic
from elemental_agents.core.memory.short_memory import ShortMemory
from elemental_agents.core.toolbox.toolbox import ToolBox
from elemental_agents.llm.data_model import Message
from elemental_agents.utils.config import ConfigModel
from elemental_agents.utils.utils import extract_tag_content


class Agent(ABC):
    """
    Generic executor class for the agent.
    """

    def __init__(
        self,
        agent_logic: AgentLogic,
        short_memory_capacity: int = -1,
        toolbox: ToolBox = None,
        termination_sequence: str = None,
    ) -> None:
        """
        Agent class. The class is meant to accommodate all operations and methods
        that are common for all agents and simplify the agent implementation to only
        include agent realization of the prompt strategy.

        :param agent_logic: The agent object to use.
        :param short_memory_capacity: The short memory capacity of the agent.
        :param toolbox: The toolbox object to use for the agent.
        :param termination_sequence: The termination sequence to use for the agent
        """

        self._agent_logic = agent_logic
        self._short_memory_capacity = short_memory_capacity
        self._short_memory = ShortMemory(short_memory_capacity)
        self._toolbox = toolbox
        self._termination_sequence = termination_sequence

        config = ConfigModel()
        self._max_iterations = config.max_agent_iterations

    def get_agent_name(self) -> str:
        """
        Get the agent's name.

        :return: The agent's name.
        """
        return self._agent_logic.get_name()

    def get_agent_persona(self) -> str:
        """
        Get the agent's persona.

        :return: The agent's persona.
        """
        return self._agent_logic.get_persona()

    def reset_short_memory(self) -> None:
        """
        Reset the agent's short memory.
        """
        self._short_memory.reset()

    def get_all_messages(self) -> List[Message]:
        """
        Get all messages from the agent's short memory.

        :return: The list of messages.
        """

        history = self._short_memory.get_all()
        return history

    def select_tag_from_response(self, response: str, tag: str) -> str:
        """
        Select a specific tag from the response.

        :param response: The response to remove the tag from.
        :param tag: The tag to remove from the response.
        :return: The response with the tag removed.
        """
        content = extract_tag_content(response, tag)
        response = content[0]
        return response

    def get_termination_sequence(self) -> str:
        """
        Get the termination sequence for the agent.

        :return: The termination sequence.
        """

        return self._termination_sequence

    def describe_toolbox(self) -> str:
        """
        Describe the toolbox for the agent.

        :return: The description of the toolbox.
        """
        return self._toolbox.describe_short()

    def update_prompt_strategy(self, agent_description: dict) -> None:
        """
        Update the prompt strategy for the agent.

        :param agent_description: The description of the agent.
        """
        from elemental_agents.core.agent_logic.conv_planreact_agent import (  # pylint: disable=import-outside-toplevel
            ConvPlanReActAgentLogic,
        )

        if isinstance(self._agent_logic, ConvPlanReActAgentLogic):
            self._agent_logic.update_prompt_strategy(agent_description)

    @abstractmethod
    def process_response(self, response: str) -> str:
        """
        Process the response from the LLM. This method should be used to
        post-process the raw response and extract only the final result.

        :param response: The raw response from the LLM.
        :return: The processed response.
        """

    @abstractmethod
    def run(self, task: str | List[str], input_session: str) -> str:
        """
        Run the agent's main logic function in the iterative fashion.

        :param task: The task to run the agent with.
        :param input_session: The input session for the agent.
        """

    @abstractmethod
    def run_instruction(
        self, instruction: str, original_instruction: str = "", input_session: str = ""
    ) -> Tuple[bool, str]:
        """
        Run the agent's main logic function on a single instruction.

        :param instruction: The instruction to run the agent with.
        :param original_instruction: The original instruction to run the agent with.
        :param input_session: The input session for the agent.
        :return: Tuple of a boolean indicating if the response is terminal and the response.
        """

    @abstractmethod
    def run_instruction_inference(
        self, instruction: str, original_instruction: str = "", input_session: str = ""
    ) -> str:
        """
        Run the agent's main logic function on a single instruction inference,
        only LLM portion of agent logic.

        :param instruction: The instruction to run the agent with.
        :param original_instruction: The original instruction to run the agent
            with.
        :param input_session: The input session for the agent.
        :return: The response from the agent.
        """

    @abstractmethod
    def run_instruction_action(self, agent_response: str) -> str:
        """
        Run the agent's requested actions.

        :param agent_response: The raw response from the agent.
        :return: Result of the actions.
        """
