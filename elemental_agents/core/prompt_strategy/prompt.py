"""
Prompt strategy module.
"""

from abc import ABC, abstractmethod
from typing import List

from elemental_agents.core.prompt_strategy.prompt_template import PromptTemplate
from elemental_agents.llm.data_model import Message


class PromptStrategy(ABC):
    """
    Interface for the prompt strategy object.
    """

    def __init__(self, system_template: List[PromptTemplate] | PromptTemplate) -> None:
        """
        Initialize the PromptStrategy object with the given parameters.

        :param system_template: The system template to use.
        """
        if isinstance(system_template, PromptTemplate):
            self._system_template = [system_template]
        else:
            self._system_template = system_template

    @abstractmethod
    def render(
        self, instruction: str, history: List[Message] | None = None
    ) -> List[Message]:
        """
        Prepare the list of message for the given input and history using the system template.

        :param instruction: The input to process.
        :param history: The history of messages.
        :return: The list of messages to send to LLM.
        """


class NonIterativePrompt(PromptStrategy):
    """
    Non-iterative prompt strategy.
    """

    def __init__(self, system_template: PromptTemplate) -> None:
        """
        Initialize basic non-iterative prompt strategy object with the given template.

        :param system_template: The system template to use.
        """

        super().__init__(system_template=system_template)

    # @abstractmethod
    # def render(self, instruction: str, history: List[Message] | None = None) -> List[Message]:
    #     """
    #     Prepare the list of message for the given input and history using the system template.

    #     :param input: The input to process.
    #     :param history: The history of messages.
    #     :return: The list of messages to send to LLM.
    #     """
    #     pass


class IterativePrompt(PromptStrategy):
    """
    Iterative prompt strategy.
    """

    def __init__(self, system_template: PromptTemplate) -> None:
        """
        Initialize iterative prompt strategy object with the given template.

        :param system_template: The system template to use.
        """

        super().__init__(system_template=system_template)

    # @abstractmethod
    # def render(self, input: str, history: List[Message] | None = None) -> List[Message]:
    #     """
    #     Prepare the list of message for the given input and history using the
    #     system template.

    #     :param input: The input to process.
    #     :param history: The history of messages.
    #     :return: The list of messages to send to LLM.
    #     """
    #     pass


class ChainedPrompt(PromptStrategy):
    """
    Chained prompt strategy.
    """

    def __init__(self, system_template: List[PromptTemplate] | PromptTemplate) -> None:
        """
        Initialize chained prompt strategy object with the set of given templates.

        :param system_template: The system template or list of templates to use.
        """
        super().__init__(system_template=system_template)  # type: ignore

    # @abstractmethod
    # def render(self, input: str, history: List[Message] | None = None) -> List[Message]:
    #     """
    #     Prepare the list of message for the given input and history using the system template.

    #     :param input: The input to process.
    #     :param history: The history of messages.
    #     :return: The list of messages to send to LLM.
    #     """
    #     pass
