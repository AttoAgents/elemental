"""
Class representing a basic chain of prompts.
"""

from itertools import cycle
from typing import Generator, List

from elemental_agents.core.prompt_strategy.prompt import ChainedPrompt
from elemental_agents.core.prompt_strategy.prompt_template import PromptTemplate
from elemental_agents.llm.data_model import Message


class BasicChain(ChainedPrompt):
    """
    Basic chain of prompts.
    """

    def __init__(self, system_template: List[PromptTemplate] | PromptTemplate) -> None:

        super().__init__(system_template=system_template)

    def get_system_template(self) -> Generator[PromptTemplate, None, None]:
        """
        Get the system template.

        :return: The system template.
        """
        yield from cycle(self._system_template)

    def render(
        self, instruction: str, history: List[Message] | None = None
    ) -> List[Message]:

        template = self.get_system_template()

        result = []

        system = next(template).render()
        system_message = Message(role="system", content=system)

        user_message = Message(role="user", content=instruction)

        # System message
        result.append(system_message)

        # If history is provided, add all the messages to the result
        if history:
            result.extend(history)

        # User message
        result.append(user_message)

        return result
