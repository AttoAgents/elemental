"""
Basic prompt class.
"""

from typing import List

from loguru import logger

from elemental_agents.core.prompt_strategy.prompt import NonIterativePrompt
from elemental_agents.core.prompt_strategy.prompt_template import PromptTemplate
from elemental_agents.llm.data_model import Message


class BasicPrompt(NonIterativePrompt):
    """
    Basic prompt strategy. This strategy assumes non-iterative prompts with only
    user and system exchanges where user messages are always provided as input.
    """

    def __init__(self, system_template: PromptTemplate) -> None:
        """
        Constructor for the BasicPrompt class. The system template is a
        PromptTemplate object that will be used to render the system messages.
        The context is a dictionary with all the variables needed for the
        template. In this class system message is always rendered on the fly.

        :param system_template: PromptTemplate object to render the system
            messages
        """

        super().__init__(system_template=system_template)

    def render_system(self) -> str:
        """
        Render the system message. This method is used to render the system
        message on the fly.

        :return: System message
        """

        prompt = self._system_template[0].render()
        return prompt

    def render(
        self, instruction: str, history: List[Message] | None = None
    ) -> List[Message]:
        """
        Render the prompt. The input is the user message that will be added to
        the history. The history is a list of Message objects with the previous
        messages in the conversation. History is optional. System prompt is
        rendered on the fly.

        :param input: User message
        :param history: List of previous messages in the conversation
        :return: List of Message objects with the system and user messages
        """

        result_prompt = []

        system = self.render_system()
        system_message = Message(role="system", content=system)

        user_message = Message(role="user", content=instruction)

        # System message
        result_prompt.append(system_message)

        # If history is provided, add all the messages to the result
        if history:
            result_prompt.extend(history)

        # User message
        result_prompt.append(user_message)

        return result_prompt


if __name__ == "__main__":

    from .prompt_template import FileTemplate

    context = {
        "agent_name": "ResearchAgent",
        "agent_persona": "Researcher always following scientific method",
    }

    test_system_template: PromptTemplate = FileTemplate(context, "test.template")
    basic_prompt = BasicPrompt(system_template=test_system_template)

    result = basic_prompt.render("Hello")

    logger.info(result)
