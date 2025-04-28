"""
Prompt strategy for ConvPlanReAct agent.
"""

from typing import Dict, List

from loguru import logger

from elemental_agents.core.prompt_strategy.prompt import IterativePrompt
from elemental_agents.core.prompt_strategy.prompt_template import PromptTemplate
from elemental_agents.llm.data_model import Message


class ConvPlanReactPrompt(IterativePrompt):
    """
    Conversational modification of the ReAct prompt strategy. This strategy is
    addition to tool description brings other agents in the unit and describes
    them in a similar way to the tools.
    """

    def __init__(
        self,
        system_template: PromptTemplate,
        tool_dictionary: Dict[str, str],
        agents_dictionary: Dict[str, str],
    ) -> None:
        """
        Constructor for the ReactPrompt class. The system template is a
        PromptTemplate object that will be used to render the system messages
        which describes the iterative behavior of the model. The context is a
        dictionary with all the variables needed for the template. The tool
        dictionary is a dictionary with the tool names and their descriptions.
        The termination sequence is a string that signals the end ReAct loop.
        The stop words is a list of words that are used stop LLM generation
        at a given iteration.

        :param system_template: PromptTemplate object to render the system
            messages
        :param tool_dictionary: Dictionary with the tool names and their
            descriptions
        :param agents_dictionary: Dictionary with the agent names and their
            descriptions
        """

        super().__init__(system_template=system_template)

        self._tool_dictionary = tool_dictionary
        self._system_template[0]._context["toolbox_description"] = self._tool_dictionary
        self._system_template[0]._context["agents_description"] = agents_dictionary

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

        result = []

        system = self.render_system()
        system_message = Message(role="system", content=system)

        # System message
        result.append(system_message)

        # If history is provided, add all the messages to the result
        if history:
            result.extend(history)

        # User message
        user_message = Message(role="user", content=instruction)
        result.append(user_message)

        return result


if __name__ == "__main__":

    from .prompt_template import FileTemplate

    agent_context = {
        "agent_name": "ResearchAgent",
        "agent_persona": "Researcher always following scientific method.",
    }

    agent_system_template = FileTemplate(agent_context, "ReAct.template")
    agent_prompt = ConvPlanReactPrompt(
        system_template=agent_system_template, tool_dictionary={}, agents_dictionary={}
    )

    react_result = agent_prompt.render("What is the weather today?")

    logger.debug(f"Agent prompt result: {react_result}")
