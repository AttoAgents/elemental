"""
RePlanner agent implementation.
"""

from typing import Optional

from loguru import logger

from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.generic_agent import GenericAgentLogic
from elemental_agents.core.prompt_strategy.basic_prompt import BasicPrompt
from elemental_agents.core.prompt_strategy.template_factory import TemplateFactory
from elemental_agents.llm.llm import LLM


class ReplannerAgentLogic(GenericAgentLogic):
    """
    RePlanner agent class.
    """

    def __init__(
        self,
        model: LLM,
        context: AgentContext,
        default_template_name: str = "RePlanner.template",
        template: Optional[str] = None,
    ) -> None:
        """
        Initialize the Planner Agent object.

        :param llm: The LLM object to use for the agent.
        :param context: The context dictionary for the agent.
        :param default_template_name: The default template file name to use if
            no template is provided.
        :param template: The template string to use for the agent.
        """

        self._template = TemplateFactory.create_template(
            context=context,
            template=template,
            default_template_name=default_template_name,
        )
        self._strategy = BasicPrompt(system_template=self._template)

        super().__init__(
            context=context, model=model, prompt_strategy=self._strategy, stop_word=None
        )
        logger.debug(f"RePlannerAgent initialized with context: {context}")

    def process_response(self, response: str) -> str:
        """
        Extract the result section from the response

        :param response: The raw response from the LLM.
        :return: The final response including the <plan> tag from the agent's response.
        """

        output = response

        return output
