"""
Composer non-iterative agent class definition.
"""

from typing import Optional

from loguru import logger

from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.generic_agent import GenericAgentLogic
from elemental_agents.core.prompt_strategy.basic_prompt import BasicPrompt
from elemental_agents.core.prompt_strategy.prompt_template import (
    FileTemplate,
    StringTemplate,
)
from elemental_agents.llm.llm import LLM


class ComposerAgentLogic(GenericAgentLogic):
    """
    Simple definition of a Composer type agent.
    """

    def __init__(
        self, model: LLM, context: AgentContext, template: Optional[str] = None
    ) -> None:
        """
        Initialize the Composer Agent Logic object for simple
        definition of standard agents.

        :param model: The LLM object to use for the agent.
        :param context: The context (name, persona) for the agent.
        :param template: The template to use for the agent.
        """

        self._context = context.model_dump()
        self._template: StringTemplate | FileTemplate

        if template:
            self._template = StringTemplate(self._context, template)
        else:
            self._template = FileTemplate(self._context, "Composer.template")

        # Reuse Basic prompt strategy type for Simple agent with different
        # template file
        self._strategy = BasicPrompt(system_template=self._template)

        super().__init__(
            context=context, model=model, prompt_strategy=self._strategy, stop_word=None
        )
        logger.debug(f"ComposerAgentLogic initialized with context: {self._context}")
