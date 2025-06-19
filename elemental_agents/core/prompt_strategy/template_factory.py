"""
This module provides a factory for creating templates based on the provided context.

Usage in agent:

class SimpleAgentLogic(GenericAgentLogic):
    def __init__(self, config: BaseAgentConfig):
        self._template = TemplateFactory.create_template(
            config.context,
            config.template,
            "Simple.template"
        )
"""

from typing import Optional, Union

from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.prompt_strategy.prompt_template import (
    FileTemplate,
    StringTemplate,
)


class TemplateFactory:
    """
    Factory for creating templates with consistent logic
    based on the provided context.
    """

    @staticmethod
    def create_template(
        context: AgentContext,
        template: Optional[str] = None,
        default_template_name: str = "default.template",
    ) -> Union[StringTemplate, FileTemplate]:
        """
        Create a template based on the provided context.
        :param context: The agent context containing model data.
        :param template: Optional string template to use.
        :param default_template_name: Default template file name if no string template is provided.
        :return: An instance of StringTemplate or FileTemplate.
        """
        context_dict = context.model_dump()

        if template:
            return StringTemplate(context_dict, template)
        return FileTemplate(context_dict, default_template_name)
