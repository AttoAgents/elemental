"""
Agent Selector Factory class to create agent selector instances based on the selector name.
"""

from loguru import logger

from elemental_agents.core.selector.agent_selector import (
    AgentSelector,
    ConversationalSelector,
    IdentitySelector,
    IterativeSelector,
)
from elemental_agents.utils.config import ConfigModel


class AgentSelectorFactory:
    """
    Agent Selector Factory class to create agent selector instances based on the selector name.
    """

    def __init__(self) -> None:
        """
        Initialize the agent selector factory with the configuration model.
        """

        self._config = ConfigModel()

    def create(self, selector_name: str, lead_agent: str) -> AgentSelector:
        """
        Create an agent selector instance based on the selector name. If the selector name is
        not provided, the default selector is used that is specified in the
        configuration file.

        :param selector_name: The name of the selector to use.
        :return: An instance of the AgentSelector class.
        """

        local_selector_name = selector_name or self._config.unit_default_selector

        if local_selector_name == "identity":

            logger.debug("Creating IdentitySelector instance.")

            return IdentitySelector(lead_agent)

        if local_selector_name == "iterative":

            logger.debug("Creating IterativeSelector instance.")

            return IterativeSelector(lead_agent)

        if local_selector_name == "conversational":

            logger.debug("Creating ConversationalSelector instance.")

            return ConversationalSelector(lead_agent)

        # Default selector
        return ConversationalSelector(lead_agent)
