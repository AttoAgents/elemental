"""
Simple non-iterative agent class definition.
"""

from typing import Optional

from loguru import logger

from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.generic_agent import GenericAgentLogic
from elemental_agents.core.prompt_strategy.basic_prompt import BasicPrompt

from elemental_agents.core.prompt_strategy.template_factory import TemplateFactory
from elemental_agents.llm.llm import LLM


class SimpleAgentLogic(GenericAgentLogic):
    """
    Simple definition of a Simple type agent.
    """

    def __init__(
        self,
        model: LLM,
        context: AgentContext,
        default_template_name: str = "Simple.template",
        template: Optional[str] = None
    ) -> None:
        """
        Initialize the Simple Agent object for simple definition of standard
        agents.

        :param model: The LLM object to use for the agent.
        :param context: The context (name, persona) for the agent.
        :param default_template_name: The default template file name to use if
            no template is provided.
        :param template: The template to use for the agent.
        """

        self._template = TemplateFactory.create_template(
            context=context, template=template, default_template_name=default_template_name
        )
        self._strategy = BasicPrompt(system_template=self._template)

        super().__init__(
            context=context, model=model, prompt_strategy=self._strategy, stop_word=None
        )
        logger.debug(f"SimpleAgent initialized with context: {context}")


if __name__ == "__main__":

    from rich.console import Console

    from elemental_agents.core.memory.short_memory import ShortMemory
    from elemental_agents.llm.llm_factory import LLMFactory

    console = Console()

    llm_factory = LLMFactory()
    llm = llm_factory.create()

    agent_context = AgentContext(
        agent_name="SimpleAgent", agent_persona="Simple and informative agent."
    )

    # Setup the Simple Agent
    agent = SimpleAgentLogic(llm, agent_context)

    # Execute the agent
    short_memory = ShortMemory()

    INSTRUCTION = "Why is the sky blue?"
    result = agent.run(INSTRUCTION, short_memory)

    console.print(result)
