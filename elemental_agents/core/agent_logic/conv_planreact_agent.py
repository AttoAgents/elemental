"""
ConvPlanReAct Agent class.
"""

from typing import Dict, Optional

from loguru import logger

from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.generic_agent import GenericAgentLogic
from elemental_agents.core.prompt_strategy.conversational_planreact_prompt import (
    ConvPlanReactPrompt,
)
from elemental_agents.core.prompt_strategy.prompt_template import (
    FileTemplate,
    StringTemplate,
)
from elemental_agents.core.toolbox.toolbox import ToolBox
from elemental_agents.llm.llm import LLM


class ConvPlanReActAgentLogic(GenericAgentLogic):
    """
    Simple definition of a ConvPlanReAct type agent.
    """

    def __init__(
        self,
        model: LLM,
        context: AgentContext,
        toolbox: ToolBox,
        template: Optional[str] = None,
    ) -> None:
        """
        Initialize the ConvPlanReAct Agent object for simple definition of standard
        ConvPlanReAct type agents.

        :param model: The LLM object to use for the agent.
        :param context: The context (name, persona) for the agent.
        :param toolbox: The toolbox object to use for the agent.
        :param template: The template string to use for the agent.
        """

        self._context = context.model_dump()
        self._toolbox = toolbox

        self._template: StringTemplate | FileTemplate

        if template:
            self._template = StringTemplate(self._context, template)
        else:
            self._template = FileTemplate(self._context, "ConvPlanReAct.template")

        # Reuse ReAct prompt strategy type for ConvPlanReAct agent with different
        # template file including the <planning> stage
        self._strategy = ConvPlanReactPrompt(
            system_template=self._template,
            tool_dictionary=self._toolbox.describe(),
            agents_dictionary={},
        )

        super().__init__(
            context=context,
            model=model,
            prompt_strategy=self._strategy,
            stop_word="<PAUSE>",
        )
        logger.debug(f"ConvPlanReActAgent initialized with context: {self._context}")

    def update_prompt_strategy(self, agents_dictionary: Dict[str, str]) -> None:
        """
        Update the prompt strategy for the agent.
        """
        self._strategy = ConvPlanReactPrompt(
            system_template=self._template,
            tool_dictionary=self._toolbox.describe(),
            agents_dictionary=agents_dictionary,
        )
        self._prompt_strategy = self._strategy


if __name__ == "__main__":

    from elemental_agents.core.memory.short_memory import ShortMemory
    from elemental_agents.llm.llm_factory import LLMFactory
    from elemental_agents.tools.calculator import Calculator, CalculatorParams
    from elemental_agents.tools.list_files import ListFiles, ListFilesParams

    llm_factory = LLMFactory()

    # User setup
    llm = llm_factory.create()

    my_toolbox = ToolBox()
    my_toolbox.register_tool("Calculator", Calculator, CalculatorParams)  # type: ignore
    my_toolbox.register_tool("ListFiles", ListFiles, ListFilesParams)  # type: ignore

    my_context = AgentContext(
        agent_name="ConvPlanReActAgent",
        agent_persona="Researcher always following scientific method",
    )

    # Setup the agent
    agent = ConvPlanReActAgentLogic(llm, my_context, my_toolbox)

    # Execute the agent
    short_memory = ShortMemory()

    INSTRUCTION = "What is square root of 25?"
    result = agent.run(INSTRUCTION, short_memory)

    logger.debug("Agent's raw response:")
    logger.info(f"Agent response: {result}")
