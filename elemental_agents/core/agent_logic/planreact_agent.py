"""
PlanReAct Agent class.
"""

from typing import Optional

from loguru import logger

from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.generic_agent import GenericAgentLogic
from elemental_agents.core.prompt_strategy.prompt_template import (
    FileTemplate,
    StringTemplate,
)
from elemental_agents.core.prompt_strategy.react_prompt import ReactPrompt
from elemental_agents.core.toolbox.toolbox import ToolBox
from elemental_agents.llm.llm import LLM


class PlanReActAgentLogic(GenericAgentLogic):
    """
    Simple definition of a PlanReAct type agent.
    """

    def __init__(
        self,
        model: LLM,
        context: AgentContext,
        toolbox: ToolBox,
        template: Optional[str] = None,
    ) -> None:
        """
        Initialize the PlanReAct Agent object for simple definition of standard
        PlanReAct type agents.

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
            self._template = FileTemplate(self._context, "PlanReAct.template")

        # Reuse ReAct prompt strategy type for PlanReAct agent with different
        # template file including the <planning> stage
        self._strategy = ReactPrompt(
            system_template=self._template, tool_dictionary=self._toolbox.describe()
        )

        super().__init__(
            context=context,
            model=model,
            prompt_strategy=self._strategy,
            stop_word="<PAUSE>",
        )
        logger.debug(f"PlanReActAgent initialized with context: {self._context}")


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
        agent_name="PlanReActAgent",
        agent_persona="Researcher always following scientific method",
    )

    # Setup the agent
    agent = PlanReActAgentLogic(llm, my_context, my_toolbox)

    # Execute the agent
    short_memory = ShortMemory()

    INSTRUCTION = "What is square root of 25?"
    result = agent.run(INSTRUCTION, short_memory)

    logger.info(f"Agent response: {result}")
