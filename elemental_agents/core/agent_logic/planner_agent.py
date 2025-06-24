"""
Planner agent implementation.
"""

from typing import Optional

from loguru import logger

from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.generic_agent import GenericAgentLogic
from elemental_agents.core.prompt_strategy.basic_prompt import BasicPrompt

from elemental_agents.core.prompt_strategy.template_factory import TemplateFactory
from elemental_agents.llm.llm import LLM
from elemental_agents.utils.utils import extract_tag_content


class PlannerAgentLogic(GenericAgentLogic):
    """
    Planer logic agent class.
    """

    def __init__(
        self, model: LLM, context: AgentContext,
        default_template_name: str = "Planner.template",
        template: Optional[str] = None
    ) -> None:
        """
        Initialize the Planner Agent object.

        :param llm: The LLM object to use for the agent.
        :param context: The context (name, persona) for the agent.
        :param default_template_name: The default template file name to use if
            no template is provided.
        :param template: The template string to use for the agent.
        """

        self._template = TemplateFactory.create_template(
            context=context, template=template, default_template_name=default_template_name
        )
        self._strategy = BasicPrompt(system_template=self._template)

        super().__init__(
            context=context, model=model, prompt_strategy=self._strategy, stop_word=None
        )
        logger.debug(f"PlannerAgentLogic initialized with context: {context}")

    def process_response(self, response: str) -> str:
        """
        Extract the result section from the response

        :param response: The raw response from the LLM.
        :return: The final <plan> tag from the agent's response.
        """
        output = extract_tag_content(response, "plan")[0]

        return output


if __name__ == "__main__":

    import json

    from elemental_agents.core.memory.short_memory import ShortMemory
    from elemental_agents.llm.llm_factory import LLMFactory

    llm_factory = LLMFactory()
    llm = llm_factory.create()

    agent_context = AgentContext(
        agent_name="PlannerAgent",
        agent_persona="Helpful planner.",
    )

    # Setup the Simple Agent
    agent = PlannerAgentLogic(model=llm, context=agent_context)

    # Execute the agent
    short_memory = ShortMemory()

    INSTRUCTION = "What is the difference between BMW X3 and X4?"
    result = agent.run(INSTRUCTION, short_memory)

    logger.debug("Agent's raw response:")
    logger.debug(result)

    # Parse the plan from the agent's response
    plan = extract_tag_content(result, "JSON")
    parsed_plan = [json.loads(p) for p in plan]
    logger.debug("Agent's parsed plan:")
    logger.debug(parsed_plan)
