"""
Plan verifier agent implementation.
"""

from typing import Optional

from loguru import logger

from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.generic_agent import GenericAgentLogic
from elemental_agents.core.prompt_strategy.basic_prompt import BasicPrompt

from elemental_agents.core.prompt_strategy.template_factory import TemplateFactory
from elemental_agents.llm.llm import LLM
from elemental_agents.utils.utils import extract_tag_content


class PlanVerifierAgentLogic(GenericAgentLogic):
    """
    Plan Verifier agent class.
    """

    def __init__(
        self, 
        model: LLM, 
        context: AgentContext,
        default_template_name: str = "Composer.template",
        template: Optional[str] = None
    ) -> None:
        """
        Initialize the Plan Verifier Agent object.

        :param llm: The LLM object to use for the agent.
        :param context: The context (name, persona) for the agent.
        :param default_template_name: The default template file name to use if
            no template is provided.
        :param template: The template file to use for the agent.
        """

        self._template = TemplateFactory.create_template(
            context=context, template=template, default_template_name=default_template_name
        )

        self._strategy = BasicPrompt(system_template=self._template)

        super().__init__(
            context=context, model=model, prompt_strategy=self._strategy, stop_word=None
        )
        logger.debug(f"PlanVerifierAgent initialized with context: {context}")

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
    llm = llm_factory.create("openai|gpt-4o-mini")

    agent_context = AgentContext(
        agent_name="PlanVerifierAgent",
        agent_persona="Plan verifier agent.",
    )

    # Setup the Simple Agent
    agent = PlanVerifierAgentLogic(model=llm, context=agent_context)

    # Execute the agent
    short_memory = ShortMemory()

    INSTRUCTION = """
    <instruction>
    Calculate the resistance of copper wire having a length of 1 km and diameter 0.5 mm. (Resistivity of copper 1.7e-8Ωm).
    </instruction>
    <plan>
    <JSON> { "id": 1, "description": "Calculate the radius of the copper wire using its diameter (0.5 mm).", "dependencies": [] } </JSON>
    <JSON> { "id": 2, "description": "Convert the length from kilometers to meters.", "dependencies": [] } </JSON>
    <JSON> { "id": 3, "description": "Calculate the cross-sectional area of the copper wire using its radius and π (approximately 3.14159).", "dependencies": [1] } </JSON>
    <JSON> { "id": 4, "description": "Convert the resistivity from Ωm to a value that matches the units of length and area.", "dependencies": [] } </JSON>
    <JSON> { "id": 5, "description": "Calculate the resistance using the formula R = ρL/A with appropriate unit conversions.", "dependencies": [2,3,4] } </JSON>
    </plan>
    """

    result = agent.run(INSTRUCTION, short_memory)

    logger.debug("Agent's raw response:")
    logger.debug(result)

    # Parse the plan from the agent's response
    plan = extract_tag_content(result, "JSON")
    parsed_plan = [json.loads(p) for p in plan]
    logger.debug("Agent's parsed plan:")
    logger.debug(parsed_plan)
