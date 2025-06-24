"""
Verifier Agent class.
"""

from typing import Optional

from loguru import logger

from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.generic_agent import GenericAgentLogic
from elemental_agents.core.prompt_strategy.basic_prompt import BasicPrompt
from elemental_agents.core.prompt_strategy.template_factory import TemplateFactory
from elemental_agents.llm.llm import LLM
from elemental_agents.utils.utils import extract_tag_content


class VerifierAgentLogic(GenericAgentLogic):
    """
    Simple definition of a Verifier type agent.
    """

    def __init__(
        self,
        model: LLM,
        context: AgentContext,
        default_template_name: str = "Verifier.template",
        template: Optional[str] = None,
    ) -> None:
        """
        Initialize the Verifier Agent object.

        :param model: The LLM object to use for the agent.
        :param context: The context (name, persona) for the agent.
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
        logger.debug(f"VerifierAgent initialized with context: {context}")

    def process_response(self, response: str) -> str:
        """
        Extract the result section from the response

        :param response: The raw response from the LLM.
        :return: The final <verification> tag from the agent's response.
        """
        result = extract_tag_content(response, "verification")[0]

        return result


if __name__ == "__main__":

    from rich.console import Console

    from elemental_agents.core.memory.short_memory import ShortMemory
    from elemental_agents.llm.llm_factory import LLMFactory

    console = Console()

    llm_factory = LLMFactory()
    llm = llm_factory.create()

    agent_context = AgentContext(
        agent_name="VerifierAgent",
        agent_persona="Result verifier.",
    )

    # Setup the Simple Agent
    agent = VerifierAgentLogic(llm, agent_context)

    # Execute the agent
    short_memory = ShortMemory()

    INSTRUCTION = "What is the difference between BMW X3 and X4?"
    RESPONSE_TO_VERIFY = """The BMW X3 is a compact luxury crossover SUV
                         manufactured by German automaker BMW since 2003. Based
                         on the BMW 3 Series platform, and now in its third
                         generation, BMW markets the crossover as a Sports
                         Activity Vehicle, the company's proprietary descriptor
                         for its X-line of vehicles. The first generation of the
                         X3 was designed by BMW in conjunction with Magna Steyr
                         of Graz, Austria—who also manufactured all X3s under
                         contract to BMW. The second generation of the X3 was
                         revealed on 14 July 2010, and went on sale in November
                         2010. The third generation X3 was revealed in June of
                         2017. The BMW X4 is a compact luxury SUV introduced in
                         2014, manufactured by German automaker BMW at its
                         United States factory in South Carolina. It was
                         initially announced in 2013 as a concept car and was
                         officially presented at the 2014 New York International
                         Auto Show. The X4 was launched in 2014 and is based on
                         the same platform as the BMW X3."""
    full_instruction = (
        f"<request>{INSTRUCTION}</request>\n <response>{RESPONSE_TO_VERIFY}</response>"
    )

    final_result = agent.run(full_instruction, short_memory)

    logger.debug("Agent's raw response:")
    console.print(final_result)
