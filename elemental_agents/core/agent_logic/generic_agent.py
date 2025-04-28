"""
Implementation of the generic agent class. This agent is used to interact with
the LLM and the prompt strategy. It should represent most generic iterative
agent class.
"""

from typing import Optional

from loguru import logger

from elemental_agents.core.agent_logic.agent_logic import AgentLogic
from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.memory.short_memory import ShortMemory
from elemental_agents.core.prompt_strategy.prompt import PromptStrategy
from elemental_agents.llm.llm import LLM


class GenericAgentLogic(AgentLogic):
    """
    Generic agent class. This agent is used to interact with the LLM and the
    prompt strategy. It should represent most generic iterative agent class.
    """

    def __init__(
        self,
        context: AgentContext,
        model: LLM,
        prompt_strategy: PromptStrategy,
        stop_word: Optional[str] = None,
    ) -> None:
        """
        Initialize the GenericAgent object.

        :param context: The name and persona of the agent.
        :param model: The large language model to use.
        :param prompt_strategy: The prompt strategy to use.
        :param stop_word: The stop word to use for the model to stop the
            generation (included in the strategy template).
        """

        super().__init__(context, model, prompt_strategy)

        self._stop_word = stop_word

    def process_response(self, response: str) -> str:
        """
        Process the response from the LLM. This method should be used to
        post-process the raw response from the LLM for better control of the
        adherence to the prompt strategy.

        :param response: The raw response from the LLM.
        :return: The processed response.
        """

        # Add stop word to the response since we stopped the model generation
        # and it is required by the prompt strategy. Only enter the stop word
        # back if this is regular iteration and not the last one,
        # i.e. only if <action> is present.
        if self._stop_word:
            if not response.endswith(self._stop_word) and "<action>" in response:

                response += f"{self._stop_word}"

        return response

    def run(self, instruction: str, short_memory: ShortMemory) -> str:
        """
        Run the generic agent. This is the main method that should be used to
        run the agent. It represents a single iteration of the agent.

        :param instruction: The instruction to run the agent with - user message
            or a task description in the first iteration, observations in the
            following iterations.
        :param short_memory: The short memory object to use for the agent.
        :return: The response from the agent.
        """

        logger.debug(f"Running the agent with instruction: {instruction}")

        # History
        history = short_memory.get_all()

        # Assemble the messages
        msgs = self._prompt_strategy.render(instruction, history)

        logger.debug(f"Messages to LLM: {msgs}")

        # Run the LLM
        output = self._llm.run(msgs, self._stop_word)

        logger.debug(f"Agents raw response: {output}")

        # Process the response
        response = self.process_response(output)

        return response


if __name__ == "__main__":

    from elemental_agents.core.prompt_strategy.prompt_template import FileTemplate
    from elemental_agents.core.prompt_strategy.react_prompt import ReactPrompt
    from elemental_agents.llm.llm_factory import LLMFactory

    # Test the GenericAgent

    agent_context = AgentContext(
        agent_name="TestAgent",
        agent_persona="Researcher always following scientific method",
    )

    llm_factory = LLMFactory()
    llm_model = llm_factory.create()

    template = FileTemplate(agent_context.model_dump(), "ReAct.template")
    strategy = ReactPrompt(
        system_template=template,
        tool_dictionary={},
    )

    sm = ShortMemory(capacity=5)

    agent = GenericAgentLogic(
        context=agent_context,
        model=llm_model,
        prompt_strategy=strategy,
        stop_word="<PAUSE>",
    )

    INSTRUCTION = "Why is the sky blue?"
    result = agent.run(instruction=INSTRUCTION, short_memory=sm)

    logger.info(f"Agent response: {result}")
