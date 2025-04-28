"""
Executor for the simple agent. This executor is non-iterative and
executes the simple agent in a single step.
"""

from loguru import logger

from elemental_agents.core.agent.generic_agent import GenericAgent
from elemental_agents.core.agent_logic.agent_logic import AgentLogic
from elemental_agents.core.agent_logic.agent_model import AgentContext


class SimpleAgent(GenericAgent):
    """
    Simple non-iterative executor for the simple agent.
    """

    def __init__(self, agent: AgentLogic, short_memory_capacity: int = -1) -> None:
        """
        Initialize the SimpleExecutor to run the simple agent.

        :param agent: The agent object to use.
        :param short_memory_capacity: The short memory capacity of the agent.
        :param toolbox: The toolbox object to use for the agent.
        :param termination_sequence: The termination sequence to use for the agent
        """

        super().__init__(agent, short_memory_capacity, None, "")

    def process_response(self, response: str) -> str:
        """
        Process the response from the LLM. This method should be used to
        post-process the raw response and extract only the final result
        section e.g. <result> section of the response.

        :param response: The raw response from the LLM.
        :return: The final result from the agent.
        """

        # Extract the result section from the response
        final_response = response

        return final_response


if __name__ == "__main__":

    from rich.console import Console

    from elemental_agents.core.agent_logic.simple_agent import SimpleAgentLogic
    from elemental_agents.llm.llm_factory import LLMFactory

    console = Console()

    llm_factory = LLMFactory()
    llm = llm_factory.create()

    context = AgentContext(
        agent_name="AssistantAgent", agent_persona="Helpful and informative assistant."
    )

    # Setup the Simple Agent
    test_agent_logic = SimpleAgentLogic(llm, context)

    # Execute the agent

    executor = SimpleAgent(test_agent_logic, -1)

    INSTRUCTION = "Why is the sky blue?"
    test_result = executor.run(INSTRUCTION, "TestSession")

    console.print(test_result)

    INSTRUCTION = "Is it the same on Mars?"
    result = executor.run(INSTRUCTION, "TestSession")

    console.print(test_result)

    logger.debug("SimpleExecutor executed successfully.")
    logger.debug("Short memory contents:")
    console.print(executor.get_all_messages())
