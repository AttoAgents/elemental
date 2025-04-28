"""
Generic unit class to run a multiple agents in a collaborative fashion.
"""

from typing import List

from loguru import logger

from elemental_agents.core.agent_team.agent_team import AgentTeam
from elemental_agents.core.selector.agent_selector import (
    AgentSelector,
    AgentSelectorParameters,
)
from elemental_agents.llm.data_model import Message
from elemental_agents.utils.config import ConfigModel
from elemental_agents.utils.exceptions import AgentException
from elemental_agents.utils.utils import extract_tag_content


class GenericAgentTeam(AgentTeam):
    """
    Generic unit class to run a multiple agents in a collaborative fashion.
    """

    def __init__(self, selector: AgentSelector = None):
        """
        Initialize the generic unit with the specified agent selector.

        :param selector: The agent selector to use for selecting agents.
        """
        super().__init__(selector=selector)

        config = ConfigModel()
        self._max_iterations = config.max_multiagent_iterations

        logger.info("Generic unit initialized.")

    def run(self, task: str | List[str], input_session: str) -> str:
        """
        Run the generic unit with the specified task.

        :param task: The task to run the agents on.
        :param input_session: The input session to run the agents with.
        :return: The final result from the agents.
        """

        agents = self.get_agents()
        selector = self.get_selector()

        instruction = None
        if isinstance(task, list):
            instruction = task[-1]
        else:
            instruction = task

        # Select the first agent based on the instruction
        selector_parameters = AgentSelectorParameters(
            agents=agents, last_message=instruction
        )
        agent_executor = selector.select(selector_parameters)
        agent_name = agent_executor.get_agent_name()
        logger.info(f"Selected agent: {agent_name}")

        # Check if the task is a list of instructions. The intended use case is
        # to allow conversation history to be passed to the agent. The last
        # instruction in the list is the main instruction to run the agent on.
        if isinstance(task, list):

            # Reset the short memory and add all the instructions except the
            # last one, this recovers the history of the conversation for the
            # follow up instructions
            agent_executor.reset_short_memory()
            mgs_type = ["user", "assistant"]
            idx = 0
            for item in task[:-1]:
                msg = Message(role=mgs_type[idx % 2], content=item)
                agent_executor._short_memory.add(msg)
                idx += 1
        else:

            agent_executor.reset_short_memory()

        # Reset short memory of remaining agents
        for agent, executor in agents.items():
            if agent != agent_name:
                executor.reset_short_memory()

        original_instruction = instruction

        # Main agents iteration loop
        iteration = 0

        while True:

            # Check if the agents have reached the maximum number of iterations
            if iteration > self._max_iterations:
                logger.error("Agent reached maximum iterations.")
                raise AgentException(
                    "Agent reached maximum iterations in the multi-agent unit."
                )

            # Run the agent iteration
            agent_name = agent_executor.get_agent_name()
            logger.info(f"Running the agent iteration with agent: {agent_name}")

            agent_response = agent_executor.run_instruction_inference(
                instruction=instruction,
                original_instruction=original_instruction,
                input_session=input_session,
            )

            if agent_executor.get_termination_sequence() in agent_response:
                # Task done, response is the final response
                response = agent_response
                logger.info(f"Response: {response}")
                return response

            # Check if the agent includes other agent in the <next> tag
            selector_parameters = AgentSelectorParameters(
                agents=agents, last_message=agent_response
            )
            agent_executor = selector.select(selector_parameters)
            new_agent_name = agent_executor.get_agent_name()

            # Continue with @Self
            if new_agent_name == agent_name:

                # Run actions, observations become the next instruction
                observation = agent_executor.run_instruction_action(agent_response)
                instruction = f"<observation>\n{observation}\n</observation>"

            else:

                # If the agent executor has changed, update the instruction by
                # selecting <message> tag from the response, this becomes the new
                # instruction for the next agent

                new_message = self.select_message_for_next_agent(agent_response)
                instruction = f"From @{agent_name}: {new_message}"

            iteration += 1

    def select_message_for_next_agent(self, response: str) -> str:
        """
        Select the message for the next agent from the response.

        :param response: The response to select the message from.
        :return: The selected message.
        """
        content = extract_tag_content(response, "message")
        response = content[0]
        return response


if __name__ == "__main__":

    # Example usage of the generic unit
    from elemental_agents.core.agent.generic_agent import GenericAgent
    from elemental_agents.core.agent_logic.agent_model import AgentContext
    from elemental_agents.core.agent_logic.react_agent import ReActAgentLogic
    from elemental_agents.core.selector.agent_selector import IdentitySelector
    from elemental_agents.core.toolbox.toolbox import ToolBox
    from elemental_agents.llm.llm_factory import LLMFactory
    from elemental_agents.tools.calculator import Calculator, CalculatorParams
    from elemental_agents.tools.list_files import ListFiles, ListFilesParams

    # Agent setup
    llm_factory = LLMFactory()
    llm = llm_factory.create()

    toolbox = ToolBox()

    toolbox.register_tool("Calculator", Calculator, CalculatorParams)  # type: ignore
    toolbox.register_tool("ListFiles", ListFiles, ListFilesParams)  # type: ignore

    context = AgentContext(
        agent_name="TestAgent",
        agent_persona="Researcher always following scientific method",
    )

    agent_logic = ReActAgentLogic(model=llm, context=context, toolbox=toolbox)

    agent = GenericAgent(
        agent_logic=agent_logic,
        short_memory_capacity=-1,
        toolbox=toolbox,
        termination_sequence="<result>",
    )

    # Initialize the generic unit
    generic_team = GenericAgentTeam(selector=IdentitySelector(lead_agent="TestAgent"))
    generic_team.register_agent("generic_agent", agent, agent_type="react")
