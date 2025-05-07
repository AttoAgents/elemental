"""
Executor class for the generic agent.
"""

import json
from typing import List, Tuple

from loguru import logger

from elemental_agents.core.agent.agent import Agent
from elemental_agents.core.agent_logic.agent_logic import AgentLogic
from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.toolbox.toolbox import ToolBox
from elemental_agents.llm.data_model import Message
from elemental_agents.observability.observer import observer
from elemental_agents.utils.exceptions import AgentException
from elemental_agents.utils.utils import (
    extract_tag_content,
    extract_unclosed_tag_content,
)


class GenericAgent(Agent):
    """
    Executor class for the generic agent.
    """

    def __init__(
        self,
        agent_logic: AgentLogic,
        short_memory_capacity: int,
        toolbox: ToolBox,
        termination_sequence: str,
    ) -> None:
        """
        Initialize the GenericExecutor to run the generic agent.

        :param agent_logic: The agent object to use.
        :param short_memory_capacity: The short memory capacity of the agent.
        :param toolbox: The toolbox object to use for the agent.
        :param termination_sequence: The termination sequence to use for the agent
        """

        super().__init__(
            agent_logic, short_memory_capacity, toolbox, termination_sequence
        )

    def process_response(self, response: str) -> str:
        """
        Process the response from the LLM. This method should be used to
        post-process the raw response and extract only the final result
        section e.g. <result> section of the response.

        :param response: The raw response from the LLM.
        :return: The final result from the agent.
        """

        # Extract the result section from the response
        final_response = extract_tag_content(response, "result")

        # Only one response is expected
        try:
            processed = final_response[0]
        except IndexError:
            logger.error("No fully closed <result> found in the agent's response.")
            final_response = extract_unclosed_tag_content(response, "result")
            processed = final_response[0]

        return processed

    def run(self, task: str | List[str], input_session: str) -> str:
        """
        Run the generic agent.

        :param task: The task to run the agent on.
        :param input_session: The input session for the agent.
        :return: The final result from the agent.
        """

        # Check if the task is a list of instructions. The intended use case is
        # to allow conversation history to be passed to the agent. The last
        # instruction in the list is the main instruction to run the agent on.
        if isinstance(task, list):
            instruction = task[-1]

            # Reset the short memory and add all the instructions except the
            # last one, this recovers the history of the conversation for the
            # follow up instructions
            self._short_memory.reset()
            mgs_type = ["user", "assistant"]
            idx = 0
            for item in task[:-1]:
                msg = Message(role=mgs_type[idx % 2], content=item)
                self._short_memory.add(msg)
                idx += 1
        else:
            instruction = task
            # self._short_memory.reset()

        original_instruction = instruction

        # Main agent loop
        iteration = 0

        while True:

            # Check if the agent has reached the maximum number of iterations
            if iteration > self._max_iterations:
                logger.error("Agent reached maximum iterations.")
                raise AgentException("Agent reached maximum iterations.")

            # Run the instruction
            (terminate, result) = self.run_instruction(
                instruction, original_instruction, input_session
            )

            # Check if the agent has completed the task
            if terminate:
                return result

            instruction = result
            iteration += 1

    def parse_action(self, result: str) -> List[str]:
        """
        Parse the actions from the agent's response. The action section is
        expected to be in the format: <action> {
            "name" : "tool_name", "parameters": {
                "param1": "value1", "param2": "value2"
            }
        } </action>

        :param result: The agent's response.
        :return: The list of actions to execute.
        """
        action = extract_tag_content(result, "action")

        return action

    def run_instruction(
        self, instruction: str, original_instruction: str = "", input_session: str = ""
    ) -> Tuple[bool, str]:
        """
        Run the generic agent single iteration.

        :param instruction: The task/instruction to run the agent on.
        :param original_instruction: The original instruction to run the agent on.
        :param input_session: The input session for the agent.
        :return: The result from the agent's iteration. The result is a tuple
            with the first element being a boolean indicating if the agent has
            completed the task and the second element being the result from the
            agent's iteration.
        """

        # Run the agent's inference portion
        result = self.run_instruction_inference(
            instruction, original_instruction, input_session
        )

        # Check if the agent has completed the task
        if self._termination_sequence in result:

            # Process the final result
            final_result = self.process_response(result)
            return (True, final_result)

        elif "<action>" in result:

            # Run the agent's action portion
            observation = self.run_instruction_action(result)

            # Update the instruction
            iteration_result = f"<observation>\n{observation}\n</observation>"

            return (False, iteration_result)

        elif self._relaxed_react:
            # Agent most likely did not follow the <action> or <result> tags
            # in the response. We will flag this as <result> and return the
            # result as is. Relaxed ReAct mode is enabled - Function calling 
            # agent behavior.
            logger.debug(
                "Agent did not follow the <action> or <result> tags in the response."
                "Returning the result as is due to relaxed_react mode."
            )
            final_result = result
            return (True, final_result)
        else:
            # Agent most likely did not follow the <action> or <result> tags
            # in the response. We will flag this as result and return the
            # result as is with failure.
            logger.error(
                "Agent did not follow the <action> or <result> tags in the response."
                "Returning the result as is."
            )
            final_result = result
            return (False, final_result)

    def run_instruction_inference(
        self, instruction: str, original_instruction: str = "", input_session: str = ""
    ) -> str:
        """
        Run the generic agent single iteration for inference.

        :param instruction: The task/instruction to run the agent on.
        :param original_instruction: The original instruction to run the agent on.
        :param input_session: The input session for the agent.
        :return: Raw response from the agent.
        """

        agent_name = self._agent_logic.get_name()

        new_user_message = Message(role="user", content=instruction)
        observer.log_message(
            input_session=input_session,
            message=new_user_message,
            agent_name=agent_name,
            task_description=original_instruction,
        )

        # Run the agent's logic
        result = self._agent_logic.run(instruction, self._short_memory)

        new_assistant_message = Message(role="assistant", content=result)
        observer.log_message(
            input_session=input_session,
            message=new_assistant_message,
            agent_name=agent_name,
            task_description=original_instruction,
        )

        # Update memory
        self._short_memory.add(new_user_message)
        self._short_memory.add(new_assistant_message)

        return result

    def run_instruction_action(self, agent_response: str) -> str:
        """
        Run the generic agent single iteration for action.

        :param agent_response: The agent's response.
        :return: Result of the action execution.
        """

        observation = ""

        # Run actions based on the agent's responses
        if "<action>" in agent_response:
            try:
                actions = self.parse_action(agent_response)

                logger.debug(f"Actions: {actions}")

                for item in actions:

                    action = json.loads(item)

                    logger.debug(f"Action - Executing tool: {action}")

                    action_result = self._toolbox.call_tool(
                        action["name"], json.dumps(action["parameters"])
                    )

                    logger.debug(f"Action result: {action_result}")

                    observation += f'{action["name"]}: {action_result}\n'
            except Exception as e:
                logger.error(f"Error executing actions: {e}")
                observation = (
                    f"Error in executing requested tool." f"Captured error: {e}"
                )

        else:
            logger.error("No action found in the agent's response.")
            observation = (
                    "No action found in the agent's response. "
                    f"Specify <action> or repeat response in {self._termination_sequence} XML tags to give final result."
                )

        return observation


if __name__ == "__main__":

    from rich.console import Console

    from elemental_agents.core.agent_logic.generic_agent import GenericAgentLogic
    from elemental_agents.core.prompt_strategy.prompt_template import FileTemplate
    from elemental_agents.core.prompt_strategy.react_prompt import ReactPrompt
    from elemental_agents.llm.llm_factory import LLMFactory

    console = Console()

    # LLM
    llm_factory = LLMFactory()
    llm = llm_factory.create()

    # Toolbox
    box = ToolBox()
    box.register_tool_by_name("Calculator")
    box.register_tool_by_name("ListFiles")

    # Context
    context = AgentContext(
        agent_name="TestAgent",
        agent_persona="Researcher always following scientific method",
    )

    # Prompt strategy
    template = FileTemplate(context.model_dump(), "ReAct.template")
    strategy = ReactPrompt(system_template=template, tool_dictionary=box.describe())

    # Setup the agent
    test_agent_logic = GenericAgentLogic(
        context=context,
        model=llm,
        prompt_strategy=strategy,
        stop_word="<PAUSE>",
    )

    # Execute the agent

    # Executor
    executor = GenericAgent(test_agent_logic, -1, box, "<result>")

    TEST_TASK = "Why is the sky blue?"
    test_result = executor.run(TEST_TASK, "TestSession")

    console.print(test_result)

    console.print(executor.get_all_messages())
