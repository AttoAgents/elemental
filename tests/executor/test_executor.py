"""
Test agent executor classes.
"""

import json
import unittest

from elemental_agents.core.agent.generic_agent import GenericAgent
from elemental_agents.core.agent.simple_agent import SimpleAgent
from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.planner_agent import PlannerAgentLogic
from elemental_agents.core.agent_logic.react_agent import ReActAgentLogic
from elemental_agents.core.agent_logic.simple_agent import SimpleAgentLogic
from elemental_agents.core.toolbox.toolbox import ToolBox
from elemental_agents.llm.llm_factory import LLMFactory
from elemental_agents.tools.calculator import Calculator, CalculatorParams
from elemental_agents.tools.list_files import ListFiles, ListFilesParams
from elemental_agents.utils.utils import extract_tag_content


class TestAgentExecution(unittest.TestCase):
    """
    Test execution of simple agent classes.
    """

    def test_simple_agent(self) -> None:
        """
        Test creating and running a SimpleAgent.
        """

        llm_factory = LLMFactory()
        llm = llm_factory.create()

        context = AgentContext(
            agent_name="AssistantAgent",
            agent_persona="Helpful and informative assistant.",
        )

        agent = SimpleAgentLogic(llm, context)

        executor = SimpleAgent(agent, -1)

        instruction1 = "Why is the sky blue?"
        result1 = executor.run(instruction1, input_session="TestSession")
        instruction2 = "Is it the same on Mars?"
        result2 = executor.run(instruction2, input_session="TestSession")

        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        self.assertEqual(len(executor.get_all_messages()), 4)

    def test_generic_executor_with_react_agent(self) -> None:
        """
        Test running a GenericExecutor with a ReActAgent.
        """

        llm_factory = LLMFactory()
        llm = llm_factory.create()

        toolbox = ToolBox()
        toolbox.register_tool("Calculator", Calculator, CalculatorParams)  # type: ignore
        toolbox.register_tool("ListFiles", ListFiles, ListFilesParams)  # type: ignore

        context = AgentContext(
            agent_name="TestAgent",
            agent_persona="Researcher always following scientific method",
        )

        agent = ReActAgentLogic(llm, context, toolbox)

        executor = GenericAgent(agent, -1, toolbox, "<result>")

        task = "Why is the sky blue?"
        result = executor.run(task, input_session="TestSession")

        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(executor.get_all_messages()), 2)

    def test_execution_of_planner(self) -> None:
        """
        Test execution of the PlannerAgent.
        """

        llm_factory = LLMFactory()
        llm = llm_factory.create("openai|gpt-4.1-mini")

        context = AgentContext(
            agent_name="PlannerAgent",
            agent_persona="Helpful planner.",
        )

        agent = PlannerAgentLogic(llm, context)

        executor = SimpleAgent(agent, -1)

        instruction = "What is the difference between BMW X3 and X4?"
        result = executor.run(instruction, input_session="TestSession")

        self.assertIsNotNone(result)

        plan = extract_tag_content(result, "JSON")
        parsed_plan = [json.loads(p) for p in plan]
        self.assertGreaterEqual(len(parsed_plan), 1)


if __name__ == "__main__":
    unittest.main()
