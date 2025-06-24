"""
Test agent logic factory methods.
"""

import unittest

from elemental_agents.core.agent_logic.agent_logic_factory import AgentLogicFactory
from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.composer_agent import ComposerAgentLogic
from elemental_agents.core.agent_logic.conv_planreact_agent import (
    ConvPlanReActAgentLogic,
)
from elemental_agents.core.agent_logic.plan_verifier_agent import PlanVerifierAgentLogic
from elemental_agents.core.agent_logic.planner_agent import PlannerAgentLogic
from elemental_agents.core.agent_logic.planreact_agent import PlanReActAgentLogic
from elemental_agents.core.agent_logic.react_agent import ReActAgentLogic
from elemental_agents.core.agent_logic.replanner_agent import ReplannerAgentLogic
from elemental_agents.core.agent_logic.simple_agent import SimpleAgentLogic
from elemental_agents.core.agent_logic.verifier_agent import VerifierAgentLogic
from elemental_agents.core.toolbox.toolbox import ToolBox
from elemental_agents.llm.llm_factory import LLMFactory


class TestFactory(unittest.TestCase):
    """
    Test factory way of creating agent logic instances for all agent logic classes.
    """

    def setUp(self) -> None:
        """
        Set up the test environment.
        """
        self.logic_types_supported = {
            "simple": SimpleAgentLogic,
            "planner": PlannerAgentLogic,
            "planverifier": PlanVerifierAgentLogic,
            "replanner": ReplannerAgentLogic,
            "composer": ComposerAgentLogic,
            "verifier": VerifierAgentLogic,
        }

        self.iterative_logic_types_supported = {
            "react": ReActAgentLogic,
            "planreact": PlanReActAgentLogic,
            "convplanreact": ConvPlanReActAgentLogic,
        }

        self.llm_factory = LLMFactory()
        self.llm_openai = self.llm_factory.create("openai|gpt-4.1-mini")
        self.llm_mock = self.llm_factory.create("mock")
        self.context = AgentContext(
            agent_name="TestAgent",
            agent_persona="Test persona for agent logic factory tests.",
        )
        self.box = ToolBox()

    def tearDown(self) -> None:
        """
        Clean up the test environment.
        """
        self.llm_factory = None
        self.llm_openai = None
        self.llm_mock = None
        self.context = None
        self.box = None

    def test_agent_factory_invalid_logic_type(self) -> None:
        """
        Test factory method with invalid logic type.
        """

        logic_type = "invalid_logic_type"
        with self.assertRaises(ValueError) as context:
            AgentLogicFactory.create_agent_logic(
                logic_type=logic_type, llm=self.llm_openai, context=self.context
            )

    def test_agent_factory_all_logic_types(self) -> None:
        """
        Test factory method for all supported logic types.
        """

        for logic_type, expected_class in self.logic_types_supported.items():
            agent_logic = AgentLogicFactory.create_agent_logic(
                logic_type=logic_type, llm=self.llm_openai, context=self.context
            )
            self.assertIsNotNone(agent_logic)
            self.assertIsInstance(agent_logic, expected_class)

    def test_agent_factory_all_logic_types_with_template(self) -> None:
        """
        Test factory method for all supported logic types.
        """

        for logic_type, expected_class in self.logic_types_supported.items():
            agent_logic = AgentLogicFactory.create_agent_logic(
                logic_type=logic_type,
                llm=self.llm_mock,
                template="This is a test template for agent logic.",
                context=self.context,
            )
            self.assertIsNotNone(agent_logic)
            self.assertIsInstance(agent_logic, expected_class)

    def test_agent_factory_all_iterative_types(self) -> None:
        """
        Test factory method for all supported iterative logic types.
        """

        for logic_type, expected_class in self.iterative_logic_types_supported.items():
            agent_logic = AgentLogicFactory.create_agent_logic(
                logic_type=logic_type,
                llm=self.llm_openai,
                context=self.context,
                toolbox=self.box,
            )
            self.assertIsNotNone(agent_logic)
            self.assertIsInstance(agent_logic, expected_class)

    def test_agent_factory_all_iterative_types_with_template(self) -> None:
        """
        Test factory method for all supported iterative logic types with template.
        """

        for logic_type, expected_class in self.iterative_logic_types_supported.items():
            agent_logic = AgentLogicFactory.create_agent_logic(
                logic_type=logic_type,
                llm=self.llm_openai,
                context=self.context,
                toolbox=self.box,
                template="This is a test template for iterative agent logic.",
            )
            self.assertIsNotNone(agent_logic)
            self.assertIsInstance(agent_logic, expected_class)

    def test_agent_properties(self) -> None:
        """
        Test factory method for all supported logic types.
        """

        agent_logic = AgentLogicFactory.create_agent_logic(
            logic_type="simple", llm=self.llm_openai, context=self.context
        )
        self.assertIsNotNone(agent_logic)
        self.assertIsInstance(agent_logic, SimpleAgentLogic)

        persona = agent_logic.get_persona()
        self.assertEqual(persona, "Test persona for agent logic factory tests.")
        name = agent_logic.get_name()
        self.assertEqual(name, "TestAgent")


if __name__ == "__main__":
    unittest.main()
