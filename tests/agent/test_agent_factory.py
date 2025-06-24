"""
Test agent factory methods.
"""

import unittest

from elemental_agents.core.agent.agent_factory import AgentFactory
from elemental_agents.core.agent.generic_agent import GenericAgent
from elemental_agents.core.agent.simple_agent import SimpleAgent
from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm_factory import LLMFactory


class TestFactory(unittest.TestCase):
    """
    Test factory way of creating agent instances for all supported agent types.
    """

    def setUp(self) -> None:
        """
        Set up the test environment.
        """
        self.context = AgentContext(
            agent_name="TestAgent",
            agent_persona="Test persona for agent factory tests.",
        )

        self.llm_factory = LLMFactory()
        self.llm_openai = self.llm_factory.create("openai|gpt-4.1-mini")
        self.llm_mock = self.llm_factory.create("mock")

        self.llm_parameters = ModelParameters()

        self.non_iterative_agent_types_supported = {
            "simple": SimpleAgent,
            "planner": SimpleAgent,
            "planverifier": SimpleAgent,
            "replanner": SimpleAgent,
            "composer": SimpleAgent,
            "verifier": SimpleAgent,
        }

        self.iterative_agent_types_supported = {
            "react": GenericAgent,
            "planreact": GenericAgent,
            "convplanreact": GenericAgent,
        }

    def tearDown(self) -> None:
        """
        Clean up the test environment.
        """
        self.context = None
        self.llm_openai = None
        self.llm_mock = None

    def test_create_simple_agents(self) -> None:
        """
        Test the factory creates simple agents.
        """
        for agent_type, agent_class in self.non_iterative_agent_types_supported.items():
            with self.subTest(agent_type=agent_type):
                agent = AgentFactory.create(
                    agent_name=self.context.agent_name,
                    agent_type=agent_type,
                    llm_model="openai|gpt-4.1-mini",
                    model_parameters=self.llm_parameters,
                    memory_capacity=10,
                    agent_persona=self.context.agent_persona,
                    tools=[],
                    template=None,
                )
                self.assertIsInstance(agent, agent_class)

    def test_create_iterative_agents(self) -> None:
        """
        Test the factory creates simple agents.
        """
        for agent_type, agent_class in self.iterative_agent_types_supported.items():

            with self.subTest(agent_type=agent_type):
                agent = AgentFactory.create(
                    agent_name=self.context.agent_name,
                    agent_type=agent_type,
                    llm_model="openai|gpt-4.1-mini",
                    model_parameters=self.llm_parameters,
                    memory_capacity=10,
                    agent_persona=self.context.agent_persona,
                    tools=["Calculator"],
                    template=None,
                )
                self.assertIsInstance(agent, agent_class)

    def test_create_agent_with_invalid_type(self) -> None:
        """
        Test the factory raises an exception for invalid agent type.
        """
        with self.assertRaises(ValueError) as context:
            AgentFactory.create(
                agent_name=self.context.agent_name,
                agent_type="invalid_agent_type",
                llm_model="openai|gpt-4.1-mini",
                model_parameters=self.llm_parameters,
                memory_capacity=10,
                agent_persona=self.context.agent_persona,
                tools=[],
                template=None,
            )


if __name__ == "__main__":
    unittest.main()
