"""
Basic agent executor class for non-interactive agents.
"""

import unittest

from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.simple_agent import SimpleAgentLogic
from elemental_agents.core.memory.short_memory import ShortMemory
from elemental_agents.llm.llm_factory import LLMFactory


class TestSimpleAgent(unittest.TestCase):
    """
    Basic tests for the SimpleAgent class.
    """

    def test_simple_agent(self) -> None:
        """
        Test creating and running a SimpleAgent.
        """

        llm_factory = LLMFactory()
        llm = llm_factory.create("ollama|gemma3n:e4b")

        context = AgentContext(
            agent_name="AssistantAgent",
            agent_persona="Helpful and informative assistant.",
        )

        agent = SimpleAgentLogic(llm, context)

        # Execute the agent
        short_memory = ShortMemory()

        instruction = "Why is the sky blue?"
        result = agent.run(instruction, short_memory)

        self.assertIsNotNone(result)

    def test_simple_agent_mock(self) -> None:
        """
        Test creating and running a SimpleAgent with a mock LLM.
        """

        llm_factory = LLMFactory()
        llm = llm_factory.create("mock")

        context = AgentContext(
            agent_name="AssistantAgent",
            agent_persona="Helpful and informative assistant.",
        )

        agent = SimpleAgentLogic(llm, context)

        # Execute the agent
        short_memory = ShortMemory()

        instruction = "Why is the sky blue?"
        result = agent.run(instruction, short_memory)

        self.assertIsNotNone(result)

    def test_simple_agent_multiple_instructions(self) -> None:
        """
        Test creating and running a SimpleAgent with multiple instructions.
        """

        llm_factory = LLMFactory()
        llm = llm_factory.create("mock")

        context = AgentContext(
            agent_name="AssistantAgent",
            agent_persona="Helpful and informative assistant.",
        )

        agent = SimpleAgentLogic(llm, context)

        # Execute the agent
        short_memory = ShortMemory()

        instruction1 = "Why is the sky blue?"
        result1 = agent.run(instruction1, short_memory)

        # result is returned
        self.assertIsNotNone(result1)

        # memory is not updated by the agent itself
        self.assertEqual(len(short_memory.get_all()), 0)

        instruction2 = "Why is the sky blue?"
        result2 = agent.run(instruction2, short_memory)

        self.assertIsNotNone(result2)
        self.assertEqual(len(short_memory.get_all()), 0)


if __name__ == "__main__":
    unittest.main()
