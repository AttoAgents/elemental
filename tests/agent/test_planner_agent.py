"""
Test agent with PlanReAct strategy.
"""

import json
import unittest

from loguru import logger
from rich.console import Console

from elemental_agents.core.agent_logic.agent_model import AgentContext
from elemental_agents.core.agent_logic.plan_verifier_agent import PlanVerifierAgentLogic
from elemental_agents.core.agent_logic.planner_agent import PlannerAgentLogic
from elemental_agents.core.memory.short_memory import ShortMemory
from elemental_agents.llm.llm_factory import LLMFactory
from elemental_agents.utils.utils import extract_tag_content


class TestPlannerAgent(unittest.TestCase):
    """
    Test planner agent.
    """

    def test_planner_agent(self) -> None:
        """
        Test the planner agent on simple instruction.
        """

        console = Console()

        llm_factory = LLMFactory()
        llm = llm_factory.create("openai|gpt-4.1-mini")

        agent_context = AgentContext(
            agent_name="PlannerAgent",
            agent_persona="Helpful planner.",
        )

        # Setup the Simple Agent
        agent = PlannerAgentLogic(model=llm, context=agent_context)

        # Execute the agent
        short_memory = ShortMemory()

        instruction = "What is the difference between BMW X3 and X4?"
        result = agent.run(instruction, short_memory)

        logger.debug("Agent's raw response:")
        console.print(result)

        # Parse the plan from the agent's response
        plan = extract_tag_content(result, "JSON")
        parsed_plan = [json.loads(p) for p in plan]
        console.print(parsed_plan)

        self.assertGreater(len(parsed_plan), 0)

    def test_planner_cases(self) -> None:
        """
        Test the planner agent on simple instruction.
        """

        console = Console()

        llm_factory = LLMFactory()
        llm = llm_factory.create("openai|gpt-4.1-mini")

        agent_context = AgentContext(
            agent_name="PlannerAgent",
            agent_persona="Helpful planner.",
        )

        # Setup the Simple Agent
        agent = PlannerAgentLogic(model=llm, context=agent_context)

        # Execute the agent
        short_memory = ShortMemory()

        instruction_set = [
            "What is the difference between BMW X3 and X4?",
            "Calculate the resistance of copper wire having a length of 1 km and diameter 0.5 mm. (Resistivity of copper 1.7e-8Ωm)",
            "Calculate the volume of a sphere with radius 5 cm.",
            "Does electric range of Tesla Model 3 allow for weekly commute between San Francisco and Los Angeles?",
        ]

        for instruction in instruction_set:
            result = agent.run(instruction, short_memory)

            logger.debug(f"Plan for instruction: {instruction}")
            logger.debug("Agent's raw response:")
            console.print(result)

            # Parse the plan from the agent's response
            plan = extract_tag_content(result, "JSON")
            parsed_plan = [json.loads(p) for p in plan]
            console.print(parsed_plan)

            self.assertGreater(len(parsed_plan), 0)

    def test_planner_cases_explicit_model(self) -> None:
        """
        Test the planner agent on simple instruction.
        """

        console = Console()

        llm_factory = LLMFactory()
        llm = llm_factory.create("openai|gpt-4o-mini")

        agent_context = AgentContext(
            agent_name="PlannerAgent",
            agent_persona="Helpful planner.",
        )

        # Setup the Simple Agent
        agent = PlannerAgentLogic(model=llm, context=agent_context)

        # Execute the agent
        short_memory = ShortMemory()

        instruction_set = [
            "What is the difference between BMW X3 and X4?",
            "Calculate the resistance of copper wire having a length of 1 km and diameter 0.5 mm. (Resistivity of copper 1.7e-8Ωm)",
            "Calculate the volume of a sphere with radius 5 cm.",
            "Does electric range of Tesla Model 3 allow for weekly commute between San Francisco and Los Angeles?",
        ]

        for instruction in instruction_set:
            result = agent.run(instruction, short_memory)

            logger.debug(f"Plan for instruction: {instruction}")
            logger.debug("Agent's raw response:")
            console.print(result)

            # Parse the plan from the agent's response
            plan = extract_tag_content(result, "JSON")
            parsed_plan = [json.loads(p) for p in plan]
            console.print(parsed_plan)

            self.assertGreater(len(parsed_plan), 0)

    def test_planner_and_verifier_cases_default_model(self) -> None:
        """
        Test the planner agent on simple instruction.
        """

        console = Console()

        llm_factory = LLMFactory()
        llm = llm_factory.create("openai|gpt-4.1-mini")

        agent_context = AgentContext(
            agent_name="PlannerAgent",
            agent_persona="Helpful planner.",
        )

        # Setup the Simple Agent
        agent = PlannerAgentLogic(model=llm, context=agent_context)
        verifier = PlanVerifierAgentLogic(model=llm, context=agent_context)

        # Execute the agent
        short_memory = ShortMemory()

        instruction_set = [
            "What is the difference between BMW X3 and X4?",
            "Calculate the resistance of copper wire having a length of 1 km and diameter 0.5 mm. (Resistivity of copper 1.7e-8Ωm)",
            "Calculate the volume of a sphere with radius 5 cm.",
            "Does electric range of Tesla Model 3 allow for weekly commute between San Francisco and Los Angeles?",
        ]

        for instruction in instruction_set:
            result = agent.run(instruction, short_memory)

            logger.debug(f"Plan for instruction: {instruction}")
            logger.debug("Agent's raw response:")
            console.print(result)

            # Parse the plan from the agent's response
            plan = extract_tag_content(result, "JSON")
            parsed_plan = [json.loads(p) for p in plan]
            console.print(parsed_plan)

            self.assertGreater(len(parsed_plan), 0)

            # Verify the plan
            verifier_instruction = (
                f"<instruction>{instruction}</instruction>\n<plan>{result}</plan>"
            )
            verifier_result = verifier.run(verifier_instruction, short_memory)
            console.print(verifier_result)
            verifier_plan = extract_tag_content(verifier_result, "JSON")
            verifier_parsed_plan = [json.loads(p) for p in verifier_plan]
            console.print(verifier_parsed_plan)

            self.assertGreater(len(verifier_parsed_plan), 0)

    def test_planner_and_verifier_cases_explicit_model(self) -> None:
        """
        Test the planner agent on simple instruction.
        """

        console = Console()

        llm_factory = LLMFactory()
        llm = llm_factory.create("openai|gpt-4.1-mini")

        agent_context = AgentContext(
            agent_name="PlannerAgent",
            agent_persona="Helpful planner.",
        )

        # Setup the Simple Agent
        agent = PlannerAgentLogic(model=llm, context=agent_context)
        verifier = PlanVerifierAgentLogic(model=llm, context=agent_context)

        # Execute the agent
        short_memory = ShortMemory()

        instruction_set = [
            "What is the difference between BMW X3 and X4?",
            "Calculate the resistance of copper wire having a length of 1 km and diameter 0.5 mm. (Resistivity of copper 1.7e-8Ωm)",
            "Calculate the volume of a sphere with radius 5 cm.",
            "Does electric range of Tesla Model 3 allow for weekly commute between San Francisco and Los Angeles?",
        ]

        for instruction in instruction_set:
            result = agent.run(instruction, short_memory)

            logger.debug(f"Plan for instruction: {instruction}")
            logger.debug("Agent's raw response:")
            console.print(result)

            # Parse the plan from the agent's response
            plan = extract_tag_content(result, "JSON")
            parsed_plan = [json.loads(p) for p in plan]
            console.print(parsed_plan)

            self.assertGreater(len(parsed_plan), 0)

            # Verify the plan
            verifier_instruction = (
                f"<instruction>{instruction}</instruction>\n<plan>{result}</plan>"
            )
            verifier_result = verifier.run(verifier_instruction, short_memory)
            console.print(verifier_result)
            verifier_plan = extract_tag_content(verifier_result, "JSON")
            verifier_parsed_plan = [json.loads(p) for p in verifier_plan]
            console.print(verifier_parsed_plan)

            self.assertGreater(len(verifier_parsed_plan), 0)


if __name__ == "__main__":
    unittest.main()
