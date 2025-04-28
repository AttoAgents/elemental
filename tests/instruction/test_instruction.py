"""
Test the Instruction class.
"""

import unittest
from typing import List

from elemental_agents.core.instruction.instruction import Instruction
from elemental_agents.core.taskqueue.task import Task


class TestInstruction(unittest.TestCase):
    """
    Test formatting of instructions with additional context.
    """

    def test_instruction(self) -> None:
        """
        Test simple instruction formatting.
        """

        t = Task(
            description="Why is the sky blue?",
            id="1",
            dependencies=[],
            context={"Task": "Previously done task."},
        )

        memories: List[str] = []

        a = Instruction(task=t, memories=memories)
        self.assertEqual(
            str(a.render()),
            "<task>Why is the sky blue?</task><context>Task - Previously done task.</context>",
        )

    def test_simple_description(self) -> None:
        """
        Test description formatting for a task.
        """

        t = Task(
            description="Why is the sky blue?", id="1", dependencies=[], context={}
        )

        a = Instruction(task=t)

        self.assertEqual(str(a.render()), "<task>Why is the sky blue?</task>")

    def test_all_fields(self) -> None:
        """
        Test all fields of the instruction.
        """

        memories = [
            "Solution from similar task.",
            "Another example that is relevant.",
        ]

        t = Task(
            description="Why is the sky blue?",
            id="1",
            dependencies=[],
            context={"Task": "Previously done task."},
        )

        a = Instruction(task=t, memories=memories)

        self.assertEqual(
            str(a.render()),
            "<task>Why is the sky blue?</task><context>Task - Previously done task.</context>\n<memory> Solution from similar task. </memory>\n<memory> Another example that is relevant. </memory>",
        )


if __name__ == "__main__":
    unittest.main()
