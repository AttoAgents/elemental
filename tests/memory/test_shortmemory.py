"""
Test cases for ShortMemory class.
"""

import unittest

from elemental_agents.core.memory.short_memory import ShortMemory
from elemental_agents.llm.data_model import Message


class TestShortMemory(unittest.TestCase):
    """
    Test short memory class.
    """

    def test_add(self) -> None:
        """
        Test adding a message to short memory.
        """

        short_memory = ShortMemory(capacity=-1)
        message = Message(role="user", content="Hello")
        short_memory.add(message)

        self.assertEqual(len(short_memory.get_all()), 1)

    def test_get_all(self) -> None:
        """
        Test get all messages from short memory.
        """

        short_memory = ShortMemory()
        messages = [Message(role="user", content="Hello")]
        short_memory.add(messages[0])
        all_messages = short_memory.get_all()

        self.assertEqual(all_messages, messages)

    def test_get_last(self) -> None:
        """
        Get the last message from short memory.
        """

        short_memory = ShortMemory(capacity=-1)
        message = Message(role="user", content="Hello")
        short_memory.add(message)

        self.assertEqual(short_memory.get_last().content, "Hello")

    def test_get_all_with_capacity(self) -> None:
        """
        Test get all messages from short memory with capacity limitted to certain value.
        """

        capacity = 2
        short_memory = ShortMemory(capacity=capacity)
        messages = [
            Message(role="user", content="Hello"),
            Message(role="system", content="How can I help you?"),
            Message(role="system", content="What day is today?"),
        ]
        short_memory.add(messages[0])
        short_memory.add(messages[1])
        short_memory.add(messages[2])
        all_messages = short_memory.get_all()

        self.assertEqual(len(all_messages), capacity)
        self.assertEqual(all_messages, messages[1:])


if __name__ == "__main__":
    unittest.main()
