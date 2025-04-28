"""
Implementation of the ShortMemory class.
"""

from typing import List

from elemental_agents.core.memory.memory import Memory
from elemental_agents.llm.data_model import Message


class ShortMemory(Memory[Message]):
    """
    ShortMemory class that represents the short-term memory of an agent.
    """

    def __init__(self, capacity: int = -1) -> None:
        """
        Initialize the short-term memory.

        :param capacity: The capacity of the short-term memory. If the capacity
            is set, it will only store the last n messages. If the capacity is
            not set, it will store all messages. Default is -1 which is no
            capacity.
        """
        super().__init__()

        self._container: List[Message] = []
        self._capacity: int = capacity

    def reset(self) -> None:
        """
        Reset the short-term memory.
        """

        self._container = []

    def add(self, item: Message) -> None:
        """
        Add a message to the short-term memory.

        :param message: The message to add to the short-term memory.
        """

        if isinstance(item, str):
            item = Message(role="user", content=item)

        self._container.append(item)

    def get(self, number: int) -> List[Message]:
        """
        Get the last number of items from the short-term memory.

        :param number: The number of items to get from the short-term memory.
        :return: The last number of items from the short-term memory.
        """
        if number > len(self._container):
            number = len(self._container)

        return self._container[-number:]

    def get_all(self) -> List[Message]:
        """
        Return all messages in the short-term memory. If the capacity is set, it
        will return the last n messages. If the capacity is not set, it will
        return all messages.

        :return: List of all messages in the short-term memory.
        """
        if self._capacity > 0:
            return self._container[-min(self._capacity, len(self._container)) :]
        return self._container

    def get_last(self) -> Message:
        """
        Return the last message in the short-term memory.

        :return: The last message in the short-term memory.
        """

        return self._container[-1]
