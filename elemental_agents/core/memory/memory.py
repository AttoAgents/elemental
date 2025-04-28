"""
Interface for the memory objects.
"""

from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

# from elemental.llm.data_model import Message

T = TypeVar("T")


class Memory(ABC, Generic[T]):
    """
    Interface for the memory object.
    """

    def __init__(self) -> None:
        """
        Initialize the memory object.
        """

    @abstractmethod
    def add(self, item: T) -> None:
        """
        Add item to memory.
        """

    @abstractmethod
    def get(self, number: int) -> List[T]:
        """
        Get the last number of items from memory.
        """
