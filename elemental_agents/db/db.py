"""
Database interface.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class DB(ABC):
    """
    Standard interface for database clients.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def add(self, data: Any) -> None:
        """
        Add data to the database.

        :param data: The data to add.
        """

    @abstractmethod
    def query(
        self, search_item: Any, top_n: int, threshold: Optional[float]
    ) -> List[Any] | None:
        """
        Query data from the database.

        :param search_item: The search item.
        :param top_n: The number of search results to return.
        :param threshold: The similarity threshold for the search results.
        :return: The search results.
        """
