"""
Interface for vector databases based on DB module.
"""

from abc import abstractmethod
from typing import Any, List, Optional

from elemental_agents.db.db import DB


class VectorDB(DB):
    """
    Vector database interface.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def add(self, data: Any) -> None:
        """
        Add data to the Vector DB.

        :param data: The data to add.
        """

    @abstractmethod
    def query(
        self, search_item: Any, top_n: int, threshold: Optional[float]
    ) -> List[Any] | None:
        """
        Query the Vector DB for similar embeddings.

        :param search_item: The search item.
        :param top_n: The number of search results to return.
        :param threshold: The similarity threshold for the search results.
        :return: The search results.
        """
