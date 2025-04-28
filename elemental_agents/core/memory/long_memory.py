"""
Long Memory module that represents the storage of an agent going beyond the
short-term (single task) memory.
"""

import os
from typing import List

from loguru import logger

from elemental_agents.core.memory.memory import Memory
from elemental_agents.db.db_factory import DBFactory
from elemental_agents.embeddings.embeddings_factory import EmbeddingsFactory
from elemental_agents.llm.data_model import Message
from elemental_agents.tools.semantic_search import (
    SemanticSearch,
    SemanticSearchInitParams,
    SemanticSearchParams,
)


class LongMemory(Memory[str]):
    """
    LongMemory class that represents the storage of an agent going beyond the
    short-term (single task) memory.
    """

    def __init__(
        self, embeddings_engine: str, max_results: int, threshold: float = 0.5
    ) -> None:
        """
        Initialize the long-term memory.

        :param embeddings_engine: The embeddings engine to use for the memory.
        :param max_results: The maximum number of search results to return.
        """
        super().__init__()

        long_memory_file = os.getenv("LONG_MEMORY_FILE")
        data_base = f"chromadb|{long_memory_file}"

        self._container = DBFactory().create(data_base)
        self._max_results = max_results
        self._threshold = threshold

        # Initialize the embeddings model
        embeddings_model_factory = EmbeddingsFactory()
        self._embeddings_model = embeddings_model_factory.create(
            engine_name=embeddings_engine
        )

        # Initialize the Semantic Search tool
        init_parameters = SemanticSearchInitParams(
            database=self._container,
            embeddings=self._embeddings_model,
        )

        # Set the semantic search tool
        self._semantic_search_tool = SemanticSearch(init_params=init_parameters)

    def add(self, item: Message | str) -> None:
        """
        Add a message to the long-term memory.

        :param item: The message to add.
        """

        if isinstance(item, Message):
            text = item.content
        else:
            text = item

        embedding = self._embeddings_model.run(text)
        data = ([embedding], None)
        self._container.add(data)

    def get(self, number: int, query: str = None) -> List[str]:
        """
        Search the long-term memory for the query phrase.

        :param number: The number of results to return.
        :param query: The query phrase.
        :return: The search results.
        """

        if query is None:
            raise ValueError("Query cannot be None")

        result: List[str] = []

        semantic_search_params = SemanticSearchParams(
            query=query, top_n=number, threshold=self._threshold
        )
        search_result = self._semantic_search_tool.run(semantic_search_params)

        logger.debug(f"Long memory :: Search results: {search_result.results}")

        result = search_result.results

        return result
