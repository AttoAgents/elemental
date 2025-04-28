"""
Simple RAG method for in memory refinement of externally processed data or other
temporary use.
"""

from typing import List

from loguru import logger
from pydantic import BaseModel

from elemental_agents.db.db_chromadb import DBChromaDB
from elemental_agents.embeddings.embeddings_factory import EmbeddingsFactory
from elemental_agents.utils.utils import split_text_into_chunks


class RAGParameters(BaseModel):
    """
    Parameters for the RAG.

    :param top_n: The number of top results to return.
    :param chunk_length: The length of each chunk.
    :param chunk_overlap: The number of words to overlap between chunks.
    """

    top_n: int = 5
    chunk_length: int = 100
    chunk_overlap: int = 20


class RAG:
    """
    Simple RAG with in memory vector storage.
    """

    def __init__(self, parameters: RAGParameters = RAGParameters()) -> None:
        """
        Initialize the RAG.
        """

        self._db = DBChromaDB(scope="in_memory", file_name=None)

        factory = EmbeddingsFactory()
        self._embeddings = factory.create()

        self._parameters = parameters

        logger.info("Temporary RAG initialized.")

    def add_data(self, data: str) -> None:
        """
        Add data to the RAG.

        :param data: The data be vectorized and added to the data base.
        """

        # Split data into smaller chunks
        chunks = split_text_into_chunks(
            text=data,
            chunk_length=self._parameters.chunk_length,
            chunk_overlap=self._parameters.chunk_overlap,
        )

        # Vectorize each chunk separately
        vectors = []
        for chunk in chunks:
            vector = self._embeddings.run(chunk)
            vectors.append(vector)

        # Add vectorized data to the data base
        self._db.add((vectors, chunks))

        logger.debug("Data added to the temporary RAG database.")

    def search(self, query: str) -> List[str]:
        """
        Search the RAG for the most similar data to the query.

        :param query: The query to search for.
        :return: The most similar data to the query.
        """

        # Vectorize the query
        query_vector = self._embeddings.run(query)

        # Search vector database for the most similar data
        data = self._db.query(
            search_item=query_vector.embedding, top_n=self._parameters.top_n
        )

        text_data = []
        for item in data:
            text_data.append(item.text)

        return text_data

    def run(self, query_text: str, data: str) -> List[str]:
        """
        Run the in memory RAG by vectorizing the data and searching for the query text.

        :param query_text: The query text to search for.
        :param data: The data to add to the RAG.
        :return: The most similar data to the query.
        """

        self.add_data(data)
        result = self.search(query_text)
        return result
