"""
Semantic Search tool that performs semantic search on a database of embeddings.
"""

from typing import List, Optional

from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import (
    Tool,
    ToolInitParameters,
    ToolParameters,
    ToolResult,
)
from elemental_agents.db.db import DB
from elemental_agents.embeddings.embeddings import Embeddings
from elemental_agents.utils.exceptions import EmbeddingError, SearchError


# Define specific Pydantic model for Semantic Search input
class SemanticSearchParams(ToolParameters):
    """
    Parameters for the Semantic Search tool to perform semantic search on a database of embeddings.

    :param query: Query string for semantic search
    :param top_n: Number of results to return
    """

    query: str = Field(..., description="Query string for semantic search")
    top_n: int = Field(default=10, description="Number of results to return")
    threshold: float = Field(
        default=0.5,
        description=(
            "Threshold for the similarity distance between query "
            "item and DB entries for the search results"
        ),
    )


class SemanticSearchInitParams(ToolInitParameters):
    """
    Initialization parameters for the Semantic Search tool.

    :param database: Database
    :param embeddings: Embeddings model
    """

    database: DB = Field(..., description="Database")
    embeddings: Embeddings = Field(..., description="Embeddings model")

    model_config = {"arbitrary_types_allowed": True}


class SemanticSearchResult(ToolResult):
    """
    Result for the Semantic Search tool.

    :param results: List of search results
    """

    results: List[str] = Field(default=[], description="List of search results")

    def __str__(self) -> str:
        return f"{self.results}"


# Semantic Search tool with class-level name and description
class SemanticSearch(Tool):
    """
    Semantic Search tool that performs semantic search on a database of embeddings.
    """

    name = "SemanticSearch"
    description = "Performs semantic search on a database of embeddings"

    def __init__(self, init_params: Optional[SemanticSearchInitParams] = None) -> None:
        """
        Initialize the Semantic Search tool with database and embeddings model
        to allow standalone execution.

        :param init_params: SemanticSearchInitParams object containing the
            initialization parameters
        """
        if init_params is not None:
            self._database = init_params.database
            self._embeddings = init_params.embeddings

    def run(self, parameters: SemanticSearchParams) -> SemanticSearchResult:  # type: ignore
        """
        Run the Semantic Search tool with the given parameters.

        :param parameters: The parameters for the Semantic Search tool.
        :return: The results of the semantic search.
        """
        try:
            # Get the embeddings for the query string
            query_embedding = self._embeddings.run(parameters.query)
            vector = query_embedding.embedding
        except EmbeddingError as e:
            logger.error(f"Embedding generation failed: {e}")

        try:
            # Search vectors in the database that are closest to the query vector
            text_only_result = []
            results = self._database.query(
                vector, parameters.top_n, parameters.threshold
            )

            if results is not None:
                for item in results:
                    text_only_result.append(item.text)

        except SearchError as e:
            logger.error(f"Database search failed: {e}")

        final_result = SemanticSearchResult(results=text_only_result)
        return final_result
