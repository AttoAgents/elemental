"""
Tool to interact with the Elemental Knowledge Base API.
"""

import os
from typing import List

import requests
from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult


class KnowledgeBaseSearchParams(ToolParameters):
    """
    Defines the input parameters for the KnowledgeBaseSearch tool.

    :param query: Query to search in the Knowledge Base
    """

    query: str = Field(
        description="Query to search in the application internal Knowledge Base."
    )
    top_n: int = Field(default=5, description="Number of top results to return.")
    collection: str = Field(description="Name of the collection to search in.")


class KnowledgeBaseSearchResult(ToolResult):
    """
    Defines the output result for the KnowledgeBaseSearch tool.
    """

    results: List[str] = Field(
        default=[], description="List of top results from the Knowledge Base"
    )
    status: str = Field(
        default="OK", description="Status of the Knowledge Base API call"
    )

    def __str__(self) -> str:
        return f"{self.results, self.status}"


class KnowledgeBaseSearch(Tool):
    """
    Tool to interact with the Elemental Knowledge Base API.
    """

    name = "KnowledgeBaseSearch"
    description = (
        "Search internal Knowledge Base for the specific topic or query. "
        "You have access to specific collections that include data on selected topic."
    )

    def run(
        self, parameters: KnowledgeBaseSearchParams  # type: ignore
    ) -> KnowledgeBaseSearchResult:
        """
        Search the Knowledge Base for the specified query.

        :param parameters: KnowledgeBaseSearchParams object with the query
        :return: List of top results from the KnowledgeBaseSearch
        """

        url = os.getenv("ELEMENTAL_KB_API_URL")
        if not url:
            message = "Knowledge Base API URL not found"
            logger.error(message)
            return KnowledgeBaseSearchResult(status=f"ERROR: {message}")

        payload = {
            "query": parameters.query,
            "collection_name": parameters.collection,
            "top_n": parameters.top_n,
        }
        response = requests.post(
            url, json=payload, headers={"Content-Type": "application/json"}, timeout=10
        )

        if response.status_code != 200:
            message = f"Failed to get response from Knowledge Base API: {response.text}"
            logger.error(message)
            return KnowledgeBaseSearchResult(status=f"ERROR: {message}")

        data = response.json()
        results = [item["text"] for item in data["results"]]

        return KnowledgeBaseSearchResult(results=results, status="OK")
