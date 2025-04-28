"""
This module provides a class for searching Google.
"""

from typing import Any, Dict, List, Optional

import requests
from loguru import logger
from pydantic import Field
from rich.console import Console

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult
from elemental_agents.utils.config import ConfigModel
from elemental_agents.utils.exceptions import ToolException


class GoogleSearchParams(ToolParameters):
    """
    Parameters for the Google Search tool

    :param query: The search query to perform
    """

    query: str = Field(..., description="The search query to perform")
    num_results: int = Field(10, description="Number of search results to return")


class GoogleSearchResult(ToolResult):
    """
    Result for the Google Search tool

    :param urls: List of dicts representing the search results
    """

    data: List[Dict[str, str]] = Field(
        ..., description="List of dictionaries of URLs representing the search results"
    )

    def __str__(self) -> str:
        return f"{self.data}"


class GoogleSearch(Tool):
    """
    Tool for searching Google. This tool takes a search query and returns the
    search results as a list of URLs.
    """

    name = "GoogleSearch"
    description = "Search Google and return results with titles, snippets and URLs."

    def run(self, parameters: GoogleSearchParams) -> GoogleSearchResult:  # type: ignore
        """
        Run the Google search tool with the given parameters.

        :param parameters: The parameters for the Google search tool.
        :return: A list of URLs representing the search results.
        """

        config = ConfigModel()

        # Base URL for Google Custom Search API
        url = "https://www.googleapis.com/customsearch/v1"

        query = parameters.query
        api_key = config.google_search_api_key
        cse_id = config.google_cse_id
        num_results = parameters.num_results
        timeout_duration = config.google_search_timeout

        # Parameters to send in the GET request
        params: Optional[Dict[str, Any]] = {
            "q": query,  # Search query
            "key": api_key,  # Your API key
            "cx": cse_id,  # Custom Search Engine ID
            "num": num_results,  # Number of results to return
        }

        try:
            # Make the request and get the response
            response = requests.get(url=url, params=params, timeout=timeout_duration)

            # If the request is successful (status code 200)
            if response.status_code == 200:
                search_result = response.json()  # Parse and return the JSON response

                final_result = []
                for item in search_result.get("items", []):
                    final_result.append(
                        {
                            "title": item["title"],
                            "url": item["link"],
                            "snippet": item["snippet"],
                        }
                    )

                logger.debug(f"Google search result: {final_result}")

                return GoogleSearchResult(data=final_result)

        except requests.exceptions.Timeout as e:
            logger.error(f"Request timed out after {timeout_duration} seconds.")
            raise ToolException(
                f"Request timed out after {timeout_duration} seconds)"
            ) from e
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred with request call in Google Search: {e}")
            raise ToolException(
                f"An error occurred with request call in Google Search: {e}"
            ) from e


if __name__ == "__main__":

    console = Console()

    # Example usage of the GoogleSearch tool
    SEARCH_QUERY = "Python programming"
    NUM_RESULTS = 10

    google_search_tool = GoogleSearch()
    search_params = GoogleSearchParams(query=SEARCH_QUERY, num_results=NUM_RESULTS)

    result = google_search_tool.run(search_params)

    if result:
        logger.info(f"Search results for '{SEARCH_QUERY}':")

        console.print(result.data)

    else:
        logger.error("An error occurred during the search.")
