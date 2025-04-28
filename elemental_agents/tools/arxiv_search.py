"""
ArXiv search tools.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List

import requests
from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult
from elemental_agents.utils.config import ConfigModel
from elemental_agents.utils.exceptions import ToolException


class ArXivSearchParams(ToolParameters):
    """
    Parameters for the ArXiv search tool.

    :param phrase: The search phrase to look up on arXiv
    :param max_results: The maximum number of results to return
    """

    phrase: str = Field(..., description="The phrase to search on arXiv")
    max_results: int = Field(5, description="The maximum number of results to return")


class ArXivSearchResult(ToolResult):
    """
    Result for the ArXiv search tool.

    :param papers: List of papers found in the search
    """

    papers: List[Dict[str, str]] = Field(
        ..., description="List of papers found in the search"
    )

    def __str__(self) -> str:
        return f"{self.papers}"


class ArXivSearch(Tool):
    """
    Tool that searches for a term on arXiv and returns a list of papers.
    """

    name = "ArXivSearch"
    description = """Search for a term on arXiv and return a list of
    papers with titles, summaries, publication dates and PDF URLs."""

    def run(self, parameters: ArXivSearchParams) -> ArXivSearchResult:  # type: ignore
        """
        Search for the given term on arXiv and return a list of papers.

        :param parameters: The search parameters.
        :return: List of papers found in the search.
        """

        config = ConfigModel()

        search_phrase = parameters.phrase
        max_results = parameters.max_results
        timeout = config.arxiv_search_timeout

        # Search the arXiv API
        papers = self.search_arxiv(search_phrase, max_results, timeout)
        final_result = ArXivSearchResult(papers=papers)
        return final_result

    def search_arxiv(
        self, search_phrase: str, max_results: int = 5, timeout: int = 10
    ) -> List[Dict[str, str]]:
        """
        Search the arXiv API for papers related to a given phrase.

        :param phrase: The search phrase to look up on arXiv
        :param max_results: The maximum number of results to return
        :param timeout: The timeout for the request
        :return: List of papers found in the search
        """

        try:
            # Construct the arXiv API URL
            url = (
                f"http://export.arxiv.org/api/query?search_query=all:{search_phrase}"
                f"&start=0&max_results={max_results}"
            )

            # Send the request to the arXiv API
            response = requests.get(url, timeout=timeout)

            # Check if the request was successful
            if response.status_code != 200:
                logger.error(f"Error fetching data: {response.status_code}")
                raise ToolException(f"Error fetching data: {response.status_code}")

            # Parse the XML response
            root = ET.fromstring(response.content)

            # Extract relevant data from the XML
            results = []
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                title = entry.find("{http://www.w3.org/2005/Atom}title").text
                summary = entry.find("{http://www.w3.org/2005/Atom}summary").text
                published = entry.find("{http://www.w3.org/2005/Atom}published").text
                pdf_url = entry.find("{http://www.w3.org/2005/Atom}id").text

                # Collect the result
                results.append(
                    {
                        "title": title.strip(),
                        "summary": summary.strip(),
                        "published": published,
                        "pdf_url": pdf_url,
                    }
                )
            if len(results) == 0:
                logger.error("No results found for the given phrase.")
                raise ToolException("No results found for the given phrase.")
            return results

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            raise ToolException(f"An error occurred: {str(e)}") from e
