"""
Wikipedia search tool. Search for a term on Wikipedia and return the whole text content.
"""

import re

import wikipediaapi
from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult
from elemental_agents.utils.config import ConfigModel
from elemental_agents.utils.exceptions import ToolException


class WikipediaException(ToolException):
    """
    Exception for Wikipedia search tool.
    """


class WikipediaSearchParams(ToolParameters):
    """
    Parameters for the Wikipedia search tool.

    :param phrase: The search phrase to look up on Wikipedia
    """

    phrase: str = Field(..., description="The phrase to search on Wikipedia")


class WikipediaSearchResult(ToolResult):
    """
    Result for the Wikipedia search tool.

    :param content: The text content of the Wikipedia page
    """

    content: str = Field(..., description="The text content of the Wikipedia page")

    def __str__(self) -> str:
        return f"{self.content}"


class WikipediaSearch(Tool):
    """
    Tool that searches for a term on Wikipedia and returns the whole text content.
    """

    name = "WikipediaSearch"
    description = "Search for a term on Wikipedia and return the whole text content."

    def run(self, parameters: WikipediaSearchParams) -> WikipediaSearchResult:  # type: ignore
        """
        Search for the given term on Wikipedia and return the whole text content.

        :param parameters: The search parameters.
        :return: The text content of the Wikipedia page.
        """

        search_phrase = parameters.phrase
        logger.info(f"Searching Wikipedia for: {search_phrase}")

        wikipedia_result = self.search_wikipedia_for_phrase(search_phrase)

        result = WikipediaSearchResult(content=wikipedia_result)
        return result

    def search_wikipedia_for_phrase(self, phrase: str) -> str:
        """
        Search for the given phrase on Wikipedia and return the full text content.

        :param phrase: The phrase to search on Wikipedia.
        :return: The full text content of the Wikipedia page.
        """

        config = ConfigModel()

        user_agent = config.wikipedia_user_agent
        try:
            # Initialize Wikipedia API for English
            wiki_wiki = wikipediaapi.Wikipedia(user_agent, "en")

            # Get the Wikipedia page for the given phrase
            page = wiki_wiki.page(phrase)

            # Check if the page exists
            if page.exists():
                # Get the full text content of the page
                page_content = page.text

                # Remove references in the format [1], [2], etc.
                page_content = re.sub(r"\[\d+\]", "", page_content)

                # Remove any content in parentheses (e.g., disambiguation content)
                page_content = re.sub(r"\([^)]*\)", "", page_content)

                # Return the cleaned content
                return page_content.strip()

            return "No results found for the given phrase."

        except Exception as e:
            logger.error(f"Error searching Wikipedia: {e}")
            raise WikipediaException(f"API error occurred: {str(e)}") from e
