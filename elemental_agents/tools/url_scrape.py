"""
Web scraping tool that extracts text content from a webpage and returns it in
Markdown format.
"""

from typing import Dict

import requests
from bs4 import BeautifulSoup, Comment
from loguru import logger
from markitdown import MarkItDown
from pydantic import Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult
from elemental_agents.utils.exceptions import ToolException


class ScrapeURLParams(ToolParameters):
    """
    Parameters for the ScrapeURL tool

    :param url: The URL of the webpage to scrape
    """

    url: str = Field(..., description="The URL of the webpage to scrape")


class ScrapeURLResult(ToolResult):
    """
    Result for the ScrapeURL tool

    :param text: The scraped text in Markdown format
    """

    text: str = Field(..., description="The scraped text in Markdown format")

    def __str__(self) -> str:
        return f"{self.text}"


class ScrapeURL(Tool):
    """
    Tool that scrapes text content from a webpage and returns it in Markdown
    format using BeautifulSoup.
    """

    name = "ScrapeURL"
    description = "Scrape text content from a webpage and return it in Markdown format"

    def run(self, parameters: ScrapeURLParams) -> ScrapeURLResult:  # type: ignore
        """
        Scrape the text content from the given URL and return it in Markdown format.

        Parameters:
        url (str): The URL of the webpage to scrape.

        Returns:
        str or None: The scraped text in Markdown format, or None if an error occurs.
        """

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.3),
            retry=retry_if_exception_type(requests.exceptions.RequestException),
        )
        def fetch_content(url: str, headers: Dict[str, str]) -> requests.Response:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response

        try:
            # Set headers to mimic a regular browser request
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; ScrapeBot/1.0; +http://example.com/bot)"
                )
            }

            # Fetch the webpage content with retries using Tenacity
            response = fetch_content(parameters.url, headers)

            # Determine the encoding and decode the content
            encoding = (
                response.encoding if response.encoding else response.apparent_encoding
            )
            decoded_content = response.content.decode(encoding, errors="replace")

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(decoded_content, "html.parser")

            # Remove script, style elements, and comments
            for element in soup(["script", "style"]):
                element.decompose()
            for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
                comment.extract()

            # Optionally, remove empty tags
            for tag in soup.find_all():
                if not tag.get_text(strip=True):
                    tag.decompose()

            # Convert the cleaned HTML to Markdown
            html = str(soup)

            md = MarkItDown()
            markdown_text = md.convert(html)

            scraped_text = ScrapeURLResult(text=markdown_text)

            return scraped_text

        except requests.exceptions.RequestException as e:
            # Handle network-related exceptions
            logger.error(f"Network error occurred: {e}")
            raise ToolException("Network error occurred") from e

        except UnicodeDecodeError as e:
            # Handle decoding errors
            logger.error(f"Decoding error occurred: {e}")
            raise ToolException("Decoding error occurred") from e

        except (AttributeError, TypeError) as e:
            # Handle parsing errors
            logger.error(f"Parsing error occurred: {e}")
            raise ToolException("Parsing error occurred") from e

        except Exception as e:
            # Handle other unforeseen exceptions
            logger.error(f"An unexpected error occurred: {e}")
            raise ToolException("An unexpected error occurred") from e


if __name__ == "__main__":

    # Example usage of the ScrapeURL tool
    # TEST_URL = "https://en.wikipedia.org/wiki/Web_scraping"
    TEST_URL = "https://www.scdmvonline.com/Vehicle-Owners/Titles"

    tool = ScrapeURL()
    params = ScrapeURLParams(url=TEST_URL)
    result = tool.run(params)
    if result:
        logger.info(result.text)
    else:
        logger.error("An error occurred while scraping the URL.")
