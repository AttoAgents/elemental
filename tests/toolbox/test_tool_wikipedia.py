"""
Test the Wikipedia tool.
"""

import json
import unittest

from loguru import logger

from elemental_agents.tools.wikipedia_search import (
    WikipediaSearch,
    WikipediaSearchParams,
    WikipediaSearchResult,
)


class TestToolWikipedia(unittest.TestCase):
    """
    Test the Wikipedia search tool.
    """

    def test_search(self) -> None:
        """
        Test searching for a single term.
        """

        wikipedia_search = WikipediaSearch()

        wikipedia_search_json = '{"phrase": "Python (programming language)"}'
        params = json.loads(wikipedia_search_json)
        result = wikipedia_search.run(WikipediaSearchParams(**params))

        logger.debug(f"Search result {str(result)}")

        self.assertIsInstance(result, WikipediaSearchResult)
        self.assertIn(
            """Python is a high-level, general-purpose""",
            result.content,
        )


if __name__ == "__main__":
    unittest.main()
