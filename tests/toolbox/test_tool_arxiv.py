"""
Test ArXiv search tool.
"""

import json
import unittest

from loguru import logger

from elemental_agents.tools.arxiv_search import (
    ArXivSearch,
    ArXivSearchParams,
    ArXivSearchResult,
)
from elemental_agents.utils.exceptions import ToolException


class TestToolArxiv(unittest.TestCase):
    """
    Test the ArXiv search tool.
    """

    def test_arxiv_search(self) -> None:
        """
        Test searching for a single term.
        """

        arxiv_search = ArXivSearch()

        arxiv_search_json = '{"phrase": "AI agents", "max_results": 2}'
        params = json.loads(arxiv_search_json)
        result = arxiv_search.run(ArXivSearchParams(**params))

        logger.debug(f"Search result {str(result)}")

        self.assertIsInstance(result, ArXivSearchResult)
        self.assertTrue(len(result.papers) > 0)

    def test_arxiv_search_exception(self) -> None:
        """
        Test searching for a single term.
        """

        arxiv_search = ArXivSearch()

        arxiv_search_json = '{"phrase": "ijeorfjoje", "max_results": 10}'
        params = json.loads(arxiv_search_json)

        with self.assertRaises(ToolException):
            arxiv_search.run(ArXivSearchParams(**params))


if __name__ == "__main__":
    unittest.main()
