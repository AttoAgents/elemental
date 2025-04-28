"""
Test simple text chunking.
"""

import unittest

from elemental_agents.utils.utils import split_text_into_chunks


class TestSplitTextIntoChunks(unittest.TestCase):
    """
    Test splitting a text into chunks with a specified length and overlap.
    """

    def test_split_text_into_chunks(self) -> None:
        """
        Test splitting a text into chunks with a specified length and overlap.
        """

        text = "This is a test sentence to split into chunks."
        chunk_length = 5
        chunk_overlap = 2

        chunks = split_text_into_chunks(text, chunk_length, chunk_overlap)

        self.assertEqual(
            chunks,
            [
                "This is a test sentence",
                "test sentence to split into",
                "split into chunks.",
            ],
        )


if __name__ == "__main__":
    unittest.main()
