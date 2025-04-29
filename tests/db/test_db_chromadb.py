"""
Test Chroma vector database operations.
"""

import unittest

from elemental_agents.db.db_chromadb import DBChromaDB
from elemental_agents.embeddings.data_model import Embedding
from elemental_agents.utils.utils import get_random_string, remove_directory_if_exists


class TestDBChromaDB(unittest.TestCase):
    """
    Test Chroma vector database operations.
    """

    def test_create_in_memory(self) -> None:
        """
        Test creating in memory database.
        """
        collection_name = "test_collection"
        db = DBChromaDB(scope="in_memory", file_name=None)

        self.assertIsNotNone(db)

    def test_create_persistent(self) -> None:
        """
        Test creating a persistent database.
        """
        file_name = get_random_string(6) + ".db"
        collection_name = "test_collection"

        db = DBChromaDB("persistent", file_name)

        self.assertIsNotNone(db)
        remove_directory_if_exists(file_name)

    def test_add_and_query_persistent(self) -> None:
        """
        Test adding and querying embeddings in an persistent database.
        """
        file_name = get_random_string(6) + ".db"
        collection_name = "test_collection"

        embeddings = [
            Embedding(text="text1", embedding=[1, 2, 3]),
            Embedding(text="text2", embedding=[4, 5, 6]),
        ]

        db = DBChromaDB("persistent", file_name)

        data = (embeddings, ["id1", "id2"])
        db.add(data)

        query_vector = [7.0, 8.0, 9.0]
        result = db.query(query_vector, 2)
        self.assertEqual(len(result), 2)

        remove_directory_if_exists(file_name)

    def test_add_and_query_in_memory(self) -> None:
        """
        Test adding and querying embeddings in an in-memory database.
        """

        collection_name = "test_collection"

        embeddings = [
            Embedding(text="text01", embedding=[1.0, 2.0, 3.0]),
            Embedding(text="text02", embedding=[4.0, 5.0, 6.0]),
            Embedding(text="text03", embedding=[7.0, 8.0, 9.0]),
            Embedding(text="text04", embedding=[1.5, 2.5, 3.5]),
            Embedding(text="text05", embedding=[4.5, 5.5, 6.5]),
            Embedding(text="text06", embedding=[7.5, 8.5, 9.5]),
            Embedding(text="text07", embedding=[1.1, 2.1, 3.1]),
            Embedding(text="text08", embedding=[4.1, 5.1, 6.1]),
            Embedding(text="text09", embedding=[7.1, 8.1, 9.1]),
            Embedding(text="text10", embedding=[1.9, 2.9, 3.9]),
            Embedding(text="text11", embedding=[4.9, 5.9, 6.9]),
            Embedding(text="text12", embedding=[7.9, 8.9, 9.9]),
        ]

        db = DBChromaDB("in_memory", file_name=None)

        data = (
            embeddings,
            [
                "id1",
                "id2",
                "id3",
                "id4",
                "id5",
                "id6",
                "id7",
                "id8",
                "id9",
                "id10",
                "id11",
                "id12",
            ],
        )
        db.add(data)

        query_vector = [7.0, 8.0, 9.0]

        max_elements = 5
        result = db.query(query_vector, max_elements)

        self.assertNotEqual(len(result), 0)
        self.assertEqual(len(result), max_elements)


if __name__ == "__main__":
    unittest.main()
