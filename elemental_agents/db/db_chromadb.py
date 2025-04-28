"""
ChromeDB database abstraction implementation in accordance with DB interface.
"""

from typing import List, Mapping, Sequence, Tuple

from chromadb import Collection, EphemeralClient, PersistentClient
from chromadb.api.types import IncludeEnum
from loguru import logger

from elemental_agents.db.vector_db import VectorDB
from elemental_agents.embeddings.data_model import Embedding
from elemental_agents.utils.exceptions import SearchError
from elemental_agents.utils.utils import get_random_string


class DBChromaDB(VectorDB):
    """
    ChromaDB database class.
    """

    def __init__(self, scope: str, file_name: str = None) -> None:
        """
        Initialize the ChromaDB client. Two scopes of clients are supported:
        1. persistent: Persistent client that stores data on disk.
        2. in_memory: In memory client that stores data in memory.

        :param scope: Type of the client to create.
        :param collection_name: Name of the collection to create.
        :param file_name: File name for the persistent client
        """

        super().__init__()
        self._collection_name = "default_collection"
        self._collection: Collection = None

        match scope:
            case "persistent":
                self._file_name = file_name
                self._client = PersistentClient(path=file_name)
                # self._collection = self._create_collection(collection_name)
                logger.debug(
                    f"ChromaDB client created with persistent storage at {file_name}."
                )

            case "in_memory":
                self._client = EphemeralClient()
                # self._collection = self._create_collection(collection_name)
                logger.debug("ChromaDB client created with in-memory storage.")

            case _:
                logger.error("Invalid type provided for ChromaDB client.")
                raise ValueError("Invalid type provided for ChromaDB client.")

    def _create_collection(self, collection_name: str) -> Collection:
        """
        Create a collection in the database.

        :param collection_name: Name of the collection to create.
        :return: Collection object.
        """
        collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        return collection

    def add(self, data: Tuple[List[Embedding], List[str]]) -> None:
        """
        Add data to the collection.

        :param data: Tuple of embeddings and their corresponding IDs.
        """

        if self._collection is None:
            self._collection = self._create_collection(self._collection_name)

        embeddings, ids = data

        if not embeddings:
            raise ValueError("Embeddings list cannot be None or empty.")

        if ids is None:
            ids = [get_random_string() for _ in range(len(embeddings))]
        elif len(ids) != len(embeddings):
            raise ValueError("Length of ids must match length of embeddings.")

        texts = [embedding.text for embedding in embeddings]
        embedding_vectors = [embedding.embedding for embedding in embeddings]

        self._collection.add(ids=ids, embeddings=embedding_vectors, documents=texts)

        logger.debug(f"Added {len(embeddings)} embeddings to the collection.")

    def query(
        self, search_item: Sequence[float], top_n: int = 10, threshold: float = 0.5
    ) -> List[Embedding]:
        """
        Query the ChromaDB database with the given query vector.

        :param query_vector: Query vector to search for.
        :param top_n: Number of results to return.
        :param threshold: Threshold for the similarity metric (distance) of the
            items in the search.
        :return: List of Embedding objects matching the query.
        """
        if not search_item:
            raise ValueError("Query vector cannot be None or empty.")

        try:
            collection = self.get_collection(self._collection_name)

            data = collection.query(
                query_embeddings=[search_item],
                n_results=top_n,
                include=[
                    IncludeEnum.distances,
                    IncludeEnum.documents,
                    IncludeEnum.embeddings,
                ],
            )
            logger.debug(f"Query result data: {data}")
        except Exception as e:
            logger.error(f"Error querying the database with vector {search_item}: {e}")
            raise SearchError(f"Error querying the database: {e}") from e

        if not data["ids"]:
            return []

        embeddings = []
        for doc_list, emb_list, dist_list in zip(
            data["documents"], data["embeddings"], data["distances"]
        ):

            for text, vector, distance in zip(doc_list, emb_list, dist_list):  # type: ignore
                if distance < threshold:
                    embeddings.append(Embedding(text=text, embedding=vector))

        # Remove potential duplicates
        unique_embeddings = []
        seen = set()
        for emb in embeddings:
            identifier = (emb.text, tuple(emb.embedding))
            if identifier not in seen:
                seen.add(identifier)
                unique_embeddings.append(emb)

        return unique_embeddings

    def get_collection(self, collection_name: str = None) -> Collection:
        """
        Get the collection object.

        :return: Collection object.
        """

        if collection_name is not None:
            collection = self._client.get_collection(collection_name)
        else:
            collection = self._collection

        return collection

    def list_collections(self) -> Sequence[str]:
        """
        List all collections in the database.

        :return: List of collection names.
        """

        all_collections = self._client.list_collections()

        return all_collections

    def create_custom_collection(self, collection_name: str) -> Collection:
        """
        Create a custom collection in the database.

        :param collection_name: Name of the collection to create.
        :return: Collection object.
        """

        logger.debug(f"Creating custom collection {collection_name}.")
        new_collection = self._create_collection(collection_name)

        if new_collection.name != collection_name:
            raise ValueError("Collection creation failed.")

        return new_collection

    def delete_custom_collection(self, collection_name: str) -> None:
        """
        Delete a custom collection from the database.

        :param collection_name: Name of the collection to delete.
        """

        logger.debug(f"Deleting custom collection {collection_name}.")
        collection_names = self.list_collections()

        if collection_name not in collection_names:
            raise ValueError("Collection does not exist.")
        # if collection_name == self._collection_name:
        #     raise ValueError("Cannot delete the default collection.")

        self._client.delete_collection(collection_name)

    def add_to_custom_collection(
        self,
        data: Tuple[List[Embedding], List[str], List[Mapping[str, str | int | float]]],
        collection_name: str = None,
    ) -> None:
        """
        Add data to a custom collection.

        :param collection_name: Name of the collection to add data to.
        :param data: Tuple of embeddings and their corresponding IDs.
        """

        if collection_name is None:
            col_name = self._collection_name
        else:
            col_name = collection_name

        collection = self._client.get_or_create_collection(col_name)
        embeddings, ids, metadata = data

        if not embeddings:
            raise ValueError("Embeddings list cannot be None or empty.")

        if ids is None:
            ids = [get_random_string() for _ in range(len(embeddings))]
        elif len(ids) != len(embeddings):
            raise ValueError("Length of ids must match length of embeddings.")

        if metadata is not None:
            if len(metadata) != len(embeddings):
                raise ValueError("Length of metadata must match length of embeddings.")

        texts = [embedding.text for embedding in embeddings]
        embedding_vectors = [embedding.embedding for embedding in embeddings]

        collection.add(
            ids=ids, embeddings=embedding_vectors, documents=texts, metadatas=metadata
        )

    def query_custom_collection(
        self, search_item: Sequence[float], collection_name: str = None, top_n: int = 10
    ) -> List[Tuple[Embedding, Mapping[str, str | int | float]]]:
        """
        Query a custom collection with the given query vector.

        :param collection_name: Name of the collection to query.
        :param query_vector: Query vector to search for.
        :param top_n: Number of results to return.
        :return: List of Embedding objects matching the query.
        """

        logger.info(
            f"Querying custom collection {collection_name} for vector {search_item}."
        )

        if collection_name is None:
            col_name = self._collection_name
        else:
            col_name = collection_name

        collection = self._client.get_collection(col_name)

        if not search_item:
            raise ValueError("Query vector cannot be None or empty.")

        try:
            data = collection.query(
                query_embeddings=[search_item],
                n_results=top_n,
                include=[
                    IncludeEnum.distances,
                    IncludeEnum.documents,
                    IncludeEnum.embeddings,
                    IncludeEnum.metadatas,
                ],
            )
            logger.debug(f"Query result data: {data}")
        except Exception as e:
            logger.error(f"Error querying the database with vector {search_item}: {e}")
            raise SearchError(f"Error querying the database: {e}") from e

        if not data["ids"]:
            return []

        embeddings = []
        for doc_list, emb_list, meta_list in zip(
            data["documents"], data["embeddings"], data["metadatas"]
        ):
            for text, vector, meta in zip(doc_list, emb_list, meta_list):  # type: ignore
                embeddings.append((Embedding(text=text, embedding=vector), meta))

        # Remove potential duplicates
        unique_embeddings = []
        seen = set()
        for emb, meta in embeddings:
            identifier = (emb.text, tuple(emb.embedding))
            if identifier not in seen:
                seen.add(identifier)
                unique_embeddings.append((emb, meta))

        return unique_embeddings


if __name__ == "__main__":

    COLLECTION_NAME = "TestCollection"
    FILE_NAME = "test.db"

    embeddings_data = [
        Embedding(text="text1", embedding=[1, 2, 3]),
        Embedding(text="text2", embedding=[4, 5, 6]),
    ]

    db = DBChromaDB("persistent", FILE_NAME)

    db.add((embeddings_data, None))
