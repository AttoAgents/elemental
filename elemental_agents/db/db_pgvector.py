"""
PgVector database client for PostgreSQL database with pgvector extension.
Abstraction for DB interface.
"""

from typing import List, Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String, create_engine
from sqlalchemy import text as sqlalchemy_text
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, sessionmaker

from elemental_agents.db.vector_db import VectorDB
from elemental_agents.embeddings.data_model import Embedding
from elemental_agents.utils.config import ConfigModel
from elemental_agents.utils.exceptions import SearchError

Base = declarative_base()
config = ConfigModel()
VECTOR_SIZE = config.default_vector_size


class Item(Base):
    """
    Item class to represent the items in the database.
    """

    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[Vector] = mapped_column(Vector(VECTOR_SIZE))

    def __init__(self, text: str, embedding: List[float] | Sequence[float]) -> None:
        self.text = text
        self.embedding = embedding


class DBPgvector(VectorDB):
    """
    Support for PostgreSQL database with pgvector extension as
    vector database.
    """

    def __init__(self, connection_string: str) -> None:
        """
        Initialize the PostgreSQL database client.

        :param connection_string: Connection string for the PostgreSQL database.
        """

        self._engine = create_engine(connection_string)

        self._session = sessionmaker(bind=self._engine)()

        # Create the pgvector extension if it does not exist.
        with self._engine.connect() as conn:
            conn.execute(sqlalchemy_text("CREATE EXTENSION IF NOT EXISTS vector"))

        Base.metadata.create_all(self._engine)

    def add(self, data: List[Embedding]) -> None:
        """
        Add embeddings to the PostgreSQL database.

        :param embeddings: List of embeddings to add to the database.
        """

        if (data is None) or (len(data[0].embedding)) == 0:
            raise ValueError("Empty vector embedding.")

        session = self._session

        for embedding in data:
            item = Item(text="example text", embedding=[0.1, 0.2, 0.3])
            item = Item(embedding.text, embedding.embedding)
            session.add(item)

        session.commit()
        session.close()

    def query(
        self, search_item: List[float] = None, top_n: int = 10, threshold: float = 0.5
    ) -> List[Embedding]:
        """
        Query the PostgreSQL database with the given query vector.

        :param query_vector: Query vector to search for.
        :param top_n: Number of results to return.
        :param threshold: Threshold for the distance metric of the search.
        :return: List of Embedding objects.
        """

        if len(search_item) == 0:
            raise ValueError("Query vector must be of length greater than 0.")

        session = self._session
        try:
            results = (
                session.query(Item)
                .order_by(Item.embedding.l2_distance(search_item))
                .limit(top_n)
                .all()
            )
        except Exception as e:
            raise SearchError(f"Error querying the database: {e}") from e
        session.close()

        result_embeddings = [
            Embedding(text=result.text, embedding=result.embedding)
            for result in results
        ]

        # Remove potential duplicates
        result_embeddings = list(set(result_embeddings))

        return result_embeddings
