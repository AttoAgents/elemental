"""
Data model for the embeddings.
"""

from typing import Any, List, Sequence

from pydantic import BaseModel


class Embedding(BaseModel):
    """
    Embedding class that represents an embedding in the framework.
    """

    text: str
    embedding: List[float] | Sequence[float]

    def __hash__(self) -> int:
        """
        Hash based on the tuple of text and tuple of embedding.
        """
        return hash((self.text, tuple(self.embedding)))

    def __eq__(self, other: Any) -> bool:
        """
        Compare based on the tuple of text and tuple of embedding.
        """
        if isinstance(other, Embedding):
            return self.text == other.text and self.embedding == other.embedding
        return False
