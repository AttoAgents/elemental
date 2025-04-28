"""
Interface for the Embeddings in the framework.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Sequence

import requests.exceptions
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from elemental_agents.embeddings.data_model import Embedding
from elemental_agents.utils.exceptions import EmbeddingError


class Embeddings(ABC):
    """
    Interface for the Embeddings in the framework.
    """

    def __init__(self, max_retries: int = 3) -> None:
        """
        Initialize the Embeddings object.
        """

        self._max_retries = max_retries

    def _validate_text(self, text: str) -> None:
        """
        Validate the text input.
        """
        if len(text) == 0:
            logger.error("No text has been provided to the model.")
            raise ValueError("No text has been provided to the model.")

    def _validate_embedding(self, embed: List[float] | Sequence[float]) -> None:
        """
        Validate the embedding output.
        """
        if len(embed) == 0 or embed is None:
            logger.error("No embedding has been generated.")
            raise EmbeddingError("No embedding has been generated.")

    def _get_retry_decorator(self) -> Callable:
        """
        Returns a configured retry decorator using the instance's max_retries value.
        """
        return retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(
                (
                    requests.exceptions.RequestException,
                    ConnectionError,
                    TimeoutError,
                    Exception,
                )
            ),
            reraise=True,
        )

    @abstractmethod
    def run(self, text: str) -> Embedding:
        """
        Calculate the embeddings for the given text.
        """
