"""
Implementation of the Embeddings class for the embeddings model served with llama-cpp.
"""

from typing import List

from llama_cpp import Llama

from elemental_agents.embeddings.data_model import Embedding
from elemental_agents.embeddings.embeddings import Embeddings


class LlamaCppEmbeddings(Embeddings):
    """
    Embeddings class for the embeddings model served with llama-cpp.
    """

    def __init__(self, model_name: str) -> None:
        """
        Initialize the LlamaCppEmbeddings object.

        :param model_name: The name of the model to use.
        """
        super().__init__()
        self._model_name = model_name
        self._llm = Llama(model_path=self._model_name, embedding=True)

    def run(self, text: str) -> Embedding:
        """
        Run the embeddings model with the given text.

        :param text: The text to embed.
        :return: The embedding, text pair object.
        """

        self._validate_text(text)

        embeddings = self._llm.create_embedding(text)
        embed: List[float] = embeddings["data"][0]["embedding"][0]  # type: ignore

        self._validate_embedding(embed)

        result = Embedding(text=text, embedding=embed)

        return result
