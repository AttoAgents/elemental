"""
Implementation of the Embeddings class for the embeddings model served with Ollama.
"""

from ollama import Client

from elemental_agents.embeddings.data_model import Embedding
from elemental_agents.embeddings.embeddings import Embeddings


class OllamaEmbeddings(Embeddings):
    """
    Embeddings class for the embeddings model served with Ollama.
    """

    def __init__(
        self,
        model_name: str,
        ollama_url: str = None,
        ollama_port: int = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the OllamaEmbeddings object.

        :param model_name: The name of the model to use.
        """
        super().__init__(max_retries=max_retries)

        if ollama_url is None and ollama_port is None:
            self._client = Client()
        else:
            url: str = f"http://{ollama_url}:{ollama_port}"
            self._client = Client(host=url)

        self._model_name = model_name

    def run(self, text: str) -> Embedding:
        """
        Run the embeddings model with the given text.
        Retry on failure.

        :param text: The text to embed.
        :return: The embedding, text pair object.
        """

        @self._get_retry_decorator()
        def _run(text: str) -> Embedding:
            self._validate_text(text)

            response = self._client.embeddings(model=self._model_name, prompt=text)
            embed = response["embedding"]

            self._validate_embedding(embed)

            return Embedding(text=text, embedding=embed)

        return _run(text)
