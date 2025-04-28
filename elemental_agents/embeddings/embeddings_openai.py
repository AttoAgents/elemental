"""
Implementation of the Embeddings class using models available with OpenAI service.
"""

from openai import OpenAI

from elemental_agents.embeddings.data_model import Embedding
from elemental_agents.embeddings.embeddings import Embeddings


class OpenAIEmbeddings(Embeddings):
    """
    Embeddings class for the embeddings model served with OpenAI.
    """

    def __init__(
        self,
        model_name: str,
        openai_api_key: str,
        openai_api_url: str = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the OpenAIEmbeddings object.

        :param model_name: The name of the model to use.
        :param openai_api_key: The OpenAI API key.
        :param openai_api_url: The OpenAI API URL.
        :param max_retries: The maximum number of retries for the request.
        """
        super().__init__(max_retries=max_retries)

        self._client = OpenAI(api_key=openai_api_key, base_url=openai_api_url)

        self._model_name = model_name

    def run(self, text: str) -> Embedding:
        """
        Run the embeddings model with the given text.

        :param text: The text to embed.
        :return: The embedding, text pair object.
        """

        @self._get_retry_decorator()
        def _run(text: str) -> Embedding:
            self._validate_text(text)

            response = self._client.embeddings.create(
                input=text, model=self._model_name
            )
            embed = response.data[0].embedding

            self._validate_embedding(embed)

            return Embedding(text=text, embedding=embed)

        return _run(text)


if __name__ == "__main__":

    from rich.console import Console

    from ..utils.config import ConfigModel

    config = ConfigModel()

    OPENAI_MODEL_NAME = "text-embedding-3-small"
    API_KEY = config.openai_api_key

    embeddings = OpenAIEmbeddings(model_name=OPENAI_MODEL_NAME, openai_api_key=API_KEY)
    embedding_result = embeddings.run("This is a test.")

    console = Console()
    console.print(embedding_result)
