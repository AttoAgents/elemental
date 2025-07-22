"""
Embedding factory module that creates embeddings instances based on the
configuration file. The embeddings instances are used to generate embeddings
for the input text data.
"""

from loguru import logger

from elemental_agents.embeddings.embeddings import Embeddings

from elemental_agents.utils.config import ConfigModel
from elemental_agents.utils.exceptions import EmbeddingTypeError


class EmbeddingsFactory:
    """
    Factory class for creating embeddings instances with parameters from the
    configuration file.
    """

    def __init__(self) -> None:
        """
        Initialize the embeddings factory with the configuration model.
        """
        self._config = ConfigModel()
        self._local_engine_name: str = None

    def create(self, engine_name: str = None) -> Embeddings:
        """
        Create an embeddings instance based on the engine name. If the engine
        name is not provided, the default engine is used that is specified in
        the configuration file.

        :param engine_name: The name of the engine to use.
        :return: An instance of the Embeddings class.
        """

        embeddings_parameters = []

        if engine_name is None:
            local_engine_name = self._config.default_engine
            self._local_engine_name = local_engine_name
        else:
            embeddings_parameters = engine_name.split("|")
            local_engine_name = embeddings_parameters[0]
            self._local_engine_name = local_engine_name

        if self._local_engine_name == "ollama":

            from elemental_agents.embeddings.embeddings_ollama import OllamaEmbeddings

            local_model_name = (
                embeddings_parameters[1]
                if len(embeddings_parameters) > 1
                else self._config.ollama_embedding_model_name
            )

            logger.debug("Creating Ollama embeddings instance.")
            logger.debug(f"Model name: {local_model_name}")

            ollama_embeddings = OllamaEmbeddings(model_name=local_model_name)
            return ollama_embeddings

        if self._local_engine_name == "openai":

            from elemental_agents.embeddings.embeddings_openai import OpenAIEmbeddings

            local_model_name = (
                embeddings_parameters[1]
                if len(embeddings_parameters) > 1
                else self._config.openai_embedding_model_name
            )

            logger.debug("Creating OpenAI embeddings instance.")
            logger.debug(f"Model name: {local_model_name}")

            openai_embeddings = OpenAIEmbeddings(
                model_name=local_model_name,
                openai_api_key=self._config.openai_api_key,
            )
            return openai_embeddings

        if self._local_engine_name == "llama-cpp":

            from elemental_agents.embeddings.embeddings_llama_cpp import LlamaCppEmbeddings

            local_model_name = (
                embeddings_parameters[1]
                if len(embeddings_parameters) > 1
                else self._config.llama_cpp_embedding_model_name
            )

            logger.debug("Creating LlamaCpp embeddings instance.")
            logger.debug(f"Model name: {local_model_name}")

            llama_cpp_embeddings = LlamaCppEmbeddings(model_name=local_model_name)
            return llama_cpp_embeddings

        logger.error(f"Unknown embeddings engine name: {self._local_engine_name}.")
        raise EmbeddingTypeError(
            f"Unknown embeddings engine name: {self._local_engine_name}."
        )

    def get_vector_size(self) -> int:
        """
        Get the vector size for the embeddings engine.

        :return: The vector size for the embeddings engine.
        """
        if self._local_engine_name == "ollama":
            return self._config.ollama_vector_size
        if self._local_engine_name == "openai":
            return self._config.openai_vector_size
        logger.error(f"Unknown embeddings engine name: {self._local_engine_name}.")
        raise ValueError(f"Unknown embeddings engine name: {self._local_engine_name}.")


if __name__ == "__main__":

    from rich.console import Console

    embeddings_factory = EmbeddingsFactory()
    embeddings = embeddings_factory.create()

    em = embeddings.run("Hello, world!")

    console = Console()
    console.print(em)
