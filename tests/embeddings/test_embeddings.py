"""
Test embedding classes.
"""

import unittest

from elemental_agents.embeddings.embeddings_factory import EmbeddingsFactory
from elemental_agents.embeddings.embeddings_llama_cpp import LlamaCppEmbeddings
from elemental_agents.embeddings.embeddings_ollama import OllamaEmbeddings
from elemental_agents.embeddings.embeddings_openai import OpenAIEmbeddings
from elemental_agents.utils.config import ConfigModel
from elemental_agents.utils.exceptions import EmbeddingTypeError


class TestOllamaEmbeddings(unittest.TestCase):
    """
    Test the OllamaEmbeddings class for text embeddings models from Ollama.
    """

    def setUp(self) -> None:

        self.model_name = "nomic-embed-text"

    def test_run_with_text(self) -> None:
        """
        Run the OllamaEmbeddings class with a text input.
        """

        embeddings = OllamaEmbeddings(model_name=self.model_name)
        result = embeddings.run("The sky is blue")

        self.assertIsNotNone(result.embedding)
        self.assertIsNotNone(result.text)

    def test_run_with_empty_text(self) -> None:
        """
        Test running the OllamaEmbeddings class with an empty text input.
        """
        embeddings = OllamaEmbeddings(model_name=self.model_name)

        with self.assertRaises(ValueError):
            embeddings.run("")


class TestOpenAIEmbeddings(unittest.TestCase):
    """
    Test the OpenAIEmbeddings class for text embeddings models from OpenAI.
    """

    def setUp(self) -> None:

        config = ConfigModel()
        self.api_key = config.openai_api_key
        self.model_name = "text-embedding-3-small"
        # self.vector_size = config.openai_vector_size

    def test_run_with_text(self) -> None:
        """
        Test running the OpenAIEmbeddings class with a text input.
        """

        embeddings = OpenAIEmbeddings(
            model_name=self.model_name, openai_api_key=self.api_key
        )
        result = embeddings.run("The sky is blue")

        self.assertIsNotNone(result.embedding)
        self.assertIsNotNone(result.text)
        # self.assertEqual(len(result.embedding), self.vector_size)

    def test_run_with_empty_text(self) -> None:
        """
        Test running the OpenAIEmbeddings class with an empty text input.
        """

        embeddings = OpenAIEmbeddings(
            model_name=self.model_name, openai_api_key=self.api_key
        )

        with self.assertRaises(ValueError):
            embeddings.run("")


class TestEmbeddingsFactory(unittest.TestCase):
    """
    Test embeddings factory class.
    """

    def test_create_ollama_embeddings(self) -> None:
        """
        Ollama embeddings from the embeddings factory.
        """

        factory = EmbeddingsFactory()
        embeddings = factory.create("ollama|nomic-embed-text")

        self.assertIsInstance(embeddings, OllamaEmbeddings)

    # def test_get_vector_size(self) -> None:
    #     """
    #     Test vector length from the embeddings factory.
    #     """

    #     factory = EmbeddingsFactory()
    #     embeddings = factory.create("ollama|nomic-embed-text")

    #     # vector_size = factory.get_vector_size()

    #     self.assertIsInstance(embeddings, OllamaEmbeddings)
    # self.assertEqual(vector_size, 768)

    def test_create_openai_embeddings(self) -> None:
        """
        OpenAI embeddings from the embeddings factory.
        """

        factory = EmbeddingsFactory()
        embeddings = factory.create("openai|text-embedding-3-small")

        # vector_size = factory.get_vector_size()

        emb = embeddings.run("The sky is blue")

        self.assertIsInstance(embeddings, OpenAIEmbeddings)
        # self.assertEqual(vector_size, 1536)
        self.assertEqual(len(emb.embedding), 1536)

    def test_create_unknown_embeddings(self) -> None:
        """
        Unknown embeddings from the embeddings factory.
        """

        factory = EmbeddingsFactory()

        with self.assertRaises(EmbeddingTypeError):
            factory.create("unknown")


# class TestLlamaCppEmbeddings(unittest.TestCase):
# """
# Test embeddings from llama_cpp module.
# """

# def setUp(self) -> None:

#     config = ConfigModel()
#     self.model_name = config.llama_cpp_embedding_model_name
#     self.vector_size = config.llama_cpp_vector_size

# def test_run_with_text(self) -> None:
#     """
#     Test running the LlamaCppEmbeddings class with a text input.
#     """

#     embeddings = LlamaCppEmbeddings(model_name=self.model_name)
#     result = embeddings.run("The sky is blue")

#     self.assertIsNotNone(result.embedding)
#     self.assertIsNotNone(result.text)
#     self.assertEqual(len(result.embedding), self.vector_size)

# def test_run_with_empty_text(self) -> None:
#     """
#     Test running the LlamaCppEmbeddings class with an empty text input.
#     """

#     embeddings = LlamaCppEmbeddings(model_name=self.model_name)

#     with self.assertRaises(ValueError):
#         embeddings.run("")

# def test_run_with_llama_cpp_from_factory(self) -> None:
#     """
#     Test running the LlamaCppEmbeddings class from the embeddings factory.
#     """

#     factory = EmbeddingsFactory()
#     embeddings = factory.create("llama-cpp")

#     result = embeddings.run("The sky is blue")

#     self.assertIsNotNone(result.embedding)
#     self.assertIsNotNone(result.text)
#     self.assertEqual(len(result.embedding), self.vector_size)


if __name__ == "__main__":
    unittest.main()
