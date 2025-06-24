"""
Test generic factory function.
"""

import unittest

from elemental_agents.embeddings.embeddings_ollama import OllamaEmbeddings
from elemental_agents.utils.factory import factory


class TestFactory(unittest.TestCase):
    """
    Test instantiation of classes from strings.
    """

    def test_factory_method(self) -> None:
        """
        Test factory method.
        """

        model_name = "nomic-embed-text"
        vector_size = 768

        instance = factory(
            full_class_string="elemental_agents.embeddings.embeddings_ollama.OllamaEmbeddings",
            model_name=model_name,
        )
        embed = instance.run("The sky is blue")

        self.assertTrue(instance is not None)
        self.assertTrue(embed is not None)
        self.assertTrue(len(embed.embedding) == vector_size)
        self.assertIsInstance(instance, OllamaEmbeddings)


if __name__ == "__main__":
    unittest.main()
