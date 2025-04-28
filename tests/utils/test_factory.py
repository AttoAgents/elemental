"""
Test generic factory function.
"""

import unittest

from elemental_agents.utils.config import ConfigModel
from elemental_agents.utils.factory import factory


class TestFactory(unittest.TestCase):
    """
    Test instantiation of classes from strings.
    """

    def test_factory_method(self) -> None:
        """
        Test factory method.
        """

        config = ConfigModel()
        model_name = config.llama_cpp_embedding_model_name
        vector_size = config.llama_cpp_vector_size

        instance = factory(
            full_class_string="atomic.embeddings.embeddings_llama_cpp.LlamaCppEmbeddings",
            model_name=model_name,
        )
        embed = instance.run("The sky is blue")

        self.assertTrue(instance is not None)
        self.assertTrue(embed is not None)
        self.assertTrue(len(embed.embedding) == vector_size)


if __name__ == "__main__":
    unittest.main()
