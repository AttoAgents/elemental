"""
This is an example of how to use the ConfigModel
class to create a configuration model and print
its parameters.
"""

from pprint import pprint

from elemental_agents.utils.config import ConfigModel


# Create a configuration model instance
config = ConfigModel()

pprint(config.model_dump())
