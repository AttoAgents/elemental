"""
Example of LLM module usage with images.
"""

from loguru import logger

from elemental_agents.llm_v.data_model import Message, ModelParameters
from elemental_agents.llm_v.llm_factory import LLMFactory

llm_parameters = ModelParameters()
llm_factory = LLMFactory()

models = {
        "openai": "openai|gpt-4.1", 
        "anthropic": "anthropic|claude-sonnet-4-20250514", 
        "ollama": "ollama|gemma3:latest" 
    }

for model_name, model_type in models.items():

    logger.info(f"Creating LLM instance for model: {model_name}")
    llm = llm_factory.create(engine_name=model_type, model_parameters=llm_parameters)

    message = Message(role="user", content="What's in this image?")
    message.add_image("./examples/12-image-interface/Sunflowers_helianthus_annuus.jpg")

    logger.info(message.model_dump())

    result = llm.run(messages=[message])
    logger.info(f"LLM response with image: {result}")
