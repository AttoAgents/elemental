"""
Example of using the Bedrock Anthropic LLM with Elemental. 
Factory-based approach.
"""

from loguru import logger

from elemental_agents.llm.data_model import Message, ModelParameters

from elemental_agents.utils.config import ConfigModel
from elemental_agents.llm.llm_factory import LLMFactory

config = ConfigModel()
llm_factory = LLMFactory()

parameters = ModelParameters(
    temperature=0.0,
    top_p=1.0,
    max_tokens=10000,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["<PAUSE>", "<STOP>"],
)

llm = llm_factory.create(
    "bedrock_anthropic|anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_parameters=parameters
)

msgs = [Message(role="user", content="What is the capital of France?")]

response = llm.run(messages=msgs, stop_word=None)

logger.info(f"Response: {response}")
