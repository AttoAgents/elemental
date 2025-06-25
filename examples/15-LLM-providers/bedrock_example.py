"""
Example of using the Bedrock Anthropic LLM with Elemental.
"""

from loguru import logger

from elemental_agents.llm.data_model import Message, ModelParameters
from elemental_agents.llm.llm_bedrock_anthropic import BedrockAnthropicLLM
from elemental_agents.utils.config import ConfigModel

config = ConfigModel()

parameters = ModelParameters(
    temperature=0.0,
    top_p=1.0,
    max_tokens=10000,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["<PAUSE>", "<STOP>"],
)

llm = BedrockAnthropicLLM(
    model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
    message_stream=False,
    stream_url=None,
    parameters=parameters,
    max_retries=3,
    aws_access_key_id=config.aws_access_key_id,
    aws_secret_access_key=config.aws_secret_access_key,
    region_name=config.aws_region,
    anthropic_version=config.anthropic_version
)

msgs = [Message(role="user", content="What is the capital of France?")]

response = llm.run(messages=msgs, stop_word=None)

logger.info(f"Response: {response}")
