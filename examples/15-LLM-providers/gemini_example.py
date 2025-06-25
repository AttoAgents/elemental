"""
Example of using the Gemini LLM with Elemental.
"""

from loguru import logger

from elemental_agents.llm.data_model import Message, ModelParameters
from elemental_agents.llm.llm_gemini import GeminiLLM
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

llm = GeminiLLM(
    model_name="gemini-2.5-flash",
    message_stream=False,
    stream_url=None,
    parameters=parameters,
    gemini_api_key=config.gemini_api_key,
    max_retries=3,
)

msgs = [Message(role="user", content="How to calculate value of pi using Monte Carlo method?")]

response = llm.run(messages=msgs, stop_word=None)

logger.info(f"Response: {response}")
