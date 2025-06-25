"""
Framework configuration class to handle setting and default values of components
independent of the configuration of the workflow.

Using the programmatic updates:

config = ConfigModel()
ConfigModel.update_parameter("db_host", "runtime_host_value")
config = ConfigModel.reload()
"""

import json
import os
from typing import Any, List, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ATTO_ENV_FILE = os.getenv("ATTO_ENV_FILE", ".env")


class ConfigModel(BaseSettings):
    """
    Configuration settings.
    """

    model_config = SettingsConfigDict(env_file=ATTO_ENV_FILE, env_file_encoding="utf-8")

    default_db_type: str = "chromadb"
    default_db_connection_string: str = "chroma.db"

    # LLM
    ollama_llm_model_name: str = ""
    ollama_embedding_model_name: str = ""
    ollama_vector_size: int = 0
    ollama_temperature: float = 0.0
    ollama_max_tokens: int = 0
    ollama_stop_words: List[str] = []
    ollama_streaming: bool = False
    ollama_url: str = "http://localhost:11434"

    # OpenAI configuration
    openai_api_key: str = "Not set"
    openai_llm_model_name: str = ""
    openai_embedding_model_name: str = ""
    openai_vector_size: int = 0
    openai_temperature: float = 0.0
    openai_max_tokens: int = 0
    openai_stop_words: List[str] = []
    openai_streaming: bool = False
    openai_url: str = "https://api.openai.com/v1"

    # Custom OpenAI Compatible LLM API configuration
    custom_api_key: str = "Not set"
    custom_llm_model_name: str = ""
    custom_embedding_model_name: str = ""
    custom_vector_size: int = 0
    custom_temperature: float = 0.0
    custom_max_tokens: int = 0
    custom_stop_words: List[str] = []
    custom_streaming: bool = False
    custom_url: str = ""

    # Anthropic configuration
    anthropic_api_key: str = "Not set"
    anthropic_llm_model_name: str = ""
    anthropic_temperature: float = 0.0
    anthropic_max_tokens: int = 0
    anthropic_stop_words: List[str] = []
    anthropic_streaming: bool = False

    # Llama.cpp configuration
    llama_llm_model_name: str = ""
    llama_cpp_embedding_model_name: str = ""
    llama_cpp_vector_size: int = 0
    llama_cpp_temperature: float = 0.0
    llama_cpp_max_tokens: int = 0
    llama_cpp_stop_words: List[str] = []
    llama_cpp_streaming: bool = False
    llama_cpp_model_directory: str = "models"

    # Azure OpenAI configuration
    azure_openai_api_key: str = "Not set"
    azure_openai_endpoint: str = ""
    azure_openai_deployment_name: str = ""
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_openai_streaming: bool = False

    # AWS Bedrock configuration
    aws_access_key_id: str = "Not set"
    aws_secret_access_key: str = "Not set"
    aws_session_token: str = ""
    aws_region: str = "us-east-1"
    bedrock_anthropic_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    bedrock_anthropic_streaming: bool = False
    aws_anthropic_version: str = "bedrock-2023-05-31"

    # Google Gemini
    gemini_api_key: str = "Not set"
    gemini_llm_model_name: str = "gemini-2.5-flash"
    gemini_streaming: bool = False
    gemini_temperature: float = 0.0
    gemini_max_tokens: int = 0
    gemini_stop_words: List[str] = []

    default_engine: str = ""
    default_model_name: str = ""
    default_vector_size: int = 0
    default_embedding_model_name: str = ""

    # Observer
    observer_destination: Literal["screen", "file", "db", "webhook", "none"] = "screen"
    observer_file_name: str = "observer.log"
    observer_db_name: str = "observer_db"
    observer_database_connection_string: str = "sqlite:///observer.db"
    observer_webhook_url: str = ""

    # Streaming
    websocket_url: str = ""

    # Agent
    max_agent_iterations: int = 15
    max_multiagent_iterations: int = 20
    relaxed_react: bool = False

    unit_default_selector: str = "identity"
    agent_default_type: str = "ReAct"

    # Tools
    google_search_api_key: str = "Not set"
    google_cse_id: str = "Not set"
    google_search_timeout: int = 5

    wikipedia_user_agent: str = "ExampleAgent/1.0 (info@example.com)"
    arxiv_search_timeout: int = 20

    # Memory
    use_long_memory: bool = False
    long_memory_items: int = 5
    long_memory_threshold: float = 0.25
    long_memory_db_string: str = "chromadb|chroma.db"
    long_memory_embeddings_engine: str = ""

    short_memory_items: int = -1

    # MCP
    mcpServers: str = ""

    # Context
    max_file_size: int = 1024 * 1024  # 1 MB

    @classmethod
    def update_parameter(cls, key: str, value: Any) -> None:
        """
        Update a configuration parameter during runtime (for current session only).
        If the value is a list, serialize it to JSON.
        """
        if isinstance(value, list):
            os.environ[key] = json.dumps(value)
        else:
            os.environ[key] = str(value)

    @classmethod
    def reload(cls) -> "ConfigModel":
        """
        Reload the configuration to reflect updated environment variables.
        Automatically deserialize lists if necessary.
        """
        new_config = cls()

        # Automatically detect and deserialize list-type attributes
        for field_name, field_type in cls.__annotations__.items():
            if field_type == List[str] and field_name in os.environ:
                try:
                    setattr(new_config, field_name, json.loads(os.environ[field_name]))
                except json.JSONDecodeError:
                    pass  # Leave it unchanged if decoding fails

        return new_config
