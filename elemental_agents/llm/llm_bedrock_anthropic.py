"""
Abstraction of AWS Bedrock Anthropic module for the LLM class.

Experimental for now.
"""

import json
from typing import Any, Dict, List, Optional

import boto3
from loguru import logger

from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm import LLM


class BedrockAnthropicLLM(LLM):
    """
    BedrockAnthropicLLM class that represents a language model inference type in the
    agent framework. This class is used to interact with Anthropic models via AWS Bedrock.
    """

    def __init__(
        self,
        model_name: str,
        message_stream: bool = False,
        stream_url: str = None,
        parameters: ModelParameters = ModelParameters(),
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        aws_session_token: str = None,
        region_name: str = "us-east-1",
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the BedrockAnthropicLLM object with the given parameters.

        :param model_name: The model ID to use (e.g., 'anthropic.claude-3-sonnet-20240229-v1:0').
        :param message_stream: Whether to stream messages.
        :param stream_url: The URL for streaming messages.
        :param parameters: Model parameters for the LLM.
        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token (for temporary credentials).
        :param region_name: AWS region name.
        :param max_retries: Maximum number of retries for API calls.
        """
        super().__init__(
            model_name, message_stream, stream_url, parameters, max_retries
        )

        # Initialize Bedrock client
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
        )
        
        self._client = session.client("bedrock-runtime")
        self._region_name = region_name

    def _extract_system_message(self, messages: List[Dict]) -> tuple:
        """
        Extract the system message from the list of messages.

        :param messages: List of messages in internal format
        :return: Tuple containing the system message and the remaining messages
        """
        if not messages:
            return "", []

        system_message = ""
        remaining_messages = []

        if messages[0].get("role") == "system":
            content = messages[0].get("content", "")
            if isinstance(content, str):
                system_message = content
            else:
                # Extract only text parts from structured content
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                system_message = " ".join(text_parts)
            remaining_messages = messages[1:]
        else:
            remaining_messages = messages

        return system_message, remaining_messages

    def _convert_message_for_bedrock(self, message: Dict) -> Dict:
        """
        Convert message format to Bedrock Anthropic's expected format.

        :param message: Message in internal format
        :return: Message in Bedrock format
        """
        content = message.get("content", "")

        if isinstance(content, str):
            return {"role": message.get("role", "user"), "content": content}

        # Handle structured content
        bedrock_content = []

        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")

                if part_type == "text":
                    bedrock_content.append(
                        {"type": "text", "text": part.get("text", "")}
                    )
                elif part_type == "image":
                    # Convert ImageContent to Bedrock format
                    try:
                        base64_data = None
                        media_type = part.get("media_type", "image/png")

                        if "base64_data" in part and part["base64_data"]:
                            base64_data = part["base64_data"]
                        elif "file_path" in part and part["file_path"]:
                            # Convert file to base64
                            import base64
                            import os

                            file_path = part["file_path"]
                            if os.path.exists(file_path):
                                with open(file_path, "rb") as image_file:
                                    base64_data = base64.b64encode(
                                        image_file.read()
                                    ).decode("utf-8")
                            else:
                                logger.warning(f"Image file not found: {file_path}")
                                continue

                        if base64_data:
                            bedrock_content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_data,
                                    },
                                }
                            )
                    except Exception as e:
                        logger.error(f"Error processing image for Bedrock: {e}")

        return {
            "role": message.get("role", "user"),
            "content": bedrock_content if bedrock_content else content,
        }

    def _run_non_streaming(self, messages: List[Dict], stop_list: List[str]) -> str:
        """
        Run the model in non-streaming mode.
        """
        system_message, remaining_messages = self._extract_system_message(messages)

        # Convert messages to Bedrock format
        bedrock_messages = [
            self._convert_message_for_bedrock(msg) for msg in remaining_messages
        ]

        # Prepare request body
        request_body = {
            "max_tokens": self._max_tokens,
            "messages": bedrock_messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
        }

        if stop_list:
            request_body["stop_sequences"] = stop_list

        if system_message:
            request_body["system"] = system_message

        # Add thinking parameters for supported models
        if self._thinking_enabled and (
            "claude-3-5" in self._model or "claude-3-7" in self._model
        ):
            thinking_config: Dict[str, Any] = {"type": "enabled"}
            if self._thinking_budget_tokens:
                thinking_config["budget_tokens"] = self._thinking_budget_tokens
            request_body["thinking"] = thinking_config
            logger.debug(f"Using thinking mode with config: {thinking_config}")

        try:
            response = self._client.invoke_model(
                modelId=self._model,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())

            # Handle thinking content
            if "content" in response_body and isinstance(response_body["content"], list):
                result_parts = []
                for block in response_body["content"]:
                    if isinstance(block, dict):
                        if block.get("type") == "thinking":
                            logger.debug(f"Thinking: {block.get('thinking', '')}")
                        elif block.get("type") == "text":
                            result_parts.append(block.get("text", ""))
                return " ".join(result_parts)
            else:
                return str(response_body.get("content", ""))

        except Exception as e:
            logger.error(f"Error in Bedrock Anthropic API call: {str(e)}")
            raise

    async def _process_stream(self, messages: List[Dict], stop_list: List[str]) -> Any:
        """
        Process the stream from the model.
        """
        system_message, remaining_messages = self._extract_system_message(messages)
        bedrock_messages = [
            self._convert_message_for_bedrock(msg) for msg in remaining_messages
        ]

        request_body = {
            "max_tokens": self._max_tokens,
            "messages": bedrock_messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
        }

        if stop_list:
            request_body["stop_sequences"] = stop_list
        if system_message:
            request_body["system"] = system_message

        if self._thinking_enabled and (
            "claude-3-5" in self._model or "claude-3-7" in self._model
        ):
            thinking_config: Dict[str, Any] = {"type": "enabled"}
            if self._thinking_budget_tokens:
                thinking_config["budget_tokens"] = self._thinking_budget_tokens
            request_body["thinking"] = thinking_config

        try:
            response = self._client.invoke_model_with_response_stream(
                modelId=self._model,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            def stream_generator() -> Any:
                """                
                Generator to yield chunks from the Bedrock response stream.
                """
                for event in response["body"]:
                    if "chunk" in event:
                        chunk_data = json.loads(event["chunk"]["bytes"].decode())
                        yield chunk_data

            return stream_generator()

        except Exception as e:
            logger.error(f"Error in Bedrock Anthropic streaming call: {str(e)}")
            raise

    def _extract_content_from_chunk(self, chunk: Any) -> Optional[str]:
        """
        Extract content from a chunk in the stream.
        """
        try:
            if isinstance(chunk, dict):
                if chunk.get("type") == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "text_delta":
                        return delta.get("text", "")
                elif chunk.get("type") == "content_block_start":
                    content_block = chunk.get("content_block", {})
                    if content_block.get("type") == "text":
                        return content_block.get("text", "")
            return None
        except Exception as e:
            logger.error(f"Error extracting content from Bedrock chunk: {str(e)}")
            return None
