"""
Helper functions for integrating context manager with LLM messages.
"""

from pathlib import Path
from typing import List, Optional, Union

from elemental_agents.core.context.context_manager import (
    ContextConfig,
    create_file_context,
)
from elemental_agents.llm_v.data_model import Message


def create_context_message(
    directory_path: Union[str, Path],
    include_content: bool = True,
    config: Optional[ContextConfig] = None,
    role: str = "system",
    prefix: str = "Here is the current directory context:",
) -> Message:
    """
    Create a Message object with directory context.

    :param directory_path: Path to analyze
    :param include_content: Whether to include file contents
    :param config: Context configuration
    :param role: Message role
    :param prefix: Prefix text for the context
    :return: Message object with context
    """
    context = create_file_context(directory_path, include_content, config)
    content = f"{prefix}\n\n{context}"

    return Message(role=role, content=content)


def create_code_analysis_messages(
    directory_path: Union[str, Path], user_question: str = "Please analyze this code."
) -> List[Message]:
    """
    Create messages for code analysis with context.

    :param directory_path: Path to analyze
    :param user_question: User's question
    :return: List of messages
    """
    system_msg = create_context_message(
        directory_path=directory_path,
        include_content=True,
        config=ContextConfig(
            include_extensions=[".py", ".js", ".ts", ".md", ".json"],
            include_line_numbers=True,
            max_content_length=15000,
        ),
        role="system",
        prefix="You are a code analysis assistant. Here is the codebase:",
    )

    user_msg = Message(role="user", content=user_question)

    return [system_msg, user_msg]


def create_file_summary_message(directory_path: Union[str, Path]) -> Message:
    """
    Create a message with just file listing (no content).

    :param directory_path: Path to analyze
    :return: Message with file summary
    """
    context = create_file_context(directory_path, include_content=False)

    return Message(
        role="system",
        content=f"Here is the file structure of the directory:\n\n{context}",
    )
