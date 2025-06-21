"""
Enhanced Generic Agent with Context Manager integration and auto-refresh capabilities.
"""

from pathlib import Path
from typing import List, Optional, Union

from loguru import logger

from elemental_agents.core.agent.generic_agent import GenericAgent
from elemental_agents.core.agent_logic.agent_logic import AgentLogic
from elemental_agents.core.context.context_manager import (
    LLMContextManager,
)
from elemental_agents.core.toolbox.toolbox import ToolBox
from elemental_agents.llm.data_model import Message
from elemental_agents.observability.observer import observer


class ContextAwareGenericAgent(GenericAgent):
    """
    Context-Aware Generic Agent that uses LLMContextManager for all context operations.
    """

    def __init__(
        self,
        agent_logic: AgentLogic,
        short_memory_capacity: int,
        toolbox: ToolBox,
        termination_sequence: str,
        context_manager: Optional[LLMContextManager] = None,
        auto_context_directories: Optional[List[Union[str, Path]]] = None,
        auto_context_files: Optional[List[Union[str, Path]]] = None,
        enable_auto_context: bool = True,
    ) -> None:
        """
        Initialize the Context-Aware Generic Agent.

        :param agent_logic: The agent object to use
        :param short_memory_capacity: The short memory capacity of the agent
        :param toolbox: The toolbox object to use for the agent
        :param termination_sequence: The termination sequence to use for the agent
        :param context_manager: LLMContextManager instance (creates default if None)
        :param auto_context_directories: Directories to automatically include in context
        :param auto_context_files: Individual files to automatically include in context
        :param enable_auto_context: Whether to automatically include context in instructions
        """
        super().__init__(
            agent_logic, short_memory_capacity, toolbox, termination_sequence
        )

        # Initialize or use provided context manager
        self.context_manager = context_manager or LLMContextManager()
        self.enable_auto_context = enable_auto_context

        # Add auto-context directories and files
        if auto_context_directories:
            for directory in auto_context_directories:
                self.context_manager.add_auto_context_directory(directory)

        if auto_context_files:
            for file_path in auto_context_files:
                self.context_manager.add_auto_context_file(file_path)

        logger.info(
            f"Context-aware agent initialized with auto-context: {self.enable_auto_context}"
        )

    def run_instruction_inference(
        self, instruction: str, original_instruction: str = "", input_session: str = ""
    ) -> str:
        """
        Enhanced inference with automatic context injection and refresh.

        :param instruction: The task/instruction to run the agent on.
        :param original_instruction: The original instruction to run the agent on.
        :param input_session: The input session for the agent.
        :return: Raw response from the agent.
        """
        # Enhance instruction with context if enabled
        if self.enable_auto_context:
            enhanced_instruction = (
                self.context_manager.enhance_instruction_with_context(
                    instruction,
                    force_refresh=(
                        self.context_manager.context_refresh_mode == "always"
                    ),
                )
            )
        else:
            enhanced_instruction = instruction

        agent_name = self._agent_logic.get_name()

        # Log the original instruction (not the enhanced one with context)
        new_user_message = Message(role="user", content=instruction)
        observer.log_message(
            input_session=input_session,
            message=new_user_message,
            agent_name=agent_name,
            task_description=original_instruction,
        )

        # Run the agent's logic with enhanced instruction
        result = self._agent_logic.run(enhanced_instruction, self._short_memory)

        new_assistant_message = Message(role="assistant", content=result)
        observer.log_message(
            input_session=input_session,
            message=new_assistant_message,
            agent_name=agent_name,
            task_description=original_instruction,
        )

        # Update memory with original instruction (not enhanced)
        self._short_memory.add(new_user_message)
        self._short_memory.add(new_assistant_message)

        return result

    def add_context_directory(self, directory_path: Union[str, Path]) -> None:
        """
        Add a directory to auto-context list.

        :param directory_path: Path to the directory to add.
        """
        self.context_manager.add_auto_context_directory(directory_path)

    def add_context_file(self, file_path: Union[str, Path]) -> None:
        """
        Add a file to auto-context list.

        :param file_path: Path to the file to add.
        """
        self.context_manager.add_auto_context_file(file_path)

    def remove_context_directory(self, directory_path: Union[str, Path]) -> bool:
        """
        Remove a directory from auto-context list.

        :param directory_path: Path to the directory to remove.
        :return: True if the directory was removed, False if it was not found.
        """
        return self.context_manager.remove_auto_context_directory(directory_path)

    def remove_context_file(self, file_path: Union[str, Path]) -> bool:
        """
        Remove a file from auto-context list.

        :param file_path: Path to the file to remove.
        :return: True if the file was removed, False if it was not found.
        """
        return self.context_manager.remove_auto_context_file(file_path)

    def set_context_refresh_mode(self, mode: str) -> None:
        """
        Set the context refresh mode.

        :param mode: Refresh mode to set. Options are "always", "on-demand", or "never".
        """
        self.context_manager.set_context_refresh_mode(mode)

    def clear_context_cache(self) -> None:
        """
        Clear the context cache.
        """
        self.context_manager.clear_context_cache()

    def get_context_status(self) -> dict:
        """
        Get status information about the context configuration.

        :return: Dictionary with context status information
        """
        status = self.context_manager.get_context_status()
        status["auto_context_enabled"] = self.enable_auto_context
        return status

    def enable_auto_context_mode(self) -> None:
        """
        Enable automatic context inclusion.
        """
        self.enable_auto_context = True
        logger.info("Auto-context mode enabled")

    def disable_auto_context_mode(self) -> None:
        """
        Disable automatic context inclusion.
        """
        self.enable_auto_context = False
        logger.info("Auto-context mode disabled")

    def get_directory_context(
        self, directory_path: Union[str, Path], include_content: bool = True
    ) -> str:
        """
        Get formatted context for a directory.

        :param directory_path: Path to the directory to get context for.
        :param include_content: Whether to include file content in the context.
        :return: Formatted context string for the directory.
        """
        return self.context_manager.get_directory_context_cached(
            directory_path, include_content
        )

    def get_file_context(self, file_path: Union[str, Path]) -> str:
        """
        Get formatted context for a file.

        :param file_path: Path to the file to get context for.
        :return: Formatted context string for the file.
        """
        return self.context_manager.get_file_context_cached(file_path)
        """
        Get status information about the context configuration.

        :return: Dictionary with context status information
        """
        return {
            "directories": [str(d) for d in self.auto_context_directories],
            "files": [str(f) for f in self.auto_context_files],
            "refresh_mode": self.context_refresh_mode,
            "include_file_content": self.include_file_content,
            "cached_contexts": len(self._context_cache),
            "tracked_hashes": len(self._last_context_hash),
        }
