"""
Agent Context Manager for file listing and content management.
"""

import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field, validator


class ContextConfig(BaseModel):
    """Configuration for context gathering."""

    # File filtering
    max_file_size: int = Field(
        default=1024 * 1024, description="Maximum file size in bytes (1MB)"
    )
    max_files: int = Field(
        default=100, description="Maximum number of files to include"
    )
    max_content_length: int = Field(
        default=10000, description="Maximum content length per file"
    )

    # File extensions
    include_extensions: Optional[List[str]] = Field(
        default=None, description="File extensions to include (e.g., ['.py', '.md'])"
    )
    exclude_extensions: List[str] = Field(
        default_factory=lambda: [
            ".pyc",
            ".pyo",
            ".pyd",
            ".so",
            ".dll",
            ".exe",
            ".bin",
            ".log",
            ".tmp",
            ".cache",
            ".DS_Store",
        ],
        description="File extensions to exclude",
    )

    # Directory filtering
    exclude_directories: List[str] = Field(
        default_factory=lambda: [
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".env",
            "dist",
            "build",
            ".pytest_cache",
            ".mypy_cache",
        ],
        description="Directory names to exclude",
    )

    # Content options
    include_hidden: bool = Field(
        default=False, description="Include hidden files and directories"
    )
    include_line_numbers: bool = Field(
        default=True, description="Add line numbers to file content"
    )
    truncate_content: bool = Field(default=True, description="Truncate long content")

    @validator("include_extensions")
    def validate_extensions(cls, v):
        if v is not None:
            return [ext if ext.startswith(".") else f".{ext}" for ext in v]
        return v

    @validator("exclude_extensions")
    def validate_exclude_extensions(cls, v):
        return [ext if ext.startswith(".") else f".{ext}" for ext in v]


class FileInfo(BaseModel):
    """Information about a single file."""

    name: str = Field(..., description="File name")
    path: str = Field(..., description="Full file path")
    relative_path: str = Field(..., description="Relative path from base directory")
    size: int = Field(..., description="File size in bytes")
    extension: str = Field(..., description="File extension")
    modified: str = Field(..., description="Last modified timestamp")
    is_text: bool = Field(default=True, description="Whether file is text-based")
    content: Optional[str] = Field(
        default=None, description="File content if requested"
    )
    content_truncated: bool = Field(
        default=False, description="Whether content was truncated"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if file couldn't be read"
    )


class DirectoryInfo(BaseModel):
    """Information about a directory."""

    name: str = Field(..., description="Directory name")
    path: str = Field(..., description="Full directory path")
    relative_path: str = Field(..., description="Relative path from base directory")


class ContextData(BaseModel):
    """Complete context data for a directory."""

    base_path: str = Field(..., description="Base directory path")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    total_files: int = Field(..., description="Total number of files found")
    total_directories: int = Field(..., description="Total number of directories found")
    total_size: int = Field(..., description="Total size of all files in bytes")
    files: List[FileInfo] = Field(default_factory=list, description="List of files")
    directories: List[DirectoryInfo] = Field(
        default_factory=list, description="List of directories"
    )
    config: ContextConfig = Field(..., description="Configuration used")
    errors: List[str] = Field(
        default_factory=list, description="Any errors encountered"
    )


class LLMContextManager(BaseModel):
    """
    Simplified context manager for gathering file listings and content.
    """

    config: ContextConfig = Field(default_factory=ContextConfig)

    class Config:
        arbitrary_types_allowed = True

    def _is_text_file(self, file_path: Path) -> bool:
        """
        Determine if a file is likely to be text-based.

        :param file_path: Path to the file
        :return: True if file is likely text-based
        """
        # Check by extension first
        text_extensions = {
            ".txt",
            ".md",
            ".py",
            ".js",
            ".ts",
            ".html",
            ".css",
            ".json",
            ".xml",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            ".sh",
            ".bat",
            ".ps1",
            ".sql",
            ".r",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".cs",
            ".php",
            ".rb",
            ".go",
            ".rs",
            ".swift",
            ".kt",
            ".scala",
            ".clj",
            ".hs",
            ".ml",
            ".fs",
            ".vb",
            ".pl",
            ".lua",
            ".dart",
            ".elm",
            ".ex",
            ".exs",
            ".jl",
            ".nim",
            ".zig",
        }

        if file_path.suffix.lower() in text_extensions:
            return True

        # For files without extension or unknown extensions, try to read a small sample
        try:
            with open(file_path, "rb") as f:
                sample = f.read(1024)  # Read first 1KB

            # Check if sample contains mostly printable characters
            if not sample:
                return True  # Empty file is considered text

            # Count printable characters
            printable_chars = sum(
                1 for byte in sample if 32 <= byte <= 126 or byte in [9, 10, 13]
            )
            ratio = printable_chars / len(sample)

            return ratio > 0.7  # If more than 70% printable, consider it text

        except Exception:
            return False

    def _read_file_content(
        self, file_path: Path
    ) -> tuple[Optional[str], bool, Optional[str]]:
        """
        Read file content with proper error handling.

        :param file_path: Path to the file
        :return: Tuple of (content, was_truncated, error_message)
        """
        try:
            # Check if it's a text file
            if not self._is_text_file(file_path):
                return None, False, "Binary file"

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.config.max_file_size:
                return None, False, f"File too large: {file_size} bytes"

            # Try to read as text with multiple encodings
            content = None
            for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                return None, False, "Could not decode file"

            # Check if content needs truncation
            was_truncated = False
            if (
                self.config.truncate_content
                and len(content) > self.config.max_content_length
            ):
                content = content[: self.config.max_content_length]
                was_truncated = True

            # Add line numbers if requested
            if self.config.include_line_numbers and content:
                lines = content.split("\n")
                numbered_lines = [f"{i+1:4d}: {line}" for i, line in enumerate(lines)]
                content = "\n".join(numbered_lines)

            return content, was_truncated, None

        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return None, False, str(e)

    def _should_include_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be included based on configuration.

        :param file_path: Path to the file
        :return: True if file should be included
        """
        # Check hidden files
        if not self.config.include_hidden and file_path.name.startswith("."):
            return False

        # Check excluded extensions
        if file_path.suffix.lower() in self.config.exclude_extensions:
            return False

        # Check included extensions (if specified)
        if (
            self.config.include_extensions
            and file_path.suffix.lower() not in self.config.include_extensions
        ):
            return False

        return True

    def _should_include_directory(self, dir_path: Path) -> bool:
        """
        Determine if a directory should be included based on configuration.

        :param dir_path: Path to the directory
        :return: True if directory should be included
        """
        # Check hidden directories
        if not self.config.include_hidden and dir_path.name.startswith("."):
            return False

        # Check excluded directories
        if dir_path.name in self.config.exclude_directories:
            return False

        return True

    def gather_context(
        self, directory_path: Union[str, Path], include_content: bool = True
    ) -> ContextData:
        """
        Gather context data for a directory.

        :param directory_path: Path to the directory to analyze
        :param include_content: Whether to include file contents
        :return: ContextData object with all gathered information
        """
        base_path = Path(directory_path).resolve()

        if not base_path.exists():
            return ContextData(
                base_path=str(base_path),
                total_files=0,
                total_directories=0,
                total_size=0,
                config=self.config,
                errors=[f"Path does not exist: {base_path}"],
            )

        if not base_path.is_dir():
            return ContextData(
                base_path=str(base_path),
                total_files=0,
                total_directories=0,
                total_size=0,
                config=self.config,
                errors=[f"Path is not a directory: {base_path}"],
            )

        files = []
        directories = []
        total_size = 0
        errors = []
        file_count = 0

        try:
            for item in base_path.rglob("*"):
                # Stop if we've reached the file limit
                if file_count >= self.config.max_files:
                    errors.append(
                        f"Reached maximum file limit ({self.config.max_files})"
                    )
                    break

                try:
                    if item.is_file():
                        if not self._should_include_file(item):
                            continue

                        file_size = item.stat().st_size
                        total_size += file_size

                        # Read content if requested
                        content = None
                        content_truncated = False
                        error = None

                        if include_content:
                            content, content_truncated, error = self._read_file_content(
                                item
                            )

                        file_info = FileInfo(
                            name=item.name,
                            path=str(item),
                            relative_path=str(item.relative_to(base_path)),
                            size=file_size,
                            extension=item.suffix.lower(),
                            modified=datetime.fromtimestamp(
                                item.stat().st_mtime
                            ).isoformat(),
                            is_text=self._is_text_file(item),
                            content=content,
                            content_truncated=content_truncated,
                            error=error,
                        )

                        files.append(file_info)
                        file_count += 1

                    elif item.is_dir():
                        if not self._should_include_directory(item):
                            continue

                        dir_info = DirectoryInfo(
                            name=item.name,
                            path=str(item),
                            relative_path=str(item.relative_to(base_path)),
                        )

                        directories.append(dir_info)

                except (PermissionError, OSError) as e:
                    errors.append(f"Cannot access {item}: {e}")
                    continue

        except Exception as e:
            errors.append(f"Error scanning directory: {e}")

        return ContextData(
            base_path=str(base_path),
            total_files=len(files),
            total_directories=len(directories),
            total_size=total_size,
            files=files,
            directories=directories,
            config=self.config,
            errors=errors,
        )

    def format_context(
        self, context_data: ContextData, include_content: bool = True
    ) -> str:
        """
        Format context data as a string for LLM consumption.

        :param context_data: Context data to format
        :param include_content: Whether to include file contents in output
        :return: Formatted context string
        """
        lines = []

        # Header
        lines.append("# Directory Context")
        lines.append(f"**Generated:** {context_data.timestamp}")
        lines.append(f"**Base Path:** {context_data.base_path}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append(f"- **Total Files:** {context_data.total_files}")
        lines.append(f"- **Total Directories:** {context_data.total_directories}")
        lines.append(f"- **Total Size:** {context_data.total_size:,} bytes")
        lines.append("")

        # Errors (if any)
        if context_data.errors:
            lines.append("## Errors")
            for error in context_data.errors:
                lines.append(f"- {error}")
            lines.append("")

        # Directory structure
        if context_data.directories:
            lines.append("## Directories")
            for directory in sorted(
                context_data.directories, key=lambda d: d.relative_path
            ):
                lines.append(f"- {directory.relative_path}/")
            lines.append("")

        # File listing
        if context_data.files:
            lines.append("## Files")
            for file_info in sorted(context_data.files, key=lambda f: f.relative_path):
                size_str = (
                    f"{file_info.size:,} bytes" if file_info.size > 0 else "empty"
                )
                lines.append(f"- **{file_info.relative_path}** ({size_str})")
                if file_info.error:
                    lines.append(f"  - Error: {file_info.error}")
            lines.append("")

        # File contents
        if include_content and context_data.files:
            lines.append("## File Contents")
            lines.append("")

            for file_info in sorted(context_data.files, key=lambda f: f.relative_path):
                if file_info.content is not None:
                    lines.append(f"### {file_info.relative_path}")
                    lines.append(f"**Size:** {file_info.size:,} bytes")
                    if file_info.content_truncated:
                        lines.append("**Note:** Content was truncated")
                    lines.append("")
                    lines.append("```")
                    lines.append(file_info.content)
                    lines.append("```")
                    lines.append("")
                elif file_info.error:
                    lines.append(f"### {file_info.relative_path}")
                    lines.append(f"**Error:** {file_info.error}")
                    lines.append("")

        return "\n".join(lines)

    def get_file_list(self, directory_path: Union[str, Path]) -> List[FileInfo]:
        """
        Get just the file list without content.

        :param directory_path: Path to analyze
        :return: List of FileInfo objects without content
        """
        context_data = self.gather_context(directory_path, include_content=False)
        return context_data.files

    def get_file_content(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Get content of a specific file.

        :param file_path: Path to the file
        :return: File content or None if couldn't read
        """
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return None

        content, _, error = self._read_file_content(path)
        if error:
            logger.warning(f"Error reading {file_path}: {error}")

        return content


# Convenience functions
def create_file_context(
    directory_path: Union[str, Path],
    include_content: bool = True,
    config: Optional[ContextConfig] = None,
) -> str:
    """
    Create a formatted context string for a directory.

    :param directory_path: Path to analyze
    :param include_content: Whether to include file contents
    :param config: Optional configuration
    :return: Formatted context string
    """
    manager = LLMContextManager(config=config or ContextConfig())
    context_data = manager.gather_context(directory_path, include_content)
    return manager.format_context(context_data, include_content)


def create_code_file_context(directory_path: Union[str, Path]) -> str:
    """
    Create a context string optimized for code files.

    :param directory_path: Path to analyze
    :return: Formatted context string
    """
    config = ContextConfig(
        include_extensions=[
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
        ],
        max_file_size=512 * 1024,  # 512KB
        include_line_numbers=True,
        max_content_length=20000,
    )

    return create_file_context(directory_path, include_content=True, config=config)
