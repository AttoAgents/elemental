"""
Agent Context Manager for file listing and content management.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator


class ContextConfig(BaseModel):
    """
    Configuration for context gathering.
    """

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

    @field_validator("include_extensions")
    @classmethod
    def validate_extensions(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """
        Validate and format include_extensions to ensure they start with a dot.
        If None, return as is.

        :param v: List of file extensions
        :return: List of formatted file extensions or None if not specified"""
        if v is not None:
            return [ext if ext.startswith(".") else f".{ext}" for ext in v]
        return v

    @field_validator("exclude_extensions")
    @classmethod
    def validate_exclude_extensions(cls, v: List[str]) -> List[str]:
        """
        Validate and format exclude_extensions to ensure they start with a dot.
        If they don't, prepend a dot.

        :param v: List of file extensions
        :return: List of formatted file extensions
        """
        return [ext if ext.startswith(".") else f".{ext}" for ext in v]


class FileInfo(BaseModel):
    """
    Information about a single file.
    """

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
    """
    Information about a directory.
    """

    name: str = Field(..., description="Directory name")
    path: str = Field(..., description="Full directory path")
    relative_path: str = Field(..., description="Relative path from base directory")


class ContextData(BaseModel):
    """
    Complete context data for a directory.
    """

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
    Enhanced context manager for gathering file listings and content with advanced features.
    """

    config: ContextConfig = Field(default_factory=ContextConfig)

    # Context tracking and caching - Using PrivateAttr for private fields
    _context_cache: Dict[str, str] = PrivateAttr(default_factory=dict)
    _context_hash_cache: Dict[str, str] = PrivateAttr(default_factory=dict)

    # Auto-context configuration
    auto_context_directories: List[Path] = Field(default_factory=list)
    auto_context_files: List[Path] = Field(default_factory=list)

    # Context behavior settings
    context_refresh_mode: str = Field(
        default="always"
    )  # "always", "on_change", "manual"
    include_file_content_default: bool = Field(default=True)
    context_section_name: str = Field(default="## Available Context")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_auto_context_directory(self, directory_path: Union[str, Path]) -> None:
        """
        Add a directory to auto-context list.

        :param directory_path: Directory to add
        """
        path = Path(directory_path).resolve()
        if path not in self.auto_context_directories:
            self.auto_context_directories.append(path)
            logger.info(f"Added directory to auto-context: {path}")
        else:
            logger.debug(f"Directory already in auto-context: {path}")

    def add_auto_context_file(self, file_path: Union[str, Path]) -> None:
        """
        Add a file to auto-context list.

        :param file_path: File to add
        """
        path = Path(file_path).resolve()
        if path not in self.auto_context_files:
            self.auto_context_files.append(path)
            logger.info(f"Added file to auto-context: {path}")
        else:
            logger.debug(f"File already in auto-context: {path}")

    def remove_auto_context_directory(self, directory_path: Union[str, Path]) -> bool:
        """
        Remove a directory from auto-context list.

        :param directory_path: Directory to remove
        :return: True if removed, False if not found
        """
        path = Path(directory_path).resolve()
        if path in self.auto_context_directories:
            self.auto_context_directories.remove(path)
            logger.info(f"Removed directory from auto-context: {path}")
            return True
        return False

    def remove_auto_context_file(self, file_path: Union[str, Path]) -> bool:
        """
        Remove a file from auto-context list.

        :param file_path: File to remove
        :return: True if removed, False if not found
        """
        path = Path(file_path).resolve()
        if path in self.auto_context_files:
            self.auto_context_files.remove(path)
            logger.info(f"Removed file from auto-context: {path}")
            return True
        return False

    def set_context_refresh_mode(self, mode: str) -> None:
        """
        Set the context refresh mode.

        :param mode: Refresh mode ("always", "on_change", "manual")
        """
        valid_modes = ["always", "on_change", "manual"]
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid refresh mode: {mode}. Must be one of {valid_modes}"
            )

        self.context_refresh_mode = mode
        logger.info(f"Context refresh mode set to: {mode}")

    def _calculate_path_hash(self, path: Path) -> str:
        """
        Calculate a hash for a path to detect changes.

        :param path: Path to calculate hash for
        :return: Hash string representing current state
        """
        try:
            if not path.exists():
                return "not_found"

            if path.is_file():
                stat = path.stat()
                # Include file size and modification time
                content = f"{stat.st_mtime}_{stat.st_size}"
                return hashlib.md5(content.encode()).hexdigest()[:16]

            elif path.is_dir():
                # For directories, create hash based on all files
                file_info = []
                try:
                    for file_path in sorted(path.rglob("*")):
                        if file_path.is_file() and self._should_include_file(file_path):
                            try:
                                stat = file_path.stat()
                                rel_path = file_path.relative_to(path)
                                file_info.append(
                                    f"{rel_path}_{stat.st_mtime}_{stat.st_size}"
                                )
                            except (OSError, PermissionError):
                                continue
                except (OSError, PermissionError):
                    return "access_denied"

                if not file_info:
                    return "empty_or_no_access"

                content = "_".join(file_info)
                return hashlib.md5(content.encode()).hexdigest()[:16]
            else:
                return "unknown_type"

        except Exception as e:
            logger.warning(f"Error calculating hash for {path}: {e}")
            return f"error_{hash(str(e)) % 10000}"

    def _should_refresh_context(self, path: Path, cache_key: str) -> bool:
        """
        Determine if context should be refreshed for a given path.

        :param path: Path to check
        :param cache_key: Cache key for the context
        :return: True if context should be refreshed
        """
        if self.context_refresh_mode == "always":
            return True
        elif self.context_refresh_mode == "manual":
            return cache_key not in self._context_cache
        elif self.context_refresh_mode == "on_change":
            current_hash = self._calculate_path_hash(path)
            cached_hash = self._context_hash_cache.get(cache_key)

            if current_hash != cached_hash:
                self._context_hash_cache[cache_key] = current_hash
                return True
            return cache_key not in self._context_cache
        else:
            logger.warning(
                f"Unknown refresh mode: {self.context_refresh_mode}, defaulting to 'always'"
            )
            return True

    def get_directory_context_cached(
        self,
        directory_path: Union[str, Path],
        include_content: Optional[bool] = None,
        force_refresh: bool = False,
    ) -> str:
        """
        Get formatted context for a directory with caching.

        :param directory_path: Path to analyze
        :param include_content: Whether to include file contents
        :param force_refresh: Force refresh regardless of refresh mode
        :return: Formatted context string
        """
        path = Path(directory_path).resolve()

        if include_content is None:
            include_content = self.include_file_content_default

        cache_key = f"dir_{path}_{include_content}"

        # Check if refresh is needed
        should_refresh = force_refresh or self._should_refresh_context(path, cache_key)

        if not should_refresh and cache_key in self._context_cache:
            logger.debug(f"Using cached context for directory {path}")
            return self._context_cache[cache_key]

        # Generate new context
        logger.debug(f"Generating fresh context for directory {path}")
        try:
            context_data = self.gather_context(path, include_content)
            formatted_context = self.format_context(context_data, include_content)

            # Cache the result
            self._context_cache[cache_key] = formatted_context

            return formatted_context
        except Exception as e:
            error_msg = f"Error generating context for directory {path}: {e}"
            logger.error(error_msg)
            # Cache error to avoid repeated attempts
            self._context_cache[cache_key] = error_msg
            return error_msg

    def get_file_context_cached(
        self,
        file_path: Union[str, Path],
        force_refresh: bool = False,
        include_metadata: bool = True,
    ) -> str:
        """
        Get formatted context for a single file with caching.

        :param file_path: Path to the file
        :param force_refresh: Force refresh regardless of refresh mode
        :param include_metadata: Include file metadata in output
        :return: Formatted context string
        """
        path = Path(file_path).resolve()
        cache_key = f"file_{path}_{include_metadata}"

        # Check if refresh is needed
        should_refresh = force_refresh or self._should_refresh_context(path, cache_key)

        if not should_refresh and cache_key in self._context_cache:
            logger.debug(f"Using cached context for file {path}")
            return self._context_cache[cache_key]

        # Generate new context
        logger.debug(f"Generating fresh context for file {path}")
        try:
            if not path.exists():
                error_msg = f"File does not exist: {path}"
                self._context_cache[cache_key] = error_msg
                return error_msg

            if not path.is_file():
                error_msg = f"Path is not a file: {path}"
                self._context_cache[cache_key] = error_msg
                return error_msg

            # Read file content
            content = self.get_file_content(path)

            if content is None:
                error_msg = f"Could not read file: {path}"
                self._context_cache[cache_key] = error_msg
                return error_msg

            # Format as context
            context_parts = []

            if include_metadata:
                file_size = path.stat().st_size
                modified_time = datetime.fromtimestamp(path.stat().st_mtime).isoformat()

                context_parts.extend(
                    [
                        f"### {path.name}",
                        f"**Path:** {path}",
                        f"**Size:** {file_size:,} bytes",
                        f"**Modified:** {modified_time}",
                        "",
                    ]
                )
            else:
                context_parts.extend([f"### {path.name}", ""])

            context_parts.extend(["```", content, "```", ""])

            formatted_context = "\n".join(context_parts)

            # Cache the result
            self._context_cache[cache_key] = formatted_context

            return formatted_context

        except Exception as e:
            error_msg = f"Error generating context for file {path}: {e}"
            logger.error(error_msg)
            self._context_cache[cache_key] = error_msg
            return error_msg

    def generate_auto_context(
        self, force_refresh: bool = False, include_content: Optional[bool] = None
    ) -> str:
        """
        Generate the complete auto-context from configured directories and files.

        :param force_refresh: Force refresh of all contexts
        :param include_content: Override default include_content setting
        :return: Complete formatted context string
        """
        if not self.auto_context_directories and not self.auto_context_files:
            return ""

        if include_content is None:
            include_content = self.include_file_content_default

        context_parts = [f"{self.context_section_name}\n"]

        # Process directories
        for directory in self.auto_context_directories:
            try:
                if not directory.exists():
                    context_parts.append(f"\n### Directory: {directory} (NOT FOUND)\n")
                    continue

                context = self.get_directory_context_cached(
                    directory,
                    include_content=include_content,
                    force_refresh=force_refresh,
                )
                context_parts.append(f"\n{context}\n")

            except Exception as e:
                logger.warning(f"Failed to get context for directory {directory}: {e}")
                context_parts.append(f"\n### Directory: {directory} - Error: {e}\n")

        # Process individual files
        for file_path in self.auto_context_files:
            try:
                context = self.get_file_context_cached(
                    file_path, force_refresh=force_refresh, include_metadata=True
                )
                context_parts.append(f"\n{context}\n")

            except Exception as e:
                logger.warning(f"Failed to get context for file {file_path}: {e}")
                context_parts.append(f"\n### File: {file_path} - Error: {e}\n")

        return "".join(context_parts)

    def enhance_instruction_with_context(
        self,
        instruction: str,
        force_refresh: bool = False,
        include_content: Optional[bool] = None,
    ) -> str:
        """
        Enhance an instruction with auto-context.

        :param instruction: Original instruction
        :param force_refresh: Force refresh of all contexts
        :param include_content: Override default include_content setting
        :return: Enhanced instruction with context
        """
        context = self.generate_auto_context(force_refresh, include_content)

        if not context.strip():
            return instruction

        return f"{instruction}\n\n{context}"

    def clear_context_cache(self) -> None:
        """Clear all context caches."""
        self._context_cache.clear()
        self._context_hash_cache.clear()
        logger.debug("All context caches cleared")

    def clear_context_cache_for_path(self, path: Union[str, Path]) -> None:
        """
        Clear context cache for a specific path.

        :param path: Path to clear cache for
        """
        path_resolved = str(Path(path).resolve())
        keys_to_remove = [
            key for key in self._context_cache.keys() if path_resolved in key
        ]

        for key in keys_to_remove:
            del self._context_cache[key]

        hash_keys_to_remove = [
            key for key in self._context_hash_cache.keys() if path_resolved in key
        ]

        for key in hash_keys_to_remove:
            del self._context_hash_cache[key]

        logger.debug(f"Cleared cache for path: {path_resolved}")

    def get_context_status(self) -> Dict:
        """
        Get comprehensive status information about the context manager.

        :return: Dictionary with context status information
        """
        return {
            "auto_directories": [str(d) for d in self.auto_context_directories],
            "auto_files": [str(f) for f in self.auto_context_files],
            "refresh_mode": self.context_refresh_mode,
            "include_file_content_default": self.include_file_content_default,
            "context_section_name": self.context_section_name,
            "cached_contexts": len(self._context_cache),
            "cached_hashes": len(self._context_hash_cache),
            "config": {
                "max_files": self.config.max_files,
                "max_file_size": self.config.max_file_size,
                "max_content_length": self.config.max_content_length,
                "include_extensions": self.config.include_extensions,
                "exclude_extensions": self.config.exclude_extensions,
                "exclude_directories": self.config.exclude_directories,
            },
        }

    def get_cache_statistics(self) -> Dict:
        """
        Get detailed cache statistics.

        :return: Dictionary with cache statistics
        """
        cache_sizes = {}
        total_cache_size = 0

        for key, value in self._context_cache.items():
            size = len(value.encode("utf-8"))
            cache_sizes[key] = size
            total_cache_size += size

        return {
            "total_cached_items": len(self._context_cache),
            "total_cache_size_bytes": total_cache_size,
            "total_cache_size_mb": round(total_cache_size / (1024 * 1024), 2),
            "individual_cache_sizes": cache_sizes,
            "hash_cache_items": len(self._context_hash_cache),
        }

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
    ) -> Tuple[Optional[str], bool, Optional[str]]:
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

        files: List[FileInfo] = []
        directories: List[DirectoryInfo] = []
        total_size = 0
        errors: List[str] = []
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
        lines: List[str] = []

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
