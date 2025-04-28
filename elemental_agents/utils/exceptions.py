"""
Custom exceptions for Atomic framework.
"""


class AgentException(Exception):
    """
    Exception raised when agent execution fails.
    """


class EmbeddingTypeError(Exception):
    """
    Exception raised when embedding type is not supported.
    """


class EmbeddingError(Exception):
    """
    Exception raised when embedding generation fails.
    """


class SearchError(Exception):
    """
    Exception raised when database search fails.
    """


class AgentTypeException(Exception):
    """
    Exception raised when an agent type is not supported.
    """


class ToolException(Exception):
    """
    Exception raised when a tool execution fails.
    """
