"""
Models for the atomic main API requests and responses.
"""

from typing import Any, Dict, List

from pydantic import BaseModel


# Pydantic models for the request and response data
class ChatRequest(BaseModel):
    """
    Request model for the chat endpoint.
    """

    input: str
    session_id: str


class CustomAgentRequest(BaseModel):
    """
    Request model for the custom agent endpoint.
    """

    input: str
    session_id: str
    workflow_yaml: Dict[str, Any]
    settings: Dict[str, Any]
    history: List[str]


class RestartAgentRequest(BaseModel):
    """
    Request model for the restart agent endpoint.
    """

    input: str
    session_id: str
    workflow_yaml: Dict[str, Any]
    settings: Dict[str, Any]
    created_tasks: List[Dict[str, Any]]
