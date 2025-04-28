"""
Data model for agent creation.
"""

from pydantic import BaseModel


class AgentContext(BaseModel):
    """
    Data model for agent context.
    """

    agent_name: str
    agent_persona: str
