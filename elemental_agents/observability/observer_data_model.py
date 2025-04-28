"""
Data model for the Observer service.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from elemental_agents.core.taskqueue.task import Status


class ObserverSession(BaseModel):
    """
    ObserverSession class to log the session details.

    observer_id: Unique identifier for the observer instance.
    session_id: Unique identifier for the session.
    task: Description of the task.
    timestamp: Timestamp of the event.
    """

    observer_id: str = Field(
        ..., description="Unique identifier for the observer instance."
    )
    session_id: str = Field(..., description="Unique identifier for the session.")
    first_message: str = Field(..., description="First message in the session.")
    timestamp: str = datetime.now().isoformat()
    workflow_id: str = Field("", description="Unique identifier for the workflow.")


class ObserverInteraction(BaseModel):
    """
    ObserverInteractionRecord class to log the interactions.

    id: Unique identifier for the observer isntance.
    session_id: Unique identifier for the session.
    agent: Name of the agent.
    task: Description of the task.
    interaction: Interaction type.
    timestamp: Timestamp of the event.
    """

    observer_id: str = Field(
        ..., description="Unique identifier for the observer instance."
    )
    session_id: str = Field(..., description="Unique identifier for the session.")
    role: str = Field(..., description="Role in the interaction.")
    message: str = Field(..., description="Content in the interaction.")
    timestamp: str = datetime.now().isoformat()


class ObserverTask(BaseModel):
    """
    ObserverTaskRecord class to log the tasks and their changes.

    id: Unique identifier for the observer isntance.
    task_id: Unique identifier for the task.
    description: Description of the task.
    status: Status of the task.
    result: Result of the task.
    dependencies: List of task IDs that this task depends on.
    context: Dictionary to store the results of the tasks that this task depends on.
    timestamp: Timestamp of the event.
    """

    observer_id: str = Field(
        ..., description="Unique identifier for the observer instance."
    )
    session_id: str = Field(..., description="Unique identifier for the session.")
    task_id: str = Field(..., description="Unique identifier for the task.")
    description: str = Field(..., description="Description of the task.")
    status: Status = Field(Status.BLOCKED, description="Status of the task.")
    result: str = Field("", description="Result of the task.")
    dependencies: List[str] = Field(
        [], description="List of task IDs that this task depends on."
    )
    context: Dict[str, str] = Field(
        {},
        description="""Dictionary to store the results
        of the tasks that this task depends on.""",
    )
    timestamp: str = datetime.now(timezone.utc).isoformat()
    origin: str = Field("", description="Origin of the task.")


class ObserverMessage(BaseModel):
    """
    ObserverMessageRecord class to log the events.

    id: Unique identifier for the observer isntance.
    role: Role in the Message format.
    text: Text in the Message format.
    agent: Agent name.
    task: Task description.
    timestamp: Timestamp of the event.
    """

    observer_id: str = Field(
        ..., description="Unique identifier for the observer instance."
    )
    session_id: str = Field(..., description="Unique identifier for the session.")
    role: str = Field(..., description="Role in the Message format.")
    text: str = Field(..., description="Content in the Message format.")
    agent: str = Field(
        None, description="Name of the agent that generated this message."
    )
    task: str = Field(
        None, description="Description of the task that this message is related to."
    )
    timestamp: str = datetime.now(timezone.utc).isoformat()


class ObserverWorkflow(BaseModel):
    """
    Workflow data model for the Observer service.
    """

    workflow: Dict[str, Any] = Field(
        ...,
        description="Configuration of workflow to be included in JSON/dict format.",
    )
    workflow_name: str = Field(..., description="Name of the workflow.")
    workflow_id: str = Field(..., description="Unique identifier for the workflow.")
    timestamp: str = datetime.now().isoformat()
