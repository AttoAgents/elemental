"""
Database module for the Observer service.
"""

import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, List

from loguru import logger
from sqlalchemy import JSON, Column, DateTime, Enum, String, Text, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, joinedload, sessionmaker

from elemental_agents.core.taskqueue.task import Status
from elemental_agents.observability.observer_data_model import (
    ObserverInteraction,
    ObserverMessage,
    ObserverSession,
    ObserverTask,
    ObserverWorkflow,
)

# Define the base class for declarative models
Base = declarative_base()


class ObserverSessionRecord(Base):  # type: ignore
    """
    ObserverSessionRecord schema
    """

    __tablename__ = "observer_session_record"

    id = Column(
        String(36),  # 36 characters for a UUID string
        primary_key=True,
        default=lambda: str(uuid.uuid4()),  # Generate UUID as string
        nullable=False,
        unique=True,
        comment="Unique identifier for the session.",
    )
    observer_id = Column(
        String,
        nullable=False,
        comment="Unique identifier for the observer instance.",
    )
    session_id = Column(
        String,
        nullable=False,
        comment="Unique identifier for the session.",
    )
    first_message = Column(
        String,
        nullable=False,
        comment="First message in the session.",
    )
    timestamp = Column(
        DateTime,
        default=datetime.now,
        nullable=False,
        comment="Timestamp of the session creation.",
    )
    workflow_id = Column(
        String,
        nullable=True,
        comment="Unique identifier for the workflow.",
    )


class ObserverInteractionRecord(Base):  # type: ignore
    """
    ObserverInteractionRecord schema
    """

    __tablename__ = "observer_interaction_record"

    id = Column(
        String(36),  # 36 characters for a UUID string
        primary_key=True,
        default=lambda: str(uuid.uuid4()),  # Generate UUID as string
        nullable=False,
        unique=True,
        comment="Unique identifier for the interaction record.",
    )
    observer_id = Column(
        String,
        nullable=False,
        comment="Unique identifier for the observer instance.",
    )
    session_id = Column(
        String,
        nullable=False,
        comment="Unique identifier for the session.",
    )
    role = Column(
        String,
        nullable=False,
        comment="Role of the interaction.",
    )
    message = Column(
        String,
        nullable=False,
        comment="Content of the interaction.",
    )
    timestamp = Column(
        DateTime,
        default=datetime.now,
        nullable=False,
        comment="Timestamp of the interaction.",
    )


class ObserverTaskRecord(Base):  # type: ignore
    """
    ObserverTaskRecord schema
    """

    __tablename__ = "observer_task_record"

    id = Column(
        String(36),  # 36 characters for a UUID string
        primary_key=True,
        default=lambda: str(uuid.uuid4()),  # Generate UUID as string
        nullable=False,
        unique=True,
        comment="Unique identifier for the task record.",
    )
    observer_id = Column(
        String, nullable=False, comment="Unique identifier for the observer instance."
    )
    session_id = Column(
        String, nullable=False, comment="Unique identifier for the session."
    )
    task_id = Column(String, nullable=False, comment="Unique identifier for the task.")
    description = Column(Text, nullable=True, comment="Description of the task.")
    status = Column(
        Enum(Status),
        default=Status.BLOCKED,
        nullable=False,
        comment="Status of the task.",
    )
    result = Column(Text, default="", nullable=True, comment="Result of the task.")
    dependencies = Column(
        JSON,
        default=[],
        nullable=True,
        comment="List of task IDs that this task depends on.",
    )
    context = Column(
        JSON,
        default={},
        nullable=True,
        comment="Dictionary to store the results of dependent tasks.",
    )
    timestamp = Column(
        DateTime,
        default=datetime.now,
        nullable=False,
        comment="Timestamp of the event.",
    )
    origin = Column(
        String,
        nullable=True,
        comment="Original instruction of the task.",
    )


class ObserverMessageRecord(Base):  # type: ignore
    """
    ObserverMessageRecord schema
    """

    __tablename__ = "observer_message_record"

    id = Column(
        String(36),  # 36 characters for a UUID string
        primary_key=True,
        default=lambda: str(uuid.uuid4()),  # Generate UUID as string
        nullable=False,
        unique=True,
        comment="Unique identifier for the message record.",
    )
    observer_id = Column(
        String, nullable=False, comment="Unique identifier for the observer instance."
    )
    session_id = Column(
        String, nullable=False, comment="Unique identifier for the session."
    )
    role = Column(String, nullable=False, comment="Role in the Message format.")
    text = Column(Text, nullable=False, comment="Content in the Message format.")
    agent = Column(
        String, nullable=True, comment="Name of the agent that generated this message."
    )
    task = Column(
        String,
        nullable=True,
        comment="Description of the task that this message is related to.",
    )
    timestamp = Column(
        DateTime,
        default=datetime.now,
        nullable=False,
        comment="Timestamp of the event.",
    )


class ObserverWorkflowRecord(Base):  # type: ignore
    """
    ObserverWorkflowRecord schema
    """

    __tablename__ = "observer_workflow_record"

    id = Column(
        String(36),  # 36 characters for a UUID string
        primary_key=True,
        default=lambda: str(uuid.uuid4()),  # Generate UUID as string
        nullable=False,
        unique=True,
        comment="Unique identifier for the workflow record.",
    )
    workflow_id = Column(
        String,
        unique=True,
        nullable=False,
        comment="Unique identifier for the workflow.",
    )
    workflow_name = Column(
        String,
        nullable=False,
        comment="Name of the workflow.",
    )
    workflow = Column(JSON, nullable=False, comment="Workflow JSON configuration.")
    timestamp = Column(
        String,
        default=datetime.now,
        nullable=False,
        comment="Timestamp of the workflow creation.",
    )


class ObserverToolCallRecord(Base):  # type: ignore
    """
    ObserverToolCallRecord schema
    """

    __tablename__ = "observer_tool_call_record"

    id = Column(
        String(36),  # 36 characters for a UUID string
        primary_key=True,
        default=lambda: str(uuid.uuid4()),  # Generate UUID as string
        nullable=False,
        unique=True,
        comment="Unique identifier for the tool call record.",
    )
    observer_id = Column(
        String, nullable=False, comment="Unique identifier for the observer instance."
    )
    session_id = Column(
        String, nullable=False, comment="Unique identifier for the session."
    )
    tool_name = Column(String, nullable=False, comment="Name of the tool called.")
    parameters = Column(JSON, nullable=False, comment="Parameters passed to the tool.")
    result = Column(JSON, nullable=True, comment="Result returned by the tool.")
    timestamp = Column(
        DateTime,
        default=datetime.now,
        nullable=False,
        comment="Timestamp of the event.",
    )


# ObserverDatabase class to handle database operations
class ObserverDatabase:
    """
    Database class to handle the connection to the observer database.
    """

    def __init__(self, connection_string: str) -> None:
        """
        Initializes the database engine and session.

        :param connection_string: The database connection string.
        """
        self._engine = create_engine(connection_string, echo=False)
        self._session_local = sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )

    def create(self) -> None:
        """
        Creates all tables in the database based on the models.
        """
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """
        Provides a transactional scope around a series of operations.

        :return: An iterator over a SQLAlchemy session.
        """
        session: Session = self._session_local()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Observer Database Error: {e}")
        finally:
            session.close()

    def get_all_sessions(self) -> List[ObserverSessionRecord]:
        """
        Retrieves all sessions from the ObserverSessionRecord table.

        :return: A list of all session records.
        """
        with self.session_scope() as session:
            return session.query(ObserverSessionRecord).all()

    def add_session(self, session_data: ObserverSession) -> None:
        """
        Adds a new session to the ObserverSessionRecord table.

        :param session_data: A dictionary containing session details.
        """
        with self.session_scope() as session:
            new_session = ObserverSessionRecord(
                observer_id=session_data.observer_id,
                session_id=session_data.session_id,
                first_message=session_data.first_message,
                workflow_id=session_data.workflow_id,
            )
            session.add(new_session)

    def delete_session(self, session_id: str) -> bool:
        """
        Deletes a session by its ID.

        :param session_id: The ID of the session.
        :return: True if the session was deleted, False otherwise
        """
        with self.session_scope() as session:
            session_record = (
                session.query(ObserverSessionRecord)
                .filter_by(session_id=session_id)
                .first()
            )
            if session_record:
                session.delete(session_record)
                # Delete all interactions, tasks, and messages associated with the session
                # to maintain referential integrity - current schema does not have foreign keys
                session.query(ObserverInteractionRecord).filter_by(
                    session_id=session_id
                ).delete()
                session.query(ObserverTaskRecord).filter_by(
                    session_id=session_id
                ).delete()
                session.query(ObserverMessageRecord).filter_by(
                    session_id=session_id
                ).delete()

                return True
            return False

    def get_session_by_id(self, session_id: str) -> ObserverSessionRecord:
        """
        Retrieves a session by its ID.

        :param session_id: The ID of the session.
        :return: The session record.
        """
        with self.session_scope() as session:
            return (
                session.query(ObserverSessionRecord)
                .filter_by(session_id=session_id)
                .first()
            )

    def upsert_session(self, session_data: ObserverSession) -> None:
        """
        Inserts a new session if it doesn't exist, or updates an existing session in
        the ObserverSessionRecord table.

        :param session_data: An ObserverSession pydantic model containing session details.
        """
        with self.session_scope() as session:
            # Try to find the existing session by session_id
            existing_session = (
                session.query(ObserverSessionRecord)
                .filter_by(session_id=session_data.session_id)
                .first()
            )

            if existing_session:
                # If the session exists, update its fields
                existing_session.first_message = session_data.first_message
            else:
                # If the session doesn't exist, create a new one
                new_session = ObserverSessionRecord(
                    observer_id=session_data.observer_id,
                    session_id=session_data.session_id,
                    first_message=session_data.first_message,
                    workflow_id=session_data.workflow_id,
                )
                session.add(new_session)

    def add_interaction(self, interaction_data: ObserverInteraction) -> None:
        """
        Adds a new interaction to the ObserverInteractionRecord table.

        :param interaction_data: A dictionary containing interaction details.
        """
        with self.session_scope() as session:
            new_interaction = ObserverInteractionRecord(
                observer_id=interaction_data.observer_id,
                session_id=interaction_data.session_id,
                role=interaction_data.role,
                message=interaction_data.message,
            )
            session.add(new_interaction)

    def add_task(self, task_data: ObserverTask) -> None:
        """
        Adds a new task to the ObserverTaskRecord table.

        :param task_data: An ObserverTask pydantic model containing task details.
        """
        with self.session_scope() as session:
            new_task = ObserverTaskRecord(
                observer_id=task_data.observer_id,
                session_id=task_data.session_id,
                task_id=task_data.task_id,
                description=task_data.description,
                status=task_data.status,
                result=task_data.result,
                dependencies=task_data.dependencies,
                context=task_data.context,
                origin=task_data.origin,
            )
            session.add(new_task)

    def upsert_task(self, task_data: ObserverTask) -> None:
        """
        Inserts a new task if it doesn't exist, or updates an existing task in
        the ObserverTaskRecord table.

        :param task_data: An ObserverTask pydantic model containing task details.
        """
        with self.session_scope() as session:
            # Try to find the existing task by task_id
            existing_task = (
                session.query(ObserverTaskRecord)
                .filter_by(task_id=task_data.task_id)
                .first()
            )

            if existing_task:
                # If the task exists, update its fields
                existing_task.description = task_data.description
                existing_task.status = task_data.status
                existing_task.result = task_data.result
                existing_task.dependencies = task_data.dependencies
                existing_task.context = task_data.context
            else:
                # If the task doesn't exist, create a new one
                new_task = ObserverTaskRecord(
                    observer_id=task_data.observer_id,
                    session_id=task_data.session_id,
                    task_id=task_data.task_id,
                    description=task_data.description,
                    status=task_data.status,
                    result=task_data.result,
                    dependencies=task_data.dependencies,
                    context=task_data.context,
                    origin=task_data.origin,
                )
                session.add(new_task)

    def add_message(self, message_data: ObserverMessage) -> None:
        """
        Adds a new message to the ObserverMessageRecord table.

        :param message_data: An ObserverMessage pydantic model containing message details.
        """
        with self.session_scope() as session:
            new_message = ObserverMessageRecord(
                observer_id=message_data.observer_id,
                session_id=message_data.session_id,
                role=message_data.role,
                text=message_data.text,
                agent=message_data.agent,
                task=message_data.task,
            )
            session.add(new_message)

    def get_task_by_id(self, task_id: str) -> ObserverTaskRecord:
        """
        Retrieves a task by its ID.

        :param task_id: The ID of the task.
        :return: The task record.
        """
        with self.session_scope() as session:
            return session.query(ObserverTaskRecord).filter_by(task_id=task_id).first()

    def get_message_by_id(self, message_id: str) -> ObserverMessageRecord:
        """
        Retrieves a message by its ID.

        :param message_id: The ID of the message.
        :return: The message record.
        """
        with self.session_scope() as session:
            return session.query(ObserverMessageRecord).filter_by(id=message_id).first()

    def update_task_status(self, task_id: str, new_status: Status) -> None:
        """
        Updates the status of a task.

        :param task_id: The ID of the task.
        :param new_status: The new status to set.
        """
        with self.session_scope() as session:
            task = session.query(ObserverTaskRecord).filter_by(task_id=task_id).first()
            if task:
                task.status = new_status

    def delete_task(self, task_id: str) -> None:
        """
        Deletes a task by its ID.

        :param task_id: The ID of the task.
        """
        with self.session_scope() as session:
            task = session.query(ObserverTaskRecord).filter_by(task_id=task_id).first()
            if task:
                session.delete(task)

    def delete_message(self, message_id: str) -> None:
        """
        Deletes a message by its ID.

        :param message_id: The ID of the message.
        """
        with self.session_scope() as session:
            message = (
                session.query(ObserverMessageRecord).filter_by(id=message_id).first()
            )
            if message:
                session.delete(message)

    def add_workflow(self, workflow_data: ObserverWorkflow) -> None:
        """
        Adds a new workflow to the ObserverWorkflowRecord table.

        :param workflow_data: An ObserverWorkflow pydantic model containing workflow details.
        """
        with self.session_scope() as session:
            new_workflow = ObserverWorkflowRecord(
                workflow_id=workflow_data.workflow_id,
                workflow_name=workflow_data.workflow_name,
                workflow=workflow_data.workflow,
            )
            session.add(new_workflow)

    def get_workflow_by_id(self, workflow_id: str) -> ObserverWorkflowRecord:
        """
        Retrieves a workflow by its ID.

        :param workflow_id: The ID of the workflow.
        :return: The workflow record.
        """
        with self.session_scope() as session:
            return (
                session.query(ObserverWorkflowRecord)
                .filter_by(workflow_id=workflow_id)
                .first()
            )

    def delete_workflow(self, workflow_id: str) -> None:
        """
        Deletes a workflow by its ID.

        :param workflow_id: The ID of the workflow.
        """
        with self.session_scope() as session:
            workflow = (
                session.query(ObserverWorkflowRecord)
                .filter_by(workflow_id=workflow_id)
                .first()
            )
            if workflow:
                session.delete(workflow)

    def get_all_workflows(self) -> List[ObserverWorkflowRecord]:
        """
        Retrieves all workflows from the ObserverWorkflowRecord table.

        :return: A list of all workflow records.
        """
        with self.session_scope() as session:
            return session.query(ObserverWorkflowRecord).options(joinedload("*")).all()

    def upsert_workflow(self, workflow_data: ObserverWorkflow) -> None:
        """
        Inserts a new workflow if it doesn't exist, or updates an existing workflow in
        the ObserverWorkflowRecord table.

        :param workflow_data: An ObserverWorkflow pydantic model containing workflow details.
        """
        with self.session_scope() as session:
            # Try to find the existing workflow by workflow_id
            existing_workflow = (
                session.query(ObserverWorkflowRecord)
                .filter_by(workflow_id=workflow_data.workflow_id)
                .first()
            )

            if existing_workflow:
                # If the workflow exists, update its fields
                existing_workflow.workflow_name = workflow_data.workflow_name
                existing_workflow.workflow = workflow_data.workflow
            else:
                # If the workflow doesn't exist, create a new one
                new_workflow = ObserverWorkflowRecord(
                    workflow_id=workflow_data.workflow_id,
                    workflow_name=workflow_data.workflow_name,
                    workflow=workflow_data.workflow,
                )
                session.add(new_workflow)

    def add_tool_call(self, tool_call_data: ObserverToolCallRecord) -> None:
        """
        Adds a new tool call to the ObserverToolCallRecord table.

        :param tool_call_data: An ObserverToolCallRecord pydantic model containing tool call details.
        """
        with self.session_scope() as session:
            new_tool_call = ObserverToolCallRecord(
                observer_id=tool_call_data.observer_id,
                session_id=tool_call_data.session_id,
                tool_name=tool_call_data.tool_name,
                parameters=tool_call_data.parameters,
                result=tool_call_data.result,
            )
            session.add(new_tool_call)
