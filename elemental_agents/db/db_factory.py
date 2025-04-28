"""
This module contains the database factory class that is responsible for creating
database instances based on the database name provided. The database instances
are used to abstract interaction with the database.
"""

from typing import Optional

from elemental_agents.db.db import DB
from elemental_agents.db.db_chromadb import DBChromaDB
from elemental_agents.db.db_pgvector import DBPgvector
from elemental_agents.utils.config import ConfigModel


class DBFactory:
    """
    Factory class for creating database instances with parameters from the
    string representation of the database name and connection string.
    """

    db_type: Optional[str] = None
    db_connection_string: Optional[str] = None

    def __init__(self) -> None:
        """
        Initialize the database factory.
        """
        self._config = ConfigModel()

    def create(self, db_name: Optional[str] = None) -> DB:
        """
        Create a database instance based on the database name. If the database
        name is not provided, the default database is used that is specified in
        the configuration file.

        :param db_name: The name of the database to use.
        :return: An instance of the DB class.
        """

        db_parameters = []

        if db_name is None:
            db_type = self._config.default_db_type
            db_connection_string = self._config.default_db_connection_string
        else:
            db_parameters = db_name.split("|")
            db_type = db_parameters[0]
            db_connection_string = db_parameters[1]

        # Example "chromadb|chroma.db"
        if db_type == "chromadb":
            chroma_db = DBChromaDB(scope="persistent", file_name=db_connection_string)
            return chroma_db

        # Example "pgvector|postgresql://user:password@localhost:5432/db"
        if db_type == "pgvector":
            pgvector_db = DBPgvector(connection_string=db_connection_string)
            return pgvector_db

        raise ValueError(f"Database type {db_type} is not supported.")
