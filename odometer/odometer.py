"""
session odometer: in memory sqlite3 (conn = sqlite3.connect(":memory:")
conversation odomoter: saved in a sqlite3 file (similar naming convention to history store)
persistent odometer: saved in postgres
"""

from collections import DefaultDict
from Chain.odometer.TokenEvent import TokenEvent
from pathlib import Path
from typing import override
import sqlite3

class Odometer:

    # Init
    def __init__(self):
        """
        Initialize clean database.
        """
        self.db = self._load_db()
        ...

    # CRUD
    def _load_db(self) -> "db":
        """
        session odometer: in memory sqlite3 (conn = sqlite3.connect(":memory:")
        conversation odomoter: saved in a sqlite3 file (similar naming convention to history store)
        persistent odometer: saved in postgres
        """
        ...

    def _create_table(self):
        """
        CREATE TABLE token_usage (
            id SERIAL PRIMARY KEY,
            provider VARCHAR(50) NOT NULL,
            model VARCHAR(100) NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            timestamp BIGINT NOT NULL,
            host VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX idx_token_usage_timestamp ON token_usage(timestamp);
        CREATE INDEX idx_token_usage_provider_model ON token_usage(provider, model);
        """
        ...

    def _ingest_token_event(self, token_event: TokenEvent):
        """
        Use self.db to ingest TokenEvent stats.

        provider: str
        model: str
        input_tokens: int
        output_tokens: int
        timestamp: int
        host: str
        """
        ...

    # Query methods
    def stats(self):
        """
        Pretty-print the stats from the odometer.
        """
        ...

class SessionOdometer(Odometer): ...
    """
    Attaches to Model as a singleton (._session_odomoter).
    Always loads by default.
    Updates PersistentOdometer on app exit / ctrl-c / app failure.
    """
    @override
    def _load_db(self):
        """
        Use sqlite in-memory.
        """
        ...

class ConversationOdometer(Odemeter) ...
    """
    Attaches to ModelStore on instance level. (self.conversation_odometer)
    Automatically updates SessionOdometer.
    """
    @override
    def _load_db(self):
        """
        Use sqlite lite in a .db file, follow conventions from MessageStore.
        """
        ...

class PersistentOdometer(Odemeter) ...
    """
    Not attached to anything; summoned when user wants to query the time series or when app updates (on completion / exit / failure).
    """
    @override
    def _load_db(self):
        """
        Use postgres.
        """
        ...

