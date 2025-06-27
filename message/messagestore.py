"""
A MessageStore object inherits from Messages and adds persistence and logging capabilities.
With MessageStore, you can:
- Use all Messages methods (append, extend, indexing, etc.) directly
- Automatically persist changes to TinyDB
- Log conversations in human-readable format
- Manage multiple conversation sessions
- All while being a drop-in replacement for Messages objects

The MessageStore IS a Messages object with superpowers.
"""

from Chain.message.message import Message
from Chain.message.messages import Messages
from Chain.message.imagemessage import ImageMessage
from Chain.message.audiomessage import AudioMessage
from rich.console import Console
from rich.rule import Rule
from pydantic import BaseModel
from tinydb import TinyDB, Query
from typing import Optional
from pathlib import Path
from datetime import datetime
import os


class MessageStore(Messages):
    """
    A Messages object with persistence and logging capabilities.
    Inherits all list-like behavior from Messages while adding database storage.
    """

    def __init__(
        self,
        messages: list[Message] = None,
        console: Console | None = None,
        history_file: str | Path = "",
        log_file: str | Path = "",
        pruning: bool = False,
        session_id: str | None = None,
        auto_save: bool = True,
    ):
        """
        Initialize MessageStore with optional persistence and logging.

        Args:
            messages: Initial list of messages (same as Messages class)
            console: Rich console for formatting output
            history_file: Path to TinyDB database file (.json extension recommended)
            log_file: Path to human-readable log file
            pruning: Whether to automatically prune old messages
            session_id: Optional session identifier for grouping messages
            auto_save: Whether to automatically save changes to database
        """
        # Initialize parent Messages class
        super().__init__(messages)

        # Use existing console or create a new one
        if not console:
            self.console = Console(width=100)
        else:
            self.console = console

        # Generate session ID if not provided
        self.session_id = session_id or datetime.now().isoformat()
        self.auto_save = auto_save

        # Config history database if requested
        if history_file:
            # Ensure .json extension for TinyDB
            if not str(history_file).endswith(".json"):
                history_file = str(history_file) + ".json"
            self.history_file = Path(history_file)
            self.persistent = True
            # Initialize TinyDB
            self.db = TinyDB(self.history_file)
            self.messages_table = self.db.table("messages")
        else:
            self.history_file = Path()
            self.persistent = False
            self.db = None
            self.messages_table = None

        # Config log file if requested
        if log_file:
            self.log_file = Path(log_file)
            with open(self.log_file, "a"):
                os.utime(self.log_file, None)
            self.logging = True
        else:
            self.log_file = Path()
            self.logging = False

        # Set the prune flag
        self.pruning = pruning

    def _auto_save_if_enabled(self):
        """Save to database if auto_save is enabled."""
        if self.auto_save and self.persistent:
            self.save()

    def _log_message_if_enabled(self, message: Message):
        """Log message if logging is enabled."""
        if self.logging:
            self.write_to_log(message)

    # Override Messages methods to add persistence and logging
    def append(self, message: Message) -> None:
        """Add a message to the end of the list with persistence."""
        super().append(message)
        self._log_message_if_enabled(message)
        self._auto_save_if_enabled()

    def extend(self, messages: list[Message]) -> None:
        """Extend the list with multiple messages with persistence."""
        super().extend(messages)
        for message in messages:
            self._log_message_if_enabled(message)
        self._auto_save_if_enabled()

    def insert(self, index: int, message: Message) -> None:
        """Insert a message at the specified index with persistence."""
        super().insert(index, message)
        self._log_message_if_enabled(message)
        self._auto_save_if_enabled()

    def remove(self, message: Message) -> None:
        """Remove the first occurrence of a message with persistence."""
        super().remove(message)
        self._auto_save_if_enabled()

    def pop(self, index: int = -1) -> Message:
        """Remove and return message at index with persistence."""
        message = super().pop(index)
        self._auto_save_if_enabled()
        return message

    def clear(self) -> None:
        """Remove all messages with persistence."""
        super().clear()
        self._auto_save_if_enabled()

    def __setitem__(self, key, value):
        """Set message(s) by index or slice with persistence."""
        super().__setitem__(key, value)
        if isinstance(value, Message):
            self._log_message_if_enabled(value)
        elif isinstance(value, list):
            for msg in value:
                if isinstance(msg, Message):
                    self._log_message_if_enabled(msg)
        self._auto_save_if_enabled()

    def __delitem__(self, key):
        """Delete message(s) by index or slice with persistence."""
        super().__delitem__(key)
        self._auto_save_if_enabled()

    def __iadd__(self, other) -> "MessageStore":
        """In-place concatenation with persistence."""
        result = super().__iadd__(other)
        if isinstance(other, (list, Messages)):
            for message in other:
                if isinstance(message, Message):
                    self._log_message_if_enabled(message)
        self._auto_save_if_enabled()
        return self

    # MessageStore-specific methods
    def add_new(self, role: str, content: str) -> None:
        """
        Create and add a new message (convenience method).
        """
        message = Message(role=role, content=content)
        self.append(message)  # This will handle persistence and logging

    def write_to_log(self, item: str | BaseModel) -> None:
        """
        Writes a log to the log file.
        """
        if not self.logging:
            return

        if isinstance(item, str):
            with open(self.log_file, "a", encoding="utf-8") as file:
                file_console = Console(file=file, force_terminal=True)
                file_console.print(f"[bold magenta]{item}[/bold magenta]\n")

        elif isinstance(item, Message):
            with open(self.log_file, "a", encoding="utf-8") as file:
                file_console = Console(file=file, force_terminal=True)
                file_console.print(Rule(title="Message", style="bold green"))
                file_console.print(f"[bold cyan]{item.role}:[/bold cyan]")
                try:
                    if item.role == "user":
                        file_console.print(f"[yellow]{item.content}[/yellow]\n")
                    elif item.role == "assistant":
                        file_console.print(f"[blue]{item.content}[/blue]\n")
                    elif item.role == "system":
                        file_console.print(f"[green]{item.content}[/green]\n")
                    else:
                        file_console.print(f"[white]{item.content}[/white]\n")
                except Exception as e:
                    print(f"MessageStore error: {e}")

    def save(self):
        """
        Saves the current messages to TinyDB.
        """
        if not self.persistent or not self.messages_table:
            return

        try:
            # Clear existing messages for this session
            MessageQuery = Query()
            self.messages_table.remove(MessageQuery.session_id == self.session_id)

            # Save all current messages
            for i, message in enumerate(self.messages):
                if message:  # Handle None messages
                    doc = {
                        "session_id": self.session_id,
                        "timestamp": datetime.now().isoformat(),
                        "message_index": i,
                        "message_data": message.to_cache_dict(),
                    }
                    self.messages_table.insert(doc)

        except Exception as e:
            print(f"Error saving history: {e}")

    def load(self, session_id: str | None = None):
        """
        Loads messages from TinyDB, replacing current messages.

        Args:
            session_id: If provided, loads only messages from this session.
                       If None, loads messages from current session.
        """
        if not self.persistent or not self.messages_table:
            print("This message store is not persistent.")
            return

        try:
            # Determine which session to load
            target_session = session_id or self.session_id

            # Query messages from the specified session
            MessageQuery = Query()
            message_docs = self.messages_table.search(
                MessageQuery.session_id == target_session
            )

            # Sort by timestamp to maintain order
            message_docs.sort(key=lambda x: x.get("timestamp", ""))

            # Deserialize messages
            messages_list = []
            for doc in message_docs:
                message_data = doc["message_data"]
                message_type = message_data.get("message_type", "Message")

                if message_type == "ImageMessage":
                    messages_list.append(ImageMessage.from_cache_dict(message_data))
                elif message_type == "AudioMessage":
                    messages_list.append(AudioMessage.from_cache_dict(message_data))
                else:
                    messages_list.append(Message.from_cache_dict(message_data))

            # Replace current messages (disable auto_save temporarily)
            old_auto_save = self.auto_save
            self.auto_save = False

            self.clear()
            self.extend(messages_list)

            self.auto_save = old_auto_save

            if self.pruning:
                self.prune()

        except Exception as e:
            print(f"Error loading history: {e}. Starting with empty history.")
            super().clear()  # Don't trigger auto_save for error case

    def prune(self):
        """
        Prunes the history to the last 20 messages.
        """
        if len(self) > 20:
            # Keep last 20 messages
            pruned_messages = list(self)[-20:]

            # Disable auto_save temporarily to avoid multiple saves
            old_auto_save = self.auto_save
            self.auto_save = False

            self.clear()
            self.extend(pruned_messages)

            self.auto_save = old_auto_save

            # Save the pruned version
            if self.persistent:
                self.save()

    def query_failed(self):
        """
        Removes the last message if it's a user message (handles failed queries).
        """
        if self and self.last() and self.last().role == "user":
            self.pop()  # This will auto-save
            if self.logging:
                self.write_to_log("Query failed, removing last user message.")

    def view_history(self):
        """
        Pretty prints the history.
        """
        if not self:
            self.console.print("No history (yet).", style="bold red")
            return

        for index, message in enumerate(self):
            if message:
                content = str(message.content)[:50].replace("\n", " ")
                self.console.print(
                    f"[green]{index+1}.[/green] [bold white]{message.role}:[/bold white] [yellow]{content}[/yellow]"
                )

    def clear_logs(self):
        """Clears the log file."""
        if self.logging:
            with open(self.log_file, "w") as file:
                file.write("")

    # Session management methods
    def list_sessions(self) -> list[str]:
        """Lists all available session IDs in the database."""
        if not self.persistent or not self.messages_table:
            return []

        try:
            all_docs = self.messages_table.all()
            sessions = list(set(doc["session_id"] for doc in all_docs))
            return sorted(sessions)
        except Exception as e:
            print(f"Error listing sessions: {e}")
            return []

    def delete_session(self, session_id: str):
        """Deletes all messages from a specific session."""
        if not self.persistent or not self.messages_table:
            return

        try:
            MessageQuery = Query()
            self.messages_table.remove(MessageQuery.session_id == session_id)

            # If we deleted the current session, clear current messages
            if session_id == self.session_id:
                self.clear()
        except Exception as e:
            print(f"Error deleting session: {e}")

    def switch_session(self, session_id: str):
        """
        Switch to a different session, saving current session and loading the new one.
        """
        if not self.persistent:
            print("Cannot switch sessions without persistence enabled.")
            return

        # Save current session
        self.save()

        # Switch to new session
        self.session_id = session_id

        # Load the new session
        self.load(session_id)

    def delete_database(self):
        """Deletes the TinyDB database file."""
        if self.persistent and self.history_file.exists():
            try:
                if self.db:
                    self.db.close()
                self.history_file.unlink()
                self.db = None
                self.messages_table = None
            except Exception as e:
                print(f"Error deleting database: {e}")

    # Enhanced Methods
    def get(self, index: int) -> Optional[Message]:
        """Gets a message by 1-based index (convenience method)."""
        if 1 <= index <= len(self):
            return self[index - 1]
        return None

    def copy(self) -> "MessageStore":
        """Return a copy of the MessageStore (without persistence)."""
        return MessageStore(
            messages=list(self.messages),
            console=self.console,
            # Don't copy persistence settings - new object manages its own state
        )

    def __repr__(self) -> str:
        """Enhanced representation showing persistence status."""
        persistent_info = f"persistent={self.persistent}"
        if self.persistent:
            persistent_info += f", session={self.session_id[:8]}..."
        return f"MessageStore({len(self)} messages, {persistent_info})"
