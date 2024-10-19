"""
A MessageStore object is a class that stores messages in a list and provides methods to interact with the list.
With MessageStore, you can:
- implement a log for tracing chains (logging mode: openfile with active updates, you can tail -f the file)
- view the history of messages
- dequeue the last message per a history size
- clear the history
- add a message to the history
- get a message from the history
- automatically convert pydantic objects to readable strings

Under the hood, a messagestore is a list that may contain either Response or Message objects,
or a mix of both, but the interaction with them is as a list of messages unless otherwise specified.
"""

from message import Message
from rich.console import Console
import os
import json
from pathlib import Path

dir_path = Path(__file__).parent


class MessageStore:
    """
    Defines a message store object.
    """

    def __init__(self, console: Console = [], json_file: str = "", log_file: str = ""):
        """
        Initializes the message store, and loads the history from a file.
        """
        if not console:
            self.console = Console(width=100)
        else:
            self.console = console
        self.messages = []
        self.json_file = dir_path / json_file
        self.persistent = False
        self.log_file = dir_path / log_file
        self.logging = False
        if json_file:
            self.persistent = True
            self.load()
        if log_file:
            self.logging = True

    def load(self):
        """
        Loads the history from a file.
        """
        if not self.persistent:
            print("This message store is not persistent.")
            return
        try:
            with open(self.file_path, "rb") as file:
                self.messages = json.loads(file)
            self.prune()
        except FileNotFoundError:
            self.save()

    def save(self):
        """
        Saves the history to a file.
        """
        if not self.messages:
            return
        if not self.persistent:
            print("This message store is not persistent.")
            return
        if self.persistent:
            with open(self.file_path, "wb") as file:
                json.dumps(self.messages, file)
        if self.logging:
            with open(self.log_file, "a") as file:
                for message in self.messages:
                    file.write(f"{message.role}: {message.content}\n\n")

    def prune(self):
        """
        Prunes the history to the last 20 messages.
        """
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

    def add(self, role: str, content: str):
        """
        Adds a message to the history.
        """
        self.messages.append(Message(role=role, content=content))
        self.prune()
        self.save()

    def last(self):
        """
        Gets the last message from the history.
        """
        if self.messages:
            return self.messages[-1]

    def get(self, index: int):
        """
        Gets a message from the history.
        """
        if self.messages:
            return self.messages[index - 1]

    def delete(self):
        """
        Deletes the history file.
        """
        if self.persistent:
            try:
                os.remove(self.json_file)
            except FileNotFoundError:
                pass
        if self.logging:
            try:
                os.remove(self.log_file)
            except FileNotFoundError:
                pass

    def view_history(self):
        """
        Pretty prints the history.
        """
        for index, message in enumerate(self.messages):
            content = message.content[:50].replace("\n", " ")
            self.console.print(
                f"[green]{index+1}.[/green] [bold white]{message.role}:[/bold white] [yellow]{content}[/yellow]"
            )

    def clear(self):
        """
        Clears the history.
        """
        self.messages = []
        if self.persistent:
            self.save()

    def clear_logs(self):
        """
        Clears the logs.
        """
        if self.logging:
            with open(self.log_file, "w") as file:
                # clear the file
                pass

    def __getitem__(self, index: int):
        """
        MessageStore can act as a list.
        """
        return self.messages[index]

    def __len__(self):
        return len(self.messages)

    def __repr__(self):
        return repr(self.messages)
