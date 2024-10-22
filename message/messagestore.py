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

"History" vs. "Log":
    - The history is a hardcode list of messages in json format.
    - The log is a file that is automatically updated with the messages, and is formatted for human readability.
    - History is invoked by the user.
    - Log is automatically updated with the messages and therefore a flag for several methods.
"""

from Chain.message.message import Message
from rich.console import Console
from rich.rule import Rule
from pydantic import BaseModel
import os
import json


class MessageStore:
    """
    Defines a message store object.
    """

    def __init__(
        self,
        console: Console = [],
        history_file: str = "",
        log_file: str = "",
        pruning: bool = False,
    ):
        """
        Initializes the message store, and loads the history from a file.
        """
        # Use existing console or create a new one
        if not console:
            self.console = Console(width=100)
        else:
            self.console = console
        self.messages = []  # The list of messages
        # Config history and log if requested
        if history_file:
            self.history_file = history_file
            self.persistent = True
        else:
            self.history_file = ""
            self.persistent = False
        if log_file:
            self.log_file = log_file
            self.logging = True
        else:
            self.log_file = ""
            self.logging = False
        # Set the prune flag
        self.pruning = pruning

    def write_to_log(self, item: "str | BaseModel") -> None:
        """
        Writes a log to the log file.
        Need to handle individual strings (i.e. prompts and text completions) as well as pydantic objects.
        """
        if isinstance(item, str):
            # We're using a Rich console to write the log to the file with colors
            with open(self.log_file, "w") as file:
                file_console = Console(
                    file=file, force_terminal=True
                )  # force_terminal = treat file as terminal
                # Write the formatted text to the file (won't be printed to terminal)
                file_console.print(f"[bold magenta]{item}[/bold magenta]\n")
        if isinstance(item, Message):
            with open(self.log_file, "w") as file:
                file_console = Console(file=file, force_terminal=True)
                file_console.print(Rule(title="Message", style="bold green"))
                file_console.print(f"[bold cyan]{item.role}:[/bold cyan]")
                if item.role == "user":
                    file_console.print(f"[yellow]{item.content}[/yellow]\n")
                elif item.role == "assistant":
                    file_console.print(f"[blue]{item.content}[/blue]\n")
                elif item.role == "system":
                    file_console.print(f"[green]{item.content}[/green]\n")
                else:
                    file_console.print(f"[white]{item.content}[/white]\n")

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
            if self.pruning:
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

    def prune(self):
        """
        Prunes the history to the last 20 messages.
        """
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

    def add(self, message: "Message | list[Message]"):
        """
        Add an existing message object to messages.
        """
        if isinstance(message, Message):
            self.messages.append(message)
            if self.logging:
                self.write_to_log(message)
        elif isinstance(message, list):
            self.messages.extend(message)
            if self.logging:
                for msg in message:
                    self.write_to_log(msg)
        if self.persistent:
            self.save()

    def add_new(self, role: str, content: str):
        """
        Adds a message to the history, constructed from role and content vars.
        """
        self.messages.append(Message(role=role, content=content))
        if self.persistent:
            self.save()
        if self.logging:
            self.write_to_log(self.messages[-1])

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

    def __bool__(self):
        """
        We want this to return True if the object is initialized.
        We wouldn't need this if we didn't also want a __len__ method.
        (If __len__ returns 0, bool() returns False for your average object).
        """
        return True

    def __len__(self):
        """
        Note: see the __bool__ method above for extra context.
        """
        return len(self.messages)

    def __repr__(self) -> str:
        """
        Standard for all of my classes; changes how the object is represented when invoked in interpreter.
        """
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"
