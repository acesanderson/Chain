"""

"""
from message import Message, Messages
from pydantic import BaseModel
from rich.console import Console
import pickle

class MessageStore():
    """
    Defines a message store object.
    """

    def __init__(self, console: Console, file_path: str):
        """
        Initializes the message store, and loads the history from a file.
        """
        self.console = console
        self.file_path = file_path
        self.messages: list[Message] = []
        self.load()
    
    def load(self):
        """
        Loads the history from a file.
        """
        try:
            with open(self.file_path, "rb") as file:
                self.messages = pickle.load(file)
            self.prune()
        except FileNotFoundError:
            self.save()
    
    def save(self):
        """
        Saves the history to a file.
        """
        with open(self.file_path, "wb") as file:
            pickle.dump(self.messages, file)
    
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
            return self.messages[index-1]

    def view_history(self):
        """
        Pretty prints the history.
        """
        for index, message in enumerate(self.messages):
            content = message.content[:50].replace("\n", " ")
            self.console.print(f"[green]{index+1}.[/green] [bold white]{message.role}:[/bold white] [yellow]{content}[/yellow]")
    
    def clear(self):
        """
        Clears the history.
        """
        self.messages = []
        self.save()
    
    def __len__(self):
        return len(self.messages)
