"""
Extensible chat class for CLI chat applications.
Add more commands by extending the commands list + defining a "command_" method.

TODO:
- more commands
    - [x] dynamic registration of new command methods
    - allow getting a message from history
    - allow branching from somewhere in history
    - allow pruning of history
    - allow saving history, saving a message
    - allow capturing screenshots for vision-based queries
- Leviathan extension
    - create a Chat app in Leviathan that incorporates ask, tutorialize, cookbook, etc.
- Mentor extension
    - system message
    - RAG: cosmo data from postgres, similarity search from chroma, direct access of courses from mongod
    - Curation objects: create, view, edit
    - blacklist course (for session)
- Learning from the Mentor implementation for more Agentic use cases
    - prompts
    - resources
    - tools
"""

from Chain.model.model import Model
from Chain.message.messagestore import MessageStore, Message
from rich.console import Console
from rich.markdown import Markdown
import re
from pydantic import BaseModel
from functools import partial


class Chat:
    """
    Basic CLI chat implementation.
    """

    def __init__(self, model: Model):
        self.model = model
        self.console = Console(width=100)
        self.regex_command = re.compile("/([^ ]+)( |$)")
        self.regex_command_show = re.compile("/show ([^ ]+)")
        self.regex_command_set = re.compile("/set ([^ ]+) ([^ ]+)")
        self.messagestore = None  # This will be initialized in the chat method.
        self.welcome_message = "[green]Hello! Type /exit to exit.[/green]"

    def parse_input(self, input: str):
        """
        Commands start with a slash. This method parses the input and returns the corresponding method.
        There are three levels of commands:
        1. Base commands (e.g. /exit)
        2. Show commands (e.g. /show model)
        3. Set commands (e.g. /set model gpt)
        """
        commands = self.get_commands()

        # Not a command; return None
        if not input.startswith("/"):
            return None

        # Subtype: show. Has a base command ("show") and a sub command (e.g. "model").
        if input.startswith("/show"):
            show_match = re.search(self.regex_command_show, input)
            if show_match:
                show_command = "command_show_" + show_match.group(1)
                if show_command in commands:
                    return getattr(self, show_command)
                else:
                    return None
            else:
                raise ValueError("Regex error.")

        # Subtype: set. Has a base command ("set"), a sub command (e.g. "model"), and a parameter.
        # Set commands should always declare their parameter as "param" in the method signature, as we are assembling partial functions here.
        elif input.startswith("/set"):
            set_match = re.search(self.regex_command_set, input)
            if set_match:
                set_command = "command_set_" + set_match.group(1)
                set_parameter = set_match.group(2)
                if set_command in commands:
                    bare_func = getattr(self, set_command)
                    return partial(bare_func, set_parameter)
                else:
                    return None
            else:
                raise ValueError("Regex error.")

        # Base commands
        else:
            command_match = re.search(self.regex_command, input)
            if command_match:
                command = "command_" + command_match.group(1)
                if command in commands:
                    return getattr(self, command)
                else:
                    return None
            else:
                raise ValueError("Regex error.")

    def get_commands(self):
        """
        Dynamic inventory of "command_" methods.
        If you extend this with more methods, make sure they follow the "command_" naming convention.
        """
        commands = [attr for attr in dir(self) if attr.startswith("command_")]
        return commands

    # Command methods
    def command_exit(self):
        """
        Exit the chat.
        """
        exit()

    def command_help(self):
        """
        Display the help message.
        """
        commands = sorted(self.get_commands())
        help_message = "Commands:\n"
        for command in commands:
            command_name = command.replace("command_", "").replace("_", " ")
            command_func = getattr(self, command)
            command_docs = command_func.__doc__.strip()
            help_message += (
                f"/[purple]{command_name}[/purple]: [green]{command_docs}[/green]\n"
            )
        self.console.print(help_message)

    def command_clear(self):
        """
        Clear the chat history.
        """
        if self.messagestore:
            self.messagestore.clear()

    def command_show_history(self):
        """
        Display the chat history.
        """
        if self.messagestore:
            self.messagestore.view_history()

    def command_show_model(self):
        """
        Display the current model.
        """
        self.console.print(f"Current model: {self.model.model}", style="green")

    def command_set_model(self, param: str):
        """
        Set the current model.
        """
        try:
            self.model = Model(param)
            self.console.print(f"Set model to {param}", style="green")
        except ValueError:
            self.console.print("Invalid model.", style="red")

    # Main query method
    def query_model(self, input: str | list[Message]) -> str | BaseModel:
        """
        Takes either a string or a list of Message objects.
        """
        return self.model.query(input, verbose=False)

    # Main chat loop
    def chat(self):
        self.console.print(self.welcome_message)
        self.messagestore = MessageStore(console=self.console)
        try:
            while True:
                user_input = input(">> ")
                if user_input.startswith("/"):
                    command = self.parse_input(user_input)
                    if callable(command):
                        command()
                        continue
                    else:
                        self.console.print("Invalid command.", style="red")
                        continue
                else:
                    self.messagestore.add_new(role="user", content=user_input)
                    with self.console.status(
                        "[green]Thinking[/green]...", spinner="dots"
                    ):
                        if self.messagestore.messages:
                            response = self.query_model(self.messagestore.messages)
                        else:
                            response = self.query_model(user_input)
                    self.messagestore.add_new(role="assistant", content=str(response))
                    self.console.print(Markdown(str(response) + "\n"), style="blue")
                    continue
        except KeyboardInterrupt:
            self.console.print("\nGoodbye!", style="green")


def main():
    c = Chat(Model("gpt"))
    c.chat()


if __name__ == "__main__":
    main()
