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
from typing import Callable


class Chat:
    """
    Basic CLI chat implementation.
    """

    def __init__(self, model: Model):
        self.model = model
        self.console = Console(width=100)
        self.regex_command_no_params = re.compile("/([^ ]+)( |$)")
        self.regex_command_one_param = re.compile("/([^ ]+) ([^ ]+)")
        self.regex_command_two_params = re.compile("/([^ ]+) ([^ ]+) (.+)")
        self.messagestore = None  # This will be initialized in the chat method.
        self.welcome_message = "[green]Hello! Type /exit to exit.[/green]"
        # Command syntax -- can be extended
        ## Commands that have one parameter:
        self.one_param_commands = ["show"]
        ## Commands that have two parameters:
        self.two_param_commands = ["set"]

    def parse_input(self, input: str) -> Callable | partial | None:
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

        # Subtype: transitive commands. Has a base command (like "show") and a sub command (e.g. "model").
        if any(
            [input.startswith("/" + command) for command in self.one_param_commands]
        ):
            match = re.search(self.regex_command_one_param, input)
            if match:
                command = "command_" + match.group(1) + "_" + match.group(2)
                if command in commands:
                    return getattr(self, command)
                else:
                    return None
            else:
                raise ValueError("Regex error.")

        # Subtype: ditransitive commands (with two parameters). Has a base command ("set"), a sub command (e.g. "model"), and a parameter.
        # These commands should always declare their parameter as "param" in the method signature, as we are assembling partial functions here.
        elif any(
            [input.startswith("/" + command) for command in self.two_param_commands]
        ):
            match = re.search(self.regex_command_two_params, input)
            if match:
                command = "command_" + match.group(1) + "_" + match.group(2)
                parameter = match.group(3)
                if command in commands:
                    bare_func = getattr(self, command)
                    return partial(bare_func, parameter)
                else:
                    return None
            else:
                raise ValueError("Regex error.")

        # Base commands -- intransitive
        else:
            command_match = re.search(self.regex_command_no_params, input)
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

    def command_show_models(self):
        """
        Display available models.
        """
        self.console.print(Model.models, style="green")

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
