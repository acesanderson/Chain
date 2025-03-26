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

from Chain import Chain, Model, MessageStore, Message
from rich.console import Console
from rich.markdown import Markdown
from instructor.exceptions import InstructorRetryException
from functools import partial
from typing import Callable
import sys
import inspect
from pathlib import Path


class Chat:
    """
    Basic CLI chat implementation.
    """

    def __init__(self, model: Model):
        self.model = model
        self.console = Console(width=100)
        self.messagestore = None  # This will be initialized in the chat method.
        self.welcome_message = "[green]Hello! Type /exit to exit.[/green]"
        self.system_message: Message | None = None
        self.commands = self.get_commands()
        self.log_file: str | Path = ""  # Off by default, but can be initialized.

    def parse_input(self, input: str) -> Callable | partial | None:
        """
        Commands start with a slash. This method parses the input and returns the corresponding method.
        If command takes a param, this returns a partial function.
        If command is not found, it returns None (and the chat loop will handle it).
        """
        commands = self.get_commands()

        # Not a command; return None.
        if not input.startswith("/"):
            return None

        # Parse for the type of command; this also involves catching parameters.
        for command in commands:
            command_string = command.replace("command_", "").replace("_", " ")
            if input.startswith("/" + command_string):
                # Check if the command has parameters
                sig = inspect.signature(getattr(self, command))
                if sig.parameters:
                    parametrized = True
                else:
                    parametrized = False
                # Check if input has parameters
                param = input[len(command_string) + 2 :]
                # Conditional return
                if param and parametrized:
                    return partial(getattr(self, command), param)
                elif param and not parametrized:
                    raise ValueError("Command does not take parameters.")
                elif not param and parametrized:
                    raise ValueError("Command requires parameters.")
                else:
                    return getattr(self, command)
        # Command not found
        raise ValueError("Command not found.")

    def get_commands(self) -> list[str]:
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
        self.console.print("Goodbye!", style="green")
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
            try:
                command_docs = command_func.__doc__.strip()
            except AttributeError:
                print(f"Command {command_name} is missing a docstring.")
                sys.exit()
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

    def command_show_models(self):
        """
        Display available models.
        """
        self.console.print(Model.models, style="green")

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
    def query_model(self, input: list[Message]) -> str | None:
        """
        Takes either a string or a list of Message objects.
        """
        if self.messagestore:
            self.messagestore.add_new(role="user", content=str(input[-1].content))
        response = str(self.model.query(input, verbose=False))
        if self.messagestore:
            self.messagestore.add_new(role="assistant", content=str(response))
        return response

    # Main chat loop
    def chat(self):
        self.console.clear()
        self.console.print(self.welcome_message)
        Chain._message_store = MessageStore(
            console=self.console, log_file=self.log_file
        )
        self.messagestore = Chain._message_store
        if self.system_message:
            self.messagestore.add(self.system_message)
        try:
            while True:
                try:
                    user_input = self.console.input("[bold gold3]>> [/bold gold3]")
                    # Capture empty input
                    if not user_input:
                        continue
                    # Process commands
                    if user_input.startswith("/"):
                        command = self.parse_input(user_input)
                        if callable(command):
                            try:
                                command()
                                continue
                            except KeyboardInterrupt:
                                # User can cancel commands with Ctrl+C
                                self.console.print("\nCommand canceled.", style="green")
                                continue
                        else:
                            self.console.print("Invalid command.", style="red")
                            continue
                    else:
                        # Process query
                        try:
                            with self.console.status(
                                "[green]Thinking[/green]...", spinner="dots"
                            ):
                                if self.messagestore.messages:
                                    self.messagestore.add_new(
                                        role="user", content=user_input
                                    )
                                    response = self.query_model(
                                        self.messagestore.messages
                                    )
                                else:
                                    response = self.query_model(
                                        [Message(role="user", content=user_input)]
                                    )
                            self.console.print(
                                Markdown(str(response) + "\n"), style="blue"
                            )
                            continue
                        except KeyboardInterrupt:
                            # User can cancel query with Ctrl+C
                            self.console.print("\nQuery canceled.", style="green")
                            continue
                        except InstructorRetryException:
                            # This exception is raised if there is some network failure from instructor.
                            self.console.print(
                                "Network error. Please try again.", style="red"
                            )
                except ValueError as e:
                    # If command not found, or commands throw an error, catch it and continue.
                    self.console.print(str(e), style="red")
                    continue
        except KeyboardInterrupt:
            # User can exit the chat with Ctrl+C
            self.console.print("\nGoodbye!", style="green")


def main():
    c = Chat(Model("gpt"))
    c.chat()


if __name__ == "__main__":
    main()
