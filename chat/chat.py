"""
Extensible chat class for CLI chat applications.
Add more commands by extending the commands list + defining a "command_" method.
"""

from Chain.model.model import Model
from Chain.message.messagestore import MessageStore, Message
from rich.console import Console
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
        self.command_regex = re.compile("/([^ ]+)( |$)")
        self.command_show_regex = re.compile("/show ([^ ]+)")
        self.command_set_regex = re.compile("/set ([^ ]+) ([^ ]+)")
        self.messagestore = None  # This will be initialized in the chat method.
        self.welcome_message = "[green]Hello! Type /exit to exit.[/green]"
        # Command registry. To extend in a subclass, add to this + add command_ method.
        self.commands = {
            "exit": self.command_exit,
            "help": self.command_help,
            "clear": self.command_clear,
            "show": {
                "model": self.command_show_model,
                "history": self.command_show_history,
            },
            "set": {
                "model": self.command_set_model,
            },
        }

    def parse_input(self, input: str):
        """
        Commands start with a slash. This method parses the input and returns the corresponding method.
        There are three levels of commands:
        1. Base commands (e.g. /exit)
        2. Show commands (e.g. /show model)
        3. Set commands (e.g. /set model gpt)
        """
        # Not a command; return None
        if not input.startswith("/"):
            return None

        # Subtype: show. Has a base command ("show") and a sub command (e.g. "model").
        if input.startswith("/show"):
            show_match = re.search(self.command_show_regex, input)
            if show_match:
                show_command = show_match.group(1)
                if show_command in self.commands["show"].keys():  # type: ignore
                    return self.commands["show"][show_command]  # type: ignore
                else:
                    self.console.print("Invalid show command.", style="red")
                    return None
            else:
                raise ValueError("Regex error.")

        # Subtype: set. Has a base command ("set"), a sub command (e.g. "model"), and a parameter.
        # Set commands should always declare their parameter as "param" in the method signature, as we are assembling partial functions here.
        elif input.startswith("/set"):
            set_match = re.search(self.command_set_regex, input)
            if set_match:
                set_command = set_match.group(1)
                set_parameter = set_match.group(2)
                bare_func = self.commands["set"][set_command]  # type: ignore
                return partial(bare_func, set_parameter)
            else:
                raise ValueError("Regex error.")

        # Base commands
        else:
            command_match = re.search(self.command_regex, input)
            if command_match:
                command = command_match.group(1)
                if command in self.commands.keys():
                    return self.commands[command]  # type: ignore
                else:
                    return None
            else:
                raise ValueError("Regex error.")

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
        help_message = "Commands:\n"
        for command in self.commands.keys():
            if isinstance(self.commands[command], dict):
                for subcommand in self.commands[command].keys():
                    help_message += f"/[purple]{command} {subcommand}[/purple]: [green]{self.commands[command][subcommand].__doc__.strip()}[/green]\n"
            else:
                help_message += f"/[purple]{command}[/purple]: [green]{self.commands[command].__doc__.strip()}[/green]\n"
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
                    self.console.print(str(response) + "\n", style="blue")
                    continue
        except KeyboardInterrupt:
            self.console.print("\nGoodbye!", style="green")


def main():
    c = Chat(Model("gpt"))
    c.chat()


if __name__ == "__main__":
    main()
