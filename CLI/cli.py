"""
Considerations:
- purpose of this is to abstract and simplify what I have in all my chat CLI apps (ask, tutorialize, leviathan, twig, cookbook)
- need to figure out how users can override the query class
- allow for grabbing stdin (so apps can be piped into)
- consider abstracting out the basic Chain functionality as a mixin
- consider making a Command class instead of using functions like objects -- it's currently a little too clever and likely unreadable. Extensibility can be done by making Command classes. Maybe.

Usage:
- In its base form, the CLI class can be used to create a simple chat application, using Claude.
- You can extend it by adding your own arguments and functions.
- To add a function, create a method with the prefix "arg_" and decorate it with the cli_arg decorator. The abbreviation for the argument should be passed as an argument to the decorator (like @cli_arg("-m")).
"""

from Chain import MessageStore, Model, Prompt, Chain
import argparse
from rich.console import Console
from utils import print_markdown
from inspect import signature
from typing import Callable
import sys


def cli_arg(abbreviation):
    """
    Decorator for adding arguments to the CLI.
    This is used to define the abbreviation for the argument.
    """

    def decorator(func):
        func.abbreviation = abbreviation
        return func

    return decorator


class CLI:
    """
    CLI class for Chain module: use this to create command line-based chat applications.
    This has a base set of parsers and commands that can be extended by the user.
    """

    def __init__(
        self,
        name: str,
        history_file: str = "",
        log_file: str = "",
        pruning: bool = False,
    ):
        """
        Initialize the CLI object with a name and optional history and log files.
        """
        self.name = name
        self.preferred_model = Model("claude")
        self.console = Console(width=120)
        self.messagestore = MessageStore(self.console, history_file, log_file, pruning)
        self.catalog = {}
        self.parser = self.init_parser()
        self.raw = False

    def init_parser(self):
        """
        Initialize the parser with all our arguments.
        """
        _parser = argparse.ArgumentParser(description=self.name)

        def catalog_arg(arg_func: Callable):
            """
            Take an argument function and add it to the parser.
            """
            arg_name = arg_func.__name__[4:]
            arg_doc = arg_func.__doc__
            # Capture the default arg
            if len(signature(arg_func).parameters) == 1 and arg_func.abbreviation == "":  # type: ignore
                _parser.add_argument(arg_name, nargs="*", help=arg_doc)

            # Transitive functions
            elif len(signature(arg_func).parameters) == 1:
                arg_abbreviation = arg_func.abbreviation  # type: ignore
                _parser.add_argument(
                    arg_abbreviation, type=str, nargs="?", dest=arg_name, help=arg_doc
                )

            # Next, intransitive functions
            elif len(signature(arg_func).parameters) == 0:
                arg_abbreviation = arg_func.abbreviation  # type: ignore
                _parser.add_argument(
                    arg_abbreviation, action="store_true", dest=arg_name, help=arg_doc
                )
            return arg_name

        methods = [method for method in dir(self) if method.startswith("arg_")]
        for method in methods:
            method_name = catalog_arg(getattr(self, method))
            self.catalog[method_name] = getattr(self, method)
        return _parser

    def run(self):
        args = self.parser.parse_args()
        args = vars(args)
        for arg in args:
            if str(args[arg]) in ["True", "False"]:
                if args[arg]:
                    self.catalog[arg]()
            elif args[arg] != None and args[arg] != []:
                self.catalog[arg](args[arg])
        sys.exit()

    # Our arg methods
    @cli_arg("")
    def arg_query(self, param):
        """
        Send a message.
        """
        # This is the default argument. Should suppress this if inherited class has its own "" argument.
        query = " ".join(param)
        self.messagestore.add_new(role="user", content=query)
        prompt = Prompt(query)
        chain = Chain(prompt=prompt, model=self.preferred_model)
        response = chain.run()
        if response.content:
            self.messagestore.add_new(role="assistant", content=str(response.content))
            if self.raw:
                print(response)
            else:
                print_markdown(str(response.content), console=self.console)
        else:
            raise ValueError("No response found.")

    @cli_arg("-hi")
    def arg_history(self):
        """
        Print the last 10 messages.
        """
        self.messagestore.view_history()

    @cli_arg("-l")
    def arg_last(self):
        """
        Print the last message.
        """
        last_message = self.messagestore.last()
        if last_message:
            if self.raw:
                print(last_message.content)
            else:
                print_markdown(str(last_message.content), console=self.console)
        else:
            self.console.print("No messages yet.")

    @cli_arg("-g")
    def arg_get(self, param):
        """
        Get a specific answer from the history.
        """
        if not param.isdigit():
            self.console.print("Please enter a valid number.")
            return
        retrieved_message = self.messagestore.get(int(param))
        if retrieved_message:
            try:
                if self.raw:
                    print(retrieved_message.content)
                else:
                    print_markdown(str(retrieved_message.content), console=self.console)
            except ValueError:
                self.console.print("Message not found.")
        else:
            self.console.print("Message not found.")

    @cli_arg("-c")
    def arg_clear(self):
        """
        Clear the message history.
        """
        self.messagestore.clear()
        self.console.print("Message history cleared.")

    @cli_arg("-m")
    def arg_model(self, param):
        """
        Specify a model.
        """
        self.preferred_model = Model(param)
        self.console.print(f"Model set to {param}.")

    @cli_arg("-r")
    def arg_raw(self):
        """
        Print raw output.
        """
        self.raw = True


if __name__ == "__main__":
    c = CLI(name="Chain Chat", history_file=".cli_history.log")
    c.run()
