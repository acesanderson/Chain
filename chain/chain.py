"""
A Chain is a convenience wrapper for models, prompts, parsers, messages, and response objects.
A chain needs to have at least a prompt and a model.
Chains are immutable, treat them like tuples.
This used to be a monolith, but now I've separated out various classes, and created Message as a new one.
"""

import time  # for timing our query calls (saved in Response object)

# The rest of our package.
from Chain.prompt.prompt import Prompt
from Chain.model.model import Model
from Chain.response.response import Response
from Chain.parser.parser import Parser
from Chain.message.message import Message
from Chain.message.messagestore import MessageStore
from typing import overload


class Chain:
    """
    How we chain things together.
    Instantiate with:
    - a prompt (a string that is ready for jinja2 formatting),
    - a model (a name of a model (full list of accessible models in Model.models))
    - a parser (a function that takes a string and returns a string)
    """

    # If you want logging, initialize a message store with log_file_path parameter, and assign it to your Chain class as a singleton.
    _message_store: MessageStore | None = None

    def __init__(
        self, model: Model, prompt: Prompt | None = None, parser: Parser | None = None
    ):
        self.prompt = prompt
        self.model = model
        self.parser = parser
        if self.prompt:
            self.input_schema = self.prompt.input_schema()  # this is a set
        else:
            self.input_schema = set()

    @overload
    def run(self, input_variables: dict, verbose: bool = True) -> Response: ...

    @overload
    def run(
        self, input_variables: dict, verbose: bool, messages: list[Message]
    ) -> Response: ...

    def run(self, input_variables: dict = {}, verbose=True, messages=[]):
        """
        Input should be a dict with named variables that match the prompt.
        """
        # Render our prompt with the input_variables if variables are passed. Should throw a jinja error if it doesn't match.
        if input_variables and self.prompt:
            prompt = self.prompt.render(input_variables=input_variables)
        elif self.prompt:
            prompt = self.prompt.prompt_string
        else:
            prompt = None
        # Route input; if string, if message
        if messages:
            result = self.run_messages(
                prompt=prompt, messages=messages, verbose=verbose
            )
        elif prompt:
            result = self.run_completion(prompt=prompt, verbose=verbose)
        else:
            raise ValueError("No prompt or messages passed to Chain.run.")
        return result

    @overload
    def run_messages(self, messages: list[Message]) -> Response: ...

    @overload
    def run_messages(self, messages: list[Message], prompt: str) -> Response: ...

    @overload
    def run_messages(
        self, messages: list[Message], prompt: str, verbose=True
    ) -> Response: ...

    def run_messages(
        self, messages: list[Message], prompt: str | None = None, verbose=True
    ):
        """
        Special version of Chain.run that takes a messages object.
        Converts input + prompt into a message object, appends to messages list, and runs to Model.chat.
        Input should be a dict with named variables that match the prompt.
        """
        if prompt:
            # Add new query to messages list
            message = Message(role="user", content=prompt)
            messages.append(message)
        # If we have class-level logging
        if Chain._message_store:
            Chain._message_store.add(messages)
        # Run our query
        time_start = time.time()
        if self.parser:
            result = self.model.query(
                messages, verbose=verbose, pydantic_model=self.parser.pydantic_model
            )
        else:
            result = self.model.query(messages, verbose=verbose)
        time_end = time.time()
        duration = time_end - time_start
        # Convert result to a string
        assistant_message = Message(role="assistant", content=result)
        # If we have class-level logging
        if Chain._message_store:
            Chain._message_store.add(assistant_message)
        messages.append(assistant_message)
        # Return a response object
        response = Response(
            content=result,
            status="success",
            prompt=prompt,
            model=self.model.model,
            duration=duration,
            messages=messages,
        )
        return response

    def run_completion(self, prompt: str, verbose=True):
        """
        Standard version of Chain.run which returns a string (i.e. a completion).
        Input should be a dict with named variables that match the prompt.
        """
        time_start = time.time()
        user_message = Message(role="user", content=prompt)
        # If we have class-level logging
        if Chain._message_store:
            Chain._message_store.add(user_message)
        if self.parser:
            result = self.model.query(
                prompt, verbose=verbose, pydantic_model=self.parser.pydantic_model
            )
        else:
            result = self.model.query(prompt, verbose=verbose)
        time_end = time.time()
        duration = time_end - time_start
        # Create a new messages object, to be passed to Response object.
        assistant_message = Message(role="assistant", content=result)
        # If we have class-level logging
        if Chain._message_store:
            Chain._message_store.add(assistant_message)
        new_messages_object = [user_message, assistant_message]
        response = Response(
            content=result,
            status="success",
            prompt=prompt,
            model=self.model.model,
            duration=duration,
            messages=new_messages_object,
        )
        return response

    def __repr__(self) -> str:
        """
        Standard for all of my classes; changes how the object is represented when invoked in interpreter.
        """
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"
