"""
A Chain is a convenience wrapper for models, prompts, parsers, messages, and response objects.
A chain needs to have at least a prompt and a model.
Chains are immutable, treat them like tuples.
"""

import time  # for timing our query calls (saved in Response object)

# The rest of our package.
from Chain.prompt.prompt import Prompt
from Chain.model.model import Model
from Chain.result.response import Response
from Chain.parser.parser import Parser
from Chain.message.message import Message
from Chain.message.messagestore import MessageStore
from Chain.message.imagemessage import ImageMessage
from Chain.message.audiomessage import AudioMessage
from typing import TYPE_CHECKING, Optional

# Our TYPE_CHECKING imports, these ONLY load for IDEs, so you can still lazy load in production.
if TYPE_CHECKING:
    from rich.console import Console


class Chain:
    """
    How we chain things together.
    Instantiate with:
    - a prompt (a string that is ready for jinja2 formatting),
    - a model (a name of a model (full list of accessible models in Model.models))
    - a parser (a function that takes a string and returns a string)
    """

    # If you want logging, initialize a message store with log_file_path parameter, and assign it to your Chain class as a singleton.
    _message_store: Optional[MessageStore] = None
    # If you want rich progress reporting, add a rich.console.Console object to Chain. (also can be added at Model level)
    _console: Optional["Console"] = None

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

    def run(
        self,
        input_variables: dict | None = None,
        messages: list[Message] | None = [],
        verbose: bool = True,
        stream: bool = False,
        cache: bool = True,
        index: int = 0,
        total: int = 0,
    ) -> Response:
        """
        Executes the Chain, processing the prompt and interacting with the language model.

        This method acts as a central dispatcher, routing the request based on
        whether 'messages' are provided or if a streaming response is requested.
        It renders the prompt with `input_variables` if a `Prompt` object is
        associated with the Chain.

        Args:
            input_variables (dict | None): A dictionary of variables to render
                the prompt template. Required if the Chain's prompt contains
                Jinja2 placeholders. Defaults to None.
            messages (list[Message] | None): A list of `Message` objects
                representing a conversation history or a single message. If
                provided, the Chain will operate in chat mode. Defaults to an
                empty list.
            verbose (bool): If True, displays progress information during the
                model query. This is managed by the `progress_display` decorator
                on the underlying `Model.query` call. Defaults to True.
            stream (bool): If True, attempts to stream the response from the
                model. Note that streaming requests do not return a `Response`
                object directly but rather a generator. Defaults to False.
            cache (bool): If True, the response will be cached if caching is
                enabled on the `Model` class. Defaults to True.
            index (int): The current index of the item being processed in a
                batch operation. Used for progress display (e.g., "[1/100]").
                Requires `total` to be provided. Defaults to 0.
            total (int): The total number of items in a batch operation. Used
                for progress display (e.g., "[1/100]"). Requires `index` to be
                provided. Defaults to 0.

        Returns:
            Response: A `Response` object containing the model's output, status,
            duration, and associated messages. Returns a generator if `stream`
            is True.

        Raises:
            ValueError: If neither a prompt nor messages are provided.
            ValueError: If `index` is provided without `total`, or vice-versa.
        """
        # Render our prompt with the input_variables if variables are passed. Should throw a jinja error if it doesn't match.
        if input_variables and self.prompt:
            prompt = self.prompt.render(input_variables=input_variables)
        elif self.prompt:
            prompt = self.prompt.prompt_string
        else:
            prompt = None
        # Route a stream request (these don't return Response objects)
        # Route input; if string, if message
        if messages:
            result = self.run_messages(
                prompt=prompt,
                messages=messages,
                verbose=verbose,
                cache=cache,
                index=index,
                total=total,
                stream = stream,
            )
        elif prompt:
            result = self.run_completion(
                prompt=prompt,
                verbose=verbose,
                cache=cache,
                index=index,
                total=total,
                stream = stream,
            )
        else:
            raise ValueError("No prompt or messages passed to Chain.run.")
        return result

    def run_messages(
        self,
        messages: list[Message | ImageMessage | AudioMessage],
        prompt: str | None = None,
        verbose=True,
        cache=True,
        index: int = 0,
        total: int = 0,
        stream: bool = False,
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
                messages,
                verbose=verbose,
                parser=self.parser,
                cache=cache,
                index=index,
                total=total,
            )
        else:
            result = self.model.query(messages, verbose=verbose, cache=cache)
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

    # In chain/chain.py - update run_completion method
    def run_completion(
        self,
        prompt: str,
        verbose=True,
        stream=False,
        cache=True,
        index: int = 0,
        total: int = 0,
    ):
        """Updated to properly handle streaming responses"""
        time_start = time.time()
        user_message = Message(role="user", content=prompt)
        
        if Chain._message_store:
            Chain._message_store.add(user_message)
        
        if stream:
            # For streaming, return the stream object directly
            if self.parser:
                # Streaming with structured output is complex - disable for now
                raise ValueError("Streaming is not supported with parsers yet")
            
            stream_response = self.model.query(
                prompt,
                verbose=verbose,
                cache=cache,
                stream=True
            )
            return stream_response  # Return raw stream object
        else:
            # Non-streaming path (existing logic)
            if self.parser:
                result = self.model.query(
                    prompt,
                    verbose=verbose,
                    parser=self.parser,
                    cache=cache,
                    index=index,
                    total=total,
                )
            else:
                result = self.model.query(prompt, verbose=verbose, cache=cache)
            
            time_end = time.time()
            duration = time_end - time_start
            
            assistant_message = Message(role="assistant", content=result)
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
