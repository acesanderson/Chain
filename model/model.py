from Chain.cache.cache import ChainCache, check_cache_and_query
from Chain.message.message import Message
from Chain.parser.parser import Parser
from Chain.progress.wrappers import progress_display
from Chain.model.params.params import Params
from Chain.model.models.models import ModelStore
from pydantic import BaseModel
from typing import Optional, TYPE_CHECKING, Literal
from pathlib import Path
import importlib

dir_path = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from rich.console import Console
    from openai import Stream  # For type hinting only, to avoid circular imports
    from anthropic import (
        Stream as AnthropicStream,
    )  # For type hinting only, to avoid circular imports


class Model:
    # Class singletons
    _clients = {}  # Store lazy-loaded client instances at the class level
    _chain_cache: Optional[ChainCache] = (
        None  # If you want to add a cache, add it at class level as a singleton.
    )
    _console: Optional["Console"] = (
        None  # For rich console output, if needed. This is overridden in the Chain class.
    )
    _debug: bool = False  # If True, you can see contents of requests and responses

    # Object methods
    def __init__(self, model: str = "gpt-4o", console: Optional["Console"] = None):
        self.model = ModelStore._validate_model(model)
        self._client_type = self._get_client_type(self.model)
        self._client = self.__class__._get_client(self._client_type)
        self._console = console

    @property
    def console(self):
        """
        Returns the effective console (hierarchy: instance -> Chain/AsyncChain class -> None)
        """
        if self._console:
            return self._console

        import sys

        # Check for Chain._console
        if "Chain.chain.chain" in sys.modules:
            Chain = sys.modules["Chain.chain.chain"].Chain
            chain_console = getattr(Chain, "_console", None)
            if chain_console:
                return chain_console

        # Check for AsyncChain._console
        if "Chain.chain.asyncchain" in sys.modules:
            AsyncChain = sys.modules["Chain.chain.asyncchain"].AsyncChain
            async_console = getattr(AsyncChain, "_console", None)
            if async_console:
                return async_console

        return None

    @console.setter
    def console(self, console: "Console"):
        """
        Sets the console object for rich output.
        This is useful if you want to override the default console for a specific instance.
        """
        self._console = console

    def _get_client_type(self, model: str) -> tuple:
        """
        Setting client_type for Model object is necessary for loading the correct client in the query functions.
        Returns a tuple with client type (which informs the module title) and the client class name (which is used to instantiate the client).
        """
        model_list = ModelStore.models()
        if model in model_list["openai"]:
            return "openai", "OpenAIClientSync"
        elif model in model_list["anthropic"]:
            return "anthropic", "AnthropicClientSync"
        elif model in model_list["google"]:
            return "google", "GoogleClientSync"
        elif model in model_list["ollama"]:
            return "ollama", "OllamaClientSync"
        elif model in model_list["groq"]:
            return "groq", "GroqClientSync"
        elif model in model_list["deepseek"]:
            return "deepseek", "DeepSeekClient"
        elif model in model_list["perplexity"]:
            return "perplexity", "PerplexityClientSync"
        else:
            raise ValueError(f"Model {model} not found in models")

    @classmethod
    def _get_client(cls, client_type: tuple):
        # print(f"client type: {client_type}")
        if client_type[0] not in cls._clients:
            try:
                module = importlib.import_module(
                    f"Chain.model.clients.{client_type[0].lower()}_client"
                )
                client_class = getattr(module, f"{client_type[1]}")
                cls._clients[client_type[0]] = client_class()
            except ImportError as e:
                raise ImportError(f"Failed to import {client_type} client: {str(e)}")
        client_object = cls._clients[client_type[0]]
        if not client_object:
            raise ValueError(f"Client {client_type} not found in clients")
        return client_object

    @progress_display
    def query(
        self,
        query_input: str | list | Message | None = None,
        parser: Parser | None = None,
        cache=True,
        temperature: Optional[float] = None,
        stream: bool = False,
        # For progress reporting decorator
        verbose: bool = True,
        index: int = 0,
        total: int = 0,
        # Options for debugging
        params: Optional[Params] = None,
        return_params: bool = False,
    ) -> "BaseModel | str | Stream | AnthropicStream":
        """
        Execute a query against the language model with optional progress tracking.

        Args:
            input: The query text or list of messages to send to the model
            parser: Optional parser to structure the response
            cache: Whether to use response caching (default: True)
            verbose: Whether to display progress information (default: True)
            index: Current item number for batch progress display (requires total)
            total: Total number of items for batch progress display (requires index)
            temperature: Optional temperature setting for the model (default: None)
            stream: Whether to stream the response (default: False)
            params: Optional Params object to override default parameters
            return_params: If True, returns the Params object instead of the response

        Returns:
            str: The model's response, optionally parsed if parser provided

        Raises:
            ValueError: If only one of index/total is provided

        Examples:
            # Basic usage
            response = model.query("What is 2+2?")

            # Batch processing with progress
            for i, item in enumerate(items):
                response = model.query(item, index=i+1, total=len(items))
                # Shows: ⠋ gpt-4o | [1/100] Processing item...

            # Suppress progress
            response = model.query("What is 2+2?", verbose=False)
        """
        # Construct Params object if not provided (majority of cases)
        if not params:
            # Here's the magic -- kwargs goes right into our Params object.
            import inspect

            frame = inspect.currentframe()
            args, _, _, values = inspect.getargvalues(frame)

            query_args = {k: values[k] for k in args if k != "self"}
            query_args["model"] = self.model
            cache = query_args.pop("cache", False)
            params = Params(**query_args)
        # We should have a Params object now, either provided or constructed.
        assert isinstance(
            params, Params
        ), f"params must be an instance of Params or None, got {type(params)}"
        # For debug, return params if requested
        if return_params:
            return params
        # If self._debug == True, print the params
        if self._debug == True:
            print(params.model_dump_json())
        # Caching
        if cache and self._chain_cache:
            return check_cache_and_query(
                self, params, lambda: self._client.query(params)
            )
        else:
            return self._client.query(params)

    def tokenize(self, text: str) -> int:
        """
        Get the token length for the given model.
        Implementation at the client level.
        """
        return self._client.tokenize(model=self.model, text=text)

    def pretty(self, user_input):
        pretty = user_input.replace("\n", " ").replace("\t", " ").strip()
        return pretty[:60] + "..." if len(pretty) > 60 else pretty

    def __repr__(self):
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"
