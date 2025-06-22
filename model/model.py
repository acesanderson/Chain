from Chain.cache.cache import ChainCache, CachedRequest
from Chain.message.message import Message
from Chain.message.imagemessage import ImageMessage
from Chain.message.audiomessage import AudioMessage
from Chain.parser.parser import Parser
from Chain.progress.wrappers import progress_display
from pydantic import BaseModel
from typing import Optional, TYPE_CHECKING
from pathlib import Path
import importlib, json, itertools

dir_path = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from rich.console import Console

class Model:
    # Class singletons
    _clients = {} # Store lazy-loaded client instances at the class level
    _chain_cache: Optional[ChainCache] = None # If you want to add a cache, add it at class level as a singleton.
    _console: Optional["Console"] = None  # For rich console output, if needed. This is overridden in the Chain class.


    # Class methods
    @classmethod
    def models(cls):
        """ Definitive list of models supported by Chain library. """
        with open(dir_path / "clients/models.json") as f:
            return json.load(f)

    @classmethod
    def aliases(cls):
        """ Definitive list of model aliases supported by Chain library. """
        with open(dir_path / "aliases.json") as f:
            return json.load(f)
    
    @classmethod
    def is_supported(cls, model: str) -> bool:
        """
        Check if the model is supported by the Chain library.
        Returns True if the model is supported, False otherwise.
        """
        in_aliases = model in cls.aliases().keys()
        in_models = model in list(itertools.chain.from_iterable(cls.models().values()))
        return in_aliases or in_models

    @classmethod
    def _validate_model(cls, model: str) -> str:
        """
        Validate the model name against the supported models and aliases.
        Converts aliases to their corresponding model names if necessary.
        """
        # Load aliases
        aliases = cls.aliases()
        # Assign models based on aliases
        if model in cls.aliases().keys():
            model = aliases[model]
        elif cls.is_supported(model):
            model = model
        else:
            ValueError(
                f"WARNING: Model not found locally: {model}. This may cause errors."
            )
        return model

    # Object methods
    def __init__(self, model: str = "gpt-4o", console: Optional["Console"] = None):
        self.model = self._validate_model(model)
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
        model_list = self.__class__.models()
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
        input: str | list | Message | ImageMessage | AudioMessage,
        verbose: bool = True,
        parser: Parser | None = None,
        raw=False,
        cache=True,
        temperature: Optional[float] = None,  # None means just use the defaults
    ) -> BaseModel | str:
        """
        Execute a query against the language model with optional progress tracking.
        
        Args:
            input: The query text or list of messages to send to the model
            parser: Optional parser to structure the response
            cache: Whether to use response caching (default: True)
            verbose: Whether to display progress information (default: True)
            index: Current item number for batch progress display (requires total)
            total: Total number of items for batch progress display (requires index)
            **kwargs: Additional model-specific parameters
            
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
        if Model._chain_cache and cache:
            cached_request = Model._chain_cache.cache_lookup(input, self.model)
            if cached_request:
                print("Cache hit!")
                if parser:
                    try:
                        cached_request_dict = json.loads(cached_request)
                        obj = parser.pydantic_model(**cached_request_dict)  # type: ignore
                        if raw:
                            return (obj, cached_request)  # type: ignore
                        if not raw:
                            return obj
                    except Exception as e:
                        print(f"Failed to parse cached request: {e}")
                return cached_request
        if parser == None:
            llm_output = self._client.query(
                self.model, input, raw=False, temperature=temperature
            )
        else:
            obj, llm_output = self._client.query(self.model, input, parser, raw=True)
        if Model._chain_cache and cache:
            cached_request = CachedRequest(
                user_input=input, model=self.model, llm_output=llm_output
            )
            Model._chain_cache.insert_cached_request(cached_request)
        if parser and not raw:
            return obj  # type: ignore
        elif parser and raw:
            return obj, llm_output  # type: ignore
        else:
            return llm_output

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

    def stream(
        self,
        input: str | list,
        verbose: bool = True,
        parser: Parser | None = None,
        temperature: Optional[float] = None,
    ):
        if verbose:
            print(f"Model: {self.model}   Query: " + self.pretty(str(input)))
        if Model._chain_cache:
            cached_request = Model._chain_cache.cache_lookup(input, self.model)
            if cached_request:
                print("Cache hit!")
                return cached_request
        results = self._client.query(self.model, input, parser)
        if Model._chain_cache:
            cached_request = CachedRequest(
                user_input=input, model=self.model, llm_output=results
            )
            Model._chain_cache.insert_cached_request(cached_request)
        stream = self._client.stream(self.model, input, parser, temperature)
        return stream
