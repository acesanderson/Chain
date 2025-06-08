from pathlib import Path
import importlib
import json
import itertools
from Chain.cache.cache import ChainCache, CachedRequest
from Chain.message.message import Message
from Chain.message.imagemessage import ImageMessage
from Chain.message.audiomessage import AudioMessage
from pydantic import BaseModel
from typing import Optional

dir_path = Path(__file__).resolve().parent


class Model:
    # Some class variables: models, clients, chain_cache
    # Load models from the JSON file. Why classmethod?G
    # Because models is a class-level method (Model.models, not model.models).
    # We want it to dynamically load the models from the models file everytime you access the attribute, because Ollama models can change.
    @classmethod
    def models(cls):
        with open(dir_path / "clients/models.json") as f:
            return json.load(f)

    # Store lazy-loaded client instances at the class level
    _clients = {}
    # If you want to add a cache, add it at class level as a singleton.
    _chain_cache: ChainCache | None = None

    def __init__(self, model: str = "gpt-4o"):
        self.model = self._validate_model(model)
        self._client_type = self._get_client_type(self.model)
        # Add client loading logic
        self._client = self.__class__._get_client(self._client_type)

    def _validate_model(cls, model: str) -> str:
        """
        This is where you can put in any model aliases you want to support.
        """
        # Load our aliases from aliases.json
        try:
            with open(dir_path / "aliases.json", "r") as f:
                aliases = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"WARNING: aliases.json not found. This may cause errors."
            )
        except json.JSONDecodeError:
            raise ValueError(
                f"WARNING: aliases.json is not a valid JSON file. This may cause errors."
            )

        # Check data quality.
        for value in aliases.values():
            if value not in list(itertools.chain.from_iterable(cls.models().values())):
                raise ValueError(
                    f"WARNING: This model declared in aliases.json is not available: {value}."
                )
        # Assign models based on aliases
        if model in aliases.keys():
            model = aliases[model]
        elif model in list(
            itertools.chain.from_iterable(cls.models().values())
        ):  # any other model we support (flattened the list)
            model = model
        else:
            ValueError(
                f"WARNING: Model not found locally: {model}. This may cause errors."
            )
        return model

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

    def query(
        self,
        input: str | list | Message | ImageMessage | AudioMessage,
        verbose: bool = True,
        pydantic_model: BaseModel | None = None,
        raw=False,
        cache=True,
        temperature: Optional[float] = None,  # None means just use the defaults
    ) -> BaseModel | str:
        if verbose:
            print(
                f"Model: {self.model}  Temperature: {temperature}  Query: "
                + self.pretty(str(input))
            )
        if Model._chain_cache and cache:
            cached_request = Model._chain_cache.cache_lookup(input, self.model)
            if cached_request:
                print("Cache hit!")
                if pydantic_model:
                    try:
                        cached_request_dict = json.loads(cached_request)
                        obj = pydantic_model(**cached_request_dict)  # type: ignore
                        if raw:
                            return (obj, cached_request)  # type: ignore
                        if not raw:
                            return obj
                    except Exception as e:
                        print(f"Failed to parse cached request: {e}")
                return cached_request
        if pydantic_model == None:
            llm_output = self._client.query(
                self.model, input, raw=False, temperature=temperature
            )
        else:
            obj, llm_output = self._client.query(
                self.model, input, pydantic_model, raw=True
            )
        if Model._chain_cache and cache:
            cached_request = CachedRequest(
                user_input=input, model=self.model, llm_output=llm_output
            )
            Model._chain_cache.insert_cached_request(cached_request)
        if pydantic_model and not raw:
            return obj  # type: ignore
        elif pydantic_model and raw:
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
        pydantic_model: BaseModel | None = None,
        temperature: Optional[float] = None,
    ):
        if verbose:
            print(f"Model: {self.model}   Query: " + self.pretty(str(input)))
        if Model._chain_cache:
            cached_request = Model._chain_cache.cache_lookup(input, self.model)
            if cached_request:
                print("Cache hit!")
                return cached_request
        results = self._client.query(self.model, input, pydantic_model)
        if Model._chain_cache:
            cached_request = CachedRequest(
                user_input=input, model=self.model, llm_output=results
            )
            Model._chain_cache.insert_cached_request(cached_request)
        stream = self._client.stream(self.model, input, pydantic_model, temperature)
        return stream


from Chain.api.client.ChainClient import ChainClient, ChainRequest, get_url


class ModelClient(Model):
    """
    Model for interacting with a ChainServer.
    Primary use case: ollama models using my desktop with RTX 5090.
    """

    _client = ChainClient()

    def __init__(
        self, url: str = get_url()
    ):  # get_url is me defaulting to my own server
        """
        Initialize the ModelClient with the ChainClient.
        """
        super().__init__()
        self.model = "chain"
        self._client_type = self._get_client_type(self.model)
        self._client = self.__class__._get_client(self._client_type)

    def query(
        self,
        input: str | list | Message | ImageMessage | AudioMessage,
        verbose: bool = True,
        pydantic_model: BaseModel | None = None,
        raw=False,
        cache=True,
        temperature: Optional[float] = None,  # None means just use the defaults
    ) -> BaseModel | str:
        if verbose:
            print(
                f"Model: {self.model}  Temperature: {temperature}  Query: "
                + self.pretty(str(input))
            )
        if Model._chain_cache and cache:
            cached_request = Model._chain_cache.cache_lookup(input, self.model)
            if cached_request:
                print("Cache hit!")
                if pydantic_model:
                    try:
                        cached_request_dict = json.loads(cached_request)
                        obj = pydantic_model(**cached_request_dict)  # type: ignore
                        if raw:
                            return (obj, cached_request)  # type: ignore
                        if not raw:
                            return obj
                    except Exception as e:
                        print(f"Failed to parse cached request: {e}")
                return cached_request
        if pydantic_model == None:
            llm_output = self._client.query(
                self.model, input, raw=False, temperature=temperature
            )
        else:
            obj, llm_output = self._client.query(
                self.model, input, pydantic_model, raw=True
            )
        if Model._chain_cache and cache:
            cached_request = CachedRequest(
                user_input=input, model=self.model, llm_output=llm_output
            )
            Model._chain_cache.insert_cached_request(cached_request)
        if pydantic_model and not raw:
            return obj  # type: ignore
        elif pydantic_model and raw:
            return obj, llm_output  # type: ignore
        else:
            return llm_output
