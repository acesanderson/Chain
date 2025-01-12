from pathlib import Path
import importlib
import json
import itertools
from Chain.cache.cache import ChainCache, CachedRequest
from pydantic import BaseModel

dir_path = Path(__file__).resolve().parent


class Model:
    # Some class variables: models, context sizes, clients
    # Load models from the JSON file. Why classmethod and property?
    # Because models is a class-level variable (Model.models, not model.models).
    # We want it to dynamically load the models from the models file everytime you access the attribute, because Ollama models can change.
    @classmethod
    @property
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
            if value not in list(itertools.chain.from_iterable(cls.models.values())):
                raise ValueError(
                    f"WARNING: This model declared in aliases.json is not available: {value}."
                )
        # Assign models based on aliases
        if model in aliases.keys():
            model = aliases[model]
        elif model in list(
            itertools.chain.from_iterable(cls.models.values())
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
        model_list = self.__class__.models
        if model in model_list["openai"]:
            return "openai", "OpenAIClientSync"
        elif model in model_list["anthropic"]:
            return "anthropic", "AnthropicClientSync"
        elif model in model_list["google"]:
            return "google", "GoogleClient"
        elif model in model_list["ollama"]:
            return "ollama", "OllamaClient"
        elif model in model_list["groq"]:
            return "groq", "GroqClient"
        elif model in model_list["deepseek"]:
            return "deepseek", "DeepSeekClient"
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
        input: str | list,
        verbose: bool = True,
        pydantic_model: BaseModel | None = None,
    ) -> BaseModel | str:
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
        return self._client.query(self.model, input, pydantic_model)

    def pretty(self, user_input):
        pretty = user_input.replace("\n", " ").replace("\t", " ").strip()
        return pretty[:100]

    def __repr__(self):
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"


class ModelAsync(Model):

    def _get_client_type(self, model: str) -> tuple:
        """
        Overrides the parent method to return the async version of each client type.
        """
        model_list = self.__class__.models
        if model in model_list["openai"]:
            return "openai", "OpenAIClientAsync"
        elif model in model_list["anthropic"]:
            return "anthropic", "AnthropicClientAsync"
        # elif model in model_list["google"]:
        #     return "google", "GoogleClient"
        # elif model in model_list["ollama"]:
        #     return "ollama", "OllamaClient"
        # elif model in model_list["groq"]:
        #     return "groq", "GroqClient"
        else:
            raise ValueError(f"Model {model} not found in models")

    async def query(
        self,
        input: str | list,
        verbose: bool = True,
        pydantic_model: BaseModel | None = None,
    ) -> BaseModel | str:
        if Model._chain_cache:
            cached_request = Model._chain_cache.cache_lookup(input, self.model)
            if cached_request:
                return cached_request
        if verbose:
            print(f"Model: {self.model}   Query: " + self.pretty(str(input)))
        results = await self._client.query(self.model, input, pydantic_model)
        if Model._chain_cache:
            cached_request = CachedRequest(
                user_input=input, model=self.model, llm_output=results
            )
            Model._chain_cache.insert_cached_request(cached_request)
        return results
