from Chain.model.model import Model
from Chain.cache.cache import CachedRequest
from Chain.parser.parser import Parser
import importlib, json
from pydantic import BaseModel


class ModelAsync(Model):
    _async_clients = {}  # Separate from Model._clients

    def _get_client_type(self, model: str) -> tuple:
        """
        Overrides the parent method to return the async version of each client type.
        """
        model_list = self.__class__.models
        if model in model_list()["openai"]:
            return "openai", "OpenAIClientAsync"
        elif model in model_list()["anthropic"]:
            return "anthropic", "AnthropicClientAsync"
        elif model in model_list()["ollama"]:
            return "ollama", "OllamaClientAsync"
        elif model in model_list()["google"]:
            return "google", "GoogleClientAsync"
        # elif model in model_list["groq"]:
        #     return "groq", "GroqClient"
        else:
            raise ValueError(f"Model {model} not found in models")

    @classmethod
    def _get_client(cls, client_type: tuple):
        # print(f"client type: {client_type}")
        if client_type[0] not in cls._async_clients:
            try:
                module = importlib.import_module(
                    f"Chain.model.clients.{client_type[0].lower()}_client"
                )
                client_class = getattr(module, f"{client_type[1]}")
                cls._async_clients[client_type[0]] = client_class()
            except ImportError as e:
                raise ImportError(f"Failed to import {client_type} client: {str(e)}")
        client_object = cls._async_clients[client_type[0]]
        if not client_object:
            raise ValueError(f"Client {client_type} not found in clients")
        return client_object

    async def query_async(
        self,
        input: str | list,
        verbose: bool = True,
        parser: Parser | None = None,
        raw=False,
        cache=True,
        print_response=False,
    ) -> BaseModel | str:
        if verbose:
            from datetime import datetime

            print(f"Model: {self.model}   TIME: {datetime.now().strftime('%H:%M:%S')}")
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
                    if print_response:
                        print(f"Response: {cached_request}")
                return cached_request
        if parser == None:
            llm_output = await self._client.query(self.model, input, raw=False)
        else:
            obj, llm_output = await self._client.query(
                self.model, input, parser, raw=True
            )
        if Model._chain_cache and cache:
            cached_request = CachedRequest(
                user_input=input, model=self.model, llm_output=llm_output
            )
            Model._chain_cache.insert_cached_request(cached_request)
        if parser and not raw:
            if print_response:
                print(f"Response: {llm_output}")
            return obj  # type: ignore
        elif parser and raw:
            if print_response:
                print(f"Response: {llm_output}")
            return obj, llm_output  # type: ignore
        else:
            if print_response:
                print(f"Response: {llm_output}")
            return llm_output
