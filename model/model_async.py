from Chain.model.model import Model
from Chain.model.models.models import ModelStore
from Chain.parser.parser import Parser
from Chain.model.params.params import Params
from Chain.progress.wrappers import progress_display
import importlib
from pydantic import BaseModel


class ModelAsync(Model):
    _async_clients = {}  # Separate from Model._clients

    def _get_client_type(self, model: str) -> tuple:
        """
        Overrides the parent method to return the async version of each client type.
        """
        model_list = ModelStore.models()
        if model in model_list["openai"]:
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

    @progress_display
    async def query_async(
        self,
        query_input: str | list,
        verbose: bool = True,
        parser: Parser | None = None,
        raw=False,
        cache=True,
        print_response=False,
    ) -> BaseModel | str:
        """
        Asynchronously executes a query against the language model with optional
        progress tracking.

        This method handles asynchronous interaction with the underlying LLM
        client, applying caching and parsing as configured. It's designed to
        be awaited in an async context, often used within `AsyncChain` for
        concurrent processing.
        """
        # Here's the magic -- kwargs goes right into our Params object.
        import inspect
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        
        query_args = {k: values[k] for k in args if k != "self"}
        query_args["model"] = self.model
        params = Params(**query_args)
        # We need to handle the following:
        ## Chaincache implementation (if cache = True)
        response: str | BaseModel = await self._client.query(params)
        return response
