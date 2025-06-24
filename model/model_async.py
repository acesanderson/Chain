from Chain.model.model import Model
from Chain.model.models.models import ModelStore
from Chain.parser.parser import Parser
from Chain.model.params.params import Params
from Chain.progress.wrappers import progress_display
from Chain.message.message import Message
from Chain.result.result import ChainResult
from Chain.result.response import Response
from Chain.result.error import ChainError
from Chain.cache.cache import check_cache, update_cache
import importlib
from typing import Optional
from time import time
from pydantic import ValidationError


class ModelAsync(Model):
    _async_clients = {}  # Separate from Model._clients

    def _get_client_type(self, model: str) -> tuple:
        """
        Overrides the parent method to return the async version of each client type.
        """
        model_list = ModelStore.models()
        if model in model_list["openai"]:
            return "openai", "OpenAIClientAsync"
        elif model in model_list["anthropic"]:
            return "anthropic", "AnthropicClientAsync"
        elif model in model_list["ollama"]:
            return "ollama", "OllamaClientAsync"
        elif model in model_list["google"]:
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
        cache=False,
        print_response=False,
        params: Optional[Params] = None
    ) -> ChainResult:
        
        try:
            if params == None:
                import inspect
                frame = inspect.currentframe()
                args, _, _, values = inspect.getargvalues(frame)

                query_args = {k: values[k] for k in args if k != "self"}
                query_args["model"] = self.model
                params = Params(**query_args)
            
            assert params and isinstance(params, Params), f"params should be a Params object, not {type(params)}"
            
            # Check cache first
            if cache and self._chain_cache:
                cached_result = check_cache(self, params)
                if cached_result is not None:
                    return cached_result  # This should be a Response

            # Execute the query
            start_time = time()
            result = await self._client.query(params)
            stop_time = time()
            
            # Create Response object
            if isinstance(result, Response):
                response = result
            elif isinstance(result, str):
                from Chain.message.messages import Messages
                user_message = Message(role="user", content=params.query_input or "")
                assistant_message = Message(role="assistant", content=result)
                messages = Messages([user_message, assistant_message])
                
                response = Response(
                    messages=messages,
                    params=params,
                    duration=stop_time - start_time,
                )
            else:
                # Handle other result types (BaseModel, etc.)
                from Chain.message.messages import Messages
                user_message = Message(role="user", content=params.query_input or "")
                assistant_message = Message(role="assistant", content=result)
                messages = Messages([user_message, assistant_message])
                
                response = Response(
                    messages=messages,
                    params=params,
                    duration=stop_time - start_time,
                )

            # Update cache after successful query
            if cache and self._chain_cache:
                update_cache(self, params, response)

            return response  # Always return Response (part of ChainResult)

        except ValidationError as e:
            from Chain.result.error import ChainError
            return ChainError.from_exception(
                e,
                code="validation_error",
                category="client",
                request_params=params.model_dump() if params else {}
            )
        except Exception as e:
            from Chain.result.error import ChainError
            return ChainError.from_exception(
                e,
                code="async_query_error",
                category="client",
                request_params=params.model_dump() if params else {}
            )
