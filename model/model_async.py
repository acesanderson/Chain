from Chain.model.model import Model
from Chain.cache.cache import CachedRequest
from Chain.parser.parser import Parser
from Chain.progress.wrappers import progress_display
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

    @progress_display
    async def query_async(
        self,
        input: str | list,
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

        Args:
            input (str | list): The query text (string) or a list of `Message`
                objects to send to the model. Supports multimodal messages
                (e.g., `ImageMessage`, `AudioMessage`) if the model client
                supports them.
            verbose (bool): If True, displays progress information (e.g., a
                spinner) for this individual query. Defaults to True.
            parser (Parser | None): An optional `Parser` object to structure
                the model's response into a Pydantic `BaseModel`. Defaults to None.
            raw (bool): If True and a `parser` is provided, returns a tuple
                containing both the parsed Pydantic object and the raw JSON/text
                output from the model. This is primarily used internally for caching.
                Defaults to False.
            cache (bool): If True, the query attempts to retrieve a response
                from the `Model._chain_cache` before making an API call. If a
                response is not found, it is cached upon successful completion.
                Defaults to True.
            print_response (bool): If True, prints the model's raw output to
                the console after the query is complete. Useful for debugging.
                Defaults to False.

        Returns:
            BaseModel | str | tuple[BaseModel, str]:
                - If `parser` is provided and `raw` is False: the parsed Pydantic model.
                - If `parser` is not provided: the raw text response from the model.
                - If `parser` is provided and `raw` is True: a tuple containing
                  the parsed Pydantic model and its raw JSON/text representation.

        Examples:
            >>> import asyncio
            >>> from Chain import ModelAsync
            >>>
            >>> async def main():
            >>>     model = ModelAsync("gpt-4o-mini")
            >>>     # Basic async query
            >>>     response_text = await model.query_async("Explain asynchronous programming.")
            >>>     print(f"Response: {response_text[:50]}...")
            >>>
            >>>     # Query with a parser
            >>>     from pydantic import BaseModel
            >>>     from Chain import Parser
            >>>
            >>>     class CapitalInfo(BaseModel):
            >>>         city: str
            >>>         country: str
            >>>
            >>>     parser = Parser(CapitalInfo)
            >>>     capital_obj = await model.query_async("What is the capital of France?", parser=parser)
            >>>     print(f"Capital: {capital_obj.city}, Country: {capital_obj.country}")
            >>>
            >>> asyncio.run(main())
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
