"""
Client subclass for Ollama models.
This doesn't require an API key since these are locally hosted models.
We can use openai api calls to the ollama server, but we use the instructor library to handle the API calls.
This has special logic for updating the models.json file, since the available Ollama models will depend on what we have pulled.
We define preferred defaults for context sizes in a separate json file.
"""

from Chain.model.clients.client import Client
from Chain.parser.parser import Parser
from Chain.message.message import Message
from Chain.message.audiomessage import AudioMessage
from Chain.message.imagemessage import ImageMessage
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI, Stream
from typing import Optional
from pathlib import Path
import instructor, ollama, json
from collections import defaultdict


dir_path = Path(__file__).resolve().parent


class OllamaClient(Client):
    """
    This is a base class; we have two subclasses: OpenAIClientSync and OpenAIClientAsync.
    Don't import this.
    """

    # Load Ollama context sizes from the JSON file
    with open(dir_path / "ollama_context_sizes.json") as f:
        _ollama_context_data = json.load(f)

    # Use defaultdict to set default context size to 4096 if not specified
    _ollama_context_sizes = defaultdict(lambda: 32768)
    _ollama_context_sizes.update(_ollama_context_data)

    def __init__(self):
        self._client = self._initialize_client()
        self.update_ollama_models()  # This allows us to keep the model file up to date.

    def _initialize_client(self):
        """
        Logic for this is unique to each client (sync / async).
        """
        pass

    def _get_api_key(self):
        """
        Best thing about Ollama; no API key needed.
        """
        return ""

    def update_ollama_models(self):
        """
        Updates the list of Ollama models.
        We run is every time ollama is initialized.
        """
        # Lazy load ollama module
        ollama_models = [m.model for m in ollama.list()["models"]]
        with open(dir_path / "models.json", "r") as f:
            model_list = json.load(f)
        model_list["ollama"] = ollama_models
        with open(dir_path / "models.json", "w") as f:
            json.dump(model_list, f)

    def tokenize(self, model: str, text: str) -> int:
        """
        Count tokens using Ollama's generate API via the official library.
        This actually runs a text generation, but only only for one token to minimize compute, since we only want the count of input tokens.
        """
        response = ollama.generate(
            model=model,
            prompt=text,
            options={"num_predict": 1},  # Set to minimal generation
        )
        return int(response.get("prompt_eval_count", 0))


class OllamaClientSync(OllamaClient):
    def _initialize_client(self):
        client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )
        return client

    def query(
        self,
        model: str,
        input: "str | list",
        parser: Parser | None = None,
        raw=False,
        temperature: Optional[float] = None,
    ) -> str | BaseModel | tuple[BaseModel, str]:
        messages = []
        if isinstance(input, str):
            messages = [Message(role="user", content=input)]
        if isinstance(input, Message):
            messages = [input]
        if isinstance(input, list):
            messages = input
        # Dev: we should have a list of message at this point
        assert isinstance(messages, list)
        # Convert messages to OpenAI format
        converted_messages = []
        for message in messages:
            if isinstance(message, ImageMessage):
                converted_messages.append(message.to_openai().model_dump())
            elif isinstance(message, AudioMessage):
                converted_messages.append(message.to_openai().model_dump())
            elif isinstance(message, Message):
                converted_messages.append(message.model_dump())
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        # Construct params
        params = {
            "model": model,
            "messages": converted_messages,
            "response_model": parser.pydantic_model if parser else None,
            "extra_body": {"options": {"num_ctx": self._ollama_context_sizes[model]}},
        }
        if temperature:
            params["temperature"] = temperature
        # If you are passing pydantic models and also want the text response, you need to set raw=True.
        if raw and parser:
            obj, raw_response = self._client.chat.completions.create_with_completion(**params)
            raw_text = raw_response.choices[0].message.content
            return obj, raw_text
        # Default behavior is to return only the pydantic model.
        elif parser:
            obj = self._client.chat.completions.create(**params)
            return obj
        # If you are not passing pydantic models, you will get the text response.
        else:
            response = self._client.chat.completions.create(**params)
            return response.choices[0].message.content

    def stream(
        self,
        model: str,
        input: "str | list",
        parser: Parser | None = None,
        temperature: Optional[float] = None,
    ) -> "str | BaseModel":
        messages = []
        if isinstance(input, str):
            messages = [Message(role="user", content=input)]
        if isinstance(input, Message):
            messages = [input]
        if isinstance(input, list):
            messages = input
        # Dev: we should have a list of message at this point
        assert isinstance(messages, list)
        # Convert messages to OpenAI format
        converted_messages = []
        for message in messages:
            if isinstance(message, ImageMessage):
                converted_messages.append(message.to_openai().model_dump())
            elif isinstance(message, AudioMessage):
                converted_messages.append(message.to_openai().model_dump())
            elif isinstance(message, Message):
                converted_messages.append(message.model_dump())
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        # Construct params
        params = {
            "model": model,
            "messages": converted_messages,
            "response_model": parser.pydantic_model if parser else None,
            "stream": True
        }
        if temperature:
            params["temperature"] = temperature
        stream: Stream = self._client.chat.completions.create(**params)
        return stream


class OllamaClientAsync(OllamaClient):
    def _initialize_client(self):
        """
        This is just ollama's async client.
        """
        ollama_async_client = instructor.from_openai(
            AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
            mode=instructor.Mode.JSON,
        )
        return ollama_async_client

    async def query(
        self,
        model: str,
        input: "str | list",
        parser: Parser | None = None,
        raw=False,
        temperature: Optional[float] = None,
    ) -> str | BaseModel | tuple[BaseModel, str]:
        messages = []
        if isinstance(input, str):
            messages = [Message(role="user", content=input)]
        if isinstance(input, Message):
            messages = [input]
        if isinstance(input, list):
            messages = input
        # Dev: we should have a list of message at this point
        assert isinstance(messages, list)
        # Convert messages to OpenAI format
        converted_messages = []
        for message in messages:
            if isinstance(message, ImageMessage):
                converted_messages.append(message.to_openai().model_dump())
            elif isinstance(message, AudioMessage):
                converted_messages.append(message.to_openai().model_dump())
            elif isinstance(message, Message):
                converted_messages.append(message.model_dump())
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        # Construct params
        params = {
            "model": model,
            "messages": converted_messages,
            "response_model": parser.pydantic_model if parser else None,
            "extra_body": {"options": {"num_ctx": self._ollama_context_sizes[model]}},
        }
        if temperature:
            params["temperature"] = temperature
        if raw and parser:
            obj, raw_response = (
                await self._client.chat.completions.create_with_completion(**params))
            raw_text = raw_response.choices[0].message.content
            return obj, raw_text
        # Default behavior is to return only the pydantic model.
        elif parser:
            obj = await self._client.chat.completions.create(**params)
            return obj
        # If you are not passing pydantic models, you will get the text response.
        else:
            response = await self._client.chat.completions.create(**params)
            return response.choices[0].message.content
