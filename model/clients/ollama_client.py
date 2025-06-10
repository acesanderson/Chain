"""
Client subclass for Ollama models.
This doesn't require an API key since these are locally hosted models.
We can use openai api calls to the ollama server, but we use the instructor library to handle the API calls.
This has special logic for updating the models.json file, since the available Ollama models will depend on what we have pulled.
We define preferred defaults for context sizes in a separate json file.
"""

from Chain.model.clients.client import Client
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
import instructor
import ollama
from typing import Optional

# Logic for updating the models.json and for setting the context sizes for Ollama models.
from pathlib import Path
import json
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
        ollama_models = [m["name"] for m in ollama.list()["models"]]
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
        pydantic_model: BaseModel | list[BaseModel] | None = None,
        raw=False,
        temperature: Optional[float] = None,
    ) -> str | BaseModel | tuple[BaseModel, str]:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]

        # If you are passing pydantic models and also want the text response, you need to set raw=True.
        if raw and pydantic_model:
            obj, raw_response = self._client.chat.completions.create_with_completion(
                model=model,
                response_model=pydantic_model,
                messages=input,
                extra_body={"options": {"num_ctx": self._ollama_context_sizes[model]}},
                temperature=temperature,
            )
            raw_text = raw_response.choices[0].message.content
            return obj, raw_text
        # Default behavior is to return only the pydantic model.
        elif pydantic_model:
            obj = self._client.chat.completions.create(
                model=model,
                response_model=pydantic_model,
                messages=input,
                extra_body={"options": {"num_ctx": self._ollama_context_sizes[model]}},
                temperature=temperature,
            )
            return obj
        # If you are not passing pydantic models, you will get the text response.
        else:
            response = self._client.chat.completions.create(
                model=model,
                response_model=None,
                messages=input,
                extra_body={"options": {"num_ctx": self._ollama_context_sizes[model]}},
                temperature=temperature,
            )
            return response.choices[0].message.content

    def stream(
        self,
        model: str,
        input: "str | list",
        pydantic_model: BaseModel | list[BaseModel] | None = None,
        temperature: Optional[float] = None,
    ) -> "str | BaseModel":
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        stream = self._client.chat(model=model, messages=input, stream=True)
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
        pydantic_model: BaseModel | list[BaseModel] | None = None,
        raw=False,
        temperature: Optional[float] = None,
    ) -> str | BaseModel | tuple[BaseModel, str]:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        # If you are passing pydantic models and also want the text response, you need to set raw=True.
        if raw and pydantic_model:
            obj, raw_response = (
                await self._client.chat.completions.create_with_completion(
                    model=model,
                    response_model=pydantic_model,
                    messages=input,
                    extra_body={
                        "options": {"num_ctx": self._ollama_context_sizes[model]}
                    },
                )
            )
            raw_text = raw_response.choices[0].message.content
            return obj, raw_text
        # Default behavior is to return only the pydantic model.
        elif pydantic_model:
            obj = await self._client.chat.completions.create(
                model=model,
                response_model=pydantic_model,
                messages=input,
                extra_body={"options": {"num_ctx": self._ollama_context_sizes[model]}},
            )
            return obj
        # If you are not passing pydantic models, you will get the text response.
        else:
            response = await self._client.chat.completions.create(
                model=model,
                response_model=None,
                messages=input,
                extra_body={"options": {"num_ctx": self._ollama_context_sizes[model]}},
            )
            return response.choices[0].message.content
