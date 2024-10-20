"""
Client subclass for Anthropic models.
This doesn't require an API key since these are locally hosted models.
This has special logic for updating the models.json file, since the available Ollama models will depend on what we have pulled.
"""

from .client import Client
import ollama
from pydantic import BaseModel

# Logic for updating the models.json and for setting the context sizes for Ollama models.
from pathlib import Path
import json
from collections import defaultdict

dir_path = Path(__file__).resolve().parent


class OllamaClient(Client):
    # Load Ollama context sizes from the JSON file
    with open(dir_path / "ollama_context_sizes.json") as f:
        _ollama_context_data = json.load(f)

    # Use defaultdict to set default context size to 4096 if not specified
    _ollama_context_sizes = defaultdict(lambda: 4096)
    _ollama_context_sizes.update(_ollama_context_data)

    def __init__(self):
        self._client = self._initialize_client()
        self.update_ollama_models()  # This allows us to keep the model file up to date.

    def _initialize_client(self):
        """
        As of 10/14/2024, I can't figure out how use Instructor with Ollama while also being able to set num_ctx.
        Best thing to do would be to implement function calling using Ollama SDK (not instructor) and then use pydantic models.
        For now, we're just using this for text generation.
        """
        return ollama

    def _get_api_key(self):
        """
        Best thing about Ollama; no API key needed.
        """
        pass

    def query(
        self, model: str, input: "str | list", pydantic_model: BaseModel = None
    ) -> "str | BaseModel":
        """
        Handles all synchronous requests from Ollama models.
        DOES NOT SUPPORT PYDANTIC MODELS, SINCE I CANNOT FIGURE HOW TO SET CTX FOR INSTRUCTOR.
        num_ctx = context window size, which is set in Chain.ollama_context_sizes.
        """
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        elif type(input[0]) == dict:
            input = input
        else:
            raise ValueError(
                f"Input not recognized as a valid input type: {type:input}: {input}"
            )
        # call our client
        response = ollama.chat(
            model=model,
            messages=input,
            options={"num_ctx": self._ollama_context_sizes[model]},
        )
        if pydantic_model:
            print("Pydantic model not supported for Ollama models currently.")
        else:
            return response["message"]["content"]

    async def query_async(
        self, model: str, input: "str | list", pydantic_model: "BaseModel" = None
    ) -> "BaseModel | str":
        # Implement asynchronous query logic here
        # This would be similar to the synchronous version but using async calls
        pass

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