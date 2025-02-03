"""
For DeepSeek's online models.
"""

from Chain.model.clients.client import Client
from Chain.model.clients.load_env import load_env
from openai import OpenAI
import instructor
from pydantic import BaseModel


class DeepSeekClient(Client):
    def __init__(self):
        self._client = self._initialize_client()

    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        As of 11-8-2024, Gemini is now usable with OpenAI SDK.
        """
        deepseek_client = OpenAI(
            api_key=self._get_api_key(),
            base_url="https://api.deepseek.com",
        )
        return instructor.from_openai(deepseek_client)

    def _get_api_key(self):
        api_key = load_env("DEEPSEEK_API_KEY")
        return api_key

    def query(
        self, model: str, input: "str | list", pydantic_model: BaseModel = None
    ) -> "str | BaseModel":
        """
        Handles all synchronous requests from DeepSeek's models.
        Possibilities:
        - pydantic object not provided, input is string -> return string
        - pydantic object provided, input is string -> return pydantic object
        Google doesn't take message objects, apparently. (or it's buried in their documentation)
        """
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        # call our client
        response = self._client.chat.completions.create(
            model=model,
            response_model=pydantic_model,
            messages=input,
        )
        if pydantic_model:
            return response
        else:
            return response.choices[0].message.content
