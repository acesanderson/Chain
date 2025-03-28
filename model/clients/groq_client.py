"""
For Groq's online models.
"""

from Chain.model.clients.client import Client
from Chain.model.clients.load_env import load_env
from openai import OpenAI
import instructor
from pydantic import BaseModel
import os
from groq import Groq


class GroqClient(Client):
    """
    This is a base class; we have two subclasses: OpenAIClientSync and OpenAIClientAsync.
    Don't import this.
    """

    def __init__(self):
        self._client = self._initialize_client()

    def _initialize_client(self):
        """
        Logic for this is unique to each client (sync / async).
        """
        pass

    def _get_api_key(self):
        api_key = load_env("GROQ_API_KEY")
        return api_key

    def query(
        self,
        model: str,
        input: "str | list",
        pydantic_model: BaseModel | None = None,
        raw=False,
    ):
        """
        Logic for this is unique to each client (sync / async).
        """
        pass


class GroqClientSync(GroqClient):
    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        client = Groq(api_key=self._get_api_key())
        return instructor.from_groq(client)

    def query(
        self,
        model: str,
        input: "str | list",
        pydantic_model: BaseModel | None = None,
        raw=False,
    ) -> str | BaseModel | tuple[BaseModel, str]:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]

        # If you are passing pydantic models and also want the text response, you need to set raw=True.
        if raw and pydantic_model:
            obj, raw_response = self._client.chat.completions.create_with_completion(
                model=model, response_model=pydantic_model, messages=input
            )
            raw_text = raw_response.choices[0].message.tool_calls[0].function.arguments
            return obj, raw_text
        # Default behavior is to return only the pydantic model.
        elif pydantic_model:
            obj = self._client.chat.completions.create(
                model=model, response_model=pydantic_model, messages=input
            )
            return obj
        # If you are not passing pydantic models, you will get the text response.
        else:
            response = self._client.chat.completions.create(
                model=model, response_model=None, messages=input
            )
            return response.choices[0].message.content

    def stream(
        self, model: str, input: "str | list", pydantic_model: BaseModel = None
    ) -> "str | BaseModel":
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]

        stream = self._client.chat.completions.create(
            model=model, response_model=pydantic_model, messages=input, stream=True
        )
        return stream
