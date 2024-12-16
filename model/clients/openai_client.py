"""
Our first client subclass.
"""

from Chain.model.clients.client import Client
from Chain.model.clients.load_env import load_env
from openai import OpenAI, AsyncOpenAI
import instructor
from pydantic import BaseModel


class OpenAIClient(Client):
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
        api_key = load_env("OPENAI_API_KEY")
        return api_key

    def query(self, model: str, input: "str | list", pydantic_model: BaseModel = None):
        """
        Logic for this is unique to each client (sync / async).
        """
        pass


class OpenAIClientSync(OpenAIClient):
    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        openai_client = OpenAI(api_key=self._get_api_key())
        return instructor.from_openai(openai_client)

    def query(
        self, model: str, input: "str | list", pydantic_model: BaseModel = None
    ) -> "str | BaseModel":
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]

        response = self._client.chat.completions.create(
            model=model, response_model=pydantic_model, messages=input
        )

        if pydantic_model:
            return response
        else:
            return response.choices[0].message.content


class OpenAIClientAsync(OpenAIClient):
    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        openai_async_client = AsyncOpenAI(api_key=self._get_api_key())
        return instructor.from_openai(openai_async_client)

    async def query(
        self, model: str, input: "str | list", pydantic_model: BaseModel = None
    ) -> "str | BaseModel":
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]

        response = await self._client.chat.completions.create(
            model=model, response_model=pydantic_model, messages=input
        )

        if pydantic_model:
            return response
        else:
            return response.choices[0].message.content
