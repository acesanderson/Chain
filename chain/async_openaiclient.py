"""
Our first client subclass.
"""

from Chain.model.clients.client import Client
from Chain.model.clients.load_env import load_env
from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel


class AsyncOpenAIClient(Client):
    def __init__(self):
        self._client = self._initialize_client()

    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        openai_async_client = AsyncOpenAI(api_key=self._get_api_key())
        return instructor.from_openai(openai_async_client)

    def _get_api_key(self):
        api_key = load_env("OPENAI_API_KEY")
        return api_key

    def query():
        pass

    async def query_async(
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
