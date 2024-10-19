"""
Our first client subclass.
"""

from .client import Client
from openai import OpenAI
import instructor
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()


class OpenAIClient(Client):
    def __init__(self):
        self._client = self._initialize_client()

    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        openai_client = OpenAI(api_key=self._get_api_key())
        return instructor.from_openai(openai_client)

    def _get_api_key(self):
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("No OpenAI API key found in environment variables")
        else:
            return os.getenv("OPENAI_API_KEY")

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

    async def query_async(
        self, model: str, input: "str | list", pydantic_model: "BaseModel" = None
    ) -> "BaseModel | str":
        # Implement asynchronous query logic here
        # This would be similar to the synchronous version but using async calls
        pass
