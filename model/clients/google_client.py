"""
For Google Gemini models.
"""

from Chain.model.clients.client import Client
from Chain.model.clients.load_env import load_env
from openai import OpenAI
import instructor
from pydantic import BaseModel


class GoogleClient(Client):
    def __init__(self):
        self._client = self._initialize_client()

    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        As of 11-8-2024, Gemini is now usable with OpenAI SDK.
        """
        gemini_client = OpenAI(
            api_key=self._get_api_key(),
            base_url="https://generativelanguage.googleapis.com/v1beta/",
        )
        return instructor.from_openai(gemini_client)

    def _get_api_key(self):
        api_key = load_env("GOOGLE_API_KEY")
        return api_key

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
