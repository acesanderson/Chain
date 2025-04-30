"""
For Google Gemini models.
"""

from Chain.model.clients.client import Client
from Chain.model.clients.load_env import load_env
from openai import OpenAI, AsyncOpenAI
import instructor
from pydantic import BaseModel


class GoogleClient(Client):
    """
    This is a base class; we have two subclasses: GoogleClientSync and GoogleClientAsync.
    Don't import this.
    """

    def __init__(self):
        self._client = self._initialize_client()

    def _get_api_key(self):
        api_key = load_env("GOOGLE_API_KEY")
        return api_key

    def tokenize(self, model: str, text: str) -> int:
        """
        Get the token count per official tokenizer (through API).
        """
        # Example using google-generativeai SDK for estimation
        import google.generativeai as genai

        client = genai.GenerativeModel(model_name=model)
        response = client.count_tokens(text)
        token_count = response.total_tokens
        return token_count


class GoogleClientSync(GoogleClient):
    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        client = instructor.from_openai(
            OpenAI(
                api_key=self._get_api_key(),
                base_url="https://generativelanguage.googleapis.com/v1beta/",
            ),
            mode=instructor.Mode.JSON,
        )
        return client

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
        self, model: str, input: "str | list", pydantic_model: BaseModel | None = None
    ) -> str | BaseModel:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]

        stream = self._client.chat.completions.create(
            model=model, response_model=pydantic_model, messages=input, stream=True
        )
        return stream


class GoogleClientAsync(GoogleClient):
    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        client = instructor.from_openai(
            AsyncOpenAI(
                api_key=self._get_api_key(),
                base_url="https://generativelanguage.googleapis.com/v1beta/",
            ),
            mode=instructor.Mode.JSON,
        )
        return client

    async def query(
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
            obj, raw_response = (
                await self._client.chat.completions.create_with_completion(
                    model=model, response_model=pydantic_model, messages=input
                )
            )
            raw_text = raw_response.choices[0].message.tool_calls[0].function.arguments
            return obj, raw_text
        # Default behavior is to return only the pydantic model.
        elif pydantic_model:
            obj = await self._client.chat.completions.create(
                model=model, response_model=pydantic_model, messages=input
            )
            return obj
        # If you are not passing pydantic models, you will get the text response.
        else:
            response = await self._client.chat.completions.create(
                model=model, response_model=None, messages=input
            )
            return response.choices[0].message.content
