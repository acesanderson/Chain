"""
For Google Gemini models.
"""

from Chain.model.clients.client import Client, Usage
from Chain.model.clients.load_env import load_env
from Chain.request.request import Request
from openai import OpenAI, AsyncOpenAI, Stream
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
        request: Request,
    ) -> tuple:
        structured_response = None
        if request.response_model is not None:
            # We want the raw response from Google, so we use `create_with_completion`
            structured_response, result = (
                self._client.chat.completions.create_with_completion(
                    **request.to_openai()
                )
            )
        else:
            # Use the standard completion method
            result = self._client.chat.completions.create(**request.to_openai())
        # Capture usage
        usage = Usage(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
        )
        if structured_response is not None:
            # If we have a structured response, return it along with usage
            return structured_response, usage
        # First try to get text content from the result
        try:
            result = result.choices[0].message.content
            return result, usage
        except AttributeError:
            pass
        if isinstance(result, BaseModel):
            return result, usage
        elif isinstance(result, Stream):
            # Handle streaming response if needed
            return result, usage


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
        request: Request,
    ) -> tuple:
        structured_response = None
        if request.response_model is not None:
            # We want the raw response from OpenAI, so we use `create_with_completion`
            structured_response, result = (
                await self._client.chat.completions.create_with_completion(
                    **request.to_openai()
                )
            )
        else:
            # Use the standard completion method
            result = await self._client.chat.completions.create(**request.to_openai())
        # Capture usage
        usage = Usage(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
        )
        if structured_response is not None:
            # If we have a structured response, return it along with usage
            return structured_response, usage
        # First try to get text content from the result
        try:
            result = result.choices[0].message.content
            return result, usage
        except AttributeError:
            pass
        if isinstance(result, BaseModel):
            return result, usage
        elif isinstance(result, Stream):
            # Handle streaming response if needed
            return result, usage
