"""
Client subclass for Anthropic models.
TBD: implement streaming support.
"""

from Chain.model.clients.client import Client
from Chain.model.params.params import Params
from Chain.model.clients.load_env import load_env
from anthropic import Anthropic, AsyncAnthropic, Stream
from pydantic import BaseModel
import instructor, os


class AnthropicClient(Client):

    def __init__(self):
        self._client = self._initialize_client()

    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        anthropic_client = Anthropic(api_key=self._get_api_key())
        return instructor.from_anthropic(anthropic_client)

    def _get_api_key(self):
        load_env("ANTHROPIC_API_KEY")
        if os.getenv("ANTHROPIC_API_KEY") is None:
            raise ValueError("No ANTHROPIC_API_KEY found in environment variables")
        else:
            return os.getenv("ANTHROPIC_API_KEY")

    def tokenize(self, model: str, text: str) -> int:
        """
        Get token count per official Anthropic api endpoint.
        """
        # Convert text to message format
        anthropic_client = Anthropic(api_key=self._get_api_key())
        messages = [{"role": "user", "content": text}]
        token_count = anthropic_client.messages.count_tokens(
            model=model,
            messages=messages,
        )
        return token_count.input_tokens


class AnthropicClientSync(AnthropicClient):

    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        anthropic_client = Anthropic(api_key=self._get_api_key())
        return instructor.from_anthropic(anthropic_client)

    def query(
        self,
        params: Params,
    ) -> str | BaseModel | Stream | None:
        result = self._client.chat.completions.create(**params.to_anthropic())
        # First try to get text content from the result
        try:
            result = result.content[0].text
            return result
        except AttributeError:
            pass
        if isinstance(result, BaseModel):
            return result
        elif isinstance(result, Stream):
            # Handle streaming response if needed
            return result


class AnthropicClientAsync(AnthropicClient):
    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        anthropic_async_client = AsyncAnthropic(api_key=self._get_api_key())
        return instructor.from_anthropic(anthropic_async_client)

    async def query(
        self,
        params: Params,
        ) -> str | BaseModel | Stream | None:
        result = await self._client.chat.completions.create(**params.to_anthropic())
        # First try to get text content from the result
        try:
            result = result.content[0].text
            return result
        except AttributeError:
            pass
        if isinstance(result, BaseModel):
            return result
        elif isinstance(result, Stream):
            # Handle streaming response if needed
            return result
