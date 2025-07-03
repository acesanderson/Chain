from Chain.model.clients.client import Client, Usage
from Chain.model.clients.load_env import load_env
from Chain.model.params.params import Params
from Chain.logs.logging_config import get_logger
from openai import OpenAI, AsyncOpenAI, Stream
from pydantic import BaseModel
import instructor, tiktoken


logger = get_logger(__name__)

class OpenAIClient(Client):
    """
    This is a base class; we have two subclasses: OpenAIClientSync and OpenAIClientAsync.
    Don't import this.
    """

    def __init__(self):
        self._client = self._initialize_client()

    def _get_api_key(self) -> str:
        api_key = load_env("OPENAI_API_KEY")
        return api_key

    def tokenize(self, model: str, text: str) -> int:
        """
        Return the token count for a string, per model's tokenization function.
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(text))
        return token_count


class OpenAIClientSync(OpenAIClient):
    def _initialize_client(self) -> object:
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        openai_client = OpenAI(api_key=self._get_api_key())
        return instructor.from_openai(openai_client)

    def query(
        self,
        params: Params,
    ) -> tuple:
        structured_response = None
        if params.response_model is not None:
            # We want the raw response from OpenAI, so we use `create_with_completion`
            structured_response, result = self._client.chat.completions.create_with_completion(
                **params.to_openai()
            )
        else:
            # Use the standard completion method
            result = self._client.chat.completions.create(**params.to_openai())
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
            # If the result is a BaseModel or Stream, handle accordingly
            pass
        if isinstance(result, BaseModel):
            return result, usage
        elif isinstance(result, Stream):
            # Handle streaming response if needed
            return result, usage

class OpenAIClientAsync(OpenAIClient):
    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        openai_async_client = AsyncOpenAI(api_key=self._get_api_key())
        return instructor.from_openai(openai_async_client)

    async def query(
        self,
        params: Params,
    ) -> tuple:
        structured_response = None
        if params.response_model is not None:
            # We want the raw response from OpenAI, so we use `create_with_completion`
            structured_response, result = await self._client.chat.completions.create_with_completion(
                **params.to_openai()
            )
        else:
            # Use the standard completion method
            result = await self._client.chat.completions.create(**params.to_openai())
        # Capture usage
        usage = Usage(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
        )
        if structured_response is not None:
            # If we have a structured response, return it along with usage
            return structured_response, usage
        try:
            result = result.choices[0].message.content
            return result, usage
        except AttributeError:
            # If the result is a BaseModel or Stream, handle accordingly
            pass
