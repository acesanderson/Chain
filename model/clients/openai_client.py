from Chain.model.clients.client import Client
from Chain.model.clients.load_env import load_env
from Chain.message.message import Message
from Chain.message.imagemessage import ImageMessage
from Chain.message.audiomessage import AudioMessage
from openai import OpenAI, AsyncOpenAI, Stream
import instructor
from pydantic import BaseModel
import tiktoken
from typing import Optional


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
        model: str,
        input: str | list | Message | ImageMessage | AudioMessage,
        pydantic_model: BaseModel | None = None,
        raw=False,
        temperature: Optional[float] = None,
    ) -> str | BaseModel | tuple[BaseModel, str]:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        elif isinstance(input, ImageMessage):
            input = [input.to_openai().model_dump()]
        elif isinstance(input, AudioMessage):
            input = [input.to_openai().model_dump()]
        elif isinstance(input, Message):
            input = [input.model_dump()]
        # Process custom message types
        elif isinstance(input, list):
            # First, ImageMessage
            input = [
                (
                    item.to_openai().model_dump()
                    if isinstance(item, ImageMessage)
                    else item.model_dump()
                )
                for item in input
            ]
            # Now, AudioMessage
            input = [
                (
                    item.to_openai().model_dump()
                    if isinstance(item, AudioMessage)
                    else item
                )
                for item in input
            ]
        params = {"model": model, "messages": input, "response_model": pydantic_model}
        # Determine if model takes temperature (reasoning models -- starting with 'o' -- don't)
        if temperature:
            if model.startswith("o"):
                raise ValueError("OpenAI reasoning models don't take temperature.")
            if temperature < 0 or temperature > 2:
                raise ValueError("OpenAI models need a temperature between 0 and 2.")
            params.update({"temperature": temperature})
        # If you are passing pydantic models and also want the text response, you need to set raw=True.
        if raw and pydantic_model:
            obj, raw_response = self._client.chat.completions.create_with_completion(
                **params
            )
            raw_text = raw_response.choices[0].message.tool_calls[0].function.arguments
            return obj, raw_text
        # Default behavior is to return only the pydantic model.
        elif pydantic_model:
            obj = self._client.chat.completions.create(**params)
            return obj
        # If you are not passing pydantic models, you will get the text response.
        else:
            response = self._client.chat.completions.create(**params)
            return response.choices[0].message.content

    def stream(
        self,
        model: str,
        input: "str | list",
        pydantic_model: BaseModel | None = None,
        temperature: Optional[float] = None,
    ) -> Stream:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        # Build our params; pydantic_model will be None if we didn't request it.
        params = {
            "model": model,
            "messages": input,
            "response_model": pydantic_model,
            "stream": True,
        }
        # Determine if model takes temperature (reasoning models -- starting with 'o' -- don't)
        if not model.startswith("o"):
            params.update({"temperature": temperature})
        stream: Stream = self._client.chat.completions.create(**params)
        return stream


class OpenAIClientAsync(OpenAIClient):
    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        openai_async_client = AsyncOpenAI(api_key=self._get_api_key())
        return instructor.from_openai(openai_async_client)

    async def query(
        self,
        model: str,
        input: "str | list",
        pydantic_model: BaseModel | None = None,
        raw=False,
        temperature: Optional[float] = None,
    ) -> str | BaseModel | tuple[BaseModel, str]:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        # Build our params; pydantic_model will be None if we didn't request it.
        params = {"model": model, "messages": input, "response_model": pydantic_model}
        # Determine if model takes temperature (reasoning models -- starting with 'o' -- don't)
        if not model.startswith("o"):
            params.update({"temperature": temperature})
        # If you are passing pydantic models and also want the text response, you need to set raw=True.
        if raw and pydantic_model:
            obj, raw_response = (
                await self._client.chat.completions.create_with_completion(**params)
            )
            raw_text = raw_response.choices[0].message.tool_calls[0].function.arguments
            return obj, raw_text
        # Default behavior is to return only the pydantic model.
        elif pydantic_model:
            obj = await self._client.chat.completions.create(**params)
            return obj
        # If you are not passing pydantic models, you will get the text response.
        else:
            response = await self._client.chat.completions.create(**params)
            return response.choices[0].message.content
