from Chain.model.clients.client import Client
from Chain.model.clients.load_env import load_env
from Chain.message.message import Message
from Chain.message.imagemessage import ImageMessage
from Chain.message.audiomessage import AudioMessage
from Chain.parser.parser import Parser
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
        parser: Parser | None = None,
        raw=False,
        temperature: Optional[float] = None,
    ) -> str | BaseModel | tuple[BaseModel, str]:
        messages = []
        if isinstance(input, str):
            messages = [Message(role="user", content=input)]
        elif isinstance(input, Message):
            messages = [input]
        elif isinstance(input, list):
            messages = input
        # Dev: we should have a list of messages at this point.
        assert isinstance(messages, list)
        # Convert messages to OpenAI format
        converted_messages = []
        for message in messages:
            if isinstance(message, ImageMessage):
                converted_messages.append(message.to_openai().model_dump())
            elif isinstance(message, AudioMessage):
                if not model == "gpt-4o-audio-preview":
                    raise ValueError(
                        "AudioMessage can only be used with the gpt-4o-audio-preview model."
                    )
                converted_messages.append(message.to_openai().model_dump())
            if isinstance(message, Message):
                converted_messages.append(message.model_dump())
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        # Construct params
        params = {
            "model": model,
            "messages": converted_messages,
            "response_model": parser.pydantic_model if parser else None,
        }
        # Determine if model takes temperature (reasoning models -- starting with 'o' -- don't)
        if temperature:
            if model.startswith("o"):
                raise ValueError("OpenAI reasoning models don't take temperature.")
            if temperature < 0 or temperature > 2:
                raise ValueError("OpenAI models need a temperature between 0 and 2.")
            params.update({"temperature": temperature})
        # If you are passing pydantic models and also want the text response, you need to set raw=True.
        if raw and parser:
            obj, raw_response = self._client.chat.completions.create_with_completion(
                **params
            )
            raw_text = raw_response.choices[0].message.tool_calls[0].function.arguments
            return obj, raw_text
        # Default behavior is to return only the pydantic model.
        elif parser:
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
        parser: Parser | None = None,
        temperature: Optional[float] = None,
    ) -> Stream:
        messages = []
        if isinstance(input, str):
            messages = [Message(role="user", content=input)]
        elif isinstance(input, Message):
            messages = [input]
        elif isinstance(input, list):
            messages = input
        # Dev: we should have a list of messages at this point.
        assert isinstance(messages, list)
        assert len(messages) > 0, "Input messages cannot be empty."
        # Convert messages to OpenAI format
        converted_messages = []
        for message in messages:
            if isinstance(message, ImageMessage):
                converted_messages.append(message.to_openai().model_dump())
            elif isinstance(message, AudioMessage):
                if not model == "gpt-4o-audio-preview":
                    raise ValueError(
                        "AudioMessage can only be used with the gpt-4o-audio-preview model."
                    )
                converted_messages.append(message.to_openai().model_dump())
            if isinstance(message, Message):
                converted_messages.append(message.model_dump())
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        # Build our params; pydantic_model will be None if we didn't request it.
        params = {
            "model": model,
            "messages": converted_messages,
            "response_model": parser.pydantic_model if parser else None,
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
        parser: Parser | None = None,
        raw=False,
        temperature: Optional[float] = None,
    ) -> str | BaseModel | tuple[BaseModel, str]:
        if isinstance(input, str):
            messages = [Message(role="user", content=input)]
        elif isinstance(input, Message):
            messages = [input]
        elif isinstance(input, list):
            messages = input
        # Dev: we should have a list of messages at this point.
        assert isinstance(messages, list)
        assert len(messages) > 0, "Input messages cannot be empty."
        # Convert messages to OpenAI format
        converted_messages = []
        for message in messages:
            if isinstance(message, ImageMessage):
                converted_messages.append(message.to_openai().model_dump())
            elif isinstance(message, AudioMessage):
                if not model == "gpt-4o-audio-preview":
                    raise ValueError(
                        "AudioMessage can only be used with the gpt-4o-audio-preview model."
                    )
                converted_messages.append(message.to_openai().model_dump())
            if isinstance(message, Message):
                converted_messages.append(message.model_dump())
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        # Build our params; pydantic_model will be None if we didn't request it.
        params = {
            "model": model,
            "messages": converted_messages,
            "response_model": parser.pydantic_model if parser else None,
        }
        # Determine if model takes temperature (reasoning models -- starting with 'o' -- don't)
        if not model.startswith("o"):
            params.update({"temperature": temperature})
        # If you are passing pydantic models and also want the text response, you need to set raw=True.
        if raw and parser:
            obj, raw_response = (
                await self._client.chat.completions.create_with_completion(**params)
            )
            raw_text = raw_response.choices[0].message.tool_calls[0].function.arguments
            return obj, raw_text
        # Default behavior is to return only the pydantic model.
        elif parser:
            obj = await self._client.chat.completions.create(**params)
            return obj
        # If you are not passing pydantic models, you will get the text response.
        else:
            response = await self._client.chat.completions.create(**params)
            return response.choices[0].message.content
