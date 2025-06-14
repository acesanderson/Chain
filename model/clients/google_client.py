"""
For Google Gemini models.
"""

from Chain.model.clients.client import Client
from Chain.model.clients.load_env import load_env
from Chain.message.message import Message
from Chain.message.imagemessage import ImageMessage
from Chain.message.audiomessage import AudioMessage
from Chain.parser.parser import Parser
from openai import OpenAI, AsyncOpenAI
import instructor
from pydantic import BaseModel
from typing import Optional


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
        # Dev: we should have a list of Message objects now.
        assert isinstance(messages, list)
        # Convert messages to OpenAI format
        converted_messages = []
        for message in messages:
            if isinstance(message, Message):
                converted_messages.append(message.model_dump())
            elif isinstance(message, ImageMessage):
                converted_messages.append(message.to_openai().model_dump())
            elif isinstance(message, AudioMessage):
                converted_messages.append(message.to_openai().model_dump())
            else:
                raise TypeError("Unsupported message type: {}".format(type(message)))
        # Construct params
        params = {
            "model": model,
            "messages": converted_messages,
            "response_model": parser.pydantic_model if parser else None,
        }
        # Determine if model takes temperature (reasoning models -- starting with 'o' -- don't)
        if temperature:
            if temperature < 0 or temperature > 2:
                raise ValueError("Gemini models need a temperature between 0 and 2.")
            params.update({"temperature": temperature})
        # If you are passing pydantic models and also want the text response, you need to set raw=True.
        if raw and parser:
            obj, raw_response = self._client.chat.completions.create_with_completion(
                **params
            )
            raw_text = raw_response.choices[0].message.tool_calls[0].function.arguments
            return obj, raw_text
        elif parser:
            obj = self._client.chat.completions.create(**params)
            return obj
        # Default is just return the response, i.e. just BaseModel or text.
        else:
            response = self._client.chat.completions.create(**params)
            return response.choices[0].message.content

    def stream(
        self,
        model: str,
        input: "str | list",
        pydantic_model: BaseModel | list[BaseModel] | None = None,
        temperature: Optional[float] = None,
    ) -> str | BaseModel:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        # Build our params; pydantic_model will be None if we didn't request it.
        params = {"model": model, "messages": input, "response_model": pydantic_model}
        # Determine if model takes temperature (reasoning models -- starting with 'o' -- don't)
        if temperature:
            if temperature < 0 or temperature > 2:
                raise ValueError("OpenAI models need a temperature between 0 and 2.")
            params.update({"temperature": temperature})
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
        temperature: Optional[float] = None,
    ) -> str | BaseModel | tuple[BaseModel, str]:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        # Build our params; pydantic_model will be None if we didn't request it.
        params = {"model": model, "messages": input, "response_model": pydantic_model}
        # Determine if model takes temperature (reasoning models -- starting with 'o' -- don't)
        if temperature:
            if temperature < 0 or temperature > 2:
                raise ValueError("Gemini models need a temperature between 0 and 2.")
            params.update({"temperature": temperature})
        # If you are passing pydantic models and also want the text response, you need to set raw=True.
        if raw and pydantic_model:
            obj, raw_response = (
                await self._client.chat.completions.create_with_completion(**params)
            )
            raw_text = raw_response.choices[0].message.tool_calls[0].function.arguments
            return obj, raw_text
        # Default is just return the response, i.e. just BaseModel or text.
        else:
            response = await self._client.chat.completions.create(**params)
            return response.choices[0].message.content
