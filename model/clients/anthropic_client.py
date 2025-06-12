"""
Client subclass for Anthropic models.
"""

from Chain.model.clients.client import Client
from Chain.message.message import Message
from Chain.message.imagemessage import ImageMessage
from Chain.model.clients.load_env import load_env
from Chain.parser.parser import Parser
from anthropic import Anthropic, AsyncAnthropic
import instructor
from pydantic import BaseModel
import os
import json
from typing import Optional


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
        model: str,
        input: str | list | Message | ImageMessage,
        parser: Parser | None = None,
        raw=False,
        temperature: Optional[float] = None,
    ) -> str | BaseModel | tuple[BaseModel, str]:
        """
        Handles all synchronous requests from Anthropic's models.
        Possibilities:
        - parser not provided, input is string -> return string
        - parser provided, input is string -> return pydantic object
        - if raw=True, return a tuple of (pydantic object, raw text)
         Anthropic is quirky about system messsages (The Messages API accepts a top-level "system" parameter, not "system" as an input message role.)
        """
        # Anthropic requires a system variable
        system = ""
        if isinstance(input, str):
            input = [Message(role="user", content=input)]
        elif isinstance(input, ImageMessage):
            input = [input.to_anthropic().model_dump()]
        elif isinstance(input, Message):
            input = [input.model_dump()]
        elif isinstance(input, list):
            input = input
            # This is anthropic quirk; we remove the system message and set it as a query parameter.
            if input[0].role == "system":
                system = input[0]["content"]
                input = input[1:]
                # Remote "system" role from any messages in input. Another annoying quirk.
                for message in input:
                    if message.role == "system":
                        message.role = "user"
            # Process image messages if present; if there is an ImageMessage, change it to a dict: ImageMessage.to_anthropic().model_dump()
            for message in input:
                if isinstance(message, ImageMessage):
                    input = [
                        (
                            message.to_anthropic().model_dump()
                            if isinstance(message, ImageMessage)
                            else message
                        )
                        for message in input
                    ]
        else:
            raise ValueError(
                f"Input not recognized as a valid input type: {type(input)}: {input}"
            )
        params = {
            "messages": input,
            "model": model,
            "response_model": None if not parser else parser.pydantic_model,
            "max_retries": 0,
            "system": system,
        }
        # set max_tokens based on model
        if model == "claude-3-5-sonnet-20240620":
            params["max_tokens"] = 8192
        else:
            params["max_tokens"] = 8192
        if temperature:
            if temperature < 0 or temperature > 1:
                raise ValueError(
                    "Temperature for Anthropic models needs to be between 0 and 1."
                )
            else:
                params["temperature"] = temperature
        # Pydantic models always return the tuple at client level (Model does further parsing)
        if raw and parser:
            obj, raw_response = self._client.chat.completions.create_with_completion(
                **params
            )
            raw_text = json.dumps(raw_response.content[0].input)
            return obj, raw_text
        # Return just the string.
        else:
            response = self._client.chat.completions.create(**params)
            return response.content[0].text


class AnthropicClientAsync(AnthropicClient):
    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        anthropic_async_client = AsyncAnthropic(api_key=self._get_api_key())
        return instructor.from_anthropic(anthropic_async_client)

    async def query(
        self,
        model: str,
        input: "str | list",
        parser: Parser | None = None,
        raw=False,
        temperature: Optional[float] = None,
    ) -> str | BaseModel | tuple[BaseModel, str]:
        """
        Handles all asynchronous requests from Anthropic's models.
        Possibilities:
        - pydantic object not provided, input is string -> return string
        - pydantic object provided, input is string -> return pydantic object
        - if raw=True, return a tuple of (pydantic object, raw text)
        Anthropic is quirky about system messsages (The Messages API accepts a top-level "system" parameter, not "system" as an input message role.)
        """
        # Anthropic requires a system variable
        system = ""
        if isinstance(input, str):
            input = [Message(role="user", content=input)]
        elif isinstance(input, list):
            input = input
            # This is anthropic quirk; we remove the system message and set it as a query parameter.
            if input[0].role == "system":
                system = input[0]["content"]
                input = input[1:]
                # Remote "system" role from any messages in input. Another annoying quirk.
                for message in input:
                    if message.role == "system":
                        message.role = "user"
        else:
            raise ValueError(
                f"Input not recognized as a valid input type: {type(input)}: {input}"
            )
        params = {
            "messages": input,
            "model": model,
            "response_model": None if not parser else parser.pydantic_model,
            "max_retries": 0,
            "system": system,
        }
        # set max_tokens based on model
        if model == "claude-3-5-sonnet-20240620":
            params["max_tokens"] = 8192
        else:
            params["max_tokens"] = 8192
        if temperature:
            if temperature < 0 or temperature > 1:
                raise ValueError(
                    "Temperature for Anthropic models needs to be between 0 and 1."
                )
            else:
                params["temperature"] = temperature
        # call our client
        if raw and parser:
            obj, raw_response = (
                await self._client.chat.completions.create_with_completion(**params)
            )
            raw_text = json.dumps(raw_response.content[0].input)
            return obj, raw_text
        elif parser:
            obj = await self._client.chat.completions.create(**params)
            return obj
        # If you are not passing pydantic models, you will get the text response.
        else:
            response = await self._client.chat.completions.create(**params)
            return response.content[0].text
