"""
Client subclass for Anthropic models.
"""

from .client import Client
from ...message.message import Message
from Chain.model.clients.load_env import load_env
from anthropic import Anthropic
import instructor
from pydantic import BaseModel
import os


class AnthropicClient(Client):
    def __init__(self):
        self._client = self._initialize_client()

    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        api_key = load_env("ANTHROPIC_API_KEY")
        anthropic_client = Anthropic(api_key=self._get_api_key())
        return instructor.from_anthropic(anthropic_client)

    def _get_api_key(self):
        if os.getenv("ANTHROPIC_API_KEY") is None:
            raise ValueError("No ANTHROPIC_API_KEY found in environment variables")
        else:
            return os.getenv("ANTHROPIC_API_KEY")

    def query(
        self, model: str, input: "str | list", pydantic_model: BaseModel = None
    ) -> "str | BaseModel":
        """
        Handles all synchronous requests from Anthropic's models.
        Possibilities:
        - pydantic object not provided, input is string -> return string
        - pydantic object provided, input is string -> return pydantic object
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

        # set max_tokens based on model
        if model == "claude-3-5-sonnet-20240620":
            max_tokens = 8192
        else:
            max_tokens = 4096
        # call our client
        response = self._client.chat.completions.create(
            # model = self.model,
            model=model,
            max_tokens=max_tokens,
            max_retries=0,
            system=system,  # This is the system message we grabbed earlier
            messages=input,
            # Splatting our pydantic models to dict
            response_model=pydantic_model,
        )
        # only two possibilities here
        if pydantic_model:
            return response
        else:
            return response.content[0].text

    async def query_async(
        self, model: str, input: "str | list", pydantic_model: "BaseModel" = None
    ) -> "BaseModel | str":
        # Implement asynchronous query logic here
        # This would be similar to the synchronous version but using async calls
        pass
