"""
For Perplexity models.
NOTE: these use standard OpenAI SDK, the only difference in processing is that the response object has an extra 'citations' field.
You want both the 'content' and 'citations' fields from the response object.
Perplexity inputs the footnotes within the content.
For this reason, we define a Pydantic class as our framework is fine with BaseModels as a response. You can still access it as a string and get the content if needed. Citations can be access by choice.
"""

from Chain.model.clients.client import Client
from Chain.model.clients.load_env import load_env
from Chain.message.message import Message
from Chain.message.imagemessage import ImageMessage
from Chain.message.audiomessage import AudioMessage
from Chain.parser.parser import Parser
from openai import OpenAI
import instructor, tiktoken
from pydantic import BaseModel
from typing import Optional, Any


class PerplexityClient(Client):
    """
    This is a base class; we have two subclasses: OpenAIClientSync and OpenAIClientAsync.
    Don't import this.
    """

    def __init__(self):
        self._client = self._initialize_client()

    def _initialize_client(self):
        """
        Logic for this is unique to each client (sync / async).
        """
        pass

    def _get_api_key(self):
        api_key = load_env("PERPLEXITY_API_KEY")
        return api_key

    def tokenize(self, model: str, text: str) -> int:
        """
        Return the token count for a string, per model's tokenization function.
        cl100k_base is good enough for Perplexity, per Perplexity documentation.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(text))
        return token_count


class PerplexityClientSync(PerplexityClient):
    def __init__(self):
        self._client = self._initialize_client()

    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        perplexity_client = OpenAI(
            api_key=self._get_api_key(), base_url="https://api.perplexity.ai"
        )
        return instructor.from_perplexity(perplexity_client)

    def query(
        self,
        model: str,
        input: str | list | Message | ImageMessage,
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
                raise ValueError("Perplexity does not support audio messages yet.")
            if isinstance(message, Message):
                converted_messages.append(message.model_dump())
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        # Construct params
        # from Chain.model.models.ModelSpec import ModelSpec

        params = {
            "model": model,
            "messages": converted_messages,
            "response_model": parser.to_perplexity() if parser else None,
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
            try:
                obj, raw_response = (
                    self._client.chat.completions.create_with_completion(**params)
                )
                raw_text = (
                    raw_response.choices[0].message.tool_calls[0].function.arguments
                )
                return obj, raw_text
            except AttributeError:
                # Fallback: get the structured response and use model_dump_json()
                obj = self._client.chat.completions.create(**params)
                if isinstance(obj, BaseModel):
                    raw_text = (
                        obj.model_dump_json()
                    )  # This gives you the JSON string for caching
                elif isinstance(obj, list):
                    from pydantic import TypeAdapter

                    # Create a TypeAdapter for the list type
                    adapter = TypeAdapter(list[Any])
                    # Convert list to JSON string
                    raw_text = adapter.dump_json(obj).decode("utf-8")
                return obj, raw_text

        # Default behavior is to return only the pydantic model.
        elif parser:
            obj = self._client.chat.completions.create(**params)
            return obj
        # If you are not passing pydantic models, you will get the text response.
        else:
            response = self._client.chat.completions.create(**params)
            return response.choices[0].message
