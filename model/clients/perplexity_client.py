"""
For Perplexity models.
NOTE: these use standard OpenAI SDK, the only difference in processing is that the response object has an extra 'citations' field.
You want both the 'content' and 'citations' fields from the response object.
Perplexity inputs the footnotes within the content.
For this reason, we define a Pydantic class as our framework is fine with BaseModels as a response. You can still access it as a string and get the content if needed. Citations can be access by choice.
"""

from Chain.model.clients.client import Client
from Chain.model.clients.load_env import load_env
from openai import OpenAI
import instructor
from pydantic import BaseModel
import tiktoken
from typing import Optional


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
        return instructor.from_openai(perplexity_client)

    def query(
        self,
        model: str,
        input: "str | list",
        pydantic_model: BaseModel | None = None,
        raw=False,
        temperature: Optional[float] = None,
    ) -> str | BaseModel | tuple[BaseModel, str]:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        """
        Don't bother with function calling with Perplexity -- it already structures its responses (i.e. citations).
        """
        # call our client
        response = self._client.chat.completions.create(
            model=model,
            response_model=None,
            messages=input,
        )
        # This is the custom handling for Perplexity -- just put citations in xml tag.
        perplexity_response = (
            response.choices[0].message.content
            + "\n<citations>\n"
            + "\n".join(response.citations)
            + "\n</citations>\n"
        )
        return perplexity_response
