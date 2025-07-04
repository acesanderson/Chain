"""
For Perplexity models.
NOTE: these use standard OpenAI SDK, the only difference in processing is that the response object has an extra 'citations' field.
You want both the 'content' and 'citations' fields from the response object.
Perplexity inputs the footnotes within the content.
For this reason, we define a Pydantic class as our framework is fine with BaseModels as a response. You can still access it as a string and get the content if needed. Citations can be access by choice.
"""

from Chain.model.clients.client import Client, Usage
from Chain.model.clients.load_env import load_env
from Chain.request.request import Request
from openai import OpenAI, AsyncOpenAI, Stream
from openai.types.chat.chat_completion import ChatCompletion
from typing import Optional
from pydantic import BaseModel
import instructor, tiktoken


class PerplexityCitation(BaseModel):
    title: str
    url: str
    date: Optional[str]


class PerplexityContent(BaseModel):
    text: str
    citations: list[PerplexityCitation]


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
        request: Request,
    ) -> tuple:
        structured_response = None
        if request.response_model is not None:
            # We want the raw response from OpenAI, so we use `create_with_completion`
            structured_response, result = (
                self._client.chat.completions.create_with_completion(
                    **request.to_perplexity()
                )
            )
        else:
            # Use the standard completion method
            result = self._client.chat.completions.create(**request.to_perplexity())
        # Capture usage
        usage = Usage(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
        )
        if structured_response is not None:
            # If we have a structured response, return it along with usage
            return structured_response, usage
        if isinstance(result, ChatCompletion):
            # Construct a PerplexityContent object from the response
            citations = result.search_results
            assert isinstance(citations, list) and all(
                [isinstance(citation, dict) for citation in citations]
            ), "Citations should be a list of dicts"
            citations = [PerplexityCitation(**citation) for citation in citations]
            content = PerplexityContent(
                text=result.choices[0].message.content, citations=citations
            )
            return content, usage
        if isinstance(result, BaseModel):
            return result, usage
        else:
            raise ValueError("Unexpected result type: {}".format(type(result)))


class PerplexityClientAsync(PerplexityClient):
    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        perplexity_client = AsyncOpenAI(
            api_key=self._get_api_key(), base_url="https://api.perplexity.ai"
        )
        return instructor.from_perplexity(perplexity_client)

    async def query(
        self,
        request: Request,
    ) -> tuple:
        result = await self._client.chat.completions.create(**request.to_perplexity())
        # Capture usage
        usage = Usage(
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
        )
        if isinstance(result, ChatCompletion):
            # Construct a PerplexityContent object from the response
            citations = result.search_results
            assert isinstance(citations, list) and all(
                [isinstance(citation, dict) for citation in citations]
            ), "Citations should be a list of dicts"
            citations = [PerplexityCitation(**citation) for citation in citations]
            content = PerplexityContent(
                text=result.choices[0].message.content, citations=citations
            )
            return content, usage
        if isinstance(result, BaseModel):
            return result, usage
        else:
            raise ValueError("Unexpected result type: {}".format(type(result)))
