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


# Our custom response class
class PerplexityResponse(BaseModel):
    content: str
    citations: list[str]

    def __str__(self) -> str:
        return self.content


class PerplexityClient(Client):
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

    def _get_api_key(self):
        api_key = load_env("PERPLEXITY_API_KEY")
        return api_key

    def query(
        self, model: str, input: "str | list", pydantic_model: BaseModel = None
    ) -> "str | BaseModel":
        """
        Handles all synchronous requests from Perplexity's models.
        Possibilities:
        - pydantic object not provided, input is string -> return string
        - pydantic object provided, input is string -> return pydantic object
        Google doesn't take message objects, apparently. (or it's buried in their documentation)
        """
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        # call our client
        response = self._client.chat.completions.create(
            model=model,
            response_model=pydantic_model,
            messages=input,
        )
        # This is the custom handling for Perplexity -- since we want message + citations, we return our pydantic object.
        return PerplexityResponse(
            content=response.choices[0].message.content,
            citations=response.citations,
        )
