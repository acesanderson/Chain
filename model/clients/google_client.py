"""
For Google Gemini models.
"""

from .client import Client
import google.generativeai as genai
import instructor
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()


class GeminiClient(Client):
    def __init__(self):
        self._client = self._initialize_client()

    def _initialize_client(self):
        """
        We use the Instructor library by default, as this offers a great interface for doing function calling and working with pydantic objects.
        """
        genai.configure(api_key=api_keys["GOOGLE_API_KEY"])
        return instructor.from_gemini(genai)

    def _get_api_key(self):
        if os.getenv("GOOGLE_API_KEY") is None:
            raise ValueError("No Google API key found in environment variables")
        else:
            return os.getenv("GOOGLE_API_KEY")

    def _query_google(
        self, model: str, input: "str | list", pydantic_model: BaseModel = None
    ) -> "str | BaseModel":
        """
        Google doesn't take message objects, apparently. (or it's buried in their documentation)
        """
        if verbose:
            print(f"Model: {self.model}   Query: " + self.pretty(str(input)))
        if isinstance(input, str):
            input = input
        elif is_messages_object(input):
            input = input["content"]
        else:
            raise ValueError(
                f"Input not recognized as a valid input type: {type:input}: {input}"
            )
        # call our client
        gemini_client_model = self._client(
            client=self._client.GenerativeModel(
                model_name=self.model,
            ),
            mode=instructor.Mode.GEMINI_JSON,
        )
        response = gemini_client_model.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": input,
                }
            ],
            response_model=pydantic_model,
        )
        if pydantic_model:
            return response
        else:
            return response.candidates[0].content.parts[0].text
