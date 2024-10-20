
"""
For Google Gemini models.
"""

from .client import Client
import google.generativeai
import instructor
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()


class OpenAIClient(Client):
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

    def query(
        self, model: str, input: "str | list", pydantic_model: BaseModel = None
    ) -> "str | BaseModel":
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]

        response = self._client.chat.completions.create(
            model=model, response_model=pydantic_model, messages=input
        )

        if pydantic_model:
            return response
        else:
            return response.choices[0].message.content

    async def query_async(
        self, model: str, input: "str | list", pydantic_model: "BaseModel" = None
    ) -> "BaseModel | str":
        # Implement asynchronous query logic here
        # This would be similar to the synchronous version but using async calls
        pass



















dotenv.load_dotenv(dotenv_path=env_path)
api_keys = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
}

genai = importlib.import_module("google.generativeai")
instructor = importlib.import_module("instructor")
genai.configure(api_key=api_keys["GOOGLE_API_KEY"])


    def _query_google(
        self,
        input: Union[str, list],
        verbose: bool = True,
        pydantic_model: Optional[Type[BaseModel]] = None,
    ) -> Union[BaseModel, str]:
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
        gemini_client_model = instructor.from_gemini(
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
