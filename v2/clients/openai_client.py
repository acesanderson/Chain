from openai import OpenAI
import instructor
from typing import Union, Type, Optional
from pydantic import BaseModel
from .client import Client

class OpenAIClient(Client):
    def __init__(self):
        self._client = self._initialize_client()

    def _initialize_client(self):
        openai_client = OpenAI(api_key=self._get_api_key())
        return instructor.from_openai(openai_client)

    def _get_api_key(self):
        # Implement your logic to securely retrieve the API key
        # This could involve environment variables, a config file, or a secure key management system
        pass

    def query(self, model: str, input: Union[str, list], pydantic_model: Optional[Type[BaseModel]] = None) -> Union[BaseModel, str]:
        if isinstance(input, str):
            input = [{"role": "user", "content": input}]
        
        response = self._client.chat.completions.create(
            model=model,
            response_model=pydantic_model,
            messages=input
        )

        if pydantic_model:
            return response
        else:
            return response.choices[0].message.content

    async def query_async(self, model: str, input: Union[str, list], pydantic_model: Optional[Type[BaseModel]] = None) -> Union[BaseModel, str]:
        # Implement asynchronous query logic here
        # This would be similar to the synchronous version but using async calls
        pass