"""
Here's boilerplate code for instructor which should be incorporated into my Chain clients.
This will allow a "raw" option for pydantic model answers.
This will be important for ChainCache to serialize and retrieve pydantic models with cloudpickle.
"""

from openai import OpenAI
import instructor
import os
from pydantic import BaseModel


class Frog(BaseModel):
    species: str
    name: str
    no_legs: int


api_key = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=api_key)
client = instructor.from_openai(openai_client)

obj, raw_response = client.chat.completions.create_with_completion(
    model="o3-mini",
    response_model=Frog,
    messages=[{"role": "user", "content": "Create a frog."}],
)

raw_text = raw_response.choices[0].message.tool_calls[0].function.arguments
