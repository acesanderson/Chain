"""
Very lightweight pydantic class, used to validate messages.
"""

from pydantic import BaseModel
from ..prompt.prompt import Prompt


class Message(BaseModel):
    """
    Industry standard, more or less, for messaging with LLMs.
    System roles can have some weirdness (like with Anthropic), but role/content is standard.
    """

    role: str
    content: str


class Messages(BaseModel):
    """
    Wrapper for a list of Message objects.
    """

    messages: list[Message]


# Some helpful functions


def create_messages(system_prompt: str, input_variables=None) -> list[dict]:
    """
    Takes a system prompt object (Prompt()) or a string, an optional input object, and returns a list of messages.
    """
    if isinstance(system_prompt, str):
        system_prompt = Prompt(system_prompt)
    if input_variables:
        messages = [
            Message(
                role="system",
                content=system_prompt.render(input_variables=input_variables),
            )
        ]
    else:
        messages = [Message(role="system", content=system_prompt.prompt_string)]
    return messages
