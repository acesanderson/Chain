"""
Very lightweight pydantic class, used to validate messages.
"""

from pydantic import BaseModel
from Chain.prompt.prompt import Prompt
from enum import Enum


class Role(Enum):
    """
    Enum for the role of the message.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """
    Industry standard, more or less, for messaging with LLMs.
    System roles can have some weirdness (like with Anthropic), but role/content is standard.
    """

    role: str | Role
    content: str | BaseModel | list[BaseModel]

    def __str__(self):
        """
        Returns the message in a human-readable format.
        """
        return f"{self.role}: {self.content}"

    def __getitem__(self, key):
        """
        Allows for dictionary-style access to the object.
        """
        return getattr(self, key)


class Messages(BaseModel):
    """
    Wrapper for a list of Message objects.
    """

    messages: list[Message]


# Some helpful functions


def create_system_message(
    system_prompt: str | Prompt, input_variables=None
) -> list[Message]:
    """
    Takes a system prompt object (Prompt()) or a string, an optional input object, and returns a Message object.
    """
    if isinstance(system_prompt, str):
        system_prompt = Prompt(system_prompt)
    if input_variables:
        system_message = [
            Message(
                role="system",
                content=system_prompt.render(input_variables=input_variables),
            )
        ]
    else:
        system_message = [Message(role="system", content=system_prompt.prompt_string)]
    return system_message
