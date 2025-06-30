"""
Message is the default message type recognized as industry standard (role + content).
Our Message class is inherited from specialized types like AudioMessage, ImageMessage, etc.
We define list[Message] as Messages in a parallel file, this class can handle the serialization / deserialization needed for message historys, api calls, and caching.
"""

from Chain.prompt.prompt import Prompt
from Chain.logging.logging_config import get_logger
from Chain.cache.cacheable import CacheableMixin
from enum import Enum
from pydantic import BaseModel
from typing import Any

logger = get_logger(__name__)


class Role(Enum):
    """
    Enum for the role of the message.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel, CacheableMixin):
    """
    Industry standard, more or less, for messaging with LLMs.
    System roles can have some weirdness (like with Anthropic), but role/content is standard.
    """

    role: str | Role
    content: Any

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
