"""
A successful Result.
"""

from Chain.message.message import Message
from Chain.model.params.params import Params
from pydantic import BaseModel
from typing import Optional, Any

class Response(BaseModel):
    messages: list[Message]
    params: Params
    duration: Optional[float]

    @property
    def prompt(self) -> str | None:
        """
        This is the last user message.
        """
        if self.messages and isinstance(self.messages[-1], Message):
            return self.messages[-1].content
        return None

    @property
    def content(self) -> Any:
        """
        This is the last assistant message content.
        """
        if self.messages and isinstance(self.messages[-1], Message):
            return self.messages[-1].content
        return None

    @property
    def model(self) -> str:
        """
        This is the model used for the response.
        """
        return self.params.model

    def __repr__(self):
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"

    def __str__(self):
        """
        We want to pass as string when possible.
        Allow json objects (dict) to be pretty printed.
        Not sure what this does if we have pydantic objects.
        """
        return str(self.content) if self.content is not None else ""

    def __len__(self):
        """
        We want to be able to check the length of the content.
        """
        return len(self.__str__())
