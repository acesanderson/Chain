"""
Simple class for responses.
A string isn't enough for debugging purposes; you want to be able to see the prompt, for example.
Should read content as string when invoked as such.
"""

from Chain.message.message import Message
from Chain.message.imagemessage import ImageMessage
from pydantic import BaseModel


class Response(BaseModel):
    content: str | BaseModel | list[BaseModel]
    status: str
    prompt: str | None
    model: str
    duration: float | None
    messages: list[Message | ImageMessage]

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
        if isinstance(self.content, BaseModel):
            return str(self.content)
        elif isinstance(self.content, list):
            return str(self.content)
        else:
            return self.content

    def __len__(self):
        """
        We want to be able to check the length of the content.
        """
        return len(self.__str__())
