"""
Simple class for responses.
A string isn't enough for debugging purposes; you want to be able to see the prompt, for example.
Should read content as string when invoked as such.
"""

from Chain.message.message import Message
from pydantic import BaseModel


class Response:
    def __init__(
        self,
        content: str | BaseModel = "",
        status="N/A",
        prompt: str | None = "",
        model="",
        duration=0.0,
        messages: list = [],
    ):
        self.content: str | BaseModel = (
            content  # This is the content of the last message (the text completion)
        )
        self.status: str = (
            status  # This is the status of the response (e.g. "success", "error")
        )
        self.prompt: str | None = (
            prompt  # This is the last prompt that was sent to the model (content of the last user message)
        )
        self.model: str = model  # This is the model name (e.g. "gpt-4o")
        self.duration: float = duration  # This is how long the request took.
        self.messages: list[Message] = (
            messages  # This is the message history; it's at least the last two messages (user and model)
        )

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
