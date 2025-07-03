"""
A successful Result.
"""

from Chain.message.message import Message
from Chain.message.messages import Messages
from Chain.model.params.params import Params
from Chain.logs.logging_config import get_logger
from Chain.progress.display_mixins import (
    RichDisplayResponseMixin,
    PlainDisplayResponseMixin,
)
from pydantic import BaseModel, Field
from datetime import datetime

logger = get_logger(__name__)


class Response(BaseModel, RichDisplayResponseMixin, PlainDisplayResponseMixin):
    """
    Our class for a successful Result.
    We mixin display modules so that Responses can to_plain, to_rich as part of our progress tracking / verbosity system.
    """

    # Core attributes
    message: Message
    params: Params
    input_tokens: int
    output_tokens: int
    duration: float

    # Initialization attributes
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp of the response creation",
    )

    def to_cache_dict(self) -> dict:
        """
        Serialize Response to cache-friendly dictionary.
        """
        return {
            "message": self.message.to_cache_dict(),
            "params": self.params.to_cache_dict(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "duration": self.duration,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_cache_dict(cls, cache_dict: dict) -> "Response":
        """
        Deserialize Response from cache dictionary.
        """
        return cls(
            message=Message.from_cache_dict(cache_dict["message"]),
            params=Params.from_cache_dict(cache_dict["params"]),
            input_tokens=cache_dict["input_tokens"],
            output_tokens=cache_dict["output_tokens"],
            duration=cache_dict["duration"],
            timestamp=cache_dict.get("timestamp", datetime.now().isoformat()),
        )

    @property
    def prompt(self) -> str | None:
        """
        This is the last user message.
        """
        return self.params.messages[-1].content

    @property
    def messages(self) -> Messages:
        return Messages(messages=self.params.messages + [self.message])

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def content(self) -> str | BaseModel | list[BaseModel] | list[str]:
        """
        This is the last assistant message content.
        """
        return self.message.content

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
        """
        return str(self.content) if self.content is not None else ""

    def __len__(self):
        """
        We want to be able to check the length of the content.
        """
        return len(self.__str__())
