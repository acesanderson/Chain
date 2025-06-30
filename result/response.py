"""
A successful Result.
"""

from Chain.message.message import Message
from Chain.message.messages import Messages
from Chain.model.params.params import Params
from Chain.logging.logging_config import get_logger
from Chain.progress.display_mixins import (
    RichDisplayResponseMixin,
    PlainDisplayResponseMixin,
)
from pydantic import BaseModel
from typing import Optional, Any, Dict
from datetime import datetime

logger = get_logger(__name__)


class Response(BaseModel, RichDisplayResponseMixin, PlainDisplayResponseMixin):
    """
    Our class for a successful Result.
    We mixin display modules so that Responses can to_plain, to_rich as part of our progress tracking / verbosity system.
    """

    # Core attributes
    messages: Messages
    params: Params
    input_tokens: int
    output_tokens: int
    duration: Optional[float]

    # Initialization attributes
    timestamp: Optional[str] = None

    def model_post_init(self, __context):
        """
        Post-initialization hook to set the timestamp and ensure Messages type.
        """
        super().model_post_init(__context)  # Call the mixin's post_init first!
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        # Ensure messages is always a Messages object
        if not isinstance(self.messages, Messages):
            self.messages = Messages(self.messages)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_cache_dict(self) -> Dict[str, Any]:
        """
        Serialize Response to cache-friendly dictionary.
        """
        # return {
        #     "messages": self._serialize_messages(),
        #     "params": self._serialize_params(),
        #     "duration": self.duration,
        #     "timestamp": self.timestamp,
        # }
        pass

    @classmethod
    def from_cache_dict(cls, data: Dict[str, Any]) -> "Response":
        """
        Deserialize Response from cache dictionary.
        """
        # from Chain.message.messages import Messages  # Import at the top
        #
        # # Deserialize messages
        # messages_list = cls._deserialize_messages(data["messages"])
        # messages = Messages(messages_list)  # Wrap in Messages object
        #
        # # Deserialize params
        # params = cls._deserialize_params(data["params"])
        #
        # # Create instance
        # instance = cls(
        #     messages=messages,  # Now it's a Messages object
        #     params=params,
        #     duration=data["duration"],
        #     timestamp=data["timestamp"],
        # )
        #
        # return instance
        pass

    @property
    def prompt(self) -> str | None:
        """
        This is the last user message.
        """
        if self.messages and isinstance(self.messages[-1], Message):
            return self.messages[-1].content
        return None

    @property
    def message(self) -> Message:
        """
        Return last message (good for messagestore handling).
        """
        if self.messages and isinstance(self.messages[-1], Message):
            return self.messages[-1]
        raise ValueError("No messages available in the response.")

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
