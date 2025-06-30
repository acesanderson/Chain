"""
A successful Result.
"""

from Chain.message.message import Message
from Chain.message.messages import Messages
from Chain.model.params.params import Params
from Chain.logging.logging_config import get_logger
from Chain.progress.display_mixins import RichDisplayResponseMixin, PlainDisplayResponseMixin
from Chain.cache.cacheable import CacheableMixin
from pydantic import BaseModel
from typing import Optional, Any, Dict
from datetime import datetime

logger = get_logger(__name__)


class Response(BaseModel, CacheableMixin, RichDisplayResponseMixin, PlainDisplayResponseMixin):
    """
    Our class for a successful Result.
    We mixin display modules so that Responses can to_plain, to_rich as part of our progress tracking / verbosity system.
    """
    # Core attributes
    messages: Messages
    params: Params
    duration: Optional[float]

    # Initialization attributes
    timestamp: Optional[str] = None

    def model_post_init(self, __context):
        """
        Post-initialization hook to set the timestamp and ensure Messages type.
        """
        super().model_post_init(__context) # Call the mixin's post_init first!
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        # Ensure messages is always a Messages object
        if not isinstance(self.messages, Messages):
            self.messages = Messages(self.messages)

    def _serialize_messages(self) -> list[Dict[str, Any]]:
        """
        Serialize the messages list, handling different message types.
        """
        serialized_messages = []
        for message in self.messages:
            if hasattr(message, "to_cache_dict"):
                # Message has its own serialization method
                serialized_messages.append(message.to_cache_dict())
            else:
                # Fallback for basic Message objects
                serialized_messages.append(
                    {
                        "message_type": "Message",
                        "role": (
                            message.role.value
                            if hasattr(message.role, "value")
                            else message.role
                        ),
                        "content": {"type": "string", "data": str(message.content)},
                    }
                )
        return serialized_messages

    @classmethod
    def _deserialize_messages(
        cls, messages_data: list[Dict[str, Any]]
    ) -> list[Message]:
        """
        Deserialize messages list, handling different message types.
        """
        messages = []
        for msg_data in messages_data:
            message_type = msg_data.get("message_type", "Message")

            if message_type == "ImageMessage":
                from Chain.message.imagemessage import ImageMessage

                messages.append(ImageMessage.from_cache_dict(msg_data))
            elif message_type == "AudioMessage":
                from Chain.message.audiomessage import AudioMessage

                messages.append(AudioMessage.from_cache_dict(msg_data))
            else:
                # Standard Message
                messages.append(Message.from_cache_dict(msg_data))

        return messages

    def _serialize_params(self) -> Dict[str, Any]:
        """
        Serialize the Params object using its own serialization method.
        """
        return self.params.to_cache_dict()

    @classmethod
    def _deserialize_params(cls, params_data: Dict[str, Any]) -> Params:
        """
        Deserialize the Params object using its own deserialization method.
        """
        from Chain.model.params.params import Params

        return Params.from_cache_dict(params_data)

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
