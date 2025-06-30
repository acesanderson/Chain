"""
Message is the default message type recognized as industry standard (role + content).
Our Message class is inherited from specialized types like TextMessage, AudioMessage, ImageMessage, etc.
"""

from abc import abstractmethod, ABC
from Chain.logging.logging_config import get_logger
from pydantic import BaseModel
from typing import Literal, Any

logger = get_logger(__name__)

# Useful type aliases
Role = Literal["user", "assistant", "system"]
MessageType = Literal["text", "audio", "image"]


class Message(BaseModel, ABC):
    """Base message class - abstract with required Pydantic functionality"""
    message_type: MessageType
    role: Role
    content: Any

    @abstractmethod
    def to_cache_dict(self) -> dict:
        """
        Serializes the message to a dictionary for caching.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
   
    @classmethod
    def from_cache_dict(cls, cache_dict: dict) -> "Message":
        """Parse JSON with lazy imports to avoid circular dependencies"""
        message_type = cache_dict["message_type"]
        
        # Import only when needed
        if message_type == "text":
            from Chain.message.textmessage import TextMessage
            return TextMessage.from_cache_dict(cache_dict)
        elif message_type == "audio":
            from Chain.message.audiomessage import AudioMessage  
            return AudioMessage.from_cache_dict(cache_dict)
        elif message_type == "image":
            from Chain.message.imagemessage import ImageMessage
            return ImageMessage.from_cache_dict(cache_dict)
        else:
            raise ValueError(f"Unknown message type: {message_type}")

