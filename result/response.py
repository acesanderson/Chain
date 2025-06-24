"""
A successful Result.
"""

from Chain.message.message import Message
from Chain.message.messages import Messages
from Chain.model.params.params import Params
from pydantic import BaseModel
from typing import Optional, Any, Dict
from datetime import datetime


class Response(BaseModel):
    # Core attributes
    messages: Messages
    params: Params
    duration: Optional[float]

    # Initialization attributes
    timestamp: Optional[str] = None

    def model_post_init(self, __context__):
        """
        Post-initialization hook to set the timestamp and ensure Messages type.
        """
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        
        # Ensure messages is always a Messages object
        if not isinstance(self.messages, Messages):
            self.messages = Messages(self.messages)


    def to_cache_dict(self) -> Dict[str, Any]:
        """
        Serialize Response to cache-friendly dictionary.
        """
        return {
            "messages": self._serialize_messages(),
            "params": self._serialize_params(),
            "duration": self.duration,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_cache_dict(cls, data: Dict[str, Any]) -> "Response":
        """
        Deserialize Response from cache dictionary.
        """
        from Chain.message.messages import Messages  # Import at the top
        
        # Deserialize messages
        messages_list = cls._deserialize_messages(data["messages"])
        messages = Messages(messages_list)  # Wrap in Messages object
        
        # Deserialize params
        params = cls._deserialize_params(data["params"])

        # Create instance
        instance = cls(
            messages=messages,  # Now it's a Messages object
            params=params,
            duration=data["duration"],
            timestamp=data["timestamp"]
        )

        return instance

    def _serialize_messages(self) -> list[Dict[str, Any]]:
        """
        Serialize the messages list, handling different message types.
        """
        serialized_messages = []
        for message in self.messages:
            if hasattr(message, 'to_cache_dict'):
                # Message has its own serialization method
                serialized_messages.append(message.to_cache_dict())
            else:
                # Fallback for basic Message objects
                serialized_messages.append({
                    "message_type": "Message",
                    "role": message.role.value if hasattr(message.role, 'value') else message.role,
                    "content": {
                        "type": "string",
                        "data": str(message.content)
                    }
                })
        return serialized_messages

    @classmethod
    def _deserialize_messages(cls, messages_data: list[Dict[str, Any]]) -> list[Message]:
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
