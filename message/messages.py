from typing import Any, Iterator, Optional
from pydantic import BaseModel, Field
from Chain.message.message import Message

class Messages(BaseModel):
    """
    A Pydantic BaseModel that contains a list of Message objects.
    Behaves like a list through dunder methods while being fully Pydantic-compatible.
    Supports Message, ImageMessage, and AudioMessage objects (all inherit from Message).
    """
    
    messages: list[Message] = Field(default_factory=list, description="List of Message objects (including ImageMessage and AudioMessage)")

    def __init__(self, messages: list[Message] = None, **kwargs):
        """
        Initialize with optional list of messages.

        Args:
            messages: list of Message objects (including ImageMessage and AudioMessage) to initialize with
        """
        if messages is not None:
            super().__init__(messages=messages, **kwargs)
        else:
            super().__init__(**kwargs)

    # List-like interface methods
    def append(self, message: Message) -> None:
        """Add a message to the end of the list."""
        self.messages.append(message)

    def extend(self, messages: list[Message]) -> None:
        """Extend the list with multiple messages."""
        self.messages.extend(messages)

    def insert(self, index: int, message: Message) -> None:
        """Insert a message at the specified index."""
        self.messages.insert(index, message)

    def remove(self, message: Message) -> None:
        """Remove the first occurrence of a message."""
        self.messages.remove(message)

    def pop(self, index: int = -1) -> Message:
        """Remove and return message at index (default last)."""
        return self.messages.pop(index)

    def clear(self) -> None:
        """Remove all messages."""
        self.messages.clear()

    def index(self, message: Message, start: int = 0, stop: Optional[int] = None) -> int:
        """Return the index of the first occurrence of message."""
        if stop is None:
            return self.messages.index(message, start)
        return self.messages.index(message, start, stop)

    def count(self, message: Message) -> int:
        """Return the number of occurrences of message."""
        return self.messages.count(message)

    def reverse(self) -> None:
        """Reverse the messages in place."""
        self.messages.reverse()

    def sort(self, *, key=None, reverse: bool = False) -> None:
        """Sort the messages in place."""
        self.messages.sort(key=key, reverse=reverse)

    def copy(self) -> 'Messages':
        """Return a shallow copy of the Messages object."""
        return Messages(self.messages.copy())

    # Dunder methods for list-like behavior
    def __len__(self) -> int:
        """Return the number of messages."""
        return len(self.messages)

    def __getitem__(self, key):
        """Get message(s) by index or slice."""
        result = self.messages[key]
        if isinstance(key, slice):
            return Messages(result)
        return result

    def __setitem__(self, key, value):
        """Set message(s) by index or slice."""
        self.messages[key] = value

    def __delitem__(self, key):
        """Delete message(s) by index or slice."""
        del self.messages[key]

    def __iter__(self) -> Iterator[Message]:
        """Iterate over messages."""
        return iter(self.messages)

    def __reversed__(self) -> Iterator[Message]:
        """Iterate over messages in reverse order."""
        return reversed(self.messages)

    def __contains__(self, message: Message) -> bool:
        """Check if message is in the list."""
        return message in self.messages

    def __bool__(self) -> bool:
        """Return True if there are any messages."""
        return bool(self.messages)

    def __eq__(self, other) -> bool:
        """Check equality with another Messages object or list."""
        if isinstance(other, Messages):
            return self.messages == other.messages
        elif isinstance(other, list):
            return self.messages == other
        return False

    def __add__(self, other) -> 'Messages':
        """Concatenate with another Messages object or list."""
        if isinstance(other, Messages):
            return Messages(self.messages + other.messages)
        elif isinstance(other, list):
            return Messages(self.messages + other)
        return NotImplemented

    def __iadd__(self, other) -> 'Messages':
        """In-place concatenation with another Messages object or list."""
        if isinstance(other, Messages):
            self.messages.extend(other.messages)
        elif isinstance(other, list):
            self.messages.extend(other)
        else:
            return NotImplemented
        return self

    def __mul__(self, other: int) -> 'Messages':
        """Repeat messages n times."""
        if isinstance(other, int):
            return Messages(self.messages * other)
        return NotImplemented

    def __imul__(self, other: int) -> 'Messages':
        """In-place repeat messages n times."""
        if isinstance(other, int):
            self.messages *= other
        else:
            return NotImplemented
        return self

    # Serialization methods (updated for BaseModel)
    def to_cache_dict(self) -> dict[str, Any]:
        """
        Serialize Messages to cache-friendly dictionary.
        """
        return {
            "messages": [msg.to_cache_dict() for msg in self.messages]
        }

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> "Messages":
        """
        Deserialize Messages from cache dictionary.
        """
        messages_list = []

        for msg_data in data["messages"]:
            # Check if this is a specialized message type
            message_type = msg_data.get("message_type", "Message")

            if message_type == "ImageMessage":
                # Import here to avoid circular imports
                from Chain.message.imagemessage import ImageMessage
                messages_list.append(ImageMessage.from_cache_dict(msg_data))
            elif message_type == "AudioMessage":
                # Import here to avoid circular imports
                from Chain.message.audiomessage import AudioMessage
                messages_list.append(AudioMessage.from_cache_dict(msg_data))
            else:
                # Standard Message
                from Chain.message.message import Message
                messages_list.append(Message.from_cache_dict(msg_data))

        return cls(messages=messages_list)

    # Chain-specific convenience methods
    def add_new(self, role: str, content: str) -> None:
        """
        Create and add a new message to the list.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        from Chain.message.message import Message
        self.append(Message(role=role, content=content))

    def last(self) -> Optional[Message]:
        """
        Get the last message in the list.

        Returns:
            Last Message object or None if empty
        """
        return self.messages[-1] if self.messages else None

    def get_by_role(self, role: str) -> list[Message]:
        """
        Get all messages with a specific role.

        Args:
            role: Role to filter by (user, assistant, system)

        Returns:
            list of messages with the specified role
        """
        return [msg for msg in self.messages if msg.role == role]

    def user_messages(self) -> list[Message]:
        """Get all user messages."""
        return self.get_by_role("user")

    def assistant_messages(self) -> list[Message]:
        """Get all assistant messages."""
        return self.get_by_role("assistant")

    def system_messages(self) -> list[Message]:
        """Get all system messages."""
        return self.get_by_role("system")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return str(self.messages)

    def __str__(self) -> str:
        """
        String representation showing message count and types.
        """
        return self.__repr__()

