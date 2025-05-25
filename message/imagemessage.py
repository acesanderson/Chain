"""
There are two basic message formats:
1. OpenAI (applicable to ollama, groq, gemini, etc.)
2. Anthropic (the one hold out)

We have a basic ImageMessage class, which is a wrapper for the OpenAI and Anthropic formats.
"""

from pydantic import BaseModel, Field
from Chain.message.message import Message
import re

# Map PIL formats to MIME types
format_to_mime = {
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


def is_base64_simple(s):
    """
    Simple validation for base64 strings.
    """
    return bool(re.match(r"^[A-Za-z0-9+/]*={0,2}$", s)) and len(s) % 4 == 0


# Anthropic schema
"""
"content": [
    {
        "type": "image", 
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": image_base64
        }
    },
    {
        "type": "text",
        "text": "What's in this image?"
    }
]
"""


# Anthropic-specific message classes
class AnthropicTextContent(BaseModel):
    type: str = "text"
    text: str


class AnthropicImageSource(BaseModel):
    type: str = "base64"
    media_type: str
    data: str


class AnthropicImageContent(BaseModel):
    type: str = "image"
    source: AnthropicImageSource


class AnthropicImageMessage(Message):
    """
    ImageMessage should have a single ImageContent and a single TextContent object.

        role: str
        content: list[ImageContent | TextContent]
    """

    role: str
    content: list[AnthropicImageContent | AnthropicTextContent]  # type: ignore


# OpenAI-specific message classes
"""
Role is the same; but content is different.
        "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"},
        ],
"""


class OpenAITextContent(BaseModel):
    type: str = "input_text"
    text: str = Field(description="The text content of the message, i.e. the prompt.")


class OpenAIImageContent(BaseModel):
    """
    image_url is a special string, composed of:
    data:{mime_type};base64,{b64_image}
    """

    type: str = "input_image"
    image_url: str = Field(description="The image URL, i.e. the base64-encoded image.")


class OpenAIImageMessage(Message):
    """
    ImageMessage should have a single ImageContent and a single TextContent object.

        role: str
        content: list[ImageContent | TextContent]
    """

    role: str
    content: list[OpenAIImageContent | OpenAITextContent]  # type: ignore


# Our base ImageMessage class, with a factory method to convert to OpenAI or Anthropic format.
class ImageMessage(BaseModel):
    """
    ImageMessage should have a single ImageContent and a single TextContent object.

        role: str
        text_content: str
        image_content: str
        mime_type: str
    """

    role: str = Field(
        description="The role of the message, e.g. 'user', 'assistant', or 'system'."
    )
    text_content: str = Field(
        description="The text content of the message, i.e. the prompt."
    )
    image_content: str = Field(description="The base64-encoded image.")
    mime_type: str = Field(
        description="The MIME type of the image, e.g. 'image/jpeg', 'image/png'."
    )

    def __init__(self, **data):
        super().__init__(**data)
        # Validate the MIME type
        if self.mime_type not in format_to_mime.values():
            raise ValueError(f"Unsupported MIME type: {self.mime_type}")
        # Validate the image content
        if not is_base64_simple(self.image_content):
            raise ValueError("Image content must be a base64-encoded string.")

    def __repr__(self):
        return f"ImageMessage(role={self.role}, text_content={self.text_content}, image_content={self.image_content:<.10}..., mime_type={self.mime_type})"

    def to_anthropic(self) -> AnthropicImageMessage:
        """
        Converts the ImageMessage to the Anthropic format.
        """
        image_source = AnthropicImageSource(
            type="base64", media_type=self.mime_type, data=self.image_content
        )
        image_content = AnthropicImageContent(source=image_source)
        text_content = AnthropicTextContent(text=self.text_content)
        return AnthropicImageMessage(
            role=self.role, content=[image_content, text_content]
        )

    def to_openai(self) -> OpenAIImageMessage:
        """
        Converts the ImageMessage to the OpenAI format.
        """
        image_url = f"data:{self.mime_type};base64,{self.image_content}"
        image_content = OpenAIImageContent(image_url=image_url)
        text_content = OpenAITextContent(text=self.text_content)
        return OpenAIImageMessage(role=self.role, content=[image_content, text_content])
