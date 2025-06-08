"""
There are two basic message formats:
1. OpenAI (applicable to ollama, groq, gemini, etc.)
2. Anthropic (the one hold out)

We have a basic ImageMessage class, which is a wrapper for the OpenAI and Anthropic formats.
"""

from pydantic import BaseModel, Field
from Chain.message.message import Message
from pathlib import Path
import re
import base64
from PIL import Image
import io

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
OpenAI expects this structure:
{
    "content": [
        {"type": "text", "text": prompt},
        {
            "type": "image_url", 
            "image_url": {
                "url": "data:image/png;base64,{b64_image}"
            }
        }
    ]
}
"""


class OpenAITextContent(BaseModel):
    type: str = "text"  # Changed from "input_text"
    text: str = Field(description="The text content of the message, i.e. the prompt.")


class OpenAIImageUrl(BaseModel):
    """Nested object for OpenAI image URL structure"""

    url: str = Field(description="The data URL with base64 image")


class OpenAIImageContent(BaseModel):
    """
    OpenAI requires image_url to be an object, not a string
    """

    type: str = "image_url"
    image_url: OpenAIImageUrl = Field(description="The image URL object")


class OpenAIImageMessage(Message):
    """
    ImageMessage should have a single ImageContent and a single TextContent object.

        role: str
        content: list[OpenAIImageContent | OpenAITextContent]
    """

    role: str
    content: list[OpenAIImageContent | OpenAITextContent]  # type: ignore


# Our base ImageMessage class, with a factory method to convert to OpenAI or Anthropic format.
class ImageMessage(Message):
    """
    ImageMessage should have a single ImageContent and a single TextContent object.

        role: str
        text_content: str
        image_content: str
        mime_type: str

    You can splat it to an OpenAI or Anthropic message; with the to_openai() and to_anthropic() methods.
    Model dump into the API query.
    """

    content: list[BaseModel] = Field(default=None)
    text_content: str = Field(
        description="The text content of the message, i.e. the prompt."
    )
    image_content: str = Field(description="The base64-encoded image.")
    mime_type: str = Field(
        description="The MIME type of the image, e.g. 'image/jpeg', 'image/png'."
    )

    def model_post_init(self, __context) -> None:
        """Called after model initialization to construct content field."""
        self.content = [self.image_content, self.text_content]

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
        # Create the nested URL object
        image_url_obj = OpenAIImageUrl(
            url=f"data:{self.mime_type};base64,{self.image_content}"
        )

        # Create image content with the nested object
        image_content = OpenAIImageContent(image_url=image_url_obj)

        # Create text content
        text_content = OpenAITextContent(text=self.text_content)

        return OpenAIImageMessage(
            role=self.role,
            content=[text_content, image_content],  # Note: text first, then image
        )


# Helper functions
def image_to_base64(file_path):
    """
    Simple version - load any image and convert to base64
    """
    with Image.open(file_path) as img:
        # Get actual format
        img_format = img.format.lower()

        # Convert to RGB if needed (for JPEG compatibility)
        if img.mode in ("RGBA", "LA", "P") and img_format in ["jpeg", "jpg"]:
            img = img.convert("RGB")

        # Save to buffer
        buffer = io.BytesIO()
        save_format = "JPEG" if img_format in ["jpeg", "jpg"] else img_format.upper()
        img.save(buffer, format=save_format)

        # Get base64
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Determine MIME type
        mime_map = {
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime_type = mime_map.get(img_format, "image/jpeg")

        return base64_data, mime_type


def create_image_message(image_file_path: str | Path, prompt_str: str) -> ImageMessage:
    """
    Function to generate an image message from an image file path and a text prompt.
    """
    image_base64, mime_type = image_to_base64(image_file_path)
    imagemessage = ImageMessage(
        role="user",
        text_content=prompt_str,
        image_content=image_base64,
        mime_type=mime_type,
    )
    return imagemessage
