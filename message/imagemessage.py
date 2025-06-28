"""
There are two basic message formats:
1. OpenAI (applicable to ollama, groq, gemini, etc.)
2. Anthropic (the one hold out)

We have a basic ImageMessage class, which is a wrapper for the OpenAI and Anthropic formats.
"""

from pydantic import BaseModel, Field, ValidationError
from Chain.message.message import Message
from Chain.message.convert_image import convert_image, convert_image_file
from Chain.logging.logging_config import get_logger
from pathlib import Path
from typing import Dict, Any
import re

logger = get_logger(__name__)

# Map PIL formats to MIME types
format_to_mime = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def is_base64_simple(s):
    """
    Simple validation for base64 strings.
    """
    return bool(re.match(r"^[A-Za-z0-9+/]*={0,2}$", s)) and len(s) % 4 == 0


def extension_to_mimetype(image_file: Path) -> str:
    """
    Given a Path object, return the mimetype.
    """
    extension = image_file.suffix.lower()
    try:
        mimetype = format_to_mime[extension]
        return mimetype
    except:
        raise ValueError(
            f"Unsupported image format: {extension}. Supported formats are: {', '.join(format_to_mime.keys())}"
        )


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
    """

    role: str
    content: list[AnthropicImageContent | AnthropicTextContent]  # type: ignore


# OpenAI-specific message classes
class OpenAITextContent(BaseModel):
    type: str = "text"
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
    """

    role: str
    content: list[OpenAIImageContent | OpenAITextContent]  # type: ignore


# Our base ImageMessage class with serialization support
class ImageMessage(Message):
    """
    ImageMessage with serialization/deserialization support.

    You can instantiate it with either:
    1. text_content + image_content + mime_type
    2. text_content + image_file (conversion happens automatically)

    You can convert it to provider formats with to_openai() and to_anthropic() methods.
    """

    content: list[BaseModel | str] | None = Field(default=None)
    text_content: str = Field(
        description="The text content of the message, i.e. the prompt."
    )
    image_file: str | Path = Field(
        description="File name for the image, if available.", default=""
    )
    image_content: str = Field(description="The base64-encoded image.", default="")
    mime_type: str = Field(
        description="The MIME type of the image, e.g. 'image/jpeg', 'image/png'.",
        default="",
    )

    def model_post_init(self, __context) -> None:
        """Called after model initialization to construct content field."""
        # Skip post_init if this is a cache restoration (empty image_file and existing content)
        if not self.image_file or self.image_file == "":
            if self.image_content and self.mime_type:
                # This is cache restoration - content already exists
                if not hasattr(self, "content") or not self.content:
                    self.content = [self.image_content, self.text_content]
                return

        # If user adds image_content and mime_type, convert them to PNG, downsample, and update base64 string.
        if self.image_content and self.mime_type:
            # Convert the image_content to a base64-encoded PNG if it's not already in that format.
            if self.mime_type != "image/png":
                self.image_content = convert_image(self.image_content)
                self.mime_type = "image/png"
        # If user submits a image_file instead of the mimetype / image_content
        if self.image_file and not self.image_content:
            # Convert the image_file to a Path object if it's a string
            if isinstance(self.image_file, str):
                self.image_file = Path(self.image_file)
            if self.image_content and self.mime_type:
                raise ValueError(
                    "Can't instantiate ImageMessage with both file_name and image_content at the same time."
                )
            self.mime_type = extension_to_mimetype(Path(self.image_file))
            self.image_content = convert_image_file(self.image_file)

        # Validate the MIME type
        if self.mime_type not in format_to_mime.values():
            raise ValueError(f"Unsupported MIME type: {self.mime_type}")
        # Validate the image content
        if not is_base64_simple(self.image_content):
            raise ValueError("Image content must be a base64-encoded string.")

        # Construct our content
        self.content = [self.image_content, self.text_content]

        # Change image_file back to a string for consistency
        if isinstance(self.image_file, Path):
            self.image_file = str(self.image_file)
        # Raise an error if we have an incomplete object at the end of this process.
        if self.image_content == "" or not self.mime_type or not self.content:
            raise ValidationError("Incorrect initialization for some reason.")

    def to_cache_dict(self) -> Dict[str, Any]:
        """
        Serialize ImageMessage to cache-friendly dictionary.
        """
        return {
            "message_type": "ImageMessage",
            "role": self.role.value if hasattr(self.role, "value") else self.role,
            "text_content": self.text_content,
            "image_file": str(self.image_file),
            "image_content": self.image_content,
            "mime_type": self.mime_type,
        }

    @classmethod
    def from_cache_dict(cls, data: Dict[str, Any]) -> "ImageMessage":
        """
        Deserialize ImageMessage from cache dictionary.
        Temporarily removes image_file to avoid validation conflict.
        """
        # Create instance with minimal data first
        instance = cls.model_construct(
            role=data["role"],
            text_content=data["text_content"],
            image_file="",  # Empty to avoid conflict
            image_content=data["image_content"],
            mime_type=data["mime_type"],
        )

        # Manually set the remaining fields after construction
        instance.content = [data["image_content"], data["text_content"]]
        instance.image_file = data["image_file"]  # Set the real file path

        return instance

    def __repr__(self):
        return f"ImageMessage(role={self.role}, text_content={self.text_content}, image_content={self.image_content[:10]}..., mime_type={self.mime_type})"

    def display(self):
        """
        Display a base64-encoded image using chafa.
        Your mileage may vary depending on the terminal and chafa version.
        """
        import subprocess, base64, os

        try:
            image_data = base64.b64decode(self.image_content)
            cmd = ["chafa", "-"]

            # If in tmux or SSH, force text mode for consistency
            if (
                os.environ.get("TMUX")
                or os.environ.get("SSH_CLIENT")
                or os.environ.get("SSH_CONNECTION")
            ):
                cmd.extend(["--format", "symbols", "--symbols", "block"])
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            process.communicate(input=image_data)
        except Exception as e:
            print(f"Error: {e}")

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
