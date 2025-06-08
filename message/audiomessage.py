"""
There are two basic message formats:
1. OpenAI (applicable to ollama, groq, gemini, etc.)
2. Anthropic (the one hold out)

We have a basic ImageMessage class, which is a wrapper for the OpenAI and Anthropic formats.
"""

from pydantic import BaseModel, Field
from Chain.message.message import Message
from Chain.message.imagemessage import OpenAITextContent
from typing import Literal
from pathlib import Path
import re


def is_base64_simple(s):
    """
    Simple validation for base64 strings.
    """
    return bool(re.match(r"^[A-Za-z0-9+/]*={0,2}$", s)) and len(s) % 4 == 0


# Gemini schema
"""
Gemini (through openai sdk) supports audio input through the chat/completions endpoint. It expects a message format like this:
{
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Transcribe this audio",
        },
        {
            "type": "input_audio",
            "input_audio": {"data": base64_audio, "format": "mp3"},
        },
    ],
}

"""


class GeminiInputAudio(BaseModel):
    """
    We are using Gemini through the OpenAI SDK, so we need to define the input audio format.
    Gemini usually supports a range of audio filetypes, but when used with OpenAI SDK, it's only mp3 and wav.
    """

    data: str = Field(description="The base64-encoded audio data.")
    format: Literal["mp3", "wav"] = Field(
        description="The format of the audio data, must be 'mp3' or 'wav'."
    )


class GeminiAudioContent(BaseModel):
    """
    Gemini AudioContent should have a single AudioContent object.
    NOTE: since we are using the OpenAI SDK, we use OpenAITextContent for text.

        type: str
        input_audio: dict
    """

    type: str = Field(
        default="input_audio", description="The type of content, must be 'input_audio'."
    )
    input_audio: GeminiInputAudio = Field(
        description="The input audio data, must be a base64-encoded string with format 'mp3' or 'wav'."
    )


class GeminiAudioMessage(Message):
    """
    Gemini AudioMessage should have a single AudioContent and a single TextContent object.
    NOTE: since we are using the OpenAI SDK, we use OpenAITextContent for text.

        role: str
        content: list[GeminiAudioContent | OpenAITextContent]
    """

    role: str  # type: ignore
    content: list[GeminiAudioContent | OpenAITextContent]  # type: ignore


# OpenAI-specific message classes
"""
Note: OpenAI has a completely different endpoint for audio transcriptions, so we need to handle that separately.

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
with open("example.m4a", "rb") as audio_file:
    audio_file = open("example.m4a", "rb")

transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
"""


class OpenAIAudioMessage(Message):
    role: str = Field(
        description="The role of the message, e.g. 'user', 'assistant', or 'system'."
    )
    file: str | Path


# class OpenAITextContent(BaseModel):
#     type: str = "text"  # Changed from "input_text"
#     text: str = Field(description="The text content of the message, i.e. the prompt.")
#
#
# class OpenAIImageUrl(BaseModel):
#     """Nested object for OpenAI image URL structure"""
#
#     url: str = Field(description="The data URL with base64 image")
#
#
# class OpenAIImageContent(BaseModel):
#     """
#     OpenAI requires image_url to be an object, not a string
#     """
#
#     type: str = "image_url"
#     image_url: OpenAIImageUrl = Field(description="The image URL object")
#
#
# class OpenAIImageMessage(Message):
#     """
#     ImageMessage should have a single ImageContent and a single TextContent object.
#
#         role: str
#         content: list[OpenAIImageContent | OpenAITextContent]
#     """
#
#     role: str
#     content: list[OpenAIImageContent | OpenAITextContent]  # type: ignore
#


# Our base ImageMessage class, with a factory method to convert to OpenAI or Anthropic format.
class AudioMessage(BaseModel):
==============NOTE: this needs to be implemented===================
    """
    ImageMessage should have a single ImageContent and a single TextContent object.

        role: str
        text_content: str
        image_content: str
        mime_type: str

    You can splat it to an OpenAI or Anthropic message; with the to_openai() and to_anthropic() methods.
    Model dump into the API query.
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

    def play(self):
        """
        Play the audio.
        """
        from pydub import AudioSegment
        from pydub.playback import play
        from pathlib import Path

        dir_path = Path(__file__).parent
        audio = AudioSegment.from_file(str(dir_path / "allhands.m4a"))

        play(audio)
