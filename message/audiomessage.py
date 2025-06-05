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
import base64, re


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

        input_audio: dict
        type: str (Default variable)
    """

    # Init variables
    input_audio: GeminiInputAudio = Field(
        description="The input audio data, must be a base64-encoded string with format 'mp3' or 'wav'."
    )

    # Default variables
    type: str = Field(
        default="input_audio", description="The type of content, must be 'input_audio'."
    )


class GeminiAudioMessage(Message):
    """
    Gemini AudioMessage should have a single AudioContent and a single TextContent object.
    NOTE: since we are using the OpenAI SDK, we use OpenAITextContent for text.

        role: str
        content: list[GeminiAudioContent | OpenAITextContent]
    """

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
    file: str | Path


# Our base ImageMessage class, with a factory method to convert to OpenAI or Anthropic format.
class AudioMessage(BaseModel):
    """
    ImageMessage should have a single ImageContent and a single TextContent object.

    You can splat it to an OpenAI or Gemini message; with the to_openai() and to_gemini() methods.
    """

    # Init variables
    role: str = Field(
        description="The role of the message, e.g. 'user', 'assistant', or 'system'."
    )
    text_content: str = Field(
        description="The text content of the message, i.e. the prompt."
    )
    audio_file: str | Path = Field(description="The path to the audio file to be sent.")

    # Post-init variables
    format: Literal["wav", "mp3", ""] = Field(
        description="The audio format.", default=""
    )
    audio_content: str = Field(
        description="The base64-encoded audio data.", default="", repr=False
    )

    def model_post_init(self, __context):  # type: ignore
        """
        Convert the audio file to base64 string.
        """
        # Check if the audio file exists
        if isinstance(self.audio_file, str):
            self.audio_file = Path(self.audio_file)
        if not self.audio_file.exists():
            raise FileNotFoundError(f"Audio file {self.audio_file} does not exist.")
        # Convert the audio file to base64 string
        self.audio_content = self._convert_audio_to_base64(self.audio_file)
        if not is_base64_simple(self.audio_content):
            raise ValueError("Audio content is not a valid base64 string.")
        # Infer the audio format from the file extension if not provided
        if self.format == "":
            if self.audio_file.suffix.lower()[1:] in ["mp3", "wav"]:
                self.format = self.audio_file.suffix.lower()[1:]
        # Change audio_file to str so it can be used in OpenAI API
        self.audio_file = str(self.audio_file)

    def _convert_audio_to_base64(self, file_path: Path) -> str:
        """
        Convert the audio file to base64 string.
        """
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def __repr__(self):
        return f"AudioMessage(role={self.role}, text_content={self.text_content}, audio_file={self.audio_file}, format={self.format})"

    def to_gemini(self) -> GeminiAudioMessage:
        """
        Converts the ImageMessage to the Gemini format.
        """
        geminiinputaudio = GeminiInputAudio(data=self.audio_content, format=self.format)
        geminiaudiocontent = GeminiAudioContent(input_audio=geminiinputaudio)
        text_content = OpenAITextContent(text=self.text_content)
        return GeminiAudioMessage(
            role=self.role, content=[text_content, geminiaudiocontent]
        )

    def to_openai(self) -> OpenAIAudioMessage:
        """
        Converts the ImageMessage to the OpenAI format.
        """
        return OpenAIAudioMessage(
            role=self.role, content=self.text_content, file=self.audio_file
        )
