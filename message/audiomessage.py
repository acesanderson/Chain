from Chain.message.message import Message
from Chain.message.imagemessage import OpenAITextContent
from Chain.logging.logging_config import get_logger
from pydantic import BaseModel, Field
from typing import Literal, Any, Optional
from pathlib import Path
import base64, re

logger = get_logger(__name__)


def is_base64_simple(s):
    """
    Simple validation for base64 strings.
    """
    return bool(re.match(r"^[A-Za-z0-9+/]*={0,2}$", s)) and len(s) % 4 == 0


class OpenAIInputAudio(BaseModel):
    """
    We are using Gemini through the OpenAI SDK, so we need to define the input audio format.
    Gemini usually supports a range of audio filetypes, but when used with OpenAI SDK, it's only mp3 and wav.
    """

    data: str = Field(description="The base64-encoded audio data.")
    format: Literal["mp3", "wav", ""] = Field(
        description="The format of the audio data, must be 'mp3' or 'wav'."
    )


class OpenAIAudioContent(BaseModel):
    """
    Gemini AudioContent should have a single AudioContent object.
    NOTE: since we are using the OpenAI SDK, we use OpenAITextContent for text.
    """

    input_audio: OpenAIInputAudio = Field(
        description="The input audio data, must be a base64-encoded string with format 'mp3' or 'wav'."
    )
    type: str = Field(
        default="input_audio", description="The type of content, must be 'input_audio'."
    )


class OpenAIAudioMessage(Message):
    """
    Gemini AudioMessage should have a single AudioContent and a single TextContent object.
    NOTE: since we are using the OpenAI SDK, we use OpenAITextContent for text.
    """

    content: list[OpenAIAudioContent | OpenAITextContent]  # type: ignore


class AudioMessage(Message):
    """
    AudioMessage with serialization/deserialization support.
    You can splat it to an OpenAI or Gemini message; with the to_openai() method.
    """

    content: Optional[Any] = Field(default=None)

    # AudioMessage-specific fields
    text_content: str = Field(
        description="The text content of the message, i.e. the prompt."
    )
    audio_file: str | Path = Field(default="", description="The path to the audio file to be sent.")

    # Post-init variables
    format: Literal["wav", "mp3", ""] = Field(
        description="The audio format.", default=""
    )
    audio_content: str = Field(
        description="The base64-encoded audio data.", default="", repr=False
    )

    def model_post_init(self, __context):  # type: ignore
        """
        Convert the audio file to base64 string and set up the content field.
        """
        super().model_post_init(__context) # Call the mixin's post_init first!
        # Skip post_init if this is a cache restoration (empty audio_file)
        if not self.audio_file or self.audio_file == "":
            if self.audio_content and self.format:
                # This is cache restoration - set content field
                self.content = [self.audio_content, self.text_content]
                return

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

        # Set up the content field to match Message interface
        self.content = [self.audio_content, self.text_content]

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

    def to_openai(self) -> OpenAIAudioMessage:
        """
        Converts the AudioMessage to the OpenAI format.
        """
        openaiinputaudio = OpenAIInputAudio(data=self.audio_content, format=self.format)
        openaiaudiocontent = OpenAIAudioContent(input_audio=openaiinputaudio)
        text_content = OpenAITextContent(text=self.text_content)
        return OpenAIAudioMessage(
            role=self.role, content=[text_content, openaiaudiocontent]
        )

    def play(self):
        """
        Play the audio.
        """
        from pydub import AudioSegment
        from pydub.playback import play

        audio = AudioSegment.from_file(self.audio_file, format=self.format)
        play(audio)
