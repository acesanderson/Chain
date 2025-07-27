"""
Dummy implementation, purely for getting imagegen to work.
See OLD_huggingface_client.py for the previous ideas for a HuggingFace client.
Keep this simple; HuggingFace is a complex suite, it's possible we'll only have very custom configurations in here, for imagegen, audiogen, audio transcription, image analysis.
"""

from Chain.model.clients.client import Client, Usage
from Chain.request.request import Request
from Chain.logs.logging_config import get_logger


logger = get_logger(__name__)


class HuggingFaceClient(Client):
    """
    This is a base class; we have two subclasses: OpenAIClientSync and OpenAIClientAsync.
    Don't import this.
    """

    def __init__(self):
        self._client = self._initialize_client()

    def _get_api_key(self) -> str:
        pass

    def tokenize(self, model: str, text: str) -> int:
        """
        Return the token count for a string, per model's tokenization function.
        """
        pass


class HuggingFaceClientSync(HuggingFaceClient):
    def _initialize_client(self) -> object:
        pass

    def query(
        self,
        request: Request,
    ) -> tuple:
        """
        Return a tuple of result, usage.
        """
        print("HUGGINGFACE REQUEST DETECTED")
        match request.output_type:
            case "image":
                from Chain.tests.fixtures.sample_objects import sample_image_message

                result = (
                    sample_image_message.image_content
                )  # base64 data of a generated image
                usage = Usage(
                    input_tokens=100, output_tokens=100
                )  # dummy usage data, for now
                return result, usage
            case "audio":
                from Chain.tests.fixtures.sample_objects import sample_audio_message

                result = (
                    sample_audio_message.audio_content
                )  # base64 data of a generated audio
                usage = Usage(
                    input_tokens=100, output_tokens=100
                )  # dummy usage data, for now
                return result, usage


"""
from pathlib import Path
from transformers import pipeline
import torch

# Import our centralized logger - no configuration needed here!
from Siphon.logs.logging_config import get_logger

# Get logger for this module - will inherit config from retrieve_audio.py
logger = get_logger(__name__)


# Transcript workflow
def transcribe(file_name: str | Path) -> str:
    ""
    Use Whisper to retrieve text content + timestamps.
    ""
    transcriber = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        # model="openai/whisper-large-v3",
        return_timestamps="sentence",
        device=0,
        torch_dtype=torch.float16,
    )
    logger.info(f"Transcribing file: {file_name}")
    result = transcriber(file_name)
    return result
"""
