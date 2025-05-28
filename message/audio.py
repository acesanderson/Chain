"""
Roundup:
- Anthropic has NO support for voice
- OpenAI is the pioneer here, with their Whisper model
- Gemini support:
    Audio input (transcription): Supported by Gemini 2.5 Pro, 2.5 Flash, 2.0 Flash, and 2.0 Flash-Lite.
    Audio output (TTS): Supported by Gemini 2.5 Flash (preview TTS) and Gemini 2.0 Flash (experimental native audio).
    Both: Gemini 2.5 Flash and Gemini 2.0 Flash support both audio input and output.

For TTS:
*Gemini is the leader is accurate for enterprise and productivity scenarios, but OpenAI’s speech-to-speech technology is currently more advanced for nuanced, expressive, and interactive voice experiences*
"""

# Gemini code example
from openai import OpenAI
import base64
import instructor
import os

client = instructor.from_openai(
    OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Fixed endpoint
    )
)

with open("output.mp3", "rb") as audio_file:
    base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")

response = (
    client.chat.completions.create(  # Use chat completions, not audio.transcriptions
        model="gemini-2.0-flash",  # Correct model name
        response_model=None,
        messages=[
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
        ],
    )
)

print(response.choices[0].message.content)

# OpenAI code example for transcription
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
audio_file = open("example.m4a", "rb")

transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)

print(transcription.text)

# Conversion boilerplate
from pydub import AudioSegment

audio = AudioSegment.from_file("example.m4a", format="m4a")
audio.export("output.mp3", format="mp3")
