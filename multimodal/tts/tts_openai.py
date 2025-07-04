from openai import OpenAI
import os, io
from pydub import AudioSegment
from pydub.playback import play


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def play_openai_audio(audio_data):
    audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
    play(audio)


response = client.audio.speech.create(
    model="tts-1",  # or "tts-1-hd"
    voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
    input="Today is a wonderful day to build something people love!",
)

play_openai_audio(response.content)
