from google import genai
from google.genai import types
from pydub import AudioSegment
from pydub.playback import play
import os, io

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# For Gemini TTS (L16 PCM format)
def play_gemini_audio(audio_data):
    audio = AudioSegment(data=audio_data, sample_width=2, frame_rate=24000, channels=1)
    play(audio)


# For OpenAI TTS (MP3 format)
def play_openai_audio(audio_data):
    audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
    play(audio)


# Generate response
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-tts",
    contents="Say cheerfully: Have a wonderful day!",
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],  # CRITICAL: Only AUDIO
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Kore",  # 30 voice options available
                )
            )
        ),
    ),
)

breakpoint()

# Usage:
play_gemini_audio(response.candidates[0].content.parts[0].inline_data.data)
# OpenAI: play_openai_audio(response.content)
