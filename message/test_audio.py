from pathlib import Path
from Chain.message.audiomessage import AudioMessage
from Chain.chain.chain import Chain
from Chain.model.model import Model

dir_path = Path(__file__).parent
output_mp3 = dir_path / "output.mp3"
# convert output_mp3 to base64 string

audiomessage = AudioMessage(
    role="user",
    text_content="Transcribe this audio with high fidelity.",
    audio_file=output_mp3,
    format="mp3",
)


model = Model("flash")
# model = Model("gpt-4o-audio-preview")
chain = Chain(model=model)

response = model.query(audiomessage)

print(response)
