from Chain.message.audiomessage import AudioMessage
from Chain.model.model import Model
from pathlib import Path

dir_path = Path(__file__).parent
assets_dir = dir_path / "assets"
example_file = assets_dir / "output.mp3"

role = "user"
text_content = (
    "Please fully transcribe this audio file, with the text returned verbatim."
)
audio_message = AudioMessage(
    role=role, text_content=text_content, audio_file=example_file
)

# model = Model("gpt-4o-audio-preview")
model = Model("gemini")
response = model.query(input=[audio_message])
print(response)
