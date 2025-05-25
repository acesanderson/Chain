from Chain.message.imagemessage import ImageMessage
from Chain.message.message import Message
from Chain.chain.chain import Chain
from Chain.prompt.prompt import Prompt
from Chain.model.clients.anthropic_client import AnthropicClientSync
from Chain.model.model import Model
import base64
from PIL import Image
import io
from pathlib import Path
import os

dir_path = Path(__file__).parent
image_path = dir_path / "tr.jpg"


# Map PIL formats to MIME types
format_to_mime = {
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


def image_to_base64(file_path):
    """
    Simple version - load any image and convert to base64
    """
    with Image.open(file_path) as img:
        # Get actual format
        img_format = img.format.lower()

        # Convert to RGB if needed (for JPEG compatibility)
        if img.mode in ("RGBA", "LA", "P") and img_format in ["jpeg", "jpg"]:
            img = img.convert("RGB")

        # Save to buffer
        buffer = io.BytesIO()
        save_format = "JPEG" if img_format in ["jpeg", "jpg"] else img_format.upper()
        img.save(buffer, format=save_format)

        # Get base64
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Determine MIME type
        mime_map = {
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime_type = mime_map.get(img_format, "image/jpeg")

        return base64_data, mime_type


#
# def call_claude_with_image(imagemessage: ImageMessage) -> str:
#     client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
#     # Get our variables
#     role = imagemessage.role
#     text_content = imagemessage.text_content
#     image_content = imagemessage.image_content
#     mime_type = imagemessage.mime_type
#     # Make API call
#     response = client.messages.create(
#         model="claude-sonnet-4-20250514",
#         max_tokens=1024,
#         messages=[imagemessage.to_anthropic().model_dump()],
#     )
#     return response.content[0].text
#

if __name__ == "__main__":
    # a = AnthropicClientSync()
    # model_str = Model.models()["anthropic"][0]
    base64, mime_type = image_to_base64(image_path)
    prompt_str = "Name the movie this is from."
    imagemessage = ImageMessage(
        role="user", text_content=prompt_str, image_content=base64, mime_type=mime_type
    )
    # message = Message(
    #     role="user",
    #     content="name ten mammals",
    # )
    # # input = [imagemessage]
    # response = a.query(input=message, model=model_str)
    # print(response)
    m = Model("gpt")
    c = Chain(model=m)
    # response = m.query(imagemessage)
    response = c.run(messages=[imagemessage])
    print(response)
