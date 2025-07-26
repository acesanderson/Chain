"""
Ollama does NOT do image gen. need hugging face.
"""

import torch
from diffusers import FluxPipeline

# pipe = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
# )
#

pipe = FluxPipeline.from_pretrained("Jlonge4/flux-dev-fp8", torch_dtype=torch.bfloat16)

# pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe.enable_sequential_cpu_offload()

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]


import io
import base64

# Convert PIL image to base64 string
buffered = io.BytesIO()
image.save(buffered, format="PNG")  # or "JPEG"
img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

from Chain.message.imagemessage import ImageMessage

im = ImageMessage.from_base64(image_content=img_str, text_content="generated image")
im.display()
