from Chain.tests.fixtures.sample_objects import sample_image_message
from Chain.result.response import Response
from Chain.model.model import Model
from Chain.message.imagemessage import ImageMessage

m = Model("Jlonge4/flux-dev-fp8")
response = m.query(query_input="create an image of st. dymphna", output_type="image")

breakpoint()

assert isinstance(response, Response)
assert isinstance(response.message, ImageMessage)
response.message.display()
