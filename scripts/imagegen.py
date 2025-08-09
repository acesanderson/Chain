from Chain.model.model import Model
import argparse


def generate_image(model_name, prompt, output_path):
    """
    Generate an image using the specified model and save it to the output path.

    :param model_name: Name of the model to use for image generation.
    :param prompt: Text prompt to guide the image generation.
    :param output_path: Path where the generated image will be saved.
    """
    model = Model(model_name)
