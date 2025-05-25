"""
Prompt class -- coordinates templates, input variables, and rendering.
"""

from jinja2 import Environment, StrictUndefined, meta

# Define jinja2 environment that we will use across all prompts.
env = Environment(
    undefined=StrictUndefined
)  # set jinja2 to throw errors if a variable is undefined


class Prompt:
    """ "
    Takes a jinja2 ready string (note: not an actual Template object; that's created by the class).
    The three stages of prompt creation:
    - the prompt string, which is provided to the class
    - the jinja template created from the prompt string
    - the rendered prompt, which is returned by the class and submitted to the LLM model.
    """

    def __init__(self, prompt_string: str):
        self.prompt_string = prompt_string
        self.template = env.from_string(prompt_string)

    def render(self, input_variables: dict) -> str:
        """
        takes a dictionary of variables
        """
        rendered_prompt = self.template.render(
            **input_variables
        )  # this takes all named variables from the dictionary we pass to this.
        return rendered_prompt

    def input_schema(self) -> set:
        """
        Returns a set of variable names from the template.
        This can be used to validate that the input variables match the template.
        """
        parsed_content = env.parse(self.prompt_string)
        return meta.find_undeclared_variables(parsed_content)

    def __repr__(self):
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"


class ImagePrompt(Prompt):
    """
    Extends the Prompt class to include image content.
    You need to pass in the image content as a base64 string and the mimetype.
    """

    def __init__(self, prompt_string: str, mimetype: str, image_content: str):
        super().__init__(prompt_string)
        self.mimetype = mimetype
        self.image_content = image_content
