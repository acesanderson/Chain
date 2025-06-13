"""
Fragments that can be used to generate ModelSpecs using Perplexity API.
NOTE: the list[BaseModel] usage of Parser requires more refactoring, as list[BaseModel] is not an object and therefore cannot be passed to Parser like I try below.

This might work -- it's really a matter of debugging the fact that I use isinstance(message, BaseModel) in my conditionals in query functions. Need to expand it to GenericAlias (which list[BaseModel] is).
```python
from typing import get_origin, get_args
from types import GenericAlias

def is_valid_pydantic_type(obj):
    # Check for Pydantic model class
    if isinstance(obj, type) and issubclass(obj, BaseModel):
        return True

    # Check for list[PydanticModel]
    if isinstance(obj, GenericAlias):
        origin = get_origin(obj)
        args = get_args(obj)
        if origin is list and args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return True

    return False

# Usage:
if pydantic_model and is_valid_pydantic_type(pydantic_model):
    # Handle structured output
```
"""

from Chain.chain.chain import Chain
from Chain.parser.parser import Parser
from Chain.prompt.prompt import Prompt
from Chain.model.model import Model
from Chain.model.models.ModelStore import ModelStore
from Chain.model.models.ModelSpec import ModelSpec, ModelSpecs

model_store = ModelStore()


def generate_model_specs(prompt_str: str, model: str = "sonar-pro") -> ModelSpecs:
    """
    Generate model specifications using a prompt and a specified model.

    Args:
        prompt_str (str): The prompt string to use for generating model specs.
        model (str): The model to use for generation, default is "sonar_pro".

    Returns:
        ModelSpecs: The generated model specifications.
    """
    prompt = Prompt(prompt_str)
    parser = Parser(ModelSpecs)
    model_obj = Model(model)
    chain = Chain(model=model_obj, parser=parser, prompt=prompt)
    model = Model(model)
    response = chain.run()
    return response.content


if __name__ == "__main__":
    # Example usage
    prompt = "You will help me create an inventory of AI models available from OpenAI. Please return a list of models with their specifications, including provider, deployment type, capabilities, formats, context window size, knowledge cutoff date, and a brief description. The output should be in the format of ModelSpecs. If a model DOESN'T meet the schema, please do not include. For example I don't want embedding models included."
    specs = generate_model_specs(prompt, model="gpt")
    print(specs)
