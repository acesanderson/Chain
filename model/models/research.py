"""
Fragments that can be used to generate ModelSpecs using Perplexity API.
"""

from Chain.chain.chain import Chain
from Chain.parser.parser import Parser
from Chain.prompt.prompt import Prompt
from Chain.model.model import Model
from Chain.model.models.ModelStore import ModelStore
from Chain.model.models.ModelSpec import ModelSpec, ModelSpecs

model_store = ModelStore()


def generate_model_specs(prompt_str: str, model: str = "sonar") -> ModelSpecs:
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
    response = chain.run()
    return response.content


if __name__ == "__main__":
    # Example usage
    prompt = "You will help me create an inventory of AI models available from OpenAI. Please return a list of models with their specifications, including provider, deployment type, capabilities, formats, context window size, knowledge cutoff date, and a brief description. The output should be in the format of ModelSpecs. If a model DOESN'T meet the schema, please do not include. For example I don't want embedding models included."
    specs = generate_model_specs(prompt, model="claude")
    print(specs)
