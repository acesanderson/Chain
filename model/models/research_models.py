from Chain.model.models.modelspec import ModelSpecList, ModelSpec
from Chain.model.models.modelspecs_CRUD import create_modelspecs_from_scratch, add_modelspec, get_all_modelspecs, get_modelspec_by_name, update_modelspec, delete_modelspec, in_db
from Chain import Model, Chain, Prompt, Parser
from rich.console import Console

console = Console()
Chain._console = console  # Set the console for Chain to use

prompt_str = """
You are an assistant who will help me identify the capabilities of a list of LLMs.

<provider>
{{provider}}
</provider>

<model_list>
{{model_list}}
</model_list>

You will be identifying, for each model, whether it supports the following functionalities:

- **text_completion**: The ability to generate, continue, or complete text based on prompts and context
- **image_analysis**: The capability to analyze, understand, and describe visual content in images
- **image_gen**: The ability to create or generate new images from text descriptions or prompts
- **audio_analysis**: The capability to process, transcribe, or analyze audio content and speech
- **audio_gen**: The ability to synthesize speech or generate audio content from text or other inputs
- **video_analysis**: The capability to analyze, understand, and extract information from video content
- **video_gen**: The ability to create or generate video content from text descriptions or other inputs
- **reasoning**: The capability to perform logical reasoning, problem-solving, and complex analytical tasks

For each model, please return a ModelCapabilities object with boolean values for each capability, indicating whether the model supports (True) or does not support (False) each functionality.

You should also generate factual description of each model (50-80 words), including:
- The model's architecture type and parameter size (if known)
- Specific capabilities it supports (multimodal inputs, function calling, etc.)
- Context window size and key technical specifications
- Release date or version information
- Primary intended use cases or design focus
- Any notable technical features or limitations

Avoid promotional language or subjective quality assessments. Focus on objective, verifiable information about what the model can do and its technical characteristics.

Return a different object for each model -- since there are {{length}} models in our list, return {{length}} ModelCapabilities objects.
""".strip()

def get_capabilities_by_provider(provider: str, model_list: list[str]) -> list[ModelSpec]:
    length = len(model_list)
    model = Model("sonar-pro")
    prompt = Prompt(prompt_str)
    parser = Parser(ModelSpecList)
    chain = Chain(model=model, prompt=prompt, parser=parser)
    response = chain.run(input_variables = {"provider": provider, "model_list": model_list, "length": length})
    return response.content.specs

def get_all_capabilities() -> list[ModelSpec]:
    """
    Get capabilities for all models across all providers. This shouldn't need to be run often, since it replaces the entire database of model specs.
    """
    all_models = Model.models()
    all_specs = []
    for index, (provider, models) in enumerate(all_models.items()):
        print(f"Processing {index + 1}/{len(all_models)}: {provider} with {len(models)} models")
        specs = get_capabilities_by_provider(provider=provider, model_list=models)
        all_specs.extend(specs)
    return all_specs

def create_from_scratch() -> None:
    """
    Create a new empty database for ModelSpecs and populate it with capabilities from all providers.
    This will overwrite any existing data in modelspecs.json.
    """
    all_specs = get_all_capabilities()
    create_modelspecs_from_scratch(all_specs)
    print(f"Populated ModelSpecs database with {len(all_specs)} entries.")
    # Test retrieval of all specs
    retrieved_specs = get_all_modelspecs()
    assert len(retrieved_specs) == len(all_specs), "Retrieved specs do not match created specs."

if __name__ == "__main__":
    create_from_scratch()
