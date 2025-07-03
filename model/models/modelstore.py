from Chain.model.models.providerstore import ProviderStore
from Chain.model.models.provider import Provider
from Chain.model.models.modelspec import ModelSpec
from Chain.model.models.modelspecs_CRUD import get_modelspec_by_name, get_all_modelspecs
from pathlib import Path
import json, itertools

dir_path = Path(__file__).parent
models_path = dir_path / "models.json"
aliases_path = dir_path / "aliases.json"
ollama_context_sizes_path = dir_path / "ollama_context_sizes.json"

class ModelStore:
    """
    Class to manage model information for the Chain library.
    Provides methods to retrieve supported models, aliases, and validate model names.
    """
    @classmethod
    def models(cls):
        """Definitive list of models supported by Chain library."""
        with open(models_path) as f:
            return json.load(f)

    @classmethod
    def providers(cls) -> list[str]:
        """Definitive list of providers supported by Chain library."""
        providers = ProviderStore.get_all_providers()
        return [provider.provider for provider in providers]

    @classmethod
    def aliases(cls):
        """Definitive list of model aliases supported by Chain library."""
        with open(aliases_path) as f:
            return json.load(f)

    @classmethod
    def is_supported(cls, model: str) -> bool:
        """
        Check if the model is supported by the Chain library.
        Returns True if the model is supported, False otherwise.
        """
        in_aliases = model in cls.aliases().keys()
        in_models = model in list(itertools.chain.from_iterable(cls.models().values()))
        return in_aliases or in_models

    @classmethod
    def _validate_model(cls, model: str) -> str:
        """
        Validate the model name against the supported models and aliases.
        Converts aliases to their corresponding model names if necessary.
        """
        # Load aliases
        aliases = cls.aliases()
        # Assign models based on aliases
        if model in cls.aliases().keys():
            model = aliases[model]
        elif cls.is_supported(model):
            model = model
        else:
            ValueError(
                f"WARNING: Model not found locally: {model}. This may cause errors."
            )
        return model

    @classmethod
    def get_num_ctx(cls, ollama_model: str) -> int:
        """
        Get the preferred num_ctx for a given ollama model.
        """
        default_value = 32768
        if ollama_context_sizes_path.exists():
            with open(ollama_context_sizes_path) as f:
                context_sizes = json.load(f)
            if ollama_model in context_sizes:
                return context_sizes[ollama_model]
            else:
                print(
                    f"WARNING: Model {ollama_model} not found in context sizes file. Using default value of {default_value}."
                )
                return default_value  # Default context size if not specified -- may throw an error for smaller models
        else:
            raise FileNotFoundError(
                f"Context sizes file not found: {ollama_context_sizes_path}"
            )

    @classmethod
    def display(cls):
        from rich.console import Console
        from rich.columns import Columns
        from rich.text import Text
        
        console = Console()
        models = cls.models()
        
        # Calculate total items and split points
        all_items = []
        for provider, model_list in models.items():
            all_items.append((provider, None))  # Provider header
            for model in model_list:
                all_items.append((provider, model))
        
        # Split into three roughly equal columns
        total_items = len(all_items)
        col1_end = total_items // 3
        col2_end = 2 * total_items // 3
        
        # Create three columns
        left_column = Text()
        middle_column = Text()
        right_column = Text()
        
        for i, (provider, model) in enumerate(all_items):
            if i < col1_end:
                target_column = left_column
            elif i < col2_end:
                target_column = middle_column
            else:
                target_column = right_column
            
            if model is None:  # Provider header
                target_column.append(f"{provider.upper()}\n", style="bold cyan")
            else:  # Model name
                target_column.append(f"  {model}\n", style="white")
        
        console.print(Columns([left_column, middle_column, right_column], equal=True, expand=True))

    # Getters
    @classmethod
    def get_model(cls, model: str) -> ModelSpec:
        """
        Get the model name, validating against aliases and supported models.
        """
        model = cls._validate_model(model)
        try:
            return get_modelspec_by_name(model)
        except ValueError:
            raise ValueError(f"Model '{model}' not found in the database.")

    @classmethod
    def get_all_models(cls) -> list[ModelSpec]:
        """
        Get all models as ModelSpec objects.
        """
        return get_all_modelspecs()

    ## Get subsets of models by provider
    @classmethod
    def by_provider(cls, provider: Provider) -> list[str]:
        """
        Get a list of models for a specific provider.
        """
        models = cls.models()
        return models.get(provider, [])

    ## Get lists of models by capability
    @classmethod
    def image_analysis_models(cls) -> list[str]:
        """
        Get a list of models that support image analysis.
        """
        return [model for model, capabilities in cls.models().items() if capabilities.get("image_analysis", False)]

    @classmethod
    def image_gen_models(cls) -> list[str]:
        """
        Get a list of models that support image generation.
        """
        return [model for model, capabilities in cls.models().items() if capabilities.get("image_gen", False)]

    @classmethod
    def audio_analysis_models(cls) -> list[str]:
        """
        Get a list of models that support audio analysis.
        """
        return [model for model, capabilities in cls.models().items() if capabilities.get("audio_analysis", False)]

    @classmethod
    def audio_gen_models(cls) -> list[str]:
        """
        Get a list of models that support audio generation.
        """
        return [model for model, capabilities in cls.models().items() if capabilities.get("audio_gen", False)]

    @classmethod
    def video_analysis_models(cls) -> list[str]:
        """
        Get a list of models that support video analysis.
        """
        return [model for model, capabilities in cls.models().items() if capabilities.get("video_analysis", False)]

    @classmethod
    def video_gen_models(cls) -> list[str]:
        """
        Get a list of models that support video generation.
        """
        return [model for model, capabilities in cls.models().items() if capabilities.get("video_gen", False)]

    @classmethod
    def reasoning_models(cls) -> list[str]:
        """
        Get a list of models that support reasoning.
        """
        return [model for model, capabilities in cls.models().items() if capabilities.get("reasoning", False)]

    @classmethod
    def text_completion_models(cls) -> list[str]:
        """
        Get a list of models that support text completion.
        """
        return [model for model, capabilities in cls.models().items() if capabilities.get("text_completion", False)]
