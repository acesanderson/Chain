from pathlib import Path
import json, itertools

dir_path = Path(__file__).parent
models_path = dir_path.parent / "clients" / "models.json"
aliases_path = dir_path.parent / "aliases.json"

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

