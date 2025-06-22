import json
from typing import Optional
from pathlib import Path
from Chain.model.models.ModelSpec import ModelSpec


class ModelStore:
    def __init__(self, spec_file: Optional[Path] = None):
        self.spec_file = spec_file or Path(__file__).parent / ".model_specs.json"
        self._specs: dict[str, ModelSpec] = {}
        self._load_specs()

    def _load_specs(self) -> None:
        """Load model specs from JSON file"""
        if self.spec_file.exists():
            with open(self.spec_file) as f:
                data = json.load(f)
                self._specs = {
                    spec_data["model_id"]: ModelSpec(**spec_data) for spec_data in data
                }

    def get_spec(self, model_id: str) -> ModelSpec:
        """Get spec for a specific model"""
        if model_id not in self._specs:
            raise ValueError(f"Model {model_id} not found in model store")
        return self._specs[model_id]

    def has_model(self, model_id: str) -> bool:
        """Check if model exists"""
        return model_id in self._specs

    def list_models(
        self, provider: Optional[str] = None, **capability_filters
    ) -> list[str]:
        """list models, optionally filtered by provider or capabilities"""
        models = []
        for model_id, spec in self._specs.items():
            # Filter by provider
            if provider and spec.provider != provider:
                continue

            # Filter by capabilities (e.g., function_calling=True)
            if capability_filters:
                spec_capabilities = spec.capabilities.model_dump()
                if not all(
                    spec_capabilities.get(cap) == value
                    for cap, value in capability_filters.items()
                ):
                    continue

            models.append(model_id)
        return models

    def get_providers(self) -> list[str]:
        """Get list of all providers"""
        return list(set(spec.provider for spec in self._specs.values()))

    def validate_capability(self, model_id: str, capability: str) -> bool:
        """Check if model supports a capability"""
        spec = self.get_spec(model_id)
        return getattr(spec.capabilities, capability, False)

    def validate_format(
        self, model_id: str, format_type: str, file_format: str
    ) -> bool:
        """Check if model supports a specific file format"""
        spec = self.get_spec(model_id)
        formats = getattr(spec.formats, format_type, [])
        return file_format.lower() in [f.lower() for f in formats]
