from pydantic import BaseModel
from typing import Any
import importlib
import warnings
from datetime import datetime

class CacheableMixin(object): # Keep this as object for Pydantic v2.x compatibility
    _CACHED_AT_KEY = "cached_at"
    _CACHE_VERSION_KEY = "cache_version"
    _CLASS_PATH_KEY = "class_path" # Store original class path for deserialization

    def to_cache_dict(self) -> dict[str, Any]:
        """
        Serializes the object to a dictionary suitable for caching.
        This method must be implemented by subclasses.
        It should produce a dictionary that `from_cache_dict` can use to reconstruct the object.
        """
        if not isinstance(self, BaseModel):
            raise TypeError(f"{self.__class__.__name__} must inherit from BaseModel to use to_cache_dict.")

        # Default serialization for Pydantic models: dump all fields
        data = self.model_dump()

        # Add metadata for deserialization
        data[self._CLASS_PATH_KEY] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        data[self._CACHED_AT_KEY] = datetime.now().isoformat() # Use isoformat for datetime
        data[self._CACHE_VERSION_KEY] = 1 # Increment as schema changes

        # Recursively handle nested CacheableMixins/BaseModels if any fields are complex
        for key, value in data.items():
            if isinstance(value, BaseModel) and hasattr(value, 'to_cache_dict'):
                data[key] = value.to_cache_dict()
            elif isinstance(value, list) and all(isinstance(item, BaseModel) and hasattr(item, 'to_cache_dict') for item in value):
                data[key] = [item.to_cache_dict() for item in value]
            elif isinstance(value, list) and all(isinstance(item, dict) and self._is_cached_item_dict(item) for item in value):
                # If it's already a serialized list of cached items, keep it
                pass
            elif isinstance(value, dict) and self._is_cached_item_dict(value):
                # If it's already a serialized cached item, keep it
                pass
            # Add handling for other complex types if needed (e.g., enums)
            # Your Verbosity enum has a to_json_serializable, perhaps handle it here
            elif hasattr(value, 'to_json_serializable') and callable(getattr(value, 'to_json_serializable')):
                data[key] = value.to_json_serializable()


        return data

    @staticmethod
    def _is_cached_item_dict(data: dict[str, Any]) -> bool:
        """Helper to check if a dict looks like a serialized CacheableMixin item."""
        return (
            isinstance(data, dict) and
            CacheableMixin._CLASS_PATH_KEY in data and
            CacheableMixin._CACHED_AT_KEY in data
        )

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> "CacheableMixin":
        """
        Reconstructs the object from a dictionary, using stored class path.
        This class method will find the original class and call its model_validate.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary for deserialization, got {type(data)}")

        class_path = data.pop(cls._CLASS_PATH_KEY, None)
        if not class_path:
            warnings.warn(f"Cannot deserialize: '{cls._CLASS_PATH_KEY}' not found in data. Data: {data.keys()}")
            # Fallback for old cache entries or malformed data
            # Try to validate with current cls, but it might fail
            if issubclass(cls, BaseModel):
                return cls.model_validate(data)
            raise ValueError(f"Cannot reconstruct {cls.__name__} without '{cls._CLASS_PATH_KEY}' and not a BaseModel.")

        try:
            module_name, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            target_cls = getattr(module, class_name)

            if not issubclass(target_cls, BaseModel):
                raise TypeError(f"Target class {class_name} ({module_name}) is not a BaseModel.")

            # Remove cache metadata before passing to model_validate
            data.pop(cls._CACHED_AT_KEY, None)
            data.pop(cls._CACHE_VERSION_KEY, None)

            # Recursively deserialize nested CacheableMixins/BaseModels
            for key, value in data.items():
                if isinstance(value, dict) and cls._is_cached_item_dict(value):
                    try:
                        # Attempt to deserialize nested CacheableMixin objects
                        data[key] = CacheableMixin.from_cache_dict(value)
                    except Exception as e:
                        warnings.warn(f"Failed to reconstruct nested object for field '{key}' with path '{value.get(cls._CLASS_PATH_KEY)}': {e}. Skipping item.")
                        # Decide how to handle: keep as dict, set to None, or raise
                        # For now, keep as dict and let downstream validation handle if it expects BaseModel
                        data[key] = value
                elif isinstance(value, list) and all(isinstance(item, dict) and cls._is_cached_item_dict(item) for item in value):
                    # For lists of CacheableMixin objects
                    reconstructed_list = []
                    for i, item_data in enumerate(value):
                        try:
                            reconstructed_list.append(CacheableMixin.from_cache_dict(item_data))
                        except Exception as e:
                            warnings.warn(f"Failed to reconstruct list item for field '{key}' at index {i} with path '{item_data.get(cls._CLASS_PATH_KEY)}': {e}. Skipping item.")
                            # Handle partially failed lists as needed
                            pass # skip item from list
                    data[key] = reconstructed_list
                # Handle Verbosity enum deserialization
                elif isinstance(value, str) and " (" in value and value.endswith(")"): # Heuristic for Verbosity string
                    try:
                        from Chain.progress.verbosity import Verbosity
                        data[key] = Verbosity.from_json_serializable(value)
                    except (ImportError, ValueError):
                        pass # Keep as string if it's not a Verbosity enum

            # Pydantic v2.x way to create instance from dictionary, handles nested models based on schema types
            return target_cls.model_validate(data)

        except Exception as e:
            # Re-raise with context to help debugging which class/data failed
            raise ValueError(f"Failed to reconstruct {class_path}: {e}")

