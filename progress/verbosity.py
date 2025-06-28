from enum import Enum
from typing import Dict, Any

class Verbosity(Enum):
    """
    SILENT - Obviously nothing shown
    PROGRESS - Just the spinner/completion (your current default)
    SUMMARY - Basic request/response info (level 2 in your spec)
    DETAILED - Truncated messages in panels (level 3)
    COMPLETE - Full messages in panels (level 4)
    DEBUG - Full JSON with syntax highlighting (level 5)
    """

    SILENT = 0
    PROGRESS = 1
    SUMMARY = 2
    DETAILED = 3
    COMPLETE = 4
    DEBUG = 5

    @classmethod
    def from_input(cls, value):
        """
        Converts various input types to a Verbosity instance.
        """
        if value is False:
            return cls.SILENT
        elif value is True:
            return cls.PROGRESS
        elif isinstance(value, cls):
            return value
        elif isinstance(value, str):
            # Map string values to enum members
            string_map = {
                "": cls.SILENT,
                "v": cls.PROGRESS,
                "vv": cls.SUMMARY,
                "vvv": cls.DETAILED,
                "vvvv": cls.COMPLETE,
                "vvvvv": cls.DEBUG
            }
            if value in string_map:
                return string_map[value]
            raise ValueError(f"Invalid verbosity: {value}")
        else:
            raise ValueError(f"Invalid verbosity type: {type(value)}")

    def __bool__(self) -> bool:
        """
        Returns True if the verbosity level is not SILENT.
        """
        return self != Verbosity.SILENT

    def __lt__(self, other):
        """Less than comparison based on enum values."""
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        """Less than or equal comparison based on enum values."""
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        """Greater than comparison based on enum values."""
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        """Greater than or equal comparison based on enum values."""
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def to_cache_dict(self) -> Dict[str, Any]:
        """
        Serialize Verbosity to cache-friendly dictionary.
        
        Returns:
            Dictionary with verbosity name and value for reconstruction
        """
        return {
            "verbosity_type": "Verbosity",
            "name": self.name,
            "value": self.value
        }

    @classmethod
    def from_cache_dict(cls, data: Dict[str, Any]) -> "Verbosity":
        """
        Deserialize Verbosity from cache dictionary.
        
        Args:
            data: Dictionary containing verbosity information
            
        Returns:
            Verbosity enum instance
            
        Raises:
            ValueError: If the data doesn't contain valid verbosity information
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")
        
        if data.get("verbosity_type") != "Verbosity":
            raise ValueError(f"Invalid verbosity_type: {data.get('verbosity_type')}")
        
        # Try to reconstruct by name first (most reliable)
        if "name" in data:
            try:
                return cls[data["name"]]
            except KeyError:
                pass
        
        # Fallback to value
        if "value" in data:
            try:
                return cls(data["value"])
            except ValueError:
                pass
        
        raise ValueError(f"Could not reconstruct Verbosity from data: {data}")

    def to_json_serializable(self) -> str:
        """
        Convert to JSON-serializable string format.
        Useful for display methods and JSON serialization.
        
        Returns:
            String representation like "PROGRESS (1)"
        """
        return f"{self.name} ({self.value})"

    @classmethod
    def from_json_serializable(cls, value: str) -> "Verbosity":
        """
        Reconstruct from JSON-serializable string format.
        
        Args:
            value: String like "PROGRESS (1)" or just "PROGRESS"
            
        Returns:
            Verbosity enum instance
        """
        if " (" in value and value.endswith(")"):
            # Extract name from "PROGRESS (1)" format
            name = value.split(" (")[0]
        else:
            # Just the name
            name = value
        
        try:
            return cls[name]
        except KeyError:
            raise ValueError(f"Invalid verbosity name: {name}")

    def __str__(self) -> str:
        """String representation for logging and display."""
        return self.name

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"Verbosity.{self.name}"
