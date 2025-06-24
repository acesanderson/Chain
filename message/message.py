"""
Message is the default message type recognized as industry standard (role + content).
Our Message class is inherited from specialized types like AudioMessage, ImageMessage, etc.
We define list[Message] as Messages in a parallel file, this class can handle the serialization / deserialization needed for message historys, api calls, and caching.
"""

from pydantic import BaseModel
from Chain.prompt.prompt import Prompt
from enum import Enum
from typing import Any
import importlib, json, warnings

class Role(Enum):
    """
    Enum for the role of the message.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """
    Industry standard, more or less, for messaging with LLMs.
    System roles can have some weirdness (like with Anthropic), but role/content is standard.
    """

    role: str | Role
    content: str | BaseModel | list[BaseModel]

    def to_cache_dict(self) -> dict[str, Any]:
        """
        Serialize Message to cache-friendly dictionary with type preservation.
        """
        serialized = {
            "message_type": self.__class__.__name__,
            "role": self.role.value if isinstance(self.role, Role) else self.role,
        }
        
        # Handle different content types
        if isinstance(self.content, str):
            serialized["content"] = {
                "type": "string",
                "data": self.content
            }
        elif isinstance(self.content, BaseModel):
            serialized["content"] = {
                "type": "pydantic_model",
                "model_class": f"{self.content.__class__.__module__}.{self.content.__class__.__name__}",
                "data": self.content.model_dump()
            }
        elif isinstance(self.content, list):
            serialized["content"] = {
                "type": "pydantic_list",
                "data": []
            }
            for item in self.content:
                if isinstance(item, BaseModel):
                    serialized["content"]["data"].append({
                        "model_class": f"{item.__class__.__module__}.{item.__class__.__name__}",
                        "data": item.model_dump()
                    })
                else:
                    # Fallback for non-BaseModel items
                    serialized["content"]["data"].append({
                        "model_class": "str",
                        "data": str(item)
                    })
        else:
            # Fallback for unknown types
            serialized["content"] = {
                "type": "string",
                "data": str(self.content)
            }
            
        return serialized

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any]) -> "Message":
        """
        Deserialize Message from cache dictionary with type reconstruction.
        """
        role = data["role"]
        content_info = data["content"]
        
        # Reconstruct content based on type
        if content_info["type"] == "string":
            content = content_info["data"]
        elif content_info["type"] == "pydantic_model":
            # Dynamically import and reconstruct the Pydantic model
            model_class_path = content_info["model_class"]
            content = cls._reconstruct_pydantic_model(model_class_path, content_info["data"])
        elif content_info["type"] == "pydantic_list":
            content = []
            for item in content_info["data"]:
                if item["model_class"] == "str":
                    content.append(item["data"])
                else:
                    reconstructed = cls._reconstruct_pydantic_model(
                        item["model_class"], 
                        item["data"]
                    )
                    content.append(reconstructed)
        else:
            # Fallback
            content = content_info["data"]
            
        return cls(role=role, content=content)

    @staticmethod
    def _reconstruct_pydantic_model(model_class_path: str, data: dict[str, Any]) -> BaseModel | str:
        """
        Dynamically import and reconstruct a Pydantic model from its class path and data.
        If reconstruction fails (class changed/moved/deleted), returns JSON string with warning.
        """
        try:
            # Split module and class name
            module_path, class_name = model_class_path.rsplit(".", 1)
            
            # Import the module and get the class
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            
            # Reconstruct the model
            return model_class.model_validate(data)
        except ImportError as e:
            # Module not found - class was moved or deleted
            json_str = json.dumps(data, indent=2)
            warnings.warn(
                f"Cache deserialization failed: Module '{module_path}' not found. "
                f"Original model class '{model_class_path}' may have been moved or deleted. "
                f"Falling back to JSON string representation. "
                f"Error: {e}",
                UserWarning,
                stacklevel=3
            )
            return json_str
        except AttributeError as e:
            # Class not found in module - class was renamed or deleted
            json_str = json.dumps(data, indent=2)
            warnings.warn(
                f"Cache deserialization failed: Class '{class_name}' not found in module '{module_path}'. "
                f"Original model class '{model_class_path}' may have been renamed or deleted. "
                f"Falling back to JSON string representation. "
                f"Error: {e}",
                UserWarning,
                stacklevel=3
            )
            return json_str
        except (ValueError, TypeError) as e:
            # Validation failed - class schema changed incompatibly
            json_str = json.dumps(data, indent=2)
            warnings.warn(
                f"Cache deserialization failed: Model validation failed for '{model_class_path}'. "
                f"The class schema may have changed incompatibly with cached data. "
                f"Falling back to JSON string representation. "
                f"Error: {e}",
                UserWarning,
                stacklevel=3
            )
            return json_str

    def __str__(self):
        """
        Returns the message in a human-readable format.
        """
        return f"{self.role}: {self.content}"

    def __getitem__(self, key):
        """
        Allows for dictionary-style access to the object.
        """
        return getattr(self, key)

# Some helpful functions
def create_system_message(
    system_prompt: str | Prompt, input_variables=None
) -> list[Message]:
    """
    Takes a system prompt object (Prompt()) or a string, an optional input object, and returns a Message object.
    """
    if isinstance(system_prompt, str):
        system_prompt = Prompt(system_prompt)
    if input_variables:
        system_message = [
            Message(
                role="system",
                content=system_prompt.render(input_variables=input_variables),
            )
        ]
    else:
        system_message = [Message(role="system", content=system_prompt.prompt_string)]
    return system_message
