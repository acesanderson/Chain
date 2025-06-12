"""
Instructor library does most of the heavy lifting here -- great piece of software.
We have to get a little fancy to allow list[BaseModel] as an option -- for some APIs (i.e. perplexity) you need to submit that to Instructor in order to get multiple pydantic objects back.
"""

from typing import get_origin, get_args, Union, Type
from pydantic import BaseModel


class Parser:
    def __init__(self, pydantic_model: Union[Type[BaseModel], type]):
        self.original_spec = pydantic_model
        self.is_list = self._is_list_of_basemodel(pydantic_model)

        if self.is_list:
            # Extract the inner BaseModel type
            inner_type = get_args(pydantic_model)[0]
            # For Instructor, we need to use List[ModelSpec] not list[ModelSpec]
            from typing import List

            self.pydantic_model = List[inner_type]
        else:
            self.pydantic_model = pydantic_model

    def _is_list_of_basemodel(self, obj) -> bool:
        """Check if the object is a list of BaseModel instances."""
        if hasattr(obj, "__origin__"):
            origin = get_origin(obj)
            if origin is list:
                args = get_args(obj)
                if args and len(args) > 0:
                    return issubclass(args[0], BaseModel)
        return False
