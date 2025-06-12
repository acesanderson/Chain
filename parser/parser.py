"""
Instructor library does most of the heavy lifting here -- great piece of software.
We have to get a little fancy to allow list[BaseModel] as an option -- for some APIs (i.e. perplexity) you need to submit that to Instructor in order to get multiple pydantic objects back.
"""

from typing import get_origin, get_args, Union, Type
from pydantic import BaseModel


class Parser:
    def __init__(self, pydantic_model: Union[Type[BaseModel], type]):
        """
        Initialize the Parser with a model specification.
        :param pydantic_model: A Pydantic BaseModel class or a list of BaseModel classes.
        """
        self.original_spec = pydantic_model
        self.is_list = self._is_list_of_basemodel(pydantic_model)

        self.pydantic_model = pydantic_model

    def _is_list_of_basemodel(self, obj) -> bool:
        """
        Check if the object is a list of BaseModel instances.
        """
        if hasattr(obj, "__origin__"):
            return get_origin(obj) is list and issubclass(get_args(obj)[0], BaseModel)
        return False

    def __repr__(self):
        return f"Parser({self.original_spec})"
