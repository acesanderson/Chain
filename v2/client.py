"""
Base class for clients; openai, anthropic, etc. inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Union, Type, Optional
from pydantic import BaseModel

class Client(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _initialize_client(self):
        pass

    @abstractmethod
    def _get_api_key(self):
        pass

    @abstractmethod
    def query(self, model: str, input: Union[str, list], pydantic_model: Optional[Type[BaseModel]] = None) -> Union[BaseModel, str]:
        pass

    @abstractmethod
    async def query_async(self, model: str, input: Union[str, list], pydantic_model: Optional[Type[BaseModel]] = None) -> Union[BaseModel, str]:
        pass