"""
Instructor library does most of the heavy lifting here -- great piece of software.
We have to get a little fancy to allow list[BaseModel] as an option -- for some APIs (i.e. perplexity) you need to submit that to Instructor in order to get multiple pydantic objects back.
"""

from typing import Union, Type
from pydantic import BaseModel
from Chain.logging.logging_config import get_logger

logger = get_logger(__name__)


class Parser:
    def __init__(self, pydantic_model: Union[Type[BaseModel], type]):
        """
        Initialize the Parser with a model specification.
        :param pydantic_model: A Pydantic BaseModel class or a list of BaseModel classes.
        """
        self.original_spec = pydantic_model

        self.pydantic_model = pydantic_model

    def __repr__(self):
        return f"Parser({self.original_spec})"

    def to_perplexity(self) -> BaseModel | type:
        """
        Convert the Pydantic model to a type suitable for Perplexity API.
        For wrapper classes with single list fields, extract the inner type
        so Instructor can handle multiple tool calls properly.
        """
        import typing

        # Handle wrapper models with a single list field
        if isinstance(self.pydantic_model, type) and issubclass(
            self.pydantic_model, BaseModel
        ):
            # Pydantic v2
            if hasattr(self.pydantic_model, "model_fields"):
                fields = self.pydantic_model.model_fields
            # Pydantic v1 fallback
            else:
                fields = self.pydantic_model.__fields__

            # Check if it's a wrapper with a single list field
            if len(fields) == 1:
                field_name, field_info = next(iter(fields.items()))
                field_type = (
                    field_info.annotation
                    if hasattr(field_info, "annotation")
                    else field_info.type_
                )

                # Check if it's list[SomeBaseModel]
                field_origin = typing.get_origin(field_type)
                if field_origin is list:
                    args = typing.get_args(field_type)
                    if args and issubclass(args[0], BaseModel):
                        # Return List[InnerModel] for Instructor
                        return list[args[0]]

        # Handle direct list[BaseModel] type annotations
        origin = typing.get_origin(self.pydantic_model)
        if origin is list:
            args = typing.get_args(self.pydantic_model)
            if args and issubclass(args[0], BaseModel):
                return self.pydantic_model  # Return list[SomeModel] directly

        # For non-wrapper classes, return as-is
        return self.pydantic_model

    #
    #
    #
    # def to_perplexity(self) -> BaseModel | type:
    #     """
    #     Convert the Pydantic model to a type suitable for Perplexity API.
    #     """
    #     import typing
    #
    #     # Handle direct list[BaseModel] type annotations
    #     origin = typing.get_origin(self.pydantic_model)
    #     if origin is list:
    #         args = typing.get_args(self.pydantic_model)
    #         if args and issubclass(args[0], BaseModel):
    #             return self.pydantic_model  # Return list[SomeModel] directly
    #
    #     # Handle wrapper models with a single list field
    #     if isinstance(self.pydantic_model, type) and issubclass(
    #         self.pydantic_model, BaseModel
    #     ):
    #         # Pydantic v2
    #         if hasattr(self.pydantic_model, "model_fields"):
    #             fields = self.pydantic_model.model_fields
    #         # Pydantic v1 fallback
    #         else:
    #             fields = self.pydantic_model.__fields__
    #
    #         if len(fields) == 1:
    #             field_name, field_info = next(iter(fields.items()))
    #             field_type = (
    #                 field_info.annotation
    #                 if hasattr(field_info, "annotation")
    #                 else field_info.type_
    #             )
    #
    #             # Check if it's list[SomeBaseModel]
    #             field_origin = typing.get_origin(field_type)
    #             if field_origin is list:
    #                 args = typing.get_args(field_type)
    #                 if args and issubclass(args[0], BaseModel):
    #                     return list[args[0]]
    #
    #     return self.pydantic_model
