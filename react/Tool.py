from typing import Callable
from inspect import signature


# Tool class
class Tool:
    def __init__(self, function: Callable):
        self.function = function
        try:
            self.description = function.__doc__.strip()
        except AttributeError:
            print("Function needs a docstring")
        try:
            self.args = self.validate_parameters()
        except AttributeError:
            print("Function needs type annotations. Does this have any parameters?")
        self.name = function.__name__

    def validate_parameters(self):
        sig = signature(self.function)
        params = sig.parameters
        param_types = {
            name: param.annotation.__name__ for name, param in params.items()
        }
        return str(param_types)

    def __call__(self, **kwargs):
        return self.function(**kwargs)
