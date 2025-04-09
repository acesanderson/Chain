"""
Resources are parameterless functions that return a static resource (typically a string but could be anything that an LLM would interpret).

Example usage:

```python
resource_registry = ResourceRegistry()

@resource_registry.register
def my_resource():
    "" This is a resource that returns a string. ""
    return "This is my resource."
```

name = my_resource (the function name)
description = "This is a resource that returns a string." (the docstring)
"""

from typing import Callable
from inspect import signature


# Our registry; .register to be used as a decorator.
class ResourceRegistry:
    def __init__(self):
        self.resources = {}

    def register(self, func):
        resource = Resource(func)
        self.resources[resource.name] = resource
        return func


# Tool class
class Resource:
    def __init__(self, function: Callable):
        self.function = function
        try:
            self.description = function.__doc__.strip()  # type: ignore
        except AttributeError:
            print("Function needs a docstring")
        try:
            # If function has parameters, raise an error
            self.args = self.validate_parameters()
            if self.args:
                raise ValueError("Resource function should not have parameters.")
        except ValueError:
            print("Resource function should not have parameters.")
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
