"""
Resources are parameterless functions that return a static resource (typically a string but could be anything that an LLM would interpret).

Example usage:

```python
tool_registry = ToolRegistry()

@tool_registry.register
def my_tool(param1: str, param2: int):
    "" This is a tool that returns an answer. ""
    return param1 + str(param2)
```

name = my_tool (the function name)
description = "This is a tool that returns a string." (the docstring)
"""

from typing import Callable
from inspect import signature


# Our registry; .register to be used as a decorator.
class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, func):
        tool = Tool(func)
        self.tools[tool.name] = tool
        return func


# Tool class
class Tool:
    def __init__(self, function: Callable):
        self.function = function
        try:
            self.description = function.__doc__.strip()  # type: ignore
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
