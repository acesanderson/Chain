"""
Check out how progress looks for these use cases:
- individual model calls
- ask, tutorialize, leviathan
- sequence of sync model calls
- asynchronous
 - prompt strings
 - input variables
 - pydantic models
"""
from rich.console import Console
console = Console()

# Individual model call
from Chain import Model


# m=Model("gpt", console=console)
# m=Model("gpt")
# response = m.query("name ten mammals")
# print(response)
# Spinner ain't spinnin'

# Series of sync model calls
from Chain import Prompt, Chain
prompt_str = """
Name ten {{things}}.
"""
prompt = Prompt(prompt_str)
# # model = Model("gpt")
# model = Model("gpt", console=console)
# chain = Chain(prompt=prompt, model=model)
things = [
    "birds",
    "cars",
    "card games",
    "african countries",
    "famous villains"
]
# for index, thing in enumerate(things):
#     response = chain.run(input_variables={"things": thing}, index=index+1, total=len(things))
#     print(response)

from Chain import AsyncChain, ModelAsync
model = ModelAsync("gpt", console = console)
# model = ModelAsync("gpt")
# prompt = Prompt(prompt_str)
# chain = AsyncChain(prompt=prompt, model=model)
#
# things = [{"things":thing} for thing in things]
#
# response = chain.run(input_variables_list=things)
# print(response)
#

# async with prompt_str
prompt_strings = []
for thing in things:
    prompt_strings.append(prompt.render({"things": thing}))

print(prompt_strings)

chain = AsyncChain(model=model)
response = chain.run(prompt_strings=prompt_strings)
print(response)
