from Chain.model.model_async import ModelAsync
from Chain.chain.asyncchain import AsyncChain
from Chain.prompt.prompt import Prompt
from Chain.parser.parser import Parser
from pydantic import BaseModel

class Frog(BaseModel):
    name: str
    no_of_legs: int
    species: str
    occupation: str
    country: str
    color: str

prompt_str = """
Make a new frog.
""".strip()

model = ModelAsync("cogito:14b")
prompt = Prompt(prompt_str)
parser = Parser(Frog)

prompt_strings = [prompt_str] * 8
print(prompt_strings)

chain = AsyncChain(model=model, parser=parser)
responses = chain.run(prompt_strings = prompt_strings)
print(responses)
