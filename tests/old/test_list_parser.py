from Chain.parser.parser import Parser
from Chain.chain.chain import Chain
from Chain.model.model import Model
from Chain.prompt.prompt import Prompt
from pydantic import BaseModel


class Frog(BaseModel):
    species: str
    name: str
    no_of_legs: int


class Frogs(BaseModel):
    army: list[Frog]


prompt_str = """
Create a frog.
""".strip()


if __name__ == "__main__":
    prompt = Prompt(prompt_str)
    model = Model("gemini")
    # model = Model("sonar")
    parser = Parser(Frog)  # Single frog
    # parser = Parser(Frogs)
    chain = Chain(prompt=prompt, model=model, parser=parser)
    response = chain.run()
    print(response)
