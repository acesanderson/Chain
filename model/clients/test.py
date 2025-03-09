# from Chain.model.clients.openai_client import OpenAIClientSync
from openai import OpenAI
from pydantic import BaseModel
from Chain import Model, ChainCache
from Chain.model.clients.ollama_client import OllamaClientSync

Model._chain_cache = ChainCache()
# model = Model("gpt-3.5-turbo-0125")
# model = Model("claude")
model = Model("llama3.1:latest")
# model = Model("llama3-70b-8192")


class Frog(BaseModel):
    species: str
    name: str
    age: int
    no_legs: int

    def __repr__(self) -> str:
        return f"Frog(species={self.species}, name={self.name}, age={self.age}, no_legs={self.no_legs})"

    def __str__(self) -> str:
        return self.__repr__()


print("Not Raw -------------")
obj = model.query(input="create a frog", pydantic_model=Frog)  # , model="o3-mini")

print("Raw -------------")
obj, raw_text = model.query(input="Create a frog", pydantic_model=Frog, raw=True)
print(obj)
print(raw_text)
