from pydantic import BaseModel
from Chain import Model, ChainCache, Prompt, ModelAsync, AsyncChain, Parser
from Chain.model.clients.ollama_client import OllamaClientSync

Model._chain_cache = ChainCache()
# model = Model("gpt-3.5-turbo-0125")
# model = Model("claude")
# model = Model("llama3.1:latest")
# model = Model("llama3-70b-8192")
# model = Model("gemini-2.0-flash-001")
# model = Model("sonar")


class Frog(BaseModel):
    species: str
    name: str
    age: int
    no_legs: int

    def __repr__(self) -> str:
        return f"Frog(species={self.species}, name={self.name}, age={self.age}, no_legs={self.no_legs})"

    def __str__(self) -> str:
        return self.__repr__()


# names = ["Freddy", "Frodo", "Fiona", "Frankie", "Fergus", "Felicity"]
# input_variables_list = [{"name": name} for name in names]
#
# model = ModelAsync("gpt-3.5-turbo-0125")
# prompt = Prompt("Create a frog with this name: {{name}}")
# parser = Parser(Frog)
# chain = AsyncChain(prompt=prompt, model=model, parser=parser)
# response = chain.run(input_variables_list=input_variables_list)
# print(response)

prompts = ["name ten birds", "name ten mammals", "name ten reptiles"]
model = ModelAsync("claude")
chain = AsyncChain(model=model)
responses = chain.run(prompt_strings=prompts)
print(responses)

# print("Not Raw -------------")
# # obj = model.query(input="create a frog", pydantic_model=Frog)
# obj = model.query(
#     input="Provide some recommendations for small cap ETFs or index funds."
# )
# print(obj)
# print("Raw -------------")
# obj, raw_text = model.query(input="Create a frog", pydantic_model=Frog, raw=True)
# print(obj)
# print(raw_text)
