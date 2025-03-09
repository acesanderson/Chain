# from Chain.model.clients.openai_client import OpenAIClientSync
from pydantic import BaseModel
from Chain import Model

model = Model("gpt-3.5-turbo-0125")


class Frog(BaseModel):
    species: str
    name: str
    age: int
    no_legs: int


# openai = OpenAIClientSync()
# print(openai.query(input="name ten mammals", model="o3-mini"))
# obj = openai.query(input="Create a frog", pydantic_model=Frog, model="o3-mini")
# obj, raw_text = openai.query(
#     input="Create a frog", model="gpt-3.5-turbo-0125", pydantic_model=Frog, raw=True
# )
obj, raw_text = model.query(
    input="Create a frog", pydantic_model=Frog, raw=True
)  # , model="o3-mini")
print(obj)
print(raw_text)
