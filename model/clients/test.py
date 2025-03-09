# from Chain.model.clients.openai_client import OpenAIClientSync
from openai import OpenAI
from pydantic import BaseModel
from Chain import Model, ChainCache
from Chain.model.clients.ollama_client import OllamaClientSync

Model._chain_cache = ChainCache()
# model = Model("gpt-3.5-turbo-0125")
# model = Model("claude")
model = Model("llama3.1:latest")


class Frog(BaseModel):
    species: str
    name: str
    age: int
    no_legs: int

    def __repr__(self) -> str:
        return f"Frog(species={self.species}, name={self.name}, age={self.age}, no_legs={self.no_legs})"

    def __str__(self) -> str:
        return self.__repr__()


# openai = OpenAIClientSync()
# print(openai.query(input="name ten mammals", model="o3-mini"))
# obj = openai.query(input="Create a frog", pydantic_model=Frog, model="o3-mini")
# obj, raw_text = openai.query(
#     input="Create a frog", model="gpt-3.5-turbo-0125", pydantic_model=Frog, raw=True
# )
# print("not raw -------------")
# obj = model.query(input="create a frog", pydantic_model=frog)  # , model="o3-mini")
# print(obj)
#
#
# print("Raw -------------")
# obj, raw_text = model.query(
#     input="Create a frog", pydantic_model=Frog, raw=True
# )  # , model="o3-mini")
# print(obj)
# print(raw_text)

print("Not Raw -------------")
obj = model.query(input="create a frog", pydantic_model=Frog)  # , model="o3-mini")

print("Raw -------------")
obj, raw_text = model.query(input="Create a frog", pydantic_model=Frog, raw=True)
print(obj)
print(raw_text)


# client = instructor.from_openai(
#     OpenAI(
#         base_url="http://localhost:11434/v1",
#         api_key="ollama",  # required, but unused
#     ),
#     mode=instructor.Mode.JSON,
# )
#
# resp, raw_resp = client.chat.completions.create_with_completion(
#     model="llama3.1:latest",
#     messages=[
#         {
#             "role": "user",
#             "content": "Create a frog",
#         }
#     ],
#     response_model=Frog,
#     extra_body={"options": {"num_ctx": 4096}},
# )
#
# print(resp)
# print(raw_resp)
