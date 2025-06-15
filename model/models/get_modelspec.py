from ModelSpec import ModelSpec
from Chain import Model, Prompt, Parser, Chain

prompt_str = "Create a ModelSpec for this model from OpenAI: o4-mini"
prompt = Prompt(prompt_str)
parser = Parser(ModelSpec)
# model = Model("sonar-pro")
model = Model("gpt")
chain = Chain(prompt=prompt, model=model, parser=parser)
response = chain.run()
content = response.content
content.card()

              
