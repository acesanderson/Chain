from ModelSpec import ModelSpec
from Chain.model.model import Model
from Chain.chain.chain import Chain
from Chain.prompt.prompt import Prompt
from Chain.parser.parser import Parser

prompt_str = "Create a ModelSpec for this model from OpenAI: o4-mini"
prompt = Prompt(prompt_str)
parser = Parser(ModelSpec)
# model = Model("sonar-pro")
model = Model("gpt")
chain = Chain(prompt=prompt, model=model, parser=parser)
response = chain.run()
content = response.content
content.card()
