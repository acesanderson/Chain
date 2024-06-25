from Chain import Model, Chain, Prompt, Parser
from pydantic import BaseModel                          # for our input_schema and output_schema; starting with List Parser.
from typing import List, Optional, Type, Union          # for type hints

class Example_List(BaseModel):
	examples: List[str]



if __name__ == "__main__":
	# print("OLLAMA\n====================================")
	# print(Model.query_ollama("sing a sing about john henry", model="mistral:latest"))
	# print("OPENAI\n====================================")
	# print(Model.query_openai("sing a sing about calamity jane", model="gpt-4o"))
	# print("ANTHROPIC\n====================================")
	# print(Model.query_anthropic("sing a sing about Paul Bunyan", model="claude-3-5-sonnet-20240620"))
	# print("Now with a pydantic model\n====================================")
	# print("OPENAI\n====================================")
	# response = Model.query_openai("Name ten mammals", pydantic_model=Example_List)
	# print(response)
	# print("ANTHROPIC\n====================================")
	# response = Model.query_anthropic("Name ten mammals", pydantic_model=Example_List)
	# print(response)
	# print("OLLAMA\n====================================")
	# response = Model.query_ollama("Name ten mammals", pydantic_model=Example_List)
	# print(response)
	prompt = Prompt("sing a song about john henry")
	parsed_prompt = Prompt("Name five birds.")
	model = Model('gpt')
	parser = Parser(Example_List)
	chain = Chain(prompt, model)
	parsed_chain = Chain(parsed_prompt, model, parser)
	print(chain.run())
	print(parsed_chain.run())

