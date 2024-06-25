# for testing -- remove whern incorporating code into main
# also look at self.model in the query functions -- need to reinstate
# what do I want to return from query functions? str/pydantic object. not Message object, that's handled outside of query functions.
# Parser should set number of retries, and it should carry over to the query functions.

from Chain import Chain
from Chain import api_keys

# all our packages
from jinja2 import Environment, meta, StrictUndefined   # we use jinja2 for prompt templates
from openai import OpenAI                               # GPT
import google.generativeai as genai                     # Google's models
from openai import AsyncOpenAI                          # for async; not implemented yet
from anthropic import Anthropic                         # claude
from groq import Groq                                   # groq
import ollama                                           # local llms
import re                                               # for regex
import os                                               # for environment variables
import dotenv                                           # for loading environment variables
import itertools                                        # for flattening list of models
import json                                             # for our jsonparser
import time                                             # for timing our query calls (saved in Response object)
import textwrap                                         # to allow for indenting of multiline strings for code readability
from pydantic import BaseModel                          # for our input_schema and output_schema; starting with List Parser.
from typing import List, Optional, Type, Union          # for type hints
import instructor                                       # for parsing objects from LLMs

# set up our environment: dynamically setting the .env location considered best practice for complex projects.
dir_path = os.path.dirname(os.path.realpath(__file__))
# Construct the path to the .env file
env_path = os.path.join(dir_path, '.env')
# Load the environment variables
dotenv.load_dotenv(dotenv_path=env_path)
api_keys = {}
api_keys['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
api_keys['ANTHROPIC_API_KEY'] = os.getenv("ANTHROPIC_API_KEY")
api_keys['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
api_keys['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
client_openai = instructor.from_openai(OpenAI(api_key = api_keys['OPENAI_API_KEY']))
client_anthropic = instructor.from_anthropic(Anthropic(api_key = api_keys['ANTHROPIC_API_KEY']))
genai.configure(api_key=api_keys["GOOGLE_API_KEY"])
# client_google = instructor.from_gemini(
#     client=genai.GenerativeModel(
#         model_name="models/gemini-1.5-flash-latest",  # model defaults to "gemini-pro"
#     ),
#     mode=instructor.Mode.GEMINI_JSON,
# )
# https://pypi.org/project/google-generativeai/
async_client_openai = instructor.from_openai(AsyncOpenAI(api_key = api_keys["OPENAI_API_KEY"]))
# async_client_anthropic: TBD
# async_client_google: TBD
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))    # Instructor not support!
client_ollama = instructor.from_openai(
	OpenAI(
		base_url="http://localhost:11434/v1",
		api_key="ollama",  # required, but unused
	),
	mode=instructor.Mode.JSON,
)
env = Environment(undefined=StrictUndefined)            # # set jinja2 to throw errors if a variable is undefined

"""
the other stuff
"""

class Message(BaseModel):
	"""
	Industry standard, more or less, for messaging with LLMs.
	System roles can have some weirdness (like with Anthropic), but role/content is standard.
	"""
	role: str
	content: str

class Messages(BaseModel):
	"""
	Industry standard, more or less, for messaging with LLMs.
	System roles can have some weirdness (like with Anthropic), but role/content is standard.
	"""
	messages: List[Message]

class Example_List(BaseModel):
	examples: List[str]

def query(input: Union[str, list], verbose: bool=True, model: str = 'mistral:latest', pydantic_model: Optional[Type[BaseModel]] = None) -> Union[BaseModel, str, List]:
	# input can be message or str
	if model in Chain.models['openai']:
		return query_openai(input, verbose, model, response_model)
	if model in Chain.models['anthropic']:
		return query_anthropic(input, verbose, model, response_model)
	if model in Chain.models['Google']:
		return query_google(input, verbose, model, response_model)
	if model in Chain.models['Ollama']:
		return query_ollama(input, verbose, model, response_model)
	if model in Chain.models['Groq']:
		return query_groq(input, verbose, model, response_model)
	if model in Chain.models['testing']:
		return query_testing(input, verbose, model, response_model)
	else:
		raise ValueError(f"Model {model} not found in Chain.models")

# we no longer need "Chat" functions or a separate Instructor function.
def query_openai(input: Union[str, list], verbose: bool=True, model: str = 'gpt-4o', pydantic_model: Optional[Type[BaseModel]] = None) -> Union[BaseModel, str]:
	"""
	Handles all synchronous requests from OpenAI's models.
	Possibilities:
	- pydantic object not provided, input is string -> return string
	- pydantic object provided, input is string -> return pydantic object
	- pydantic object not provided, input is a Messages object -> return Messages object
	- pydantic object provided, input is a Messages object -> return Messages object (with pydantic object as the Content of the last Message)
	"""
	if isinstance(input, str):
		input = [{"role": "user", "content": input}]
	elif Messages.model_validate(input):
		input = input
	else:
		raise ValueError(f"Input not recognized as a valid input type: {type:input}: {input}")
	# call our client
	response = client_openai.chat.completions.create(
		# model=self.model,
		model = model,
		response_model = pydantic_model,
		messages = input
	)
	if pydantic_model:
		return response
	else:
		return response.choices[0].message.content

def query_anthropic(input: Union[str, list], verbose: bool=True, model: str = 'claude-3-5-sonnet-20240620', pydantic_model: Optional[Type[BaseModel]] = None) -> Union[BaseModel, str]:
	"""
	Handles all synchronous requests from Anthropic's models.
	Possibilities:
	- pydantic object not provided, input is string -> return string
	- pydantic object provided, input is string -> return pydantic object
	- pydantic object not provided, input is a Messages object -> return Messages object
	- pydantic object provided, input is a Messages object -> return Messages object (with pydantic object as the Content of the last Message)
	"""
	if isinstance(input, str):
		input = [{"role": "user", "content": input}]
	elif Messages.model_validate(input):
		input = input
	else:
		raise ValueError(f"Input not recognized as a valid input type: {type:input}: {input}")
	# call our client
	response = client_anthropic.chat.completions.create(
		# model = self.model,
		model = model,
		max_tokens = 1024,
		max_retries = 0,
		messages = input,
		response_model = pydantic_model,
	)
	# only two possibilities here
	if pydantic_model:
		return response
	else:
		return response.content[0].text

def query_ollama(input: Union[str, list], verbose: bool=True, model: str = 'mistral:latest', pydantic_model: Optional[Type[BaseModel]] = None) -> Union[BaseModel, str]:
	"""
	Handles all synchronous requests from OpenAI's models.
	Possibilities:
	- pydantic object not provided, input is string -> return string
	- pydantic object provided, input is string -> return pydantic object
	- pydantic object not provided, input is a Messages object -> return Messages object
	- pydantic object provided, input is a Messages object -> return Messages object (with pydantic object as the Content of the last Message)
	"""
	if isinstance(input, str):
		input = [{"role": "user", "content": input}]
	elif Messages.model_validate(input):
		input = input
	else:
		raise ValueError(f"Input not recognized as a valid input type: {type:input}: {input}")
	# call our client
	response = client_ollama.chat.completions.create(
		# model=self.model,
		model = model,
		response_model = pydantic_model,
		messages = input
	)
	if pydantic_model:
		return response
	else:
		return response.choices[0].message.content

if __name__ == "__main__":
	# print("OLLAMA\n====================================")
	# print(query_ollama("sing a sing about john henry", model="mistral:latest"))
	# print("OPENAI\n====================================")
	# print(query_openai("sing a sing about calamity jane", model="gpt-4o"))
	# print("ANTHROPIC\n====================================")
	# print(query_anthropic("sing a sing about Paul Bunyan", model="claude-3-5-sonnet-20240620"))
	# print("Now with a pydantic model\n====================================")
	# print("OPENAI\n====================================")
	response = query_openai("Name ten mammals", pydantic_model=Example_List)
	print(response)
	print("ANTHROPIC\n====================================")
	response = query_anthropic("Name ten mammals", pydantic_model=Example_List)
	print(response)
	print("OLLAMA\n====================================")
	response = query_ollama("Name ten mammals", pydantic_model=Example_List)
	print(response)


