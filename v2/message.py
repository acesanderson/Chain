"""
This module provides a standard way to generate messages.
All LLM SDKs work with some variant of a Message object, which is a role/content pair.

Message objects allow us to:
- quickly validate and maintain consistency in our chains
- see the history of messages within a Response object
- easily serialize and deserialize messages
- use the same message objects across all LLM SDKs (while accounting for differences like with Anthropic)
- system message generation for more complex chains
"""
from pydantic import BaseModel
from prompt import Prompt

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
	messages: list[Message]

def is_messages_object(input):
	if not input:
		return False
	try:
		Messages(messages=input)
		return True
	except Exception as e:
		print(e)
		return False

def create_messages(system_prompt: str, input_variables = None) -> list[dict]:
	"""
	Takes a system prompt object (Prompt()) or a string, an optional input object, and returns a list of messages.
	"""
	if isinstance(system_prompt, str):
		system_prompt = Prompt(system_prompt)
	if input_variables:
		messages = [{'role': 'system', 'content': system_prompt.render(input_variables=input_variables)}]
	else:
		messages = [{'role': 'system', 'content': system_prompt.string}]
	return messages
