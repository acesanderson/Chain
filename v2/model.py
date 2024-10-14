"""
Modularized version of Model class.

NEXT BIG THING TO DO:
- lazy load should happen on model object initialization, not on query.
"""

import importlib
from typing import Union, Type, Optional
from pydantic import BaseModel
import json
import itertools
from collections import defaultdict

class Model:
	# Some class variables: models, context sizes, clients
	# Load models from the JSON file
	with open('models.json') as f:
		_models = json.load(f)

	# Load Ollama context sizes from the JSON file
	with open('ollama_context_sizes.json') as f:
		ollama_context_data = json.load(f)

	# Use defaultdict to set default context size to 4096 if not specified
	_ollama_context_sizes = defaultdict(lambda: 4096)
	_ollama_context_sizes.update(ollama_context_data)

	# Store lazy-loaded client instances at the class level
	_clients = {}
	
	def __init__(self, model: str = "gpt-4o"):
		self.model = self._validate_model(model)
		self._client_type = self._get_client_type()
		self._client = None  # Will be lazy-loaded when needed

	def _validate_model(cls, model: str) -> str:
		"""
		This is where you can put in any model aliases you want to support.
		"""
		# Load our aliases from aliases.json
		with open('aliases.json', 'r') as f:
			aliases = json.load(f)
		# Check data quality.
		for value in aliases.values():
			if value not in list(itertools.chain.from_iterable(cls._models.values())):
				raise ValueError(f"WARNING: This model declared in aliases.json is not available: {value}.")
		# Assign models based on aliases
		if model in aliases.keys():
			model = aliases[model]
		elif model in list(itertools.chain.from_iterable(cls._models.values())):   # any other model we support (flattened the list)
			model = model
		else:
			ValueError(f"WARNING: Model not found locally: {model}. This may cause errors.")
		return model

	def _get_client_type(self):
		"""
		Setting client_type for Model object is necessary for loading the correct client in the query functions.
		"""
		model_list = self.__class__._models
		if self.model in model_list['openai']:
			return 'openai'
		elif self.model in model_list['anthropic']:
			return 'anthropic'
		elif self.model in model_list['google']:
			return 'google'
		elif self.model in model_list['ollama']:
			return 'ollama'
		elif self.model in model_list['groq']:
			return 'groq'
		elif self.model in model_list['testing']:
			return 'testing'
		else:
			raise ValueError(f"Model {self.model} not found in Chain.models")
	
	@classmethod
	def _get_client(cls, client_type: str):
		if client_type not in cls._clients:
			try:
				# Dynamically import the client module
				module = importlib.import_module(f'clients.{client_type.lower()}_client')
				client_class = getattr(module, f'{client_type.capitalize()}Client')
				cls._clients[client_type] = client_class()
			except ImportError as e:
				raise ImportError(f"Failed to import {client_type} client: {str(e)}")
		return cls._clients[client_type]

	def query(self, input: Union[str, list], verbose: bool = True, pydantic_model: Optional[Type[BaseModel]] = None) -> Union[BaseModel, str]:
		if self._client is None:
			self._client = self._get_client(self._client_type)
		if verbose:
			print(f"Model: {self.model}   Query: " + self.pretty(str(input)))
		return self._client.query(self.model, input, pydantic_model)

	async def query_async(self, input: Union[str, list], verbose: bool = True, pydantic_model: Optional[Type[BaseModel]] = None) -> Union[BaseModel, str]:
		if self._client is None:
			self._client = self._get_client(self._client_type)
		if verbose:
			print(f"Model: {self.model}   Query: " + self.pretty(str(input)))
		return await self._client.query_async(self.model, input, pydantic_model)

	@classmethod
	def update_ollama(cls):
		"""
		Updates the list of Ollama models.
		"""
		# Lazy load ollama module
		import ollama
		ollama_models = [m['name'] for m in ollama.list()['models']]
		with open('models.json', 'r') as f:
			model_list = json.load(f)
		model_list['ollama'] = ollama_models
		with open('models.json', 'w') as f:
			json.dump(model_list, f)
		cls.models = model_list

	def pretty(self, user_input):
		pretty = user_input.replace('\n', ' ').replace('\t', ' ').strip()
		return pretty[:150]

	# Other utility methods can remain in the Model class