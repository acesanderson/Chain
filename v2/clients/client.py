"""
Base class for clients; openai, anthropic, etc. inherit from this class.
ABC abstract methods are like pydantic validators for classes. They ensure that the child classes implement the methods defined in the parent class.
If a client subclass doesn't implement _get_api_key, for example, the code will raise an error when trying to instantiate the subclass.
This guarantees all client subclasses have the methods below.
"""

from abc import ABC, abstractmethod

class Client(ABC):
	@abstractmethod
	def __init__(self):
		"""
		Typically we should see this here:
		self._client = self._initialize_client()
		"""
		pass

	@abstractmethod
	def _initialize_client(self):
		"""
		This method should initialize the client object, this is idiosyncratic to each SDK.
		"""
		pass

	@abstractmethod
	def _get_api_key(self):
		"""
		API keys are accessed via dotenv.
		This should be in an .env file in the root directory.
		"""
		pass

	@abstractmethod
	def query(self, model: str, input: 'str | list', pydantic_model: 'BaseModel' = None) -> 'BaseModel | str': # type: ignore
		"""
		All client subclasses must have a query function that can take:
		- a model name
		- input in the form of EITHER a string or a Messages-style list of dictionaries
		- an optional Pydantic model to validate the response
		And returns
		- either a string (i.e. text generation) or a Pydantic model (function calling)
		"""
		pass

	@abstractmethod
	async def query_async(self, model: str, input: 'str | list', pydantic_model: 'BaseModel' = None) -> 'BaseModel | str': # type: ignore
		pass

	def __repr__(self):
		"""
		Standard repr.
		"""
		attributes = ', '.join([f'{k}={repr(v)[:50]}' for k, v in self.__dict__.items()])
		return f"{self.__class__.__name__}({attributes})"